"""Train a LoRA adapter for the Codenames spymaster with GRPO.

The script keeps dataset creation and reward calculation usable without the
training stack, which makes the environment/reward contract easy to smoke test
before downloading a model or starting a GPU job.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import random
from typing import Any, Callable, Sequence

from codenames_env import (
    BoardWord,
    CodenamesBoard,
    DEFAULT_VOCABULARY,
    EmbeddingGuesser,
    Guesser,
    OrderedGuesser,
    generate_board,
    simulate_turn,
    add_abstention_args,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    board_words: tuple[str, ...]
    board_labels: tuple[str, ...]
    board_revealed: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "board_words": list(self.board_words),
            "board_labels": list(self.board_labels),
            "board_revealed": list(self.board_revealed),
        }


def load_vocabulary_file(path: Path) -> tuple[str, ...]:
    """Load one board-word vocabulary item per line.

    Blank lines and comments are ignored so generated vocabulary files can
    include their source and filtering metadata at the top.
    """
    words = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip().lower()
        if line:
            words.append(line)
    if len(words) < 16:
        raise ValueError(f"Vocabulary file must contain at least 16 words: {path}")
    if len(set(words)) != len(words):
        raise ValueError(f"Vocabulary file contains duplicate words: {path}")
    return tuple(words)


def validate_initial_adapter(path: Path, model_name: str) -> dict[str, Any]:
    """Check a warm start before loading expensive policy weights."""
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ValueError("--initial-adapter must be a LoRA adapter")
    def canonical(name: str) -> str:
        return name.rsplit("/", 1)[-1].lower().replace("-unsloth-bnb-4bit", "")
    if canonical(config["base_model_name_or_path"]) != canonical(model_name):
        raise ValueError("--model-name differs from the initial adapter's base_model_name_or_path")
    return config


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the highest numbered Trainer checkpoint in an output directory."""
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def build_training_examples(
    num_boards: int = 2_000,
    seed: int = 10_000,
    vocabulary: Sequence[str] = DEFAULT_VOCABULARY,
    max_revealed: int = 0,
) -> list[TrainingExample]:
    """Procedurally create the prompt dataset; there are no ground-truth clues."""
    examples = []
    if not 0 <= max_revealed <= 12:
        raise ValueError("max_revealed must be between 0 and 12")
    for offset in range(num_boards):
        board = generate_board(seed=seed + offset, vocabulary=vocabulary)
        if max_revealed:
            rng = random.Random(seed + offset + 10_000_000)
            # Leave at least one word for each team, and never reveal ASSASSIN.
            protected = {rng.choice([item.word for item in board.words if item.label == label])
                         for label in ("BLUE", "RED")}
            revealable = [item.word for item in board.words
                          if item.label != "ASSASSIN" and item.word not in protected]
            board = CodenamesBoard(board.words, tuple(rng.sample(revealable, rng.randint(0, max_revealed))))
        examples.append(
            TrainingExample(
                prompt=board.prompt(),
                board_words=tuple(item.word for item in board.words),
                board_labels=tuple(item.label for item in board.words),
                board_revealed=board.revealed,
            )
        )
    return examples


def format_prompt_for_model(prompt: str, tokenizer: Any) -> str:
    """Use the instruct model's chat template when one is available."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _board_from_columns(words: Sequence[str], labels: Sequence[str], revealed: Sequence[str] = ()) -> CodenamesBoard:
    if len(words) != len(labels):
        raise ValueError("board_words and board_labels must have the same length")
    return CodenamesBoard(tuple(BoardWord(word, label) for word, label in zip(words, labels)), tuple(revealed))


def _completion_text(completion: Any) -> str:
    """Accept standard string completions and conversational TRL completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def _semantic_margin_bonus(
    result: Any,
    board: CodenamesBoard,
    guesser: Guesser,
    safety_weight: float,
    blue_weight: float,
) -> float:
    """Reward a clue whose BLUE associations beat its dangerous associations.

    The game reward is discrete and can make generic clues look attractive.
    This bounded auxiliary signal gives GRPO a smoother gradient while keeping
    the simulator's actual score as the dominant objective.
    """
    if result.clue is None or (safety_weight == 0 and blue_weight == 0):
        return 0.0
    similarity_fn = getattr(guesser, "similarities", None)
    if similarity_fn is None:
        return 0.0
    scores = similarity_fn(result.clue.word, [item.word for item in board.active_words])
    blue_scores = sorted(
        (scores[item.word] for item in board.active_words if item.label == "BLUE"),
        reverse=True,
    )
    dangerous_scores = [
        scores[item.word]
        for item in board.active_words
        if item.label in {"RED", "NEUTRAL", "ASSASSIN"}
    ]
    if not blue_scores or not dangerous_scores:
        return 0.0
    target_count = min(result.clue.number, len(blue_scores))
    blue_signal = sum(blue_scores[:target_count]) / target_count
    danger_signal = max(dangerous_scores)
    margin = max(-1.0, min(1.0, blue_signal - danger_signal))
    blue_signal = max(-1.0, min(1.0, blue_signal))
    return safety_weight * margin + blue_weight * blue_signal


def make_codenames_reward(
    guesser: Guesser,
    debug: bool = False,
    safety_weight: float = 0.5,
    blue_weight: float = 0.25,
    duplicate_penalty: float = 0.25,
) -> Callable[..., list[float]]:
    """Build the callable consumed by GRPOTrainer's ``reward_funcs`` argument."""

    def codenames_reward(
        completions: Sequence[Any],
        board_words: Sequence[Sequence[str]],
        board_labels: Sequence[Sequence[str]],
        board_revealed: Sequence[Sequence[str]] | None = None,
        **_: Any,
    ) -> list[float]:
        evaluated: list[tuple[Any, str, float, tuple, str | None]] = []
        for index, (completion, words, labels) in enumerate(zip(completions, board_words, board_labels)):
            revealed = board_revealed[index] if board_revealed is not None else ()
            board = _board_from_columns(words, labels, revealed)
            completion_text = _completion_text(completion)
            result = simulate_turn(completion_text, board, guesser)
            shaping = _semantic_margin_bonus(result, board, guesser, safety_weight, blue_weight)
            clue_key = result.clue.word if result.clue else None
            board_key = (tuple(words), tuple(labels), tuple(sorted(revealed)))
            evaluated.append((result, completion_text, shaping, board_key, clue_key))

        clue_counts = Counter(
            (board_key, clue_key)
            for _, _, _, board_key, clue_key in evaluated
            if clue_key is not None
        )
        rewards: list[float] = []
        for index, (result, completion_text, shaping, board_key, clue_key) in enumerate(evaluated):
            duplicate = 0.0
            if clue_key is not None:
                duplicate = duplicate_penalty * max(0, clue_counts[(board_key, clue_key)] - 1)
            reward = float(result.reward + shaping - duplicate)
            if debug:
                print(
                    f"[reward-debug {index}] completion={completion_text!r} "
                    f"game={result.reward:.2f} shaping={shaping:.2f} "
                    f"duplicate=-{duplicate:.2f} total={reward:.2f}"
                )
            rewards.append(reward)
        return rewards

    # TRL uses the callable's name when recording reward metrics.
    codenames_reward.__name__ = "codenames_reward"
    return codenames_reward


def run_reward_smoke_test() -> None:
    """Exercise the exact reward callback shape without importing TRL/Torch."""
    example = build_training_examples(num_boards=1, seed=123)[0]
    board = _board_from_columns(example.board_words, example.board_labels)
    blue_words = [item.word for item in board.words if item.label == "BLUE"]
    reward = make_codenames_reward(OrderedGuesser(blue_words))
    values = reward(
        prompts=[example.prompt],
        completions=["CLUE: galaxy\nNUMBER: 1"],
        board_words=[list(example.board_words)],
        board_labels=[list(example.board_labels)],
    )
    if len(values) != 1 or not isinstance(values[0], float):
        raise AssertionError(f"Unexpected reward callback output: {values!r}")
    print(f"Reward smoke test passed: reward={values[0]:.2f}")


def _load_training_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are not installed. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return torch, Dataset, LoraConfig, AutoTokenizer, GRPOConfig, GRPOTrainer


def build_grpo_config(grpo_config_cls: Any, args: argparse.Namespace, torch_module: Any) -> Any:
    """Construct only arguments supported by the installed GRPOConfig version."""
    cuda_available = bool(torch_module.cuda.is_available())
    bf16_supported = bool(cuda_available and torch_module.cuda.is_bf16_supported())
    use_bf16 = bool(cuda_available and (args.bf16 or bf16_supported))
    model_dtype = torch_module.bfloat16 if use_bf16 else torch_module.float32
    values: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": max(1, args.save_steps),
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        # The non-reentrant implementation avoids the LoRA input-gradient
        # edge case seen with newer PyTorch releases. We also explicitly enable
        # those gradients below for compatibility with older TRL versions.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": use_bf16,
        "fp16": bool(cuda_available and not use_bf16),
        "tf32": bool(args.tf32 and cuda_available),
        # Qwen's published checkpoint is BF16. Explicitly passing the dtype
        # prevents Transformers from expanding it to FP32 unnecessarily.
        "model_init_kwargs": {"dtype": model_dtype},
    }
    supported = set(inspect.signature(grpo_config_cls).parameters)
    filtered = {key: value for key, value in values.items() if key in supported}
    return grpo_config_cls(**filtered)


def train(args: argparse.Namespace) -> None:
    torch, Dataset, LoraConfig, AutoTokenizer, GRPOConfig, GRPOTrainer = _load_training_stack()

    vocabulary = load_vocabulary_file(args.vocabulary_file) if args.vocabulary_file else DEFAULT_VOCABULARY
    examples = build_training_examples(args.num_boards, args.seed, vocabulary=vocabulary,
                                       max_revealed=args.max_revealed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset_rows = []
    for example in examples:
        row = example.as_dict()
        row["prompt"] = format_prompt_for_model(row["prompt"], tokenizer)
        dataset_rows.append(row)
    dataset = Dataset.from_list(dataset_rows)

    config = build_grpo_config(GRPOConfig, args, torch)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    reward = make_codenames_reward(
        EmbeddingGuesser(args.embedding_model, device="cpu", min_similarity=args.min_similarity,
                         abstention_mode=args.abstention_mode),
        debug=args.debug_rewards,
        safety_weight=args.safety_weight,
        blue_weight=args.blue_weight,
        duplicate_penalty=args.duplicate_penalty,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": args.model_name,
        "reward_funcs": reward,
        "args": config,
        "train_dataset": dataset,
        "peft_config": lora_config,
    }
    if args.initial_adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        validate_initial_adapter(args.initial_adapter, args.model_name)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **config.model_init_kwargs)
        trainer_kwargs["model"] = PeftModel.from_pretrained(base_model, args.initial_adapter, is_trainable=True)
        trainer_kwargs.pop("peft_config")
        config.model_init_kwargs = None
    trainer_parameters = inspect.signature(GRPOTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        # Compatibility fallback for older TRL releases.
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError("Installed GRPOTrainer exposes neither processing_class nor tokenizer")

    trainer = GRPOTrainer(**trainer_kwargs)
    resume_checkpoint = find_latest_checkpoint(args.output_dir) if args.resume else None
    if args.resume and resume_checkpoint is None:
        raise FileNotFoundError(
            f"--resume was requested, but no checkpoint-* directory exists in {args.output_dir}"
        )
    # TRL 0.14 constructs its own GenerationConfig and otherwise drops the
    # base checkpoint's top-k/top-p/repetition settings. Restore Qwen's
    # published sampling defaults to avoid repetitive garbage rollouts.
    trainer.generation_config.top_p = args.top_p
    trainer.generation_config.top_k = args.top_k
    trainer.generation_config.repetition_penalty = args.repetition_penalty
    # TRL 0.14 also omits the model's EOS IDs. Without them, Qwen can produce
    # a valid answer and then continue sampling garbage until max length.
    trainer.generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    trainer.generation_config.pad_token_id = tokenizer.pad_token_id
    if args.debug_rewards:
        print(f"Generation config: {trainer.generation_config}")
        debug_inputs = tokenizer(dataset[0]["prompt"], return_tensors="pt").to(trainer.model.device)
        with torch.inference_mode():
            debug_output = trainer.model.generate(
                **debug_inputs,
                generation_config=trainer.generation_config,
            )
        debug_text = tokenizer.decode(
            debug_output[0, debug_inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        print(f"Pre-training model probe: {debug_text!r}")
    if config.gradient_checkpointing:
        # PEFT + gradient checkpointing needs the hidden-state inputs to carry
        # gradients even though the base model weights are frozen.
        if hasattr(trainer.model, "enable_input_require_grads"):
            trainer.model.enable_input_require_grads()
        elif hasattr(trainer.model, "base_model") and hasattr(trainer.model.base_model, "enable_input_require_grads"):
            trainer.model.base_model.enable_input_require_grads()
        if hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = False
    print(
        f"Training {args.model_name} on {len(dataset)} boards for {args.max_steps} steps "
        f"with {args.num_generations} generations per prompt."
    )
    print(f"Board vocabulary: {len(vocabulary)} words from {args.vocabulary_file or 'built-in default'}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)} | BF16: {config.bf16} | embedding guesser: CPU")
    else:
        print("CUDA is unavailable; training will run on CPU and will be very slow.")
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Test board/reward plumbing without model dependencies")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--initial-adapter", type=Path, help="Warm-start GRPO from an SFT LoRA adapter")
    parser.add_argument("--max-revealed", type=int, default=0, help="Randomly reveal up to this many words (0-12)")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    add_abstention_args(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codenames-grpo"))
    parser.add_argument(
        "--vocabulary-file",
        type=Path,
        help="Optional board vocabulary, one word per line; comments beginning with # are ignored",
    )
    parser.add_argument("--num-boards", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint-* directory inside --output-dir",
    )
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=48)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    # TRL 0.14 samples rollouts while the PEFT model is in train mode, so
    # nonzero LoRA dropout would also perturb generation. Keep it deterministic
    # for this small policy; GRPO already supplies exploration through sampling.
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable checkpointing (may corrupt TRL 0.14 rollouts)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast when CUDA is available")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on supported NVIDIA GPUs")
    parser.add_argument("--debug-rewards", action="store_true", help="Print sampled completions and their rewards")
    parser.add_argument("--safety-weight", type=float, default=0.5)
    parser.add_argument("--blue-weight", type=float, default=0.25)
    parser.add_argument("--duplicate-penalty", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.smoke_test:
        run_reward_smoke_test()
    else:
        train(cli_args)
