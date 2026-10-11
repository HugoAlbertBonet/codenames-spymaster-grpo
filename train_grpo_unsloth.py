"""Optional Unsloth/GRPO trainer for the Codenames experiment.

This lives beside ``train_grpo.py`` so the working Transformers/TRL
environment remains reproducible. Install Unsloth in a separate environment
before running this script.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

from codenames_env import DEFAULT_VOCABULARY, EmbeddingGuesser, add_abstention_args
from train_grpo import (
    DEFAULT_MODEL,
    build_training_examples,
    find_latest_checkpoint,
    format_prompt_for_model,
    load_vocabulary_file,
    make_codenames_reward,
    validate_initial_adapter,
)


def _configure_flashinfer(no_flashinfer: bool, fast_inference: bool) -> bool:
    """Disable FlashInfer when the CUDA Toolkit is not actually usable.

    PyTorch CUDA wheels provide the runtime needed by torch, but FlashInfer's
    JIT path additionally needs the CUDA compiler and headers. In WSL it is
    possible for ``nvcc`` to appear on PATH while ``CUDA_HOME`` still points to
    a missing directory, so checking only ``shutil.which('nvcc')`` is not
    sufficient.
    """
    cuda_roots: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            cuda_roots.append(Path(value))
    cuda_roots.append(Path("/usr/local/cuda"))

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_root = Path(nvcc).resolve().parent.parent
        cuda_roots.append(nvcc_root)

    usable_root = next(
        (
            root
            for root in cuda_roots
            if (root / "bin" / "nvcc").exists() and (root / "include").exists()
        ),
        None,
    )
    if usable_root is not None:
        os.environ.setdefault("CUDA_HOME", str(usable_root))
        os.environ.setdefault("CUDA_PATH", str(usable_root))

    disable = no_flashinfer or (fast_inference and usable_root is None)
    if disable:
        os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
        # Some vLLM/Unsloth combinations independently enable the FlashInfer
        # sampler even when the Unsloth attention backend is disabled. Force
        # the sampler off as well, otherwise sampling still JIT-compiles CUDA.
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        if os.environ.get("VLLM_ATTENTION_BACKEND") == "FLASHINFER":
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        if fast_inference:
            reason = "--no-flashinfer was supplied" if no_flashinfer else "no usable CUDA Toolkit (nvcc + headers) was found"
            print(f"FlashInfer disabled: {reason}.")
    return disable


def _supported_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(cls).parameters)
    return {key: value for key, value in values.items() if key in supported}


def train(args: argparse.Namespace) -> None:
    if args.beta is not None and (not math.isfinite(args.beta) or args.beta < 0):
        raise ValueError("--beta must be finite and nonnegative")
    if min(args.max_steps, args.num_boards, args.save_steps, args.save_total_limit) < 1:
        raise ValueError("Steps, board count and checkpoint retention must be positive")
    initial_config = validate_initial_adapter(args.initial_adapter, args.model_name) if args.initial_adapter else None
    if args.reference_policy == "sft" and (initial_config is None or args.beta is None or args.beta <= 0):
        raise ValueError("--reference-policy sft requires --initial-adapter and positive --beta")
    if args.game_reward_only:
        args.safety_weight = args.blue_weight = args.duplicate_penalty = 0.0
    # FlashInfer JIT compilation needs the system CUDA toolkit (nvcc), which
    # is often absent in WSL environments that only have PyTorch CUDA wheels.
    # Unsloth documents this environment variable as the fallback.
    _configure_flashinfer(args.no_flashinfer, args.fast_inference)
    # Patch before importing TRL: this is the order used by Unsloth's GRPO
    # examples and prevents the standard trainer from being constructed first.
    try:
        import torch
        from unsloth import FastLanguageModel, PatchFastRL, is_bf16_supported
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth training dependencies are missing. Use a separate environment "
            "and install requirements-unsloth.txt."
        ) from exc

    PatchFastRL("GRPO", FastLanguageModel)
    from trl import GRPOConfig, GRPOTrainer

    vocabulary = load_vocabulary_file(args.vocabulary_file) if args.vocabulary_file else DEFAULT_VOCABULARY
    if args.fresh_boards:
        from outcome_data import (embedding_artifact_hashes, file_hash, fresh_boards,
                                  known_boards, prepare_run)
        if args.vocabulary_file is None:
            raise ValueError("--fresh-boards requires --vocabulary-file")
        run_config = {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items() if key != "resume"}
        run_config["vocabulary_sha256"] = file_hash(args.vocabulary_file)
        run_config["embedding_hashes"] = embedding_artifact_hashes(args.embedding_model)
        run_config["source_hashes"] = {name: file_hash(Path(name)) for name in
                                       ("train_grpo_unsloth.py", "train_grpo.py",
                                        "grpo_reference.py", "codenames_env.py")}
        if args.initial_adapter:
            run_config["initial_adapter_hashes"] = {
                path.name: file_hash(path) for path in
                [args.initial_adapter / "adapter_config.json", *args.initial_adapter.glob("adapter_model.*")]}

        def plan():
            excluded, sources = known_boards()
            return fresh_boards(args.num_boards, args.seed, args.vocabulary_file,
                                args.max_revealed, "grpo-training", excluded), sources

        dataset_rows, _ = prepare_run(args.output_dir, run_config, plan, resume=args.resume)
    else:
        dataset_rows = [example.as_dict() for example in build_training_examples(
            args.num_boards, args.seed, vocabulary=vocabulary, max_revealed=args.max_revealed)]

    use_bf16 = bool(is_bf16_supported())
    max_seq_length = args.max_prompt_length + args.max_completion_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.initial_adapter) if args.initial_adapter else args.model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=initial_config["r"] if initial_config else args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if initial_config is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    else:
        FastLanguageModel.for_training(model)
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Loaded initial adapter has no trainable parameters")
    reference = None
    if args.reference_policy == "sft":
        from grpo_reference import adapter_digest, make_sft_reference
        reference = make_sft_reference(model, args.initial_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for row in dataset_rows:
        row["prompt"] = format_prompt_for_model(row["prompt"], tokenizer)
    longest_prompt = max(len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
                         for row in dataset_rows)
    if longest_prompt > args.max_prompt_length:
        raise ValueError(f"Prompt needs {longest_prompt} tokens; increase --max-prompt-length to avoid truncating rules/boards")
    dataset = Dataset.from_list(dataset_rows)

    config_values = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": max(1, args.save_steps),
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": False,
        "lr_scheduler_type": "cosine",
        "warmup_steps": math.ceil(args.max_steps * 0.1),
        "seed": args.seed,
        "data_seed": args.seed,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": "adamw_8bit",
        "use_vllm": args.fast_inference,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.beta is not None:
        config_values["beta"] = args.beta
    config = GRPOConfig(**_supported_kwargs(GRPOConfig, config_values))
    reward = make_codenames_reward(
        EmbeddingGuesser(args.embedding_model, device="cpu", min_similarity=args.min_similarity,
                         abstention_mode=args.abstention_mode),
        debug=args.debug_rewards,
        safety_weight=args.safety_weight,
        blue_weight=args.blue_weight,
        duplicate_penalty=args.duplicate_penalty,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": reward,
        "args": config,
        "train_dataset": dataset,
    }
    trainer_parameters = inspect.signature(GRPOTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        # Unsloth wraps GRPOTrainer with a ``(*args, **kwargs)`` signature,
        # so introspection cannot see the forwarded processing_class keyword.
        # Current Unsloth GRPO examples use processing_class.
        trainer_kwargs["processing_class"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    if reference is not None:
        if trainer.beta != args.beta or getattr(config, "sync_ref_model", False):
            raise RuntimeError("SFT reference/KL configuration was not applied")
        # TRL 0.24's PEFT default disables LoRA, yielding the raw base policy.
        # Override that with a frozen copy of the SFT adapter. Forward delegates
        # to the already Unsloth-patched policy, including hidden-state scoring.
        trainer.ref_model = reference
        reference.audit()
        print(f"KL reference: frozen SFT adapter | beta={trainer.beta}")
    # Standard TRL exposes generation_config, while recent Unsloth wraps the
    # trainer without forwarding that attribute. The sampling values above are
    # already in GRPOConfig; only apply the extra token IDs when the attribute
    # exists.
    generation_config = getattr(trainer, "generation_config", None)
    if generation_config is not None:
        generation_config.repetition_penalty = args.repetition_penalty
        generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        generation_config.pad_token_id = tokenizer.pad_token_id

    resume_checkpoint = find_latest_checkpoint(args.output_dir) if args.resume else None
    if args.resume and resume_checkpoint is None:
        raise FileNotFoundError(
            f"--resume was requested, but no checkpoint-* directory exists in {args.output_dir}"
        )
    print(
        f"Unsloth training {args.model_name} on {len(dataset)} boards for {args.max_steps} steps "
        f"with {args.num_generations} generations per prompt."
    )
    print(f"Board vocabulary: {len(vocabulary)} words from {args.vocabulary_file or 'built-in default'}")
    print(f"Fast inference/vLLM: {args.fast_inference} | 4-bit: {args.load_in_4bit} | BF16: {use_bf16}")
    print(f"Longest prompt: {longest_prompt} tokens (limit {args.max_prompt_length})")
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    if reference is not None:
        audit = reference.audit()
        if audit["forward_calls"] == 0:
            raise RuntimeError("Trainer did not use the SFT reference")
        audit["policy_adapter_sha256"] = adapter_digest(model, "default")
        audit["beta"] = trainer.beta
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "reference_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Unsloth LoRA adapter and tokenizer to {args.output_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--initial-adapter", type=Path, help="Warm-start GRPO from an SFT LoRA adapter")
    parser.add_argument("--reference-policy", choices=("base", "sft"), default="base",
                        help="KL reference: raw base model or a frozen copy of the initial SFT adapter")
    parser.add_argument("--beta", type=float, help="Explicit GRPO KL coefficient")
    parser.add_argument("--game-reward-only", action="store_true",
                        help="Disable semantic bonuses and duplicate penalties")
    parser.add_argument("--fresh-boards", action="store_true",
                        help="Save a reproducible board plan excluding prior SFT/DPO/evaluation boards")
    parser.add_argument("--max-revealed", type=int, default=0, help="Randomly reveal up to this many words (0-12)")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    add_abstention_args(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codenames-unsloth"))
    parser.add_argument(
        "--vocabulary-file",
        type=Path,
        help="Optional board vocabulary, one word per line; comments beginning with # are ignored",
    )
    parser.add_argument("--num-boards", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=48)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-inference", action="store_true", help="Use Unsloth's vLLM-backed rollout path")
    parser.add_argument("--no-flashinfer", action="store_true", help="Disable FlashInfer JIT kernels")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--debug-rewards", action="store_true")
    parser.add_argument("--safety-weight", type=float, default=0.5)
    parser.add_argument("--blue-weight", type=float, default=0.25)
    parser.add_argument("--duplicate-penalty", type=float, default=0.25)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
