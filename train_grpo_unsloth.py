"""Optional Unsloth/GRPO trainer for the Codenames experiment.

This lives beside ``train_grpo.py`` so the working Transformers/TRL
environment remains reproducible. Install Unsloth in a separate environment
before running this script.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import shutil
from typing import Any

from codenames_env import DEFAULT_VOCABULARY, EmbeddingGuesser
from train_grpo import (
    DEFAULT_MODEL,
    build_training_examples,
    find_latest_checkpoint,
    format_prompt_for_model,
    load_vocabulary_file,
    make_codenames_reward,
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
    # FlashInfer JIT compilation needs the system CUDA toolkit (nvcc), which
    # is often absent in WSL environments that only have PyTorch CUDA wheels.
    # Unsloth documents this environment variable as the fallback.
    _configure_flashinfer(args.no_flashinfer, args.fast_inference)
    # Patch before importing TRL: this is the order used by Unsloth's GRPO
    # examples and prevents the standard trainer from being constructed first.
    try:
        import torch
        from datasets import Dataset
        from unsloth import FastLanguageModel, PatchFastRL, is_bf16_supported
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth training dependencies are missing. Use a separate environment "
            "and install requirements-unsloth.txt."
        ) from exc

    PatchFastRL("GRPO", FastLanguageModel)
    from trl import GRPOConfig, GRPOTrainer

    vocabulary = load_vocabulary_file(args.vocabulary_file) if args.vocabulary_file else DEFAULT_VOCABULARY
    examples = build_training_examples(
        args.num_boards,
        args.seed,
        vocabulary=vocabulary,
    )
    tokenizer = None
    dataset_rows = []
    for example in examples:
        row = example.as_dict()
        # The tokenizer returned by FastLanguageModel is loaded below. Keeping
        # this loop after model loading lets us use its chat template exactly.
        dataset_rows.append(row)

    use_bf16 = bool(is_bf16_supported())
    max_seq_length = args.max_prompt_length + args.max_completion_length
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for row in dataset_rows:
        row["prompt"] = format_prompt_for_model(row["prompt"], tokenizer)
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
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": False,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": "adamw_8bit",
        "use_vllm": args.fast_inference,
        "repetition_penalty": args.repetition_penalty,
    }
    config = GRPOConfig(**_supported_kwargs(GRPOConfig, config_values))
    reward = make_codenames_reward(
        EmbeddingGuesser(args.embedding_model, device="cpu"),
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
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Unsloth LoRA adapter and tokenizer to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
