"""SFT-initialized LoRA DPO with frozen, cached SFT reference scores.

Default: 4-bit Unsloth on one CUDA GPU. No new adapter is attached, no raw-base
reference is used, and gameplay/board metadata never enters the model's loss.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path

from dpo_data import PreferenceCollator, load_preferences, tokenize_preference
from dpo_loss import forward_response_logps, make_dpo_trainer_class, preference_metrics
from outcome_data import file_hash
from train_grpo import find_latest_checkpoint
from train_sft import calculate_warmup_steps


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_args(args) -> None:
    for name in ("epochs", "learning_rate", "beta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.max_steps != -1 and args.max_steps < 1:
        raise ValueError("max_steps must be -1 or positive")
    for name in ("batch_size", "gradient_accumulation_steps", "reference_batch_size", "save_steps",
                 "save_total_limit", "max_seq_length"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_train_pairs", "max_validation_pairs", "stop_after_steps"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")


def build_dpo_arguments(args, count, cls, *, cuda, bf16):
    """Use explicit warmup steps, supported by both installed Transformers versions."""
    values = dict(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1, save_strategy="steps", save_steps=args.save_steps, eval_steps=args.save_steps,
        save_total_limit=args.save_total_limit, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, report_to="none",
        bf16=bf16, fp16=bool(cuda and not bf16), optim="adamw_torch", weight_decay=0.0,
        lr_scheduler_type="cosine", warmup_steps=calculate_warmup_steps(args, count),
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
        # Forward labels are masks for our loss, not a request for model SFT loss.
        label_names=[], gradient_checkpointing=bool(args.gradient_checkpointing and args.backend == "transformers"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    supported = inspect.signature(cls).parameters
    values["eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"] = "steps"
    unsupported = values.keys() - supported.keys()
    if unsupported:
        raise RuntimeError(f"Unsupported TrainingArguments: {', '.join(sorted(unsupported))}")
    return cls(**values)


def run_identity(args, manifest, tokenizer, train_tokens, validation_tokens, dtype) -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}
    if args.backend == "unsloth":
        versions["unsloth"] = importlib.metadata.version("unsloth")
    settings = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
                if name not in {"resume", "dry_run", "stop_after_steps", "output_dir"}}
    return {"schema_version": 1, "objective": "sigmoid_dpo_summed_response_logps_v1", "settings": settings,
            "versions": versions, "dtype": str(dtype),
            "train_sha256": file_hash(args.data_dir / "train.jsonl"),
            "validation_sha256": file_hash(args.data_dir / "validation.jsonl"),
            "audit_sha256": file_hash(args.data_dir / "audit.json"),
            "source_adapter_hashes": manifest["config"]["sampling"]["adapter_hashes"],
            "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "tokens_sha256": hashlib.sha256(json.dumps([train_tokens, validation_tokens], sort_keys=True).encode()).hexdigest(),
            "teammate": {key: manifest["config"][key] for key in ("embedding_model", "min_similarity", "abstention_mode")}}


def validate_reference_cache(cache: dict, identity: dict, split_sizes: dict) -> None:
    if cache["identity"] != identity:
        raise ValueError("Reference cache provenance differs from this run")
    if set(cache["logps"]) != set(split_sizes):
        raise ValueError("Reference cache split names differ")
    for split, size in split_sizes.items():
        entries = cache["logps"][split]
        if len(entries) > size:
            raise ValueError("Reference cache has too many pairs")
        for entry in entries:
            if len(entry) != 2 or any(not math.isfinite(value) or value > 1e-5 for value in entry):
                raise ValueError("Invalid reference response log probabilities")


def precompute_reference(model, splits, collator, cache_path, identity, batch_size, dtype):
    """Freeze SFT scores before updates; an interrupted cache resumes by pair index."""
    import torch
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {
        "identity": identity, "logps": {split: [] for split in splits}}
    validate_reference_cache(cache, identity, {key: len(rows) for key, rows in splits.items()})
    model.eval()
    device = next(model.parameters()).device
    for split, rows in splits.items():
        entries = cache["logps"][split]
        for offset in range(len(entries), len(rows), batch_size):
            selected = rows[offset:offset + batch_size]
            batch = {key: value.to(device) for key, value in collator(selected).items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logps = forward_response_logps(model, batch).cpu().tolist()
            size = len(selected)
            entries.extend([[logps[i], logps[size + i]] for i in range(size)])
            validate_reference_cache(cache, identity, {key: len(values) for key, values in splits.items()})
            atomic_json(cache_path, cache)
            if len(entries) % 20 == 0 or len(entries) == len(rows):
                print(f"Frozen SFT reference: {split} {len(entries)}/{len(rows)} pairs", flush=True)
    for split, rows in splits.items():
        for row, (chosen, rejected) in zip(rows, cache["logps"][split]):
            row["ref_chosen_logps"], row["ref_rejected_logps"] = chosen, rejected
    return cache


def trainable_hash(model) -> str:
    """Detect inadvertent updates during the no-grad reference phase."""
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"Expected only LoRA trainable parameters, found {name}")
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train(args) -> None:
    validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This laptop DPO trainer currently supports a single process/GPU")
    train_rows, validation_rows, manifest = load_preferences(args.data_dir, args.initial_adapter, args.model_name)
    if args.max_train_pairs:
        train_rows = train_rows[:args.max_train_pairs]
    if args.max_validation_pairs:
        validation_rows = validation_rows[:args.max_validation_pairs]
    if (args.output_dir.resolve() == args.initial_adapter.resolve() or
            args.initial_adapter.resolve() in args.output_dir.resolve().parents):
        raise ValueError("DPO output must not overwrite the SFT reference adapter")
    if not args.dry_run:
        if args.resume:
            if not (args.output_dir / "run_config.json").exists():
                raise FileNotFoundError("No DPO run configuration to resume")
        elif args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("DPO output directory is nonempty; choose a new directory or --resume")

    fast_model = None
    if args.backend == "unsloth" and not args.dry_run:
        from unsloth import FastLanguageModel
        fast_model = FastLanguageModel
    import torch
    from transformers import AutoTokenizer, TrainingArguments, set_seed
    torch.set_num_threads(4)
    cuda = torch.cuda.is_available()
    if cuda and not args.dry_run:
        torch.cuda.reset_peak_memory_stats()
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    training_args = build_dpo_arguments(args, len(train_rows), TrainingArguments, cuda=cuda, bf16=bf16)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.initial_adapter)
    if not tokenizer.chat_template:
        raise ValueError("DPO requires the SFT instruct tokenizer's chat template")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    def encode(tok):
        return {split: [tokenize_preference(row, tok, args.max_seq_length) for row in rows]
                for split, rows in (("train", train_rows), ("validation", validation_rows))}
    splits = encode(tokenizer)
    lengths = [len(row[f"{side}_input_ids"]) for rows in splits.values() for row in rows for side in ("chosen", "rejected")]
    print(f"DPO: {len(train_rows)} train / {len(validation_rows)} validation pairs; max tokens={max(lengths)}; "
          f"beta={args.beta}; lr={args.learning_rate}; reference=SFT adapter; "
          f"pair mode={manifest['config'].get('pair_mode', 'gameplay')}", flush=True)
    if args.dry_run:
        print("Dry run passed: audited data, matching SFT weights, full responses/end markers and prompt masks validated")
        return
    if not cuda and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable: run in your GPU terminal, or explicitly use --allow-cpu")
    if not cuda and (args.backend == "unsloth" or args.load_in_4bit):
        raise ValueError("CPU smoke tests require --backend transformers --no-load-in-4bit --allow-cpu")
    set_seed(args.seed)
    if fast_model is not None:
        model, actual_tokenizer = fast_model.from_pretrained(
            model_name=str(args.initial_adapter), max_seq_length=args.max_seq_length,
            dtype=dtype, load_in_4bit=args.load_in_4bit, fast_inference=False)
        fast_model.for_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        actual_tokenizer.pad_token = actual_tokenizer.pad_token or actual_tokenizer.eos_token
        actual_tokenizer.padding_side = "right"
        actual_splits = encode(actual_tokenizer)
        if actual_splits != splits:
            raise ValueError("Loaded model tokenizer differs from the validated SFT tokenizer")
        tokenizer = actual_tokenizer
    else:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training
        kwargs = {"dtype": dtype}
        if args.load_in_4bit:
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
                          device_map={"": torch.cuda.current_device()})
        base = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
        if args.load_in_4bit:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(base, args.initial_adapter, is_trainable=True)
        model.to(torch.device("cuda" if cuda else "cpu"))
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("SFT adapter has no trainable LoRA parameters")
    model.print_trainable_parameters()
    identity = run_identity(args, manifest, tokenizer, splits["train"], splits["validation"], dtype)
    lock_path = args.output_dir / "run_config.json"
    if args.resume:
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        if locked["identity"] != identity:
            raise ValueError("Resume settings, data, tokenizer, model or installed versions changed")
        expected_ref_hash = locked.get("reference_cache_sha256")
        if expected_ref_hash and file_hash(args.output_dir / "reference_logps.json") != expected_ref_hash:
            raise ValueError("Frozen reference cache changed after training began")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        locked = {"identity": identity, "status": "reference_precompute"}
        atomic_json(lock_path, locked)
    collator = PreferenceCollator(tokenizer.pad_token_id)
    before_reference = trainable_hash(model)
    precompute_reference(model, splits, collator, args.output_dir / "reference_logps.json", identity,
                         args.reference_batch_size, dtype)
    if trainable_hash(model) != before_reference:
        raise RuntimeError("Reference precomputation changed SFT adapter weights")
    reference_hash = file_hash(args.output_dir / "reference_logps.json")
    locked.update(reference_cache_sha256=reference_hash, status="training")
    atomic_json(lock_path, locked)
    if fast_model is not None:
        fast_model.for_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.config.use_cache = False
    from datasets import Dataset
    from transformers import TrainerCallback
    Trainer = make_dpo_trainer_class()
    kwargs = dict(model=model, args=training_args, beta=args.beta, data_collator=collator,
                  train_dataset=Dataset.from_list(splits["train"]), eval_dataset=Dataset.from_list(splits["validation"]),
                  compute_metrics=preference_metrics)
    from transformers import Trainer as BaseTrainer
    kwargs["processing_class" if "processing_class" in inspect.signature(BaseTrainer).parameters else "tokenizer"] = tokenizer
    class VerifyUpdates(TrainerCallback):
        changed = False
        def on_step_end(self, training_args, state, control, model=None, **_):
            if not self.changed:
                self.changed = trainable_hash(model) != before_reference
            return control
    update_check = VerifyUpdates()
    kwargs["callbacks"] = [update_check]
    if args.stop_after_steps:
        class StopAfterSteps(TrainerCallback):
            def on_step_end(self, training_args, state, control, **_):
                if state.global_step >= args.stop_after_steps:
                    control.should_training_stop = True
                    control.should_save = True
                    control.should_evaluate = True
                return control
        kwargs["callbacks"].append(StopAfterSteps())
    trainer = Trainer(**kwargs)
    resume = find_latest_checkpoint(args.output_dir) if args.resume else None
    if args.resume and resume is None:
        print("No saved optimizer checkpoint: restarting training at step 0, reusing frozen reference scores", flush=True)
    if resume:
        print(f"Resuming DPO optimizer/scheduler/RNG from {resume}; reference remains original SFT", flush=True)
    baseline = locked.get("initial_metrics")
    previous_report = args.output_dir / "dpo_run.json"
    if baseline is None and resume and previous_report.exists():
        baseline = json.loads(previous_report.read_text(encoding="utf-8")).get("initial_metrics")
    if not resume:
        baseline = trainer.evaluate(metric_key_prefix="initial")
    if baseline is not None:
        locked["initial_metrics"] = baseline
        atomic_json(lock_path, locked)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    metrics = trainer.evaluate()
    if file_hash(args.output_dir / "reference_logps.json") != reference_hash:
        raise RuntimeError("Reference cache was modified during training")
    final_hash = trainable_hash(trainer.model)
    if not update_check.changed:
        print("Warning: no LoRA update observed (a warmup-only stop can have zero learning rate)", flush=True)
    if final_hash == before_reference:
        print("Exported best validation checkpoint equals the initial SFT weights; no improvement established", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    complete = trainer.state.global_step >= trainer.state.max_steps
    locked.update(status="complete" if complete else "stopped_at_checkpoint")
    atomic_json(lock_path, locked)
    atomic_json(args.output_dir / "dpo_run.json", {
        "identity": identity, "initial_metrics": baseline, "validation_metrics": metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint, "global_step": trainer.state.global_step,
        "status": locked["status"], "reference_cache_sha256": reference_hash,
        "initial_trainable_sha256": before_reference, "exported_trainable_sha256": final_hash,
        "optimizer_updated_lora": update_check.changed,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if cuda else None,
    })
    print(f"Saved DPO adapter to {args.output_dir}; {locked['status']}; evaluate against SFT with the same teammate")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dpo"))
    parser.add_argument("--initial-adapter", type=Path, default=Path("outputs/codenames-sft-1p5b"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer-name", help="Optional local tokenizer path")
    parser.add_argument("--backend", choices=("unsloth", "transformers"), default="unsloth")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codenames-dpo-1p5b"))
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1, help="Pairs per device batch (two responses per pair)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--reference-batch-size", type=int, default=1,
                        help="Pairs per reference batch; 1 matches default train/eval padding and numerical behavior")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=160_000)
    parser.add_argument("--max-train-pairs", type=int, help="Limit pairs for a smoke test")
    parser.add_argument("--max-validation-pairs", type=int, help="Limit pairs for a smoke test")
    parser.add_argument("--stop-after-steps", type=int, help="Checkpoint and stop early without changing the planned LR schedule")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
