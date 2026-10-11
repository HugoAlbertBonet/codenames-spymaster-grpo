"""Response-only SFT LoRA, with an optional 4-bit Unsloth backend.

Start with --dry-run to validate data and token boundaries without loading a
policy model. Fine-tuning uses the shared board prompt, never target annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

from sft_data import load_records, tokenize_record
from train_grpo import find_latest_checkpoint


def calculate_warmup_steps(args: argparse.Namespace, num_examples: int, world_size: int = 1) -> int:
    """Express the 5% warmup as optimizer steps, compatible with Transformers 4/5."""
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        batches_per_epoch = math.ceil(num_examples / (args.batch_size * world_size))
        updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
        total_steps = math.ceil(updates_per_epoch * args.epochs)
    return math.ceil(total_steps * 0.05)


def build_training_arguments(args, num_examples, training_args_cls, *, cuda, bf16, world_size=1):
    """Construct/check the training options before allocating model weights."""
    values = dict(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10, save_strategy="steps", save_steps=args.save_steps,
        eval_steps=args.save_steps, save_total_limit=args.save_total_limit,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=bf16, fp16=bool(cuda and not bf16), optim="adamw_torch",
        lr_scheduler_type="cosine", warmup_steps=calculate_warmup_steps(args, num_examples, world_size),
        weight_decay=0.01, report_to="none", seed=args.seed, data_seed=args.seed,
    )
    supported = inspect.signature(training_args_cls).parameters
    values["eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"] = "steps"
    unsupported = values.keys() - supported.keys()
    if unsupported:
        raise RuntimeError(f"Installed TrainingArguments does not support: {', '.join(sorted(unsupported))}")
    return training_args_cls(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer-name", help="Optional local tokenizer path for offline checks")
    parser.add_argument("--backend", choices=("unsloth", "transformers"), default="unsloth")
    parser.add_argument("--train-file", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/sft/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codenames-sft-1p5b"))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=74000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    train_rows, validation_rows = load_records(args.train_file), load_records(args.validation_file)
    train_boards = {tuple(sorted(zip(row["board_words"], row["board_labels"]))) for row in train_rows}
    if any(tuple(sorted(zip(row["board_words"], row["board_labels"]))) in train_boards for row in validation_rows):
        raise ValueError("A board occurs in both train and validation")
    if {row["clue_family"] for row in train_rows} & {row["clue_family"] for row in validation_rows}:
        raise ValueError("Clue families occur in both train and validation")
    resume = find_latest_checkpoint(args.output_dir) if args.resume else None
    if args.resume and resume is None:
        raise FileNotFoundError(f"No checkpoint-* found in {args.output_dir}")
    if not args.resume and (args.output_dir / "adapter_config.json").exists():
        raise FileExistsError(f"Adapter already exists at {args.output_dir}; choose a new output directory")
    fast_model = None
    if args.backend == "unsloth" and not args.dry_run:
        # Unsloth must patch Transformers before its training imports.
        from unsloth import FastLanguageModel
        fast_model = FastLanguageModel
    import torch
    from transformers import AutoTokenizer, TrainingArguments
    cuda = torch.cuda.is_available()
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    training_args = build_training_arguments(
        args, len(train_rows), TrainingArguments, cuda=cuda, bf16=bf16,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)
    if not tokenizer.chat_template:
        raise ValueError("SFT requires an instruct tokenizer with a chat template")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    def encode(rows, tok):
        return [tokenize_record(row, tok, args.max_seq_length) for row in rows]

    train_tokens, validation_tokens = encode(train_rows, tokenizer), encode(validation_rows, tokenizer)
    token_lengths = [len(row["input_ids"]) for row in train_tokens + validation_tokens]
    supervised = [sum(label != -100 for label in row["labels"]) for row in train_tokens + validation_tokens]
    print(f"SFT: {len(train_rows)} train / {len(validation_rows)} validation; "
          f"max tokens={max(token_lengths)}; assistant tokens={min(supervised)}..{max(supervised)}")
    print(f"Learning-rate warmup: {training_args.warmup_steps} optimizer steps")
    if args.dry_run:
        print("Dry run passed: training options supported, prompts masked, complete assistant responses and end markers supervised")
        return

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, set_seed
    set_seed(args.seed)
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if fast_model is not None:
        model, tokenizer = fast_model.from_pretrained(
            model_name=args.model_name, max_seq_length=args.max_seq_length,
            dtype=dtype, load_in_4bit=args.load_in_4bit, fast_inference=False,
        )
        model = fast_model.get_peft_model(
            model, r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0,
            target_modules=modules, bias="none", use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "right"
        # Train with exactly the tokenizer Unsloth returned, never an assumed one.
        train_tokens, validation_tokens = encode(train_rows, tokenizer), encode(validation_rows, tokenizer)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model_kwargs = {"dtype": dtype}
        if args.load_in_4bit:
            if not cuda:
                raise ValueError("4-bit training requires CUDA; use --no-load-in-4bit for a CPU smoke test")
            from transformers import BitsAndBytesConfig
            model_kwargs.update(
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
                device_map={"": torch.cuda.current_device()},
            )
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0,
            target_modules=modules, bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.print_trainable_parameters()
    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=Dataset.from_list(train_tokens), eval_dataset=Dataset.from_list(validation_tokens),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer_kwargs["processing_class" if "processing_class" in inspect.signature(Trainer).parameters else "tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(
        train_sha256=hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(args.validation_file.read_bytes()).hexdigest(),
        validation_metrics=metrics, best_checkpoint=trainer.state.best_model_checkpoint,
    )
    (args.output_dir / "sft_run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved SFT adapter to {args.output_dir}; evaluate game behavior before GRPO")


if __name__ == "__main__":
    train(parse_args())
