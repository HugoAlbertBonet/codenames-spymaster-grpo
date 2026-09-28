"""Evaluate the base and GRPO-trained Codenames spymasters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from codenames_env import (
    EmbeddingGuesser,
    Guesser,
    LLMGuesser,
    TurnResult,
    CodenamesBoard,
    BoardWord,
    DEFAULT_VOCABULARY,
    simulate_turn,
)
from train_grpo import (
    TrainingExample,
    build_training_examples,
    format_prompt_for_model,
    load_vocabulary_file,
)


@dataclass(frozen=True)
class EvaluationRecord:
    model: str
    board_index: int
    board_seed: int
    completion: str
    clue: str
    number: int | None
    guesses: tuple[str, ...]
    guessed_labels: tuple[str, ...]
    reward: float
    invalid: bool
    invalid_reason: str | None
    correct_guesses: int
    board_words: tuple[str, ...]
    board_labels: tuple[str, ...]

    @classmethod
    def from_turn(
        cls,
        model: str,
        board_index: int,
        board_seed: int,
        result: TurnResult,
        board: CodenamesBoard,
    ) -> "EvaluationRecord":
        return cls(
            model=model,
            board_index=board_index,
            board_seed=board_seed,
            completion=result.completion,
            clue=result.clue.word if result.clue else "",
            number=result.clue.number if result.clue else None,
            guesses=result.guesses,
            guessed_labels=result.guessed_labels,
            reward=result.reward,
            invalid=result.invalid,
            invalid_reason=result.invalid_reason,
            correct_guesses=sum(label == "BLUE" for label in result.guessed_labels),
            board_words=tuple(item.word for item in board.words),
            board_labels=tuple(item.label for item in board.words),
        )


def board_from_example(example: TrainingExample) -> CodenamesBoard:
    return CodenamesBoard(
        tuple(BoardWord(word, label) for word, label in zip(example.board_words, example.board_labels))
    )


def evaluate_completions(
    examples: Sequence[TrainingExample],
    completion_fn: Callable[[str], str],
    model_name: str,
    guesser: Guesser,
    seed: int,
) -> list[EvaluationRecord]:
    """Run identical boards through an injected spymaster completion function."""
    records: list[EvaluationRecord] = []
    for board_index, example in enumerate(examples):
        completion = completion_fn(example.prompt)
        result = simulate_turn(completion, board_from_example(example), guesser)
        records.append(
            EvaluationRecord.from_turn(
                model_name, board_index, seed + board_index, result, board_from_example(example)
            )
        )
    return records


def summarize_records(records: Sequence[EvaluationRecord]) -> dict[str, float | int | str]:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    return {
        "model": records[0].model,
        "episodes": len(records),
        "average_reward": sum(record.reward for record in records) / len(records),
        "invalid_clue_percent": 100.0 * sum(record.invalid for record in records) / len(records),
        "average_correct_guesses": sum(record.correct_guesses for record in records) / len(records),
    }


CSV_FIELDS = (
    "model", "board_index", "board_seed", "completion", "clue", "number",
    "guesses", "guessed_labels", "reward", "invalid", "invalid_reason", "correct_guesses",
    "board_words", "board_labels",
)


def write_csv(records: Sequence[EvaluationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {
                "model": record.model,
                "board_index": record.board_index,
                "board_seed": record.board_seed,
                "completion": record.completion,
                "clue": record.clue,
                "number": record.number if record.number is not None else "",
                "guesses": json.dumps(record.guesses),
                "guessed_labels": json.dumps(record.guessed_labels),
                "reward": record.reward,
                "invalid": record.invalid,
                "invalid_reason": record.invalid_reason or "",
                "correct_guesses": record.correct_guesses,
                "board_words": json.dumps(record.board_words),
                "board_labels": json.dumps(record.board_labels),
            }
            writer.writerow(row)


def write_transcripts(
    base_records: Sequence[EvaluationRecord],
    trained_records: Sequence[EvaluationRecord],
    examples: Sequence[TrainingExample],
    path: Path,
    limit: int = 3,
) -> None:
    """Save boards where the trained result is most visibly better."""
    trained_by_index = {record.board_index: record for record in trained_records}
    base_by_index = {record.board_index: record for record in base_records}
    ranked_indices = sorted(
        base_by_index,
        key=lambda index: (
            trained_by_index[index].reward - base_by_index[index].reward,
            trained_by_index[index].correct_guesses - base_by_index[index].correct_guesses,
        ),
        reverse=True,
    )[:limit]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in ranked_indices:
            base = base_by_index[index]
            trained = trained_by_index[index]
            handle.write(f"BOARD {index} (seed {base.board_seed})\n")
            handle.write(" | ".join(f"{item.word}={item.label}" for item in board_from_example(examples[index]).words))
            handle.write("\n\n")
            for record in (base, trained):
                handle.write(f"{record.model}:\n")
                handle.write(f"  completion: {record.completion!r}\n")
                handle.write(f"  guesses: {list(record.guesses)}\n")
                handle.write(f"  labels: {list(record.guessed_labels)}\n")
                handle.write(f"  reward: {record.reward:.2f}\n\n")


def _load_generation_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation dependencies are not installed. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    adapter_dir: Path | None = None,
    *,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, Any]:
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _load_generation_stack()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        # Keep compatibility with older Transformers releases that do not
        # accept ``attn_implementation`` in ``from_pretrained``.
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if attn_implementation is not None and hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(attn_implementation)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def model_completion_fn(model: Any, tokenizer: Any, device: Any, max_new_tokens: int) -> Callable[[str], str]:
    from transformers import GenerationConfig

    # Evaluation is greedy. Supplying a clean config prevents Transformers from
    # validating the training-time sampling fields inherited from Qwen.
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.1,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        pad_token_id=tokenizer.pad_token_id,
    )

    def complete(prompt: str) -> str:
        formatted_prompt = format_prompt_for_model(prompt, tokenizer)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[-1]
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
        )
        return tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()

    return complete


def evaluate_checkpoint(
    model_name: str,
    adapter_dir: Path | None,
    examples: Sequence[TrainingExample],
    model_label: str,
    guesser: Guesser,
    seed: int,
    max_new_tokens: int,
) -> list[EvaluationRecord]:
    model, tokenizer, device = load_model_and_tokenizer(model_name, adapter_dir)
    completion_fn = model_completion_fn(model, tokenizer, device, max_new_tokens)
    records = evaluate_completions(examples, completion_fn, model_label, guesser, seed)
    del model
    return records


def print_summary(records: Sequence[EvaluationRecord]) -> None:
    summary = summarize_records(records)
    print(
        f"{summary['model']:<8} | episodes={summary['episodes']} | "
        f"reward={summary['average_reward']:.3f} | "
        f"invalid={summary['invalid_clue_percent']:.1f}% | "
        f"correct guesses/turn={summary['average_correct_guesses']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/codenames-grpo"))
    parser.add_argument("--num-boards", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--guesser", choices=("embedding", "llm"), default="embedding")
    parser.add_argument("--llm-guesser-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--llm-guesser-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--llm-guesser-max-new-tokens", type=int, default=64)
    parser.add_argument("--vocabulary-file", type=Path, help="Optional board vocabulary, one word per line")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-csv", type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--transcript-file", type=Path, default=Path("results/transcripts.txt"))
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    if not args.adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {args.adapter_dir}. Run train_grpo.py first."
        )
    vocabulary = load_vocabulary_file(args.vocabulary_file) if args.vocabulary_file else DEFAULT_VOCABULARY
    examples = build_training_examples(args.num_boards, args.seed, vocabulary=vocabulary)
    if args.guesser == "embedding":
        guesser: Guesser = EmbeddingGuesser(args.embedding_model, device="cpu")
    else:
        guesser = LLMGuesser(
            model_name=args.llm_guesser_model,
            device=args.llm_guesser_device,
            max_new_tokens=args.llm_guesser_max_new_tokens,
        )
    base_records = evaluate_checkpoint(
        args.base_model, None, examples, "base", guesser, args.seed, args.max_new_tokens
    )
    trained_records = evaluate_checkpoint(
        args.base_model, args.adapter_dir, examples, "trained", guesser, args.seed, args.max_new_tokens
    )
    all_records = base_records + trained_records
    write_csv(all_records, args.output_csv)
    write_transcripts(base_records, trained_records, examples, args.transcript_file)
    print(f"Guesser: {args.guesser}")
    print_summary(base_records)
    print_summary(trained_records)
    print(f"Saved per-episode results to {args.output_csv}")
    print(f"Saved example transcripts to {args.transcript_file}")


if __name__ == "__main__":
    main()
