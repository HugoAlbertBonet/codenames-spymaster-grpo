"""Render README media from bundled evaluation rows, without loading models.

Use --evaluation-dir results to recapture the curated rows and aggregate metrics
from the original local evaluations. Otherwise the checked-in examples.json is
the input. Frames live in a temporary directory, not in the published gallery.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render import render_board, render_turn_sequence
from render_evaluation import board_from_evaluation_row, read_evaluation_rows


EXAMPLES = (
    ("base", "evaluation-sft-random.csv", 281, "base", "RAW QWEN 1.5B"),
    ("sft", "evaluation-sft-random.csv", 281, "trained", "SFT QWEN 1.5B"),
    ("oov", "evaluation-sft-oov.csv", 51, "trained", "SFT | RESERVED OOV"),
    ("failure", "evaluation-sft-random.csv", 807, "trained", "SFT | ASSASSIN FAILURE"),
)
SUITES = (
    ("evaluation-sft-random.csv", 80000, 0, "data/sft/board_vocabulary.txt", None),
    ("evaluation-sft-partial.csv", 81000, 9, "data/sft/board_vocabulary.txt", None),
    ("evaluation-sft-oov.csv", 82000, 0, "data/evaluation_vocabulary_oov.txt", None),
    ("evaluation-dpo-1000.csv", 170000, 9, "data/training_vocabulary.txt", 0.30),
    ("evaluation-dpo-clue-only.csv", 170000, 9, "data/training_vocabulary.txt", 0.30),
    ("evaluation-sft-grpo-game-only.csv", 170000, 9, "data/training_vocabulary.txt", 0.30),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows: list[dict[str, str]]) -> dict:
    """All rates include invalid turns in their denominator."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    summary = {}
    for model, records in grouped.items():
        n = len(records)
        labels = [json.loads(row["guessed_labels"]) for row in records]
        summary[model] = {
            "episodes": n,
            "average_reward": sum(float(row["reward"]) for row in records) / n,
            "invalid_clue_percent": 100 * sum(row["invalid"].lower() == "true" for row in records) / n,
            "average_correct_guesses": sum(int(row["correct_guesses"]) for row in records) / n,
            "assassin_hit_percent": 100 * sum("ASSASSIN" in guesses for guesses in labels) / n,
            "red_hit_percent": 100 * sum("RED" in guesses for guesses in labels) / n,
        }
    return summary


def capture(evaluation_dir: Path, output_dir: Path) -> None:
    cache = {name: read_evaluation_rows(evaluation_dir / name) for name, *_ in SUITES}
    examples = []
    for name, source, index, model, title in EXAMPLES:
        matching = [row for row in cache[source] if int(row["board_index"]) == index and row["model"] == model]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one {model} row on board {index} in {source}")
        examples.append({
            "name": name, "title": title, "source_csv": source,
            "source_csv_sha256": digest(evaluation_dir / source), "row": matching[0],
        })
    example_data = {
        "selection": "Hand-selected real turns: paired improvement, OOV success, assassin failure. Not a random sample.",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "sft_adapter": "outputs/codenames-sft-1p5b",
        "teammate": "sentence-transformers/all-MiniLM-L6-v2",
        "min_similarity": None, "duration_ms": 1200, "examples": examples,
    }
    summary = {"rate_denominator": "All evaluated turns, including invalid clues", "suites": {}}
    for name, seed, revealed, vocabulary, cutoff in SUITES:
        rows = cache[name]
        by_model = defaultdict(dict)
        for row in rows:
            index = int(row["board_index"])
            if index in by_model[row["model"]]:
                raise ValueError(f"Duplicate evaluation row in {name}")
            by_model[row["model"]][index] = row
        if set(by_model) != {"base", "trained"} or by_model["base"].keys() != by_model["trained"].keys():
            raise ValueError(f"Unpaired evaluation in {name}")
        for index, base in by_model["base"].items():
            trained = by_model["trained"][index]
            for field in ("board_seed", "board_words", "board_labels", "board_revealed"):
                if base[field] != trained[field]:
                    raise ValueError(f"Different paired board states in {name}, board {index}")
        summary["suites"][name] = {
            "sha256": digest(evaluation_dir / name), "seed": seed,
            "max_revealed": revealed, "vocabulary_file": vocabulary,
            "min_similarity": cutoff, "abstention_mode": "all" if cutoff else None,
            "baseline": "raw Qwen" if cutoff is None else "SFT adapter",
            "status": "SFT comparison" if cutoff is None else "reused development suite",
            "metrics": aggregate(rows),
        }
    report_path = evaluation_dir / "abstention-fresh/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary["abstention_confirmation"] = {
        "source": "abstention-fresh/report.json", "sha256": digest(report_path),
        "overall": report["overall"], "suites": report["suites"], "passed": report["passed"],
    }
    summary["sft_run"] = json.loads((ROOT / "outputs/codenames-sft-1p5b/sft_run.json").read_text())
    dpo_run = json.loads((ROOT / "outputs/codenames-dpo-1p5b/dpo_run.json").read_text())
    summary["recorded_dpo_training_versions"] = dpo_run["identity"]["versions"]
    summary["sft_adapter_hashes"] = dpo_run["identity"]["source_adapter_hashes"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "examples.json").write_text(json.dumps(example_data, indent=2) + "\n", encoding="utf-8")
    (output_dir.parent / "evaluation-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def render_examples(output_dir: Path) -> None:
    manifest = json.loads((output_dir / "examples.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="codenames-readme-") as directory:
        for example in manifest["examples"]:
            row = example["row"]
            board = board_from_evaluation_row(row)
            render_turn_sequence(
                board, json.loads(row["guesses"]), clue=row["clue"] or None,
                number=int(row["number"]) if row["number"] else None,
                completion=row["completion"], show_labels=True,
                title_prefix=f"{example['title']} | REWARD {float(row['reward']):+.1f}",
                output_dir=Path(directory) / example["name"],
                gif_path=output_dir / f"{example['name']}.gif", duration_ms=manifest["duration_ms"],
            )
            if example["name"] == "sft":
                render_board(board, [tile.word for tile in board.words],
                             title="SFT EXAMPLE | FULL ANSWER KEY", show_labels=True,
                             output_path=output_dir / "answer.png")
            print(f"Rendered {example['name']}.gif from {example['source_csv']}, board {row['board_index']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, help="Recapture original evaluations and their summaries")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/sft")
    args = parser.parse_args()
    if args.evaluation_dir:
        capture(args.evaluation_dir, args.output_dir)
    render_examples(args.output_dir)


if __name__ == "__main__":
    main()
