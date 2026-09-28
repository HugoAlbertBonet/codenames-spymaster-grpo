"""Render random turns from an evaluation CSV for visual inspection.

The evaluator writes one row per model and board. This script samples rows,
reconstructs the board, and creates a GIF for each selected turn. It also
writes an answer-key PNG with every board label visible, which makes it easier
to understand why a guess earned its reward.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

from codenames_env import BoardWord, CodenamesBoard, generate_board
from render import render_board, render_turn_sequence


def read_evaluation_rows(path: Path) -> list[dict[str, str]]:
    """Read an evaluation CSV and fail clearly when it is empty or malformed."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Evaluation file contains no rows: {path}")
    required = {"model", "board_index", "board_seed", "guesses", "reward"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Evaluation file is missing columns: {', '.join(sorted(missing))}")
    return rows


def _json_list(row: Mapping[str, str], column: str) -> list[str] | None:
    value = row.get(column, "")
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Column {column!r} must contain a JSON list of strings")
    return parsed


def board_from_evaluation_row(row: Mapping[str, str]) -> CodenamesBoard:
    """Build a board from self-contained CSV columns, with legacy fallback.

    Older evaluation files did not include board columns, but they did include
    ``board_seed``. Since board generation is deterministic, those files can
    still be rendered.
    """
    words = _json_list(row, "board_words")
    labels = _json_list(row, "board_labels")
    if words is not None or labels is not None:
        if words is None or labels is None or len(words) != len(labels):
            raise ValueError("board_words and board_labels must be matching JSON lists")
        return CodenamesBoard(tuple(BoardWord(word, label) for word, label in zip(words, labels)))
    return generate_board(seed=int(row["board_seed"]))


def select_rows(
    rows: Sequence[Mapping[str, str]],
    count: int,
    model: str,
    seed: int,
) -> list[Mapping[str, str]]:
    """Select reproducible random rows, optionally pairing base and trained."""
    if count < 1:
        raise ValueError("count must be at least 1")
    grouped: dict[int, dict[str, Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["board_index"]), {})[row["model"]] = row

    if model == "both":
        board_indices = [index for index, items in grouped.items() if {"base", "trained"} <= items.keys()]
    else:
        board_indices = [index for index, items in grouped.items() if model in items]
    if not board_indices:
        raise ValueError(f"No rows found for model={model!r}")

    chosen = random.Random(seed).sample(board_indices, min(count, len(board_indices)))
    selected: list[Mapping[str, str]] = []
    for index in chosen:
        if model == "both":
            selected.extend([grouped[index]["base"], grouped[index]["trained"]])
        else:
            selected.append(grouped[index][model])
    return selected


def render_evaluation_rows(
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    *,
    show_labels: bool = False,
    duration_ms: int = 900,
) -> list[Path]:
    """Render selected rows and return the generated GIF paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_paths: list[Path] = []
    answer_paths: set[Path] = set()
    for row in rows:
        board_index = int(row["board_index"])
        model = row["model"]
        board = board_from_evaluation_row(row)
        guesses = _json_list(row, "guesses") or []
        clue = row.get("clue") or None
        completion = row.get("completion", "")
        number_text = row.get("number", "")
        number = int(number_text) if number_text else None
        reward = float(row["reward"])
        invalid_text = " | INVALID" if row.get("invalid", "").lower() == "true" else ""
        title_prefix = f"{model.upper()} | BOARD {board_index} | REWARD {reward:+.2f}{invalid_text}"
        invalid_reason = row.get("invalid_reason", "")
        if invalid_reason:
            title_prefix = f"{title_prefix}\nWHY: {invalid_reason}"

        answer_path = output_dir / f"board_{board_index:04d}_answer.png"
        if answer_path not in answer_paths and not answer_path.exists():
            render_board(board, [item.word for item in board.words], title=f"BOARD {board_index} | ANSWER KEY", output_path=answer_path, show_labels=True)
        answer_paths.add(answer_path)

        model_dir = output_dir / f"board_{board_index:04d}_{model}"
        gif_path = output_dir / f"board_{board_index:04d}_{model}.gif"
        render_turn_sequence(
            board,
            guesses,
            clue=clue,
            number=number,
            output_dir=model_dir,
            gif_path=gif_path,
            show_labels=show_labels,
            title_prefix=title_prefix,
            duration_ms=duration_ms,
            completion=completion if not clue else None,
        )
        gif_paths.append(gif_path)
    return gif_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-file", type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--count", type=int, default=3, help="Number of random boards to inspect")
    parser.add_argument("--model", choices=("base", "trained", "both"), default="trained")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation-renders"))
    parser.add_argument("--show-labels", action="store_true", help="Show labels on revealed guesses")
    parser.add_argument("--duration-ms", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_evaluation_rows(args.evaluation_file)
    selected = select_rows(rows, args.count, args.model, args.seed)
    gif_paths = render_evaluation_rows(
        selected,
        args.output_dir,
        show_labels=args.show_labels,
        duration_ms=args.duration_ms,
    )
    print(f"Selected {len(gif_paths)} turn(s) from {args.evaluation_file}")
    for path in gif_paths:
        print(f"Saved GIF: {path}")
    print(f"Answer-key PNGs: {args.output_dir}/board_XXXX_answer.png")


if __name__ == "__main__":
    main()
