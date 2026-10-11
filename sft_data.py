"""SFT record validation, shared game-state prompts, and response-only labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codenames_env import BLUE, BoardWord, CodenamesBoard, parse_and_validate_clue


def board_from_record(row: dict[str, Any]) -> CodenamesBoard:
    words, labels = row["board_words"], row["board_labels"]
    if len(words) != len(labels):
        raise ValueError("Board words and labels have different lengths")
    return CodenamesBoard(
        tuple(BoardWord(word, label) for word, label in zip(words, labels)),
        tuple(row.get("board_revealed", [])),
    )


def validate_record(row: dict[str, Any]) -> None:
    board = board_from_record(row)
    clue = parse_and_validate_clue(row["completion"], board)
    targets = row["intended_targets"]
    if len(targets) != clue.number or len(set(targets)) != len(targets):
        raise ValueError("NUMBER must equal the number of distinct intended targets")
    if any(word not in board.by_word or board.by_word[word].label != BLUE
           or word in board.revealed for word in targets):
        raise ValueError("All intended targets must be unguessed BLUE words")
    if row["prompt"] != board.prompt():
        raise ValueError("Dataset prompt differs from the shared environment prompt")


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            validate_record(row)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid SFT record in {path}:{line_number}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty SFT dataset: {path}")
    return rows


def tokenize_record(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Use the model's own chat template; mask every user/prompt token.

    Reject truncation rather than silently clipping the answer or board. The
    assistant completion includes its chat end marker so SFT teaches stopping.
    """
    messages = [{"role": "user", "content": row["prompt"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["completion"]}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if input_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Chat template does not preserve the assistant prefix token boundary")
    if len(input_ids) > max_length:
        raise ValueError(f"Example has {len(input_ids)} tokens, exceeding --max-seq-length={max_length}")
    if len(input_ids) <= len(prefix_ids):
        raise ValueError("Example has no assistant tokens to supervise")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prefix_ids) + input_ids[len(prefix_ids):],
    }
