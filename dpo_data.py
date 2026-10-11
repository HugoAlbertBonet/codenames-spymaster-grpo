"""Validated preference records and exact, response-only DPO tokenization."""

from __future__ import annotations

import json
import math
from pathlib import Path

from codenames_env import ASSASSIN, BLUE, RED, REWARD_BY_LABEL, parse_and_validate_clue
from outcome_data import board_from_row, board_key, file_hash, read_jsonl
from sft_data import tokenize_record
from train_grpo import validate_initial_adapter


def validate_preference(row: dict, pair_mode: str | None = None) -> None:
    pair_mode = pair_mode or row.get("pair_mode", "gameplay")
    if pair_mode not in ("gameplay", "clue-only") or row.get("pair_mode", "gameplay") != pair_mode:
        raise ValueError("Preference pair mode differs from the experiment")
    board = board_from_row(row)
    if row["board_key"] != board_key(row) or row["prompt"] != [{"role": "user", "content": board.prompt()}]:
        raise ValueError("Preference prompt/state mismatch")
    results = {}
    for side in ("chosen", "rejected"):
        messages = row[side]
        if len(messages) != 1 or messages[0]["role"] != "assistant":
            raise ValueError("Expected exactly one assistant message per completion")
        clue = parse_and_validate_clue(messages[0]["content"], board)
        outcome = row[f"{side}_outcome"]
        guesses, labels = outcome["guesses"], outcome["guessed_labels"]
        if outcome["invalid"] or len(guesses) != len(labels) or len(set(guesses)) != len(guesses):
            raise ValueError("Invalid saved preference outcome")
        if len(guesses) > clue.number + 1:
            raise ValueError("Saved outcome exceeds the guess budget")
        if any(word in board.revealed or word not in board.by_word or board.by_word[word].label != label
               for word, label in zip(guesses, labels)):
            raise ValueError("Saved guesses/labels differ from the active board")
        if any(label != BLUE for label in labels[:-1]):
            raise ValueError("Saved turn continued after a non-BLUE guess")
        if not math.isclose(outcome["reward"], sum(REWARD_BY_LABEL[label] for label in labels), abs_tol=1e-8):
            raise ValueError("Saved reward differs from actual game scoring")
        results[side] = (clue, outcome)
    clue, chosen = results["chosen"]
    rejected_clue = results["rejected"][0]
    if pair_mode == "clue-only" and (clue.number != rejected_clue.number or clue.word == rejected_clue.word):
        raise ValueError("Clue-only pairs require different clue words and the same NUMBER")
    if (chosen["reward"] <= 0 or BLUE not in chosen["guessed_labels"] or
            set(chosen["guessed_labels"]) & {RED, ASSASSIN} or
            clue.number > sum(tile.label == BLUE for tile in board.active_words)):
        raise ValueError("Winner fails the safe-positive/feasible-number rule")
    gap = chosen["reward"] - results["rejected"][1]["reward"]
    if gap <= 0 or not math.isclose(gap, row["reward_gap"], abs_tol=1e-8):
        raise ValueError("Preference has no strict reward advantage")


def load_preferences(data_dir: Path, initial_adapter: Path, model_name: str) -> tuple[list[dict], list[dict], dict]:
    """Require an audit certificate covering these exact files and SFT weights."""
    audit_path = data_dir / "audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"Run build_dpo_pairs.py --output-dir {data_dir} --validate-only first")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {"manifest.json", "boards.jsonl", "candidates.jsonl", "train.jsonl", "validation.jsonl"}
    if set(audit["audited_files"]) != required:
        raise ValueError("Audit certificate does not cover all preference artifacts")
    for name, expected in audit["audited_files"].items():
        if file_hash(data_dir / name) != expected:
            raise ValueError(f"Audited artifact changed: {name}; repeat the audit")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    pair_mode = manifest["config"].get("pair_mode", "gameplay")
    if audit.get("pair_mode", "gameplay") != pair_mode:
        raise ValueError("Audit pair mode differs from the dataset")
    if pair_mode == "clue-only" and audit.get("selection_rule") != "safe_positive_same_number_different_clue_v1":
        raise ValueError("Clue-only selection rule was not audited")
    source = manifest["config"]["sampling"]
    if source["base_model"] != model_name:
        raise ValueError("DPO base model differs from the audited sampling policy")
    validate_initial_adapter(initial_adapter, model_name)
    for name, expected in source["adapter_hashes"].items():
        if file_hash(initial_adapter / name) != expected:
            raise ValueError("Initial/reference SFT adapter differs from the policy that produced these preferences")
    train, validation = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "validation.jsonl")
    if not train or not validation:
        raise ValueError("Both preference splits must be nonempty")
    seen = set()
    for split, rows in (("train", train), ("validation", validation)):
        for row in rows:
            validate_preference(row, pair_mode)
            if row["split"] != split or row["board_key"] in seen:
                raise ValueError("Duplicate labeled board or split leakage")
            seen.add(row["board_key"])
    if audit["pairs_replayed"] != len(seen):
        raise ValueError("Audit pair count differs from the data")
    return train, validation, manifest


def tokenize_preference(row: dict, tokenizer, max_length: int) -> dict:
    """Reuse SFT's full chat template, including assistant end markers."""
    result = {}
    for side in ("chosen", "rejected"):
        tokens = tokenize_record({"prompt": row["prompt"][0]["content"],
                                  "completion": row[side][0]["content"]}, tokenizer, max_length)
        result[f"{side}_input_ids"] = tokens["input_ids"]
        result[f"{side}_labels"] = tokens["labels"]
    return result


class PreferenceCollator:
    """Chosen rows followed by rejected rows; prompts and right padding stay masked."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]) -> dict:
        import torch
        sequences = [row[f"{side}_input_ids"] for side in ("chosen", "rejected") for row in rows]
        labels = [row[f"{side}_labels"] for side in ("chosen", "rejected") for row in rows]
        width = max(map(len, sequences))
        result = {
            "input_ids": torch.tensor([ids + [self.pad_token_id] * (width - len(ids)) for ids in sequences]),
            "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]),
            "labels": torch.tensor([ids + [-100] * (width - len(ids)) for ids in labels]),
        }
        if all("ref_chosen_logps" in row and "ref_rejected_logps" in row for row in rows):
            for side in ("chosen", "rejected"):
                result[f"ref_{side}_logps"] = torch.tensor([row[f"ref_{side}_logps"] for row in rows], dtype=torch.float32)
        return result
