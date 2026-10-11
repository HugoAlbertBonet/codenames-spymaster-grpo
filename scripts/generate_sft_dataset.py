"""Build 2,000 varied, checked SFT examples from assistant-authored associations.

No teacher API is called. Labels derive from a finite authored association bank,
then undergo deterministic legality checks and a frozen embedding-guesser audit.
This is a synthetic curriculum, not independent frontier-model rollouts on
uniformly random boards. Read data/sft/README.md before training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codenames_env import ASSASSIN, BLUE, RED, NEUTRAL, BoardWord, CodenamesBoard, EmbeddingGuesser, simulate_turn
from sft_data import load_records, validate_record
from train_grpo import load_vocabulary_file


def read_bank(path: Path, excluded: set[str]) -> list[tuple[str, list[str]]]:
    families = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        clue, raw_targets = line.split("\t")
        targets = [word for word in raw_targets.split() if word not in excluded
                   and clue not in word and word not in clue]
        if len(targets) >= 4:
            families.append((clue, targets))
    if len({clue for clue, _ in families}) != len(families):
        raise ValueError("Duplicate clues in association bank")
    if len(families) < 40:
        raise ValueError("Too few usable association families")
    return families


class CachedAuditGuesser:
    """Precompute pool/clue scores instead of encoding every candidate board."""

    def __init__(self, guesser: EmbeddingGuesser, pool: list[str], clues: list[str]):
        vectors = guesser.model.encode(pool, normalize_embeddings=True, batch_size=256)
        clue_vectors = guesser.model.encode(clues, normalize_embeddings=True, batch_size=128)
        self.word_index = {word: i for i, word in enumerate(pool)}
        self.scores = {clue: vector @ vectors.T for clue, vector in zip(clues, clue_vectors)}

    def similarities(self, clue: str, words: list[str]) -> dict[str, float]:
        return {word: float(self.scores[clue][self.word_index[word]]) for word in words}

    def rank(self, clue: str, words: list[str]) -> list[str]:
        scores = self.similarities(clue, words)
        return sorted(words, key=lambda word: -scores[word])


def make_example(clue: str, targets: list[str], pool: list[str], number: int,
                 revealed_count: int, rng: random.Random, guesser: CachedAuditGuesser) -> dict | None:
    intended = rng.sample(targets, number)
    target_scores = guesser.similarities(clue, intended)
    safe_limit = min(target_scores.values()) - 0.015
    distractors = [word for word in pool if word not in targets
                   and word not in clue and clue not in word
                   and float(guesser.scores[clue][guesser.word_index[word]]) < safe_limit]
    if len(distractors) < 16:
        return None
    others = rng.sample(distractors, 16 - number)
    tiles = [BoardWord(word, BLUE) for word in intended]
    labels = [BLUE] * (6 - number) + [RED] * 6 + [NEUTRAL] * 3 + [ASSASSIN]
    rng.shuffle(labels)
    tiles.extend(BoardWord(word, label) for word, label in zip(others, labels))
    rng.shuffle(tiles)
    # Keep both teams alive and never reveal the assassin; these are nonterminal
    # snapshots, not full simulated games with fabricated prior clue histories.
    protected_red = rng.choice([tile.word for tile in tiles if tile.label == RED])
    revealable = [tile.word for tile in tiles if tile.word not in intended
                  and tile.label != ASSASSIN and tile.word != protected_red]
    revealed = tuple(rng.sample(revealable, min(revealed_count, len(revealable))))
    board = CodenamesBoard(tuple(tiles), revealed)
    completion = f"CLUE: {clue}\nNUMBER: {number}"
    active = [tile.word for tile in board.active_words]
    ranked = guesser.rank(clue, active)
    scores = guesser.similarities(clue, active)
    result = simulate_turn(completion, board, guesser)
    # Check all claimed targets, plus the simulator's NUMBER+1 extra guess.
    # A neutral extra costs .3; a RED or ASSASSIN extra rejects the candidate.
    if (result.invalid or set(ranked[:number]) != set(intended)
            or result.reward < number - 0.300001
            or min(scores[word] for word in intended) < 0.20):
        return None
    dangerous = [scores[tile.word] for tile in board.active_words if tile.label != BLUE]
    row = {
        "prompt": board.prompt(), "completion": completion,
        "board_words": [tile.word for tile in tiles],
        "board_labels": [tile.label for tile in tiles],
        "board_revealed": list(revealed), "intended_targets": intended,
        "clue_family": clue, "source": "assistant_authored_associations_with_procedural_boards",
        "audit": {"reward": result.reward, "guesses": list(result.guesses),
                  "guessed_labels": list(result.guessed_labels),
                  "minimum_target_similarity": min(scores[word] for word in intended),
                  "safety_margin": min(scores[word] for word in intended) - max(dangerous)},
    }
    validate_record(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000, help="Total, including held-out validation")
    parser.add_argument("--seed", type=int, default=73000)
    parser.add_argument("--vocabulary-file", type=Path, default=Path("data/training_vocabulary.txt"))
    parser.add_argument("--exclude-file", type=Path, default=Path("data/evaluation_vocabulary_oov.txt"))
    parser.add_argument("--clue-bank", type=Path, default=Path("data/sft_clue_bank.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        train = load_records(args.output_dir / "train.jsonl")
        validation = load_records(args.output_dir / "validation.jsonl")
        if {row["clue_family"] for row in train} & {row["clue_family"] for row in validation}:
            raise ValueError("Clue families overlap between splits")
        rows = train + validation
        signatures = {tuple(sorted(zip(row["board_words"], row["board_labels"]))) for row in rows}
        if len(signatures) != len(rows):
            raise ValueError("Dataset contains repeated boards")
        excluded = set(load_vocabulary_file(args.exclude_file))
        if any(set(row["board_words"]) & excluded for row in rows):
            raise ValueError("Reserved evaluation OOV words appear in the SFT boards")
        print(f"Validated {len(train)} training and {len(validation)} validation records")
        return
    if args.count < 200:
        parser.error("--count must be at least 200 for this curriculum")
    rng = random.Random(args.seed)
    excluded = set(load_vocabulary_file(args.exclude_file))
    families = read_bank(args.clue_bank, excluded)
    pool = sorted((set(load_vocabulary_file(args.vocabulary_file))
                   | {word for _, targets in families for word in targets}) - excluded)
    guesser = CachedAuditGuesser(EmbeddingGuesser(args.embedding_model, device="cpu"),
                                pool, [clue for clue, _ in families])
    # Some human-plausible associations are weak under this particular teammate.
    # Drop those targets before constructing boards rather than filling the
    # requested count with teacher labels the configured guesser cannot follow.
    retained = []
    for clue, targets in families:
        scores = guesser.similarities(clue, targets)
        compatible = [word for word in targets if scores[word] >= 0.20]
        if len(compatible) >= 4:
            retained.append((clue, compatible))
    print(f"Embedding-compatible association families: {len(retained)}/{len(families)}", flush=True)
    families = retained
    if len(families) < 40:
        raise ValueError("Too few compatible families; inspect the association bank and guesser")
    rng.shuffle(families)
    validation_families = families[:max(1, round(len(families) * 0.1))]
    train_families = families[len(validation_families):]
    seen = set()
    splits = {}
    attempts = 0
    for split, bank, count in [("train", train_families, args.count - round(args.count * 0.1)),
                               ("validation", validation_families, round(args.count * 0.1))]:
        records = []
        for index in range(count):
            clue, targets = bank[index % len(bank)]
            number = 1 + index % 4
            revealed_count = (0, 3, 6, 9)[(index // 4) % 4]
            for _ in range(3000):
                attempts += 1
                row = make_example(clue, targets, pool, number, revealed_count, rng, guesser)
                if row is None:
                    continue
                signature = tuple(sorted(zip(row["board_words"], row["board_labels"])))
                if signature in seen:
                    continue
                seen.add(signature)
                row["id"] = f"{split}-{index:05d}"
                records.append(row)
                break
            else:
                raise RuntimeError(f"Cannot validate {clue}/{number}/{revealed_count}; revise this family")
            if (index + 1) % 100 == 0:
                print(f"{split}: {index + 1}/{count}; attempted boards={attempts}", flush=True)
        rng.shuffle(records)
        splits[split] = records
    # Only write after every requested record passed the checks.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "board_vocabulary.txt").write_text("\n".join(pool) + "\n", encoding="utf-8")
    all_rows = splits["train"] + splits["validation"]
    stats = {
        "seed": args.seed, "total": len(all_rows), "train": len(splits["train"]),
        "validation": len(splits["validation"]), "embedding_model": args.embedding_model,
        "association_bank_sha256": hashlib.sha256(args.clue_bank.read_bytes()).hexdigest(),
        "board_pool_sha256": hashlib.sha256("\n".join(pool).encode()).hexdigest(),
        "candidate_boards_attempted": attempts, "unique_clues": len(families),
        "unique_board_words": len({word for row in all_rows for word in row["board_words"]}),
        "board_pool_size": len(pool), "oov_board_overlap": len(set(pool) & excluded),
        "number_counts": dict(Counter(len(row["intended_targets"]) for row in all_rows)),
        "revealed_counts": dict(Counter(len(row["board_revealed"]) for row in all_rows)),
        "clue_counts": dict(Counter(row["clue_family"] for row in all_rows)),
        "mean_embedding_reward": sum(row["audit"]["reward"] for row in all_rows) / len(all_rows),
        "minimum_safety_margin": min(row["audit"]["safety_margin"] for row in all_rows),
        "source": "assistant-authored association bank, procedurally constructed boards",
        "limitations": ["Constructed positive boards; not uniform random-board teacher rollouts",
                        "Embedding compatibility is not human semantic verification",
                        "Guessed-word counts describe snapshots, not full game trajectories",
                        "Validation holds out clue families; separate random/OOV evaluation still needed"],
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in stats.items() if key != "clue_counts"}, indent=2))


if __name__ == "__main__":
    main()
