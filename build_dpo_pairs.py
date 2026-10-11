"""Build offline DPO preferences from real SFT clue outcomes under a confirmed teammate.

No teacher API, declared-target scoring, word substitutions or BLUE-only sampling.
Each pair consists of two complete clue+NUMBER responses to the same board.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

from codenames_env import (BLUE, RED, ASSASSIN, EmbeddingGuesser, add_abstention_args,
                          parse_clue, simulate_turn, validate_abstention)
from outcome_data import (add_sampling_args, board_from_row, board_key, collect_candidates,
                          file_hash, fresh_boards, known_boards, prepare_run, read_jsonl, embedding_artifact_hashes,
                          prime_embeddings, sampling_config, write_json, write_jsonl)
from train_grpo import load_vocabulary_file


def outcome_dict(result) -> dict:
    return {"reward": result.reward, "guesses": list(result.guesses),
            "guessed_labels": list(result.guessed_labels), "invalid": result.invalid,
            "invalid_reason": result.invalid_reason, "abstained": result.abstained,
            "stop_reason": result.stop_reason, "guess_similarities": list(result.guess_similarities)}


PAIR_MODES = ("gameplay", "clue-only")


def selection_rule(pair_mode: str) -> str:
    if pair_mode not in PAIR_MODES:
        raise ValueError(f"Unknown pair mode: {pair_mode}")
    return ("safe_positive_same_number_different_clue_v1" if pair_mode == "clue-only"
            else "safe_positive_feasible_number_v2")


def choose_pair(row: dict, guesser, min_reward_gap: float = 1.0,
                pair_mode: str = "gameplay") -> tuple[dict | None, str]:
    """Keep one safe winner and a lower-scoring gameplay alternative per board."""
    selection_rule(pair_mode)
    if not math.isfinite(min_reward_gap) or min_reward_gap <= 0:
        raise ValueError("min_reward_gap must be positive")
    board = board_from_row(row)
    seen, scored = set(), []
    for completion in row["completions"]:
        result = simulate_turn(completion, board, guesser)
        if result.invalid:
            continue  # Gameplay DPO, not a second formatting curriculum.
        key = (result.clue.word, result.clue.number)
        if key in seen:
            continue
        seen.add(key)
        scored.append(result)
    remaining_blue = sum(tile.label == BLUE for tile in board.active_words)
    safe = [result for result in scored if result.reward > 0 and BLUE in result.guessed_labels
            and result.clue.number <= remaining_blue
            and RED not in result.guessed_labels and ASSASSIN not in result.guessed_labels]
    if not safe:
        return None, "no_safe_positive_winner"
    # A lower-scoring safe winner may have an eligible same-number comparison
    # even when the global best winner does not. Search all eligible winners.
    ordered = sorted(safe, key=lambda result: (result.reward, result.guessed_labels.count(BLUE)), reverse=True)
    for chosen in ordered if pair_mode == "clue-only" else ordered[:1]:
        losers = [result for result in scored if chosen.reward - result.reward >= min_reward_gap - 1e-9
                  and (pair_mode == "gameplay" or
                       (result.clue.number == chosen.clue.number and result.clue.word != chosen.clue.word))]
        if losers:
            break
    if not losers:
        return None, ("no_same_number_clue_preference" if pair_mode == "clue-only"
                      else "no_clear_gameplay_preference")
    dangerous = [result for result in losers if RED in result.guessed_labels or ASSASSIN in result.guessed_labels]
    # Closest clearly inferior dangerous alternative is a hard negative; no need for all pair combinations.
    rejected = max(dangerous or losers, key=lambda result: result.reward)
    pair = {key: row[key] for key in ("board_words", "board_labels", "board_revealed", "board_seed", "board_key")}
    pair.update({"prompt": [{"role": "user", "content": row["prompt"]}],
                 "chosen": [{"role": "assistant", "content": chosen.completion}],
                 "rejected": [{"role": "assistant", "content": rejected.completion}],
                 "chosen_outcome": outcome_dict(chosen), "rejected_outcome": outcome_dict(rejected),
                 "reward_gap": chosen.reward - rejected.reward, "split": row["suite"],
                 "source": "sft_policy_samples_scored_by_fixed_embedding_teammate",
                 "pair_mode": pair_mode})
    return pair, "kept"


def validate_pair(pair: dict, guesser, min_reward_gap: float, pair_mode: str | None = None) -> None:
    pair_mode = pair_mode or pair.get("pair_mode", "gameplay")
    selection_rule(pair_mode)
    if pair.get("pair_mode", "gameplay") != pair_mode:
        raise ValueError("Pair mode differs from the experiment")
    board = board_from_row(pair)
    if pair["board_key"] != board_key(pair) or pair["prompt"] != [{"role": "user", "content": board.prompt()}]:
        raise ValueError("Pair board or prompt changed")
    results = []
    for side in ("chosen", "rejected"):
        messages = pair[side]
        if len(messages) != 1 or messages[0]["role"] != "assistant":
            raise ValueError("Preference completions must be single assistant messages")
        result = simulate_turn(messages[0]["content"], board, guesser)
        actual, saved = outcome_dict(result), pair[f"{side}_outcome"]
        for field in actual:
            if field == "guess_similarities":
                if len(actual[field]) != len(saved[field]) or any(abs(a-b) > 1e-5 for a, b in zip(actual[field], saved[field])):
                    raise ValueError("Saved cosine scores do not match replay")
            elif actual[field] != saved[field]:
                raise ValueError(f"Saved {side} {field} does not match replay")
        if result.invalid:
            raise ValueError("Invalid completion in gameplay preference")
        results.append(result)
    chosen, rejected = results
    if pair_mode == "clue-only" and (chosen.clue.number != rejected.clue.number or
                                     chosen.clue.word == rejected.clue.word):
        raise ValueError("Clue-only pairs require different clue words and the same NUMBER")
    if (chosen.reward <= 0 or BLUE not in chosen.guessed_labels or
            chosen.clue.number > sum(tile.label == BLUE for tile in board.active_words) or
            set(chosen.guessed_labels) & {RED, ASSASSIN} or
            chosen.reward - rejected.reward < min_reward_gap - 1e-9 or
            abs(pair["reward_gap"] - (chosen.reward - rejected.reward)) > 1e-9):
        raise ValueError("Pair fails safe-positive winner / strict reward-gap rules")


def validate_candidate_plan(plan: list[dict], rows: list[dict], count: int) -> dict[str, dict]:
    planned = {row["board_key"]: row for row in plan}
    candidates = {row["board_key"]: row for row in rows}
    if (len(planned) != len(plan) or len(candidates) != len(rows) or
            candidates.keys() != planned.keys()):
        raise ValueError("Candidate log is incomplete or contains duplicate/unknown boards")
    for key, row in candidates.items():
        if (any(row.get(field) != value for field, value in planned[key].items()) or
                len(row["completions"]) != count or key != board_key(row) or
                row["prompt"] != board_from_row(row).prompt()):
            raise ValueError("Candidate log does not match its frozen board plan")
    return candidates


def load_cached_source(directory: Path) -> tuple[dict, list[dict], list[dict], dict]:
    """Reuse only a complete audited run, including its original board splits."""
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    required = {"manifest.json", "boards.jsonl", "candidates.jsonl", "train.jsonl", "validation.jsonl"}
    if set(audit["audited_files"]) != required:
        raise ValueError("Source audit does not cover all required artifacts")
    for name, expected in audit["audited_files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Cached source changed after audit: {name}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(directory / "boards.jsonl") != manifest["boards_sha256"]:
        raise ValueError("Source board plan changed")
    plan = read_jsonl(directory / "boards.jsonl")
    rows = read_jsonl(directory / "candidates.jsonl")
    validate_candidate_plan(plan, rows, manifest["config"]["sampling"]["samples_per_board"])
    actual = Counter(row["suite"] for row in plan)
    if actual != Counter(train=manifest["config"]["num_boards"],
                         validation=manifest["config"]["validation_boards"]):
        raise ValueError("Source board split counts differ from the manifest")
    provenance = {"directory": str(directory.resolve()),
                  "artifact_hashes": {name: file_hash(directory / name) for name in sorted(required | {"audit.json"})}}
    return manifest, plan, rows, provenance


def check_cached_output(source_dir: Path, output_dir: Path) -> None:
    source, output = source_dir.resolve(), output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Cached rebuild requires a separate, non-nested output directory")


def restore_cached_settings(args, saved: dict) -> None:
    for key in ("embedding_model", "min_similarity", "abstention_mode", "min_reward_gap",
                "num_boards", "validation_boards", "max_revealed"):
        setattr(args, key, saved[key])
    sampling = saved["sampling"]
    for key in ("base_model", "samples_per_board", "max_new_tokens", "temperature", "top_p", "top_k"):
        setattr(args, key, sampling[key])
    args.adapter_dir = Path(sampling["adapter_dir"])
    args.seed = sampling["sampling_seed"]
    for key in ("vocabulary_file", "exclude_vocabulary_file", "confirmation_report"):
        if key in saved:
            setattr(args, key, Path(saved[key]))


def require_confirmation(args, sampling: dict) -> dict:
    report = json.loads(args.confirmation_report.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError("Abstention has not passed its fresh-board confirmation; do not silently retune on this test")
    config = report["config"]
    expected = (args.embedding_model, args.min_similarity, args.abstention_mode)
    actual = (config["embedding_model"], config["min_similarity"], config["abstention_mode"])
    if (expected != actual or sampling["adapter_hashes"] != config["sampling"]["adapter_hashes"] or
            sampling["base_model"] != config["sampling"]["base_model"]):
        raise ValueError("Use the same fixed teammate and SFT checkpoint as the confirmation")
    if report.get("teammate_artifact_hashes") != embedding_artifact_hashes(args.embedding_model):
        raise ValueError("Embedding weights/tokenizer differ from the confirmation; rerun confirmation if hashes are absent")
    directory = args.confirmation_report.parent
    for name, field in (("manifest.json", "manifest_sha256"), ("candidates.jsonl", "candidates_sha256")):
        if file_hash(directory / name) != report[field]:
            raise ValueError("Confirmation input artifacts changed")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_sampling_args(parser)
    add_abstention_args(parser)
    parser.set_defaults(min_similarity=0.30)
    parser.add_argument("--samples-per-board", type=int, default=8, help="One greedy plus sampled SFT completions")
    parser.add_argument("--num-boards", type=int, default=2000, help="Training boards to attempt, not guaranteed pair count")
    parser.add_argument("--validation-boards", type=int, default=250)
    parser.add_argument("--seed", type=int, default=130_000)
    parser.add_argument("--max-revealed", type=int, default=9)
    parser.add_argument("--vocabulary-file", type=Path, default=Path("data/sft/board_vocabulary.txt"))
    parser.add_argument("--exclude-vocabulary-file", type=Path, default=Path("data/evaluation_vocabulary_oov.txt"))
    parser.add_argument("--min-reward-gap", type=float, default=1.0)
    parser.add_argument("--confirmation-report", type=Path, default=Path("results/abstention-fresh/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dpo"))
    parser.add_argument("--pair-mode", choices=PAIR_MODES,
                        help="gameplay (default), or clue-only: different clue words with equal NUMBER")
    parser.add_argument("--source-dir", type=Path,
                        help="Reselect from a complete audited candidate cache; inherit its settings, never load the policy")
    parser.add_argument("--validate-only", action="store_true", help="Replay saved preferences without loading the policy")
    args = parser.parse_args(argv)
    source_manifest = source_plan = source_rows = provenance = None
    if args.validate_only and args.source_dir:
        parser.error("--validate-only reads the output dataset; omit --source-dir")
    if args.source_dir:
        check_cached_output(args.source_dir, args.output_dir)
        source_manifest, source_plan, source_rows, provenance = load_cached_source(args.source_dir)
        restore_cached_settings(args, source_manifest["config"])
    if args.validate_only:
        # Validation replays the recorded experiment, rather than relying on CLI defaults.
        saved = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
        if args.pair_mode and args.pair_mode != saved.get("pair_mode", "gameplay"):
            raise ValueError("Audit pair mode differs from the recorded dataset")
        restore_cached_settings(args, saved)
        args.pair_mode = saved.get("pair_mode", "gameplay")
    args.pair_mode = args.pair_mode or "gameplay"
    validate_abstention(args.min_similarity, args.abstention_mode)
    if not (2 <= args.samples_per_board <= 64 and args.num_boards > 0 and args.validation_boards > 0
            and args.min_reward_gap > 0 and args.temperature > 0 and 0 < args.top_p <= 1):
        parser.error("Need at least 2 candidates, positive board counts/gap/temperature, and 0 < top_p <= 1")
    sampling = sampling_config(args)
    confirmation = require_confirmation(args, sampling)
    vocabulary = set(load_vocabulary_file(args.vocabulary_file))
    reserved = set(load_vocabulary_file(args.exclude_vocabulary_file))
    if vocabulary & reserved:
        raise ValueError("DPO vocabulary overlaps reserved OOV evaluation words")
    config = {"sampling": sampling, "embedding_model": args.embedding_model,
              "min_similarity": args.min_similarity, "abstention_mode": args.abstention_mode,
              "min_reward_gap": args.min_reward_gap, "num_boards": args.num_boards,
              "validation_boards": args.validation_boards, "max_revealed": args.max_revealed,
              "vocabulary_sha256": file_hash(args.vocabulary_file),
              "reserved_vocabulary_sha256": file_hash(args.exclude_vocabulary_file),
              "confirmation_report_sha256": file_hash(args.confirmation_report)}
    config["teammate_artifact_hashes"] = confirmation["teammate_artifact_hashes"]
    if source_manifest:
        original = source_manifest["config"]
        if any(config[key] != original[key] for key in config):
            raise ValueError("Cached source settings or input hashes differ")
    if not args.validate_only or "pair_mode" in saved:
        config.update(pair_mode=args.pair_mode,
                      vocabulary_file=str(args.vocabulary_file.resolve()),
                      exclude_vocabulary_file=str(args.exclude_vocabulary_file.resolve()),
                      confirmation_report=str(args.confirmation_report.resolve()))
    if provenance:
        config["cached_source"] = provenance
    elif args.validate_only and "cached_source" in saved:
        config["cached_source"] = saved["cached_source"]
    def make_boards():
        excluded, sources = known_boards([args.confirmation_report.parent / "boards.jsonl"])
        train = fresh_boards(args.num_boards, args.seed, args.vocabulary_file,
                             args.max_revealed, "train", excluded)
        # Training plan can skip seeds; start beyond its last actual seed.
        validation = fresh_boards(args.validation_boards, train[-1]["board_seed"] + 10_000,
                                  args.vocabulary_file, args.max_revealed, "validation", excluded)
        return train + validation, sources
    if args.validate_only:
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("Validation settings differ from the dataset manifest")
        plan = read_jsonl(args.output_dir / "boards.jsonl")
        if file_hash(args.output_dir / "boards.jsonl") != manifest["boards_sha256"]:
            raise ValueError("Board plan changed")
    elif source_manifest:
        plan, _ = prepare_run(args.output_dir, config,
                              lambda: (source_plan, source_manifest["excluded_sources"]), resume=args.resume)
        for name in ("boards.jsonl", "candidates.jsonl"):
            target = args.output_dir / name
            expected = provenance["artifact_hashes"][name]
            if target.exists() and file_hash(target) != expected:
                raise ValueError(f"Cached output differs from source: {name}")
            if not target.exists():
                shutil.copyfile(args.source_dir / name, target)
    else:
        plan, _ = prepare_run(args.output_dir, config, make_boards, resume=args.resume)
    import torch
    torch.set_num_threads(4)
    guesser = EmbeddingGuesser(args.embedding_model, device="cpu", min_similarity=args.min_similarity,
                              abstention_mode=args.abstention_mode)
    if args.validate_only:
        candidates = validate_candidate_plan(plan, read_jsonl(args.output_dir / "candidates.jsonl"),
                                             args.samples_per_board)
        seen = set()
        planned = {row["board_key"]: row for row in plan}
        if "cached_source" in config:
            for name in ("boards.jsonl", "candidates.jsonl"):
                if file_hash(args.output_dir / name) != config["cached_source"]["artifact_hashes"][name]:
                    raise ValueError("Derived dataset changed the original board/candidate cache")
        prime_embeddings(guesser, list(candidates.values()))
        for split in ("train", "validation"):
            pairs = read_jsonl(args.output_dir / f"{split}.jsonl")
            if not pairs:
                raise ValueError(f"Empty {split} preference split")
            for pair in pairs:
                if pair["board_key"] in seen or planned[pair["board_key"]]["suite"] != split:
                    raise ValueError("Repeated board or split leakage")
                seen.add(pair["board_key"])
                expected = planned[pair["board_key"]]
                if any(pair[field] != expected[field] for field in
                       ("board_words", "board_labels", "board_revealed", "board_seed")):
                    raise ValueError("Pair state differs from the state used for sampling")
                if pair["teammate"] != {"embedding_model": args.embedding_model,
                                        "min_similarity": args.min_similarity,
                                        "abstention_mode": args.abstention_mode}:
                    raise ValueError("Pair teammate metadata differs from the experiment")
                if set(pair["board_words"]) & reserved:
                    raise ValueError("Reserved OOV word in DPO board")
                for side in ("chosen", "rejected"):
                    if pair[side][0]["content"] not in candidates[pair["board_key"]]["completions"]:
                        raise ValueError("Preference completion was not sampled by the SFT policy")
                validate_pair(pair, guesser, args.min_reward_gap, args.pair_mode)
            print(f"Replayed and validated {len(pairs)} {split} pairs")
        certificate = {
            "schema_version": 1,
            "audited_files": {name: file_hash(args.output_dir / name) for name in
                              ("manifest.json", "boards.jsonl", "candidates.jsonl", "train.jsonl", "validation.jsonl")},
            "pairs_replayed": len(seen), "teammate_artifact_hashes": config["teammate_artifact_hashes"],
            "selection_rule": selection_rule(args.pair_mode),
        }
        # Keep legacy certificates byte-stable: completed DPO runs pin their hash.
        if "pair_mode" in config:
            certificate["pair_mode"] = args.pair_mode
        write_json(args.output_dir / "audit.json", certificate)
        return
    # No policy generation is possible on the cache-only path, even after an interruption.
    candidates = source_rows if source_manifest else collect_candidates(plan, args.output_dir, args)
    prime_embeddings(guesser, candidates)
    pairs, skips = [], Counter()
    for row in candidates:
        pair, reason = choose_pair(row, guesser, args.min_reward_gap, args.pair_mode)
        skips[reason] += 1
        if pair:
            pair["teammate"] = {"embedding_model": args.embedding_model,
                                "min_similarity": args.min_similarity, "abstention_mode": args.abstention_mode}
            pairs.append(pair)
    stats = {"attempted_boards": len(candidates), "pairs": len(pairs), "selection_counts": dict(skips),
             "selection_rule": selection_rule(args.pair_mode), "pair_mode": args.pair_mode,
             "reference_adapter": str(args.adapter_dir), "reference_adapter_hashes": sampling["adapter_hashes"],
             "chosen_number_counts": dict(Counter(parse_clue(pair["chosen"][0]["content"]).number for pair in pairs)),
             "rejected_assassin_pairs": sum(ASSASSIN in pair["rejected_outcome"]["guessed_labels"] for pair in pairs),
             "rejected_red_pairs": sum(RED in pair["rejected_outcome"]["guessed_labels"] for pair in pairs),
             "train": sum(pair["split"] == "train" for pair in pairs),
             "validation": sum(pair["split"] == "validation" for pair in pairs)}
    clues = Counter(parse_clue(pair["chosen"][0]["content"]).word for pair in pairs)
    stats["distinct_chosen_clues"] = len(clues)
    stats["top_chosen_clues"] = clues.most_common(10)
    stats["chosen_blue_guess_counts"] = dict(Counter(pair["chosen_outcome"]["guessed_labels"].count(BLUE)
                                                     for pair in pairs))
    stats["revealed_count_distribution"] = dict(Counter(len(pair["board_revealed"]) for pair in pairs))
    stats["same_clue_different_number_pairs"] = sum(
        parse_clue(pair["chosen"][0]["content"]).word == parse_clue(pair["rejected"][0]["content"]).word
        for pair in pairs)
    stats["same_number_different_clue_pairs"] = sum(
        parse_clue(pair["chosen"][0]["content"]).number == parse_clue(pair["rejected"][0]["content"]).number
        and parse_clue(pair["chosen"][0]["content"]).word != parse_clue(pair["rejected"][0]["content"]).word
        for pair in pairs)
    stats["split_number_counts"] = {split: dict(Counter(
        parse_clue(pair["chosen"][0]["content"]).number for pair in pairs if pair["split"] == split))
        for split in ("train", "validation")}
    stats["chosen_number_above_observed_blue_guesses"] = sum(
        parse_clue(pair["chosen"][0]["content"]).number > pair["chosen_outcome"]["guessed_labels"].count(BLUE)
        for pair in pairs)
    for split in ("train", "validation"):
        write_jsonl(args.output_dir / f"{split}.jsonl", [pair for pair in pairs if pair["split"] == split])
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, indent=2))
    if not stats["train"] or not stats["validation"]:
        raise RuntimeError("Not enough clear preferences: increase fresh board/candidate counts; do not manufacture labels")
    print("Preferences saved. For DPO training, use this frozen SFT adapter as the reference, not the raw base model.")


if __name__ == "__main__":
    main()
