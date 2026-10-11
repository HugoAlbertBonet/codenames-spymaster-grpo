"""Compare a preselected abstention cutoff on fresh boards using identical SFT clues."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean, stdev

from codenames_env import EmbeddingGuesser, add_abstention_args, simulate_turn, validate_abstention
from evaluate import EvaluationRecord, print_summary, summarize_records, write_csv
from outcome_data import (add_sampling_args, board_from_row, collect_candidates, file_hash, embedding_artifact_hashes,
                          fresh_boards, known_boards, prepare_run, prime_embeddings, sampling_config, write_json)


def paired_comparison(baseline: list[EvaluationRecord], abstaining: list[EvaluationRecord]) -> dict:
    if not baseline or len(baseline) != len(abstaining):
        raise ValueError("Need matching, nonempty evaluations")
    differences = []
    assassin_differences = []
    for before, after in zip(baseline, abstaining):
        if (before.board_words, before.board_labels, before.board_revealed, before.completion) != (
                after.board_words, after.board_labels, after.board_revealed, after.completion):
            raise ValueError("Comparison must use identical boards and completions")
        differences.append(after.reward - before.reward)
        assassin_differences.append(int("ASSASSIN" in after.guessed_labels) -
                                   int("ASSASSIN" in before.guessed_labels))
    def interval(values):
        error = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return [mean(values) - error, mean(values) + error]
    before, after = summarize_records(baseline), summarize_records(abstaining)
    baseline_blue = sum(row.correct_guesses for row in baseline)
    retained = sum(row.correct_guesses for row in abstaining) / baseline_blue if baseline_blue else None
    return {"baseline": before, "abstaining": after, "reward_delta": mean(differences),
            "reward_delta_approx_95_ci": interval(differences),
            "assassin_hit_delta_approx_95_ci": interval(assassin_differences),
            "blue_retention": retained}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_sampling_args(parser)
    add_abstention_args(parser)
    parser.set_defaults(min_similarity=0.30, samples_per_board=1)
    parser.add_argument("--num-boards", type=int, default=500, help="Fresh boards per suite; three suites")
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--vocabulary-file", type=Path, default=Path("data/sft/board_vocabulary.txt"))
    parser.add_argument("--oov-vocabulary-file", type=Path, default=Path("data/evaluation_vocabulary_oov.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/abstention-fresh"))
    args = parser.parse_args()
    validate_abstention(args.min_similarity, args.abstention_mode)
    if args.min_similarity is None:
        parser.error("Choose a fixed --min-similarity before running the confirmation")
    if not 1 <= args.num_boards <= 10_000:
        parser.error("--num-boards must be 1..10000 (suite seed ranges must not overlap)")
    config = {"sampling": sampling_config(args), "num_boards_per_suite": args.num_boards,
              "seed": args.seed, "embedding_model": args.embedding_model,
              "min_similarity": args.min_similarity, "abstention_mode": args.abstention_mode,
              "vocabulary_sha256": file_hash(args.vocabulary_file),
              "oov_vocabulary_sha256": file_hash(args.oov_vocabulary_file),
              "gate": {"minimum_blue_retention": 0.90,
                       "positive_paired_reward_ci": True, "fewer_assassin_hits": True}}
    def make_boards():
        excluded, sources = known_boards()
        rows = []
        for suite, offset, vocabulary, revealed in (
            ("random", 0, args.vocabulary_file, 0),
            ("partial", 10_000, args.vocabulary_file, 9),
            ("oov", 20_000, args.oov_vocabulary_file, 0),
        ):
            rows.extend(fresh_boards(args.num_boards, args.seed + offset, vocabulary,
                                     revealed, suite, excluded))
        return rows, sources
    rows, _ = prepare_run(args.output_dir, config, make_boards, resume=args.resume)
    candidates = collect_candidates(rows, args.output_dir, args)
    import torch
    torch.set_num_threads(4)
    guesser = EmbeddingGuesser(args.embedding_model, device="cpu")
    prime_embeddings(guesser, candidates)
    before, after = [], []
    for row in candidates:
        board, completion = board_from_row(row), row["completions"][0]
        guesser.min_similarity = None
        result = simulate_turn(completion, board, guesser)
        before.append(EvaluationRecord.from_turn("sft-no-abstention", row["board_seed"],
                                                row["board_seed"], result, board))
        guesser.min_similarity, guesser.abstention_mode = args.min_similarity, args.abstention_mode
        result = simulate_turn(completion, board, guesser)
        after.append(EvaluationRecord.from_turn("sft-abstention", row["board_seed"], row["board_seed"],
                                               result, board, args.min_similarity, args.abstention_mode))
    report = {"config": config, "manifest_sha256": file_hash(args.output_dir / "manifest.json"),
              "candidates_sha256": file_hash(args.output_dir / "candidates.jsonl"),
              "teammate_artifact_hashes": embedding_artifact_hashes(args.embedding_model), "suites": {}}
    for suite in ("random", "partial", "oov"):
        indices = [i for i, row in enumerate(candidates) if row["suite"] == suite]
        baseline, abstaining = [before[i] for i in indices], [after[i] for i in indices]
        report["suites"][suite] = paired_comparison(baseline, abstaining)
        write_csv(baseline + abstaining, args.output_dir / f"evaluation-{suite}.csv")
        print(f"\n{suite.upper()}")
        print_summary(baseline)
        print_summary(abstaining)
    overall = paired_comparison(before, after)
    report["overall"] = overall
    # Do not pick a new threshold on this test. A failed gate requires a separate experiment.
    report["passed"] = bool(len(before) >= 300 and overall["blue_retention"] is not None and
                            overall["blue_retention"] >= 0.90 and
                            overall["reward_delta_approx_95_ci"][0] > 0 and
                            overall["abstaining"]["assassin_hit_percent"] <
                            overall["baseline"]["assassin_hit_percent"])
    report["limitations"] = ("Single-turn outcome test, not win rate. Approximate paired normal CIs; "
                             "suite intervals and point improvements may differ. Cosine is not a probability.")
    write_json(args.output_dir / "report.json", report)
    print(f"\nOverall reward delta: {overall['reward_delta']:+.4f}; "
          f"approximate 95% CI {overall['reward_delta_approx_95_ci']}; "
          f"BLUE retention {overall['blue_retention']}")
    print(f"Confirmation gate: {'PASS' if report['passed'] else 'NOT CONFIRMED'}. "
          f"Saved {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
