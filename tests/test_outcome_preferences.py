import json
import copy
from pathlib import Path
import tempfile
import unittest
import random
from types import SimpleNamespace
from unittest.mock import patch

from build_dpo_pairs import (choose_pair, validate_pair, require_confirmation, check_cached_output,
                             load_cached_source, validate_candidate_plan, main)
from dpo_data import validate_preference
from codenames_env import BLUE, RED, ASSASSIN, OrderedGuesser, simulate_turn, validate_abstention, parse_clue
from confirm_abstention import paired_comparison
from evaluate import EvaluationRecord, write_csv, summarize_records
from outcome_data import board_from_row, board_key, fresh_boards, prepare_run, file_hash
from train_grpo import build_training_examples


class ScoredGuesser(OrderedGuesser):
    def __init__(self, words, scores, cutoff=None, mode="all"):
        super().__init__(words)
        self.scores = scores
        self.min_similarity = cutoff
        self.abstention_mode = mode

    def similarities(self, clue, words):
        return {word: self.scores.get(word, 0.0) for word in words}


class AbstentionTests(unittest.TestCase):
    def setUp(self):
        self.row = build_training_examples(1, 321)[0].as_dict()
        self.board = board_from_row(self.row)
        self.blue = [tile.word for tile in self.board.words if tile.label == BLUE]
        self.assassin = next(tile.word for tile in self.board.words if tile.label == ASSASSIN)

    def test_default_policy_unchanged_and_threshold_blocks_assassin(self):
        guesser = ScoredGuesser([self.blue[0], self.assassin],
                                {self.blue[0]: 0.5, self.assassin: 0.29})
        before = simulate_turn("CLUE: galaxy\nNUMBER: 1", self.board, guesser)
        self.assertEqual(before.reward, -4)
        self.assertEqual(before.stop_reason, "assassin")
        guesser.min_similarity = 0.3
        after = simulate_turn(before.completion, self.board, guesser)
        self.assertEqual(after.reward, 1)
        self.assertTrue(after.abstained)
        self.assertEqual(after.guesses, (self.blue[0],))
        self.assertEqual(after.guess_similarities, (0.5,))

    def test_all_can_abstain_on_first_guess_extra_cannot(self):
        guesser = ScoredGuesser([self.assassin], {self.assassin: 0.2}, 0.3)
        after = simulate_turn("CLUE: galaxy\nNUMBER: 1", self.board, guesser)
        self.assertEqual(after.guesses, ())
        self.assertEqual(after.reward, 0)
        self.assertFalse(after.invalid)
        self.assertTrue(after.abstained)
        guesser.abstention_mode = "extra"
        self.assertEqual(simulate_turn(after.completion, self.board, guesser).reward, -5)

    def test_cutoff_equality_is_accepted_and_first_nonblue_still_ends_turn(self):
        guesser = ScoredGuesser([self.assassin, self.blue[0]], {self.assassin: 0.3}, 0.3)
        after = simulate_turn("CLUE: galaxy\nNUMBER: 1", self.board, guesser)
        self.assertEqual(after.guessed_labels, (ASSASSIN,))
        self.assertFalse(after.abstained)

    def test_revealed_words_never_guessed(self):
        self.row["board_revealed"] = [self.blue[0]]
        board = board_from_row(self.row)
        guesser = ScoredGuesser([self.blue[0], self.blue[1]], {self.blue[1]: 0.5}, 0.3)
        result = simulate_turn("CLUE: galaxy\nNUMBER: 1", board, guesser)
        self.assertEqual(result.guesses, (self.blue[1],))

    def test_bad_thresholds_and_missing_scores_rejected(self):
        for cutoff in (float("nan"), float("inf"), 1.1, -1.1):
            with self.assertRaises(ValueError):
                validate_abstention(cutoff, "all")
        with self.assertRaises(ValueError):
            validate_abstention(0.3, "bogus")
        guesser = OrderedGuesser([])
        guesser.min_similarity = 0.3
        with self.assertRaises(ValueError):
            simulate_turn("CLUE: galaxy\nNUMBER: 1", self.board, guesser)

    def test_csv_and_comparison_expose_actual_abstention(self):
        guesser = ScoredGuesser([self.blue[0], self.assassin],
                                {self.blue[0]: 0.5, self.assassin: 0.1})
        before = EvaluationRecord.from_turn("before", 0, 321,
                    simulate_turn("CLUE: galaxy\nNUMBER: 1", self.board, guesser), self.board)
        guesser.min_similarity = 0.3
        after = EvaluationRecord.from_turn("after", 0, 321,
                    simulate_turn(before.completion, self.board, guesser), self.board, 0.3)
        comparison = paired_comparison([before], [after])
        self.assertEqual(comparison["reward_delta"], 5)
        self.assertEqual(comparison["blue_retention"], 1)
        self.assertEqual(summarize_records([after])["abstention_percent"], 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.csv"
            write_csv([after], path)
            self.assertIn("similarity_below_threshold", path.read_text())

    def test_abstention_only_removes_a_suffix_never_introduces_an_assassin(self):
        rng = random.Random(83)
        for example in build_training_examples(50, 2000):
            board = board_from_row(example.as_dict())
            scores = {tile.word: rng.random() for tile in board.words}
            ranking = sorted(scores, key=lambda word: -scores[word])
            guesser = ScoredGuesser(ranking, scores)
            for number in range(1, 5):
                completion = f"CLUE: galaxy\nNUMBER: {number}"
                guesser.min_similarity = None
                baseline = simulate_turn(completion, board, guesser)
                for mode in ("all", "extra"):
                    guesser.min_similarity, guesser.abstention_mode = 0.3, mode
                    result = simulate_turn(completion, board, guesser)
                    self.assertEqual(result.guesses, baseline.guesses[:len(result.guesses)])
                    if ASSASSIN in result.guessed_labels:
                        self.assertIn(ASSASSIN, baseline.guessed_labels)


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.row = build_training_examples(1, 321)[0].as_dict()
        self.row.update(board_seed=321, suite="train", board_key=board_key(self.row))
        board = board_from_row(self.row)
        blue = [tile.word for tile in board.words if tile.label == BLUE]
        red = next(tile.word for tile in board.words if tile.label == RED)
        assassin = next(tile.word for tile in board.words if tile.label == ASSASSIN)
        class Teammate:
            min_similarity = 0.3
            abstention_mode = "all"
            def similarities(self, clue, words):
                target = {"galaxy": blue[0], "cosmos": red, "planet": assassin}.get(clue, blue[0])
                return {word: (0.5 if word == target else 0.1) for word in words}
            def rank(self, clue, words):
                scores = self.similarities(clue, words)
                return sorted(words, key=lambda word: -scores[word])
        self.guesser = Teammate()

    def test_clue_only_requires_same_number_without_rewriting_completions(self):
        self.row['completions'] = ['CLUE: galaxy\nNUMBER: 1', 'CLUE: cosmos\nNUMBER: 2']
        self.assertEqual(choose_pair(self.row, self.guesser, pair_mode='clue-only'),
                         (None, 'no_same_number_clue_preference'))
        self.row['completions'].append('CLUE: cosmos\nNUMBER: 1')
        pair, reason = choose_pair(self.row, self.guesser, pair_mode='clue-only')
        self.assertEqual(reason, 'kept')
        self.assertEqual(pair['pair_mode'], 'clue-only')
        for side in ('chosen', 'rejected'):
            self.assertIn(pair[side][0]['content'], self.row['completions'])
        validate_pair(pair, self.guesser, 1, 'clue-only')
        validate_preference(pair, 'clue-only')

    def test_clue_only_searches_below_global_best_winner(self):
        board = board_from_row(self.row)
        blue = [tile.word for tile in board.words if tile.label == BLUE]
        red = next(tile.word for tile in board.words if tile.label == RED)
        class Teammate:
            min_similarity = .3
            abstention_mode = 'all'
            def similarities(self, clue, words):
                selected = blue[:3] if clue == 'galaxy' else [red] if clue == 'cosmos' else blue[:1]
                return {word: .8 - .1 * selected.index(word) if word in selected else .1 for word in words}
            def rank(self, clue, words):
                scores = self.similarities(clue, words)
                return sorted(words, key=lambda w: -scores[w])
        self.row['completions'] = ['CLUE: galaxy\nNUMBER: 2', 'CLUE: universe\nNUMBER: 1',
                                   'CLUE: cosmos\nNUMBER: 1']
        best, _ = choose_pair(self.row, Teammate())
        self.assertEqual(best['chosen'][0]['content'], self.row['completions'][0])
        controlled, _ = choose_pair(self.row, Teammate(), pair_mode='clue-only')
        self.assertEqual(controlled['chosen'][0]['content'], self.row['completions'][1])
        validate_pair(controlled, Teammate(), 1, 'clue-only')

    def test_clue_only_audit_and_trainer_reject_different_budgets(self):
        self.row['completions'] = ['CLUE: galaxy\nNUMBER: 1', 'CLUE: cosmos\nNUMBER: 2']
        pair, _ = choose_pair(self.row, self.guesser)
        pair['pair_mode'] = 'clue-only'
        for validator in (lambda: validate_pair(pair, self.guesser, 1, 'clue-only'),
                          lambda: validate_preference(pair, 'clue-only')):
            with self.assertRaisesRegex(ValueError, 'same NUMBER'):
                validator()

    def test_clue_only_skips_same_clue_budget_preferences(self):
        board = board_from_row(self.row)
        blue = [tile.word for tile in board.words if tile.label == BLUE]
        red = next(tile.word for tile in board.words if tile.label == RED)
        guesser = ScoredGuesser(blue[:2] + [red], {blue[0]: .8, blue[1]: .7, red: .6}, .3)
        self.row['completions'] = ['CLUE: galaxy\nNUMBER: 1', 'CLUE: galaxy\nNUMBER: 2']
        self.assertEqual(choose_pair(self.row, guesser)[1], 'kept')
        self.assertEqual(choose_pair(self.row, guesser, pair_mode='clue-only')[1],
                         'no_same_number_clue_preference')

    def test_cached_plan_rejects_duplicate_boards_and_changed_completions_count(self):
        row = self.row | {'completions': ['CLUE: galaxy\nNUMBER: 1', 'CLUE: cosmos\nNUMBER: 1']}
        validate_candidate_plan([self.row], [row], 2)
        for rows in ([row, row], [row | {'completions': ['bad']}], [row | {'prompt': 'changed'}]):
            with self.assertRaises(ValueError):
                validate_candidate_plan([self.row], rows, 2)

    def test_cached_source_output_protects_original_and_parent_directories(self):
        for output in ('data/dpo', 'data/dpo/derived', 'data'):
            with self.assertRaises(ValueError):
                check_cached_output(Path('data/dpo'), Path(output))
        check_cached_output(Path('data/dpo'), Path('data/dpo-clue-only'))

    @unittest.skipUnless(Path('data/dpo/audit.json').exists(), 'local audited cache required')
    def test_complete_source_cache_is_verified_without_policy_generation(self):
        manifest, plan, rows, provenance = load_cached_source(Path('data/dpo'))
        self.assertEqual((len(plan), len(rows)), (600, 600))
        self.assertEqual(provenance['artifact_hashes']['candidates.jsonl'], file_hash(Path('data/dpo/candidates.jsonl')))
        self.assertEqual(manifest['config']['num_boards'], 500)

    @unittest.skipUnless(Path('data/dpo/audit.json').exists(), 'local audited cache required')
    def test_cache_only_pipeline_cannot_call_policy_sampler(self):
        # CPU-only integration: actual audited cached text, deterministic fake teammate.
        source_config = json.loads(Path('data/dpo/manifest.json').read_text())['config']
        def fixture_hash(path):
            # Confirmation behavior is mocked below; its original local report
            # is not shipped with the repo. Keep all dataset hashes real.
            if Path(path) == Path('results/abstention-fresh/report.json'):
                return source_config['confirmation_report_sha256']
            return file_hash(path)
        source_pairs = [json.loads(line) for split in ('train', 'validation')
                        for line in Path(f'data/dpo/{split}.jsonl').read_text().splitlines()]
        eligible = {p['board_key']: p for p in source_pairs
                    if parse_clue(p['chosen'][0]['content']).number == parse_clue(p['rejected'][0]['content']).number
                    and parse_clue(p['chosen'][0]['content']).word != parse_clue(p['rejected'][0]['content']).word}
        def cached_selection(row, guesser, gap, mode):
            pair = copy.deepcopy(eligible.get(row['board_key']))
            if pair is not None:
                pair['pair_mode'] = mode
            return pair, 'kept' if pair else 'no_same_number_clue_preference'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'clue-only'
            with patch('build_dpo_pairs.require_confirmation', return_value={'teammate_artifact_hashes':
                       source_config['teammate_artifact_hashes']}), \
                    patch('build_dpo_pairs.file_hash', side_effect=fixture_hash), \
                    patch('build_dpo_pairs.EmbeddingGuesser', return_value=self.guesser), \
                    patch('build_dpo_pairs.prime_embeddings'), \
                    patch('build_dpo_pairs.choose_pair', side_effect=cached_selection), \
                    patch('builtins.print'), \
                    patch('build_dpo_pairs.collect_candidates', side_effect=AssertionError('policy loaded')):
                main(['--source-dir', 'data/dpo', '--pair-mode', 'clue-only', '--output-dir', str(output)])
                main(['--source-dir', 'data/dpo', '--pair-mode', 'clue-only', '--output-dir', str(output), '--resume'])
                with self.assertRaises(FileExistsError):
                    main(['--source-dir', 'data/dpo', '--pair-mode', 'clue-only', '--output-dir', str(output)])
            self.assertEqual(file_hash(output / 'candidates.jsonl'), file_hash(Path('data/dpo/candidates.jsonl')))
            self.assertEqual(file_hash(output / 'boards.jsonl'), file_hash(Path('data/dpo/boards.jsonl')))

    def test_safe_winner_and_actual_dangerous_negative(self):
        self.row["completions"] = ["CLUE: galaxy\nNUMBER: 1", "CLUE: cosmos\nNUMBER: 1",
                                   "CLUE: planet\nNUMBER: 1", "invalid", "CLUE: GALAXY\nNUMBER: 1"]
        pair, reason = choose_pair(self.row, self.guesser)
        self.assertEqual(reason, "kept")
        self.assertEqual(pair["reward_gap"], 2)
        self.assertEqual(pair["chosen_outcome"]["guessed_labels"], [BLUE])
        self.assertEqual(pair["rejected_outcome"]["guessed_labels"], [RED])
        self.assertEqual(pair["rejected"][0]["content"], self.row["completions"][1])
        validate_pair(pair, self.guesser, 1)
        pair["chosen_outcome"]["reward"] = 999
        with self.assertRaises(ValueError):
            validate_pair(pair, self.guesser, 1)

    def test_ties_duplicates_invalids_and_no_safe_winner_are_skipped(self):
        for completions, reason in (
            (["CLUE: galaxy\nNUMBER: 1", "CLUE: GALAXY\nNUMBER: 1"], "no_clear_gameplay_preference"),
            (["CLUE: cosmos\nNUMBER: 1", "invalid"], "no_safe_positive_winner"),
            (["CLUE: galaxy\nNUMBER: 1", "invalid"], "no_clear_gameplay_preference"),
        ):
            self.row["completions"] = completions
            pair, actual = choose_pair(self.row, self.guesser)
            self.assertIsNone(pair)
            self.assertEqual(actual, reason)

    def test_signature_ignores_board_order_and_revealed_state(self):
        other = dict(self.row, board_words=list(reversed(self.row["board_words"])),
                     board_labels=list(reversed(self.row["board_labels"])),
                     board_revealed=[self.row["board_words"][0]])
        # Pick a non-assassin word for the alternate nonterminal state.
        other["board_revealed"] = [tile.word for tile in board_from_row(self.row).words if tile.label == BLUE][:1]
        self.assertEqual(board_key(self.row), board_key(other))

    def test_new_boards_exclude_previous_and_split_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / "vocabulary.txt"
            vocab.write_text("\n".join(self.row["board_words"]))
            excluded = {self.row["board_key"]}
            train = fresh_boards(3, 321, vocab, 9, "train", excluded)
            validation = fresh_boards(3, 321, vocab, 9, "validation", excluded)
            self.assertFalse({row["board_key"] for row in train} & {row["board_key"] for row in validation})

    def test_resume_requires_identical_settings_and_unchanged_board_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            prepare_run(output, {"cutoff": 0.3}, lambda: ([self.row], {}), resume=False)
            rows, _ = prepare_run(output, {"cutoff": 0.3}, None, resume=True)
            self.assertEqual(rows[0]["board_key"], self.row["board_key"])
            with self.assertRaises(ValueError):
                prepare_run(output, {"cutoff": 0.4}, None, resume=True)
            with self.assertRaises(FileExistsError):
                prepare_run(output, {"cutoff": 0.3}, None, resume=False)
            (output / "boards.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                prepare_run(output, {"cutoff": 0.3}, None, resume=True)

    def test_failed_or_mismatched_confirmation_blocks_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            args = SimpleNamespace(confirmation_report=path, embedding_model="fixed", min_similarity=0.3,
                                   abstention_mode="all")
            path.write_text(json.dumps({"passed": False}))
            with self.assertRaises(ValueError):
                require_confirmation(args, {})
            report = {"passed": True, "config": {"embedding_model": "fixed", "min_similarity": 0.4,
                       "abstention_mode": "all", "sampling": {"adapter_hashes": {}, "base_model": "model"}}}
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                require_confirmation(args, {"adapter_hashes": {}, "base_model": "model"})

    def test_pair_winner_cannot_be_a_zero_guess_abstention(self):
        class Abstaining(ScoredGuesser):
            pass
        self.row["completions"] = ["CLUE: galaxy\nNUMBER: 1", "CLUE: planet\nNUMBER: 1"]
        pair, reason = choose_pair(self.row, Abstaining([], {}, 0.3))
        self.assertIsNone(pair)
        self.assertEqual(reason, "no_safe_positive_winner")

    def test_winner_number_cannot_exceed_remaining_blue_words(self):
        board = board_from_row(self.row)
        blue = [tile.word for tile in board.words if tile.label == BLUE]
        self.row["board_revealed"] = blue[1:]
        self.row["prompt"] = board_from_row(self.row).prompt()
        self.row["completions"] = ["CLUE: galaxy\nNUMBER: 2", "CLUE: cosmos\nNUMBER: 1"]
        pair, reason = choose_pair(self.row, self.guesser)
        self.assertIsNone(pair)
        self.assertEqual(reason, "no_safe_positive_winner")
        self.row["completions"].append("CLUE: galaxy\nNUMBER: 1")
        pair, reason = choose_pair(self.row, self.guesser)
        self.assertEqual(reason, "kept")
        self.assertEqual(pair["chosen"][0]["content"], "CLUE: galaxy\nNUMBER: 1")


if __name__ == "__main__":
    unittest.main()
