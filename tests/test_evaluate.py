import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codenames_env import OrderedGuesser
from evaluate import (
    evaluate_completions,
    load_vocabulary_file,
    summarize_records,
    write_csv,
    write_transcripts,
    main,
    parse_args,
)
from train_grpo import build_training_examples


class EvaluationTests(unittest.TestCase):
    def test_baseline_adapter_replaces_raw_base_and_keeps_same_boards(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / 'sft'
            trained = Path(directory) / 'dpo'
            initial.mkdir()
            trained.mkdir()
            with patch('sys.argv', ['evaluate.py', '--baseline-adapter-dir', str(initial),
                    '--adapter-dir', str(trained), '--num-boards', '2',
                    '--output-csv', str(Path(directory) / 'evaluation.csv'),
                    '--transcript-file', str(Path(directory) / 'transcripts.txt')]):
                args = parse_args()
            guesser = OrderedGuesser([])
            def fake_checkpoint(model_name, adapter, examples, label, teammate, seed, max_tokens):
                return evaluate_completions(examples, lambda _: 'bad', label, teammate, seed)
            with patch('evaluate.EmbeddingGuesser', return_value=guesser), \
                    patch('evaluate.evaluate_checkpoint', side_effect=fake_checkpoint) as evaluate:
                main(args)
            self.assertEqual(evaluate.call_args_list[0].args[1], initial)
            self.assertEqual(evaluate.call_args_list[1].args[1], trained)
            self.assertIs(evaluate.call_args_list[0].args[2], evaluate.call_args_list[1].args[2])

    def test_evaluation_uses_injected_completion_function_and_reports_metrics(self):
        examples = build_training_examples(num_boards=2, seed=30)
        guesser = OrderedGuesser([item for item in examples[0].board_words])
        records = evaluate_completions(
            examples,
            lambda prompt: "CLUE: galaxy\nNUMBER: 1",
            "fake",
            guesser,
            seed=30,
        )
        self.assertEqual(len(records), 2)
        summary = summarize_records(records)
        self.assertEqual(summary["model"], "fake")
        self.assertEqual(summary["episodes"], 2)
        self.assertIn("average_reward", summary)

    def test_csv_and_transcript_outputs_are_created(self):
        examples = build_training_examples(num_boards=2, seed=31)
        guesser = OrderedGuesser(list(examples[0].board_words))
        base = evaluate_completions(examples, lambda _: "bad", "base", guesser, 31)
        trained = evaluate_completions(
            examples, lambda _: "CLUE: galaxy\nNUMBER: 1", "trained", guesser, 31
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "nested" / "results.csv"
            transcript_path = Path(directory) / "nested" / "transcripts.txt"
            write_csv(base + trained, csv_path)
            write_transcripts(base, trained, examples, transcript_path)
            self.assertTrue(csv_path.exists())
            self.assertTrue(transcript_path.exists())
            self.assertIn("model,board_index", csv_path.read_text())
            self.assertIn("board_words", csv_path.read_text())
            self.assertIn("BOARD 0", transcript_path.read_text())

    def test_custom_vocabulary_is_loaded_for_evaluation_boards(self):
        with tempfile.TemporaryDirectory() as directory:
            vocabulary_path = Path(directory) / "vocabulary.txt"
            vocabulary_path.write_text("\n".join(f"oov{i}" for i in range(16)), encoding="utf-8")
            vocabulary = load_vocabulary_file(vocabulary_path)
            examples = build_training_examples(num_boards=2, seed=32, vocabulary=vocabulary)
            self.assertEqual(len(vocabulary), 16)
            self.assertTrue(set(examples[0].board_words) <= set(vocabulary))


if __name__ == "__main__":
    unittest.main()
