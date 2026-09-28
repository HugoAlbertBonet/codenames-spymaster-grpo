import tempfile
import unittest
from pathlib import Path

from train_grpo import (
    _board_from_columns,
    build_training_examples,
    find_latest_checkpoint,
    format_prompt_for_model,
    make_codenames_reward,
)
from codenames_env import Clue, OrderedGuesser, TurnResult
from train_grpo import _semantic_margin_bonus


class TrainingIntegrationTests(unittest.TestCase):
    def test_semantic_margin_bonus_prefers_blue_over_dangerous_words(self):
        example = build_training_examples(num_boards=1, seed=22)[0]

        class FakeSimilarityGuesser:
            def similarities(self, clue, words):
                return {
                    word: (0.9 if label == "BLUE" else 0.1)
                    for word, label in zip(example.board_words, example.board_labels)
                }

        board = _board_from_columns(example.board_words, example.board_labels)
        result = TurnResult(
            "CLUE: galaxy\nNUMBER: 2",
            Clue("galaxy", 2),
            (),
            (),
            0.0,
            False,
            None,
        )
        bonus = _semantic_margin_bonus(result, board, FakeSimilarityGuesser(), 0.5, 0.25)
        self.assertGreater(bonus, 0.0)

    def test_duplicate_clues_receive_a_small_group_penalty(self):
        example = build_training_examples(num_boards=1, seed=23)[0]
        board = _board_from_columns(example.board_words, example.board_labels)
        blue_words = [item.word for item in board.words if item.label == "BLUE"]
        reward = make_codenames_reward(OrderedGuesser(blue_words), duplicate_penalty=0.25)
        values = reward(
            prompts=[example.prompt, example.prompt],
            completions=["CLUE: galaxy\nNUMBER: 1", "CLUE: galaxy\nNUMBER: 1"],
            board_words=[list(example.board_words)] * 2,
            board_labels=[list(example.board_labels)] * 2,
        )
        self.assertEqual(values, [1.75, 1.75])

    def test_model_prompt_uses_chat_template_when_available(self):
        class FakeTokenizer:
            chat_template = "available"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.arguments = (messages, tokenize, add_generation_prompt)
                return "formatted prompt"

        tokenizer = FakeTokenizer()
        self.assertEqual(format_prompt_for_model("raw prompt", tokenizer), "formatted prompt")
        self.assertTrue(tokenizer.arguments[2])

    def test_training_rows_have_prompt_and_full_board_metadata(self):
        examples = build_training_examples(num_boards=3, seed=20)
        self.assertEqual(len(examples), 3)
        self.assertEqual(len(examples[0].board_words), 16)
        self.assertEqual(len(examples[0].board_labels), 16)
        self.assertIn("CLUE:", examples[0].prompt)

    def test_reward_callback_matches_grpo_batch_shape(self):
        example = build_training_examples(num_boards=1, seed=21)[0]
        guesser = OrderedGuesser(example.board_words)
        reward = make_codenames_reward(guesser)
        values = reward(
            prompts=[example.prompt, example.prompt],
            completions=["CLUE: galaxy\nNUMBER: 1", "invalid"],
            board_words=[list(example.board_words)] * 2,
            board_labels=[list(example.board_labels)] * 2,
        )
        self.assertEqual(len(values), 2)
        self.assertEqual(values[1], -2.0)

    def test_latest_checkpoint_is_selected_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "checkpoint-50").mkdir()
            (output_dir / "checkpoint-200").mkdir()
            (output_dir / "checkpoint-not-a-step").mkdir()
            self.assertEqual(find_latest_checkpoint(output_dir), output_dir / "checkpoint-200")


if __name__ == "__main__":
    unittest.main()
