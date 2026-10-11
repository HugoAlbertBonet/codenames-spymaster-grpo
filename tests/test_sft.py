import copy
import unittest
from types import SimpleNamespace

from codenames_env import BLUE, RED, CodenamesBoard, OrderedGuesser, generate_board, simulate_turn
from sft_data import tokenize_record, validate_record
from train_grpo import build_training_examples, make_codenames_reward
from evaluate import board_from_example
from train_sft import calculate_warmup_steps


class SFTTests(unittest.TestCase):
    def test_warmup_uses_optimizer_steps_and_respects_max_steps(self):
        args = SimpleNamespace(max_steps=5, batch_size=1, gradient_accumulation_steps=8, epochs=2)
        self.assertEqual(calculate_warmup_steps(args, 1800), 1)
        args.max_steps = -1
        self.assertEqual(calculate_warmup_steps(args, 1800), 23)
        self.assertEqual(calculate_warmup_steps(args, 1800, world_size=2), 12)
        # An incomplete accumulation group is still one optimizer update.
        self.assertEqual(calculate_warmup_steps(args, 9), 1)

    def setUp(self):
        self.board = generate_board(seed=81)
        self.blue = [item.word for item in self.board.words if item.label == BLUE]

    def test_revealed_word_cannot_be_guessed_or_rewarded(self):
        board = CodenamesBoard(self.board.words, (self.blue[0],))
        result = simulate_turn("CLUE: universe\nNUMBER: 1", board, OrderedGuesser(self.blue))
        self.assertFalse(result.invalid)
        self.assertNotIn(self.blue[0], result.guesses)
        self.assertEqual(result.reward, 2.0)
        active_section = board.prompt().split("BLUE: ")[-1].split("\n")[0]
        self.assertNotIn(self.blue[0], active_section.split(", "))
        self.assertIn("ALREADY GUESSED", board.prompt())

    def test_guessed_assassin_and_completed_teams_are_rejected(self):
        assassin = next(item.word for item in self.board.words if item.label == "ASSASSIN")
        for revealed in [(assassin,), tuple(self.blue),
                         tuple(item.word for item in self.board.words if item.label == RED)]:
            with self.assertRaises(ValueError):
                CodenamesBoard(self.board.words, revealed)

    def test_random_states_propagate_through_evaluation_and_reward(self):
        examples = build_training_examples(20, seed=83, max_revealed=9)
        self.assertTrue(any(example.board_revealed for example in examples))
        for example in examples:
            board = board_from_example(example)
            self.assertEqual(board.revealed, example.board_revealed)
            blue = [item.word for item in board.active_words if item.label == BLUE]
            reward = make_codenames_reward(OrderedGuesser(blue), duplicate_penalty=0)
            scores = reward(completions=["CLUE: universe\nNUMBER: 1"],
                            board_words=[example.board_words], board_labels=[example.board_labels],
                            board_revealed=[example.board_revealed])
            self.assertEqual(scores, [float(min(2, len(blue)))])

    def test_target_annotations_reject_revealed_targets_and_stale_prompts(self):
        row = {"board_words": [item.word for item in self.board.words],
               "board_labels": [item.label for item in self.board.words],
               "board_revealed": [], "prompt": self.board.prompt(),
               "completion": "CLUE: universe\nNUMBER: 1", "intended_targets": [self.blue[0]]}
        validate_record(row)
        stale = copy.deepcopy(row)
        stale["prompt"] += " altered"
        with self.assertRaises(ValueError):
            validate_record(stale)
        guessed = copy.deepcopy(row)
        guessed["board_revealed"] = [self.blue[0]]
        guessed["prompt"] = CodenamesBoard(self.board.words, (self.blue[0],)).prompt()
        with self.assertRaises(ValueError):
            validate_record(guessed)

    def test_response_labels_exclude_prompt_and_include_end_marker(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                prefix = "<user>" + messages[0]["content"] + "<assistant>"
                return prefix if add_generation_prompt else prefix + messages[-1]["content"] + "<end>"

            def __call__(self, text, add_special_tokens):
                return {"input_ids": [ord(character) for character in text]}

        row = {"prompt": "A board", "completion": "CLUE: universe\nNUMBER: 1"}
        tokens = tokenize_record(row, Tokenizer(), 200)
        supervised = "".join(chr(label) for label in tokens["labels"] if label != -100)
        self.assertEqual(supervised, row["completion"] + "<end>")
        with self.assertRaises(ValueError):
            tokenize_record(row, Tokenizer(), 10)


if __name__ == "__main__":
    unittest.main()
