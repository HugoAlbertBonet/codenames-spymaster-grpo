import unittest

import numpy as np

from codenames_env import (
    ASSASSIN,
    BLUE,
    InvalidClue,
    NEUTRAL,
    OrderedGuesser,
    RED,
    EmbeddingGuesser,
    LLMGuesser,
    generate_board,
    parse_and_validate_clue,
    parse_clue,
    simulate_turn,
)


class CodenamesEnvironmentTests(unittest.TestCase):
    def test_llm_guesser_parser_returns_complete_board_permutation(self):
        words = ("ocean", "shore", "bridge")
        ranking = LLMGuesser.parse_ranking("1. bridge\n2. bridge\n3. ocean", words)
        self.assertEqual(ranking, ["bridge", "ocean", "shore"])

    def test_board_is_reproducible_and_has_standard_distribution(self):
        first = generate_board(seed=7)
        second = generate_board(seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first.words), 16)
        self.assertEqual([item.label for item in first.words].count(BLUE), 6)
        self.assertEqual([item.label for item in first.words].count(RED), 6)
        self.assertEqual([item.label for item in first.words].count(NEUTRAL), 3)
        self.assertEqual([item.label for item in first.words].count(ASSASSIN), 1)

    def test_prompt_contains_every_word_and_label(self):
        board = generate_board(seed=1)
        prompt = board.prompt()
        for item in board.words:
            self.assertIn(item.word, prompt)
            self.assertIn(f"{item.label}:", prompt)
        self.assertIn("NUMBER + 1 guesses", prompt)
        self.assertIn("metadata tags", prompt)

    def test_parser_accepts_only_expected_format(self):
        self.assertEqual(parse_clue("CLUE: ocean\nNUMBER: 3").word, "ocean")
        self.assertEqual(parse_clue("CLUE: OCEAN\nNUMBER: 3").number, 3)

        with self.assertRaises(InvalidClue):
            parse_clue("CLUE: T\nNUMBER: 1")
        for completion in (
            "ocean 3",
            "CLUE: two words\nNUMBER: 2",
            "CLUE: ocean\nNUMBER: 5",
            "CLUE: ocean\nNUMBER: two",
        ):
            with self.assertRaises(InvalidClue):
                parse_clue(completion)

    def test_board_overlap_is_rejected_in_both_directions(self):
        board = generate_board(seed=2)
        board_word = board.words[0].word
        with self.assertRaises(InvalidClue):
            parse_and_validate_clue(f"CLUE: {board_word}\nNUMBER: 1", board)
        with self.assertRaises(InvalidClue):
            parse_and_validate_clue(f"CLUE: {board_word}er\nNUMBER: 1", board)

    def test_role_tags_are_rejected_as_shortcut_clues(self):
        board = generate_board(seed=2)
        for tag in ("BLUE", "RED", "NEUTRAL", "ASSASSIN"):
            with self.assertRaises(InvalidClue):
                parse_and_validate_clue(f"CLUE: {tag}\nNUMBER: 1", board)

    def test_invalid_clue_gets_fixed_penalty(self):
        board = generate_board(seed=3)
        result = simulate_turn("not the format", board, OrderedGuesser([]))
        self.assertTrue(result.invalid)
        self.assertEqual(result.reward, -2.0)

    def test_turn_stops_after_first_non_blue_and_allows_number_plus_one(self):
        board = generate_board(seed=4)
        blue = [item.word for item in board.words if item.label == BLUE]
        red = [item.word for item in board.words if item.label == RED]
        neutral = [item.word for item in board.words if item.label == NEUTRAL]
        ordered = blue[:2] + [red[0]] + blue[2:] + neutral
        result = simulate_turn("CLUE: galaxy\nNUMBER: 4", board, OrderedGuesser(ordered))
        self.assertEqual(result.guesses, tuple(blue[:2] + [red[0]]))
        self.assertEqual(result.guessed_labels, (BLUE, BLUE, RED))
        self.assertEqual(result.reward, 1.0)

    def test_embedding_guesser_caches_board_and_repeated_clue_embeddings(self):
        class FakeEmbeddingModel:
            def __init__(self):
                self.calls = []

            def encode(self, texts, normalize_embeddings):
                self.calls.append(tuple(texts))
                return np.ones((len(texts), 2), dtype=np.float32)

        guesser = EmbeddingGuesser.__new__(EmbeddingGuesser)
        guesser.model = FakeEmbeddingModel()
        guesser._word_embedding_cache = {}
        guesser._clue_embedding_cache = {}
        guesser.rank("galaxy", ("apple", "bridge"))
        guesser.rank("galaxy", ("apple", "bridge"))
        self.assertEqual(guesser.model.calls, [("apple", "bridge"), ("galaxy",)])


if __name__ == "__main__":
    unittest.main()
