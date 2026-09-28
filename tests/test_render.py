import unittest
from pathlib import Path
import tempfile

from codenames_env import generate_board
import render
from render import build_reveal_states, render_turn_sequence


class RenderTests(unittest.TestCase):
    def test_reveal_states_start_hidden_and_reveal_in_order(self):
        board = generate_board(seed=40)
        guesses = [board.words[0].word, board.words[1].word]
        states = build_reveal_states(board, guesses)
        self.assertEqual(states, [(), (guesses[0],), tuple(guesses)])

    def test_unknown_guess_is_rejected(self):
        board = generate_board(seed=41)
        with self.assertRaises(ValueError):
            build_reveal_states(board, ["not-on-board"])

    def test_invalid_completion_is_visible_in_turn_title(self):
        board = generate_board(seed=42)
        captured_titles = []
        original_render_board = render.render_board

        def capture_title(*args, **kwargs):
            captured_titles.append(kwargs.get("title"))
            return Path(kwargs["output_path"])

        render.render_board = capture_title
        try:
            with tempfile.TemporaryDirectory() as directory:
                render_turn_sequence(
                    board,
                    [],
                    output_dir=Path(directory),
                    completion="CLUE: blue\nNUMBER: 2",
                )
        finally:
            render.render_board = original_render_board

        self.assertIn("OUTPUT: CLUE: blue | NUMBER: 2", captured_titles[0])


if __name__ == "__main__":
    unittest.main()
