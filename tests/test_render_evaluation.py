import csv
import json
import tempfile
import unittest
from pathlib import Path

from codenames_env import generate_board
from render_evaluation import (
    board_from_evaluation_row,
    read_evaluation_rows,
    select_rows,
)


class RenderEvaluationTests(unittest.TestCase):
    def _write_csv(self, directory: str) -> Path:
        path = Path(directory) / "evaluation.csv"
        board = generate_board(seed=123)
        fields = ["model", "board_index", "board_seed", "guesses", "reward"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for model in ("base", "trained"):
                writer.writerow(
                    {
                        "model": model,
                        "board_index": 0,
                        "board_seed": 123,
                        "guesses": json.dumps([board.words[0].word]),
                        "reward": "1.0",
                    }
                )
        return path

    def test_legacy_csv_reconstructs_board_from_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = read_evaluation_rows(self._write_csv(directory))
            board = board_from_evaluation_row(rows[0])
            self.assertEqual(board, generate_board(seed=123))

    def test_both_selects_paired_rows_for_one_board(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = read_evaluation_rows(self._write_csv(directory))
            selected = select_rows(rows, count=1, model="both", seed=7)
            self.assertEqual([row["model"] for row in selected], ["base", "trained"])
            self.assertEqual({row["board_index"] for row in selected}, {"0"})

    def test_custom_model_labels_are_selectable(self):
        rows = [{"model": "sft-abstention", "board_index": "1"}]
        self.assertEqual(select_rows(rows, 1, "sft-abstention", 7), rows)


if __name__ == "__main__":
    unittest.main()
