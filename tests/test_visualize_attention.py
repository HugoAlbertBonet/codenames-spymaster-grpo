import unittest

from visualize_attention import (
    aggregate_last_token_attention,
    find_word_token_spans,
    reorder_scores_to_board,
    replace_board_word,
)
from codenames_env import CodenamesBoard, BoardWord


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        words = text.split()
        offsets = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        return {"offset_mapping": offsets, "input_ids": list(range(len(words)))}


class AttentionVisualizationTests(unittest.TestCase):
    def test_word_spans_use_character_offsets_for_board_words(self):
        tokenizer = FakeTokenizer()
        spans = find_word_token_spans("BLUE: snow, rainbow\nRED: jacket", tokenizer, ["snow", "rainbow", "jacket"])
        self.assertEqual(spans, ((1,), (2,), (4,)))

    def test_attention_is_summed_over_each_word_span(self):
        import torch

        # batch x heads x query x key; two heads, three prompt tokens.
        attentions = [torch.tensor([[[[0.1, 0.2, 0.7]], [[0.3, 0.4, 0.3]]]])]
        scores = aggregate_last_token_attention(attentions, ((0,), (1, 2)), prompt_length=3)
        self.assertEqual(tuple(scores.shape), (1, 2, 2))
        self.assertAlmostEqual(float(scores[0, 0, 0]), 0.1)
        self.assertAlmostEqual(float(scores[0, 0, 1]), 0.9)
        self.assertAlmostEqual(float(scores[0, 1, 1]), 0.7)

    def test_replace_board_word_only_changes_one_matching_occurrence(self):
        updated = replace_board_word("BLUE: snow, RED: snow", "snow")
        self.assertEqual(updated, "BLUE: [REMOVED], RED: snow")

    def test_attention_scores_are_mapped_from_grouped_prompt_to_tile_order(self):
        board = CodenamesBoard(
            (
                BoardWord("redword", "RED"),
                BoardWord("blueword", "BLUE"),
                BoardWord("neutralword", "NEUTRAL"),
                BoardWord("assassinword", "ASSASSIN"),
                BoardWord("blue2", "BLUE"),
                BoardWord("blue3", "BLUE"),
                BoardWord("blue4", "BLUE"),
                BoardWord("blue5", "BLUE"),
                BoardWord("red2", "RED"),
                BoardWord("red3", "RED"),
                BoardWord("red4", "RED"),
                BoardWord("red5", "RED"),
                BoardWord("neutral2", "NEUTRAL"),
                BoardWord("neutral3", "NEUTRAL"),
                BoardWord("red6", "RED"),
                BoardWord("blue6", "BLUE"),
            )
        )
        prompt_order = tuple(item.word for item in board.words if item.label == "BLUE")
        prompt_order += tuple(item.word for item in board.words if item.label == "RED")
        prompt_order += tuple(item.word for item in board.words if item.label == "NEUTRAL")
        prompt_order += ("assassinword",)
        scores = list(range(16))
        mapped = reorder_scores_to_board(scores, prompt_order, board)
        self.assertEqual(mapped[0], 6)  # redword is the first RED prompt item.
        self.assertEqual(mapped[1], 0)  # blueword is the first BLUE prompt item.
        self.assertEqual(mapped[3], 15)


if __name__ == "__main__":
    unittest.main()
