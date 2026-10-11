"""Keep the public gallery replayable and its documentation links local."""

import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlparse

from PIL import Image

from codenames_env import OrderedGuesser, simulate_turn
from render_evaluation import board_from_evaluation_row
from scripts.render_readme_examples import aggregate


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_gallery_replays_saved_outcomes_and_contains_paired_board(self):
        manifest = json.loads((ROOT / 'docs/sft/examples.json').read_text())
        self.assertIsNone(manifest['min_similarity'])
        rows = {}
        for example in manifest['examples']:
            row = example['row']
            board = board_from_evaluation_row(row)
            guesses = json.loads(row['guesses'])
            result = simulate_turn(row['completion'], board, OrderedGuesser(guesses))
            self.assertEqual(list(result.guesses), guesses)
            self.assertEqual(list(result.guessed_labels), json.loads(row['guessed_labels']))
            self.assertEqual(result.invalid, row['invalid'].lower() == 'true')
            self.assertAlmostEqual(result.reward, float(row['reward']))
            self.assertEqual(sum(label == 'BLUE' for label in result.guessed_labels), int(row['correct_guesses']))
            rows[example['name']] = row
        for field in ('board_index', 'board_seed', 'board_words', 'board_labels', 'board_revealed'):
            self.assertEqual(rows['base'][field], rows['sft'][field])
        self.assertIn('ASSASSIN', json.loads(rows['failure']['guessed_labels']))

    def test_gallery_contains_real_animated_frames_and_static_answer_key(self):
        manifest = json.loads((ROOT / 'docs/sft/examples.json').read_text())
        for example in manifest['examples']:
            with Image.open(ROOT / 'docs/sft' / f"{example['name']}.gif") as image:
                self.assertEqual(image.n_frames, len(json.loads(example['row']['guesses'])) + 1)
                self.assertEqual(image.info['loop'], 0)
                self.assertEqual(image.info['duration'], manifest['duration_ms'])
                image.verify()
        with Image.open(ROOT / 'docs/sft/answer.png') as image:
            image.verify()

    def test_summary_aggregation_includes_invalid_turns_in_rates(self):
        rows = [dict(model='base', reward='-2', invalid='True', correct_guesses='0', guessed_labels='[]'),
                dict(model='base', reward='-4', invalid='False', correct_guesses='1',
                     guessed_labels='["BLUE", "ASSASSIN"]')]
        metrics = aggregate(rows)['base']
        self.assertEqual(metrics['average_reward'], -3)
        self.assertEqual(metrics['invalid_clue_percent'], 50)
        self.assertEqual(metrics['assassin_hit_percent'], 50)
        self.assertEqual(metrics['average_correct_guesses'], 0.5)

    def test_published_evaluation_protocols_remain_separate(self):
        summary = json.loads((ROOT / 'docs/evaluation-summary.json').read_text())
        for name, suite in summary['suites'].items():
            self.assertEqual(set(suite['metrics']), {'base', 'trained'})
            for metrics in suite['metrics'].values():
                self.assertEqual(metrics['episodes'], 1000)
            if name.startswith('evaluation-sft-') and 'grpo' not in name:
                self.assertIsNone(suite['min_similarity'])
                self.assertEqual(suite['baseline'], 'raw Qwen')
            else:
                self.assertEqual(suite['min_similarity'], 0.30)
                self.assertEqual(suite['baseline'], 'SFT adapter')
        self.assertTrue(summary['abstention_confirmation']['passed'])
        self.assertEqual(summary['abstention_confirmation']['overall']['baseline']['episodes'], 1500)

    def test_markdown_relative_links_and_heading_anchors_exist(self):
        paths = [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md')),
                 ROOT / 'data/sft/README.md', ROOT / 'data/dpo/README.md',
                 ROOT / 'data/dpo/review.md', ROOT / 'data/dpo-clue-only/README.md',
                 ROOT / 'data/dpo-clue-only/review.md']
        for path in paths:
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                if urlparse(target).scheme:
                    continue
                relative, _, anchor = unquote(target).partition('#')
                linked = (path.parent / relative).resolve() if relative else path
                self.assertTrue(linked.exists(), f'Broken link in {path}: {target}')
                if anchor and linked.suffix == '.md':
                    headings = re.findall(r'^#+\s+(.+)$', linked.read_text(), re.MULTILINE)
                    slugs = [re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-') for title in headings]
                    self.assertIn(anchor, slugs, f'Broken anchor in {path}: {target}')


if __name__ == '__main__':
    unittest.main()
