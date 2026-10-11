"""Render a random, inspectable sample of SFT boards to a standalone HTML file."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sft_data import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("data/sft/review.html"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    rows = load_records(args.dataset)
    selected = random.Random(args.seed).sample(rows, min(args.count, len(rows)))
    cards = []
    for row in selected:
        tiles = []
        for word, label in zip(row["board_words"], row["board_labels"]):
            classes = label.lower()
            if word in row["board_revealed"]:
                classes += " guessed"
            if word in row["intended_targets"]:
                classes += " target"
            tiles.append(f'<div class="tile {classes}"><strong>{escape(word)}</strong><small>{label}</small></div>')
        targets = ", ".join(row["intended_targets"])
        card = (
            f'<article data-number="{len(row["intended_targets"])}" data-guessed="{len(row["board_revealed"])}">'
            f'<h2>{escape(row["completion"].replace(chr(10), " · "))}</h2>'
            f'<p>{escape(row["id"])} · Already guessed: {len(row["board_revealed"])} · '
            f'Embedding turn reward: {row["audit"]["reward"]:.1f}</p>'
            f'<div class="board">{"".join(tiles)}</div><p>Intended targets: <b>{escape(targets)}</b></p>'
            f'<details><summary>Actual model prompt</summary><pre>{escape(row["prompt"])}</pre></details></article>'
        )
        cards.append(card)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Codenames SFT examples</title>
<style>
body{font:16px system-ui;margin:24px auto;max-width:1050px;padding:0 16px;background:#f5f7fa;color:#182230}
article{background:white;padding:20px;margin:20px 0;border-radius:12px;box-shadow:0 1px 5px #ccd}
h2{font-size:20px}.board{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.tile{padding:18px 4px;text-align:center;border:3px solid transparent;border-radius:8px;overflow-wrap:anywhere}
small{display:block;font-size:10px;margin-top:5px}.blue{background:#d8eaff}.red{background:#ffdada}
.neutral{background:#eee6cc}.assassin{background:#303746;color:white}.target{border-color:#1a9c55}
.guessed{opacity:.4;text-decoration:line-through}pre{white-space:pre-wrap;font-size:13px}
select{margin:0 12px 0 4px;padding:5px}[hidden]{display:none}
</style><h1>SFT board sample</h1><p>Green border = intended target. Faded/crossed words = already guessed.
Labels are shown here for inspection. The model receives the grouped text prompt below each board.</p>
<p>Number <select id="number"><option value="">All</option><option>1</option><option>2</option><option>3</option><option>4</option></select>
Already guessed <select id="guessed"><option value="">All</option><option>0</option><option>3</option><option>6</option><option>9</option></select></p>
'''
    html += "\n".join(cards) + '''<script>
const filters=[document.querySelector('#number'),document.querySelector('#guessed')];
filters.forEach(filter=>filter.addEventListener('change',()=>document.querySelectorAll('article').forEach(card=>{
card.hidden=filters.some(filter=>filter.value && card.dataset[filter.id]!==filter.value);
})));</script></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Rendered {len(selected)} examples to {args.output}")


if __name__ == "__main__":
    main()
