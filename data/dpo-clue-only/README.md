# Clue-only DPO diagnostic dataset

This is a **reselection**, not new policy generation or a new independent board
sample. It reuses all 4,800 original SFT completions on the same 500 training and
100 validation board states from `data/dpo`. Original files and checkpoints are
unchanged. Both copied JSONL cache files are byte-for-byte identical to the source.

There are **318 pairs: 261 training / 57 validation**, with 240 distinct winning
clue words. Every pair has **different clue words and the same NUMBER**. No clue,
number or completion was rewritten. All 318 pairs passed outcome/provenance
replay, and `train_dpo.py --dry-run` validates the audited data and tokenization.

## Selection

The same frozen MiniLM teammate uses cosine cutoff 0.30 before every guess,
up to NUMBER + 1 guesses, and the same BLUE/NEUTRAL/RED/ASSASSIN rewards.
The same safe-positive winner, available-BLUE count feasibility, and minimum
one-point reward gap rules apply. Invalid completions and ties remain excluded.

Search safe winners in descending reward order. For each, seek a lower-reward
response with the **same NUMBER and a different clue word**. If no compatible
loser exists for the global best winner, try the next safe winner. Prefer the
closest clearly inferior RED/ASSASSIN alternative when available. Keep at most
one pair per board; skip a board if no compatible comparison exists.

Simply filtering the old selected pairs would retain only 190 comparisons
(157 training / 33 validation). Reselecting from every cached candidate recovers
128 additional comparisons. All selected boards were already among the original
360 selected boards; this is not 128 new board samples.

## Coverage and limitations

| Declared NUMBER | Training | Validation |
| --- | ---: | ---: |
| 1 | 188 | 40 |
| 2 | 67 | 17 |
| 3 | 5 | 0 |
| 4 | 1 | 0 |

194 losers hit RED and 38 hit ASSASSIN. Winners made one BLUE guess in 133 cases,
two in 159, and three in 26. 39 winners made fewer BLUE guesses than their
declared NUMBER; declared counts are not independently verified semantic targets.
43 winners include a guess with cosine at most 0.31, just above the 0.30 cutoff.
The selected states include 37 full boards and 281 partial boards.

These are fixed-teammate outcome preferences, not human-quality annotations.
Same-number pairing removes the **direct budget contrast** from each pair;
it does not freeze the model's number behavior. The full assistant response,
including NUMBER and the end marker, remains part of the DPO loss. Number=1
examples still dominate. This dataset is suitable for a **small diagnostic
ablation**, not evidence of multi-target strategy or general gameplay improvement.
See `review.md` for the 24-pair AI-assisted qualitative spot check.

## Provenance and commands

`manifest.json` pins source artifacts, SFT weights, teammate artifacts, inputs
and the unchanged board split. `audit.json` covers exact hashes of the five
derived data/source files. Reserved OOV words are absent, and training/validation
boards do not overlap. The source train and validation assignments are preserved.

The rebuild is already complete. To repeat it in a **new** directory:

```bash
.venv/bin/python build_dpo_pairs.py \
  --source-dir data/dpo --pair-mode clue-only \
  --output-dir data/dpo-clue-only-repeat

.venv/bin/python build_dpo_pairs.py \
  --output-dir data/dpo-clue-only-repeat --validate-only
```

`--source-dir` inherits generation, board, teammate, reward-gap and input settings
from the source manifest. It never loads the spymaster or generates responses.
For legacy manifests without input paths, vocabulary/confirmation CLI paths
must match the recorded hashes. Add `--resume` to the same rebuild command after
an interruption; new nonempty outputs and source/child/parent destinations are
refused. Omitting `--pair-mode` retains the original `gameplay` selection mode.

Training has **not** been started. For the diagnostic, initialize from SFT again,
not a previous DPO adapter, and keep the first experiment's settings:

```bash
.venv-unsloth/bin/python train_dpo.py \
  --initial-adapter outputs/codenames-sft-1p5b \
  --data-dir data/dpo-clue-only --epochs 1 --learning-rate 1e-6 --beta 0.1 \
  --batch-size 1 --gradient-accumulation-steps 8 --save-steps 10 \
  --output-dir outputs/codenames-dpo-clue-only
```

One epoch is about 33 optimizer updates, versus 38 previously, because fewer
pairs survive. Thus this is not a perfectly matched-update ablation. Keep SFT
as the baseline and evaluate gameplay on the existing development boards first;
confirm any selected improvement on fresh seeds and reserved OOV boards.
