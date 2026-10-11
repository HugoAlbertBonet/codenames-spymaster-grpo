# Codenames SFT curriculum

These **2,000 synthetic examples** are derived from the association bank authored
by the assistant in `data/sft_clue_bank.tsv`, then assembled and checked by
`scripts/generate_sft_dataset.py`. They are **not 2,000 independent frontier-model
calls reasoning over uniformly random boards**. No external teacher API was
called or billed. The examples are intended as a first SFT curriculum before
game reward training.

`train.jsonl` has 1,800 examples and `validation.jsonl` has 200. Clue families
are held out between splits; related semantic concepts can still share target
words. This validation tests imitation loss on held-out clue families, not
full gameplay generalization. Fresh random-board and OOV evaluations are still
required. The 89 reserved OOV words do not occur as board words in either split.

There are 190 distinct clue families. Each number (1–4) appears 500 times;
the numbers of previously guessed words are 0, 3, 6, and 9. Tile order and
within-tag word order are shuffled. The model sees the shared environment
prompt with remaining words grouped by role. It does not see tile coordinates,
targets, audit scores, or metadata. Position has no meaning to the current
guesser. There are no fabricated prior clues: these are nonterminal board
snapshots, not trajectories through a full game.

The finite association bank provides plausible target connections. Distractors
are sampled from 5,476 words (the expanded vocabulary plus association targets,
minus reserved OOV words), constrained to rank below intended targets under
the frozen `all-MiniLM-L6-v2` teammate. This makes the curriculum easier than
unconstrained random boards. Every clue is legal under the existing strict
substring rule, targets only remaining BLUE words, and has exactly NUMBER
intended targets. The audit rejects RED/ASSASSIN guesses within NUMBER + 1;
a NEUTRAL extra guess is allowed and subtracts 0.3. This check does not certify
that every clue is natural, optimal, or safe for a human/LLM teammate.

`stats.json` contains counts, source hashes, and embedding-audit metrics. Inspect
`review.html` before training; its filters let you inspect clue counts and
partial-board states. To sample other examples:

```bash
uv run python scripts/inspect_sft_dataset.py --count 32 --seed 22
```

To reproduce/check the data:

```bash
uv run python scripts/generate_sft_dataset.py
uv run python scripts/generate_sft_dataset.py --validate-only
```

SFT trains only assistant tokens (`CLUE`, `NUMBER`, and the chat end marker).
Prompt tokens and padding have loss labels of -100. Strict token-length checks
reject truncation rather than quietly clipping boards/answers. Begin from the
original instruct model rather than an unsuccessful GRPO adapter:

```bash
.venv-unsloth/bin/python train_sft.py --dry-run
.venv-unsloth/bin/python train_sft.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 2 \
  --learning-rate 5e-5 \
  --output-dir outputs/codenames-sft-1p5b
```

The dry run also validates the installed Transformers training options before
model loading. Warmup is passed as an explicit optimizer-step count for
Transformers 4/5 compatibility: one step for a five-step smoke test, or 23 steps
for the default two epochs with 1,800 examples and accumulation of eight.

The default backend is Unsloth with 4-bit QLoRA, rank 16, and batch size 1.
The standard Transformers backend is available via `--backend transformers`.
Checkpoints are saved/evaluated every 50 optimizer steps; the lowest validation
loss checkpoint is exported. This selects imitation quality, not necessarily
the best game score. To resume the same SFT run, repeat its command with
`--resume`. Use a new output directory if changing the data or training settings.

Evaluate SFT before starting GRPO:

```bash
uv run python evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir outputs/codenames-sft-1p5b \
  --vocabulary-file data/sft/board_vocabulary.txt \
  --num-boards 1000 --seed 80000 \
  --output-csv results/evaluation-sft-random.csv \
  --transcript-file results/transcripts-sft-random.txt
```

Repeat with `--max-revealed 9` (and different output paths) for random partial
boards, and with `data/evaluation_vocabulary_oov.txt` for OOV. A good first
signal is fewer invalid clues without one clue dominating the responses; then
check correct guesses, assassin frequency, and game reward. High curriculum
reward alone does not establish an improvement on random boards.

## Completed experiments

The recorded two-epoch SFT run made 450 optimizer updates and exported
checkpoint 100, selected by validation imitation loss. On 1,000 random full
boards, reward improved from raw Qwen's −1.2836 to 0.3547 and invalid clues
fell from 55.2% to 5.6%. Assassin hits nevertheless increased from 3.8% to 5.8%.
See the [current README](../../README.md#evaluation-results) for partial/OOV
results and the real evaluation animations.

A fixed abstaining teammate improved safety in a separate fresh-board test.
Later outcome-based DPO, clue-only DPO and a 50-update game-only GRPO pilot did
not demonstrate improvement over SFT. Keep SFT as the baseline; do not assume
that extending RL will fix candidate quality or NUMBER calibration. Exact
recorded experiment commands are in the
[experiment guide](../../docs/experiments.md#sft--short-game-only-grpo).

`--initial-adapter` loads SFT weights as trainable parameters and starts a new
optimizer. `--resume` restores checkpoints within the same GRPO run. The latest
pilot uses a frozen SFT reference and disables semantic bonuses and duplicate
penalties. Clue diversity is a diagnostic, not a substitute for board-specific
game quality. Preference artifacts pin the original SFT weights; newly trained
adapters need fresh confirmation and pair collection rather than bypassing hashes.
