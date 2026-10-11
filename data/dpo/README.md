# Outcome-based DPO preferences

This dataset is generated from the existing Qwen2.5-1.5B SFT LoRA policy, not
from teacher-written winners or substituted board-word negatives. Eight actual
policy responses (one greedy, seven sampled) are collected for each of 500
training boards and 100 validation boards. Counts refer to attempted boards:
only clear gameplay preferences survive. See `stats.json` for final pair counts.

The completed initial run retained **360 pairs: 298 training and 62 validation**
from 4,800 sampled responses. Winners use 262 distinct clue words; declared
`NUMBER` counts are 1: 255, 2: 96, 3: 6, and 4: 3. Actual BLUE guesses are
1: 145, 2: 175, and 3: 40. Rejected responses include 247 RED outcomes and
41 ASSASSIN outcomes. All pairs passed outcome/provenance replay checks.

## Frozen teammate and selection

- `sentence-transformers/all-MiniLM-L6-v2`, normalized cosine ranking on CPU.
- Before **every** guess, abstain if similarity is below **0.30**. Equality is
  accepted. At most `NUMBER + 1` guesses, ending at the first non-BLUE result.
- Rewards: BLUE +1, NEUTRAL -0.3, RED -1, ASSASSIN -5; invalid clue -2.
- Winner: legal clue with at least one BLUE guess, positive reward, and no RED
  or ASSASSIN hit. Pick the highest-scoring safe response.
  Its `NUMBER` must not exceed the BLUE words still available on that board.
- Loser: another legal sampled response scoring at least one point less;
  prefer RED/ASSASSIN outcomes, choosing the closest inferior dangerous response.
- At most one pair per board. Skip ties, duplicate clue+number actions, invalid
  responses and boards without safe positive winners.

The 0.30 cutoff was fixed before evaluating 1,500 fresh boards. The paired test
in `results/abstention-fresh/report.json` passed: reward +0.0901, approximate
95% CI [+0.0562, +0.1241], 95.0% BLUE retention, and assassin hits 8.47% → 7.07%.
These are abstention results, **not evidence of DPO training improvements**.

## Files and audit

- `manifest.json`: checkpoint hashes, teammate weight/tokenizer hashes, sampling
  settings, vocabulary hashes, confirmation-report hash and excluded data sources.
- `boards.jsonl`: immutable board plan with shuffled words and 0–9 revealed words.
- `candidates.jsonl`: all actual model responses, including those later rejected.
- `train.jsonl`, `validation.jsonl`: conversational `prompt`, `chosen`, `rejected`
  messages plus board state and both simulated outcomes.
- `stats.json`: selection yield, clue/number diversity, revealed-state distribution
  and dangerous-negative counts.
- `audit.json`: replay certificate with hashes of all five source/data artifacts;
  the DPO trainer checks these hashes before using the data.
- `review.md`: full-audit summary and qualitative spot check, including weak
  associations and declared-count limitations.

Board splits exclude previously recorded SFT/evaluation/confirmation/smoke boards.
Signatures ignore tile permutation and revealed state, so another partial state
of the same labeled board cannot leak across splits. Reserved OOV vocabulary is
absent from these boards. Vocabulary words can repeat across different boards;
the validation split is not an additional OOV test.

Recompute each pair's actual outcome and check its sampled provenance:

```bash
.venv/bin/python build_dpo_pairs.py --validate-only
```

To resume sampling, repeat the original arguments and add `--resume`:

```bash
.venv/bin/python build_dpo_pairs.py --num-boards 500 --validation-boards 100 \
  --samples-per-board 8 --seed 130000 --min-similarity 0.30 \
  --output-dir data/dpo --resume
```

For DPO, initialize the trainable policy from the recorded SFT adapter and use
a **frozen copy of that same SFT policy** as the reference—not the raw Qwen base
with its adapter disabled. No DPO training is performed by the collector.

Run the separate trainer (one conservative epoch by default):

```bash
.venv-unsloth/bin/python train_dpo.py \
  --initial-adapter outputs/codenames-sft-1p5b \
  --data-dir data/dpo --learning-rate 1e-6 --beta 0.1 --epochs 1 \
  --output-dir outputs/codenames-dpo-1p5b
```

It caches the original SFT response log probabilities before any optimization,
then updates only the existing LoRA adapter. Repeat the same command with
`--resume` to restore the latest optimizer checkpoint; do not change run settings
or data while resuming. See the root README for objective, smoke tests and
SFT-versus-DPO evaluation commands.

## Trainer verification

On 2026-10-10, the four-step Unsloth/4-bit smoke test in
`outputs/codenames-dpo-smoke-v2` stopped at step 2, then resumed to step 4.
The LoRA parameters changed, SFT source files remained unchanged, and the frozen
reference cache kept the same SHA-256 before and after resume. The exported
lowest-validation-loss adapter came from checkpoint 2. Loading that adapter
through `evaluate.py --baseline-adapter-dir` also succeeded on eight fresh
partial boards. That tiny check had identical SFT/DPO gameplay summaries and
is **only a functionality check, not evidence of improved gameplay**.

The 58-test suite passes in both `.venv` and `.venv-unsloth`, including causal
response masking, end markers, pair ordering, frozen-reference gradients,
audit-tamper rejection, actual tiny-model LoRA updates and baseline evaluation:

```bash
.venv/bin/python -m unittest discover -s tests
.venv-unsloth/bin/python -m unittest discover -s tests
```

## Limitations

Preference labels optimize a deterministic embedding teammate and single-turn
reward, not human semantic quality or complete-game win rate. Some associations
are unintuitive. Winner counts are selected from actual outcomes, **not balanced
like the SFT curriculum**. Higher reward does not establish a universally best
clue. Inspect samples before training, and evaluate DPO on new boards with this
fixed teammate as well as a separate teammate if possible.
`NUMBER` is the policy's declared count, whereas `chosen_blue_guess_counts`
measures actual BLUE guesses. Those are not interchangeable: this dataset has
no independent intended-target annotations, and a guesser can stop before the
declared count. The statistics report these mismatches rather than claiming
that every declared target count is semantically verified.
