# SFT → game-only GRPO diagnostic (2026-10-10)

Completed: **this 50-update pilot did not demonstrate improvement over SFT**.
Point estimates are worse, but the paired reward interval includes zero. This
does not establish that GRPO itself is harmful or cannot work.

## Fixed experiment

- Qwen/Qwen2.5-1.5B-Instruct, Unsloth 4-bit training, initialized from the
  original SFT adapter; a frozen second SFT adapter supplies the KL reference.
- MiniLM teammate on CPU, cosine cutoff 0.30, abstention before every guess.
- Game reward only: BLUE +1, RED −1, NEUTRAL −0.3, ASSASSIN −5, invalid −2.
- Pool: 1,000 unique boards, seed 200000, 5,000-word vocabulary, 0–9 revealed
  tiles. Recorded prior SFT/DPO/evaluation/abstention boards excluded.
- 50 updates, four clues/prompt, batch size 1, accumulation 8, learning rate
  5e-7, five warmup updates, cosine decay; reference beta 0.04.
- Temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1. No vLLM.
  Prompt/completion limits 512/48; longest prompt 378 tokens.

The original SFT files were unchanged. Checkpoints save only the policy adapter;
the frozen reference is rebuilt from the pinned SFT artifact during recovery.
GPU recovery from checkpoint 10 restored policy, optimizer, scheduler and RNG
state; the first resumed metrics matched. This is not a claim of complete
bitwise reproducibility across later sampling or library versions.

Frozen reference tensors remained unchanged and policy tensors changed. All
50 updates are logged locally. Across their sampled groups, 43% had zero reward
standard deviation; seven updates had no reward contrast in either group.
Mean sampled training reward was 0.213, **not** a held-out gameplay estimate.
KL ranged approximately 0.000015–0.00211; completions were 10–12 tokens,
with none hitting the cap. About 100 prompts were visited, not all 1,000 boards
in the pool. Equal rewards do not necessarily mean identical clues.

## Paired evaluation

1,000 identical partial boards, seed 170000, 5,000-word vocabulary, 0.30/all
teammate; greedy generation using ordinary BF16 evaluation, not quantized
evaluation. This is reused **development data**, not fresh confirmation.

| Metric | SFT | SFT → GRPO |
| --- | ---: | ---: |
| Average reward | 0.1590 | 0.1332 |
| Invalid clues | 6.3% | 6.5% |
| BLUE/turn | 1.020 | 1.016 |
| Assassin hits | 7.5% | 7.8% |
| RED hits | 30.6% | 31.0% |
| Abstention | 20.1% | 20.3% |

Paired reward change **−0.0258**, percentile bootstrap 95% interval
**[−0.0575, +0.0029]** (10,000 whole-board-pair resamples, NumPy seed 42).
13 boards improved, 18 worsened and 969 tied; completions changed on 66 boards.
Five new assassin hits and two avoided hits give the net increase of three.
The avoided hits became RED guesses, not clean BLUE turns.

Reward change over all 1,000 turns decomposes as BLUE −4, RED −4, NEUTRAL +1.2,
ASSASSIN −15, invalid −4 = **−25.8**. Added assassin penalties explain most of
the decrease. All SFT completions matched the earlier clue-only DPO baseline.

## Changed-turn review

- Board 272: **security**, NUMBER 1 → 2. Both first guess BLUE **safe** and
  **opportunity**; GRPO allows a third guess, ASSASSIN **charges** (+2 → −3).
- Board 305: **finance**, NUMBER 1 → 2, adds ASSASSIN **aid** after two BLUEs.
- Board 610: SFT **time** reaches BLUE **tomorrow**, **difference** (+2);
  GRPO **transportation** immediately reaches ASSASSIN **processing** (−5).
- Board 660: **lubrication** reaches BLUE **oil** (+1); **chemistry** adds
  ASSASSIN **nervous** (−4).
- Board 672: invalid **broadcasting** becomes legal **television**, but hits
  ASSASSIN **medical**. Clue validity alone is not sufficient.

These include both count-calibration and clue-choice failures. Keep SFT as the
baseline; audit candidate quality and NUMBER before extending this run. Any
checkpoint selected on these development boards needs fresh-board confirmation.
Scores are single-turn embedding-teammate outcomes, not human clue quality or
full-game win rates.

## Reproduction and evidence

The [experiment guide](experiments.md#sft--short-game-only-grpo) includes exact
train/evaluate commands. The [published summary](evaluation-summary.json)
records aggregate metrics and original CSV hashes. The GRPO CSV SHA-256 is
`d8b13c7f7e5d8aa68b7089ef484f7de766128443a9b7132da7aa4bd3f1268f2d`.

Detailed local artifacts are `results/sft-grpo-game-only-report.md`,
`outputs/codenames-sft-grpo-game-only-50/manifest.json`, `reference_audit.json`
in that output directory, checkpoints and the original evaluation CSV/transcripts.
Those weights and run artifacts are not published with the code.
