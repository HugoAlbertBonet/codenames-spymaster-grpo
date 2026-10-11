# Experiment guide and historical GRPO results

See the [main README](../README.md) for the current Qwen2.5-1.5B SFT baseline,
updated results and animations. The opening results below describe the earlier
0.5B GRPO experiment, not the current SFT model. Later sections preserve the
SFT, abstention, DPO and game-only GRPO reproduction instructions.

Commands below show the **recorded local run paths** and assume execution from
the repository root. Model weights and confirmation artifacts are not bundled.
The published DPO data pins the original SFT weights: a newly trained adapter
needs its own abstention confirmation, collection and replay audit in new
directories (for example `results/abstention-local` and `data/dpo-local`). Pass
the new report with `--confirmation-report` and train with `--data-dir` pointing
to the new preferences. Never overwrite the bundled evidence or disable hash
checks to force a different adapter to match it.

Reinforcement-learning fine-tuning of a small language model (Qwen2.5-0.5B-Instruct)
to play the **spymaster** in Codenames. The model sees the board and its hidden
team labels and must output one clue word and a number:

```text
CLUE: <single word>
NUMBER: <integer 1-4>
```

There is no labeled "correct" clue. The policy is trained with **Group Relative
Policy Optimization (GRPO)** and LoRA: a fixed sentence-embedding guesser turns each
clue into guesses, and a game simulator scores the turn.

| Base model: invalid clue (−2.0) | GRPO-trained: `FLOWER 2` → rose, grass, star (+3.0) |
| :---: | :---: |
| ![Base model turn](example-base.gif) | ![Trained model turn](example-trained.gif) |

## Results

Held-out evaluation on **1,000 boards** (seed `30000`, never used in training).
Both models see identical boards, so every board is a paired comparison.

| Model | Average reward | Invalid clues | BLUE guesses/turn |
| --- | ---: | ---: | ---: |
| Base Qwen2.5-0.5B-Instruct | −1.463 | 63.4% | 0.249 |
| GRPO + Unsloth + LoRA | **−0.681** | **8.7%** | **0.607** |

The paired reward delta is **+0.782**: the trained model wins 526 boards, ties 360,
and loses 114.

### Does it transfer?

Two stricter tests on the same 100 boards (seed `40000`), built only from words
that never appear in training (`data/evaluation_vocabulary_oov.txt`):

| Guesser at evaluation | Model | Average reward | Invalid clues | BLUE guesses/turn |
| --- | --- | ---: | ---: | ---: |
| Embedding guesser (training-time) | Base | −1.437 | 57% | 0.36 |
| | GRPO | **−0.825** | **8%** | **0.61** |
| Frozen LLM guesser (never seen in training) | Base | −1.387 | 57% | 0.26 |
| | GRPO | **−0.976** | **8%** | **0.49** |

The gain survives this small test with a different teammate. This is encouraging,
but does not establish human-quality clues or rule out embedding-specific shortcuts.

### What did not work (yet)

- **Clue diversity.** The trained policy's most common clue is `secret`
  (425/1,000 turns). Performance is stable but relies on a few safe, generic clues.
- **Scaling to Qwen2.5-1.5B.** The 1.5B base model is already much better
  (30.5% invalid clues vs 63.4% for 0.5B), but GRPO did not improve it: one run
  raised invalid clues to 50.5%, and a lower learning rate (1e-7) left it
  unchanged (35.0% → 36.5%). Evaluated on 200 boards.

## How it works

1. **Environment** (`codenames_env.py`): generates 16-word boards (6 BLUE, 6 RED,
   3 NEUTRAL, 1 ASSASSIN), validates clues, and simulates the guesser.
2. **Prompt**: explains the turn cycle, scoring, the `NUMBER + 1` guess allowance
   and clue rules. Board words are grouped by label because tile position carries
   no meaning; the label names themselves are rejected as clues so the model
   cannot earn reward by copying `BLUE`.
3. **Guesser**: ranks the remaining words by cosine similarity with
   `sentence-transformers/all-MiniLM-L6-v2`, makes up to `NUMBER + 1` guesses, and
   stops at the first non-BLUE word.
4. **Reward** (per turn):

   | Event | Reward | Reason |
   | --- | ---: | --- |
   | BLUE guessed | `+1.0` | Progress toward the spymaster's objective |
   | NEUTRAL guessed | `-0.3` | Turn ends, game continues |
   | RED guessed | `-1.0` | Helps the opposing team and ends the turn |
   | ASSASSIN guessed | `-5.0` | Ends the game, so this risk must dominate |
   | Invalid clue | `-2.0` | Discourages format and rule violations |

   Earlier GRPO runs used a bounded semantic safety-margin bonus and a small
   duplicate-clue penalty. The latest `--game-reward-only` pilot disables both;
   evaluation always reports the unshaped game reward.
5. **Training** (`train_grpo.py`, `train_grpo_unsloth.py`): GRPO with LoRA on
   procedurally generated boards; the Unsloth path uses vLLM-backed rollouts.

**Why a language model?** The action is an open-ended word from a large
vocabulary whose value depends on its relation to several board words at once.
A fixed-output classifier does not represent that action space naturally; a small
language model gives a practical policy over it.

## Project structure

| Path | Purpose |
| --- | --- |
| `codenames_env.py` | Board generation, clue rules, embedding/LLM guessers, turn simulation, rewards |
| `train_grpo.py` | Procedural dataset, GRPO training, LoRA configuration |
| `train_grpo_unsloth.py` | Unsloth + vLLM accelerated GRPO training |
| `train_sft.py` | Response-only LoRA SFT with optional 4-bit Unsloth acceleration |
| `data/sft/`, `data/sft_clue_bank.tsv` | 2,000 checked synthetic examples, association seeds, and HTML preview |
| `evaluate.py` | Paired base-vs-trained evaluation, CSV metrics, transcripts |
| `confirm_abstention.py` | Fresh-board paired test of a preselected teammate cutoff |
| `build_dpo_pairs.py`, `outcome_data.py` | Resumable SFT clue sampling and audited outcome-based preferences |
| `render.py`, `render_evaluation.py` | Board frames and animated GIFs of evaluated turns |
| `visualize_attention.py` | Attention heatmaps and occlusion analysis per board word |
| `scripts/build_training_vocabulary.py` | Reproducibly builds the 5,000-word training vocabulary |
| `data/` | Training vocabulary and the disjoint evaluation-only vocabulary |
| `tests/` | Fast unit tests (no model downloads), run in CI |

## Setup

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                          # exact versions from uv.lock
uv run python -m unittest discover -s tests -v   # fast checks, no model downloads
uv run python train_grpo.py --smoke-test
```

With pip: `pip install -r requirements.txt`.

The ML stack (PyTorch, model weights) is large and meant for a CUDA machine. The
default model fits an 8 GB laptop GPU with LoRA, batch size 1 and short completions.

### Optional Unsloth environment

Unsloth may need newer Torch/TRL/vLLM versions than the baseline, so it lives in a
separate environment:

```bash
uv venv .venv-unsloth --python 3.12
uv pip install --python .venv-unsloth/bin/python -r requirements-unsloth.txt
```

## Training

### New experiment: SFT before GRPO

The prepared dataset contains 1,800 training and 200 validation examples with
varied clue counts (1–4), shuffled words, and partially revealed boards. Open
`data/sft/review.html` to inspect a random sample and read
[the dataset card](../data/sft/README.md) for provenance and limitations. These
labels come from assistant-authored associations, procedural construction,
and embedding checks, rather than 2,000 independent teacher-model rollouts.

```bash
.venv-unsloth/bin/python train_sft.py --dry-run
.venv-unsloth/bin/python train_sft.py --output-dir outputs/codenames-sft-1p5b
```

The dry run checks chat-template boundaries, response-only loss masks and
sequence lengths. SFT exports the adapter with lowest held-out imitation
loss; evaluate game behavior on fresh random boards before starting GRPO.
Both GRPO trainers accept `--initial-adapter` to warm-start from SFT weights
with a new optimizer. Trainers and evaluation accept `--max-revealed 9`
for random partial states. The dataset card includes evaluation and GRPO
commands. Response masks follow the
[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
label interface; loading existing adapters follows
[Unsloth continuation](https://unsloth.ai/docs/basics/continued-pretraining).

### Controlled abstention, then outcome-based DPO data

The embedding teammate can stop before guessing a word whose cosine similarity
is below a fixed threshold. This does **not** inspect its hidden label. The
default remains no abstention; add `--min-similarity 0.30` to evaluation or either
GRPO trainer. `--abstention-mode all` checks every guess, including the first;
`extra` checks only the optional `NUMBER + 1` guess. Equality to the threshold
is accepted. A valid clue with no sufficiently similar guesses earns zero,
not an invalid-clue penalty. Cosine similarity is not a calibrated probability.

First confirm the preselected cutoff, without retuning it on this test:

```bash
.venv/bin/python confirm_abstention.py --num-boards 500 --seed 90000 \
  --min-similarity 0.30 --output-dir results/abstention-fresh
```

This generates **1,500 fresh boards**: 500 full, 500 random partial (up to nine
revealed words), and 500 reserved OOV boards. It excludes labeled boards used in
the saved SFT dataset and prior self-contained evaluation CSVs, even if board
order or revealed state differs. Each SFT completion is replayed with both
teammates, so the comparison measures abstention alone. Read `report.json`
for reward, BLUE guesses, assassin/RED rates, abstentions, zero-guess turns,
and approximate paired 95% confidence intervals. The prespecified overall gate
requires at least 300 turns, positive lower reward-delta CI, fewer assassin hits,
and at least 90% of the original BLUE guesses. A pass is evidence for this
single-turn test mixture, not a guarantee for every board suite or full games.

After a pass, collect preferences using **the same SFT checkpoint and teammate**:

```bash
.venv/bin/python build_dpo_pairs.py --num-boards 500 --validation-boards 100 \
  --samples-per-board 8 --seed 130000 --max-revealed 9 \
  --min-similarity 0.30 --output-dir data/dpo

.venv/bin/python build_dpo_pairs.py --validate-only
```

The local Qwen2.5-1.5B SFT checkpoint passed the 1,500-turn fresh test at cutoff
0.30 (`all`): reward **0.2015 → 0.2916**, assassin hits **8.47% → 7.07%**,
RED hits **42.73% → 35.67%**, and BLUE guesses **1.1873 → 1.1280** per turn.
The paired reward improvement was +0.0901 (approximate 95% CI +0.0562 to +0.1241);
95.0% of BLUE guesses were retained. OOV reward was nearly unchanged; most of
the improvement came from full and partial in-vocabulary boards. These numbers
describe the **unchanged SFT policy with a different teammate**, not DPO gains.

The command above builds an initial, auditable dataset. To scale later, use
`--num-boards 2000 --validation-boards 250 --seed 150000 --output-dir data/dpo-large`;
do not change board counts and use `--resume` on the existing dataset.

These are attempted board counts, not guaranteed preference counts. For each
board the sampler produces one greedy and seven sampled complete clue responses.
The frozen teammate scores their actual turns. One safe positive winner is paired
with a response scoring at least one point less, preferentially a RED/ASSASSIN
outcome. Ties, invalid responses, duplicate actions and boards without safe winners
are skipped. A winner cannot claim more BLUE words than remain on its board.
No clues are manufactured by replacing them with board words, and
declared intended targets play no role in labeling. The script refuses to collect
data if the fresh-board confirmation failed or its policy/teammate differs.
These labels express this embedding teammate's preferences, not independently
verified human-quality associations; unintuitive high-scoring clues can remain.
Review examples before training and judge the resulting policy on fresh gameplay.

`data/dpo/train.jsonl` and `validation.jsonl` use conversational
`prompt`/`chosen`/`rejected` fields accepted by
[TRL's DPO trainer](https://huggingface.co/docs/trl/dpo_trainer), plus board and
outcome metadata for auditing. Split by labeled board, not by pair; reserve OOV
words and all confirmation boards. `stats.json` reports actual pair counts,
chosen clue counts and rejected dangerous outcomes. DPO should start from this
SFT adapter and use a **frozen copy of the same SFT adapter as its reference**,
not the adapter-disabled raw base model. This step prepares data; it does not
launch DPO training.

Both commands require CUDA for policy generation unless `--allow-cpu` is explicit,
save completed board rollouts incrementally, and accept `--resume` with the same
arguments. Model/adaptor hashes, dataset configuration and the board plan are
recorded; changing settings requires a new directory. They use the ordinary
Transformers environment and no paid model API. On this 8 GB GPU, generation is
batched with BF16 when supported; sampling does not require training gradients.
`--validate-only` reads the saved sampling/teammate settings from its manifest;
custom vocabulary and confirmation-report locations must still be supplied if
different from the defaults.

To inspect fresh abstaining turns with the existing renderer:

```bash
.venv/bin/python render_evaluation.py \
  --evaluation-file results/abstention-fresh/evaluation-partial.csv \
  --model sft-abstention --count 5 --show-labels \
  --output-dir results/abstention-fresh/renders
```

New evaluation CSVs include `abstained`, `stop_reason`, `guess_similarities`,
`min_similarity` and `abstention_mode`; older CSVs remain renderable.

### SFT-initialized DPO

The replay audit and qualitative pair review are in
[`data/dpo/review.md`](../data/dpo/review.md). The current dataset has **298 training
and 62 validation pairs** from actual SFT clue outcomes under the frozen
MiniLM teammate (cosine cutoff **0.30**, abstention before **every** guess).
These are proxy preferences, not 360 human-approved best clues: some winning
associations are weak, and NUMBER=3/4 examples are rare.

`train_dpo.py` continues the **existing SFT LoRA adapter**, with the **same SFT
policy as its frozen reference**, not raw Qwen with the adapter disabled. It
uses the standard [sigmoid DPO objective](https://huggingface.co/docs/trl/dpo_trainer):

```text
loss = -log sigmoid(beta * [(log P(chosen) - log P_ref(chosen))
                         - (log P(rejected) - log P_ref(rejected))])
```

Probabilities are summed over assistant tokens, including the chat end marker;
prompt and padding tokens are masked. A small Transformers Trainer extension
implements this objective explicitly across the two installed library versions.
It does not depend on TRL's version-specific reference-adapter defaults. There is
no SFT mixture, length normalization, or separately trained reward model.
`beta` scales the reference-adjusted preference margin; the conservative
learning rate limits how far this first experiment moves the SFT policy.

For your 8 GB GPU, reference scores are precomputed once with no gradients and
stored in `reference_logps.json`. Only one quantized 1.5B model is resident;
Unsloth gradient checkpointing and response-tail logits reduce memory use.
One batch item means **one pair / two responses**. The embedding guesser is not
run during optimization; its fixed outcomes already determined the labels.

Check data/tokenization without loading model weights:

```bash
.venv-unsloth/bin/python train_dpo.py --dry-run
```

Start the first full experiment:

```bash
.venv-unsloth/bin/python train_dpo.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --initial-adapter outputs/codenames-sft-1p5b \
  --data-dir data/dpo --epochs 1 --learning-rate 1e-6 --beta 0.1 \
  --batch-size 1 --gradient-accumulation-steps 8 --save-steps 10 \
  --output-dir outputs/codenames-dpo-1p5b
```

This is about 38 optimizer updates, not another 1,000-step GRPO run. Default
reference batch size 1 matches train/eval padding behavior to limit numerical
differences. Saving every 10 steps saves optimizer, scheduler and RNG state.
To stop, use Ctrl+C; repeat **the same command with `--resume`** to continue
from the latest complete checkpoint. Work since that checkpoint is replayed.
If interrupted during reference scoring, its cached prefix resumes instead.
Run settings, package versions, data hashes, tokenizer and SFT weights are
locked; changed settings require a new output directory. Existing nonempty
outputs and overwriting the SFT reference are refused.

For a short train/resume test, use a separate output directory:

```bash
.venv-unsloth/bin/python train_dpo.py \
  --max-train-pairs 16 --max-validation-pairs 8 --max-steps 4 \
  --save-steps 1 --gradient-accumulation-steps 2 --stop-after-steps 2 \
  --output-dir outputs/codenames-dpo-your-smoke

# Continue the same planned four-step schedule, not a new two-step schedule.
.venv-unsloth/bin/python train_dpo.py \
  --max-train-pairs 16 --max-validation-pairs 8 --max-steps 4 \
  --save-steps 1 --gradient-accumulation-steps 2 --resume \
  --output-dir outputs/codenames-dpo-your-smoke
```

The exported adapter is the checkpoint with lowest validation DPO loss; it is
**not automatically the best gameplay policy**. `dpo_run.json` records validation
preference accuracy/margin, source/cache hashes, optimizer-update verification
and the exported checkpoint. An unchanged best adapter is reported explicitly.

Compare **SFT versus DPO**, on new boards with the exact fixed teammate:

```bash
.venv/bin/python evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --baseline-adapter-dir outputs/codenames-sft-1p5b \
  --adapter-dir outputs/codenames-dpo-1p5b \
  --vocabulary-file data/training_vocabulary.txt \
  --min-similarity 0.30 --abstention-mode all \
  --num-boards 1000 --max-revealed 9 --seed 170000 \
  --output-csv results/evaluation-dpo-1000.csv \
  --transcript-file results/transcripts-dpo-1000.txt
```

With `--baseline-adapter-dir`, CSV `model=base` now means **SFT baseline**;
`model=trained` means **DPO**. Without this option, the original raw-base
comparison is unchanged. Judge gameplay reward, BLUE guesses, invalid rate and
assassin hits; preference accuracy alone is not evidence of better play. Repeat
with reserved OOV vocabulary and another fresh seed before accepting the policy.

### Clue-only DPO diagnostic

The first gameplay-DPO experiment did not demonstrate improvement over SFT on
the 1,000 seed-170000 development boards, including checkpoints 20, 30 and 38.
The next diagnostic separates clue choice from the direct guess-budget contrast:
`build_dpo_pairs.py --pair-mode clue-only` selects **different clue words with
the same NUMBER**, searching all compatible safe winners before skipping a board.
The default `gameplay` mode preserves the original selection logic.

The new dataset is already rebuilt and audited in
[`data/dpo-clue-only`](../data/dpo-clue-only/README.md): **261 training / 57 validation
pairs**, recovered from the original 4,800 cached responses. Source data and
checkpoints are unchanged. This is a same-board reselection, not fresh validation
data or new generated examples. To reproduce in a new directory:

```bash
.venv/bin/python build_dpo_pairs.py \
  --source-dir data/dpo --pair-mode clue-only \
  --output-dir data/dpo-clue-only-repeat

.venv/bin/python build_dpo_pairs.py \
  --output-dir data/dpo-clue-only-repeat --validate-only
```

Cache-only rebuilds inherit source settings and never load the spymaster or call
generation. Source hashes and board splits are pinned. Use a separate empty
output directory; add `--resume` to the same rebuild command to continue.
The DPO trainer also rejects clue-only pairs with unequal NUMBER or identical
clue words, regardless of their saved outcomes.

See the [qualitative review](../data/dpo-clue-only/review.md) before training. Weak
embedding associations remain; there are only six NUMBER=3/4 training pairs and
none in validation. Equal counts remove a direct preference for smaller budgets,
but do not freeze number behavior or balance the curriculum. Keep the objective,
reference policy, teammate and hyperparameters unchanged for the diagnostic:

```bash
.venv-unsloth/bin/python train_dpo.py \
  --initial-adapter outputs/codenames-sft-1p5b \
  --data-dir data/dpo-clue-only --epochs 1 --learning-rate 1e-6 --beta 0.1 \
  --batch-size 1 --gradient-accumulation-steps 8 --save-steps 10 \
  --output-dir outputs/codenames-dpo-clue-only
```

The clue-only run completed **33 updates**, not the old 38, because the dataset
is smaller; this is not a matched-update comparison. On seed 170000, its reward
was 0.1352 versus SFT's 0.1590, with assassin hits 8.0% versus 7.5%.
Use the SFT-versus-DPO evaluation command above with
`--adapter-dir outputs/codenames-dpo-clue-only` and distinct output filenames.
Seed 170000 is development data now; confirm any chosen improvement on fresh
seeds and OOV boards. Do not accept an adapter based only on preference loss.

### SFT → short, game-only GRPO

Neither DPO variant demonstrated improvement over SFT on the seed-170000
development boards. The next experiment samples fresh clues online and rewards
their **actual turn outcomes**, rather than learning from a fixed preference set.
Keep the teammate fixed: MiniLM embeddings, minimum similarity **0.30**, abstention
mode **all**, random partial boards with up to nine revealed words.

The 50-update pilot below has now completed, including a GPU recovery test from
checkpoint 10. On 1,000 development boards, reward was **0.1332 versus SFT's
0.1590**, with assassin hits **7.8% versus 7.5%**. Paired reward change −0.0258,
bootstrap 95% interval [−0.0575, +0.0029]: **no demonstrated improvement**.
Keep SFT as the baseline; see the
[experiment report](grpo-game-only.md) for diagnostics and
changed-turn review. To reproduce, choose a new empty output directory.

```bash
.venv-unsloth/bin/python train_grpo_unsloth.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --initial-adapter outputs/codenames-sft-1p5b \
  --reference-policy sft --beta 0.04 --game-reward-only --fresh-boards \
  --vocabulary-file data/training_vocabulary.txt \
  --min-similarity 0.30 --abstention-mode all --max-revealed 9 \
  --num-boards 1000 --seed 200000 \
  --num-generations 4 --gradient-accumulation-steps 8 \
  --learning-rate 5e-7 --max-steps 50 --save-steps 10 --save-total-limit 6 \
  --output-dir outputs/codenames-sft-grpo-game-only-50
```

This is **50 optimizer updates**, not 50 boards or 50 epochs. There are four
sampled clues per prompt; many groups can have identical rewards and contribute
no relative reward signal. The score remains BLUE +1, RED −1, NEUTRAL −0.3,
ASSASSIN −5, invalid −2. Semantic bonuses and duplicate penalties are disabled.
Beta 0.04 is a starting setting, not an established optimum. Explicit warmup is
five updates. This command uses Unsloth without vLLM to keep memory use modest.

`--reference-policy sft` loads a **frozen second SFT adapter on the same base
weights**. The reference forward temporarily switches adapters, then restores the
policy's adapter and gradient flags. Standard TRL's adapter-disabled reference
would instead be the raw base model. The trainer verifies matching initial
adapter tensors, a frozen reference after training, and actual reference use in
`reference_audit.json`. Only the policy adapter is saved in checkpoints; the
reference is reconstructed from the pinned original SFT adapter on resume. This
also avoids Transformers skipping root policy weights in mixed-adapter saves.

`--fresh-boards` saves `boards.jsonl` and a hashed `manifest.json`, excluding
previous SFT, DPO, abstention and evaluation boards. Resume with the **same command
plus `--resume`**; changing pinned settings or the vocabulary/adapter/teammate
artifacts is rejected. Use a new output directory for a new experiment.

Evaluate with the same protocol as DPO:

```bash
.venv/bin/python evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --baseline-adapter-dir outputs/codenames-sft-1p5b \
  --adapter-dir outputs/codenames-sft-grpo-game-only-50 \
  --vocabulary-file data/training_vocabulary.txt \
  --min-similarity 0.30 --abstention-mode all \
  --num-boards 1000 --max-revealed 9 --seed 170000 \
  --output-csv results/evaluation-sft-grpo-game-only.csv \
  --transcript-file results/transcripts-sft-grpo-game-only.txt
```

CSV `base` means SFT, not the untrained base model. Check reward, invalid clues,
BLUE guesses and assassin hits together. Seed 170000 is **development data**;
any checkpoint selected on it needs confirmation on fresh boards before adoption.

### Original GRPO experiments

Run a one-step GPU check first, then the full experiment:

```bash
uv run python train_grpo.py --num-boards 8 --seed 123 --max-steps 1 \
  --save-steps 1 --output-dir outputs/smoke-model

uv run python train_grpo.py --num-boards 2000 --seed 10000 --max-steps 1000 \
  --save-steps 50 --learning-rate 5e-7 \
  --output-dir outputs/codenames-grpo-prompt-v3-lr5e-7
```

The adapter reported above was trained with the Unsloth path:

```bash
.venv-unsloth/bin/python train_grpo_unsloth.py --num-boards 2000 --seed 10000 \
  --max-steps 1000 --save-steps 50 --learning-rate 5e-7 \
  --fast-inference --no-flashinfer --output-dir outputs/codenames-unsloth-prompt-v3
```

- Add `--resume` to continue from the highest-numbered `checkpoint-*` directory.
  Resuming requires the same prompt, dataset seed and main settings.
- On CUDA out-of-memory, shrink the rollout group first: `--num-generations 4
  --max-prompt-length 384 --max-completion-length 32`.
- BF16 is selected automatically when supported. The embedding guesser runs on CPU
  to leave GPU memory for rollouts, and sentence embeddings are cached per board
  and clue.
- `--no-flashinfer` is applied automatically when `nvcc` is unavailable.

### Expanded vocabulary

`data/training_vocabulary.txt` holds 5,000 frequency-ranked English words from
[`wordfreq`](https://github.com/rspeer/wordfreq) (CC BY-SA 4.0), filtered for
function words and kept disjoint from the evaluation vocabulary. Both trainers
accept `--vocabulary-file data/training_vocabulary.txt`. Regenerate it with:

```bash
uv run --with wordfreq python scripts/build_training_vocabulary.py
```

## Evaluation

```bash
uv run python evaluate.py \
  --adapter-dir outputs/codenames-unsloth-prompt-v3 \
  --num-boards 1000 --seed 30000 \
  --output-csv results/evaluation-unsloth-1000.csv \
  --transcript-file results/transcripts-unsloth-1000.txt
```

- Out-of-vocabulary boards: add `--vocabulary-file data/evaluation_vocabulary_oov.txt`.
  These words are disjoint from training, although tokenizers still split them into
  familiar subwords.
- Frozen LLM teammate: add `--guesser llm --llm-guesser-device cuda`. The LLM sees
  only the clue and the 16 words, never the labels.

Each board produces two CSV rows (`model=base` and `model=trained`) with the
completion, parsed clue and number, ordered guesses and their labels, reward,
invalid reason, and the full board. Compare rows sharing a `board_index`.
With `--baseline-adapter-dir`, `base` means that adapter rather than raw Qwen.

Render random evaluated turns as GIFs (plus an answer-key PNG):

```bash
uv run python render_evaluation.py --evaluation-file results/evaluation-unsloth-1000.csv \
  --count 5 --model both --seed 7 --show-labels --output-dir results/evaluation-renders
```

## Attention and occlusion analysis

`visualize_attention.py` reloads an evaluated turn with eager attention and draws a
Codenames-style heatmap of how much attention each board word receives, the most
board-focused layer/head, and a JSON trace. `--occlusion` removes each board word in
turn and measures the change in the clue's log-probability, a more causal companion
to raw attention. It needs the regular Transformers environment, because
Unsloth/vLLM inference does not expose attention tensors.

```bash
uv run python visualize_attention.py \
  --evaluation-file results/evaluation-unsloth-oov-100.csv \
  --adapter-dir outputs/codenames-unsloth-prompt-v3 \
  --model trained --count 5 --seed 7 --occlusion --output-dir results/attention-oov
```

## Limitations

- Episodes are single turns, not complete multi-turn games.
- The historical 0.5B adapter used the 88-word default vocabulary. Current
  SFT uses the 5,476-word curriculum pool; later DPO/GRPO use 5,000 words.
- The embedding guesser approximates human word association and is not trained
  jointly with the spymaster. Self-play between two language models would be more
  realistic but slower and harder to stabilize.
- Changing the prompt or clue rules changes the task: retrain before comparing
  adapters across prompt versions.
- The code targets the TRL 0.14 GRPO interface and filters configuration
  arguments against the installed version.

## License

MIT. See [LICENSE](../LICENSE). Vocabulary attribution is described in the main README.
