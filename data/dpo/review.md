# Preference review before DPO

Review date: 2026-10-10. These are outcome preferences for the fixed MiniLM
teammate with cosine cutoff 0.30 before every guess, not human judgments.

## Full mechanical audit

All 360 pairs were replayed against the frozen teammate and checked against
their actual sampled completions. There are 298 training and 62 validation
pairs, with no duplicate labeled boards, cross-split board overlap, overlap
with confirmation boards, or reserved OOV vocabulary in the pair boards.
Every winner is legal, positive-reward, avoids RED/ASSASSIN, and declares no
more targets than remaining BLUE words. Every loser is legal and scores at
least one point less. The audit covers the exact file hashes in `audit.json`;
training refuses changed data or different initial SFT weights.

The dataset has 262 distinct winning clues. Winning NUMBER frequencies are
1: 255, 2: 96, 3: 6, 4: 3; actual BLUE guesses are 1: 145, 2: 175, 3: 40.
247 losers hit RED and 41 hit ASSASSIN. 41 pairs keep the same clue but prefer
a different NUMBER. 43 winners obtain fewer BLUE guesses than their declared
count; a feasible count is not an independently verified intended-target count.
Winners stop at the guess limit in 188 cases, at the cosine threshold in 139,
and on NEUTRAL in 33. Revealed-state counts span 0 through 9.

## Qualitative spot check

Inspected 24 pairs: 12 randomly selected with seed 193, plus 12 with the lowest
minimum cosine among their winning guesses. This is an AI-assisted spot check,
not a human semantic review of all 360 pairs. Examples inspected include:

| Board seed | Preference | Interpretation |
| --- | --- | --- |
| 130481 | credit / 1 over credit / 2 | Both reach BLUE account and card; the larger budget adds RED oscar. A clear safety/budget preference. |
| 130478 | illumination / 1 over illumination / 2 | Both reach BLUE lighting and worn; the larger budget adds ASSASSIN basically. Strong safety evidence, but worn is a weak association. |
| 130254 | electricity / 2 over power / 1 | Electricity reaches BLUE energy, socket, band. The first two links are clear; band is a questionable bonus association. |
| 130021 | religion / 1 over equine / 1 | Religion reaches BLUE allah and then NEUTRAL africa; equine hits RED cat. Positive does not mean a flawless clue. |
| 130389 | legal / 3 over securities / 2 | Legal reaches only BLUE jury before abstaining. This does not prove three meaningful intended targets. |
| 140591 | geography / 2 over war / 2 | Geography reaches BLUE syria and usage; usage is barely above the cutoff (0.30024) and not convincing human evidence. |
| 130014 | theater / 1 over performance / 1 | Theater reaches BLUE representative at 0.30174; performance hits RED immediately. A weak proxy-dependent winner. |

## Decision

Keep the audited outcome labels unchanged for a conservative first experiment:
one epoch, learning rate 1e-6, beta 0.1, initialized from SFT, with that exact
SFT policy as the frozen DPO reference. Do not present these examples as
universally good Codenames clues or a balanced curriculum for multi-target clues.

Training can learn the fixed embedding teammate's quirks and bonus-guess budget,
not necessarily human semantic strategy. Validation DPO loss measures preference
fit, not gameplay reward. The decision to retain DPO must come from fresh-board
SFT-versus-DPO comparisons under the same cutoff, reporting reward, BLUE guesses,
invalid clues and assassin hits, followed by OOV and alternative-teammate tests.
