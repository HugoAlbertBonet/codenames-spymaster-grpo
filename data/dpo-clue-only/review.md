# Clue-only preference review

Reviewed 2026-10-10. The mechanical audit replayed all 318 preferences using the
fixed CPU MiniLM teammate, checked both completions occurred in the original
candidate cache, and enforced equal NUMBER/different clue words. All rewards
and saved similarities match replay. No training/validation board overlap or
confirmation-board overlap was found. Original source artifacts and checkpoints
were preserved; source hashes are recorded in `manifest.json`.

## Qualitative spot check

Inspected 24 unique pairs: 12 random pairs using seed 193, the eight winners
with lowest minimum guess cosine, and all six NUMBER=3/4 pairs, deduplicated.
This is an AI-assisted spot check, not a human semantic audit of every pair.

| Seed | Same-NUMBER preference | Assessment |
| --- | --- | --- |
| 130438 | aviation / 1 over automobile / 1 | Both reach BLUE engines; automobile additionally hits RED legally. Clear teammate-safety contrast, not a claim that aviation has the strongest first association. |
| 140544 | rhythm / 1 over dance / 1 | Both reach BLUE dancing; dance adds RED daddy. A plausible safe clue choice. |
| 130431 | jewelry / 1 over gemstone / 1 | Jewelry reaches BLUE necklace and stolen, while gemstone reaches necklace only. The stolen association is weaker (0.30879). |
| 130158 | tide / 2 over maritime / 2 | Tide reaches BLUE oyster, seam and fog; maritime adds RED ltd after two BLUEs. Seam/fog are debatable human links. |
| 130021 | religion / 1 over equine / 1 | Religion reaches BLUE allah then NEUTRAL africa; equine hits RED cat. The winner is not a flawless turn. |
| 140591 | geography / 2 over war / 2 | Geography gets BLUE syria and usage. Usage barely passes the cutoff (0.30024), so this remains proxy-dependent. |
| 130069 | smartphone / 4 over smart / 4 | Smartphone reaches only BLUE iphone and abstains. It does not substantiate four intended targets. |
| 130342 | school / 3 over study / 3 | School reaches BLUE classroom then NEUTRAL university; study hits university immediately. Better reward, but no evidence of a three-target clue. |
| 130389 | legal / 3 over finance / 3 | Legal reaches only BLUE jury; finance hits RED opportunities. Another overdeclared-count example. |
| 130423 | transportation / 3 over travel / 3 | Both reach BLUE carriage, airlines and affairs; travel additionally hits RED singing. The affairs association is questionable. |

Inspected seeds: 140546, 130438, 130158, 130285, 140544, 130257, 140511,
130204, 130021, 130298, 130140, 130431, 140591, 130286, 130310, 130241,
140541, 130385, 130069, 130157, 130309, 130342, 130389, 130423.

## Decision

Preserve these outcome labels without retrospective human-style relabeling or
new similarity thresholds. Use this dataset only for a conservative same-number
diagnostic experiment. It removes direct winner/loser number differences but
does not solve the weak-association proxy problem, number imbalance, or limited
data volume. There are no NUMBER=3/4 validation pairs, so that split cannot
establish high-count preference generalization.

Do not promote a resulting adapter based on DPO loss alone. Compare SFT versus
the new policy on gameplay reward, BLUE guesses, invalid clues, assassin hits
and declared-number distributions. Treat seed 170000 as development data after
the prior checkpoint comparisons; confirm a selected improvement on untouched
boards and OOV vocabulary before accepting it.

## Completed diagnostic

The subsequent 33-update run did not demonstrate improvement: on the 1,000
seed-170000 development boards, reward was 0.1352 versus SFT's 0.1590, and
assassin hits were 8.0% versus 7.5%. Keep SFT as the baseline. See the
[current results](../../README.md#what-happened-after-sft) for the comparison
with gameplay DPO and the later game-only GRPO pilot.
