# M1.3c grounding autopsy

This diagnostic asks why dc22 succeeds where bp35/cd82 do not.

## Summary

| game | new pairs | actionable-source new pairs | entering agenda | relation candidates | env actions | error |
|---|---:|---:|---:|---:|---:|---|
| bp35-0a0ad940 | 5 | 0 | 0 | 0 | 0 | not_enough_relation_candidates_for_agenda |
| cd82-fb555c5d | 4 | 0 | 0 | 0 | 0 | not_enough_relation_candidates_for_agenda |
| dc22-4c9bff3e | 6 | 3 | 1 | 4 | 1 |  |

## Grounding funnel

| game | discovered | target present | actionable source | blocked source | live-compatible | entering agenda | env-action pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| bp35-0a0ad940 | 20 | 20 | 0 | 20 | 0 | 0 | 0 |
| cd82-fb555c5d | 20 | 20 | 0 | 20 | 0 | 0 | 0 |
| dc22-4c9bff3e | 20 | 17 | 5 | 15 | 2 | 2 | 1 |

## New-pair grounding funnel

| game | discovered | target present | actionable source | blocked source | live-compatible | entering agenda | env-action pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| bp35-0a0ad940 | 5 | 5 | 0 | 5 | 0 | 0 | 0 |
| cd82-fb555c5d | 4 | 4 | 0 | 4 | 0 | 0 | 0 |
| dc22-4c9bff3e | 6 | 4 | 3 | 3 | 1 | 1 | 0 |

## dc22 live-compatible new pairs

| action | pair | support | predicates | live predicates | entering agenda |
|---|---|---:|---|---|---|
| ACTION6 | 8->2 | 291 | aligned_with, adjacent_to, paired_with | aligned_with, adjacent_to, paired_with | true |
| ACTION6 | 9->12 | 100 | same_shape, aligned_with, adjacent_to, paired_with |  | false |
| ACTION6 | 8->15 | 713 | same_shape, aligned_with, adjacent_to, paired_with |  | false |

## Interpretation

- dc22 is the positive control: at least one new M1 pair is live-compatible and enters A25.
- bp35/cd82 produce new pairs, but their new sources are not selectable from the reset grid.
- The bottleneck is now hypothesis grounding: trace anchor -> live object -> actionable source.
- This diagnostic remains analysis-only and does not count trace support as proof.
- Positive games: dc22-4c9bff3e.
- Blocked-by-source games: bp35-0a0ad940, cd82-fb555c5d.
