# SAGE12 V4.11 — Counterfactual teacher QA

Status: **CAPACITY FAILED BEFORE MODEL FITTING**.

Teacher QA checksum:
`4fcfca50eb6699f8cbbad6e11bf42a2770ba4e715c833458cae128a1d6f3da63`.

Teacher-panel SHA-256:
`0642073627de0f61a52c3adc55f984210e9304af56bf1bd1606e69ea2b74adf8`.

## Compiled corpus

The deterministic teacher compiled:

- 1,056 replay-verified panels;
- 3,914 immediate arms;
- 5,529 within-panel comparisons;
- two deterministic continuations per arm, to horizon three;
- 100% action-aligned, compass-free student graphs.

All collection, source-firewall, graph, and per-game panel-count checks passed.
Source validation, holdout, external historical data, and live environments
remained closed.

## Progress capacity

The frozen requirement was at least 20 progress-discordant panels in at least
eight games. Only seven games met it.

| Game | Progress-discordant panels | Meets 20 |
|---|---:|:---:|
| `bp35` | 96 | yes |
| `cd82` | 4 | no |
| `dc22` | 3 | no |
| `g50t` | 95 | yes |
| `ka59` | 77 | yes |
| `lf52` | 51 | yes |
| `lp85` | 3 | no |
| `sp80` | 49 | yes |
| `su15` | 42 | yes |
| `tr87` | 7 | no |
| `tu93` | 90 | yes |

There are 517 progress-discordant panels in total, but they remain concentrated
in seven games. Training a LOGO comparator would therefore test transfer from
an insufficient number of distinct mechanics for four of the outer folds.

## Immediate-effect capacity

Eight effects independently met the frozen eligibility rule of at least 100
discordant comparisons across four games:

- `changed`;
- `moved`;
- `target_removed`;
- `target_moved`;
- `local_change`;
- `contact_lost`;
- `productive`;
- `risk`.

This is useful future training material, but it cannot override the failed
progress-capacity gate. The iteration's primary claim concerns comparative
horizon progress, not only immediate effect deltas.

Completion also remained below its separate capacity rule: eight positive
arms in two games, versus the required 20 positives in four games. Completion
recall is therefore undefined rather than failed.

## Decision

`teacher_ready=false`.

No comparator, root-only control, distillation model, identity probe, world
model, EBM, or controller was fitted. This is a pre-model capacity failure,
not a negative learned-model result.

The result says that the current hand-compiled one-step score plus two
deterministic horizon-three continuations does not generate sufficiently
diverse cross-game preferences. It does not refute comparative semantics or
the global SAGE12 architecture.
