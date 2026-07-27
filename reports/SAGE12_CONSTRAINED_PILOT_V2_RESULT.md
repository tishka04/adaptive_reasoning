# SAGE12 constrained effect pilot V2 result

Date: 2026-07-27.

Outcome: **FAIL_CLOSED**. Constrained typed output and the reduced
game-signature controls passed, but the one-bit semantic representation plus
frozen Qwen encoder did not improve effect prediction over action identity.
No semantic world model or EBM was fit.

Primary result checksum:
`7440cbf5a15edd4ca2c7c70fbebdcb2ced1bdf88817bdf1f7c0f417a6db81e3a`.

## What V2 changed

V1 failed before semantic comparison because none of its 224 free-form Qwen
outputs matched the typed schema and its compact scene graphs almost perfectly
identified the source game.

V2 deliberately separated those problems:

- Qwen2.5 0.5B is a frozen encoder rather than a free-form JSON generator;
- independent class-balanced logistic heads predict four observed effect
  components;
- code renders positive decisions into strict typed hypotheses with
  `support=0`;
- the compiler checks the exact selected legal action and grounds the
  `player` role;
- the complete state representation is one binary `actor_interaction` bit:
  whether a player-role entity is near, touching, or adjacent to a non-player
  entity;
- counts, identifiers, shape, direction, relation inventory, scene signature,
  available-action set, coordinates, colors, game ID, and outcome are absent
  from the prompt.

The selected action remains visible because the scientific question is
whether the state motif adds information beyond action identity.

## Frozen order and amendment

The code, tests, two source-training preflights, protocol, and manifest were
published before V2 validation in commit `4c07670`. The first execution
completed GPU embeddings and then stopped before metrics because scikit-learn
rejected `int64` indices on a tiny sparse baseline matrix. No result or
prediction artifact was written.

Only that baseline container was changed to a dense array. Inputs, Qwen
embeddings, heads, labels, thresholds, baselines, gates, and splits remained
frozen. The amendment was published in commit `7e6cd6d`, after which the clean
evaluation restarted from zero.

The prior V1 validation result was necessarily known during V2 design. No V2
validation metric was computed before the V2 freeze. Representation reduction
used only the 1,624 source-training rows.

## Data and execution

V2 reused the exact 2,104-row source corpus:

- 1,624 rows across the 11 source-training games;
- 480 rows, 160 each, from `ls20`, `re86`, and `sc25`;
- collection-manifest checksum
  `69182fef9d397768aace54a301dc5046bd801589524ae14b7d6e8ad728ad0e05`;
- combined shard checksum
  `ce5cfe1217f9add9ab250f60315ed66d154ae8ed903e51bb572b69a4b3`;
- holdout, historical, and `ar25` remained closed.

The RTX 4050 encoded 2,584 prompts: source training, source validation, and
relation-shuffled source validation. Only 12 unique prompts exist because the
input is six action names crossed with one binary motif.

| Pass | Rows | GPU model time | Maximum tokens |
|---|---:|---:|---:|
| Source train | 1,624 | 6.855 s | 80 |
| Validation original | 480 | 2.536 s | 80 |
| Validation relation shuffle | 480 | 3.452 s | 80 |

These times cover model forward passes, not model loading, tokenization, or
linear-head fitting.

## Gate result

| Gate | Frozen requirement | Result | Pass |
|---|---:|---:|:---:|
| Strict JSON validity | 1.00 | 1.00 | yes |
| `support=0` | 1.00 | 1.00 | yes |
| Grounded emitted hypotheses | >= 0.99 | 1.00 | yes |
| Primary macro-F1 gain | >= +0.05 | -0.0645 | no |
| Relation-shuffle degradation | >= 0.05 | -0.0977 | no |
| Every validation game non-negative | required | 2/3 | no |
| Motif identity gain over majority | <= +0.10 | +0.0979 | yes |
| Motif identity gain beyond action | <= +0.10 | +0.0899 | yes |

Five gates passed and three failed. The output and leakage repairs therefore
worked as designed, but the representation did not carry a transferable
causal signal.

## Predictive comparison

The primary metric is macro-F1 over `changed` and `player_moved`.
`level_complete` and `game_over` are reported separately because source
training contains only one and 27 positive rows respectively.

| Method | Primary macro-F1 | Primary macro-recall |
|---|---:|---:|
| Action-only logistic | 0.5487 | 0.5510 |
| Direct action + motif logistic | 0.4254 | 0.3640 |
| Frozen Qwen + linear heads | 0.4842 | 0.4302 |
| Relation-shuffled Qwen | 0.5819 | 0.5353 |
| Deterministic template | 0.0000 | 0.0000 |

Qwen improved over the direct additive motif classifier but remained 0.0645
below action identity. Relation shuffling improved macro-F1 by 0.0977 instead
of degrading it.

Per label:

| Effect | Validation positives | Qwen F1 | Action-only F1 |
|---|---:|---:|---:|
| `changed` | 476 | 0.7239 | 0.6695 |
| `player_moved` | 174 | 0.2446 | 0.4280 |
| `level_complete` | 0 | 0.0000 | 0.0000 |
| `game_over` | 5 | 0.0288 | 0.0230 |

Qwen helped the nearly constant `changed` label but lost most of that gain on
the meaningful movement component. The terminal rows are too rare for a
promotion claim.

## Per-game transfer

| Game | Qwen macro-F1 | Stronger baseline | Gain |
|---|---:|---:|---:|
| `ls20` | 0.4901 | 0.3949 action-only | +0.0952 |
| `re86` | 0.4236 | 0.6607 action-only | -0.2371 |
| `sc25` | 0.4068 | 0.3794 action-only | +0.0274 |

The aggregate no-go is not a small uniform miss: most of the negative transfer
comes from `re86`.

## Post-hoc signal diagnosis

The one bit learned the following source-training association:

- without actor interaction: player movement rate 9.75%;
- with actor interaction: player movement rate 29.49%.

That association does not transfer:

- `re86`: every row moves the player, whether the bit is zero or one; Qwen
  predicts no movement for all 107 zero-bit rows;
- `sc25`: movement is 14.81% when the bit is zero and 0% when it is one,
  reversing the source-training direction;
- `ls20`: every row has the bit set, movement is only 6.25%, yet Qwen predicts
  movement on 75% of rows.

The relation permutation flips the one-bit motif on 34.38% of `ls20`, 56.88%
of `re86`, and 47.50% of `sc25` rows. Because the original bit-to-effect
association is miscalibrated across games, this corruption sometimes moves
predictions closer to the target by accident. It is not evidence that shuffled
relations are causally better.

Post-hoc diagnostic checksum:
`a5effa1c401e2ecc65f3d5d972ec715b087c329a66578813955786ea49ebb019`.

## Decision

V2 rejects the single-bit actor-interaction representation as a sufficient
semantic state for cross-game effect hypotheses. It also shows that merely
placing a frozen LLM embedding between the motif and a linear head does not
create missing causal information.

`authorized_next_stage` is `none` and `world_model_fit_started` is `false`.
The semantic world model, EBM, shadow controller, bounded controller, holdout,
historical games, and `ar25` remain untouched.

The next representation should be action-target grounded rather than globally
scene descriptive. It needs stable observed event labels such as actor
displacement, target creation/removal/motion, and terminal transition, plus
relations anchored to the exact selected target or requested movement
direction. That requires a new source-only collection format and a new frozen
pilot; it must not be retrofitted onto this result.

Predictions checksum:
`76c30cadf8ab1a0fa96eaed2944e08bc19e20dd6f4101eb945e5d784ecd7ec95`.

## Software and artifact validation

- targeted Ruff validation passed;
- 31 focused SAGE12 tests passed;
- the full repository suite passed 1,698 tests with one non-failing Joblib
  physical-core-detection warning;
- all 480 prediction rows were re-read, with exactly 160 rows for each
  validation game;
- the prediction file hash and both canonical result checksums were
  independently recomputed;
- both result artifacts confirm `world_model_fit_started=false`.

## Reproduction

```powershell
python -m theory.sage12.constrained_pilot preflight
python -m theory.sage12.constrained_pilot evaluate
python -m theory.sage12.constrained_pilot diagnose
```
