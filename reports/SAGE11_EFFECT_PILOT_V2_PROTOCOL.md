# SAGE.11 effect pilot v2 pre-registration

Status: frozen before empirical execution

Protocol date: 2026-07-26

Format: `sage11-factorized-effect-pilot-v2`

## Purpose

Pilot v1 validly rejected the coarse typed-atom to joint-effect-class pairing.
Its result and checksum
`c724aeb6d2ab71154a7c72fa381f3f5f4347a5135644ba64ac82a5542e528136`
remain immutable.

Pilot v2 asks a narrower question:

> Do leakage-free pre-action trajectory relations improve factorized
> cross-game effect prediction by at least 0.10 macro-F1 beyond a learned
> action-only classifier?

The answer is evaluated once on the same three frozen source-validation games.
There is no tuning on source validation, historical games, or
`NEURO_HOLDOUT_V1`.

## Archived-data limitation

The published 100,000-row corpus contains typed atoms, action arguments,
effect atoms, exact-state digests, and sequence identifiers. It does **not**
contain raw grids, object coordinates, contact graphs, alignment relations, or
proximity relations. Exact frames cannot be reconstructed reliably because
deduplication intentionally omitted some intermediate transitions.

Therefore v2 is a **trajectory-relational and factorized-target pilot**, not a
test of the proposed object-relation vocabulary. A no-go means richer raw
features must be collected before another pilot. A go permits work on a model
that consumes the same v2 interface; it does not authorize training the
unmodified graph model on the old 19 atoms.

## Frozen data

- Manifest:
  `d4fd8210f2015c00b906cdd98e01630b309deefa7cd9498b38aba8e55130fa1b`.
- Training: all 76,908 `source_train` rows from the 11 registered games.
- Validation: all 23,092 `source_validation` rows from exactly `re86`,
  `ls20`, and `sc25`.
- Historical, holdout, and regression-only games remain unread.
- Atom vocabulary is fitted on source-training rows only.
- Factor vocabularies are fixed from the atom schema, not learned from
  validation.

## Frozen inputs

### Learned action-only baseline

The stricter primary baseline is a classifier, not only a majority lookup. It
receives:

- one-hot current action identity (`ACTION1` through `ACTION6`);
- `has_xy`;
- target is on the 64-by-64 outer boundary;
- target is on a corner;
- target is on the main or anti-diagonal.

Raw `x/64` and `y/64`, coordinate buckets, game identity, policy arm, and
state digests are excluded. The three coordinate predicates are
grid-topological rather than identifiers of source-game locations.

The train-only per-action majority remains a reported secondary baseline for
continuity with v1.

### Full pre-action representation

The full classifier receives the action-only block plus:

- train-fitted binary presence of the current pre-action typed atoms;
- reset-position bucket: step 0, steps 1–3, 4–15, or 16+;
- accepted-state visit bucket within the current game/seed/reset:
  first, second, third/fourth, or fifth+;
- accepted-state recency bucket: new, one step, 2–4, 5–16, or 17+;
- exact-continuity flag, true only when the immediately preceding archived
  row is step `t-1` and its after-state digest equals the current before-state
  digest;
- previous action one-hot and same-action flag when continuity is true;
- previous observed changed-cells bucket and player-moved flag;
- previous observed level-complete and game-over flags;
- current-versus-previous target relations when both actions have coordinates:
  same target, same row, same column, signed x/y direction, and Manhattan
  distance bucket (zero, 1–4, 5–16, or 17+);
- current atom-set difference from the contiguous previous after-state,
  bucketed as zero, one, few, some, or many.

State digests are used only for equality, visit counts, and recency inside one
reset. Their bytes, hashes, prefixes, or fitted identities never enter the
matrix. Previous outcomes enter only on an exact contiguous predecessor, so
post-action information from the row being predicted cannot leak.

## Frozen factorized targets

Two core heads determine the go/no-go result:

1. changed-cells bucket: `zero`, `one`, `few`, `some`, `many`;
2. player-moved: `False`, `True`.

Level-complete and game-over are separate audit heads and cannot affect the
gate because the dataset contains only 44 strong terminal/level events,
below the existing 100-event threshold. Value-multiset cardinality is not a
separate head because the current extractor derives it from the same changed
cell set; double-counting it would overweight one observation.

Each head is scored independently with validation macro-F1, so correct
components receive credit even when another component is wrong.

## Frozen estimators

Every learned head uses one
`sklearn.ensemble.HistGradientBoostingClassifier` with:

- learning rate 0.08;
- maximum depth 4;
- 100 iterations;
- early stopping disabled;
- balanced class weights;
- random state 11.

For each core target, one model uses only the action block and one uses the
full block. There is no hyperparameter search, seed search, feature selection,
threshold tuning, or outcome-driven rerun.

## Metrics and decision gate

For a set of rows, the factorized composite is the unweighted mean of
changed-cells macro-F1 and player-moved macro-F1.

Pilot v2 is a **go** only if all conditions pass:

1. the overall full composite exceeds the learned action-only composite by at
   least 0.10;
2. neither core head has a negative overall full-minus-action-only delta;
3. the full-minus-action-only composite delta is non-negative on each of
   `re86`, `ls20`, and `sc25`.

Any failure is a no-go and blocks model training. No exception may be made
from strong absolute F1, a favorable single game, or the secondary majority
baseline.

## Pre-registered controls

Two controls are diagnostics and do not alter the gate:

- shuffle the complete current-action block within each validation game
  without retraining;
- shuffle only direct argument predicates within each game/action stratum
  without retraining.

Both use random state 11. These separate current-action sensitivity from
argument sensitivity while preserving observed within-game marginals more
closely than v1's six-scalar shuffle.

## Hardware rule

The laptop GPU may be used only if it accelerates the frozen estimator without
changing the method. Scikit-learn's histogram gradient booster has no CUDA
backend, and the matrix is small, so CPU is expected to be the effective
device. Hardware availability, the device decision, and timings must be
published.

## Required artifacts

The single empirical execution must publish:

- `diagnostics/sage/sage11_factorized_effect_pilot_v2.json`;
- `reports/SAGE11_EFFECT_PILOT_V2_RESULT.md`;
- aggregate and per-game metrics for both core heads and the composite;
- audit-head metrics and label support;
- continuity/recurrence coverage;
- control results, versions, hardware, timings, and an artifact checksum;
- the exact reproduction command and final go/no-go decision.
