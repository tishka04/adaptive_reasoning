# SAGE.11 small relational pilot pre-registration

Status: frozen before collection and empirical fitting

Protocol date: 2026-07-26

Collection format: `sage11-relational-pilot-collection-v1`

Transition format: `sage11-relational-transition-v1`

Pilot format: `sage11-relational-effect-logo-v1`

## Purpose

The source-train anti-shortcut audit rejected the shared 77-feature
representation for world-model training. Changed-cells full-minus-best-
baseline was -0.1026, conditional action-shuffle degradation was 0.0180, and
fixed availability/object-role signatures predicted game identity with
99.17% accuracy.

This replacement asks whether generic object geometry supplies the missing
action/state interaction. It is intentionally a small recollection, not a
second 100,000-row corpus.

## Frozen collection

Collect only the 11 registered `source_train` games with the existing
outcome-independent 70/20/10 active-controller, uniform-legal, and
frontier-stall mixture. Use seeds 0–4 and exact transition-signature
deduplication.

The frozen quotas are 1,000 transitions for each source-training game except
`lp85`, whose previously verified finite unique capacity is 27. The total is
therefore exactly 10,027:

| Game | Rows |
| --- | ---: |
| `bp35` | 1,000 |
| `cd82` | 1,000 |
| `dc22` | 1,000 |
| `g50t` | 1,000 |
| `ka59` | 1,000 |
| `lf52` | 1,000 |
| `lp85` | 27 |
| `sp80` | 1,000 |
| `su15` | 1,000 |
| `tr87` | 1,000 |
| `tu93` | 1,000 |

Do not open or collect source-validation, historical, holdout, or
regression-only games. A v2 base transition remains nested for labels, exact
deduplication, action-argument auditability, and streaming context. Raw grids
are not archived. Raw action coordinates remain in the nested audit record but
are never exposed as model columns: the streaming encoder derives only
categorical/topological predicates and the relational encoder derives only
object-relative predicates.

## Frozen relational schema

`sage11-object-relations-v1` has 52 binary columns, checksum
`84f044dd08f3240f968a6ba1bf528896eab00eb39066dcce95d0f87e9a9193f7`.
The same pure encoder is used at collection time and by future live
counterfactual inference.

The 22 state columns preserve:

- non-player object-count bucket;
- detected-player presence;
- any object-object contact, row alignment, and column alignment;
- minimum object-object proximity bucket;
- any player-object contact, row alignment, and column alignment;
- minimum player-object proximity bucket.

The 30 current-action-dependent columns preserve:

- whether the action has an `(x, y)` target;
- target inside/contact/row alignment/column alignment with objects;
- target-to-nearest-object proximity;
- target direction relative to the nearest object center;
- nearest-object size and aspect;
- target-to-player proximity and row/column alignment.

No object color/value, object identifier, game identity, state digest, raw
grid coordinate, outcome, or policy-arm feature is included. `(x, y)` is used
transiently to compute relations but is not itself archived in the relational
vector.

## Frozen model views

Reconstruct the leakage-free streaming context from each nested base
transition, then remove every `action:available(...)` and
`object:role_present(...)` column before fitting.

Evaluate four learned views:

1. **Action-only:** the 10 categorical/topological current-action columns.
2. **State-only:** all candidate-independent streaming columns after fixed-
   signature removal plus the 22 relational state columns.
3. **Full without relations:** every retained streaming column, with no
   relational columns.
4. **Full:** every retained streaming column plus all 52 relational columns.

Targets remain separate changed-cells and player-moved heads. The composite
is their unweighted macro-F1 mean; player movement cannot compensate for a
failed changed-cells condition.

## Frozen validation and estimator

Use leave-one-game-out evaluation across all 11 source-training games. Every
view, target, and fold uses one balanced
`HistGradientBoostingClassifier`:

- learning rate 0.08;
- maximum depth 4;
- 100 iterations;
- early stopping disabled;
- random state 11.

There is no hyperparameter, threshold, feature, seed, or row-count search.
The dataset manifest and all selected shard hashes must verify before fitting.

## Frozen conditional action shuffle

Within each held-out game, group rows by the exact 22-bit relational state
signature. Within each group, permute all current-action-dependent streaming
and object-relative columns together, do not retrain, and use random state
`11 + fold_index`.

This preserves the coarse relational state while breaking the current
action/state relation. Report the full-minus-shuffled composite on
concatenated out-of-fold predictions.

Also report row-weighted majority-game accuracy of the exact 22-bit
relational state signature. This is diagnostic; the state-only comparator and
conditional shuffle are the protections against state-regime shortcuts.

## Frozen go/no-go gate

The pilot passes only if all conditions hold:

1. changed-cells full macro-F1 exceeds the stronger action-only/state-only
   baseline by at least 0.10;
2. composite macro-F1 degrades by at least 0.10 under conditional action
   shuffling;
3. changed-cells full macro-F1 exceeds full-without-relations by at least
   0.05, proving the recollected relations add value;
4. changed-cells full-minus-best-baseline is non-negative on at least 9/11
   held-out games and no fold is below -0.05.

No favorable absolute score, player-moved result, subset of games, or
post-hoc analysis can waive a condition.

## Consequences

- **Pass:** extend the shared live/data interface with the frozen relational
  vector, update the compact factorized PyTorch model, and train on the RTX
  4050. The trained model must still pass every existing source-validation
  gate before shadow mode.
- **Fail:** publish the negative result and stop the current world-model
  track. Do not recollect more rows or touch source-validation, historical, or
  holdout games without a new explicit plan.

## Reproduction

After this protocol and its implementation are committed and pushed:

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.relational_pilot_collection --workers 8
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.relational_effect_pilot
```

Required artifacts:

- `training/sage11/relational_pilot_v1/manifest.json`;
- `training/sage11/relational_pilot_v1/collection_report.json`;
- `diagnostics/sage/sage11_relational_effect_pilot.json`.
