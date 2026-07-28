# SAGE12 V4.10 — Action-aligned invariant semantics result

Verdict: **ACTION_ALIGNED_INVARIANT_SEMANTICS_NOT_YET_SUPPORTED**.

This is an exploratory, source-only result. V4.10 improves average semantic
calibration and productive-action ranking over V4.9, but it does not establish
that the action-aligned neighbor relations transfer across games. It grants no
live authority and its probabilities must not yet train the semantic world
model or EBM.

## What was built and evaluated

V4.10 augmented the valid V4.9 teacher corpus with 1,587 fresh, unique,
executed source transitions targeted at rare functional interventions. The
resulting teacher corpus contains 9,128 transitions, 2,396 same-prestate
pairs, and six genuine completion events.

The student sees only the legal action and a pre-action graph expressed in the
action's frame:

- `ahead`, `behind`, `lateral_left`, `lateral_right`, `overlap`, and `radial`
  replace compass directions;
- contact topology is explicit;
- neighbor order is irrelevant;
- coordinates, compass relations, colors/raw values, object IDs, game IDs,
  frames, and future fields are excluded.

Training balances every batch across games and adds identity confusion,
latent/output distribution alignment, same-prestate productive ranking, and a
game-balanced prevalence calibration rule. Every reported probability is
strict leave-one-source-game-out. The run trained 22 models (full
action-aligned graph and root-only for each of 11 held-out games) on the laptop
RTX 4050 using `cuda:0`. It completed in 1,652.8 seconds wall time.

Source validation, holdout, historical, and live environments remained closed.

## Frozen decisions

Lower macro-Brier is better. A positive gain is the baseline Brier minus the
full-model Brier.

| Check | Frozen threshold | Result | Pass |
|---|---:|---:|:---:|
| Teacher ready | all QA checks | all pass | yes |
| Macro-Brier gain over action-only | `> 0` | +0.000416 | yes |
| Macro-Brier gain over root-only | `> 0` | −0.005207 | no |
| Macro-Brier gain over V4.9 | `> 0` | +0.010287 | yes |
| Productive-pair accuracy gain over root-only | `> 0` | −0.005690 | no |
| Relation shuffle Brier degradation | `> 0` | −0.000675 | no |
| Neighbor permutation probability delta | `≤ 1e-6` | `2.07e-7` | yes |
| Non-negative games versus action-only | `≥ 6/11` | 4/11 | no |
| Semantic-output game identity | `≤ 0.60` | 0.8551 | no |
| Identity reduction from V4.9 | `≥ 0.15` | 0.0513 | no |
| Completion recall@8 | `≥ 0.20` | 0/6 = 0.00 | no |

Four of eleven frozen checks passed. The global gate therefore fails.

## Direct semantic prediction

| Model/control | Macro-Brier ↓ | Base-seven ↓ | Functional-ten ↓ |
|---|---:|---:|---:|
| root-only | **0.072988** | **0.077882** | **0.069562** |
| relation shuffle | 0.077520 | 0.080008 | 0.075778 |
| action-aligned graph | 0.078195 | 0.081064 | 0.076187 |
| action-only | 0.078611 | 0.081772 | 0.076399 |
| V4.9 object-relative graph | 0.088482 | — | — |

The full V4.10 model is materially better than the V4.9 predictor and narrowly
better than action-only on the corpus-wide mean. This is real progress, but it
is not evidence that the neighbor relations are the cause:

- root-only is clearly better than the full graph;
- shuffling relations also improves rather than worsens Brier;
- the full model beats action-only in only four games (`dc22`, `lf52`, `tr87`,
  and marginally `tu93`).

The strongest full-model improvements over action-only are `moved` (+0.01319
Brier gain), `local_change` (+0.01304), and `path_opened` (+0.00553). Its
largest regressions are `productive` (−0.01277), `target_removed` (−0.01244),
and `contact_lost` (−0.00795). The new representation helps local physical
change but still miscalibrates several decision-critical effects.

## Productive ranking

On 703 same-prestate pairs with different teacher productive scores:

| Model/control | Accuracy |
|---|---:|
| root-only | **0.7738** |
| action-aligned graph | 0.7681 |
| relation shuffle | 0.4296 |
| action-only | 0.3713 |

This is the clearest positive result. The full model gains +0.3969 accuracy
over action-only and V4.10 raises full-model ranking from V4.9's 0.4296 to
0.7681. However, root-only remains slightly better. The relational graph is
highly relevant to pair ordering—the shuffle destroys much of the gain—but has
not shown incremental value over the simpler pre-action root representation.

## Identity and terminal semantics

A classifier reading only the 17 LOGO semantic probabilities recovers game
identity at **85.51%** accuracy, versus 90.64% in V4.9 and a 10.05% majority
baseline. The invariance objectives reduce leakage by 5.13 percentage points,
well short of the frozen 15-point reduction and 60% absolute ceiling.

Completion recall@8 remains **0/6**:

- `lp85`: 0/5;
- `sp80`: 0/1.

The corpus gained only one completion transition. Six positives across two
games remain insufficient to establish cross-game terminal retrieval.

## Interpretation

V4.10 validates several implementation hypotheses:

- the teacher can be expanded with fresh source-only functional
  interventions;
- action-aligned graphs are exactly permutation invariant;
- balancing and alignment substantially improve average calibration and
  same-prestate productive ranking;
- useful pre-action semantic signal exists beyond action identity.

It rejects the stronger claim needed for downstream use: that this frozen
action-aligned relational student supplies invariant, cross-game semantic
probabilities. Root-only wins the main calibration and ranking comparisons,
relation shuffling improves calibration, seven games regress against
action-only, game identity remains easy to decode, and terminal retrieval is
zero.

This does **not** refute the conditional SAGE12 architecture
(semantic hypotheses → world model → EBM → hierarchical controller). It
rejects V4.10 as the semantics provider for that architecture. Fitting the
world model and EBM now would mostly test game-specific rate signatures and
root/action priors, not the intended transferable relations.

The next semantics iteration should concentrate data and supervision where
V4.10 is weakest: cross-game-matched terminal/progress interventions,
explicit positive/negative causal pairs for `productive`, `target_removed`,
and `contact_lost`, and a representation-level identity criterion rather than
post-hoc output alignment. Root-only should remain the incumbent baseline.

## Published artifacts

- `logo_predictions.jsonl`: 9,128 strict LOGO predictions for all controls.
- `student_result.json`: frozen checks, folds, per-effect and per-game metrics.
- `v4_7_slot_annotations.jsonl`: 9,520 V4.7-compatible annotations covering
  4,760 slots for the action-aligned and relation-shuffle variants; audit only.
- `v4_7_slot_export.json`: annotation cardinality and checksum.

Result checksum:
`338613e30178125b7143f32bc25c8f18c66af153e7969e61f15cfe8d2d8cefe5`.

Prediction SHA-256:
`2179b14a8540fcc5392602dc3ab595d98ed321a4f3af3fac5a177bfaa0e52021`.

Slot-annotation SHA-256:
`29e8564ced710fc347768b75a6d43175db066c09c8b013c1ab90f3b55cceed17`.

Validation: Ruff passed, artifact cardinalities and checksums passed, and all
159 SAGE12 tests passed under the bundled Python 3.12 environment.
