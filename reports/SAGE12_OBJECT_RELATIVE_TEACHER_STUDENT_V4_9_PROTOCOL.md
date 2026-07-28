# SAGE12 V4.9 — Object-relative semantic teacher/student protocol

Status at freeze: **source-only exploratory protocol; no result observed**.

## Question

Can a deterministic post-transition semantic teacher produce a useful dense
causal vocabulary, and can a small pre-action object-relative predictor imitate
that teacher on a game it never saw?

This iteration addresses the semantic bottleneck isolated by V4.7 and V4.8. It
does not change the global SAGE12 hypothesis → world model → EBM → controller
architecture, and it cannot grant live authority.

## Leakage boundary

Only the eleven immutable SAGE11 `SOURCE_TRAIN` games are authorized. Evaluation
is leave-one-game-out (LOGO): every prediction for a game comes from a model
trained on the other ten.

The teacher may inspect:

- the executed pre-state, legal action, and post-state;
- the existing replay-verified physical effect labels;
- terminal state and completed-level deltas;
- auditable absolute grounding fields.

The student may inspect only:

- the legal action name, family, and direction;
- the pre-action root type, occupancy, coarse size/aspect/affordance, path and
  actor relation;
- an unordered set of at most 16 neighboring objects described by relative
  direction, proximity, relative size, coarse size/aspect, alignment, actor
  role, and boundary contact.

Student inputs exclude game IDs, object IDs, trace IDs, absolute coordinates,
seeds, step indices, raw values/colours, shape signatures, frames, and every
post-action field.

## Semantic teacher

The deterministic teacher emits seven physical predicates:

1. `changed`
2. `moved`
3. `target_created`
4. `target_removed`
5. `target_moved`
6. `level_complete`
7. `game_over`

It also emits ten functional predicates:

1. `local_change`
2. `path_opened`
3. `path_closed`
4. `actor_approached_root`
5. `contact_gained`
6. `contact_lost`
7. `reachable_area_increased`
8. `reachable_area_decreased`
9. `productive`
10. `risk`

Every predicate has an applicability mask. `productive` is a documented
composite score over terminal success, target effects, path/reachability change
and approach, with explicit penalties for terminal failure and closing access.
It is supervision and a ranking target, not a learned reward.

The two published raw corpora are deduplicated by trace digest. V4.3 pair links
retain same-prestate counterfactual structure. A pre-freeze capacity audit found
roughly 7,500 unique transitions but only five explicit completion events.
Completion labels will not be generated, duplicated, or reweighted into fake
capacity.

## Object-relative student

The student is a compact DeepSets graph encoder:

- stable hashed categorical embeddings (2,048 buckets, width 32);
- token mean pooling for the action/root;
- shared neighbor encoder with permutation-invariant mean and max pooling;
- a 96-wide trunk and 17 masked binary effect heads.

Training combines:

- masked binary teacher imitation;
- pairwise ranking when two actions from the same pre-state have different
  teacher productive scores;
- an adversarial game-identity classifier. The classifier learns from detached
  representations while the semantic trunk is pushed toward a uniform game
  distribution.

The base seven probabilities export directly as V4.7 `SlotAnnotation` records
with `support=0`. Functional predicates remain available for the later semantic
world-model/EBM extension.

## Baselines and controls

The full object-relative model is compared with:

- action-only Beta-smoothed LOGO effect rates;
- the same neural model with the neighbor set removed (root-only).

Controls are:

- a deterministic relation-direction shuffle;
- exact prediction comparison after reversing neighbor order;
- a cross-validated game classifier trained only on the 17 semantic outputs;
- per-game Brier scores;
- same-prestate productive ranking;
- completion recall@8 within each held-out game.

## Frozen exploratory decisions

All checks are required for the positive exploratory verdict:

- teacher root grounding ≥ 0.95;
- semantic macro-Brier strictly better than action-only;
- semantic macro-Brier strictly better than root-only;
- productive-pair accuracy strictly better than action-only;
- relation shuffling strictly worsens macro-Brier;
- neighbor reordering changes no probability by more than `1e-6`;
- game identity from semantic outputs ≤ 0.60 accuracy;
- completion recall@8 ≥ 0.20.

These are not architecture-wide falsification thresholds. The strict-positive
comparisons test whether relations add any transfer value beyond simpler
inputs. The high identity ceiling is deliberately permissive for an
exploratory source-only corpus. The completion threshold means capturing at
least one event at the audited five-positive capacity.

## Publication sequence

1. Publish implementation, tests, this protocol, and the frozen manifest.
2. Compile and publish the complete teacher corpus and QA.
3. Train the LOGO student (GPU when available), export V4.7 annotations, and
   publish the result whether positive or negative.

No source-validation, historical, holdout, or live environment is opened.
