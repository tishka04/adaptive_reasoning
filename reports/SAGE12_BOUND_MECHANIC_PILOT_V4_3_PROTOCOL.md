# SAGE12 V4.3 causal-binding and semantic world-model protocol

Status: **frozen before source collection**

Format: `sage12-bound-trajectory-v4.3`
Manifest checksum:
`2376ddd8c9c1c10083dc42ae92b9633ffc55272cf675770908b9467642370cea`

## Question

V4.2.1 showed strong temporal transfer but its binding permutation changed
only a small fraction of useful contexts. V4.3 asks the stronger causal
question: from an identical replay-verified pre-state, does changing the
semantic action binding predict which target effect occurs?

This is a new prospective corpus and contract. It does not reuse the opened
V4.2 or V4.2.1 prospective shards.

## Frozen collection

Each root begins after eight real chronological transitions. The collector
replays the complete reset-local action prefix, verifies the exact grid,
game-state, and level-count hash, then clones that restored state twice. It
executes two legal interventions selected without inspecting their outcomes.
Same-action/different-argument pairs are preferred; otherwise the collector
uses same-family actions with different semantic targets, then any two
distinct legal interventions.

Both outcomes are retained. Each arm is replayed recursively to depth three,
yielding a binary tree with at most seven pairs and fourteen executed branch
transitions per root. Terminal branches truncate naturally; collection never
selects or discards a branch according to its effect.

- source training: 32 roots in each of the 11 SAGE11 source-training games,
  352 roots, at most 2,464 pairs and 4,928 branch transitions;
- source validation: 64 roots in each of `re86`, `ls20`, and `sc25`, 192
  roots, at most 1,344 pairs and 2,688 branch transitions;
- source seeds: 857, 907, 953, 1009;
- validation seeds: 1061, 1103, 1151, 1201;
- action budget: 32 per reset;
- chronological repeats are retained;
- validation collection is blocked until the source representation is frozen.

Full frames, coordinates, action arguments, object IDs, hashes, seeds, reset
indices, and tree paths are audit fields. They never enter a model view.

## Binding representation

`BindingSignature` has three frozen projections:

1. `minimal`: target kind (`occupied_object`, `free_slot`, or `targetless`),
   occupancy, and path status;
2. `relational`: minimal plus requested direction, actor relation, and
   actor-relative direction;
3. `typed`: relational plus target area, aspect, and affordance buckets.

The source-only preflight evaluates every projection by leave-one-game-out
prediction. A projection is rejected if its game-identity accuracy gains more
than +0.05 over action identity. Among passing projections, the highest
structured macro-Brier skill is selected; within 0.005, the simpler projection
wins. The selection, calibration, thresholds, and stronger baseline are
frozen before validation collection.

## Binding model

`BoundMechanicRule` is a Beta-smoothed action-and-binding rule over the three
factorized target effects:

- `target_created`;
- `target_removed`;
- `target_moved`.

Rules may be exact-action or action-family and use the selected binding,
minimal-binding fallback, or `any` binding. A rule proposal always has
`support=0`. Observed support and refutation are stored only in
`BoundMechanicEvidence`. The model uses the preceding eight executed semantic
events and requires at least two applicable local observations before a local
posterior overrides its source prior.

Baselines are action plus history without binding, global action only,
binding without history, and a deterministic causal template. Platt
calibration and F1 thresholds are fit only from source-training
leave-one-game-out predictions.

## Binding gates

All gates are conjunctive:

- replay integrity, strict JSON validity, compiler grounding, and support-zero
  rate equal 1.00;
- at least 2,000 source pairs and 1,000 validation pairs;
- for every target effect, at least 75 source positives and negatives and 30
  validation positives and negatives;
- macro-Brier skill at least +0.10 and macro-F1 gain at least +0.05 over the
  frozen stronger baseline;
- swapping only the two executed query bindings reduces Brier skill by at
  least 0.05;
- discordant-pair accuracy gains at least +0.10, with a positive paired
  bootstrap 95% lower bound;
- every validation game has non-negative Brier skill;
- macro ECE is at most 0.10;
- source game-signature gain is at most +0.05;
- the same-action/different-target `sc25` subset exists and has positive
  discordant-pair accuracy gain.

Failure publishes `FAIL_CLOSED` and forbids semantic world-model fitting.

## Conditional semantic world model

Only a complete binding-model pass authorizes
`BoundSemanticWorldModel`. It rolls the calibrated factorized effects through
an explicit identity-free occupancy state for horizon three with beam width
eight. Creation is legal only for free slots; removal and movement are legal
only for occupied targets. Each predicted event updates the abstract slot
state and the eight-event temporal context.

The model is compared with all four binding-model baselines rolled out under
the same constraints. All world-model gates are conjunctive:

- productive exact three-step effect-sequence recall@8 at least 0.70;
- recall@8 gain at least +0.10 over the stronger rollout baseline;
- horizon-three macro-Brier skill at least +0.05;
- binding swapping reduces recall@8 by at least 0.10;
- macro ECE at most 0.10;
- every validation game has non-negative horizon Brier skill.

A pass authorizes only a future, separately frozen energy and safety protocol.
It does not authorize an EBM, shadow controller, bounded probe, active mode,
holdout opening, or historical evaluation.

## Exclusions and publication order

Qwen, GNNs, EBM training, and controller execution are excluded. The required
publication checkpoints are:

1. code, tests, protocol, and frozen manifest;
2. source corpus and collection manifest;
3. source preflight, projection freeze, calibration, and priors;
4. validation corpus and collection manifest;
5. binding predictions and result;
6. conditional world-model predictions and result, or an explicit skipped
   fail-closed artifact.

Commands:

```powershell
python -m theory.sage12.bound_mechanic_pilot collect-source
python -m theory.sage12.bound_mechanic_pilot preflight
python -m theory.sage12.bound_mechanic_pilot collect-validation
python -m theory.sage12.bound_mechanic_pilot evaluate-binding
python -m theory.sage12.bound_mechanic_pilot evaluate-world-model
```
