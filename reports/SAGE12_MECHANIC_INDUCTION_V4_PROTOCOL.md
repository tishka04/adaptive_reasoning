# SAGE12 temporal mechanic-induction pilot V4 protocol

Status: frozen before prospective V4 collection.

Frozen manifest checksum:
`24fc301460c015fbfa9d4647bf13733caa2ca373e07782c8c6d2bda11fdad901`.

## Question and authority boundary

V4 tests the first untested causal link in the proposed architecture:

> Can eight observed semantic transitions induce typed game-local mechanics
> that predict the next transition better than action identity, a deterministic
> template, and an equally informed local action table?

The structured Beta mechanic inducer is primary. Qwen2.5 0.5B emits the same
typed rules on a frozen subset as a secondary ablation. V4 never fits or runs
a semantic world model, trajectory ranker, EBM, shadow controller, bounded
probe, or active controller. Passing authorizes only a separately frozen V5
world-model pilot using deterministic hypotheses.

## Data and temporal firewall

Source-training development derives length-eight contiguous contexts from the
3,040 already published V3 source-training traces. A context and query must
remain inside one game, policy seed, and reset, and every adjacent
`frame_after`/`frame_before` hash must match. V3 source-validation rows are
non-gating because their outcomes have already been inspected.

After source-only preflight is frozen, V4 collects 768 fresh prospective
transitions: 256 each from `re86`, `ls20`, and `sc25`, using policy seeds 131,
173, 211, and 257. Collection is balanced across legal actions and is never
outcome-adaptive. Real chronological repeats remain in the raw audit stream;
only identical complete context/query audit digests are removed from scoring.

These games are known source-validation games. V4 can therefore establish
prospective trajectory-level adaptation, not untouched-game generalization.
The holdout, historical games, and `ar25` remain closed.

## Model-facing contract

The raw tracker may use frames, component values, and coordinates only to
associate objects inside one reset. Its output contains local semantic roles,
action identity and family, one anchor condition, effect bits, and
applicability masks. Track IDs, coordinates, colours, values, grids, hashes,
game ID, policy metadata, reset/step numbers, and the query outcome are
forbidden from the model view.

One `sage12-mechanic-window-v4` record contains:

- eight observed `SemanticTransitionEvent` values;
- one outcome-blind `MechanicQuery`;
- four audited query labels stored outside the model view;
- provenance and a checksummed audit digest.

A `MechanicRule` has an exact-action or action-family scope, one anchor
condition, one effect, and `support=0`. `MechanicEvidence` separately records
observed support, refutation, source prior, and posterior probability.

## Frozen inference and baselines

For each effect, the structured inducer considers exact/family rules with the
query anchor condition and `any` backoffs. It selects the first rule with at
least two applicable context observations, otherwise the family/any backoff.
A Beta posterior combines local evidence with a source-only prior of strength
2. The fixed threshold is 0.5 and at most eight rules are emitted.

Baselines are:

1. source-trained global action-only Beta probabilities;
2. the V3 deterministic action-target template;
3. a local action-only Beta table updated from the same eight observations.

The stronger baseline is the one with lowest validation macro Brier score.
Controls permute context outcomes, permute action-anchor bindings, remove the
context, permute source labels, and probe identity in static query features.

Qwen uses the local frozen weights on `cuda:0`, temperature zero, at most 512
input and 256 output tokens. The 128 contexts are selected before query
outcomes by game, action, anchor condition, and audit digest. No JSON repair
or prose extraction is allowed.

## Frozen gates

All structured gates must pass:

- at least 1,500 source-training and 500 prospective windows;
- at least 75 positive and negative training examples per effect;
- at least 30 positive and negative prospective examples per effect;
- actor role known at least 0.95 globally and 0.90 in every game;
- zero model-view, reset, continuity, or query-outcome leakage;
- structured JSON validity, `support=0`, and grounding exactly 1.00;
- macro Brier skill at least +0.10 over the stronger baseline;
- run-cluster bootstrap 95% lower bound strictly above zero;
- macro-F1 gain at least +0.05;
- context-outcome shuffle reduces Brier skill by at least 0.05;
- eight-transition context gains at least 0.05 Brier skill over no context;
- non-negative Brier skill in every validation game;
- macro ECE at most 0.10;
- static identity gain beyond action-only at most +0.10.

Qwen JSON, grounding, prediction, and shuffle results are published but do not
change the V4 verdict. Any structured failure produces `FAIL_CLOSED`.

## Reproduction and publication order

```powershell
python -m theory.sage12.mechanic_induction preflight
python -m theory.sage12.mechanic_collection
python -m theory.sage12.mechanic_induction evaluate
```

The implementation, tests, this protocol, and the checksummed manifest are
committed before preflight or collection. The source-only priors and preflight
are committed before prospective collection. Final shards, predictions,
controls, positive or negative result, and all documentation are published
without changing a gate.

## Source-training preflight result

The frozen source-only run produced 1,911 unique length-eight windows from all
11 source-training games. All four labels exceed the frozen 75-positive and
75-negative capacity. Static query identity accuracy gained +0.0816 over
action-only, inside the +0.10 limit, and the model-view firewall passed.

Persistent actor-role coverage remained 0.831 globally, with `cd82` at 0.049
and `sp80` at 0.244. It therefore failed the frozen 0.95 global and 0.90
per-game gates. Preflight checksum:
`5ae964387078c0b0f0ef529fc8d5bb96f05daed697540e1034ab8f5600fff44b`.
Priors checksum:
`7f9d62dddd392387d90c31409c203e0b0d23f5e7432e218f7d011d8ddc08042a`.

This is already sufficient to prevent promotion, but it does not change the
publication order. The fixed prospective collection and predictive controls
will still run so the representation receives a complete positive or negative
transfer audit. No tracker, prior, threshold, gate, or feature is changed
after this preflight.
