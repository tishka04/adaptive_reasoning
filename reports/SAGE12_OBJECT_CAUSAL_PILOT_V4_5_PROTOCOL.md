# SAGE12 V4.5 rooted intervention-event protocol

Status: **frozen before feasibility audit**

Manifest checksum:
`cfae89ac0de9f263af52dbb042e352869324f301a633012d44ad7b85ec028741`

## Question

V4.4 showed that the hand-built `BindingSignature` is not a transferable
causal discriminator. V4.5 asks whether the changed object and its local
relations can be recovered before any larger semantic model is considered:

> From two interventions executed at the same verified pre-state, can an
> identity-free graph rooted at each action target predict which arm produces
> a discovered object-relative event?

The 2,396 immutable V4.3 source pairs are a design-only feasibility corpus.
No source-validation game may be opened during this audit.

## Tri-view event compiler

Each pair provides one common pre-state and two post-states. Objects are
matched separately from the common pre-state into both arms using:

- translation-normalized and absolute cell IoU;
- relative area;
- centroid distance normalized by the grid diagonal;
- tolerant value compatibility.

A one-to-one match needs score at least `0.65` and a margin of `0.10` over its
nearest competitor. Splits and merges of two or three components require
union IoU at least `0.70` and area ratio at least `0.75`. Weak or near-tied
assignments remain explicitly ambiguous.

The compiler emits appearance, disappearance, displacement, recolouring,
reshape, split, merge, progress, and terminal events. Events are attributed
as direct, local, or collateral to the action root. Semantically equivalent
events on the same pre-state subject in both arms are cancelled as common
dynamics. The remaining supervision is intervention-exclusive.

## Source-discovered vocabulary

The source audit starts with
`locus × operation × direction × magnitude`. For each atomic event it keeps
the finest representation with:

- at least 75 discordant pairs;
- at least 10 discordant pairs in three source games.

If the full token lacks capacity, magnitude is removed, then direction.
Selected tokens are re-counted after merging and must still meet the same
capacity. At least two tokens must survive. The resulting vocabulary is
checksummed and frozen; validation cannot add or refine tokens.

## Rooted model view

`RootedTargetGraph` uses an occupied target, virtual target cell, actor, or
targetless action root. It exposes only:

- action name, family, and requested relative direction;
- contact, adjacency, near, alignment, and relative-direction counts within
  two local hops;
- actor/object roles and relative size;
- identity-free interaction-count, last-operation, and recency buckets.

Coordinates, object IDs, raw values, raw shape signatures, global counts,
global scene signatures, frames, hashes, seeds, paths, labels, and game ID
are forbidden.

The cheap primary model is the V4.4 intercept-free antisymmetric logistic
model. Every row is `left - right` and is augmented with its exact negative.
Baselines are action plus shared history without the root, action only, the
root without history, and a deterministic local template.

## Feasibility gates

All compiler gates are conjunctive:

- V4.3 source collection checksum matches;
- identical pre-state rate equals `1.0`;
- correspondence confident rate at least `0.90`;
- ambiguity rate at most `0.10`;
- root grounding rate at least `0.90`;
- exclusive-event cell localization at least `0.90`;
- at least two promoted event tokens.

All source LOGO predictive gates are also conjunctive:

- macro-Brier skill at least `+0.10` over the strongest baseline;
- macro directional-accuracy gain at least `+0.10`;
- swapping the two roots reduces accuracy by at least `0.10`;
- permuting relation semantics while preserving action and relation counts
  reduces accuracy by at least `0.10`;
- game-identity gain over action difference at most `+0.05`;
- macro ECE at most `0.10`;
- complete arm swap error at most `1e-12`;
- every scoreable game has non-negative accuracy gain;
- paired bootstrap lower 95% bound is positive.

Failure writes `FAIL_CLOSED`, creates no fresh shard, and blocks every later
stage.

## Conditional prospective stages

A feasibility pass alone would permit a fresh source replication on the same
11 SAGE11 source games:

- 32 roots per game;
- seeds 1663, 1721, 1783, and 1847;
- depth-three counterfactual trees;
- 32 actions per reset and at most 64 resets per game;
- all eight preceding full traces retained for persistent object tracking;
- at least 2,000 replay-verified pairs and track continuity at least `0.90`.

Only a complete fresh-source gate pass could open `re86`, `ls20`, and `sc25`
with 64 roots per game and seeds 1901, 1951, 2011, and 2063. Validation would
require at least 1,000 pairs, 30 discordant pairs per promoted event, and all
predictive gates unchanged.

A final validation pass would authorize only preparation of a separately
frozen semantic-world-model protocol. V4.5 never authorizes or fits Qwen, a
GNN, a world model, an EBM, or a controller.

Commands:

```powershell
python -m theory.sage12.object_causal_pilot freeze
python -m theory.sage12.object_causal_pilot feasibility
python -m theory.sage12.object_causal_pilot collect-source
python -m theory.sage12.object_causal_pilot preflight
python -m theory.sage12.object_causal_pilot collect-validation
python -m theory.sage12.object_causal_pilot evaluate
```
