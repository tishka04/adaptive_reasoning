# SAGE12 V4.19 — topological causal invariants

Status at freeze: **protocol and implementation specified; no V4.19 model or
result observed**.

## Pre-fit amendment 1 — persistent contact relations

The first compact-corpus QA, run before any estimator was fitted, reported
`contact_added` on every transition. Audit showed that the inherited V4.16
event token compared unmatched free-space nodes against object
correspondences. V4.19 now measures contact addition/removal only between
confident one-to-one persistent object matches. The initial corpus is
superseded and must be regenerated.

This amendment changes no split, target vocabulary, model, coefficient,
control, threshold or active budget. It is frozen in a separate commit before
the first fit; the original manifest and QA remain recoverable in Git history.

## Question

V4.18 showed that a true-future trajectory oracle improves action selection,
while its learned critic does not. V4.19 tests whether the missing
representation is an explicit causal topology:

> Can a compact model predict object-relative changes in connectivity,
> access, articulation and graph distance, then recover enough of the
> true-future oracle's gain to improve control on unseen games?

This is a falsifiable representation experiment. It does not promote
controller authority or open the final holdout.

## Frozen data boundary

- Human fitting games: `ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`.
- Offline transfer games: `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`,
  `tr87`, `tu93`.
- Bounded active validation: `re86`, `ls20`, `sc25`, seeds 0–2.
- Final confirmation: `NEURO_HOLDOUT_V1`, closed.

The 5,661 existing human decisions are compiled directly from followed raw
traces. Existing V4.11 same-state counterfactual panels supply offline
evaluation only. Their outcomes, game identities and future frames never
enter fitting. No new environment collection is authorized because the
current environment has no verified clone-and-restore operation.

## Storage contract

Every command runs inside the V4.18 bounded storage guard:

- one ignored, command-specific scratch directory;
- scratch and local cache each limited to 5 GiB;
- no derived file above 512 MiB;
- automatic stop above 12 GiB for the repository or below 100 GiB free;
- inventory and checksums before and after every command;
- the three regenerable V4.16 giant corpora remain explicitly ignored;
- raw frames, full graph corpora and embeddings are never persisted;
- only compact vectors, reports, manifests, decisions and a small checkpoint
  may be published;
- command scratch is removed after successful verification.

## Canonical graph

Each observation is converted in memory into object and free-region nodes.
Roles include actor and action root. Structural edges are undirected contact
or containment relations; near/alignment remain descriptive relation
features, not connectivity.

The learned view excludes:

- game and episode identity;
- palette values;
- local or persistent object identifiers;
- absolute coordinates;
- seeds;
- raw action names.

The compiler emits a multiset representation, so arbitrary node permutations
must reproduce the exact same feature vector.

## Persistent correspondence

The audited V4.16 overlap/morphology matcher aligns objects before and after
the action and labels:

- persistence;
- creation and removal;
- merge and split.

Confidence at or above 0.60 is considered grounded. The corpus report records
mean confidence, confident-correspondence rate and fully ambiguous
transitions. Representation support requires at least 90% confident
structural correspondences and fewer than 10% fully ambiguous transitions.
Contact-edge deltas use only confident one-to-one persistent object matches;
birth/death effects represent edges involving created or removed objects.

## Explicit invariants

For each state V4.19 computes:

- node, object and free-region counts;
- connected components and cycle rank;
- actor and action-root components;
- structural shortest distance from actor to action root;
- articulation points and bridges;
- whether the action root is an articulation point;
- bridges incident to the action root;
- free regions reachable from the actor;
- object holes and Euler characteristic.

The teacher factorizes the observed delta into 20 binary effects:

- birth, death, merge and split;
- relative motion and morphology change;
- contact addition/removal;
- free-region increase/decrease;
- articulation and bridge addition/removal;
- reachable-region increase/decrease;
- action-root distance decrease/increase;
- terminal progress and risk.

## Multi-horizon credit

Complete human sequences assign discounted credit at horizons 8, 16, 32 and
64 with gamma 0.97. Productive access, reduced graph distance and terminal
progress are positive; lost access, increased distance and risk are negative.
Only the executed action receives regression targets.

## Compact predictor

The frozen model is a small `512 → 128 → 64` MLP with:

- 20 factor logits;
- four bounded value heads for horizons 8, 16, 32 and 64;
- uncertainty equal to mean factor Bernoulli entropy.

Six leave-one-human-game-out folds are mandatory. The final checkpoint is
fitted only after all fold predictions are fixed.

### Representation gates

`TOPOLOGICAL_REPRESENTATION_SUPPORTED` requires:

- factor macro-F1 at least +0.10 above action-only;
- binding swap degradation at least 0.05;
- relation removal degradation at least 0.05;
- nonnegative factor gain on at least 5/6 human games;
- game-identity probe at most +0.10 above majority;
- the correspondence and permutation gates above.

All later conditions run regardless of this gate.

## Registered controls

- node permutation;
- relation removal;
- deterministic action-root binding swap;
- static invariants only;
- action-family only;
- palette and translation invariance on synthetic QA scenes.

## Offline transfer

All 768 V4.11 panels and 2,831 arms are evaluated. V4.15 supplies the policy
prior, V4.14 its temporal energy, V4.18 its learned value, and V4.17 its
published hybrid score.

The deployable V4.19 score is frozen as:

```text
z(V4.15 prior)
+ 0.5 × z(V4.19 topological value at horizon 32)
- 0.5 × z(V4.14 temporal energy)
- 0.25 × z(V4.19 uncertainty)
```

Registered lanes:

- V4.15 policy;
- V4.17 hybrid;
- V4.18 learned hybrid;
- action-only value;
- static invariants;
- V4.19 learned;
- V4.19 without relations;
- V4.19 with swapped action binding;
- local observed-topology oracle;
- multi-horizon true-future topology oracle;
- exact utility oracle.

`TOPOLOGICAL_VALUE_SUPPORTED` requires:

- a positive paired 95% utility-gain lower bound over V4.18;
- a positive paired 95% lower bound over action-only;
- relation removal and binding swap each cause positive paired degradation;
- nonnegative transfer gain on at least 5/8 games;
- at least 30% of the multi-horizon oracle's gain is recovered;
- at least one completion arm and at least half the oracle's completions.

## Active validation

Reuse by checksum the V4.15, V4.17 and V4.18 runs. Execute nine fresh V4.19
runs on `re86`, `ls20` and `sc25`, seeds 0–2, with at most 1,000 actions and
14 resets per run.

Operational support requires:

- at least one completed level;
- zero illegal proposals;
- no aggregate increase in `GAME_OVER` relative to V4.18.

No active result can rescue a failed offline causal gate.

## Verdict ladder

- `TOPOLOGICAL_OBJECTIVE_BOTTLENECK`: the multi-horizon topology oracle fails.
- `CORRESPONDENCE_OR_REPRESENTATION_BOTTLENECK`: correspondence or factor
  representation fails.
- `VALUE_LEARNING_BOTTLENECK`: invariants transfer, but learned value fails.
- `PLANNING_OR_EXECUTION_BOTTLENECK`: offline value passes, active progress
  fails.
- `TOPOLOGICAL_CAUSAL_CONTROL_SUPPORTED`: all gates pass.

## Commands

```powershell
python -m theory.sage12.topological_causal_control_v4_19 freeze
python -m theory.sage12.topological_causal_control_v4_19 compile
python -m theory.sage12.topological_causal_control_v4_19 train --device auto
python -m theory.sage12.topological_causal_control_v4_19 evaluate --device auto
python -m theory.sage12.topological_causal_control_v4_19 active --device auto
```

Artifacts are written under
`training/sage12/topological_causal_control_v4_19`; ignored checkpoints live
under `.sage12_cache/v4_19`.
