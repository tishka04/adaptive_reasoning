# SAGE12 V4.16 — morpho-topological transformation analogies

Status at implementation: **software complete; no V4.16 experimental result
observed**.

## Question

V4.14 predicts independent semantic effects and V4.15 learns a
demonstration-conditioned policy. V4.16 tests a different representation:

> Can SAGE transfer observed graph transformations across games more reliably
> than it transfers state descriptors?

V4.16 is an isolated, fail-closed research lane. It does not modify the
frozen V4.15 implementation or artifacts.

## Data boundary

- Training uses the six human-trace games: `ar25`, `bp35`, `cd82`, `cn04`,
  `dc22`, and `ft09`.
- Offline transfer uses the eight V4.11 panel games not present in the human
  split: `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`, `tr87`, and `tu93`.
- `re86`, `ls20`, and `sc25` remain closed during V4.16 fitting and
  evaluation.
- `NEURO_HOLDOUT_V1` remains closed.
- The manifest fingerprints every source JSONL and freezes all parameters
  before compilation or model fitting.

## Representation

Each grid is compiled into a palette-free graph containing connected objects,
free-space regions, action-relative morphology, contact, containment,
alignment, reachability, holes, Euler characteristic, and boundary
connectivity.

Raw values, colors, absolute coordinates, persistent object IDs, scene hashes,
game identity, and future grids are forbidden from the deployable model view.
Raw cells and coordinates exist only ephemerally while two observed states
are aligned.

The transition compiler creates deterministic one-to-one and many-to-many
correspondences. Its neutral audit vocabulary includes birth, death, motion,
growth, contraction, merge, split, morphology change, relation edits, and
signed invariant changes. These events supervise reconstruction and positive
mining but never act as cluster labels.

## Models

The shared graph encoder uses three typed message-passing layers. The offline
teacher encodes `(graph_before, action, graph_after)` into a normalized
64-dimensional transformation latent. The deployable predictor receives only
`(graph_before, action)` and predicts the teacher latent, event distribution,
and uncertainty.

Training combines:

- masked event and invariant reconstruction;
- cross-game contrastive learning over compatible invariant deltas;
- a second dropout view of every transition;
- causal query-to-teacher distillation;
- variance regularization;
- a gradient-reversal game-identity adversary.

Every training epoch is balanced across games. Same-prestate alternatives
provide hard negatives when present.

## Discovery and memory

HDBSCAN is selected on training data only from the frozen grid:

- `min_cluster_size`: 16, 32, or 64;
- `min_samples`: 5 or 10.

Selection maximizes bootstrap stability times eligible multi-game coverage.
A deployable prototype needs at least 20 observed transitions and three games.
Prototype IDs are content-addressed from the medoid, centroid, and dominant
delta signatures. Human-readable aliases such as `fusion` or `croissance` are
post-hoc descriptions and have no authority.

At runtime the causal predictor retrieves up to eight transformation
prototypes by cosine similarity and applicability. Expected productivity and
risk come only from observed transitions. Unknown or uncertain queries remain
unassigned, and V4.16 never reclusters online.

## Shadow integration

`SemanticPlanningController` accepts an optional transformation advisor. Its
default is absent/off. In shadow mode the advisor:

1. evaluates only legal candidates;
2. records the suggested action and prototype evidence;
3. leaves the symbolic or semantic action unchanged;
4. observes the executed transition afterward;
5. updates only the assigned prototype's observed evidence.

SAGE-MT failures are caught at the advisory boundary and cannot block the
existing world-model or semantic-memory update. V4.16 has no bounded or active
authority.

## Frozen gates

Shadow activation requires all of:

- no forbidden student-view field;
- median augmented-view cosine at least 0.95 and cluster consistency at least
  0.99;
- bootstrap ARI at least 0.70 and eligible coverage at least 0.60;
- cross-game recall@8 above action-only and state-similarity baselines with a
  positive paired-bootstrap lower bound;
- causal delta Brier below action-only;
- lower recall after removing topological relations;
- latent game-identity accuracy no more than 0.10 above majority;
- positive offline utility lower bound, nonnegative transfer on five of eight
  games, at least one completion, and at least 50% of oracle completion
  opportunities.

Failure writes `SAGE_MT_NOT_YET_SUPPORTED`, keeps the advisor off, opens no
additional split, and authorizes no automatic data collection.

## Commands and artifacts

```powershell
python -m theory.sage12.morpho_topological_v4_16 freeze
python -m theory.sage12.morpho_topological_v4_16 compile
python -m theory.sage12.morpho_topological_v4_16 train --device cuda:0
python -m theory.sage12.morpho_topological_v4_16 cluster
python -m theory.sage12.morpho_topological_v4_16 evaluate --device cuda:0
python -m theory.sage12.morpho_topological_v4_16 shadow
```

All artifacts are written under
`training/sage12/morpho_topological_v4_16`. `shadow_activation.json` records
whether every frozen gate passed; it never grants controller authority.
