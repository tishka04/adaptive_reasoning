# SAGE12 V4.16 — morpho-topological transformation result

## Verdict

**`SAGE_MT_NOT_YET_SUPPORTED`**

V4.16 compiled and evaluated the complete frozen transformation-learning
lane. The representation is firewalled, permutation-invariant and relatively
free of game-identity leakage, but the learned transformation space does not
form stable, transferable prototype families.

Shadow mode remains off. Active validation and the final holdout remained
closed, and no controller authority was granted.

## Corpus

The compiler retained:

- 5,344 human transitions from `ar25`, `bp35`, `cd82`, `cn04`, `dc22` and
  `ft09`;
- 2,831 transfer arms from `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`,
  `tr87` and `tu93`;
- 3,891 distinct training delta signatures;
- 862 distinct transfer delta signatures;
- zero forbidden student-view fields.

The student sees only the current palette-free morpho-topological graph and
candidate action. Before/after alignment, correspondences, raw frames, game
identity and observed delta signatures remain teacher or audit fields.

## Training and clustering

The CUDA model trained for 50 epochs over the 5,344 human records. Its
64-dimensional teacher latent used the observed transition; the causal query
latent used the current graph and action only.

HDBSCAN selected `min_cluster_size=64`, `min_samples=5`. It produced only two
eligible multi-game prototypes:

| Diagnostic | Required | Observed |
|---|---:|---:|
| bootstrap ARI | ≥ 0.70 | 0.61831 |
| eligible training coverage | ≥ 0.60 | 0.20397 |
| eligible prototypes | — | 2 |

The complete compile/train/cluster/evaluate preparation took 1,338.98
seconds.

## Transfer diagnostics

| Metric | V4.16 | Baseline / gate |
|---|---:|---:|
| augmented-view cosine median | 1.00000 | ≥ 0.95 |
| augmented cluster consistency | 1.00000 | ≥ 0.99 |
| cross-game recall@8 | 0.00000 | baseline 0.00000, strict gain required |
| relation-removed recall@8 | 0.00000 | must be lower than full |
| causal delta Brier | 0.05851 | action-only 0.04611 |
| game identity accuracy | 0.39970 | majority 0.33608 |
| utility-gain 95% lower bound | +0.00651 | > 0 |
| nonnegative transfer games | 7/8 | ≥ 5/8 |
| completion capture | 1/8 | ≥ 4/8 |

The exact gate audit is:

| Gate | Result |
|---|---|
| student view safe | pass |
| augmentation invariant | pass |
| clusters stable and sufficiently broad | fail |
| retrieval beats baselines | fail |
| causal Brier beats action-only | fail |
| topology is used | fail |
| identity leakage bounded | pass |
| global utility and completion supported | fail |

The small positive utility result is not sufficient. Recall remains zero,
relations have no measured effect at the prototype-retrieval gate, causal
delta calibration is worse than action-only and seven of eight completion
opportunities are missed.

## Interpretation

The graph representation itself is not the main failure. Node reordering does
not change the latent or assignment, forbidden fields are absent and identity
probe gain is within the frozen bound.

The bottleneck is prototype formation. The human corpus yields thousands of
nearly unique exact transformation signatures but only two stable eligible
families covering about one fifth of training transitions. The causal
predictor can therefore produce smooth latents without retrieving the
specific transformation family needed on a new game.

V4.17 consequently treats V4.16 as an explicitly weak component and tests
whether it becomes useful only after V4.15 narrows the candidate set.

## Reproducibility

```powershell
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.morpho_topological_v4_16 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.morpho_topological_v4_16 compile
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.morpho_topological_v4_16 train --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.morpho_topological_v4_16 cluster
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.morpho_topological_v4_16 evaluate --device cuda:0
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.morpho_topological_v4_16 shadow
```

Checksums:

- manifest: `192def1ee931c574e1582ad32fbc48e35b7a6b418a0b1b698eb484c54667fb0c`;
- model SHA-256:
  `37e3a368fce95d66661fea363ed8558eb81563cd5087a28f8aa988ad2ea2f26d`;
- result: `f11823db5f4bf3f0afeea397d2f82303d16172db6fdeebb9735ec4634556fda4`.
