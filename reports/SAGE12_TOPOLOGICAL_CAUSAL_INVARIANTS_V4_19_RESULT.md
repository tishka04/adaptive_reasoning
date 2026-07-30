# SAGE12 V4.19 — topological causal invariants result

Status: **complete**  
Verdict: **`TOPOLOGICAL_OBJECTIVE_BOTTLENECK`**

## Executive result

V4.19 successfully built a canonical and compact topological representation,
but the registered topological value objective did not rank useful actions
better than V4.18. The learned controller consequently reproduced V4.18's
active behavior exactly while running about four times slower.

The failure is informative: object correspondence is not the bottleneck.
Explicit topology is partially predictable across held-out human games, but
the chosen local and short-continuation topological rewards are not aligned
strongly enough with eventual game utility.

## Audit-preserving pre-fit amendment

The first compact-corpus QA reported `contact_added` on all 5,661 transitions.
Before fitting any estimator, the audit traced this to inherited V4.16 event
tokens comparing unmatched free-space nodes against object correspondences.

The corrected compiler restricts contact deltas to confident one-to-one
persistent object matches. The amendment was committed and pushed before the
first fit. It changed:

- `contact_added`: 5,661 → **140**;
- `contact_removed`: 164 → **123**.

It did not change the split, vocabulary, model, coefficients, thresholds or
active budget. The original manifest and first QA remain recoverable in Git
history.

## Corpus and correspondence

- Human decisions: **5,661**
- Uninterrupted sequences: **41**
- Human games: **6**
- Compact artifact before cleanup: **50,939,234 bytes**
- Compact artifact SHA-256:
  `27312930347d3aee9b56c2e108cdadcf7ef7caaa0264ac942dda36e56ed7b8ad`
- Confident structural correspondences: **97.31%**
- Fully ambiguous transitions: **0.247%**
- Structural correspondences audited: **201,917**
- Node-permutation failures: **0**
- Forbidden student fields: **0**
- Raw frames or full graphs persisted: **no**

The correspondence gate passed comfortably.

## Leave-one-human-game-out predictor

The compact `512 → 128 → 64` predictor trained in **4.42 seconds** on CPU.
CUDA was unavailable in the rebuilt local environment; the small fit did not
justify installing a multi-gigabyte CUDA runtime.

| Metric | Result | Gate |
|---|---:|---:|
| V4.19 factor macro-F1 | 0.2037 | — |
| Action-only macro-F1 | 0.0045 | — |
| Gain over action-only | **+0.1992** | ≥ +0.10 |
| Nonnegative human games | **5/6** | ≥ 5/6 |
| Relation-removal degradation | **+0.0370** | ≥ +0.05 |
| Binding-swap degradation | **−0.0111** | ≥ +0.05 |
| Game-identity probe increment | **+0.5257** | ≤ +0.10 |
| Value MAE | 0.3824 | — |
| Action-only value MAE | 0.4284 | — |

The factor targets are meaningfully predictable, but the representation gate
fails. Binding the action to another object does not hurt prediction, explicit
relations are below the required effect floor, and game identity remains easy
to recover from the learned latent.

## Offline transfer

All **768 panels**, **2,831 arms** and **11 registered conditions** executed.

| Condition | Mean utility | Mean regret | Completion arms |
|---|---:|---:|---:|
| V4.15 policy | 0.3932 | 0.5111 | 2 |
| V4.17 hybrid | 0.4277 | 0.4766 | 1 |
| V4.18 learned | **0.4374** | 0.4669 | 1 |
| Action-only | 0.4095 | 0.4948 | 1 |
| Static invariants | 0.4138 | 0.4905 | 1 |
| V4.19 learned | **0.4365** | 0.4678 | 1 |
| V4.19 without relations | 0.4106 | 0.4937 | 1 |
| V4.19 binding swapped | 0.4312 | 0.4731 | 1 |
| Local topology oracle | 0.4447 | 0.4596 | 4 |
| Multi-horizon topology oracle | **0.4375** | 0.4668 | 4 |
| Exact utility oracle | **0.9043** | 0.0000 | 8 |

Key paired results against V4.18:

- V4.19 learned: **−0.00090**, 95% interval
  `[−0.04090, +0.03978]`;
- local topology oracle: **+0.00734**, 95% interval
  `[−0.03160, +0.04973]`;
- multi-horizon topology oracle: **+0.00012**, 95% interval
  `[−0.03958, +0.04125]`;
- exact utility oracle: **+0.46690**, 95% interval
  `[+0.40466, +0.53028]`.

The learned lane beats action-only by +0.0270 on average, but its lower
confidence bound is negative. It beats relation removal by +0.0259 and binding
swap by only +0.0053, also with intervals crossing zero. Only 4/8 transfer
games are nonnegative against V4.18.

The decisive failure is the oracle: exact future utility has substantial
headroom, while the registered topology oracle is effectively tied with
V4.18. Therefore more capacity in the same value learner cannot solve this
objective.

## Active validation

The panel contains 27 frozen comparator runs and nine fresh V4.19 runs.

| Controller | Actions | Levels | WIN | GAME_OVER | Illegal | Mean decision latency |
|---|---:|---:|---:|---:|---:|---:|
| V4.15 + temporal EBM | 8,184 | 0 | 0 | 93 | 0 | 0.0774 s |
| V4.17 hybrid | 8,184 | 0 | 0 | 93 | 0 | 0.2479 s |
| V4.18 goal critic | 8,184 | 0 | 0 | 93 | 0 | 0.0611 s |
| V4.19 topology | **8,184** | **0** | **0** | **93** | **0** | **0.2557 s** |

Every V4.19 run exactly matched V4.18 in levels, wins and GAME_OVER. Active
progress therefore failed, and V4.19 incurred a **4.18×** decision-latency
penalty relative to V4.18.

## Storage result

All seven guarded commands completed without a budget error and removed their
unique scratch directories.

- Final repository size including Git objects: **8.30 GiB**
- Ignored local SAGE12 cache: **371.2 MiB**
- Free disk space: **585.7 GiB**
- Largest V4.19 published runtime artifact: **6.23 MiB**
- V4.16 giant corpora regenerated: **no**

The 48.6 MiB compact training projection is deterministic and regenerable. It
is removed after final checksum verification and is not published.

## Interpretation

V4.19 narrows the diagnosis:

1. Persistent correspondence is sufficiently reliable.
2. Explicit graph-delta factors can transfer better than action identity.
3. The current factor learner still uses game-specific structure and does not
   depend enough on the selected action binding.
4. Most importantly, the registered topology reward itself does not recover
   the exact oracle's utility.

The next iteration should not merely enlarge this MLP or add more of the same
topological transitions. It needs a better goal representation: learned
subgoal predicates or option termination conditions that connect topology to
long-range task progress.

No final holdout, authority promotion or unsupported performance claim
occurred.

## Validation

- Ruff: passed on every V4.19 Python file.
- Focused V4.16–V4.19 suite: **24 passed**.
- Complete repository sweep: **1,889 passed**, two failures.
- Both failures are pre-existing missing optional assets in the clean clone:
  `training/checkpoints` and
  `models/qwen2_5_0.5b_instruct/model.safetensors`.
- Manifest, QA, checkpoint metadata, active and final result checksums:
  reproduced exactly.
- Registered conditions: **11/11** complete.
- Active runs: **36/36** complete.
- Final V4.19 scratch bytes: **0**.
- Post-cleanup, pre-publication inventory checksum:
  `a183d5d48b0449e964c375df5d171bd9f6bf568c1355c35587c6a337f8571a1c`.
