# SAGE12 V4.18 — goal-conditioned trajectory value

Status at freeze: **protocol specified; no V4.18 score observed**.

## Question

V4.17 made the offline ranker measurably sensitive to morpho-topological
relations, but the signal did not improve active control. V4.18 tests whether
the missing quantity is long-horizon credit:

> Can a compact, object-relative critic learn which current actions advance a
> future subgoal when preparatory actions receive credit from complete human
> trajectory suffixes?

The experiment runs every registered offline condition and all nine bounded
active runs. No intermediate gate cancels a later condition.

## Frozen data boundary

- Human fitting games: `ar25`, `bp35`, `cd82`, `cn04`, `dc22`, `ft09`.
- Offline transfer games: `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`,
  `tr87`, `tu93`.
- Bounded active games: `re86`, `ls20`, `sc25`.
- Seeds: `0`, `1`, `2`.
- Final confirmation games: closed.

The critic is fitted only on the existing 5,661 human decisions grouped into
41 complete sequences. Transfer panels, active outcomes, game identity,
absolute coordinates, palettes, future frames and holdout data never enter
the fit.

## Strict storage contract

Every V4.18 command runs inside the same fail-closed storage guard:

- one unique ignored scratch directory per command;
- at most 5 GiB of scratch;
- at most 5 GiB in the ignored persistent V4.18 cache;
- no newly created or enlarged derived file above 512 MiB;
- at most 12 GiB for the complete repository, including `.git`;
- at least 100 GiB free on the repository volume;
- size/checksum inventory before and after every command;
- content hashes for every compact V4.18 output;
- scratch removal before the command is considered complete.

The three regenerable V4.16 corpora are explicitly ignored and are never
rebuilt:

- `train_embeddings.jsonl`;
- `train_transitions.jsonl`;
- `transfer_transitions.jsonl`.

Only code, tests, reports, manifests, compact decisions and small metadata are
publishable. Model checkpoints stay in the ignored bounded cache.

## Required checkpoint regeneration

The ignored V4.14 temporal student/EBM and V4.15 policy checkpoints are rebuilt
only from their tracked corpora and frozen manifests. The rebuild uses
cache-local copies so tracked historical artifacts cannot be overwritten.

The following parity diagnostics are recorded against the published outputs:

- V4.14 effect-bit agreement and mean probability delta;
- V4.15 chosen-action agreement and milestone agreement;
- source manifest and corpus fingerprints;
- checkpoint sizes and hashes.

Parity is diagnostic, not a gate that silently cancels V4.18. Any discrepancy
is reported and all registered comparisons still run.

## Retrospective multi-horizon teacher

For each executed human action, observed future events are discounted with
`gamma = 0.97` at horizons 8, 16, 32 and 64:

- `motion`;
- `object_change`;
- `topology`;
- `access`;
- `terminal_progress`;
- `risk`;
- `overall`.

Positive factors use the strongest discounted occurrence in the horizon.
Risk is negative. `overall` is a clipped discounted sum of productive effects
minus risk. Unexecuted candidates never receive invented regression targets;
one deterministic alternative per positive suffix supplies only a ranking
negative.

Compilation is streaming by complete sequence. Candidate graphs are hashed
directly into sparse 512-dimensional vectors, so no large graph or embedding
matrix is written.

## Compact critic

The critic is a small MLP:

- input: 512-dimensional, palette-free, object-relative graph hash;
- hidden widths: 128 then 64;
- value head: `7 goals × 4 horizons`;
- immediate-factor head: 6 factors.

It is trained with smooth-L1 trajectory regression, immediate-factor binary
cross-entropy and a positive-suffix pairwise ranking loss. Six
leave-one-human-game-out folds are evaluated before one full-data checkpoint
is fitted. CUDA is selected only if an identical training-step benchmark is
at least 1.10× faster than CPU.

## Offline comparison

All 768 existing transfer panels and 2,831 arms are scored. Registered lanes:

1. V4.15 learned policy;
2. V4.17 published hybrid;
3. action-family-only trajectory value;
4. learned V4.18 critic hybrid;
5. V4.18 hybrid with relations removed;
6. true-future trajectory-oracle hybrid;
7. true-future trajectory oracle alone;
8. exact oracle.

The learned hybrid is fixed as:

```text
z(V4.15 policy)
+ 0.5 × z(V4.18 overall value at horizon 32)
- 0.5 × z(V4.14 temporal energy)
```

The true-future lanes use already-collected continuation utility only for
diagnosis and are not deployable. Utility, regret, completion-arm capture,
per-game deltas and paired 95% bootstrap intervals are reported.

## Active comparison

Nine fresh V4.18 runs are executed:

- three games × three seeds;
- 1,000-action budget;
- maximum 14 resets;
- only legal actions are scored;
- one action is executed before replanning;
- V4.15 policy, V4.17 and V4.18 are compared on matching game/seed keys.

No online oracle is claimed because the environment has no verified
clone-and-restore operation for branching all actions from an identical live
state.

## Interpretation

- Oracle without positive gain: `OBJECTIVE_OR_INTEGRATION_BOTTLENECK`.
- Oracle strong, learned critic weak:
  `REPRESENTATION_OR_DATA_BOTTLENECK`.
- Learned critic strong offline, no active level:
  `PLANNING_OR_EXECUTION_BOTTLENECK`.
- At least one active level and no illegal action:
  `GOAL_CONDITIONED_TRAJECTORY_VALUE_SUPPORTED`.

No result opens the final holdout or promotes controller authority.

## Commands

```powershell
python -m theory.sage12.goal_conditioned_trajectory_value_v4_18 freeze
python -m theory.sage12.goal_conditioned_trajectory_value_v4_18 rebuild
python -m theory.sage12.goal_conditioned_trajectory_value_v4_18 compile
python -m theory.sage12.goal_conditioned_trajectory_value_v4_18 train
python -m theory.sage12.goal_conditioned_trajectory_value_v4_18 evaluate
python -m theory.sage12.goal_conditioned_trajectory_value_v4_18 active
```

Artifacts are written under
`training/sage12/goal_conditioned_trajectory_value_v4_18`; checkpoints and
runtime dependencies remain under ignored `.sage12_cache/v4_18`.
