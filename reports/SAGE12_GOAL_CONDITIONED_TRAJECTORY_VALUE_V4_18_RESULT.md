# SAGE12 V4.18 — goal-conditioned trajectory value result

Status: **complete**

Verdict: **`REPRESENTATION_OR_DATA_BOTTLENECK`**

V4.18 executed every pre-registered condition and all nine bounded active
runs. The result supports the long-horizon objective and its integration, but
does not support the learned critic as a transferable action-value signal.

## Storage outcome

The strict storage contract succeeded throughout all six commands:

| Command | Repository after | Local cache | Free space | Scratch removed |
|---|---:|---:|---:|---|
| `freeze` | 7.066 GiB | 0 MiB | 587.1 GiB | yes |
| `rebuild` | 7.406 GiB | 347.6 MiB | 586.7 GiB | yes |
| `compile` | 7.424 GiB | 347.6 MiB | 586.7 GiB | yes |
| `train` | 7.454 GiB | 347.9 MiB | 586.7 GiB | yes |
| `evaluate` | 7.470 GiB | 347.9 MiB | 586.7 GiB | yes |
| `active` | 7.522 GiB | 370.9 MiB | 586.6 GiB | yes |

No budget error occurred. No file exceeded 512 MiB. The complete repository
stayed below 12 GiB, the ignored cache stayed below 5 GiB and free space
stayed far above 100 GiB. The three forbidden V4.16 corpora were never
created. The final active-command inventory checksum is
`673a41bb623b43f3e0054c287dae9e53976fde08743443cd764687c5a90261d4`.

All per-command inventories and output hashes are recorded in
`training/sage12/goal_conditioned_trajectory_value_v4_18/storage_events.jsonl`.
The ignored cache may be deleted without losing any published result.

## Checkpoint reconstruction

Only the required V4.14/V4.15 components were rebuilt:

| Component | Size | SHA-256 |
|---|---:|---|
| V4.14 temporal student | 787,271 bytes | `13d494fa…03363` |
| V4.14 temporal EBM | 4,003 bytes | `debcc8ec…d8064` |
| V4.15 demonstration policy | 942,601 bytes | `666aef0a…491ab` |

The new CPU/PyTorch environment did not reproduce the historical predictions
bit-for-bit:

- V4.14 temporal effect-bit agreement: **0.8985**;
- V4.14 mean probability delta: **0.1012**;
- V4.15 selected-action agreement: **0.8006**;
- V4.15 milestone agreement: **0.7661**.

This is a reproducibility limitation of the regenerated historical
checkpoints. It was pre-registered as a reported diagnostic rather than a
gate, so it did not cancel or retune V4.18.

## Teacher and critic

The streaming compiler produced:

- 5,661 human decisions;
- 41 complete sequences;
- horizons 8, 16, 32 and 64;
- 74 terminal-progress events;
- 780 access events;
- 1,295 topology events;
- 322 risk events;
- zero invented regression targets for unexecuted actions.

The compact corpus is 19,829,686 bytes with SHA-256
`3c563ca501aa55d1c6099272d326077d48e49d44ba34224e7501ea09cb317a6f`.

The 301 KiB critic trained in 6.98 seconds on CPU. CUDA was unavailable in the
installed PyTorch runtime; installing a second CUDA stack was not justified
for this seven-second fit under the storage-first protocol.

Leave-one-human-game-out diagnostics:

- learned value MAE: **0.2876**;
- action-family-only MAE: **0.3038**;
- relation-removed MAE: **0.3214**;
- immediate-factor macro-F1: **0.1587**;
- game-identity increment: **+0.0691**, below the +0.10 ceiling.

These source diagnostics suggested a weak relational signal, which the
transfer comparison then tested directly.

## Offline transfer

All eight lanes ran on 768 panels and 2,831 arms:

| Condition | Mean utility | Gain vs V4.15 | Completion arms |
|---|---:|---:|---:|
| V4.15 policy | 0.3932 | — | 2 |
| V4.17 hybrid | 0.4277 | +0.0345 | 1 |
| action-only | 0.4721 | +0.0790 | 1 |
| V4.18 learned | 0.4374 | +0.0442 | 1 |
| V4.18 without relations | 0.4453 | +0.0522 | 1 |
| trajectory-oracle hybrid | 0.6266 | +0.2334 | 5 |
| trajectory oracle | 0.9043 | +0.5111 | 8 |
| exact oracle | 0.9043 | +0.5111 | 8 |

The trajectory-oracle hybrid has a strictly positive paired 95% interval:
`[+0.1834, +0.2870]`. The objective and integration therefore pass.

The learned critic has a smaller positive gain over V4.15:
`+0.0442`, interval `[+0.0067, +0.0851]`. It nevertheless fails the causal
comparison:

- learned minus action-only: **−0.0347**, interval
  `[−0.0810, +0.0107]`;
- learned minus relation-removed: **−0.0079**, interval
  `[−0.0467, +0.0299]`;
- oracle-gain capture: **18.95%**, below the frozen 25% floor.

The learned lane was nonnegative on 5/8 games and selected one completion arm,
but its relations did not transfer as useful causal value.

## Active validation

The minimal local runtime used `arc-agi 0.9.1` and `arcengine 0.9.3` from the
ignored cache. Nine fresh V4.18 runs executed on `re86`, `ls20` and `sc25`,
seeds 0–2:

- actions: **8,184**;
- levels: **0**;
- WINs: **0**;
- GAME_OVER events: **93**;
- illegal proposals: **0**;
- mean decision latency: **0.0611 s**.

The outcomes are exactly identical to the frozen V4.15 and V4.17 comparators
for every paired game/seed. V4.18 is faster than both published comparators
in this run, but it does not alter the useful trajectory.

## Interpretation

The oracle result rules out a defective long-horizon objective as the primary
problem. The failed action-only and relation-removal comparisons locate the
current bottleneck in the learned representation or training data:

- the critic can fit average suffix value;
- it does not recover action-conditioned relational value on unseen games;
- its score is too weak to change active behavior.

The next iteration should not enlarge this MLP or regenerate the old V4.16
corpora. It should first improve the supervision itself: persistent
object correspondence, localized before/after transformations, explicit
preconditions and contrastive alternatives from the same pre-state.

## Integrity and safety

- Manifest checksum:
  `3aa75a2e81a76f925221edf2393ccfda4481826362355813ff9937ec999e4194`.
- Final result checksum:
  `2998aaf95f290bf75124e0d61887025cf1f17b816b2bb31ea4dba6893c712adb`.
- Active checksum:
  `f09e04b919b8ebf7447267f5c2c4d02873348b6eab4e071a3bffee274e329635`.
- All registered conditions executed.
- Final holdout remained closed.
- Controller authority remained off.

Focused validation passed: Ruff is clean and all 33 V4.14–V4.18/storage tests
pass. The repository-wide run collected 1,885 tests: **1,883 passed** and two
unrelated environment-dependent tests failed because this fresh clone does not
contain the ignored historical `training/checkpoints` files or the untracked
Qwen `model.safetensors`. Those large optional assets were deliberately not
downloaded under the storage-first protocol. The only warning was the existing
Joblib fallback from physical-core to logical-core detection on Windows.
