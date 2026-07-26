# SAGE.11 cheap effect-classifier pilot result

Date: 2026-07-26

Decision: **NO-GO — do not train the graph world model**

The pre-registered cheap effect-predictability gate failed on the frozen
source-validation games. The classifier reached 0.0779 macro-F1 against
0.0490 for the train-only per-action majority baseline, an absolute gain of
0.0288. The required gain was at least 0.10.

The negative result is published in
`diagnostics/sage/sage11_effect_predictability_pilot.json`, checksum
`c724aeb6d2ab71154a7c72fa381f3f5f4347a5135644ba64ac82a5542e528136`.
No graph-world-model training, historical evaluation, or holdout evaluation
was started.

## Frozen inputs and leakage controls

- Dataset manifest:
  `d4fd8210f2015c00b906cdd98e01630b309deefa7cd9498b38aba8e55130fa1b`.
- Training: 76,908 transitions from the 11 registered source-training games.
- Evaluation: 23,092 transitions from only `re86`, `ls20`, and `sc25`.
- The 19-state-atom vocabulary and 20 effect classes were fitted on
  source-training rows only.
- Validation contained 16 effect classes. All were represented in training:
  zero validation rows had an unseen state atom or unseen effect class.
- Inputs were 19 binary pre-action typed-atom features plus the existing
  six-value action/argument vector. Game identity, policy arm, post-action
  atoms, outcomes, historical rows, and holdout rows were excluded.
- The baseline selected the training-set majority effect separately for each
  action name, with a global training majority only as an unseen-action
  fallback.

## Fixed classifier

One `HistGradientBoostingClassifier` was run with learning rate 0.08, maximum
depth 4, 100 iterations, early stopping disabled, and random state 11. There
was no hyperparameter search and no outcome-driven rerun.

The action-shuffle diagnostic permuted all six action features within each
validation game, preserved each game's action marginal, did not retrain the
classifier, and used random state 11.

## Results

| Validation set | Rows | Effect classes | Per-action majority F1 | Classifier F1 | Gain | Action-shuffled F1 | Shuffle degradation | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| All three games | 23,092 | 16 | 0.0490 | 0.0779 | +0.0288 | 0.0719 | +0.0059 | Fail |
| `re86` | 7,698 | 9 | 0.0386 | 0.0316 | -0.0070 | 0.0498 | -0.0182 | Fail |
| `ls20` | 7,697 | 8 | 0.0997 | 0.0454 | -0.0543 | 0.0414 | +0.0040 | Fail |
| `sc25` | 7,697 | 12 | 0.0477 | 0.0711 | +0.0234 | 0.0449 | +0.0261 | Fail |

None of the three source-validation games passed independently. The very
small overall action-shuffle degradation is also far below the later
world-model requirement of 0.10 and provides no evidence that the learned
mapping uses action identity in a robust, cross-game causal way.

## Interpretation

This result rejects the current **representation-and-target pairing**, not the
100,000-transition corpus as a whole. Label coverage was complete, so unseen
validation classes do not explain the failure. The likely bottleneck is that
19 coarse state atoms collapse spatially and causally different situations
into the same pre-action representation, while a joint effect class asks the
classifier to recover several consequences at once.

The result does not justify spending compute on the 1.54M-parameter graph
world model. A scientifically valid follow-up would first pre-register a new
cheap pilot using richer source-derived pre-action features, such as local
spatial relations, object counts/locations, and short transition context, or
factor the joint effect target into separately scored heads. The present
result must remain available alongside any replacement pilot.

## Hardware and runtime

The laptop exposed an NVIDIA GeForce RTX 4050 Laptop GPU through CUDA 12.1.
It was deliberately not used: the fixed scikit-learn histogram gradient
booster has no CUDA backend, the dense matrix had only 25 columns, and adding
a GPU estimator would have changed the pre-registered method for setup and
transfer overhead rather than effective acceleration.

The successful CPU run used 16 logical cores and completed in 9.764 seconds:
1.974 seconds to verify/load/encode, 6.642 seconds to fit, and 1.148 seconds
to evaluate. Joblib could not query Windows physical-core metadata and fell
back to the available logical-core count; this warning does not alter the
estimator or results.

The first shell invocation had a 10-second orchestration timeout and was
terminated before publishing an artifact. The exact same command and fixed
configuration were then allowed sufficient time; this was operational
recovery, not an experimental or tuning rerun.

## Reproduction

From the repository root:

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.effect_pilot_runner
```

The runner verifies the manifest and every referenced shard before fitting,
writes the machine-readable result atomically after evaluation, and exits
successfully for either a go or a scientifically valid no-go.

Validation:

- Ruff passed for the full `theory/sage11` package and the updated test file.
- Focused SAGE.10g/SAGE.11 regression set: 29 passed.
- Complete repository suite: 1,649 passed in 290.26 seconds.
- `git diff --check` passed.
- The only warning was the documented joblib physical-core discovery fallback.
