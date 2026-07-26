# SAGE.11 factorized effect pilot v2 result

Date: 2026-07-26

Formal decision: **GO under the frozen v2 gate**

Operational decision: **implement the v2 input interface before any model
training; do not train or promote the unmodified graph model**

The pre-registration was committed and pushed as `2660f4b` before any v2
estimator was fit. The single fixed execution produced
`diagnostics/sage/sage11_factorized_effect_pilot_v2.json`, checksum
`45f58d1537a1b1a6800636b77df401ab3bf1f94f4ed6dc3bcf2d107864f0328f`.

## Frozen gate result

The full factorized composite reached 0.5506 macro-F1 versus 0.3431 for the
learned action-only classifiers, an absolute gain of +0.2075. The required
gain was +0.10.

All three frozen conditions passed:

1. overall full-minus-action-only composite: +0.2075;
2. both core-head deltas were non-negative;
3. all three source-validation game deltas were non-negative.

No historical, holdout, or regression-only game was read. The v1 no-go result
remains unchanged.

## Aggregate metrics

| Target | Per-action majority F1 | Learned action-only F1 | Full F1 | Full − action-only | Full − majority |
|---|---:|---:|---:|---:|---:|
| Changed-cells bucket | 0.1716 | 0.1131 | 0.1562 | +0.0431 | -0.0154 |
| Player moved | 0.3917 | 0.5731 | 0.9450 | +0.3720 | +0.5533 |
| Unweighted composite | 0.2816 | 0.3431 | 0.5506 | **+0.2075** | +0.2690 |

The formal pass is real under the frozen rule, but it is highly asymmetric:
player-movement prediction supplies most of the gain. Changed-cells prediction
improves over the learned action-only classifier but remains below the simpler
per-action majority baseline.

## Per-game composite

| Game | Rows | Action-only F1 | Full F1 | Full − action-only | Full − majority | Gate condition |
|---|---:|---:|---:|---:|---:|---|
| `re86` | 7,698 | 0.2620 | 0.3704 | +0.1084 | +0.2558 | Pass |
| `ls20` | 7,697 | 0.1615 | 0.2417 | +0.0802 | -0.1455 | Pass |
| `sc25` | 7,697 | 0.3089 | 0.3418 | +0.0329 | +0.0004 | Pass |

The frozen gate compared the full representation with the learned action-only
model, not with the secondary majority lookup. Consequently `ls20` formally
passes while still trailing its per-action majority. Its changed-cells head
scored 0.0000 macro-F1, so v2 does not establish useful changed-cell dynamics
on that game.

## Shuffle controls

| Control | Composite degradation |
|---|---:|
| Current action block shuffled within game | +0.0078 |
| Direct argument predicates shuffled within game/action | -0.00002 |

These controls were explicitly diagnostic, not gate conditions. Their near
zero effects show that the pass is not evidence of strong current-action or
argument sensitivity. It would fail the later world-model action-shuffle
requirement of at least 0.10.

## Interpretation

Factorization fixed the v1 partial-credit failure: the model could score
player movement correctly without also predicting the exact changed-cell,
progress, and risk tuple. Streaming context was available on 77.2% of
validation rows, which gave the classifier materially more pre-action
structure than the v1 19-bit state.

However, the aggregate 0.9450 player-moved F1 is consistent with an easier
game/regime discrimination effect:

- `re86` has 7,646 moved rows out of 7,698;
- `ls20` has 494 out of 7,697;
- `sc25` has 83 out of 7,697;
- action availability and coarse object-role atoms are nearly fixed within a
  game and can act as an implicit game signature.

This is an inference from the label supports, per-game scores, and negligible
action-shuffle degradation. Explicit game identity was excluded, but the
current atoms can still identify game-specific marginal regimes. The result
therefore demonstrates **factorized predictability from the v2 pre-action
context**, not yet an action-conditioned, object-relational world model.

## Audit heads

- Level complete: validation contained zero positive rows. Full macro-F1 was
  0.4964 versus 1.0000 for the majority predictor; it cannot support a
  generalization claim.
- Game over: validation contained 261 positive rows. Full macro-F1 was 0.4495
  versus 0.4972 for the majority predictor.
- The terminal head remains disabled because the training corpus has only 44
  positive level-complete events, below the frozen threshold of 100.

Neither audit head affected the v2 decision.

## Inputs and coverage

- Same verified 100,000-row manifest as v1: 76,908 training and 23,092
  validation rows.
- 77 full features versus 10 learned action-only features.
- 19 current typed-atom indicators plus reset position, exact continuity,
  state recurrence/recency, previous observed action/effects, relative target
  movement, and atom-set delta.
- Exact contiguous predecessor: 85.3% of training rows and 77.2% of
  validation rows.
- Revisited-state flag: 6.3% of training rows and 14.0% of validation rows.
- Coordinate arguments: 25.9% of training rows and 18.2% of validation rows.
- Raw grids, raw coordinate values, game identity, policy arm, digest bytes,
  current-row outcomes, historical data, and holdout data were excluded.

The archived shards still lack contact, alignment, containment, proximity,
and object-relative geometry. V2 does not test those proposed features.

## Hardware and execution

The NVIDIA GeForce RTX 4050 Laptop GPU and CUDA 12.1 were detected but not
used. The frozen balanced scikit-learn histogram gradient boosters have no
CUDA backend, and changing estimator would have violated the published
protocol. CPU was effective: the complete run took 13.722 seconds, including
3.148 seconds to verify/load/encode and 10.466 seconds to fit every head.

The run used Python 3.12.13, NumPy 2.4.3, scikit-learn 1.9.0, and Torch
2.5.1+cu121. Joblib emitted the known Windows physical-core discovery warning
and used the 16 available logical cores; this does not alter the result.

## Consequence

The result permits implementation of a versioned v2 data/model interface that
reproduces these factorized heads and streaming features. It does not permit:

- training the current graph model on the old 19-atom interface;
- claiming changed-cell predictability;
- claiming current-action sensitivity;
- entering shadow, bounded, active, historical, or holdout evaluation.

Any trained model must still pass the existing changed-transition,
action-shuffle, calibration, collapse, and source-validation gates. Given the
weak shuffle and changed-cell evidence, raw object-relational data collection
remains the likely requirement for a useful action-conditioned world model.

## Reproduction

From the repository root:

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.factorized_effect_pilot_runner
```

The protocol is
`reports/SAGE11_EFFECT_PILOT_V2_PROTOCOL.md`. The result checksum independently
recomputes from the canonical JSON payload.

## Validation

- Ruff passed for the full `theory/sage11` package and updated pilot tests.
- `git diff --check` passed.
- Focused SAGE.10g/SAGE.11 regression set: 32 passed.
- Complete repository suite: 1,652 passed in 201.37 seconds.
- The only warning was the documented joblib physical-core discovery
  fallback.
