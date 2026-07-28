# SAGE12 V4.13 — unconditional semantic-bottleneck curve result

## Verdict

**`SEMANTIC_PREDICTOR_BOTTLENECK`**

All frozen conditions ran through the complete source-only
world-model → depth-three trajectory → EBM → controller chain. The upper
architecture works with true world outcomes and with perfect semantic input.
It remains robust under symmetric semantic corruption down to approximately
75% correct bits. The actual V4.12 predictor does not cross the baseline.

This is the first result in the recent sequence that localizes the failing
component after executing the global architecture rather than stopping at a
semantic gate.

## Component ladder

| Condition | Gain over primary baseline | 95% CI | Completion capture |
|---|---:|---:|---:|
| true world + learned EBM | **+0.42972** | [+0.33547, +0.54267] | 3/3 |
| perfect semantic oracle | **+0.35215** | [+0.24614, +0.46859] | 3/3 |
| 90% symmetric oracle bits | **+0.24777** | [+0.13690, +0.36001] | 3/3 |
| 75% symmetric oracle bits | **+0.19924** | [+0.07876, +0.30964] | 3/3 |
| 50% symmetric oracle bits | **−0.40500** | [−0.70736, −0.17678] | 0/3 |
| learned V4.12 semantics | −0.17871 | [−0.49502, +0.03971] | 0/3 |
| structure without teacher | **−0.41061** | [−0.71566, −0.17233] | 0/3 |

The exact leaf oracle and the true-world/learned-EBM lane both achieved mean
utility `8.04567`, oracle first-action accuracy `1.0`, oracle-leaf accuracy
`1.0`, and 3/3 completion capture. This validates the EBM/controller on the
frozen candidate-complete trees.

The perfect semantic world-model lane achieved:

- mean utility `7.96811` versus `7.61595` for the primary baseline;
- oracle first-action accuracy `0.772` versus `0.540`;
- oracle-leaf accuracy `0.6618` versus `0.3912`;
- 3/3 completion capture;
- positive performance in 9/11 games.

It also beat the structure-only world chain by `+0.76276`, with a fully
positive confidence interval `[+0.52129, +1.08322]`.

## Semantic precision curve

The observed corruption accuracies were 100.00%, 89.98%, 75.03% and 49.97%
over 52,360 semantic bits. Utility was perfectly monotonic with semantic
accuracy (Spearman `1.0`).

The lowest tested condition that retained a positive confidence interval,
completion capture and broad per-game transfer was 75.03% symmetric bit
accuracy. This is a tested lower point, not proof that exactly 75% is the
boundary; the transition lies somewhere between the 50% and 75% conditions.

World-model mean Brier followed the same pattern:

| Input | Mean Brier |
|---|---:|
| oracle 100 | **0.01355** |
| oracle 90 | 0.04003 |
| oracle 75 | 0.06317 |
| oracle 50 | 0.07814 |
| structure only | 0.07806 |
| learned V4.12 | 0.07817 |

At 50% the semantic channel is indistinguishable from no useful teacher
information and the global gain disappears.

## Why V4.12 fails despite 86% raw bit accuracy

The learned V4.12 channel has:

- raw threshold accuracy: `86.15%`;
- macro balanced accuracy: only `48.19%`;
- semantic Brier: `0.10474`.

The raw accuracy is misleading because most semantic bits are negative.
V4.12 often obtains those easy negatives while missing the action-dependent
positive effects needed by the controller. Its per-effect balanced accuracy
is near chance: `moved` 44.21%, `local_change` 44.39%, `productive` 44.71%,
`contact_lost` 48.04% and `target_removed` 48.70%.

The oracle corruption curve uses symmetric independent errors, whereas the
learned predictor makes structured, class-imbalanced errors concentrated on
the informative positives. It therefore cannot be placed at the 86% point of
the oracle curve.

Even so, the learned semantic channel contains some information: it improves
the structure-only chain by `+0.23190`, with a positive confidence interval
`[+0.08501, +0.38584]`. It simply remains `−0.17871` behind the much stronger
action-sequence baseline, is nonnegative on only 5/11 games and captures no
completion opportunity.

## Topology boundary

The positive architecture result is candidate-complete, not yet deployable.
The depth-three planner consumes true descriptors for future V4.3 nodes.

When perfect semantics are forced to reuse only current-root slots:

- gain over the primary baseline falls to `+0.05691`;
- its confidence interval crosses zero `[−0.27835, +0.27029]`;
- completion capture falls from 3/3 to 0/3.

The full oracle topology beats root reuse by `+0.29524`, with a positive
confidence interval `[+0.09752, +0.56278]`. Therefore a learned deployable
state-transition rollout remains a second genuine bottleneck. V4.13 does not
claim that the agent can already win live levels.

## What this supports and what it rejects

Supported on the frozen source trees:

- the EBM/controller can exploit correct world outcomes;
- the learned world-model/EBM/controller composition can exploit correct
  semantic inputs;
- the architecture tolerates substantial *symmetric* semantic noise;
- the current failure is not evidence against the whole architecture.

Rejected:

- the V4.12 snapshot/object-relation predictor is adequate;
- raw semantic accuracy is a useful quality measure under severe imbalance;
- root-slot reuse is an adequate replacement for a real rollout model;
- current offline completion capture can be reported as live level wins.

## Recommended next iteration

The next implementation should combine two pieces rather than run another
snapshot classifier:

1. a temporal semantic belief state that infers persistent object roles from
   executed history—movable, blocker, consumable, hazard, goal-relevant and
   controllable—and is optimized for balanced positive-effect recovery;
2. a deployable object-relative transition model that applies a candidate
   action to that belief state and generates the next semantic state without
   reading V4.3 future nodes.

The learned rollout should first be trained by distilling the already
validated perfect-semantic/world oracle on source data. The decisive test is
then whether it preserves the V4.13 75% robustness region and the 3/3
completion capture without future topology. Only after that result should
real source-game win-rate trials begin.

## Reproducibility

```powershell
python -m theory.sage12.semantic_bottleneck_curve_v4_13 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.semantic_bottleneck_curve_v4_13 evaluate
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.semantic_bottleneck_curve_v4_13 attach-fidelity
pytest -q tests/test_sage12_semantic_bottleneck_curve_v4_13.py
```

Checksums:

- frozen manifest:
  `9f9f75e1443a0af17abd01a224b5b5f7fbdc552192ff60053c49ce843202ca2e`
- result:
  `ff5c19518eb56025444b7387a3f29366409aab1aa117e310156b3e1208612898`
- decisions:
  `1d4e64ed0ebac9f872e17a7d7d1a98e8516aecc2b4ac3965eca2426560cd6412`
- folds:
  `8741199b0a0e7d0971db11b33702a429cde734a9f6e118fa9dd1a25fc07b081f`
