# SAGE.11 small relational pilot result

Status: **NO-GO — stop the current world-model track**

Execution date: 2026-07-26

Pre-registration commit: `3ac6d60`

Frozen dataset commit: `151ed9b`

Pilot format: `sage11-relational-effect-logo-v1`

Result checksum:
`272a327ab523a4f81f887e69d381d66c33b31d014bac515347f39e197b31177b`

## Decision

The 52-feature relational replacement failed every frozen gate. Do not extend
the compact model with this representation, do not train it on the RTX 4050,
and do not enter source-validation shadow mode.

| Frozen condition | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Changed-cells full minus stronger action/state baseline | at least +0.10 | -0.0059 | fail |
| Composite conditional action-shuffle degradation | at least 0.10 | 0.0048 | fail |
| Changed-cells full minus full-without-relations | at least +0.05 | -0.1202 | fail |
| Changed-cells fold robustness | at least 9/11 non-negative; worst at least -0.05 | 6/11; worst -0.4888 | fail |

Player movement was also below its stronger baseline by 0.0141 and cannot
compensate for the changed-cells failure.

## Aggregate evidence

Changed-cells macro-F1:

- per-action majority: 0.1627;
- action-only: 0.2283;
- state-only: 0.1882;
- full without relations: 0.3426;
- full with relations: 0.2224;
- conditionally action-shuffled full: 0.1988.

Player-moved macro-F1:

- per-action majority: 0.4382;
- action-only: 0.4685;
- state-only: 0.5861;
- full without relations: 0.6039;
- full with relations: 0.5720;
- conditionally action-shuffled full: 0.5859.

The full composite was 0.3972. It was only 0.0100 above the stronger
action/state composite, 0.0760 below full-without-relations, and dropped by
only 0.0048 under conditional action shuffling.

## What improved—and what did not

The new state relation signature is substantially less game-specific than the
old fixed atoms:

- 26 distinct relational state signatures;
- 10 signatures shared by multiple games;
- 13.97% of rows in game-exclusive signatures;
- majority-game prediction accuracy 64.20%, versus 99.17% for the old fixed
  availability/object-role signatures.

That is a real representation improvement: the contact/alignment/proximity
summary behaves more generically across games. It still does not predict
changed-cell dynamics. Adding all relational features reduced changed-cells
F1 by 0.1202 relative to the same streaming model without them, and breaking
the current-action relation barely changed performance.

The result distinguishes two problems:

1. fixed coarse atoms caused severe game-identity negative transfer;
2. removing that shortcut and adding aggregate geometry is still insufficient
   to model action-conditioned effects.

The remaining missing information is likely object identity/correspondence
across frames, local patch content, action semantics beyond identity, or a
target representation that predicts localized changes rather than a global
changed-cell bucket. Those are hypotheses for a new explicit plan, not
permission for post-hoc tuning in this pilot.

## Per-game robustness

Changed-cells full-minus-best-baseline was non-negative on `cd82`, `dc22`,
`ka59`, `lp85`, `su15`, and `tu93`. It was negative on `bp35`, `g50t`,
`lf52`, `sp80`, and `tr87`. The worst fold was `tr87` at -0.4888.

Relations improved changed-cells on `bp35`, `g50t`, `ka59`, `lf52`, `lp85`,
and `tu93`, but harmed it on the other five games. That heterogeneity fails
the cross-game transfer requirement.

## Firewall and compute

The runner verified manifest
`11a734063ac4be4b8cece50a4d6e7ee40bb25ccfacbc8cd703a1565845f39f2c`
and read exactly the 10,027 rows from the 11 source-training shards. It did
not open source-validation, historical, holdout, or regression-only shards.

The frozen full matrix had 113 columns: the streaming representation after
removing fixed availability/object-role atoms, plus all 52 relational
columns. The run took 62.097 seconds, including 59.983 seconds for the 88
fixed estimators.

The RTX 4050 was visible but unused because the frozen scikit-learn estimator
has no CUDA backend. GPU world-model training was conditional on a pass and
therefore did not occur.

## Final consequence

The current SAGE.11 world-model track stops fail-closed:

- no PyTorch training;
- no model checkpoint;
- no source-validation shadow run;
- no historical or holdout access;
- neural authority remains `off`;
- no further recollection or representation search under this protocol.

Any future attempt requires a new explicit, pre-registered plan. This negative
result does not invalidate the dataset as an analysis resource; it rejects
the tested aggregate-relations-to-factorized-effects pairing.

## Reproduction

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.relational_effect_pilot
```

Machine-readable artifact:
`diagnostics/sage/sage11_relational_effect_pilot.json`.
