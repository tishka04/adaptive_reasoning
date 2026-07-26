# SAGE.11 source-train anti-shortcut audit result

Status: **NO-GO — relational recollection required**

Execution date: 2026-07-26

Pre-registration commit: `c76d9ff`

Format: `sage11-anti-shortcut-logo-v1`

Result checksum:
`c4afd1adecbd40b6e3dccba96f3f2e43414d91ad9a04b1dc71f9540027e66a8a`

## Decision

The shared 77-feature representation failed three of the four frozen
conditions. The compact PyTorch world model must not train on this
representation.

| Frozen condition | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Changed-cells full minus stronger action/state baseline | at least +0.10 | -0.1026 | fail |
| Composite conditional action-shuffle degradation | at least 0.10 | 0.0180 | fail |
| Changed-cells fold robustness | at least 9/11 non-negative; worst at least -0.05 | 5/11; worst -0.2314 | fail |
| No harmful fixed-signature reliance | required | signature removal improved both metrics | pass |

Player movement cannot compensate for the failed changed-cells conditions.
The pre-registered next step is a smaller source-only corpus that preserves
contact, alignment, proximity, and object-relative action relations.

## Aggregate evidence

Changed-cells macro-F1:

- action-only: 0.2730;
- state-only: 0.1013;
- full: 0.1704;
- full without availability/object-role signature atoms: 0.3379;
- conditionally action-shuffled full: 0.1546.

Player-moved macro-F1:

- action-only: 0.4945;
- state-only: 0.6769;
- full: 0.6756;
- full without signature atoms: 0.6784;
- conditionally action-shuffled full: 0.6555.

The full composite was 0.4230. It exceeded the stronger action/state
composite by only 0.0339 and fell to 0.4050 under conditional action
shuffling.

## Fixed-signature finding

The 19 availability/object-role atoms are not harmless generic state
descriptors:

- 126 distinct fixed signatures occurred;
- only four signatures were shared by more than one game;
- 87.43% of rows had a signature exclusive to one game;
- majority-game prediction from the signature reached 99.17% accuracy.

The formal shortcut-reliance condition did not trigger because removing the
signature atoms improved rather than reduced performance. That is stronger
evidence against retaining them: changed-cells F1 rose from 0.1704 to 0.3379
without those atoms, and the composite rose from 0.4230 to 0.5082. The fixed
atoms behave like game identifiers and cause negative cross-game transfer.

## Per-game robustness

Changed-cells full-minus-best-baseline was non-negative on `bp35`, `cd82`,
`g50t`, `lf52`, and `lp85`. It was negative on `dc22`, `ka59`, `sp80`,
`su15`, `tr87`, and `tu93`. The worst fold was `tr87` at -0.2314.

Conditional action-shuffle effects were small and sometimes negative. This
confirms that the representation does not reliably encode the action/state
interaction needed for changed-cell dynamics.

## Firewall and compute

The runner verified and opened only the 76,908 rows from the 11 registered
source-training shards. It did not open source-validation (`re86`, `ls20`,
`sc25`), historical, holdout, or regression-only shards.

The audit took 154.073 seconds: 3.040 seconds to verify/load/encode and
150.908 seconds to fit the fixed leave-one-game-out estimators. The RTX 4050
was visible, but the frozen scikit-learn estimator has no CUDA backend.
Changing estimators solely to use the GPU would have violated the
pre-registration.

## Consequence

Do not train `sage11-world-model-v2`. Build a much smaller source-train-only
replacement corpus with versioned, live/data-identical object relations:

- object contact or adjacency;
- horizontal/vertical alignment;
- bucketed object/player proximity;
- action-target position relative to object boxes and centers.

The replacement pilot must preserve the same changed-cells priority,
conditional action-shuffle control, leave-one-game-out transfer test, and
fail-closed publication discipline. It must not recollect another 100,000
rows.

## Reproduction

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.anti_shortcut_audit
```

Machine-readable artifact:
`diagnostics/sage/sage11_source_train_anti_shortcut_logo.json`.
