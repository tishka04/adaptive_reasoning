# SAGE12 target-mechanic V4.2 — source-preflight result

Date: 2026-07-27

Status: `PASS_SOURCE_TRAIN_PREFLIGHT`

Preflight checksum:
`68747717f45289775cd543aaa027eb24164200b255b42b57368e4c6fba0816ff`

Manifest checksum:
`fba242f31cbc492f44333bcae8c5f9228baee0b79c15f0c68009dce7a76a6210`

## Result

All 11 frozen source gates passed. V4.2 may collect its preregistered 768
fresh source-validation transitions. This result authorizes collection only;
it does not authorize V5, a world model, Qwen authority, an EBM, or a
controller.

The immutable 3,040 V3 source-training traces produced 1,911 unique
eight-transition V4.2 windows.

| Target effect | Positive | Negative | Minimum | Result |
|---|---:|---:|---:|---|
| `target_created` | 87 | 1,490 | 75 / 75 | pass |
| `target_removed` | 226 | 1,054 | 75 / 75 | pass |
| `target_moved` | 90 | 1,190 | 75 / 75 | pass |

`actor_displaced` remained audit-only: 35 positives among 781 applicable
windows and no model or authority exposure.

## Source-only diagnostics

| Metric | Value | Frozen gate | Result |
|---|---:|---:|---|
| Static identity gain over action | +0.0387 | at most +0.05 | pass |
| Calibrated structured macro Brier | 0.0479 | diagnostic | — |
| Brier skill over local action | +0.1821 | at least +0.10 | pass |
| Calibrated macro-F1 gain | +0.0749 | at least +0.05 | pass |
| Context Brier skill gain | +0.4328 | at least +0.10 | pass |
| Calibrated macro-ECE | 0.0365 | at most 0.10 | pass |
| Calibration Brier change | −0.0045 | degradation at most +0.005 | pass |
| Qwen prompt tokens | 295–317 | maximum 384 | pass |

The stronger source baseline was calibrated local action. Structured
calibrated macro-F1 was 0.6891. The model-view firewall and exact actor-effect
exclusion both passed.

## Frozen artifacts

| Artifact | Checksum |
|---|---|
| Source priors | `27593773cdeb96ea4b740b51a2fab88046fa7813ee8ee0e72d95a3273051f1cb` |
| Calibration | `6d71f47af5c3bc2d98e11e233c3799ce1987e40fd3f2333a6cfdd1c41f79c2d8` |
| Source preflight | `68747717f45289775cd543aaa027eb24164200b255b42b57368e4c6fba0816ff` |

`source_validation_opened` remains false in the machine-readable result.
The next permitted action is exactly the frozen collection using seeds 479,
523, 569, and 617 on `re86`, `ls20`, and `sc25`.
