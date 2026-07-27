# SAGE12 V4.2.1 source preflight result

Date executed: 2026-07-27

Status: `PASS_SOURCE_TRAIN_PREFLIGHT`

Preflight checksum:
`4ce44b0a0eacaa041106813649d6782be44c21790385c31fda03dbe605abecdb`

## Result

All 14 conjunctive source gates passed without opening source-validation
outcomes. Three gates are the V4.2.1 runtime-recovery requirements; the
remaining scientific checks and thresholds are unchanged from V4.2.

The preflight derived 1,911 unique eight-transition windows from 3,040
immutable source-training traces. Each target effect met the frozen capacity:

| Effect | Positives | Negatives |
| --- | ---: | ---: |
| `target_created` | 87 | 1,490 |
| `target_removed` | 226 | 1,054 |
| `target_moved` | 90 | 1,190 |

The calibrated structured inducer achieved:

- macro Brier 0.047919 versus 0.058585 for the stronger local-action
  baseline, a +0.182060 Brier skill;
- macro-F1 0.689068, a +0.074908 gain over that baseline;
- +0.432771 context Brier skill over the no-context ablation;
- macro-ECE 0.036452;
- static identity gain +0.038723 over action identity, below the +0.05 cap.

The complete Qwen prompt range was 295-317 tokens against the 384-token
preflight cap. The source rehearsal, public-rule round trip, generic-`any`
coverage, label capacity, calibration, Brier non-degradation, utility,
identity, prompt-budget, actor-exclusion, and model-view firewall checks all
passed.

Calibration checksum:
`6d71f47af5c3bc2d98e11e233c3799ce1987e40fd3f2333a6cfdd1c41f79c2d8`

Prior checksum:
`e65255233bcaba751923810f2529523f5da458abbc332fb9a30e820526a91a1f`

## Authority

This result authorizes only the frozen V4.2.1 collection of 768 fresh
transitions under seeds 661, 709, 757, and 809. It does not authorize
prospective evaluation until that raw collection is separately published.
V5, world-model fitting, EBM fitting, and controller use remain unauthorized.
