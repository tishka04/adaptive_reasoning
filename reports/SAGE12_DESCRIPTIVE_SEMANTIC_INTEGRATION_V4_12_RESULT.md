# SAGE12 V4.12 — descriptive semantics and conditional integration result

## Verdict

**`DESCRIPTIVE_SEMANTICS_NOT_SUPPORTED`**

The eight-effect semantic gate failed. Per the pre-published V4.12 protocol,
the downstream source-only world model, trajectory generator, EBM and
controller were not fitted:

**`SKIPPED_SEMANTIC_GATE_FAILED`**

This is a negative result for the current object-relative relation
representation and distillation method. It is not a completed test of the
global architecture, because the frozen precondition for that test was not
met.

## Execution

- device: NVIDIA RTX 4050 Laptop GPU (`cuda:0`);
- outer evaluation: leave one of eleven source-train games out;
- records available to training folds: 13,042;
- primary fresh V4.11 arms: 3,914;
- applicable discordant effect pairs: 5,559;
- active effects: `changed`, `moved`, `target_removed`, `target_moved`,
  `local_change`, `contact_lost`, `productive`, `risk`;
- source validation, holdout, historical and live environments: unopened.

The initial execution exposed a performance defect in inner alpha selection:
the implementation recomputed all 17 effect calibrations for every
effect/alpha/held-game candidate. It reached the one-hour process limit before
producing any prediction or result artifact. The calculation was vectorised
and restricted to the eight already-frozen active effects. This optimization
is mathematically equivalent: data, losses, model dimensions, epochs, random
seeds, alpha grid, calibration rule, gates and thresholds were unchanged.
Targeted tests passed before the completed rerun.

## Frozen gate results

| Check | Result | Observed |
|---|---|---:|
| V4.11 collection/firewall ready | PASS | all required checks true |
| effect-pair gain over root-only, CI lower > 0 | **FAIL** | mean +0.00015; 95% CI [−0.02738, +0.02728] |
| effect-pair gain over action-only, CI lower > 0 | PASS | mean +0.03788; lower +0.00855 |
| relation-shuffle pair degradation, CI lower > 0 | **FAIL** | mean −0.03705; lower −0.05186 |
| active macro-Brier gain over root-only, CI lower > 0 | **FAIL** | mean −0.00203; lower −0.00284 |
| relation-shuffle Brier degradation, CI lower > 0 | **FAIL** | mean −0.00185; lower −0.00297 |
| ECE no worse than root-only | PASS | 0.11401 versus 0.12528 |
| pair nonnegative games ≥ 6 | **FAIL** | 4/11 |
| absolute-Brier nonnegative games ≥ 6 | **FAIL** | 4/11 |
| incremental identity upper CI ≤ 0.02 | PASS | upper −0.00856 |
| neighbour permutation delta ≤ 1e-6 | PASS | 1.79e−7 |
| pair-swap complement error ≤ 1e-6 | PASS | 2.22e−16 |

The active-effect macro-Brier scores were:

| Model | Macro-Brier (lower is better) |
|---|---:|
| action-only | **0.12077** |
| root-only | 0.13844 |
| descriptive relation model | 0.14114 |
| relation-shuffled descriptive model | 0.13870 |

## Interpretation

There are two real positives:

1. The descriptive comparator beats action-only on within-state effect
   ordering with a positive confidence interval.
2. It improves calibration error, reduces game-identity predictability
   relative to root-only, and satisfies both mathematical invariance controls.

Those positives are insufficient because the central causal control goes in
the wrong direction. The descriptive model does not reproducibly beat the
root-only anchor, and scrambling its action-relative relations improves both
pair loss and absolute Brier. Therefore the pairwise gain over action-only
cannot be attributed to useful relation semantics. The learned relational
residual is behaving as noise or harmful game-specific structure.

The per-game result reinforces this diagnosis. Descriptive pair loss beats
root-only in only `ka59`, `lf52`, `lp85` and `tr87`; it loses in the other
seven games, including large reversals on `bp35`, `dc22` and `su15`.

Absolute probabilities are weaker still: action-only beats the descriptive
model on seven of eight active effects. `local_change` is the sole sizeable
descriptive improvement over root-only, while relation shuffling improves the
descriptive score on every active effect.

## Architectural consequence

The conditional downstream gate behaved as designed:

- no semantic world model was fitted;
- no trajectory EBM was fitted;
- no controller decisions were evaluated;
- no authority was promoted.

This result narrows the problem. The obstacle is not pairwise supervision or
calibration leakage: both now behave sensibly. The remaining failure is that
the current pre-action relation vocabulary does not carry stable held-game
information about these effects.

The next iteration should not add capacity to this same encoder. A useful next
test is to learn a compact *state-change schema* from short pre-action history
and object persistence—such as movable, consumable, blocking, hazardous and
controllable roles—then evaluate whether those inferred roles survive
relation shuffling and beat the root/action controls before reopening the
global chain.

## Reproducibility

Commands:

```powershell
python -m theory.sage12.descriptive_semantic_integration_v4_12 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.descriptive_semantic_integration_v4_12 run --device cuda:0
pytest -q tests/test_sage12_counterfactual_semantic_panels_v4_11.py tests/test_sage12_descriptive_semantic_integration_v4_12.py
```

Key checksums:

- frozen manifest:
  `1071eecaa67753a887dadb783a0ce5287b4f6fa233e7c1e07b77040184d49835`
- semantic result:
  `ee4e98dcafdf45b9e19ecc36ae662348d7df48c0c3141eb3de76f9c9d5db5c7a`
- conditional integration result:
  `10dd2ca233980aaed544585cd8de23ce08f3bf29a08a868890688527192a3c3a`
