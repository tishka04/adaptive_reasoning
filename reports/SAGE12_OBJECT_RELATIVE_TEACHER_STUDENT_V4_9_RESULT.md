# SAGE12 V4.9 — Object-relative teacher/student result

Verdict: **OBJECT_RELATIVE_TEACHER_IMITATION_NOT_YET_SUPPORTED**.

This is an exploratory source-only result. It does not refute the global
SAGE12 architecture and does not grant live authority.

## What was built and evaluated

The deterministic semantic teacher passed all QA and compiled 17 physical and
functional predicates over 7,541 unique executed transitions. A compact
DeepSets student then predicted those predicates from the pre-action
object-relative graph only.

Every published student probability is strict leave-one-game-out across the 11
SAGE11 source-train games. The run trained 22 neural models (full graph and
root-only for each held-out game) on `cuda:0` and finished in 236.4 seconds
wall time. Source validation, holdout, historical and live environments
remained closed.

## Frozen decisions

| Check | Result | Pass |
|---|---:|:---:|
| Teacher ready | all QA checks pass | yes |
| Full macro-Brier better than action-only | −0.003077 gain | no |
| Full macro-Brier better than root-only | +0.000253 gain | yes |
| Productive-pair accuracy better than action-only | +0.058321 | yes |
| Relation shuffle worsens macro-Brier | −0.005422 degradation | no |
| Neighbor-order probability delta ≤ `1e-6` | `2.38e-7` | yes |
| Semantic-output game identity ≤ 0.60 | 0.9064 | no |
| Completion recall@8 ≥ 0.20 | 0/5 = 0.00 | no |

Four of eight frozen checks failed.

## Direct semantic prediction

| Model/control | Macro-Brier ↓ | Base-seven ↓ | Functional-ten ↓ |
|---|---:|---:|---:|
| action-only | **0.085405** | **0.087157** | **0.084179** |
| relation shuffle | 0.083060 | 0.084978 | 0.081718 |
| object-relative | 0.088482 | 0.090959 | 0.086749 |
| root-only | 0.088735 | 0.090775 | 0.087306 |

The full graph very slightly improves the root-only model, which demonstrates
that neighbor features are used. It does not beat the simpler action-only
effect rates. More importantly, rotating relation directions improves rather
than degrades calibration. The learned direction-specific associations
therefore transfer negatively.

The full model improves over action-only on `changed`, `contact_gained`,
`reachable_area_increased`, `reachable_area_decreased`, `risk`, and
`target_created`. Its largest regressions are `contact_lost` (−0.0200 Brier
gain), `moved` (−0.0153), `target_removed` (−0.0148), and `productive`
(−0.0058). These common game-concentrated effects dominate the useful smaller
signals.

## Productive ranking

On 703 same-prestate pairs with different teacher productive scores:

| Model/control | Accuracy |
|---|---:|
| root-only | **0.5334** |
| object-relative | 0.4296 |
| relation shuffle | 0.3841 |
| action-only | 0.3713 |

The full student does learn more causal ranking signal than action identity
alone. However, the root-only network is substantially better than the full
graph. Neighbor relations currently add unstable game-specific correlations
rather than robust productive ordering.

## Identity and terminal semantics

A classifier reading only the 17 LOGO semantic outputs recovers game identity
at **90.64%** accuracy versus a 10.04% majority baseline. The training-time
identity confusion objective is therefore insufficient. This leakage is not
from an explicit ID, coordinate, colour, value, object ID, or future frame—the
input firewall passed. It comes from the strong game-specific distribution of
effects and the fold-specific calibration left in the semantic outputs.

Completion recall@8 is **0/5**:

- `lp85`: 0/4;
- `sp80`: 0/1.

Five positives across two games are not enough to learn terminal transfer by
LOGO. This result does not show that terminal semantics are intrinsically
unlearnable; it shows that this corpus cannot support that claim.

## Interpretation

The iteration produces a valid semantic teacher and a deployable pre-action
object-relative predictor implementation, but the predictor is not yet a
trusted cross-game semantic provider.

The positive evidence is narrow but real:

- neighbor-set processing is exactly permutation invariant;
- full graphs slightly beat root-only macro calibration;
- the model beats action-only on productive pair ranking and several effects;
- all inputs and predictions respect the source-only firewall.

The negative evidence is decisive for this frozen student:

- action-only remains the best unshuffled direct semantic baseline;
- relation direction has the wrong transfer effect;
- root-only beats the full graph for productive ranking;
- semantic outputs remain a strong game signature;
- terminal events are never retrieved.

The correct conclusion is not “abandon higher semantics.” It is: **do not fit
the semantic world model or EBM from these V4.9 probabilities yet**. The next
iteration should first acquire cross-game-balanced functional interventions,
replace absolute direction tokens with action-aligned/contact-topology
relations, and calibrate outputs with a shared cross-fold invariant head.

## Published artifacts

- `teacher_corpus.jsonl`: auditable teacher corpus.
- `same_prestate_pairs.jsonl`: counterfactual pair links.
- `teacher_qa.json`: teacher capacity and QA.
- `logo_predictions.jsonl`: all four LOGO prediction variants.
- `student_result.json`: complete metrics and per-game results.
- `v4_7_slot_annotations.jsonl`: base-seven V4.7-compatible annotations,
  published for audit only.
- `v4_7_slot_export.json`: export checksum and coverage.

Result checksum:
`6bc0bb39ac7c2e9fe673b4078251755ed5adc4f05aa3b4cb896703b17c37d9af`.

Validation: Ruff passed, artifact cardinalities/checksums passed, and all 153
SAGE12 tests passed under the bundled Python 3.12 environment.
