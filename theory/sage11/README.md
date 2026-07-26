# SAGE.11 neuro-symbolic stack

SAGE.11 adds a learned world model without weakening the repository's
scientific or safety discipline. Neural outputs are candidate hypotheses with
`support=0`; only observed transitions can create symbolic evidence. The
existing controller remains the single execution path.

## What is implemented

- `splits.py` freezes all 25 games into 11 source-training games, three
  source-validation games, five `NEURO_HOLDOUT_V1` games, five historical
  report-only games, and `ar25` regression-only. Every artifact operation is
  checked against its declared purpose.
- `curriculum.py` merges per-game SAGE.10g causal-schema libraries, coalesces
  content-addressed schemas, preserves audit provenance, and rejects holdout
  sources before a target controller is created.
- `dataset.py` implements the pre-registered 70/20/10 active-controller,
  uniform-legal, and frontier-stall mixture. It enforces per-game caps,
  exact-state transition-signature deduplication, train/validation accounting,
  ACTION6 argument-coverage accounting, checksummed JSONL shards, and a
  versioned manifest.
- `source_dataset_runner.py` executes that policy on the real offline
  environments with independent per-game controllers. It resumes completed
  games and checksum-verified partial shards, rotates seeds deterministically
  in 200-reset windows with independent controllers and duplicate-streak
  counters, merges their content-addressed causal libraries, freezes all 11
  source game libraries, detects finite-state saturation only after 4,000
  consecutive duplicates on every seed, applies the approved 1,292-row
  aggregate overflow only to five source-training games, closes verified
  partial caps once aggregate unique capacity reaches the target, publishes
  deterministic exact-row prefixes, verifies the 100,000-row manifest, and
  merges the source-only SAGE.10g curriculum.
- `atoms.py` provides a shared typed vocabulary for observations, `FrameDiff`
  effects, causal-schema predicates, and neural candidate hypotheses.
- `pilot.py` defines the mandatory fixed gradient-boosted effect classifier
  and the train-only grouped-majority baseline. `effect_pilot_runner.py`
  verifies the source corpus, fits state/effect vocabularies only on
  source-training rows, evaluates the three frozen source-validation games,
  runs a within-game action-shuffle control, and publishes a checksummed
  go/no-go artifact.
- `factorized_effect_pilot_runner.py` implements the separately pre-registered
  v2 follow-up. It removes raw coordinates, factors changed-cells from
  player-movement, reconstructs only leakage-free trajectory relations from
  archived rows, compares a 77-feature full model with a learned 10-feature
  action-only model, and preserves aggregate/per-game controls and checksums.
- `streaming_features.py` owns the versioned 77-column interface used
  identically by archived-row loading and live counterfactual inference. Its
  tracker encodes multiple candidates without mutating state, then advances
  history only after the observed transition. `streaming_dataset.py` provides
  checksum-verified full-corpus and source-train-only loaders.
- `anti_shortcut_audit.py` implements the frozen source-train-only
  leave-one-game-out audit: action-only/state-only/full views, conditional
  current-action shuffling, and explicit fixed-signature identity/ablation
  tests. It does not open source-validation, historical, holdout, or
  regression-only shards.
- `model.py` implements a 1,552,178-parameter graph-atom encoder with five
  bootstrap dynamics heads. It consumes the shared 77 features and predicts
  next latent state, separate changed-cells/player-moved effects, progress,
  terminal, risk, and no-op. The terminal head is forced off until at least
  100 strong terminal/level events exist.
- `training.py` implements JEPA, factorized-effect, action-contrast,
  consistency, progress, terminal, risk, and no-op losses. Weak progress
  labels receive one-quarter weight. Its concrete optimizer trains bootstrap
  heads on resampled rows, clips gradients, and writes split/data/schema-
  checksummed checkpoints. Promotion uses the amended change-weighted
  persistence, action-shuffle, factor macro-F1, calibration, and
  latent-collapse gates.
- `bridge.py` uses the same stateful tracker for live candidates and converts
  calibrated factor-head outputs to typed, falsifiable hypotheses while
  preserving `support=0`.
- `authority.py` implements `off`, `shadow`, `bounded`, and `active`. Off does
  not call the predictor. Shadow logs rankings but returns the byte-identical
  symbolic action. Bounded/active require their gates, yield to protected
  competence, enforce the symbolic danger-memory veto, require positive
  information gain, spend at most one probe per branch/context, demote after
  two non-productive probes, and re-arm only on registered context/effect/route
  evidence.
- `adaptation.py` freezes the encoder and bounds target-local adaptation to a
  2,048-transition replay, one update opportunity every 32 observations, and
  at most four gradient steps. Replay, context, optimizer state, and counters
  reset at every game/seed boundary.
- `evaluation.py` implements shadow gates, paired bootstrap confidence
  intervals, the complete 5 games × 5 seeds holdout promotion rule, and atomic
  per-run JSON checkpoints.

## Strong and weak labels

Observed level completion, WIN, or an explicit terminal event is strong.
Frontier credit, causal-subgoal advance, confirmed route progress, and
sub-effect relay are weak. Weak events train only the progress head at reduced
weight and never count toward the 100-event terminal threshold.

## Authority invariants

1. Symbolic danger memory is a hard veto; neural risk is advisory.
2. Protected terminal competence always wins.
3. Source terminal support only orders probes and never becomes target policy
   authority.
4. Historical games are report-only after one final evaluation.
5. `NEURO_HOLDOUT_V1` is touched only by the final paired confirmation.
6. A failed gate leaves the system in shadow and must be documented as a
   negative result.

## Reproduction

From the repository root:

```powershell
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage11.audit
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.source_dataset_runner --workers 8
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.effect_pilot_runner
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.factorized_effect_pilot_runner
ARC-AGI-3-Agents\.venv\Scripts\python.exe `
  -m theory.sage11.anti_shortcut_audit
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q `
  tests\test_sage10g_i_symbolic_repairs.py `
  tests\test_sage11_splits_dataset.py `
  tests\test_sage11_streaming_features.py `
  tests\test_sage11_anti_shortcut_audit.py `
  tests\test_sage11_model_training.py `
  tests\test_sage11_authority.py
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m ruff check theory\sage11
```

Publishable JSONL shards are stored through Git LFS and referenced by
checksums in the manifest. The first capacity run stopped before publication:
two exact-dedup source games made 98,708 the optimistic upper bound under the
base 8,000/game cap. The approved amendment adds one aggregate 1,292-row
overflow pool across five high-capacity training games. See
`reports/SAGE11_SOURCE_CAPACITY_RESULT.md`. The amended corpus now verifies at
exactly 100,000 rows with manifest
`d4fd8210f2015c00b906cdd98e01630b309deefa7cd9498b38aba8e55130fa1b`;
the frozen 11-source curriculum checksum is
`d11948c5cfcb70ce888b435d63d217b95ce2a0006e4423ae7ac70374d81c630c`.
Model training has not started, and the terminal head remains disabled at
44/100 strong events.

The cheap effect pilot was executed on 2026-07-26 and failed closed:
classifier macro-F1 0.0779 versus 0.0490 for the train-only per-action
majority baseline, a gain of +0.0288 rather than the required +0.10.
Within-game action shuffling degraded macro-F1 by only +0.0059, and all three
validation games failed independently. Result checksum:
`c724aeb6d2ab71154a7c72fa381f3f5f4347a5135644ba64ac82a5542e528136`.
See `reports/SAGE11_EFFECT_PILOT_RESULT.md`. No graph-model, historical, or
holdout run followed.

Factorized pilot v2 was pre-registered in pushed commit `2660f4b` and then
executed once. It formally passed: full composite macro-F1 0.5506 versus
0.3431 for the learned action-only comparator, a +0.2075 gain; both core heads
and all three source-validation games met the frozen non-regression
conditions. Result checksum:
`45f58d1537a1b1a6800636b77df401ab3bf1f94f4ed6dc3bcf2d107864f0328f`.

The result is qualified: player movement supplies nearly all the gain,
changed-cells F1 remains 0.1562, current-action shuffle degradation is only
0.0078, and raw object relations were not archived. The exact v2 interface is
now implemented with schema checksum
`39bb692848fba64ef994e0c0a304785128e1a69adaf6308f1d22623a8f0876bd`,
and the factorized model/trainer/live bridge all consume it. GPU training is
was evaluated by the separately frozen source-train-only anti-shortcut audit.
The audit failed: changed-cells transfer was -0.1026 versus the stronger
baseline, conditional action-shuffle degradation was 0.0180, and only 5/11
folds were non-negative. Fixed signatures predicted game identity at 99.17%;
removing them improved changed-cells F1. GPU training remains blocked while a
smaller relational-data pilot is collected—not another 100,000-row corpus.
Historical and holdout games remain untouched. See
`reports/SAGE11_ANTI_SHORTCUT_AUDIT_RESULT.md`.
