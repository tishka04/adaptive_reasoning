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
  transition-signature deduplication, ACTION6 argument-coverage accounting,
  checksummed JSONL shards, and a versioned manifest.
- `atoms.py` provides a shared typed vocabulary for observations, `FrameDiff`
  effects, causal-schema predicates, and neural candidate hypotheses.
- `pilot.py` runs the mandatory cheap gradient-boosted effect classifier
  before graph-model training. A near-ceiling majority baseline is a no-go,
  not an invitation to train the larger model.
- `model.py` implements a 1,540,953-parameter graph-atom encoder with five
  bootstrap dynamics heads. It predicts next latent state, symbolic effects,
  changed/no-op, progress, terminal, and risk. The terminal head is forced off
  until at least 100 strong terminal/level events exist.
- `training.py` implements JEPA, symbolic-effect, action-contrast,
  consistency, progress, terminal, risk, and no-op losses. Weak progress labels
  receive one-quarter weight. Its concrete optimizer trains bootstrap heads on
  resampled rows, clips gradients, and writes split/data-checksummed
  checkpoints. Promotion uses the amended change-weighted persistence,
  action-shuffle, macro-F1, calibration, and latent-collapse gates.
- `bridge.py` converts calibrated model outputs to typed, falsifiable
  hypotheses while preserving `support=0`.
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
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q `
  tests\test_sage10g_i_symbolic_repairs.py `
  tests\test_sage11_splits_dataset.py `
  tests\test_sage11_model_training.py `
  tests\test_sage11_authority.py
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m ruff check theory\sage11
```

The data shards and trained checkpoint are intentionally not fabricated by
this implementation change. When collected, JSONL shards and the selected
checkpoint must be stored through Git LFS, referenced by checksums in the
manifest/model card, and pass the source-only pilot and world-model gates
before shadow evaluation begins.
