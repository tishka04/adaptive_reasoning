# SAGE12 implementation result

Date: 2026-07-27.

Outcome: the guarded high-semantic planning scaffold is implemented and
integrated; empirical promotion remains closed.

## Delivered

- grounded game-identity-free scene graphs and semantic memory;
- a typed hypothesis/effect DSL with strict JSON parsing and mandatory
  `support=0`;
- a cached local Transformers backend for a small open-weight model plus a
  deterministic template baseline;
- role grounding, precondition checks, exact legal-action compilation, and
  explicit rejection reasons;
- a Beta-smoothed semantic transition model and bounded beam trajectories;
- a six-component heuristic energy and optional tiny pairwise PyTorch EBM;
- hierarchical subgoals and one-action receding-horizon control;
- `off`, `shadow`, `bounded`, and `active` modes with independent gates;
- symbolic danger veto, protected-competence supremacy, bounded context probes,
  and outcome-only evidence updates;
- versioned semantic-trajectory audit records and append-only JSONL support;
- integration with transition observation, branch reset, decision arbitration,
  and controller summary in `UnifiedCognitiveController`.

## Software evidence

Focused validation:

```text
python -m pytest -q tests\test_sage12_semantic_planning.py
14 passed
```

The tests cover scene grounding, support-zero enforcement, illegal/unbound
proposal rejection, local-model JSON parsing, shadow identity, authority gate
downgrade, active receding-horizon selection, danger/protected blocks,
observed-only model updates, integrated-controller activation, deterministic
template behavior, and pairwise EBM optimization mechanics.

Full repository regression:

```text
ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q
1681 passed, 1 environment warning in 190.19s
```

The warning is Joblib falling back from an unavailable physical-core query to
the logical core count; it does not affect a test result. Focused Ruff
validation of `theory/sage12`, its tests, and the unified-controller
integration also passes.

Post-pilot regression:

```text
python -m pytest -q
1692 passed, 1 environment warning in 212.93s
```

The focused semantic-planning and proposal-pilot suites pass 25 tests
together, and targeted Ruff validation remains clean.

Post-V2 regression:

```text
python -m pytest -q
1698 passed, 1 environment warning in 192.04s
```

The three focused SAGE12 suites pass 31 tests together. The warning remains
the same harmless Joblib physical-core query fallback.

## GPU decision

The subsequent Stage A proposal pilot compared identical Qwen2.5 0.5B
decoding on CPU and the laptop RTX 4050. Median inference fell from 26.478
seconds to 6.953 seconds, a 3.808x speedup, so the GPU was used for the 224
clean proposal generations. This was inference only. No semantic world model
or EBM was trained because the proposal gate failed.

## Authority result

The proposal gate has now been evaluated and failed closed. Strict typed JSON,
grounding, recall gain, relation sensitivity, per-game transfer, and
game-signature leakage all failed their frozen requirements. The world-model,
energy, shadow, bounded, active, and holdout stages were therefore not run.
The integrated default remains `off`; bounded and active downgrade to shadow
without all prerequisite gates. See
`reports/SAGE12_PROPOSAL_PILOT_RESULT.md` for the complete result and artifact
checksums.

The separately preregistered constrained V2 repair passed JSON, support-zero,
grounding, and both reduced-leakage gates. It still failed predictive gates:
Qwen primary macro-F1 0.484 versus action-only 0.549, shuffle degradation
-0.098, and `re86` gain -0.237. V2 result checksum:
`7440cbf5a15edd4ca2c7c70fbebdcb2ced1bdf88817bdf1f7c0f417a6db81e3a`.
No world-model fitting followed.

## Action-target V3 implementation and result

V3 adds a separate `sage12-action-target-trace-v3` schema, deterministic
action-anchor resolution, conservative one-to-one before/after object
matching, four masked component labels, balanced/adaptive source-only
collection, source-training leakage selection, independent structured heads,
frozen Qwen embedding ablation, deterministic and action-only baselines,
shuffle/permutation controls, calibration, bootstrap intervals, and complete
artifact checksums.

Collection produced exactly 3,040 source-training and 960 source-validation
rows with no exact duplicates. The source-only preflight selected the coarse
projection and shallow gradient boosting. The once-only validation evaluation
failed closed: structured macro-F1 0.232 versus 0.371 for the stronger
template, primary gain -0.140, target-shuffle degradation 0.0005, and macro
ECE 0.397. JSON, support-zero, and grounding were all 1.00. Result checksum:
`10b1d84b6ff675c3fd05f73ad853d0618658b79045824ad4c2f9e79e6466fdb4`.

The explanatory diagnostic found only 26 unique training signatures in the
selected model view and a target shuffle that changed 12 of 960 validation
rows. This leaves the software and audit corpus available for research but
keeps all later authority closed. See
`reports/SAGE12_ACTION_TARGET_PILOT_V3_RESULT.md`.

Post-V3 focused validation:

```text
python -m pytest -q tests/test_sage12_action_target_pilot.py
tests/test_sage12_semantic_planning.py tests/test_sage12_proposal_pilot.py
tests/test_sage12_constrained_pilot.py
48 passed
```

Post-V3 full repository regression:

```text
python -m pytest -q
1715 passed, 1 environment warning in 242.33s
```

Targeted Ruff validation of the V3 schema, collector, pilot, and tests also
passes. The warning remains the same harmless Joblib physical-core query
fallback.

## Temporal mechanic-induction V4

V4 implements reset-local role tracking, bounded semantic transition events,
eight-transition windows, typed zero-support mechanic rules, separate Beta
evidence, source-only priors, outcome-blind prospective queries, balanced
chronological collection, global/local/template baselines, context, binding,
outcome, and label controls, block bootstrap, calibration, Qwen diagnostics,
and full checksummed artifacts.

Post-V4 validation:

```text
targeted SAGE12 tests: 57 passed in 12.50s
targeted Ruff checks: All checks passed!
full repository suite: 1724 passed, 1 warning in 271.84s (0:04:31)
```

The sole warning remains the harmless Joblib physical-core query fallback.

It derived 1,911 source windows and collected 768 fresh transitions yielding
576 prospective windows. Structured prediction strongly beat the local
action-only baseline, but source actor-role quality and calibration failed the
frozen conjunctive protocol. Result checksum:
`5987eb9531f568dc814dad46eb9e78d13a3813a9c30db3d6cb1fa8a319e16927`.
No world model, EBM, or controller evaluation followed.

## Clean temporal replication V4.1

V4.1 is implemented as a separate version so V4 remains reproducible. It adds
an online `translational` / `non_translational` / `ambiguous` role contract,
source leave-one-game-out Platt calibration, source-only decision thresholds,
raw-versus-calibrated replication metrics, a compact Qwen JSON compiler,
separate Qwen authority, and a per-effect V5 eligibility ledger.

The implementation, tests, protocol, and manifest are frozen before the
source preflight. No world model, EBM, or controller is fit at this checkpoint.
