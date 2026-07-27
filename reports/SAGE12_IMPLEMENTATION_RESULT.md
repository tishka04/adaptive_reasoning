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
