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

## GPU decision

No empirical SAGE12 model was trained in this implementation task, so using
the laptop GPU would not have accelerated meaningful training. The only
PyTorch run was a tiny unit-level optimization check. The local proposal
backend and future EBM training both support automatic CUDA placement, with
the reproducibility/timing rules frozen in `training/SAGE12_DATA_POLICY.md`.

## Authority result

No proposal, world-model, energy, shadow, bounded, active, or holdout gate has
been evaluated. The integrated default remains `off`; bounded and active
downgrade to shadow without all prerequisite gates. This report makes no
claim that SAGE12 improves game performance or cross-game generalization.
