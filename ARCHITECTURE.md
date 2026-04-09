# Architecture: Adaptive Reasoning as a Control Problem

## Core Principle

General intelligence is not just solving a given problem. It is **choosing the right way to think** for the problem at hand.

This architecture treats reasoning itself as a control problem:

```
problem state → choose reasoning action → observe verified outcome → choose next reasoning action
```

The system learns:
- **What** kind of reasoning to do
- **When** to do it
- **For how long** (budget)
- **With which tool** (solver selection)
- **Under which verification constraints** (strictness)

## The Ingredients and Their Roles

### A. LLM — Reasoning Operator (not unquestioned brain)

Used for:
- Parsing natural-language problems into structured objects
- Proposing decompositions
- Generating code or solver formulations
- Interpreting errors
- Proposing repairs
- Summarizing what happened

**Not relied on for guaranteed correctness.**

### B. EBM Reasoning Router

Energy-Based Model that scores candidate reasoning actions.

Each candidate is a tuple: `r = (mode, budget, strictness, tool_hint)`

Example candidates:
- `(llm_codegen, medium, verified, make_solver_skeleton)`
- `(global_opt, high, strict, feasibility_priority)`
- `(repair, low, verified, local_patch)`
- `(hierarchical, medium, fast, depth_2)`

**Low energy = "This is a promising way of thinking next."**

### C. JEPA-Style World Model

Predicts the latent future state of the reasoning process after a candidate action:

```
ẑ_{t+1} = P(z_t, r_t)
```

Does NOT reconstruct raw tokens or full observations. Predicts latent consequences:
- Expected feasibility
- Expected progress
- Expected regret
- Expected need for repair
- Expected compute cost

This gives the router **counterfactual foresight**.

### D. Dedicated Tools (Solver Portfolio)

| Mode | Use Case | Backend |
|------|----------|---------|
| `hierarchical` | Subgoal sequencing, decomposable planning | Topological sort + recursive solve |
| `global_opt` | Coupled constrained problems | OR-Tools CP-SAT |
| `repair` | Patching infeasible/failing outputs | Targeted/relaxation/local search |
| `llm_codegen` | Code synthesis, critique, repair | Instruction-tuned LLM |

### E. Verifier — The Real Source of Truth

Checks:
- Tests pass or fail
- Constraints satisfied or violated
- Feasibility holds or not
- Objective quality
- Runtime behavior
- Safety / consistency conditions

**The verifier prevents elegant nonsense.**

### F. Memory

**Episodic memory:** Previous reasoning trajectories, mode choices, failures, successful patches, solver outcomes.

**Structural memory:** Reusable templates, known formulations, common failure patterns, latent priors over domains and solver choices.

## Module Architecture

### Module 1 — Semantic Parser
- **Input:** User problem + optional structured input + previous context
- **Output:** Structured TaskObject (domain, constraints, objective, entities, ambiguities)
- **Implementation:** Compact instruction-tuned LLM (7-8B)

### Module 2 — State Encoder
- **Input:** Parsed task + partial solution + verifier feedback + recent history
- **Output:** Latent reasoning state `z_t`
- **Implementation:** MLP, ~5-20M parameters

### Module 3 — Candidate Reasoning Generator
- **Output:** 4-8 candidates with metadata (mode, budget, strictness, tool hint)
- **Implementation:** Rule-based (Phase 1), learnable later

### Module 4 — JEPA-Style Transition Predictor
- **Input:** `z_t` + candidate action `r_i`
- **Output:** Predicted `ẑ_{t+1}` + auxiliary predictions (validity, score, cost, repair need)
- **Implementation:** Gated residual MLP, ~20-100M parameters

### Module 5 — Router EBM
- **Input:** `(z_t, r_i, ẑ_{t+1})`
- **Output:** Energy score (lower = better)
- **Implementation:** 2-4 layer MLP, ~10-50M parameters

### Module 6 — Solver Portfolio
- Selected candidate activates one solver
- Each solver implements the `BaseSolver` interface

### Module 7 — Verifier
- Domain-specific verification (planning, scheduling, optimization, code)
- Returns `VerificationResult` with validity, feasibility, violations, scores

### Module 8 — Memory + Replay
- Stores full reasoning trajectories
- Updates router EBM and world model via training
- Enables long-term improvement

## Full Control Loop

```
1. Parse the problem into structured form
2. Encode current reasoning state into latent z_t
3. Generate candidate reasoning actions
4. Predict latent consequences with the world model
5. Score candidates with the router EBM
6. Execute the best one (or top-k few)
7. Verify outputs externally
8. Update state and memory
9. Repeat until verified success or budget exhaustion
```

## Training Order

| Phase | What | Details |
|-------|------|---------|
| **1** | Build solvers & verifier | Make each mode work individually on at least one domain |
| **2** | Rule-based router | Get the full loop running end-to-end |
| **3** | Train JEPA-lite world model | Predict latent next state, validity, score gain, compute cost |
| **4** | Train router EBM | Ranking loss on good vs bad reasoning actions |
| **5** | Add top-k routing | Evaluate multiple promising candidates under verification |
| **6** | Add stronger LLM usage | Code gen, critique, solver-formulation proposals, repair prompts |

## Hardware Budget (Single A100/H100)

| Component | Size | VRAM |
|-----------|------|------|
| LLM (7-8B, 4-bit quantized) | ~5GB | Inference only |
| State Encoder | ~5-20M params | Negligible |
| World Model (predictor + aux heads) | ~20-100M params | < 1GB |
| Router EBM | ~10-50M params | < 0.5GB |
| Memory (FAISS + replay buffer) | CPU/RAM | Scales with data |

## Minimal Mode Set

Four modes for the first prototype:
1. `hierarchical` — subgoal decomposition
2. `global_opt` — constrained optimization (CP-SAT)
3. `repair` — fix violations
4. `llm_codegen` — code generation / critique

## Benchmark Domains

| Domain | Examples |
|--------|----------|
| **Hierarchical dependency** | Key-door, resource assembly, tool-use planning |
| **Constraint satisfaction / scheduling** | Assignment, precedence scheduling, fairness constraints |
| **Optimization-heavy** | Energy dispatch, inventory allocation, network flow |
| **Verified coding** (optional) | Generate solver code, patch failing code, pass tests |

## Why This Is Promising

This architecture attacks the real problem: choosing the right way to think.

It explicitly supports:
- Multiple reasoning styles
- Solver selection
- Verification
- Repair
- Memory
- Latent anticipation of reasoning consequences

That is much closer to robust cognition than one-shot generation.

## Summary

> A promising path toward more general intelligence is an adaptive solver
> architecture in which an LLM proposes structured reasoning actions, a
> JEPA-style world model predicts their latent consequences, an EBM router
> scores and selects among them, specialized tools execute them, a verifier
> provides ground truth, and memory supports long-term adaptation.
