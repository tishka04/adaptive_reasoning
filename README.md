# v4.1 Adaptive Reasoning System

**An architecture that treats reasoning itself as a control problem.**

```
Problem → LLM parse → latent state → candidate reasoning actions
→ JEPA-style latent consequence prediction → EBM routing
→ specialized solver/tool execution → external verification
→ repair or memory update → repeat
```

## Core Principle

General intelligence is not just solving a given problem — it is **choosing the right way to think** for the problem at hand.

This system explicitly learns:
- **What** kind of reasoning to do
- **When** to do it
- **For how long** (budget)
- **With which tool** (solver selection)
- **Under which verification constraints** (strictness)

## Architecture Overview

| Module | Role | Implementation |
|--------|------|----------------|
| **Semantic Parser** | NL → structured task | 7-8B instruct LLM (Mistral/Llama/Qwen) |
| **State Encoder** | Context → latent z_t | MLP, ~5-20M params |
| **Candidate Generator** | Propose reasoning actions | Rule-based (Phase 1), learnable later |
| **JEPA World Model** | Predict latent consequences | Transition predictor + aux heads, ~20-100M params |
| **EBM Router** | Score & select best action | Energy-Based Model, ~10-50M params |
| **Solver Portfolio** | Execute reasoning | Hierarchical, Global Opt, Repair, LLM Codegen |
| **Verifier** | Ground truth checking | Domain-specific (planning, scheduling, optimization, code) |
| **Memory** | Long-term adaptation | Episodic replay + structural patterns (FAISS) |

## Reasoning Modes

| Mode | Use Case | Backend |
|------|----------|---------|
| `hierarchical` | Subgoal decomposition, dependency planning | Topological sort + recursive solve |
| `global_opt` | Coupled constrained optimization | OR-Tools CP-SAT |
| `repair` | Fix violations, restore feasibility | Targeted patch / relaxation / local search |
| `llm_codegen` | Code synthesis, critique, repair | Instruction-tuned LLM |

## Project Structure

```
v4_1_reasoning_system/
├── parser/
│   ├── task_schema.py          # Pydantic task representation
│   └── llm_parser.py           # LLM-based semantic parser
├── world_model/
│   ├── encoder.py              # State → latent z_t
│   ├── predictor.py            # JEPA transition predictor + action encoder
│   └── aux_heads.py            # Validity, score, cost, repair prediction
├── router/
│   ├── candidate_generator.py  # Rule-based candidate generation
│   ├── ebm_router.py           # Energy-Based Model router + rule fallback
│   └── routing_train.py        # Ranking loss training for EBM
├── solvers/
│   ├── base.py                 # BaseSolver interface + SolverResult
│   ├── hierarchical_solver.py  # Subgoal decomposition
│   ├── global_opt_solver.py    # OR-Tools CP-SAT wrapper
│   ├── repair_solver.py        # Violation-targeted / relaxation / local search
│   └── llm_codegen_solver.py   # LLM code generation + execution
├── verifier/
│   ├── base.py                 # BaseVerifier + VerificationResult
│   ├── planning_verifier.py    # Dependency, cycle, coverage checks
│   ├── scheduling_verifier.py  # Precedence, resource, makespan checks
│   ├── dispatch_verifier.py    # Bounds, constraints, objective checks
│   └── code_verifier.py        # Syntax, execution, test case checks
├── memory/
│   ├── replay.py               # Episodic trajectory replay buffer
│   └── retrieval.py            # FAISS episodic retriever + structural memory
├── orchestration/
│   ├── controller.py           # Main reasoning control loop
│   └── state_update.py         # State lifecycle management
├── training/
│   ├── collect_rollouts.py     # Synthetic problem gen + rollout collection
│   ├── train_world_model.py    # JEPA predictor + aux heads training
│   └── train_router.py         # EBM router ranking loss training
├── pyproject.toml
└── setup.py

notebooks/
├── 01_demo_inference.ipynb     # Colab demo: run the full loop
└── 02_training_pipeline.ipynb  # Colab training: Phases 2-5
```

## Training Order

| Phase | What | Details |
|-------|------|---------|
| **1** | Build solvers & verifier | Make each mode work individually |
| **2** | Rule-based router | Get the full loop running end-to-end |
| **3** | Train JEPA-lite world model | Predict latent next state + aux targets |
| **4** | Train router EBM | Ranking loss on good vs bad reasoning actions |
| **5** | Add top-k routing | Evaluate multiple candidates under verification |
| **6** | Add stronger LLM usage | Code gen, critique, solver formulation, repair |

## Quick Start

### Installation

```bash
cd v4_1_reasoning_system
pip install -e .
```

### Minimal Example (No GPU / No LLM)

```python
from v4_1_reasoning_system.orchestration.controller import ReasoningController

controller = ReasoningController(
    use_learned_router=False,
    max_iterations=5,
    device="cpu",
)

# Solve a scheduling problem with pre-parsed input
result = controller.solve_from_dict(
    problem="Schedule 3 jobs to minimize makespan",
    parsed_json={
        "domain": "scheduling",
        "description": "Schedule 3 jobs",
        "entities": ["job_0", "job_1", "job_2"],
        "constraints": [],
        "objective": {"description": "Minimize makespan", "sense": "minimize"},
        "ambiguities": [],
        "structured_data": {
            "type": "scheduling",
            "jobs": [
                {"name": "job_0", "duration": 3, "dependencies": []},
                {"name": "job_1", "duration": 5, "dependencies": ["job_0"]},
                {"name": "job_2", "duration": 2, "dependencies": ["job_0"]},
            ],
            "horizon": 20,
            "resources": {},
        },
    },
)

print(f"Valid: {result['valid']}, Score: {result['score']:.3f}")
print(f"Solution: {result['solution']}")
```

### Google Colab

1. Upload `v4_1_reasoning_system/` to Google Drive
2. Open `notebooks/01_demo_inference.ipynb` for inference demo
3. Open `notebooks/02_training_pipeline.ipynb` for training

### Collect Rollouts & Train

```python
from v4_1_reasoning_system.training.collect_rollouts import RolloutCollector

# Generate synthetic problems
problems = RolloutCollector.generate_benchmark_suite(
    n_planning=20, n_scheduling=20, n_optimization=20,
)

# Collect reasoning trajectories
collector = RolloutCollector(controller)
results = collector.collect(problems)

# Train world model (Phase 3)
from v4_1_reasoning_system.training.train_world_model import (
    WorldModelTrainer, build_world_model_dataset, WorldModelDataset
)
from torch.utils.data import DataLoader

records = build_world_model_dataset(controller.replay_buffer)
loader = DataLoader(WorldModelDataset(records), batch_size=16, shuffle=True)
trainer = WorldModelTrainer(controller.transition_predictor, controller.aux_heads)
for epoch in range(100):
    metrics = trainer.train_epoch(loader)

# Train router EBM (Phase 4)
from v4_1_reasoning_system.training.train_router import train_router_from_buffer

result = train_router_from_buffer(
    controller.ebm_router, controller.replay_buffer, num_epochs=100
)

# Switch to learned routing
controller.use_learned_router = True
```

### Save / Load Checkpoints

```python
controller.save_checkpoint("/path/to/checkpoint")
controller.load_checkpoint("/path/to/checkpoint")
```

## Hardware Requirements

| Component | Size | Notes |
|-----------|------|-------|
| LLM (parser/codegen) | 7-8B params | 4-bit quantized, ~5GB VRAM |
| State Encoder | ~5-20M params | Negligible |
| World Model | ~20-100M params | < 1GB |
| Router EBM | ~10-50M params | < 0.5GB |
| Memory (FAISS) | CPU/RAM | Scales with trajectory count |

**Total: fits comfortably on a single A100 (40/80GB) or H100.**

## Benchmark Domains

| Domain | Task Examples |
|--------|--------------|
| **Planning** | Key-door, resource assembly, tool-use planning |
| **Scheduling** | Job-shop, precedence scheduling, fairness constraints |
| **Optimization** | Knapsack, energy dispatch, network flow |
| **Coding** | Generate solver code, patch failing code, pass tests |

## License

MIT
