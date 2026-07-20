# Adaptive Reasoning for ARC-AGI-3

**A brain-inspired agent that explores, learns, and strategises to solve unknown interactive games.**

```
Game Frame → Visual Cortex (CNN) → State Description → Goal Decomposition
→ JEPA World Model predicts outcomes → EBM scores strategies
→ Actioner executes → Associative Memory consolidates → repeat
```

## Core Principle

Each ARC-AGI-3 game is unique and its rules are unknown. The agent must:
1. **Explore** the game to discover mechanics (fast random play + systematic probing)
2. **Model** the world (latent JEPA predictions + pixel-level CNN predictions)
3. **Plan** from a small trajectory-sampling core that stays measurable as we extend it
4. **Learn online** from every action, transferring knowledge across games

## Unified Live Cognitive Path

`adaptivereasoning`, the registered competition agent, routes normal planning,
fast exploration, and procedure replay through one
`UnifiedCognitiveController` in `theory/`. The controller reuses the existing
V3/V5/theory components instead of introducing another agent architecture:

1. `LiveTransitionBeliefLoop` turns every real before/action/after frame into
   objects, affordances, a structured diff, mechanic hypotheses, and verified
   symbolic rules.
2. `DiscriminatingExperimentDesigner` probes unresolved action effects;
   `GenericDiscriminatingExperimentDesigner` chooses coordinates whose
   observable structural or relational predictions disagree.
3. Generic predictions are promoted into `GameTheory` only after repeated
   support in distinct live contexts. Confirmed directed rules are compiled
   into options that can establish a missing precondition, apply a known
   operator/action, and revise themselves from the resulting transition.
4. Option value uses directed relation/color/level progress. A raw visual
   change is recorded separately, and a mechanically true but functionally
   sterile option is quarantined instead of monopolising control.
5. `OperatorInducer` compiles repeated effects into state-conditioned
   operators. Confirmed theory rules and induced operators are planning inputs,
   not one-step score decorations.
6. V5 danger memory and anti-attractor provide observation-dominated vetoes
   for lethal actions, repeated no-ops, and low-novelty loops.
7. The former v4_1 trajectory decision remains the explicit fallback when the
   scientific path has no justified experiment or plan.

Candidate hypotheses and transferred priors never count as proof: statuses are
revised only after the corresponding live action has been observed.

The paired controller-boundary benchmark is reproducible with:

```bash
python -m theory.unified_cognition_ab_benchmark --seeds 0,1 --budget 40 --resets 2
```

It creates fresh environments for both arms and checks identical reset frames,
games, seeds, reset counts, and budgets. The current five-game public-unseen
run records 20 reset attempts per arm and 800 unified actions. SAGE.8r uses 488
deliberate experiments, learns 28 state-conditioned action models, and obtains
2 downstream pursuit-progress events, versus 0 when that directional control
is ablated. It still finds no level or win gain in either arm; this is recorded
as a negative terminal result, not relabelled as ARC progress.

## Sampler Roadmap

The current branch recentres planning around a deliberately minimal sampler:

1. `V0`: observe -> infer lightweight goal context -> sample short `heuristic` and `random` trajectories -> score them -> execute the first action -> store the observed outcome.
2. `V1`: add task-program and human-prior guidance only if it improves a tracked metric.
3. `V2`: add trajectory replay and mutation only if it beats `V1`.
4. `V3`: add level-to-level continuation only if it clearly helps again.

This keeps the planning loop readable, testable, and easy to ablate instead of mixing several special-case mechanisms at once.

The runner now exposes this explicitly via `--sampler-stage v0|v1|v2|v3`. Current practical stages are `v0` and `v1`.
Use `--planner-mode hypothesis` to switch from prior-guided action sampling to the new action-dynamics hypothesis planner.

## Architecture Overview

| Module | Role | Implementation |
|--------|------|----------------|
| **Visual Cortex** | Predict next frame from current frame + action | U-Net CNN with FiLM conditioning, ~450K params |
| **State Describer** | Frame → structured `GameObservation` | Grid analysis + object detection + player tracking |
| **Goal Decomposer** | Game → overarching goal → ordered subgoals | LLM + template fallback, game-type classification |
| **Strategy Generator** | Subgoal → candidate strategies | LLM + template fallback, action-effect aware |
| **JEPA World Model** | Predict latent consequences of strategies | Encoder + Predictor + Aux heads, ~5M params |
| **EBM Energy Scorer** | Score & rank candidate strategies | Energy-Based Model with pairwise ranking loss |
| **Actioner** | Strategy → concrete game action | Handler dispatch: navigate, click, explore, undo, sequence |
| **Associative Memory** | Brain-inspired multi-system memory | LTP/LTD associations + episodic + procedural + policy NN |
| **Game Memory** | Tracks discovered game mechanics | Action profiles, player tracking, direction mapping |
| **Cross-Game Memory** | Trust-gated meta-knowledge across games and runs | Policy NN + action priors + goal hints + failure patterns → `cross_game_memory.pt` |

## Agent Phases

```
Phase 1 — Fast Exploration (time-budgeted)
  ├─ Random play with novelty-driven clicking
  ├─ Replay winning sequences (procedural memory)
  ├─ Visual cortex trains on observed transitions (50 steps)
  └─ Associative memory consolidates episodes

Phase 2 — Strategic Play (action-budgeted)
  ├─ Observe → Decompose → Strategize → Execute → Update
  ├─ Visual cortex predictions feed into:
  │   ├─ Strategy Generator (NL descriptions in prompt)
  │   ├─ Associative Memory (change rates, directions, danger, similarity)
  │   └─ Game Memory (VC-predicted movement directions)
  ├─ JEPA predicts latent outcomes, EBM scores
  ├─ Online training (world model, EBM, visual cortex)
  └─ Subgoal budget management + re-decomposition when stuck
```

## Visual Cortex → Memory Integration

The visual cortex (CNN) feeds structured predictions into the associative memory:

| Pathway | Data | Effect |
|---------|------|--------|
| **Change rates** | Predicted % cells changed per action | Biases `pick_novel_action()` weights for untried actions |
| **Directions** | Predicted (dy, dx) displacement | Fallback for navigation when no observed player movement |
| **Danger** | Predicted destruction score | Injected into `danger_map` to avoid risky actions |
| **Similarity** | Pairwise cosine of predicted grids | Regularises policy NN to generalise across similar-effect actions |

## Project Structure

```
v4_1_reasoning_system/
├── arc_agi/                        # ARC-AGI-3 game adapter (core agent logic)
│   ├── reasoning_loop.py           # Hierarchical control loop (Observe→Explore→Decompose→Strategize→Execute→Update)
│   ├── visual_cortex.py            # CNN U-Net frame predictor with FiLM action conditioning
│   ├── associative_memory.py       # Brain-inspired memory (LTP/LTD, episodic, procedural, policy NN, cross-game)
│   ├── game_memory.py              # Per-game knowledge: action profiles, player tracking, directions
│   ├── game_world_model.py         # JEPA-style world model (latent state prediction + aux heads)
│   ├── energy_scorer.py            # EBM strategy scorer with pairwise ranking loss
│   ├── strategy_generator.py       # LLM + template strategy generation
│   ├── strategy_router.py          # Rule-based strategy candidate routing
│   ├── goal_decomposer.py          # Game-type classification + subgoal decomposition
│   ├── actioner.py                 # Strategy → concrete action (navigate, click, explore, undo)
│   ├── state_describer.py          # Frame → GameObservation (grid analysis, objects, player)
│   ├── grid_analyzer.py            # Low-level grid analysis utilities
│   └── llm_cache.py               # Deterministic LLM response caching
│
├── world_model/                    # Generic JEPA components
│   ├── encoder.py                  # State → latent z_t
│   ├── predictor.py                # JEPA transition predictor
│   └── aux_heads.py                # Auxiliary prediction heads
│
├── router/                         # Generic routing components
│   ├── candidate_generator.py      # Rule-based candidate generation
│   ├── ebm_router.py              # Energy-Based Model router
│   └── routing_train.py           # Ranking loss training
│
├── training/                       # Training scripts
│   ├── train_world_model.py       # JEPA predictor + aux heads training
│   └── train_router.py            # EBM router ranking loss training
│
└── pyproject.toml

ARC-AGI-3-Agents/
├── agents/templates/
│   └── adaptive_reasoning_agent.py # Main agent entry point (Phase 1 + Phase 2)
├── test_full_agent.py              # Multi-game test with clean summary output
├── run_training_loop.py            # Iterated training: compound cross-game memory across runs
├── test_play_and_learn.py          # Fast-play training loop (associative memory only)
├── test_single_verbose.py          # Single-game verbose debugging
└── main.py                         # Competition entry point (Swarm orchestrator)
```

## Quick Start

### Installation

```bash
cd v4_1_reasoning_system
pip install -e .
```

### Run a Multi-Game Test

```bash
# 25 games, 60s per game
python ARC-AGI-3-Agents/test_full_agent.py 25 60
```

Output includes per-game results table with timing, visual cortex steps, goal pursuit stats, wins, and cross-game memory. Memory is saved to `cross_game_memory.pt` and reloaded automatically on next run.

### Run the Training Loop (compound learning across runs)

```bash
# 5 iterations × 25 games × 60s each—memory persists between iterations
python ARC-AGI-3-Agents/run_training_loop.py 5 25 60
```

Each iteration runs the full game suite. Cross-game memory compounds: action priors, policy NN weights, and goal strategy hints carry over. A comparative diagnostic table is printed after each iteration showing trends (▲/▼) and first-to-last improvement.

### Run a Single Game (Verbose)

```bash
python ARC-AGI-3-Agents/test_single_verbose.py ls20
```

### Fast Play-and-Learn (No Strategy, Memory Only)

```bash
# 10 games, 50 iterations each, 100 actions per iteration
python ARC-AGI-3-Agents/test_play_and_learn.py 10 50 100
```

### Competition Submission

```bash
python ARC-AGI-3-Agents/main.py --agent=adaptivereasoning
```

## Hardware Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Visual Cortex (U-Net) | ~450K params | CPU or GPU, trains online |
| JEPA World Model | ~5M params | CPU or GPU |
| EBM Scorer | ~1M params | CPU |
| Policy NN (actor-critic) | ~10K params | CPU, trains online |
| LLM (goal bank gen) | 494M params (Qwen2.5-0.5B) | ~1GB fp16, GPU recommended |

**All neural components auto-detect GPU (CUDA).** With an RTX 4050 (6.4GB), total VRAM usage is ~1.1GB.

Kaggle constraints: CPU/GPU ≤ 6 hrs, no internet, pre-trained models OK.

## Key Design Decisions

- **Two-phase architecture**: Fast exploration builds a model of the game, strategic phase exploits it
- **Visual cortex as shared backbone**: CNN predictions feed into both the LLM (via NL descriptions) and the memory system (via structured analysis)
- **Brain-inspired memory**: LTP/LTD synaptic learning, not just a replay buffer — enables rapid online adaptation
- **Hierarchical goals**: Game-type classification drives subgoal decomposition, preventing aimless action
- **Sceptical cross-game transfer**: Persistent memory acts as hypothesis proposer (trust=0.3), not action governor; failure patterns, overpredicted goals, and contradicted priors are persisted alongside successes
- **Dev/competition separation**: Development mode uses full persistence; competition mode halves trust (0.15) to preserve adaptation to novel games

## License

MIT
