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

SAGE.10b-SAGE.10e extend branch-local delayed credit with bounded sub-effect
relay, generalized stall detection, per-level explorer re-arming, and verified
per-level route shortening. The compact paired proof is reproducible with:

```bash
ARC-AGI-3-Agents/.venv/Scripts/python.exe -m theory.sage10b_plus_benchmark
```

All four procedural gates pass against isolated ablations. Eligibility
identity survives at most three structurally or causally linked sub-effects;
the three new stall signals do not require exact state recurrence; a level
change only re-arms the retired explorer after a new stall; and shorter routes
remain candidate-only until their own live terminal replay.

SAGE.10g-SAGE.10i and SAGE.11 add the frozen multi-source curriculum and a
firewalled neuro-symbolic world-model path. The neural path defaults to `off`;
`shadow` executes the byte-identical symbolic action, and bounded/active
authority remains inaccessible until the pre-registered source, productivity,
calibration, inference-cost, and paired holdout gates pass. Observed symbolic
danger is always a hard veto, protected terminal competence remains supreme,
and every neural hypothesis enters with support zero.

The implementation, split registry, data policy, model card, and validation
protocol are documented in:

- `theory/sage11/README.md`
- `training/SAGE11_DATA_POLICY.md`
- `models/SAGE11_MODEL_CARD.md`
- `reports/SAGE11_VALIDATION_PROTOCOL.md`
- `reports/SAGE11_EFFECT_PILOT_RESULT.md`
- `reports/SAGE11_EFFECT_PILOT_V2_PROTOCOL.md`
- `reports/SAGE11_EFFECT_PILOT_V2_RESULT.md`

The environment/model audit is reproducible with:

```bash
ARC-AGI-3-Agents/.venv/Scripts/python.exe -m theory.sage11.audit
```

The resumable real-environment source collection and SAGE.10g curriculum
freeze are reproducible with:

```bash
ARC-AGI-3-Agents/.venv/Scripts/python.exe -m theory.sage11.source_dataset_runner --workers 8
```

It attempts to publish exactly 100,000 deduplicated rows in separate
source-train and source-validation shards, verifies every row count and
checksum, rotates the five registered seeds in deterministic 200-reset windows
with independent controllers and saturation counters, resumes only
checksum-verified partial shards, and never touches the neural holdout or
historical report-only games. The first
five-seed capacity run failed closed at an optimistic maximum of 98,708 rows
under the base 8,000/game cap. The user then approved the minimum 1,292-row
global overflow on proven high-capacity training games; see
`reports/SAGE11_SOURCE_CAPACITY_RESULT.md`.

The amended run is complete: 100,000 verified rows (76,908 train / 23,092
validation), manifest
`d4fd8210f2015c00b906cdd98e01630b309deefa7cd9498b38aba8e55130fa1b`,
and an 11-source frozen curriculum
`d11948c5cfcb70ce888b435d63d217b95ce2a0006e4423ae7ac70374d81c630c`.
The terminal head remains disabled because the corpus contains 44 strong
events, below its pre-registered threshold of 100.

The source-only cheap effect pilot is reproducible with:

```bash
ARC-AGI-3-Agents/.venv/Scripts/python.exe -m theory.sage11.effect_pilot_runner
```

It failed closed: 0.0779 classifier macro-F1 versus 0.0490 for the train-only
per-action majority baseline, a gain of +0.0288 rather than the required
+0.10. All three source-validation games failed independently and
within-game action shuffling degraded macro-F1 by only +0.0059. No graph-model
training or holdout evaluation followed. The checksummed result is documented
in `reports/SAGE11_EFFECT_PILOT_RESULT.md`.

The separately pre-registered factorized follow-up is reproducible with:

```bash
ARC-AGI-3-Agents/.venv/Scripts/python.exe -m theory.sage11.factorized_effect_pilot_runner
```

Pilot v2 formally passed its frozen cheap gate: full composite macro-F1
0.5506 versus 0.3431 for a learned action-only comparator, a +0.2075 gain,
with non-negative gains on all three source-validation games. Player movement
supplied nearly all the improvement, changed-cells F1 remained 0.1562, and
action shuffling degraded the composite by only 0.0078.

The passing representation is now a shared, versioned 77-feature interface
used identically by source-row loading and live counterfactual inference
(schema checksum
`39bb692848fba64ef994e0c0a304785128e1a69adaf6308f1d22623a8f0876bd`).
The 1,552,178-parameter world model now has separate changed-cells and
player-moved heads. The pre-registered source-train-only leave-one-game-out
anti-shortcut audit then failed: changed-cells was 0.1026 below the stronger
action/state baseline, conditional action-shuffle degradation was only
0.0180, and fixed signatures predicted game identity with 99.17% accuracy.
GPU training was blocked, which activated a smaller corpus preserving
contact, alignment, proximity, and object-relative action features—not another
100,000-row recollection. That replacement was frozen before collection:
10,027 source-train rows and 52 versioned relation columns, followed by the
same LOGO/action-shuffle gates plus an explicit requirement that relations add
at least +0.05 changed-cells F1. See
`reports/SAGE11_ANTI_SHORTCUT_AUDIT_RESULT.md` and
`reports/SAGE11_RELATIONAL_PILOT_PROTOCOL.md`.

The relational collection now verifies at exactly 10,027 rows, manifest
checksum
`11a734063ac4be4b8cece50a4d6e7ee40bb25ccfacbc8cd703a1565845f39f2c`.
The frozen relational fit then failed all four gates: changed-cells
full-minus-best-baseline -0.0059, conditional shuffle degradation 0.0048,
relations contribution -0.1202, and only 6/11 non-negative folds. Result
checksum:
`272a327ab523a4f81f887e69d381d66c33b31d014bac515347f39e197b31177b`.
The current world-model track stops without GPU training or shadow
evaluation. See `reports/SAGE11_RELATIONAL_COLLECTION_RESULT.md` and
`reports/SAGE11_RELATIONAL_PILOT_RESULT.md`.

SAGE12 implements the higher-semantic replacement path without reopening the
failed SAGE11 gate. A local open-weight LLM proposes strict typed hypotheses;
a deterministic compiler grounds structural roles and legal actions; a small
semantic world model rolls out bounded trajectories; an auditable energy
function ranks them; and a hierarchical controller can execute only the first
action before replanning. LLM proposals always have `support=0`, observed
transitions are the only evidence, and symbolic danger/protected competence
remain hard vetoes.

The implementation is integrated but defaults to `off`. The frozen Stage A
pilot collected 2,104 source-only executed traces and ran 224 Qwen2.5 0.5B
generations on the laptop GPU after measuring a 3.808x median speedup over
CPU. It failed closed: strict JSON and grounded recall were zero, the
action-only baseline reached 0.895 recall, relation-shuffle degradation was
zero, and the compact scene signature identified source-training games at
99.94% accuracy. No semantic world model or EBM was fit, and no performance
promotion is claimed.

The constrained V2 repair then passed strict JSON, support-zero, grounding,
and both reduced-leakage gates. It still failed closed: Qwen-head primary
macro-F1 was 0.484 versus 0.549 for action-only, relation shuffling improved
rather than degraded the score, and `re86` transferred at -0.237. This
isolates the remaining problem as missing action-target causal information,
not output syntax. See:

V4.6 now tests the global architecture instead of applying another
component-promotion gate. It composes the real hypothesis compiler, a
cross-game semantic world model, heuristic and learned trajectory energy, and
the receding-horizon controller over 340 replay-verified V4.3 action trees.
An explicit oracle ladder localizes the first loss of utility. Qwen is frozen
on 44 outcome-blind roots with original and relation-shuffled prompts; all
work is source-only and offline, and the pilot cannot grant live authority.
See `reports/SAGE12_GLOBAL_INTEGRATION_PILOT_V4_6_PROTOCOL.md`.

V4.7 repaired candidate coverage and localized the remaining bottleneck:
zero-shot Qwen semantics lost 0.4373 utility to the fold-selected baseline,
while true semantic annotations gained 0.3978 and true world outputs let the
learned EBM exactly match the oracle.

V4.8 then trained a source-only, same-prestate paired semantic adapter over a
frozen Qwen representation and ran it through the unchanged V4.7 world model,
EBM, and controller. It improved structured-only by +0.2068 and improved V4.7
zero-shot by +0.1938, proving the adapter carries some useful signal. It still
lost 0.2435 to action-sequence-only, was non-negative on 5/11 games, selected
0/3 completion trajectories, and leaked game identity at 85.61%. True
annotations remained +0.3978 above baseline. This rejects the present
frozen-Qwen external-adapter route without refuting the conditional global
architecture. See
`reports/SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_RESULT.md`.

The completed V4.6 result is negative for the current learned chain but does
not refute the global architecture under its frozen rule. The oracle pipeline
preserved 100% of best actions and gained +0.666 utility over the stronger
simple baseline, while the myopic oracle left 0.312 utility on the table. The
first collapse was Qwen proposal/grounding: only 11/44 original roots yielded
a normalized grounded option, and even oracle ranking of those options scored
-0.284 below the stronger same-root baseline. The full chain scored -0.386,
relation shuffling improved it by 0.094, and game identity remained perfectly
decodable. No authority was promoted. See
`reports/SAGE12_GLOBAL_INTEGRATION_PILOT_V4_6_RESULT.md`.

- `theory/sage12/README.md`
- `training/SAGE12_DATA_POLICY.md`
- `models/SAGE12_MODEL_CARD.md`
- `reports/SAGE12_VALIDATION_PROTOCOL.md`
- `reports/SAGE12_IMPLEMENTATION_RESULT.md`
- `reports/SAGE12_PROPOSAL_PILOT_RESULT.md`
- `reports/SAGE12_CONSTRAINED_PILOT_V2_RESULT.md`
- `reports/SAGE12_GLOBAL_INTEGRATION_PILOT_V4_6_PROTOCOL.md`
- `reports/SAGE12_GLOBAL_INTEGRATION_PILOT_V4_6_RESULT.md`
- `reports/SAGE12_CANDIDATE_COMPLETE_SLOT_PILOT_V4_7_RESULT.md`
- `reports/SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_PROTOCOL.md`
- `reports/SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_SEMANTIC_RESULT.md`
- `reports/SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_RESULT.md`
- `reports/SAGE12_GOAL_CONDITIONED_TRAJECTORY_VALUE_V4_18_PROTOCOL.md`
- `reports/SAGE12_GOAL_CONDITIONED_TRAJECTORY_VALUE_V4_18_RESULT.md`
- `reports/SAGE12_TOPOLOGICAL_CAUSAL_INVARIANTS_V4_19_PROTOCOL.md`
- `reports/SAGE12_TOPOLOGICAL_CAUSAL_INVARIANTS_V4_19_RESULT.md`

Focused software validation:

```bash
python -m pytest -q tests/test_sage12_semantic_planning.py
python -m pytest -q tests/test_sage12_proposal_pilot.py
python -m pytest -q tests/test_sage12_constrained_pilot.py
python -m pytest -q tests/test_sage12_integration_pilot.py
python -m pytest -q tests/test_sage12_semantic_adapter_v4_8.py
```

The long-budget performance track skips ablation overhead and writes compact
level/WIN/action-efficiency history:

```bash
ARC-AGI-3-Agents/.venv/Scripts/python.exe -m theory.benchmark_score_runner --label sage10b-plus
```

Its defaults are the five `public_unseen` games, seeds 0/1, budgets
500/1500/4000, and 14 resets. Scientific attribution remains available in
the unified benchmark through schema `v42` and the isolated flags
`--disable-subeffect-eligibility-relay`,
`--disable-generalized-frontier-stall-detection`,
`--disable-per-level-frontier-rearming`,
`--disable-level-route-memory`, and
`--disable-level-route-shortening`.

It creates fresh environments for both arms and checks identical reset frames,
games, seeds, reset counts, and budgets. The current five-game public-unseen
run records 20 reset attempts per arm and 800 unified actions. SAGE.8r uses 488
deliberate experiments, learns 28 state-conditioned action models, and obtains
2 downstream pursuit-progress events, versus 0 when that directional control
is ablated. SAGE.8s can persist after such progress and compose an observed
mode bridge with its progressive follow-up, but no profitable bridge is
available in the current five-game run; its held-out metrics therefore remain
identical to SAGE.8r. SAGE.8t replaces color-only click aliases with concrete
entity/role slots plus position-invariant transfer classes: the same run learns
34 concrete identities, 28 structural classes, and makes 2 structurally
transferred selections without changing the 782 objective reductions. It still
finds no level or win gain in either arm; this is recorded as a negative
terminal result, not relabelled as ARC progress.

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

### SAGE.T12.5b.4 target-local short-program utility

T12.5b.4 is a separately frozen, two-phase source-train experiment following
the negative T12.5b.3 result. It calibrates all fixed length-2/3 programs from
one exact non-terminal detour context on lineage 8701, treats candidate
terminal outcomes as risk evidence, and permits a four-trial lineage-8705
evaluation only after a signed calibration pass. See
`reports/SAGE_T12_5B_4_LOCAL_PROGRAM_UTILITY_PROTOCOL.md` and
`reports/SAGE_T12_5B_4_LOCAL_PROGRAM_UTILITY_RUNBOOK.md`.

## License

MIT
