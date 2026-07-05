# Architecture: Adaptive Reasoning for ARC-AGI-3

## Overview

This system solves unknown interactive ARC-AGI-3 games by combining:
- **Fast exploration** to discover game mechanics
- **Mechanism-driven action selection** — learned action effects are the primary decision signal
- **Two world models** (latent JEPA + pixel-level CNN) for prediction (EBM used as tiebreaker only)
- **Brain-inspired memory** for online learning and cross-game transfer
- **Goal-directed pursuit** with hypothesis testing, binary progress measurement, and retry logic

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Goal-Directed Adaptive Loop                         │
│                                                                      │
│   ε = f(knowledge, iteration)                                        │
│                                                                      │
│   ε high (early)                    ε low (knowledgeable)            │
│   ┌──────────────────────┐          ┌───────────────────────┐       │
│   │ Informed exploration  │  ε decay │ Goal Pursuit           │       │
│   │ • GameMemory-ranked   │◀────────▶│ ┌─ Goal Bank (3-6)    │       │
│   │ • Mechanism-locked    │          │ ├─ Select best goal   │       │
│   │ • Procedure replay    │          │ ├─ Action-effect score│       │
│   │ • Continuous VC train │          │ │  (EBM = tiebreaker) │       │
│   └──────────┬───────────┘          │ ├─ Execute + Measure  │       │
│              │                      │ └─ Retry / Switch /   │       │
│              │   ε hard shift:      │    Regenerate bank    │       │
│              │   level done → 0.10  └──────────┬────────────┘       │
│              │   knowledge≥0.6 → 0.25          │                    │
│              └──────────────┬────────────────────┘                    │
│                             ▼                                        │
│              ┌──────────────────────────┐                            │
│              │ GameMemory + Assoc + VC  │                            │
│              │ (mechanism lock → ε ↓↓)   │                            │
│              └──────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘
```

## Components

### A. Visual Cortex — CNN Frame Predictor

**File:** `arc_agi/visual_cortex.py` (~450K params)

Predicts the next game frame given the current frame and an action. Operates in **pixel space** (unlike the JEPA world model which operates in latent space).

```
Input: one-hot grid (B, 16, 64, 64) + action features (B, 10)
  → Encoder: 3 ConvBlocks (16→32→64→128) with MaxPool
  → FiLM bottleneck: action embedding conditions feature maps (scale + shift)
  → Decoder: 3 upsampling stages with skip connections
  → Residual shortcut: learns delta from input
Output: per-pixel class logits (B, 16, 64, 64)
```

**Loss:** Per-pixel cross-entropy with 5× weight on changed cells.

**Outputs feed into 5 downstream consumers:**

| Consumer | Data format | Integration point |
|----------|-------------|-------------------|
| Strategy Generator | NL descriptions in `GameObservation.visual_cortex_summary` | LLM prompt context |
| AssociativeMemory (change rates) | `{act: (count, changes)}` | `pick_novel_action()` weights |
| AssociativeMemory (directions) | `{act: (dy, dx)}` | Navigation fallback |
| AssociativeMemory (danger) | `{act: danger_score}` | `danger_map` entries |
| AssociativeMemory (similarity) | `{(act_i, act_j): cosine}` | Policy NN regularisation |

### B. JEPA World Model — Latent Consequence Predictor

**File:** `arc_agi/game_world_model.py` (~5M params)

Predicts the **latent** next state from the current state and a strategy embedding. Follows the JEPA principle: predict abstract consequences, not raw observations.

```
GameStateEncoder: grid (CNN) + context → z_t
StrategyEncoder: strategy text → embedding
GamePredictor: (z_t, strategy_emb) → ẑ_{t+1}   (gated residual MLP)
GameAuxHeads: ẑ_{t+1} → validity, score gain, compute cost, repair probability
```

Used during STRATEGIZE phase to give the EBM scorer counterfactual foresight.

### C. EBM Energy Scorer (tiebreaker role)

**File:** `arc_agi/energy_scorer.py`

Scores candidate strategies by energy: **lower energy = more promising**. **Downgraded:** EBM now contributes only **10%** of the strategy selection score. The primary signal is action-effect quality from GameMemory (40%) and mechanism-lock bonuses (25%). Online EBM training is **disabled by default** (`LoopConfig.train_ebm_online = False`) to reduce learning noise.

```
Input: (z_t, strategy_embedding, ẑ_{t+1}, aux_predictions)
Output: scalar energy score
Weight in strategy selection: 10% (was 100%)
```

### D. Associative Memory — Brain-Inspired Multi-System Memory

**File:** `arc_agi/associative_memory.py`

Four memory subsystems inspired by neuroscience:

| System | Mechanism | Purpose |
|--------|-----------|---------|
| **Semantic** | (state_features, action) → strength via LTP/LTD | Cue-based action retrieval |
| **Episodic** | Full episode recordings with rewards | Experience replay |
| **Procedural** | Winning action sequences ranked by length | Sequence replay |
| **Policy NN** | Actor-critic network (6 features → 7 actions) | Learned exploration policy |

**Consolidation:** After each episode, winning actions are potentiated (LTP), failing actions are depressed (LTD), and weak associations decay.

**Visual cortex integration:** The `ingest_vc_predictions()` method accepts structured analysis from the CNN and distributes it across all four pathways:

1. **Change rate priors** — injected as virtual observations into `_vc_change_prior`, used by `pick_novel_action()` when real observation count is low
2. **Direction predictions** — stored in `_vc_directions`, surfaced via `get_vc_direction()` as navigation fallback
3. **Danger scores** — written to `danger_map` keyed by state features (capped at 0.5 to not overrule observed deaths)
4. **Similarity regularisation** — stored in `_vc_action_similarity`, applied during `_train_on_episode()` as a loss term penalising divergent logits for actions with cosine > 0.95

### E. Game Memory — Per-Game Mechanical Knowledge (primary decision maker)

**File:** `arc_agi/game_memory.py`

Tracks discovered game mechanics and is now the **primary action selection signal**:

- **Action profiles**: per-action stats (times tried, change rate, death rate, displacement)
- **Player tracking**: identified player value, position, size, confidence
- **Direction mapping**: action → (dy, dx), priority: profile > observed > VC predicted
- **Click tracking**: effective click values, game-over positions
- **State visit counts**: for novelty detection

#### Mechanism Locking

Actions with consistent, well-understood behaviour get "locked" — frozen for deterministic exploitation. An action is locked when tried ≥5 times with change_rate ≥0.80 or ≤0.15.

```python
locked = memory.get_locked_mechanisms()
# → {"ACTION1": "move_up", "ACTION3": "move_left", "ACTION5": "no_effect"}
```

Locked mechanisms provide a **0.25 bonus** in strategy scoring for strategies that use them.

#### Action Scoring

`score_action(name)` — primary signal for both exploration and strategy selection:

```
score = change_rate − death_rate × 0.8 + consistency_bonus + win_bonus
```

| Component | Value | When |
|-----------|-------|------|
| `change_rate` | 0–1 | Action reliably changes the grid |
| `death_rate × 0.8` | 0–0.8 | Penalty for lethal actions |
| `consistency_bonus` | +0.2 | Action tried ≥3× with stable rate (≥0.8 or ≤0.1) |
| `win_bonus` | +0.3 | Action has led to level completion |

`rank_actions(available)` — sorted by score, used in exploration loop (70% of the time when knowledge ≥ 0.4).

### F. Cross-Game Memory

**Class:** `CrossGameMemory` in `associative_memory.py`

Persists between games **and between test runs** via disk serialisation (`cross_game_memory.pt`). Cross-run priors act as **hypothesis proposers, not action governors** — gated by a trust system that starts low and requires in-game validation.

#### Trust System

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `INITIAL_TRUST` | 0.30 | How much to trust persistent priors at game start |
| `TRUST_DECAY_ON_DISAGREE` | 0.50 | Halve trust when current game contradicts |
| `TRUST_GROWTH_ON_AGREE` | 0.10 | Slow climb when current game confirms |
| `MAX_TRUST` | 0.70 | Priors never dominate in-game evidence |

In **competition mode**, initial trust is halved again (0.15) to preserve adaptation-to-novelty.

#### What transfers (success memory)

| Knowledge | How | Trust-gated |
|-----------|-----|-------------|
| Policy NN weights | `state_dict` deep copy | NN blend = raw_blend × trust |
| Action effect priors | EMA of change/death rates | Prior counts scaled by trust |
| Goal strategy hints | Top-3 lightweight dicts per normalised goal type | Seeded as read-only hypotheses |
| Training step count | Blend factor continuity | Indirect via NN blend |

#### What transfers (failure memory)

Remembering what *not* to trust is as important as remembering what worked.

| Knowledge | What it captures |
|-----------|------------------|
| `failure_patterns` | Goal types that gave proxy progress but no wins |
| `overpredicted_goals` | Goal families frequently over-predicted by LLM |
| `contradicted_priors` | Action priors that were contradicted in-game (reduces their count) |

#### Diagnostics-only counters

`games_played` and `games_won` are stored for reporting only. They **never** influence the control loop — a system should not become more self-confident merely because it has run many times.

#### Modes

| Mode | Trust | Use case |
|------|-------|----------|
| `development` | 0.30 | Repeated runs on public games to discover robust modules |
| `competition` | 0.15 | Novel games; heavily regularised to preserve adaptation |

Per-game state (associations, episodes, procedures, danger map, strategy outcomes) always resets.

**Persistence API:**

```python
cross_game.save("cross_game_memory.pt")
cross_game = CrossGameMemory.load("cross_game_memory.pt")
```

### G. Goal Decomposer

**File:** `arc_agi/goal_decomposer.py`

Classifies game type and produces two kinds of output:

1. **Single-goal decomposition** (`decompose()`) — hierarchical subgoals for a single best-guess goal
2. **Goal bank generation** (`generate_goal_bank()`) — 3-6 competing objective hypotheses, each with measurable progress signals

```
GameObjective (goal bank entry)
├── id: str (e.g. "obj_navigate_green")
├── description: str
├── success_condition: str
├── progress_signals: List[ProgressSignal]   ← measurable indicators
├── anti_signals: List[str]                  ← wrong-direction indicators
├── confidence: float (LLM's initial belief)
└── status: ACTIVE / SUSPENDED / REJECTED / CONFIRMED
```

LLM + template fallback. Re-generates goal bank when all goals are exhausted.

### H. Strategy Generator

**File:** `arc_agi/strategy_generator.py`

Two modes:

1. **Standard** (`generate()`) — generates candidates given observation and active subgoal
2. **Goal-conditioned** (`generate_for_goal()`) — takes a `GameObjective` + failure history, produces strategies specifically for that goal while avoiding previously-failed approaches

Supports LLM mode and template fallback. Receives visual cortex predictions in the prompt via `GameObservation.visual_cortex_summary`.

### K. Goal Progress Manager

**File:** `arc_agi/goal_pursuit.py`

Executive controller for goal-directed behaviour — the layer between the adaptive loop and the actioner.

**Responsibilities:**

| Function | Description |
|----------|-------------|
| `set_goal_bank()` | Accept 3-6 hypotheses from the decomposer |
| `select_next_goal()` | Pick highest-confidence non-terminal goal |
| `begin_strategy()` / `end_strategy()` | Lifecycle hooks that snapshot state and measure progress |
| `_measure_progress()` | Multi-signal goal-relative evaluator |
| `decide_next_action()` | Retry strategy / switch goal / regenerate bank |
| `strategy_context()` | Provide failure history to strategy generator |

**Progress measurement** — brutally simple, level-completion-dominated:

```
if level_completed:
    P = 0.90 + 0.10 · P_state     (near-certain success)
elif game_over:
    P = 0.05 · P_state              (clear failure)
else:
    P = 0.30·P_state + 0.15·P_novelty + 0.15·P_player + 0.15·P_goal
    (max ≈ 0.75 — cannot fake success without a level)
```

ARC rewards clear breakthroughs, not smooth progress. The old 6-signal weighted average produced flat ~0.47 noise across all strategies — indistinguishable. The new signal is bimodal: either something meaningful happened (level/new state) or it didn't.

**Decision logic:**

```
if progress ≥ 0.70 → "continue" (success)
if all goals exhausted → "regenerate" (new bank)
if current goal max attempts (3) with no meaningful progress → "new_goal"
if strategy failed → "new_strategy" (retry same goal)
if partial progress → "continue" (refine)
```

**Data structures:**

- `GameObjective` — goal hypothesis with progress signals and tracking
- `StrategyOutcome` — full record of one strategy attempt (stored in memory)
- `ProgressSignal` — one measurable indicator (name, direction, weight)

### I. Actioner

**File:** `arc_agi/actioner.py`

Converts strategies into concrete game actions:

| Handler | Trigger | Logic |
|---------|---------|-------|
| **navigate** | Strategy targets a position | Dot-product alignment with known action directions |
| **click** | Strategy involves clicking | Cycle through target positions via ACTION6 |
| **explore** | Strategy says explore | Systematic action cycling |
| **undo** | Strategy says undo/reset | RESET action |
| **sequence** | Procedural memory has a plan | Replay recorded sequence |

Stuck detection: 4 consecutive no-effect actions → random unstick.

### J. State Describer

**File:** `arc_agi/state_describer.py`

Converts raw game frames into structured `GameObservation`:

```python
@dataclass
class GameObservation:
    raw_grid: np.ndarray           # Current frame
    grid_hash: int                  # For state tracking
    objects: List[Dict]             # Detected objects (value, size, center)
    player_info: Optional[Dict]     # Player position if identified
    game_state: str                 # NOT_PLAYED / NOT_FINISHED / WIN / GAME_OVER
    levels_completed: int
    available_actions: List[str]
    visual_cortex_summary: str      # Injected VC predictions (NL)
    # ... more fields
```

## Unified Adaptive Loop

**File:** `agents/templates/adaptive_reasoning_agent.py` → `main()`

Instead of a hard Phase 1 → Phase 2 transition, the agent uses a **single loop** where every iteration sits on a continuous exploration ↔ exploitation spectrum controlled by **ε (exploration rate)**.

### Knowledge Metric

`knowledge_level` ∈ [0, 1] — focused on **actionable** knowledge (do we know what actions DO?):

| Signal | Weight | Range |
|--------|--------|-------|
| Action coverage (tried ≥2×) | 0.30 | 0→1 |
| Mechanism confidence (consistent change_rate) | 0.30 | 0→1 |
| Level completed | 0.25 | 0 or 1 |
| Player identified | 0.15 | 0 or 1 |

Mechanism confidence counts actions tried ≥3× where change_rate ≥0.7 or ≤0.15 (i.e. we reliably know what it does).

### Epsilon — Hard Exploitation Shifts

```
if any level completed:
    ε = 0.10                          ← exploit hard
elif knowledge ≥ 0.6:
    ε = 0.25                          ← commit to strategic play
else:
    base = max(0.15, 1.0 − knowledge × 1.5)
    iter_decay = max(0.0, 1.0 − iteration / 20)
    ε = base × (0.3 + 0.7 × iter_decay)
    ε = clamp(ε, 0.05, 1.0)
```

Key insight: ARC rewards **fast commitment** once a pattern is detected. The old gradual 80-iteration decay kept ε stuck at ~0.72, preventing the agent from ever transitioning to exploitation.

### Time Budget & Stall Restart

The entire game runs under a **single time budget** (default 60s).  There is no action-count budget.

```
TIME_BUDGET = 60s              ← total time for one game
STALL_FRACTION = 0.25          ← fraction of remaining time before restart
stall_limit = max(2s, time_left × 0.25)
```

If a strategic iteration makes no level progress for `stall_limit` seconds, it is abandoned and a fresh iteration starts with all accumulated knowledge.

### Per-Iteration Decision

```
while time_left > 0 and wins < 10:
    if ε > 0.5:
        → Exploration iteration (100 fast actions)
          • 70% GameMemory-ranked actions (when knowledge ≥ 0.4)
          • 30% novelty-biased / NN retrieval (fallback)
    else:
        → Goal pursuit iteration:
           1. Generate/regenerate goal bank if needed
           2. Switch goal if current exhausted
           3. Generate goal-conditioned strategy
           4. Score: 40% action_effect + 25% lock_bonus + 25% confidence + 10% EBM
           5. Snapshot state → Execute → Measure (binary progress)
           6. decide_next_action() → retry / switch / regenerate
```

### Feedback Loops (the bootstrap)

```
┌─ Exploration ──────────────────────────────────────────┐
│                                                         │
│  When knowledge ≥ 0.4 (70% of actions):                 │
│    rank_actions() picks by score_action()               │
│    (change_rate − death_rate + consistency + win_bonus)  │
│    ← mechanism-driven, not random                       │
│                                                         │
│  Fallback (~30%):                                       │
│    retrieve_action() or pick_novel_action()             │
│    (associations + NN + VC priors + avoidance)          │
│                                                         │
│  Mechanism locking:                                     │
│    Actions with stable behavior (≥5 tries, consistent   │
│    change rate) are frozen and exploited deterministically│
│                                                         │
│  Every transition feeds:                                │
│    • VC buffer (continuous training every 10 iters)     │
│    • Assoc memory (LTP/LTD, episodes, procedures)       │
│    • Game memory (action profiles → mechanism lock)     │
│  → mechanism lock → knowledge ↑ → ε ↓↓ hard shift      │
└─────────────────────────────────────────────────────────┘

┌─ Goal Pursuit ────────────────────────────────────────────┐
│                                                            │
│  Goal bank generation (LLM + template fallback):           │
│    → 3-6 GameObjective hypotheses with progress signals    │
│    → highest confidence selected as active goal            │
│                                                            │
│  Per-goal inner loop (up to 3 strategy attempts):          │
│    STRATEGIZE → goal-conditioned + failure-aware            │
│    SCORE     → 40% action_effect + 25% lock + 25% conf    │
│              + 10% EBM tiebreaker                          │
│    EXECUTE   → actioner runs strategy                      │
│    MEASURE   → binary progress (level or not?)             │
│    DECIDE    → retry / switch goal / regenerate bank       │
│                                                            │
│  Strategy outcomes stored in associative memory:           │
│    • Indexed by goal_id for retrieval                      │
│    • Cross-game transfer of successful strategies          │
│                                                            │
│  Discoveries feed back to memory systems:                  │
│    • VC directions → GameMemory.vc_directions              │
│    • VC danger → AssociativeMemory.danger_map              │
│    • VC similarity → policy NN regularisation              │
│  → richer memory → smarter exploration next iteration      │
│                                                            │
│  All goals exhausted → regenerate bank with new evidence   │
└────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
                    ┌──────────────┐
                    │  Game Frame  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │State Describer│
                    └──────┬───────┘
                           │
                  GameObservation
                    ┌──────▼───────────────────────────┐
                    │         Visual Cortex             │
                    │  predict_all_actions(grid, acts)  │
                    └──┬──────┬──────┬──────┬──────────┘
                       │      │      │      │
        NL summary     │   analysis  │  similarity
        (prompt)       │   (struct)  │  (cosine)
           │           │      │      │
   ┌───────▼──┐   ┌───▼──────▼──────▼───┐
   │ Strategy  │   │  Associative Memory │
   │ Generator │   │  ┌─change priors    │
   └─────┬─────┘   │  ├─directions       │──→ GameMemory.vc_directions
         │         │  ├─danger map       │
   candidates     │  ├─sim regularise   │
         │         │  └─strategy outcomes│──→ CrossGameMemory hints
   ┌─────▼─────┐   └─────────────────────┘
   │JEPA World │──→ ẑ_{t+1} + aux predictions
   │  Model    │
   └─────┬─────┘
         │
   ┌─────▼──────────┐  action scores   ┌──────────────────┐
   │Strategy Scoring │◀─── locked ─────│   Game Memory     │
   │ 40% action_eff  │   mechanisms     │ (action profiles) │
   │ 25% lock bonus  │                 │ (mechanism lock)  │
   │ 25% confidence  │                 └──────────────────┘
   │ 10% EBM energy  │
   └─────┬──────────┘                  ┌──────────────────┐
         │                            │GoalProgressManager│
         │                            │ binary progress   │
         │                            │ retry/switch/regen│
         │                            └──────────────────┘
   ┌─────▼─────┐
   │  Actioner  │──→ concrete action + data
   └─────┬─────┘
         │
   ┌─────▼──────────┐
   │   Game Engine   │──→ next frame, reward, game state
   └────────────────┘
```

## Cross-Game & Cross-Run Learning Flow

```
  ┌─────────────────── Run 1 ───────────────────┐
  │                                              │
  │  Game 1          Game 2          Game N      │
  │  ┌────────┐     ┌────────┐     ┌────────┐   │
  │  │new_game│◀xm▶│new_game│◀xm▶│new_game│   │
  │  │ play.. │     │ play.. │     │ play.. │   │
  │  │ export │─▶xm│ export │─▶xm│ export │─▶xm
  │  └────────┘     └────────┘     └────────┘   │
  │                                              │
  └──────────────────────┬───────────────────────┘
                         │ save("cross_game_memory.pt")
                         ▼
                   ┌───────────┐
                   │   DISK    │  cross_game_memory.pt
                   └─────┬─────┘
                         │ load("cross_game_memory.pt")
                         ▼
  ┌─────────────────── Run 2 ───────────────────┐
  │                                              │
  │  Game 1          Game 2          Game N      │
  │  (starts with priors from Run 1)             │
  │  ...                                         │
  └──────────────────────┬───────────────────────┘
                         │ save (cumulative)
                         ▼
                      Run 3 ...
```

**What compounds across runs (trust-gated):**

| Knowledge | Mechanism | Trust gating |
|-----------|-----------|-------------|
| Policy NN weights | Generalised exploration strategy | NN blend starts at raw × 0.3; grows only if in-game training validates |
| Action effect priors | EMA of change/death rates | Prior counts scaled by trust (0.3); in-game observations dominate after ~5 actions |
| Goal strategy hints | Top-3 outcomes per goal type | Seeded as hypotheses; current game must confirm before reliance |
| Failure patterns | Proxy-progress records, overpredicted goals | Actively reduce confidence in goal families that historically mislead |
| Contradicted priors | Action priors disagreeing with in-game by >0.3 | Prior count cap reduced per contradiction; persistent liars are weakened |

## Hardware Budget

| Component | Parameters | Memory | Training |
|-----------|-----------|--------|----------|
| Visual Cortex (U-Net + FiLM) | ~450K | < 50MB | Online, continuous (every 10 iterations) |
| JEPA World Model | ~5M | < 100MB | **Disabled by default** (`train_jepa_online=False`) |
| EBM Scorer | ~1M | < 20MB | **Disabled by default** (`train_ebm_online=False`) |
| Policy NN (actor-critic) | ~10K | < 1MB | Online, after each episode |
| Game Memory (action profiles) | < 1K | < 1MB | Instant (per-action stats) |
| Associative Memory (tabular) | Variable | < 10MB | Instant (LTP/LTD updates) |
| LLM (optional) | 7-8B | ~5GB | Inference only |

**Total without LLM: < 200MB.** Runs comfortably on CPU.

Kaggle constraints: CPU/GPU ≤ 6 hrs, no internet, pre-trained models OK.

## Training Loop

**File:** `ARC-AGI-3-Agents/run_training_loop.py`

Iterates the full 25-game test suite multiple times, compounding cross-game memory between runs.

```bash
python run_training_loop.py [iterations] [num_games] [time_budget]
# Example: 5 runs × 25 games × 60s each
python run_training_loop.py 5 25 60
```

**Features:**
- tqdm progress bar across iterations
- Parses metrics from each run (solved, wins, avg progress, NN steps, goal types, etc.)
- Prints comparative diagnostic table after each iteration with ▲/▼ trend indicators
- Shows first-to-last improvement summary (absolute + percentage change)
- Tracks which specific games were won per run
- Cross-game memory auto-loads at start, auto-saves at end of each run

**Expected learning curve:**

| Run | What improves |
|-----|---------------|
| 1 | Baseline — discovers action effects, builds initial policy NN, records failure patterns |
| 2 | Weak action priors (trust=0.3) seed slightly faster exploration; failure patterns flag misleading goals |
| 3+ | Contradicted priors are weakened; overpredicted goals are flagged; NN generalises cautiously |

**Modes:**

```bash
# Development mode (default): full persistence for iterating on public games
python run_training_loop.py 5 25 60

# Competition mode: heavily regularised persistence for novel games
python test_full_agent.py 25 60 competition
```

## Design Principles

1. **Hard exploitation shifts** — Once a level is completed (ε→0.10) or mechanics are confident (ε→0.25), the agent commits. ARC rewards fast commitment, not gradual decay
2. **Mechanism-driven action selection** — GameMemory `score_action()` is the primary decision signal; EBM is a 10% tiebreaker. Actions with consistent observed effects get locked and exploited deterministically
3. **Goal-directed hypothesis testing** — Multiple objective hypotheses compete; strategies serve specific goals; progress is binary (level or not), not smooth noise
4. **Two complementary world models** — JEPA predicts abstract consequences (inference only by default); visual cortex predicts concrete pixel changes for memory and navigation
5. **Brain-inspired memory, not just replay** — LTP/LTD synaptic learning enables rapid online adaptation with forgetting
6. **Graceful degradation** — Every LLM call has a template fallback; system works without GPU or internet
7. **Sceptical cross-game transfer** — Persistent memory acts as a hypothesis proposer, not an action governor; priors start at low trust (0.3), require in-game validation, and decay when contradicted
8. **Failure memory as first-class citizen** — Proxy-progress patterns, overpredicted goals, and contradicted action priors are persisted as aggressively as successes; remembering what *not* to trust prevents compounding errors
9. **Reduce learning noise** — JEPA and EBM online training disabled by default; only VC trains online (cheap, useful). The bottleneck is action selection quality, not representation capacity
10. **Dev/competition separation** — Development mode uses full persistence on known games; competition mode applies heavy regularisation (trust=0.15) to preserve adaptation to novelty

## Summary

> An adaptive agent architecture for ARC-AGI-3 that uses **mechanism-driven
> action selection** where learned action effects (change rates, death rates,
> consistency) are the primary decision signal. Exploration transitions hard
> to exploitation once mechanics are understood (ε→0.10 on level completion).
> Multiple goal hypotheses compete with binary progress measurement (level
> completion, not smooth noise). A CNN visual cortex predicts pixel-level
> changes, JEPA/EBM provide secondary scoring (10% weight), brain-inspired
> associative memory enables rapid online learning, and GameMemory's
> `score_action()` + mechanism locking drive both exploration and strategy
> selection — with cross-game transfer of policy weights, action priors,
> and successful goal-strategy patterns.

---

# V3 Architecture: Operator-Centric Adaptive Reasoning

V3 is a ground-up redesign that replaces V2's action-scoring and ε-based
exploration loop with **operator induction, specialist minds, beam search
over abstractions, and solution compression**.

## V3 Design Philosophy

1. **Mechanics before strategy** — infer *what actions do* and *under which
   conditions* before committing to plans.
2. **Operators before raw actions** — the internal control language is
   `move(up)`, `click(blue_object)`, `push(box, left)`, not `ACTION1`.
3. **Search over abstractions** — beam search over operators/macros,
   never over raw button presses.
4. **Exploit brutally** — once mechanics are understood, commit fast.
   ARC scoring penalises wasted actions.

## V3 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AdaptiveReasoningAgentV3                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Perception Layer                                            │
│     - Object Extractor  (connected-component by colour)         │
│     - Frame Diff         (cell/object/player-level diff)        │
│     - Affordance Mapper  (clickable/traversable/hazardous/…)    │
│     - Player Hypotheses  (ranked candidates with confidence)    │
│                                                                  │
│  2. Mechanic Inference Layer                                    │
│     - Action Profiler    (context-conditioned per-action stats)  │
│     - Operator Inducer   (move/noop/lethal/click/transform)     │
│     - Rule Engine        (symbolic death/win/blocking rules)     │
│     - Experiment Designer (max info-gain action selection)       │
│                                                                  │
│  3. Memory Layer                                                │
│     - Game Memory V3     (operators + rules + macros + failures) │
│     - Episodic Graph     (state motifs ↔ operators, relational)  │
│     - Belief Debugger    (audit & demote false beliefs)          │
│     - Cross-Game Memory  (trust-gated operator/rule/macro priors)│
│                                                                  │
│  4. Control Layer                                               │
│     - Progress Tracker    (three-tier progress: LP/SP/TP + branch killer)│
│     - Specialist Minds   (Navigator/Click/Sequence/Physics/      │
│                            Transform/Closure)                    │
│     - Arbiter            (Bayesian selection, not LLM)           │
│     - Operator Searcher  (beam search over operators/macros)     │
│     - Reactive Controller(locked ops, danger escape, greedy move)│
│     - Action Executor    (operator → primitive translation)      │
│                                                                  │
│  5. Compression Layer                                           │
│     - Solution Shortener (iterative action removal + replay)     │
│     - Macro Compiler     (repeated sequences → reusable options) │
└─────────────────────────────────────────────────────────────────┘
```

## V3 Main Decision Loop

```
while time_left > 0 and not done:
    obs = perceive(frame)          # structured observation
    update_beliefs(obs)            # profiler, operators, rules

    if should_experiment():        # uncertainty high?
        plan = experiment_designer.design(obs)
    elif should_exploit():         # mechanics understood?
        plan = reactive_controller.act(obs)
        if plan is None:
            plan = mind_proposals → arbiter → search
    else:
        plan = mind_proposals → arbiter → search

    action = executor.translate(plan, obs)
    next_frame = env.step(action)
    record_transition(obs, action, next_frame)

    if level_solved():
        compress_solution()
        compile_macros()
```

The loop does **not** ask "explore or strategize?" — it asks "do I need
**information**, **search**, or **execution** right now?"

### Mode Selection Logic

| Condition | Mode | Action |
|-----------|------|--------|
| `progress.should_kill_branch()` | Experiment (forced) | Break sterile loop, reset branch |
| Every 10th action | Search (forced) | Mind proposals → arbiter → beam search |
| Every 10th+5 action | Experiment (forced) | Try untested actions |
| `LP > 0.3 AND SP < 0.15` (stall) | SequenceMind (forced) | Permutation strategy, every 20th action |
| `TP > 0.1` | ClosureMind (forced) | Terminal probes, every 15th action |
| `inducer.best_confidence < 0.6` | Experiment | Max info-gain probes |
| `action_coverage < 0.6` | Experiment | Try untested actions |
| `levels_completed increased` | Exploit | Reactive + solution compress |
| `num_locked_operators ≥ 2` | Exploit | Reactive greedy control |
| `best_operator_confidence ≥ 0.7` | Exploit | Reactive then search |
| `knowledge_level ≥ 0.6` | Exploit | Mind proposals + search |
| Otherwise | Search | Mind proposals → arbiter → beam search |

**Progress-based biasing:**
- High TP (terminal progress) → bias toward ClosureMind, exploit
- High SP (structural progress) but low TP → bias toward search, explore new regions
- High LP (local progress) but low SP → bias toward SequenceMind, force combos
- All low → bias toward experiment, explore mechanics

## V3 Module Reference

### A. Perception Layer

#### Object Extractor (`perception/object_extractor.py`)
4-connected flood-fill extraction of coloured components.  Each object:
`object_id`, `value`, `cells`, `bbox`, `center`, `area`, `shape_signature`.

**Object tracking**: match objects across frames by value + proximity (max
distance 5.0).  Produces `created`, `removed`, `moved` lists.

**Player hypotheses**: ranked by heuristics — small area (1-4 cells),
unique value, continuity with previous frame, movement consistency.

#### Frame Diff (`perception/frame_diff.py`)
Structured diff between two frames:
- Cell-level: changed positions, before/after values
- Object-level: created, removed, moved (with displacement vectors)
- Player displacement: `(dy, dx)` or `None`
- Flags: `game_over`, `level_complete`, `is_noop`

This is the **raw material for all mechanic inference**.

#### Affordance Mapper (`perception/affordance_mapper.py`)
Maps objects → interaction possibilities: `CLICKABLE`, `TRAVERSABLE`,
`HAZARDOUS`, `COLLECTIBLE`, `MOVABLE`, `UNKNOWN`.  Uses known lethal
values, danger map, object size/proximity.

### B. Mechanic Inference Layer

#### Action Profiler (`mechanics/action_profiler.py`)
**Context-conditioned** action tracking.  Each action is profiled not just
globally but per discrete context bucket:

```python
context_key = (
    "player_yes",
    "wall_up",
    "obj_right_v3",
    "free_down",
    "free_left",
)
```

Per-context stats: `tries`, `changes`, `deaths`, `wins`, `mean_diff_size`,
`mean_disp`.  Global stats aggregate across contexts.

Key queries:
- `action_coverage(available, min_tries)` → fraction profiled
- `most_uncertain_action(available)` → fewest tries
- `is_consistent(action, min_tries)` → change rate ≥0.8 or ≤0.15
- `dominant_displacement(action)` → `(dy, dx)` if consistently moves player

#### Operator Inducer (`mechanics/operator_inducer.py`)
Transforms profiled action-effect traces into **reusable Operator schemas**.

```python
@dataclass
class Operator:
    operator_id: str
    kind: OperatorKind       # MOVE / CLICK / NOOP / LETHAL / GLOBAL_TRANSFORM / …
    parameters: dict
    preconditions: list[Predicate]
    expected_effects: list[Effect]
    confidence: float        # support / (support + contradictions), scaled by evidence
    support: int
    contradictions: int
    primitive_action: str    # mapped raw ARC action
```

**Induction families** (Phase 1):

| Family | Detection | Preconditions | Effects |
|--------|-----------|---------------|---------|
| Movement | `dominant_displacement` ≥60% of tries | `PlayerExists`, `CellFree(dy,dx)` | `PlayerDisplacement(dy,dx)` |
| No-op | ≥85% no change | — | `NoEffect` |
| Lethal | ≥50% game-over | — | `GameOverEffect` |
| Click | Has coords, change_rate>50%, not movement | `ObjectExists(value)` | `ChangedCells`, `RemovesObject` |
| Global Transform | ≥5 cells changed ≥50% of time | — | `GlobalGridChange` |

Runs every **5 actions**.  Operators with contradiction rate >50% are pruned.

#### Predicate & Effect DSL (`schemas.py`)

**Predicates** (state conditions):
`PlayerExists`, `CellFree(rel)`, `AdjacentToValue(v)`, `ObjectExists(v)`,
`InDangerZone(threshold)`, `ObjectCount(v, cmp, n)`

**Effects** (predicted outcomes):
`PlayerDisplacement(dy,dx)`, `ChangedCells(range)`, `RemovesObject(v)`,
`CreatesObject(v)`, `GlobalGridChange(min)`, `GameOverEffect`,
`LevelCompleteEffect`, `NoEffect`

Each effect has a `matches(diff)` method for automated verification.

#### Rule Engine (`mechanics/rule_engine.py`)
Proposes and verifies symbolic causal rules from repeated observations.

**Rule families**:
- **Death rules**: "moving onto value X causes game-over" (tracked by
  value + death co-occurrence)
- **Completion rules**: "removing last object of value X completes level"
- **Blocking rules**: "movement blocked by value X in direction D"
- **Removal rules**: "action A removes objects of value X"

Rules with confidence <0.3 are pruned.  Runs every **10 actions**.

#### Experiment Designer (`mechanics/experiment_designer.py`)
Maintains a posterior over **7 effect-type hypotheses** per action:
`movement`, `interaction`, `noop`, `lethal`, `global_transform`,
`conditional`, `unknown`.

Info-gain scoring:
```
info_gain = entropy(posterior) × novelty_decay(tries) − 0.5 × death_risk
```

Designs 1-3 action experiments targeting the highest-entropy actions.
Triggers when operator confidence is low, coverage is sparse, or
posteriors are high-entropy.

### C. Memory Layer

#### Game Memory V3 (`memory/game_memory.py`)
Per-game aggregate: profiler + inducer + rule engine + macros + solutions +
failure patterns + state visit counts + lethal values.

**Knowledge level** (0→1):
```
0.30 × action_coverage
0.30 × avg_operator_confidence
0.25 × level_completion (binary)
0.15 × has_movement_operators (binary)
```

#### Episodic Graph (`memory/episodic_graph.py`)
Graph-structured memory.  Nodes: `state_motif`, `operator`, `macro`,
`rule`, `failure`, `goal`.  Edges: `precedes`, `enables`, `contradicts`,
`similar`, `dangerous_after`.

Retrieval is graph traversal:
- Current state motif → activates nearby operator hypotheses
- Repeated failure motif → suppresses misleading operators
- Similar solved motif → activates known macros

Supports `decay()` for forgetting and `export_compact()` for cross-game.

#### Belief Debugger (`memory/belief_debugger.py`)
Audits all beliefs every **20 actions**.  Detects:

| Issue | Detection | Repair |
|-------|-----------|--------|
| Overconfident operator | conf>0.7 but contradiction rate>30% | Halve confidence |
| High-risk non-lethal | risk>0.5, not marked lethal | Demote |
| Contradicted rule | contradiction rate>30% | Demote (>60%: prune) |
| Unstable macro | success_rate<0.3 after 3+ uses | Prune |

#### Cross-Game Memory V3 (`memory/cross_game_memory.py`)
Trust-gated transfer of **compact abstractions only**:
- Operator templates (kind → list of lightweight dicts, max 5/kind)
- Rule priors (rule_id → dict)
- Macro schemas (macro_id → dict)
- Failure patterns (motif → dict)
- Mind reliability priors (name → accuracy)

Trust: `0.30` dev / `0.15` competition.  Growth on agreement (+0.10),
decay on disagreement (×0.50).  Max trust 0.70.  File size guard: warn
if >10MB.

### D. Control Layer

#### Progress Tracker (`control/progress_tracker.py`)
**Three-tier progress system + sterile branch detection.**

Replaces the old flat `_progress_motifs` with a more nuanced progress signal:

| Tier | Signals | Purpose |
|------|---------|---------|
| **LP (Local Progress)** | Safe moves, operator predictions matched, clicks, transforms | Measures immediate action competence |
| **SP (Structural Progress)** | Novel states, first-time class removals, new regions, rules validated | Measures meaningful state-space expansion |
| **TP (Terminal Progress)** | Remaining target objects, classes fully exhausted, closure plausibility | Measures proximity to level completion |

**Branch Killing (sterile-loop detection):**

Rolling 40-action window tracks:
- Repeated state hashes (same hash dominates 50%+ of window)
- Repeated diff signatures (same diff 60%+ of window)
- Unique states visited (<3 unique states)
- Terminal stall duration (50 actions without TP improvement)

When branch is sterile → forces experiment mode to break out of loop.

Integration:
- `progress.on_action(diff, obs)` called on every transition
- `progress.should_kill_branch()` checked in mode selection
- `progress.scores()` returns (LP, SP, TP) for mode biasing
- Branch ID increments on each kill (reported in diagnostics)

#### Specialist Minds (`control/specialist_minds.py`)
Six competing worldviews, each proposing an operator-level plan:

| Mind | Worldview | Key Signal |
|------|-----------|------------|
| **Navigator** | Spatial movement toward goals | Movement operators + target distance |
| **Click** | Object selection / click interaction | Click operators or high change-rate non-movement |
| **Sequence** | Exact short programs matter | Known sequences, permutation strategies, forced sampling on stall |
| **Physics** | Pushing, collisions, hazards | Blocking rules, push targets, danger escape |
| **Transform** | Global grid changes / toggles | Global-transform operators |
| **Closure** | Hypothesis-driven completion | Few targets remain, high-confidence ops, terminal probes |

**ClosureMind** (new):
- Activates when: ≤5 small targets remain OR (validated≥2 AND high-confidence≥0.7)
- Strategies: eliminate remaining objects, reach furthest targets, transform sequence, terminal probes (try untested actions)
- Forced every 15th action when TP > 0.1

**SequenceMind** (revived):
- Added permutation strategy: generates untried A→B→A combos from effective actions (confidence 0.35)
- Tracks tried combos to avoid repetition
- Forced every 20th action when LP > 0.3 but SP < 0.15 (stalling with local competence)

Each proposal includes: `confidence`, `expected_progress`,
`expected_info_gain`, `estimated_cost`, `estimated_risk`.

Mind tracking: `proposals_made`, `predictions_correct`,
`progress_realised` → used by arbiter for calibration.

#### Arbiter (`control/arbiter.py`)
**Performance-based, not LLM-based.**

```
score = 0.30 × expected_progress
      + 0.20 × expected_info_gain
      + 0.20 × mind_recent_accuracy
      + 0.15 × confidence
      − 0.05 × cost/10
      − 0.10 × risk
```

Tracks per-mind selection frequency.  Feeds back real outcomes to update
mind reliability.

#### Operator Searcher (`control/operator_search.py`)
Beam search over operators and macros (not raw actions).

- Beam width: **4**, max depth: **5**
- Expands only applicable operators (preconditions met), skips LETHAL/NOOP
- Macros wrapped as pseudo-operators if `success_rate ≥ 0.3`

```
beam_score = 0.35 × goal_progress
           + 0.20 × rule_consistency
           + 0.20 × operator_confidence
           + 0.15 × novelty_bonus
           − 0.10 × depth_penalty
```

#### Reactive Controller (`control/reactive_controller.py`)
Fast, cheap decisions without search:
1. **Stuck recovery**: 4 consecutive no-effect → random unstick
2. **Danger escape**: use safest movement operator
3. **Greedy movement**: one step toward target via best movement op
4. **Locked operator**: use highest-confidence applicable operator

#### Action Executor (`control/executor.py`)
Translates operator calls → primitive ARC actions.  The **only module**
that touches the raw action space.

Handles: `move_*`, `click_*`, `macro_*`, `experiment_*`, `probe_*`,
`seq_step_*`, `unstick_*`, `raw_click_*`, and operator-mapped primitives.

### E. Compression Layer

#### Solution Shortener (`compression/solution_shortener.py`)
Once a level is solved, immediately tries to compress:
1. Single-action removal (greedy, iterate until no improvement)
2. Contiguous span removal (spans of 2 and 3)
3. Repeated pattern compression (consecutive identical actions)

Requires a `replay_fn` that re-executes a candidate sequence and returns
whether it still solves the level.

#### Macro Compiler (`compression/macro_compiler.py`)
**Two-tier macro system: ephemeral patterns → promoted macros.**

- **Ephemeral cache**: frequent subsequences (2-4 steps) cached cheaply, max 30 patterns. Does NOT affect control.
- **Promoted macros**: validated subsequences that precede structural progress. Max 10 active.
- **Promotion criteria**: 3+ repeats AND mean structural gain > 0
- **Integration**: `compile_from_trace()` takes `structural_gain` parameter from ProgressTracker SP score.

Full solution compiled as macro (success_rate=1.0). Retained macros ranked by success rate.

## V3 Data Flow

```
    Frame (64×64 grid)
         │
    ┌────▼────┐
    │ Object  │── objects, player hypotheses
    │Extractor│
    └────┬────┘
         │
    ┌────▼────┐
    │  Frame  │── FrameDiff (cells, objects, displacement, flags)
    │  Diff   │
    └────┬────┘
         │
    ┌────▼────┐
    │Progress │── LP/SP/TP scores, branch kill signal
    │ Tracker │
    └────┬────┘
         │
    ┌────▼────┐
    │ Action  │── TransitionRecord → per-action, per-context stats
    │Profiler │
    └────┬────┘
         │ every 5 actions
    ┌────▼─────┐
    │ Operator │── Operator library (move/noop/lethal/click/transform)
    │ Inducer  │
    └────┬─────┘
         │ every 10 actions
    ┌────▼────┐
    │  Rule   │── symbolic rules (death/win/blocking/removal)
    │ Engine  │
    └────┬────┘
         │
    ┌────▼──────────┐        ┌──────────┐
    │   Experiment  │───────▶│ Specialist│
    │   Designer    │  info  │  Minds   │
    └───────────────┘  needs ├──────────┤
                             │ Arbiter  │
                             ├──────────┤
                             │ Operator │── beam search plan
                             │ Searcher │
                             ├──────────┤
                             │ Reactive │── fast greedy action
                             │Controller│
                             └────┬─────┘
                                  │
                             ┌────▼────┐
                             │Executor │── PrimitiveAction (button press)
                             └────┬────┘
                                  │
                                  ▼
                             env.step()
                                  │
                        on level solved:
                             ┌────▼────┐
                             │Solution │── shortened trajectory
                             │Shortener│
                             ├─────────┤
                             │ Macro   │── compiled reusable sequences (two-tier)
                             │Compiler │
                             └─────────┘
```

## V3 Key Scoring Formulas

### Operator Confidence
```
confidence = (support / (support + contradictions)) × min(1.0, evidence / 8)
```

### Knowledge Level
```
knowledge = 0.30 × coverage + 0.30 × avg_op_conf + 0.25 × level + 0.15 × has_moves
```

### Info Gain (Experiment Designer)
```
info_gain = entropy(posterior) × (1 / (1 + tries × 0.15)) − 0.5 × death_risk
```

### Arbiter Score
```
score = 0.30×progress + 0.20×info_gain + 0.20×accuracy + 0.15×confidence − 0.05×cost − 0.10×risk
```

### Beam Search Score
```
beam = 0.35×goal_progress + 0.20×rule_consistency + 0.20×op_confidence + 0.15×novelty − 0.10×depth
```

## V3 File Layout

```
v3/
├── __init__.py
├── schemas.py                         # All shared data structures, predicates, effects
├── adaptive_reasoning_agent_v3.py     # Main agent
│
├── perception/
│   ├── object_extractor.py            # Connected-component extraction + player hypotheses
│   ├── frame_diff.py                  # Structured frame diff
│   └── affordance_mapper.py           # Object → interaction possibility mapping
│
├── mechanics/
│   ├── action_profiler.py             # Context-conditioned action stats
│   ├── operator_inducer.py            # Action traces → Operator schemas
│   ├── rule_engine.py                 # Symbolic rule proposal + verification
│   └── experiment_designer.py         # Info-gain experiment design
│
├── memory/
│   ├── game_memory.py                 # Per-game knowledge store
│   ├── episodic_graph.py              # Graph-structured relational memory
│   ├── belief_debugger.py             # False belief detection + repair
│   └── cross_game_memory.py           # Trust-gated cross-game transfer
│
├── control/
│   ├── progress_tracker.py            # Three-tier progress (LP/SP/TP) + branch killing
│   ├── specialist_minds.py            # Navigator / Click / Sequence / Physics / Transform / Closure
│   ├── arbiter.py                     # Bayesian mind selection
│   ├── operator_search.py             # Beam search over operators/macros
│   ├── reactive_controller.py         # Fast locked-operator execution
│   └── executor.py                    # Operator → primitive action translation
│
└── compression/
    ├── solution_shortener.py          # Iterative action removal
    └── macro_compiler.py              # Sequence → reusable macro compilation
```

## V3 Implementation Phases

### Phase 1 — V3-lite backbone (implemented)
- FrameDiff, object extraction, player hypotheses
- Context-aware ActionProfiler
- Operator induction (move/noop/lethal/click/global_transform)
- NavigatorMind, ClickMind, SequenceMind, PhysicsMind, TransformMind, ClosureMind
- Arbiter + beam search over operators
- Solution shortener + macro compiler (two-tier: ephemeral → promoted)
- Episodic graph memory + belief debugger
- Cross-game memory V3
- Progress tracker (three-tier: LP/SP/TP + branch killing)
- Main agent with experiment/exploit/search decision loop
- Forced mind sampling (SequenceMind on stall, ClosureMind on terminal progress)
- `_has_compositional_evidence()` uses three-tier progress

### Phase 2 — Real mechanic reasoning (next)
- Richer rule proposer (toggle, spawn, conditional transforms)
- More sophisticated experiment design (multi-step hypotheses)
- Macro parameterisation (not just raw replay)
- Episodic graph cross-game transfer
- Integration with existing V2 visual cortex for danger/direction priors

### Phase 3 — Stronger transfer and abstraction
- Cross-game operator templates with analogical matching
- Object-centric latent workspace
- Optional JEPA-assisted operator ranking
- Synthetic mechanic universe pre-training
- Two-speed mind (fast reactive + slow deliberative)

## V3 vs V2 Comparison

| Dimension | V2 | V3 |
|-----------|----|----|
| Control language | Raw actions (ACTION1…7) | Operators (`move_up`, `click_blue`) |
| Exploration | ε-decay → informed probing | Info-gain experiment design |
| Exploitation | GameMemory.rank_actions() | Reactive controller + beam search |
| Strategy | LLM-generated text strategies | Specialist mind proposals |
| Scoring | 40% action_effect + 25% lock + 25% conf + 10% EBM | Arbiter (progress/info/accuracy/conf/cost/risk) |
| Progress | Binary level completion | Binary level + rule-based verification |
| Memory | Flat action profiles + associative memory | Operators + rules + episodic graph |
| Transfer | Action priors + NN weights | Operator templates + rule priors + macros |
| Compression | None | Solution shortening + macro compilation |
| World model | JEPA + EBM (10% weight) | Rules + predicates (symbolic, no training) |
| Decision loop | "explore or strategize?" | "information, search, or execution?" |

## V3 Summary

> An operator-centric, mechanic-inducing adaptive agent that discovers
> local game physics online, compiles them into reusable operators with
> symbolic preconditions and effects, searches over those operators via
> beam search, and compresses successful behaviour for action-efficient
> solving.  Six specialist minds (including new ClosureMind) with different
> inductive biases compete under Bayesian arbitration.  A belief debugger
> prevents false abstractions from persisting.  Three-tier progress tracking
> (LP/SP/TP) with branch killing prevents sterile loops.  A two-tier macro
> system (ephemeral cache → promoted) ensures only semantically useful
> sequences are compiled.  Cross-game transfer moves operator templates and
> rules, not raw action statistics.

---

# V4 Architecture: Chambered Adaptive Reasoning

V4 reorganizes the agent around four interacting currencies:

- **Surprise** - what violated expectation or exposed new structure
- **Laws** - what local mechanics and soft purposes currently explain the world
- **Projects** - what compact undertakings are worth funding right now
- **Dissent** - what part of the current explanation is probably wrong

Instead of treating the agent as a single controller, V4 treats each game
as a temporary world that must be modeled, inhabited, challenged,
compressed, and sometimes abandoned.

The current V4 implementation is intentionally more skeptical than the
initial prototype: it now emphasizes **epistemic discipline** as much as
generative richness. Abstractions are budgeted, pruned, and forced to pay
rent via predictive or control value before they stay alive.

The next planned extension, **V4-Learn**, keeps this chambered scaffold
but adds learned calibrators and value models so the agent can adapt over
attempts without giving up interpretability or grounded control.

## V4 Design Philosophy

1. **Worldviews before plans** - keep multiple ontologies alive until evidence kills them.
2. **Projects before monolithic search** - fund small, legible undertakings rather than one global plan.
3. **Skepticism as a first-class module** - false coherence must be actively attacked.
4. **Compression only after payoff** - motifs and rituals are promoted only when they repeatedly buy structure or closure.
5. **Multi-timescale memory** - branch-local memory, per-game memory, and cross-game abstractions all have different jobs.
6. **Make structure expensive to keep** - operators, rules, teleology, projects, motifs, and rituals now compete under hard budgets.
7. **Retrieve across frames, not only within them** - memory should support bissociation, not just nearest-neighbor recall.
8. **Use heuristics as priors and teachers** - learned components should modulate chamber outputs rather than replace them end-to-end.
9. **Learn trust, not only content** - the agent should learn which internal signals deserve budget, confidence, and control authority.

## V4 High-Level Architecture

```
+--------------------------------------------------------------------+
|                    AdaptiveReasoningAgentV4                         |
+--------------------------------------------------------------------+
|  A. Sensorium                                                      |
|     - FrameDiffer         (cell/object/player diffs)               |
|     - ObjectTracker       (entity extraction + player hypotheses)  |
|     - SurpriseField       (pixel/object/causal/topology surprise)  |
|     - TopologyMonitor     (reachable regions and unlocks)          |
|                                                                    |
|  B. Ontology Chamber                                               |
|     - OntologyFactory      (avatar/click/token/field/phase/transform) |
|     - OntologyCompetition  (evidence-for vs evidence-against)      |
|     - AffordanceReframer   (ontology-conditioned salience)         |
|                                                                    |
|  C. Physics Chamber                                                |
|     - ActionProfiler       (context + ontology conditioned stats)  |
|     - OperatorInducer      (move/noop/lethal/click/transform/etc.) |
|     - ConstraintEngine     (blocking/death/removal/phase rules)    |
|     - TeleologyEngine      (speculative vs validated closure laws) |
|     - LawCompetition       (rank operators, rules, teleology)      |
|                                                                    |
|  D. Strategy Chamber                                               |
|     - ProjectGenerator     (exhaust/reach/probe/replay/closure)    |
|     - ProjectMarket        (dignity, cost, fragility, payoff, pruning) |
|     - Specialist Minds     (Navigator/Click/Sequence/Physics/      |
|                              Transform/Closure)                    |
|     - Arbiter              (proposal ranking)                      |
|     - OperatorSearcher     (phase-aware operator expansion)        |
|                                                                    |
|  E. Dissent Chamber                                                |
|     - FalseProgressDetector                                        |
|     - OntologyDissenter                                            |
|     - AntiLoopCritic                                               |
|     - DissentController                                            |
|                                                                    |
|  F. Composition Chamber                                            |
|     - MotifComposer        (action/count/ontology-shift motifs)    |
|     - PrefixCompressor     (shorten successful prefixes)           |
|     - Ritualizer           (ontology-tagged reusable prefixes)     |
|     - ForgettingManager    (decay stale beliefs and motifs)        |
|                                                                    |
|  G. Execution Layer                                                |
|     - ReactiveController   (cheap grounded control)                |
|     - ActionExecutor       (intent/operator -> primitive action)   |
|     - EmergencyReset       (branch abandonment policy)             |
|                                                                    |
|  H. Memory                                                         |
|     - FastMemory          (branch-local state, plan, dissent)      |
|     - FrameMemory         (coherent interpretive bundles)          |
|     - BridgeMemory        (useful + harmful frame crossings)       |
|     - BissociationEngine  (hybrid project retrieval)               |
|     - GameMemoryV4        (laws, motifs, rituals, phase history,   |
|                              budgets and survival scores)          |
|     - CrossGameMemoryV4   (low-trust transfer of compact priors)   |
|                                                                    |
|  I. Learning Layer (planned)                                       |
|     - WorldReliabilityModel (predict belief-state usefulness)      |
|     - ProjectValueModel    (learn project payoff by context)       |
|     - ArbiterBandit        (learn which mind to trust when)        |
|     - OntologyCalibrator   (recalibrate worldview confidence)      |
|     - OperatorUtilityModel (predict strategic operator value)      |
|     - TeleologyValidator   (promote closure hypotheses cautiously) |
|     - SterilityPredictor   (predict zombie branches early)         |
|     - Bridge/Compression   (learn useful crossings and motifs)     |
|     - RetrospectiveCredit  (turn attempts into training signals)   |
+--------------------------------------------------------------------+
```

## V4 Main Runtime Loop

```
while time_left > 0 and not done:
    obs = sensorium.observe(prev_obs, prev_frame, frame)

    if previous_action_exists:
        transition = sensorium.make_transition(...)
        profiler.update(...)
        inducer.induce(...)
        constraints.update(...)
        teleology.evidence_hits(...)
        progress.on_transition(...)
        teleology.update(...)
        motif_composer.update(...)
        game_memory.enforce_budgets()

    ontology.update(obs, memory)
    current_frame = frame_memory.observe(obs, memory)
    bridge_memory.observe_shift(previous_frame, current_frame, progress_gain)
    phase = phase_controller.select_phase(obs, memory)

    if emergency_reset.should_reset(memory):
        intent = RESET
    else:
        dissent_report = dissent.update(obs, memory)
        if dissent.should_interrupt(memory):
            intent = dissent.interrupt_and_redirect(obs, memory)
        else:
            intent = strategy.propose_control(obs, memory, phase)

    primitive = executor.act(intent, obs, memory)
    env.step(primitive)

    if level_completed:
        compressed = prefix_compressor.compress(trace)
        ritualizer.compile(memory, compressed)
        forgetting.decay(memory)
```

V4 no longer asks only "what action should I take?" It continuously asks:

- What kind of world is this?
- What laws currently explain it?
- What project deserves funding?
- Why might that explanation be wrong?

## V4 Chamber Reference

| Chamber | Key files | Responsibility |
|---------|-----------|----------------|
| Sensorium | `v4/sensorium/frame_diff.py`, `object_tracker.py`, `surprise_field.py`, `topology_monitor.py` | Turn raw frames into object-level diffs, surprise, and reachability structure |
| Ontology | `v4/ontology/ontology_hypotheses.py`, `ontology_competition.py`, `affordance_reframing.py` | Maintain competing world interpretations and reweight salience |
| Physics | `v4/physics/action_profiler.py`, `operator_inducer.py`, `constraint_engine.py`, `teleology_engine.py`, `law_competition.py` | Induce operators, constraints, and split terminal reasoning into speculative vs validated teleology |
| Strategy | `v4/strategy/project_generators.py`, `project_market.py`, `specialist_minds.py`, `arbiter.py`, `operator_search.py` | Generate candidate projects, include hybrid bissociative projects, and keep only the strongest active market |
| Dissent | `v4/dissent/false_progress_detector.py`, `ontology_dissenter.py`, `anti_loop_critic.py`, `dissent_controller.py` | Detect fake progress, ontology lock-in, repeated sterile behavior, and chamber-specific abstraction inflation |
| Composition | `v4/composition/motif_composer.py`, `prefix_compressor.py`, `ritualizer.py`, `forgetting.py` | Promote motifs and rituals while decaying stale abstractions |
| Execution | `v4/execution/reactive_controller.py`, `executor.py`, `emergency_reset.py` | Resolve intents into primitive ARC actions, reject weak operators, and manage resets |
| Memory | `v4/memory/fast_memory.py`, `frame_memory.py`, `bridge_memory.py`, `bissociation_engine.py`, `game_memory.py`, `cross_game_memory.py` | Separate branch-local, per-game, cross-frame, and cross-game memory responsibilities |

## V4-Learn: Planned Learning Extension

V4 is now architecture-rich enough that the main next step is not to
replace it with end-to-end learning, but to make the architecture learn
from its own attempts. The core idea is:

- heuristics remain the scaffold
- symbolic chambers keep interpretability and strong inductive bias
- learned components estimate which abstractions, projects, and internal
  signals are actually trustworthy

In short: V4-Learn treats the current hand-designed heuristics as
**priors, targets, and teachers** for learned modulators.

### Learning timescales

V4-Learn is planned across four timescales:

- **Within-branch learning** - learn from the last few dozen actions.
- **Within-game learning** - learn across attempts inside one game.
- **Across-game meta-learning** - learn which ontologies, laws, projects, and bridges tend to matter.
- **Self-evaluation learning** - learn which internal scores predict real payoff and which only sound good.

### Planned learned modules

| Model | Planned file | Role |
|-------|--------------|------|
| World reliability model | `v4/learning/world_reliability_model.py` | Predict whether the current belief state is likely to buy SP/TP or become sterile |
| Project value model | `v4/learning/project_value_model.py` | Estimate the future payoff of a project in the current world context |
| Arbiter bandit | `v4/learning/arbiter_bandit.py` | Learn which mind proposal is worth following in context |
| Ontology calibrator | `v4/learning/ontology_calibrator.py` | Recalibrate ontology confidences from evidence patterns and later outcomes |
| Operator utility model | `v4/learning/operator_utility_model.py` | Separate operator usefulness for solving from mere predictive existence |
| Teleology validator | `v4/learning/teleology_validator.py` | Decide which speculative closure hypotheses deserve promotion |
| Sterility predictor | `v4/learning/sterility_predictor.py` | Predict zombie branches before the hand-tuned reset rules fire |
| Bridge value model | `v4/learning/bridge_value_model.py` | Learn which frame crossings are likely to yield useful hybrid projects |
| Compression value model | `v4/learning/compression_value_model.py` | Learn which motifs and rituals are worth preserving |
| World embedding model | `v4/learning/world_embedding_model.py` | Learn compact world and episode embeddings that index symbolic frame and bridge memory |
| Retrospective credit | `v4/learning/retrospective_credit.py` | Convert attempts into training examples for every learned head |

### Interaction rule

The learning layer should **modulate** chamber outputs, not replace them.

Examples:

- heuristic operator confidence says an operator is real; learned operator utility can still downweight it if it is strategically useless
- heuristic TP rises; learned teleology validation can cap that rise if closure evidence remains speculative
- heuristic arbiter ranking favors one mind; the learned bandit can shift the ranking toward another mind that historically pays off in similar contexts
- heuristic branch rules allow more search; learned sterility or world-reliability models can cut search budget earlier

This keeps V4 grounded while giving it real adaptation over attempts.

### Retrospective credit assignment

The most important learning mechanism is retrospective self-analysis after
each attempt. V4-Learn should ask questions like:

- which projects actually preceded SP increases?
- which teleology hypotheses preceded TP gains or solves?
- which ontology shifts helped escape sterile local explanations?
- which bridges created hybrid projects that paid off?
- which motifs only appeared inside elegant but useless loops?
- which dissent warnings turned out to be correct?

Those answers become training examples for the project value model,
ontology calibrator, teleology validator, arbiter bandit, sterility
predictor, bridge value model, and compression value model.

### Training signals

V4 does not need external labels for this layer. Each run already produces:

- observations and transitions
- internal chamber states
- project proposals and chosen minds
- accepted operators, rules, teleology hypotheses, motifs, and rituals
- later LP/SP/TP outcomes
- branch kills, resets, and solved flags

This supports supervised or bandit-style examples such as:

- `(branch_features, zombie_flag)`
- `(project_features, future_SP, future_TP, solved_flag)`
- `(ontology_evidence, later_usefulness)`
- `(proposal_features, realized_payoff)`
- `(bridge_features, hybrid_project_utility)`
- `(motif_features, later_reuse_value)`

### Planned learning roadmap

The intended implementation order is:

1. **Phase 1** - `arbiter_bandit.py`, `sterility_predictor.py`, `project_value_model.py`
2. **Phase 2** - `ontology_calibrator.py`, `operator_utility_model.py`, `teleology_validator.py`
3. **Phase 3** - `bridge_value_model.py`, `compression_value_model.py`, `world_reliability_model.py`
4. **Phase 4** - `world_embedding_model.py` for learned world / episode embeddings that improve frame retrieval and bissociation while still handing symbolic frames and bridges back to control

This sequence keeps the first gains practical: better arbitration,
earlier branch pruning, and more reliable project funding before deeper
meta-learning is layered on top.

## V4 Epistemic Discipline

The biggest recent V4 change is that abstractions are no longer allowed to
proliferate freely. The agent now uses hard active budgets plus survival
scoring to keep the ecology small enough to remain honest.

### Active budgets

| Structure | Active budget |
|-----------|---------------|
| Ontologies | Top 3 |
| Operators | 10 |
| Rules | 6 |
| Teleology | 3 validated + 4 speculative |
| Projects | 6 |
| Motifs | 5 |
| Rituals | 2 |

### Survival principle

Each major abstraction is now retained only if it buys predictive or
control value. In practice this means V4 maintains survival-style scores
using combinations of:

- predictive accuracy
- realized structural gain
- realized terminal gain
- contradiction rate
- support / usage
- idle penalties for unused weak hypotheses

This makes V4 much harsher than the first prototype, which tended to reward
abstraction generation more than abstraction quality.

## V4 Progress and Phase Logic

V4 keeps the V3 LP/SP/TP decomposition, but uses it inside a richer
meta-control story:

| Signal | Meaning in V4 |
|--------|----------------|
| **LP** | Immediate control competence: safe movement, predicted effects, effective click/transform behavior |
| **SP** | Structural world change: novel states, unlocked regions, first-time removals, new validated rules |
| **TP** | Conservative closure pressure: signs that the world is narrowing toward completion, but only under stronger evidence |

Phases are selected by `v4/control/phase_controller.py`:

- `sensory_ignorance`
- `mechanical_stabilization`
- `project_emergence`
- `closure_pressure`
- `compression`
- `crisis`

Branch death is still explicit. `ProgressTracker.should_kill_branch()` and
`EmergencyReset.should_reset()` together allow V4 to abandon sterile local
world-models instead of polishing them forever.

### TP hardening

TP is now intentionally difficult to inflate. Strong TP increases are
primarily reserved for:

- validated teleology hits
- replay of a ritual / useful prefix
- class exhaustion or region unlock events with supporting evidence
- actual level completion

Speculative teleology can nudge TP only slightly, and unsolved runs keep TP
capped well below certainty. This change was made to avoid the earlier V4
failure mode where TP could jump to a suspiciously high fixed value too
easily.

## V4 Teleology

`v4/physics/teleology_engine.py` now splits terminal reasoning into two
stages:

- **Speculative teleology** - cheap closure hypotheses that remain weak and do not strongly drive control.
- **Validated teleology** - hypotheses that survived repeated evidence and are allowed to meaningfully shape TP and project generation.

This keeps V4 imaginative about possible end conditions without letting
closure narratives dominate before they have earned it.

## V4 Bissociative Memory

V4 memory is now explicitly bissociative rather than purely associative.
The goal is not only to recall similar past structure, but to connect
different interpretive frames and mine the tension between them.

### New memory modules

| Module | File | Role |
|--------|------|------|
| Frame memory | `v4/memory/frame_memory.py` | Stores coherent interpretive bundles such as ontology + dominant laws + dominant project + terminal style |
| Bridge memory | `v4/memory/bridge_memory.py` | Stores useful and harmful crossings between frames |
| Bissociation engine | `v4/memory/bissociation_engine.py` | Retrieves one dominant frame, one distant-but-relevant frame, and proposes hybrid projects |

### Retrieval modes

V4 now supports three memory moves:

- **Associative retrieval** - find nearby structure within the current worldview.
- **Contrastive retrieval** - find distant frames with similar outcomes but different explanations.
- **Bissociative retrieval** - connect two frames through a bridge and generate a hybrid project.

This means V4 can now form proposals like:

- reach a target under an avatar interpretation, then click it under a click interpretation
- transform first, then probe objects immediately
- navigate to a class target and treat exhaustion as the likely closure route

Negative bridges are also stored, so memory can remember not only creative
crossings that worked, but seductive crossings that repeatedly produce
sterile loops.

## V4 Transfer Discipline

Cross-game transfer is now deliberately conservative:

- `INITIAL_TRUST = 0.10`
- `MAX_TRUST = 0.20`
- only a smaller number of operator templates and rituals are seeded
- seeded confidence is clamped low
- trust rises meaningfully only on wins or unusually strong predictive/control behavior

This is a deliberate correction to the earlier V4 prototype, where transfer
trust could grow too high relative to the system's actual reliability.

The planned learning layer extends this with learned world or episode
embeddings that **index** symbolic memory rather than replace it. Vector
retrieval should help find similar, contrastive, and bridgeable worlds,
then hand symbolic frame and bridge structures back to the strategist and
dissenter.

## V4 File Layout

```
v4/
|-- __init__.py
|-- schemas.py
|-- adaptive_reasoning_agent_v4.py
|
|-- sensorium/
|   |-- frame_diff.py
|   |-- object_tracker.py
|   |-- surprise_field.py
|   `-- topology_monitor.py
|
|-- ontology/
|   |-- ontology_hypotheses.py
|   |-- ontology_competition.py
|   `-- affordance_reframing.py
|
|-- physics/
|   |-- action_profiler.py
|   |-- operator_inducer.py
|   |-- constraint_engine.py
|   |-- teleology_engine.py
|   `-- law_competition.py
|
|-- control/
|   |-- progress_tracker.py
|   |-- phase_controller.py
|   `-- branch_scheduler.py
|
|-- strategy/
|   |-- project_generators.py
|   |-- project_market.py
|   |-- specialist_minds.py
|   |-- arbiter.py
|   `-- operator_search.py
|
|-- dissent/
|   |-- false_progress_detector.py
|   |-- ontology_dissenter.py
|   |-- anti_loop_critic.py
|   `-- dissent_controller.py
|
|-- composition/
|   |-- motif_composer.py
|   |-- prefix_compressor.py
|   |-- ritualizer.py
|   `-- forgetting.py
|
|-- execution/
|   |-- reactive_controller.py
|   |-- executor.py
|   `-- emergency_reset.py
|
`-- memory/
    |-- frame_memory.py
    |-- bridge_memory.py
    |-- bissociation_engine.py
    |-- fast_memory.py
    |-- game_memory.py
    `-- cross_game_memory.py
```

Planned learning package:

```
v4/learning/
|-- world_reliability_model.py
|-- project_value_model.py
|-- arbiter_bandit.py
|-- ontology_calibrator.py
|-- operator_utility_model.py
|-- teleology_validator.py
|-- sterility_predictor.py
|-- bridge_value_model.py
|-- compression_value_model.py
|-- world_embedding_model.py
`-- retrospective_credit.py
```

## ARC Integration

The V4 package is exposed to the ARC-AGI-3 runner through:

- `ARC-AGI-3-Agents/agents/templates/adaptive_reasoning_v4_agent.py`
- `ARC-AGI-3-Agents/test_v4_agent.py`

The bridge keeps the same outer lifecycle as V3: keep playing until the
game is won or the wall-clock budget is exhausted.

## V4 vs V3 Comparison

| Dimension | V3 | V4 |
|-----------|----|----|
| Core unit of reasoning | Operators | Ontologies + laws + projects |
| Main competition | Specialist minds | Ontologies, projects, and minds |
| Progress failure handling | Branch kill on sterile LP/SP/TP profile | Branch kill plus explicit dissent, chamber warnings, and ontology downweighting |
| Closure model | ClosureMind + TP heuristics | Speculative vs validated teleology + conservative TP + closure projects |
| Compression | Macros and shortened traces | Motifs, compressed prefixes, and rituals under hard budgets |
| Transfer | Operator/rule/macro priors | Low-trust ontology/operator/motif/ritual priors with strict seeding limits |
| Memory style | Episodic/operator association | Multi-timescale memory plus bissociative frame/bridge retrieval |
| Learning posture | Mostly heuristic adaptation | Heuristic chamber ecology with planned learned calibrators, value models, and retrospective credit |
| Main risk addressed | Unknown local mechanics | Premature coherence around the wrong world-model and abstraction inflation |

## V4 Summary

> A chambered adaptive reasoner that treats each ARC game as a temporary
> world to be constituted rather than merely searched. Sensorium measures
> surprise, the ontology chamber keeps multiple worldviews alive, the physics
> chamber induces both local mechanics and speculative/validated terminal
> laws, the strategy chamber funds a small market of compact projects, the
> dissent chamber attacks false coherence at both global and chamber levels,
> and the composition chamber promotes only those motifs and rituals that
> repeatedly buy structural or terminal progress. Memory is now explicitly
> bissociative: it stores coherent frames, bridges between frames, and hybrid
> retrieval proposals. Execution stays grounded and skeptical, transfer trust
> stays deliberately low, and structure is now expensive to keep alive. The
> planned V4-Learn layer does not replace this architecture; it teaches the
> agent which internal abstractions deserve trust, budget, and control
> authority by learning from retrospective outcomes over attempts.
