# human_trace — black-box human play for v4_1

A small, self-contained pipeline for **recording human playthroughs of the
public ARC-AGI-3 environments** and **feeding the resulting traces into the
`v4_1_reasoning_system` as priors**. It treats every environment as a hidden
transition function and records what you *see*, what you *did*, and what you
*believed you were testing* — never what the hidden code does.

The design follows the spec distilled in `../V5_PROPOSAL.md` and the user's
original proposal: black-box transition + intent tag + optional sticky
hypothesis.

## Contents

| File | Purpose |
|---|---|
| `schema.py` | `StepRecord`, `EpisodeRecord`, `IntentTag`, JSONL writer/reader |
| `recorder.py` | Pygame UI driving an `arc_agi.EnvironmentWrapper` |
| `run_recorder.py` | CLI entry point (`python -m human_trace.run_recorder ...`) |
| `loader.py` | Read traces back into an in-memory corpus |
| `integration.py` | Derive priors and seed `GameMemory` / `CrossGameMemory` |

No changes are made to `v4_1_reasoning_system/`; integration only calls its
public API.

## Install

```powershell
pip install pygame
# (torch, numpy, pydantic, arc_agi, arcengine already come from requirements.txt / .venv)
```

## Record a session

Offline (drives the local simulator under `environment_files/`):

```powershell
python -m human_trace.run_recorder --game ar25 --mode offline --out human_traces
```

Or against the ARC-AGI-3 API (same transport the Agent framework uses):

```powershell
python -m human_trace.run_recorder --game ar25 --mode online --out human_traces
```

### Controls (in the pygame window)

| Key | Effect |
|---|---|
| `1` `2` `3` `4` `5` `7` | Send ACTION1..ACTION5, ACTION7 |
| Left mouse click on grid | Send ACTION6 at the clicked `(x, y)` |
| `R` | RESET |
| `Space` | Skip (no action, but keeps you in the loop) |
| `Tab` / `Shift-Tab` | Cycle intent tag (shown on HUD) |
| `H` | Type a sticky hypothesis (Enter = save, Esc = cancel) |
| `N` | Clear current hypothesis |
| `C` | Mark `hypothesis_confirmed` on the next recorded action |
| `X` | Mark `hypothesis_rejected` on the next recorded action |
| `G` | Mark `goal_changed` on the next recorded action |
| `M` | Append a discovered mechanic line |
| `F` | Append a discovered mistake line |
| `T` | Set `game_type_guess` for the episode |
| `O` | Set `objective_guess` for the episode |
| `K` | Kill the current episode, start a fresh one |
| `Q` / `Esc` | Save & quit |

Every action produces one `StepRecord` (JSONL) with the current intent tag,
any pending cognitive event markers, and sticky hypothesis attached. Cognitive
events are one-shot labels: press `C`, `X`, or `G` when the state has just
become meaningful, then perform the next action. On WIN / GAME_OVER / quit an
`EpisodeRecord` is written.

## Data layout

For a session started at 2026-04-20T17:30Z against `ar25-e3c63847`:

```
human_traces/
├── ar25-e3c63847.20260420-173000.steps.jsonl
└── ar25-e3c63847.20260420-173000.episodes.jsonl
```

See `schema.py` for the exact field list. Each step carries the primary grid
before/after, the available action set, the action + args, the resulting
game state, the intent tag, the cognitive event markers, and the sticky
hypothesis.

## Feed traces into v4_1

```python
from human_trace import load_traces, build_prior_pack, seed_game_memory, seed_cross_game_memory
from v4_1_reasoning_system.arc_agi.game_memory import GameMemory
from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory

corpus = load_traces("human_traces")
cross  = CrossGameMemory.load("cross_game_memory_v4_1.pkl")

for game_id in corpus.by_game:
    pack = build_prior_pack(corpus, game_id)
    gm = GameMemory()
    seed_game_memory(pack, gm)
    seed_cross_game_memory(pack, cross)
    # `gm` is now pre-warmed for this game and can be handed to the reasoning loop.

cross.save("cross_game_memory_v4_1.pkl")
```

### What exactly is transferred

- **Per-game (`GameMemory`)**
  - Action profiles: tries / changes / deaths / wins per action, from replayed
    transitions via `record_action`.
  - Click effectiveness (`record_click` for every ACTION6 step).
  - Player hypotheses (automatically inferred from moved objects).
  - Human hypotheses as `memory.hypotheses[...]` keys with confidence 0.35–0.8.
  - Canonical `game_type::<goal_id>` and `objective::<short>` hypotheses that
    the `GoalDecomposer` template fallback consults.

- **Cross-game (`CrossGameMemory`)**
  - Lightweight dict hints under `goal_strategy_hints[goal_id]` (top-3 per
    goal, respects the memory-leak fix: no `StrategyOutcome` objects).
  - Failure patterns under `failure_patterns[goal_id]` for lost episodes with
    flagged mistakes or GAME_OVER.
  - Canonical goal ids: `click_puzzle`, `sequence_puzzle`, `navigate_exit`,
    `collection`, `push_puzzle`, `transform_puzzle`, `unknown`.

Trust is deliberately kept low. Human priors are *proposers*, not governors —
in-game evidence overrides them within ~20 actions.

## Record the failures too

The loop was designed so that a quit-after-dying or a GAME_OVER episode is
*first-class data*. Use `F` to log "discovered mistakes" while you play.
Losing episodes with at least one logged mistake become `failure_patterns`
entries; the `GoalDecomposer` will then actively avoid those strategies
rather than re-proposing them.
