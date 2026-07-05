"""Single-game verbose diagnostic: trace what the agent does each step."""
import logging, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_DIR = PROJECT_ROOT / "environment_files"
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", "recordings")

# Enable INFO logging for our modules
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(message)s")

from arc_agi import Arcade, OperationMode
from agents import AVAILABLE_AGENTS

arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
envs = arc.get_environments()
game_id = sys.argv[1] if len(sys.argv) > 1 else envs[0].game_id

agent_cls = AVAILABLE_AGENTS["adaptivereasoning"]
agent_cls.MAX_ACTIONS = 200  # iterative play-and-learn

env = arc.make(game_id)
agent = agent_cls(card_id=None, game_id=game_id, agent_name="adaptivereasoning",
                  ROOT_URL="", record=False, arc_env=env)

# Patch reasoning loop to log every action
_orig_step = agent.reasoning.step.__func__

_click_positions = []

def verbose_step(self, current_grid, game_state, levels_completed, available_actions):
    result = _orig_step(self, current_grid, game_state, levels_completed, available_actions)
    ac = self.state.action_counter
    phase = result.get("phase", "?")
    action = result.get("action", "?")
    strat = result.get("strategy")
    strat_type = strat.strategy_type.value if strat else "none"
    sg = result.get("subgoal")
    sg_desc = f"SG{sg.id}:{sg.description[:30]}" if sg else "no-subgoal"
    obs = result.get("observation")
    pos = ""
    if obs and obs.player_info:
        py, px = obs.player_info["y"], obs.player_info["x"]
        pos = f"@({py:.0f},{px:.0f})"
    # Show click position for ACTION6
    click = ""
    action_data = result.get("action_data")
    if action == "ACTION6" and action_data:
        cx, cy = action_data.get("x", "?"), action_data.get("y", "?")
        click = f"click({cy},{cx})"
        _click_positions.append((cy, cx))
    print(f"  [{ac:>3}] {phase:<12} -> {action:<10} {pos:<10} {click:<16} strat={strat_type:<20} {sg_desc}")
    return result

import types
agent.reasoning.step = types.MethodType(verbose_step, agent.reasoning)

print(f"\n=== Game: {game_id} ===\n")
agent.main()

s = agent.memory.summary()
effective = sum(1 for e in agent.memory.action_history if e.anything_changed)
print(f"\n=== Summary ===")
print(f"Actions: {agent.action_counter}, Effective: {effective}")
print(f"Levels: {s['max_level']}, States: {s['states_visited']}")
print(f"Player: {s['player_identified']} (val={s['player_value']})")
print(f"Movement actions: {s['movement_actions']}")
print(f"Exploration: {s['exploration_score']:.2f}")

# Show action profile
print(f"\nAction profiles:")
for name, prof in sorted(agent.memory.action_profiles.items()):
    print(f"  {name}: tried={prof.times_tried}, changed={prof.times_changed_grid}, "
          f"moved_player={prof.times_moved_player}, "
          f"disp={prof.dominant_displacement}, "
          f"no_effect={prof.times_no_effect}")

if _click_positions:
    unique = set(_click_positions)
    print(f"\nClicks: {len(_click_positions)} total, {len(unique)} unique positions")
    for y, x in sorted(unique):
        count = _click_positions.count((y, x))
        print(f"  ({y},{x}): {count}x")
