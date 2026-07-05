"""Quick multi-game test for the AdaptiveReasoning agent."""

import logging
import os
import sys
import time
from pathlib import Path

# Get project root (parent of ARC-AGI-3-Agents)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_DIR = PROJECT_ROOT / "environment_files"

os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", "recordings")

from arc_agi import Arcade, OperationMode
from agents import AVAILABLE_AGENTS

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
logger = logging.getLogger(__name__)


def test_game(arc, game_id, max_actions=100, cross_game=None):
    agent_cls = AVAILABLE_AGENTS["adaptivereasoning"]
    original_max = agent_cls.MAX_ACTIONS
    agent_cls.MAX_ACTIONS = max_actions

    env = arc.make(game_id)
    agent = agent_cls(
        card_id=None, game_id=game_id, agent_name="adaptivereasoning",
        ROOT_URL="", record=False, arc_env=env, cross_game=cross_game, arcade=arc,
    )
    start = time.time()
    try:
        agent.main()
    except Exception as e:
        print(f"  ERROR: {e}")

    elapsed = time.time() - start
    s = agent.memory.summary()

    # Count how many actions actually changed the grid
    effective = sum(
        1 for e in agent.memory.action_history if e.anything_changed
    )
    move_ids = len(s["movement_actions"])

    agent_cls.MAX_ACTIONS = original_max
    return {
        "game": game_id[:12],
        "actions": agent.action_counter,
        "time": round(elapsed, 2),
        "level": s["max_level"],
        "states": s["states_visited"],
        "explore": round(s["exploration_score"], 2),
        "player": s["player_identified"],
        "move_ids": move_ids,
        "effective": effective,
    }


def main():
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    envs = arc.get_environments()
    game_ids = [e.game_id for e in envs]

    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory
    cross_game = CrossGameMemory()

    print(f"Testing {max_games} games, {max_actions} actions each (cross-game learning)\n")
    print(f"{'Game':<15} {'Acts':>5} {'Time':>6} {'Lvl':>4} {'States':>7} {'Expl':>6} {'Player':>7} {'MvID':>5} {'Eff':>4}")
    print("-" * 75)

    results = []
    for gid in game_ids[:max_games]:
        r = test_game(arc, gid, max_actions, cross_game)
        results.append(r)
        print(f"{r['game']:<15} {r['actions']:>5} {r['time']:>6}s {r['level']:>4} {r['states']:>7} {r['explore']:>6} {str(r['player']):>7} {r['move_ids']:>5} {r['effective']:>4}")

    print("-" * 75)
    avg_explore = sum(r["explore"] for r in results) / len(results) if results else 0
    total_levels = sum(r["level"] for r in results)
    identified = sum(1 for r in results if r["player"])
    total_eff = sum(r["effective"] for r in results)
    print(f"{'TOTALS':<15} {'':>5} {'':>6}  {total_levels:>4} {'':>7} {avg_explore:>6.2f} {identified:>4}/{len(results):>2} {'':>5} {total_eff:>4}")


if __name__ == "__main__":
    main()
