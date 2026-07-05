"""
Quick offline test for the AdaptiveReasoning agent.

Runs the agent against one local game in OFFLINE mode to verify
the full pipeline: grid analysis → memory → strategy routing → action.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Get project root (parent of ARC-AGI-3-Agents)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_DIR = PROJECT_ROOT / "environment_files"

# Force offline mode before any imports read .env
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", "recordings")

from arc_agi import Arcade, OperationMode
from agents import AVAILABLE_AGENTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def test_single_game(game_id: str, max_actions: int = 50) -> dict:
    """Run the AdaptiveReasoning agent on a single game offline."""
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )

    agent_cls = AVAILABLE_AGENTS["adaptivereasoning"]

    # Override MAX_ACTIONS for quicker testing
    original_max = agent_cls.MAX_ACTIONS
    agent_cls.MAX_ACTIONS = max_actions

    env = arc.make(game_id)
    agent = agent_cls(
        card_id=None,
        game_id=game_id,
        agent_name="adaptivereasoning",
        ROOT_URL="",
        record=False,
        arc_env=env,
    )

    logger.info(f"Starting agent on game {game_id} (max {max_actions} actions)")
    start = time.time()

    try:
        agent.main()
    except Exception as e:
        logger.error(f"Agent crashed: {e}", exc_info=True)

    elapsed = time.time() - start

    # Collect results
    summary = agent.memory.summary()
    result = {
        "game_id": game_id,
        "actions_taken": agent.action_counter,
        "elapsed_seconds": round(elapsed, 2),
        "max_level": summary["max_level"],
        "states_visited": summary["states_visited"],
        "exploration_score": round(summary["exploration_score"], 3),
        "player_identified": summary["player_identified"],
        "action_semantics": summary["action_semantics"],
    }

    # Restore
    agent_cls.MAX_ACTIONS = original_max

    return result


def main():
    # Discover available games
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    game_ids = [e.game_id for e in envs]
    logger.info(f"Found {len(game_ids)} games: {game_ids[:5]}...")

    # Pick first game for testing
    target = game_ids[0] if game_ids else None
    if not target:
        logger.error("No games found in environment_files/")
        return

    # Allow command-line override
    if len(sys.argv) > 1:
        target = sys.argv[1]

    max_actions = 50
    if len(sys.argv) > 2:
        max_actions = int(sys.argv[2])

    result = test_single_game(target, max_actions=max_actions)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
