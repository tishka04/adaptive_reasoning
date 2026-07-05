"""
Transition Collector for ARC-AGI-3 games.

Plays every game in environment_files/ using a CollectorAgent (extends
the framework Agent base class) and records full transitions.

These transitions form the training data for:
  - JEPA world model  (predict z_{t+1} from z_t + action)
  - EBM scorer        (learn which actions lead to progress)

Usage:
    python training/collect_transitions.py [--actions 300] [--runs 3] [--out training/data]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Resolve paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AGENTS_DIR = PROJECT_ROOT / "ARC-AGI-3-Agents"
ENV_DIR = PROJECT_ROOT / "environment_files"

# Add ARC-AGI-3-Agents to path so we can import arc_agi and arcengine
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", "recordings")

from arc_agi import Arcade, OperationMode
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from v4_1_reasoning_system.arc_agi.grid_analyzer import GridAnalyzer, FrameDiff
from v4_1_reasoning_system.arc_agi.game_memory import GameMemory
from v4_1_reasoning_system.arc_agi.state_describer import StateDescriber

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Collector Agent: extends Agent, records transitions
# ------------------------------------------------------------------
ALL_ACTIONS = [
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6,
    GameAction.ACTION7,
]
ACTION_NAMES = [
    "ACTION1", "ACTION2", "ACTION3", "ACTION4",
    "ACTION5", "ACTION6", "ACTION7",
]


class CollectorAgent(Agent):
    """
    Agent that plays with smart-random policy and records transitions.
    """
    MAX_ACTIONS = 300

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analyzer = GridAnalyzer()
        self.memory = GameMemory()
        self.describer = StateDescriber()
        self.transitions: list[dict] = []
        self._prev_grid: np.ndarray | None = None
        self._prev_levels: int = 0
        self._last_action_name: str | None = None
        self._last_action_data: dict | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        state = latest_frame.state

        # Handle resets
        if state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last_action_name = "RESET"
            self._last_action_data = None
            if state == GameState.GAME_OVER:
                self.memory.on_game_over()
            return GameAction.RESET

        # Parse current grid
        current_grid = self.analyzer.parse_frame(latest_frame.frame)

        # Record previous transition
        if self._prev_grid is not None and self._last_action_name and self._last_action_name != "RESET":
            diff = self.analyzer.compute_diff(self._prev_grid, current_grid)

            self.memory.record_action(
                action_name=self._last_action_name,
                grid_before=self._prev_grid,
                grid_after=current_grid,
                diff=diff,
                game_state=state.name,
                levels_completed=latest_frame.levels_completed,
            )

            obs_before = self.describer.describe(
                grid=self._prev_grid,
                memory=self.memory,
                game_state="NOT_FINISHED",
                levels_completed=self._prev_levels,
                action_counter=self.action_counter - 1,
                diff=None,
            )
            obs_after = self.describer.describe(
                grid=current_grid,
                memory=self.memory,
                game_state=state.name,
                levels_completed=latest_frame.levels_completed,
                action_counter=self.action_counter,
                diff=diff,
            )

            level_changed = latest_frame.levels_completed > self._prev_levels
            game_over = state == GameState.GAME_OVER

            transition = {
                "game_id": self.game_id,
                "step": self.action_counter,
                "action": self._last_action_name,
                "action_data": self._last_action_data,
                "grid_before": self._prev_grid.tolist(),
                "grid_after": current_grid.tolist(),
                "grid_shape": list(current_grid.shape),
                "level_before": self._prev_levels,
                "level_after": latest_frame.levels_completed,
                "level_changed": level_changed,
                "game_over": game_over,
                "anything_changed": diff.anything_changed,
                "num_changes": diff.num_changes,
                "player_pos_before": (
                    [obs_before.player_info["y"], obs_before.player_info["x"]]
                    if obs_before.player_info else None
                ),
                "player_pos_after": (
                    [obs_after.player_info["y"], obs_after.player_info["x"]]
                    if obs_after.player_info else None
                ),
                "n_objects_before": len(obs_before.objects),
                "n_objects_after": len(obs_after.objects),
                "state_before": "NOT_FINISHED",
                "state_after": state.name,
                "memory_summary": self.memory.summary(),
            }
            self.transitions.append(transition)

        # Pick next action
        action, action_data = self._pick_action(current_grid)

        if action == GameAction.ACTION6 and action_data:
            action.set_data(action_data)

        self._prev_grid = current_grid
        self._prev_levels = latest_frame.levels_completed
        self._last_action_name = action.name
        self._last_action_data = action_data

        return action

    def _pick_action(
        self, grid: np.ndarray
    ) -> tuple[GameAction, dict | None]:
        """Smart-random policy."""
        action_data = None

        if self.action_counter < 14:
            idx = (self.action_counter // 2) % len(ALL_ACTIONS)
            action = ALL_ACTIONS[idx]
        else:
            move_actions = self.memory.get_movement_actions()
            if move_actions and random.random() < 0.5:
                action_name = random.choice(move_actions)
                idx = ACTION_NAMES.index(action_name) if action_name in ACTION_NAMES else 0
                action = ALL_ACTIONS[idx]
            elif random.random() < 0.15:
                action = GameAction.ACTION6
            elif random.random() < 0.1:
                action = GameAction.ACTION5
            else:
                action = random.choice(ALL_ACTIONS)

        if action == GameAction.ACTION6:
            h, w = grid.shape
            if random.random() < 0.3:
                action_data = {"x": w // 2, "y": h // 2}
            else:
                action_data = {
                    "x": random.randint(0, max(w - 1, 0)),
                    "y": random.randint(0, max(h - 1, 0)),
                }

        return action, action_data


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect ARC-AGI-3 game transitions")
    parser.add_argument("--actions", type=int, default=300, help="Max actions per game per run")
    parser.add_argument("--runs", type=int, default=3, help="Runs per game (different random seeds)")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "training" / "data"),
                        help="Output directory for transition data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    game_ids = [e.game_id for e in envs]

    if not game_ids:
        logger.error("No games found!")
        return

    logger.info(f"Found {len(game_ids)} games. Collecting {args.runs} runs x {args.actions} actions each.")

    all_stats = []

    for gid in game_ids:
        game_transitions = []

        for run in range(args.runs):
            seed = hash(f"{gid}-{run}") % (2**31)
            random.seed(seed)
            np.random.seed(seed % (2**31))

            env = arc.make(gid)
            CollectorAgent.MAX_ACTIONS = args.actions
            agent = CollectorAgent(
                card_id=None,
                game_id=gid,
                agent_name="collector",
                ROOT_URL="",
                record=False,
                arc_env=env,
            )

            try:
                agent.main()
                game_transitions.extend(agent.transitions)
            except Exception as e:
                logger.warning(f"  [{gid}] run {run} failed: {e}")

        if game_transitions:
            out_file = out_dir / f"{gid.replace('-', '_')}.json"
            with open(out_file, "w") as f:
                json.dump(game_transitions, f)

            n_level = sum(1 for t in game_transitions if t["level_changed"])
            n_go = sum(1 for t in game_transitions if t["game_over"])
            n_changed = sum(1 for t in game_transitions if t["anything_changed"])

            stat = {
                "game_id": gid,
                "transitions": len(game_transitions),
                "level_changes": n_level,
                "game_overs": n_go,
                "state_changes": n_changed,
            }
            all_stats.append(stat)

            logger.info(
                f"  {gid}: {len(game_transitions)} trans, "
                f"{n_level} lvl-ups, {n_go} GOs, "
                f"{n_changed} changes"
            )

    # Save summary
    summary_file = out_dir / "collection_summary.json"
    total_transitions = sum(s["transitions"] for s in all_stats)
    total_levels = sum(s["level_changes"] for s in all_stats)

    summary = {
        "total_games": len(all_stats),
        "total_transitions": total_transitions,
        "total_level_changes": total_levels,
        "runs_per_game": args.runs,
        "actions_per_run": args.actions,
        "per_game": all_stats,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nDone! {total_transitions} transitions from {len(all_stats)} games -> {out_dir}")
    logger.info(f"Level changes: {total_levels}, Summary: {summary_file}")


if __name__ == "__main__":
    main()
