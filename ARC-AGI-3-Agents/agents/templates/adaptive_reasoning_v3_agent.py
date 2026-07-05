"""
V3 Adaptive Reasoning Agent — ARC-AGI-3 bridge.

Wraps the V3 operator-centric architecture into the Agent interface
expected by the ARC-AGI-3-Agents framework.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..tracing import trace_agent_session

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v3.adaptive_reasoning_agent_v3 import AdaptiveReasoningAgentV3
from v3.memory.cross_game_memory import CrossGameMemoryV3

logger = logging.getLogger(__name__)


# ── Action name / enum helpers ──────────────────────────────────────

_NAME_TO_ACTION = {
    "RESET": GameAction.RESET,
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
}

_INT_TO_ACTION_NAME = {
    0: "RESET",
    1: "ACTION1", 2: "ACTION2", 3: "ACTION3",
    4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}


def _normalize_action_name(a: Any) -> str:
    if isinstance(a, int):
        return _INT_TO_ACTION_NAME.get(a, f"ACTION{a}")
    name = a.name if hasattr(a, "name") else str(a)
    return name


class AdaptiveReasoningV3(Agent):
    """
    ARC-AGI-3 agent powered by the V3 operator-centric architecture.

    Lifecycle per game:
      1. RESET to start
      2. Feed frames to V3 agent → get operator-translated primitive actions
      3. On GAME_OVER: RESET and continue with accumulated knowledge
      4. Until WIN or TIME_BUDGET exhausted
    """

    MAX_ACTIONS: int = 999_999  # not the real limit — time budget is
    TIME_BUDGET: float = 60.0

    # Stuck/loop detection parameters
    STUCK_HASH_WINDOW: int = 30    # check last N state hashes for cycles

    def __init__(
        self,
        *args: Any,
        cross_game: Optional[CrossGameMemoryV3] = None,
        arcade: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cross_game = cross_game
        self._arcade = arcade

        # Build the V3 agent
        self.v3 = AdaptiveReasoningAgentV3(
            cross_game=cross_game,
            time_budget=self.TIME_BUDGET,
        )

        # State tracking
        self._game_started: bool = False
        self._needs_reset: bool = False
        self._last_action_name: Optional[str] = None
        self._prev_levels: int = 0
        self._available_action_names: List[str] = [
            "ACTION1", "ACTION2", "ACTION3", "ACTION4",
            "ACTION5", "ACTION6", "ACTION7",
        ]

        # Loop / stuck detection
        self._recent_hashes: List[int] = []
        self._consecutive_noops: int = 0
        self._reset_count: int = 0
        self._start_time: Optional[float] = None
        self._actions_since_level: int = 0  # actions since last level completion
        self._prev_levels_wrapper: int = 0
        self._actions_per_attempt: int = 300  # force reset after this many fruitless actions
        self._base_attempt_budget: int = 300
        self._stop_reason: str = "running"

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def _time_remaining(self) -> float:
        return max(0.0, self.TIME_BUDGET - self._elapsed())

    def _deadline_reached(self) -> bool:
        return self._time_remaining() <= 0.0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Win condition
        if latest_frame.state is GameState.WIN:
            self._stop_reason = "win"
            return True

        # Time budget exhausted
        elapsed = self._elapsed()
        if elapsed >= self.TIME_BUDGET:
            self._stop_reason = "time_budget"
            return True

        # Too many resets (game-over loop — truly hopeless)
        return False

    @trace_agent_session
    def main(self) -> None:
        """Run until the game is won or the time budget is exhausted."""
        self.timer = time.time()
        self._start_time = time.monotonic()
        self._stop_reason = "running"

        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            latest_frame = self._convert_raw_frame_data(
                self.arc_env.observation_space if self.arc_env else None
            )
            if self.is_done(self.frames, latest_frame):
                break

            action = self.choose_action(self.frames, latest_frame)
            if self._deadline_reached():
                self._stop_reason = "time_budget"
                break

            if frame := self.take_action(action):
                self.append_frame(frame)
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, "
                    f"levels completed {frame.levels_completed}, avg fps {self.fps})"
                )
            self.action_counter += 1

        self.cleanup()

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if self._deadline_reached():
            return GameAction.RESET
        try:
            return self._choose_action_impl(frames, latest_frame)
        except Exception as e:
            logger.error(f"[{self.game_id}] V3 error: {e}", exc_info=True)
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return GameAction.RESET
            return random.choice([
                GameAction.ACTION1, GameAction.ACTION2,
                GameAction.ACTION3, GameAction.ACTION4,
            ])

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _choose_action_impl(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # ── Handle mandatory resets ──
        if self._deadline_reached():
            return GameAction.RESET

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._needs_reset = True
            self._reset_count += 1

        if self._needs_reset:
            self._needs_reset = False
            self._game_started = True
            self._last_action_name = "RESET"
            return GameAction.RESET

        if latest_frame.full_reset and self._last_action_name != "RESET":
            self._last_action_name = "RESET"
            return GameAction.RESET

        # ── Parse current frame ──
        grid = self._frame_to_grid(latest_frame)

        # ── Track state hash for loop detection ──
        grid_hash = hash(grid.tobytes())
        self._recent_hashes.append(grid_hash)

        # ── Trim hash buffer to avoid unbounded growth ──
        if len(self._recent_hashes) > 500:
            self._recent_hashes = self._recent_hashes[-200:]

        # ── Track progress: reset counter on level completion ──
        current_levels = latest_frame.levels_completed
        if current_levels > self._prev_levels_wrapper:
            self._prev_levels_wrapper = current_levels
            self._actions_since_level = 0
            self._actions_per_attempt = self._base_attempt_budget  # fresh budget
            self._recent_hashes.clear()
        else:
            self._actions_since_level += 1

        # ── Stuck cycle reset: same states repeating ──
        if len(self._recent_hashes) >= self.STUCK_HASH_WINDOW:
            window = self._recent_hashes[-self.STUCK_HASH_WINDOW:]
            unique_ratio = len(set(window)) / len(window)
            if unique_ratio < 0.15:  # cycling through <15% unique states
                self._actions_since_level = 0
                self._reset_count += 1
                self._recent_hashes.clear()
                self._last_action_name = "RESET"
                return GameAction.RESET

        # ── Futility reset: too many actions without level completion ──
        if self._actions_since_level >= self._actions_per_attempt:
            self._actions_since_level = 0
            self._reset_count += 1
            self._recent_hashes.clear()
            self._last_action_name = "RESET"
            # Increase budget for next attempt (maybe game needs more exploration)
            self._actions_per_attempt = min(
                1000, self._actions_per_attempt + 100
            )
            return GameAction.RESET

        if latest_frame.available_actions:
            self._available_action_names = [
                _normalize_action_name(a)
                for a in latest_frame.available_actions
                if _normalize_action_name(a) != "RESET"
            ]

        # ── Feed to V3 agent ──
        result = self.v3.choose_action(
            frames=[grid],
            available_actions=self._available_action_names,
            game_state=latest_frame.state.name,
            levels_completed=latest_frame.levels_completed,
        )

        action_name = result["action"]
        self._last_action_name = action_name
        self._prev_levels = latest_frame.levels_completed

        # Convert to GameAction
        game_action = _NAME_TO_ACTION.get(action_name, GameAction.ACTION1)

        # Handle ACTION6 click coords
        if action_name == "ACTION6" and result.get("x") is not None:
            game_action = GameAction.ACTION6
            x = result["x"]
            y = result["y"]
            game_action.set_data({
                "game_id": self.game_id,
                "x": int(x),
                "y": int(y),
            })
        else:
            game_action.set_data({"game_id": self.game_id})

        return game_action

    def _frame_to_grid(self, frame: FrameData) -> np.ndarray:
        """Convert FrameData to numpy grid (extract last 2D grid from frame list)."""
        if frame.frame:
            # frame.frame is List[List[List[int]]] — a list of 2D grids
            grid = frame.frame[-1] if isinstance(frame.frame, list) else frame.frame
            return np.array(grid, dtype=np.int32)
        return np.zeros((10, 10), dtype=np.int32)

    # ------------------------------------------------------------------
    # Game end — export to cross-game memory
    # ------------------------------------------------------------------
    def end_game(self) -> Dict[str, Any]:
        """Call at game end. Returns diagnostic summary."""
        won = self.frames[-1].state == GameState.WIN if self.frames else False
        summary = self.v3.end_game(won=won)
        summary["wrapper"] = {
            "elapsed_seconds": round(self._elapsed(), 2),
            "reset_count": self._reset_count,
            "stop_reason": self._stop_reason,
        }
        return summary
