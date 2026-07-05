"""V5 Adaptive Reasoning Agent bridge for ARC-AGI-3.

Mirrors the V4 wrapper, but instantiates `AdaptiveReasoningAgentV5`
and passes through the `game_id` plus stagnation handling.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..tracing import trace_agent_session

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v5 import AdaptiveReasoningAgentV5
from v5.memory import CrossGameMemoryV5

logger = logging.getLogger(__name__)

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
    0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3",
    4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}


def _normalize_action_name(value: Any) -> str:
    if isinstance(value, int):
        return _INT_TO_ACTION_NAME.get(value, f"ACTION{value}")
    return value.name if hasattr(value, "name") else str(value)


class AdaptiveReasoningV5(Agent):
    """ARC-AGI-3 bridge around the V5 agent."""

    MAX_ACTIONS = 999_999
    TIME_BUDGET = 60.0
    STAGNATION_SECONDS = 0.0

    def __init__(
        self,
        *args: Any,
        cross_game: Optional[CrossGameMemoryV5] = None,
        arcade: Any = None,
        stagnation_seconds: Optional[float] = None,
        use_dissent: bool = True,
        use_ontology: bool = True,
        use_rituals: bool = True,
        use_goal_skeleton: bool = True,
        use_bissociation: bool = True,
        use_learned_priors: bool = False,
        learned_priors: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cross_game = cross_game
        self._arcade = arcade
        self.v5 = AdaptiveReasoningAgentV5(
            cross_game=cross_game,
            time_budget=self.TIME_BUDGET,
            use_dissent=use_dissent,
            use_ontology=use_ontology,
            use_rituals=use_rituals,
            use_goal_skeleton=use_goal_skeleton,
            use_bissociation=use_bissociation,
            use_learned_priors=use_learned_priors,
            learned_priors=learned_priors,
        )
        self._available_action_names: List[str] = [
            "ACTION1", "ACTION2", "ACTION3", "ACTION4",
            "ACTION5", "ACTION6", "ACTION7",
        ]
        self._last_action_name: Optional[str] = None
        self._start_time: Optional[float] = None
        self._stop_reason = "running"
        self._stagnation_seconds = (
            float(stagnation_seconds)
            if stagnation_seconds is not None
            else float(self.STAGNATION_SECONDS)
        )
        self._last_level_progress_at: Optional[float] = None
        self._last_levels_seen: int = 0
        # Secondary stagnation: no LP/SP/TP improvement for the same window
        self._last_score_progress_at: Optional[float] = None
        self._best_lp: float = 0.0
        self._best_sp: float = 0.0
        self._best_tp: float = 0.0

    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def _deadline_reached(self) -> bool:
        return self._elapsed() >= self.TIME_BUDGET

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            self._stop_reason = "win"
            return True
        if self._deadline_reached():
            self._stop_reason = "time_ceiling"
            return True
        if self._stagnation_seconds > 0:
            levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
            now = time.monotonic()

            # ── Channel 1: no NEW level for the window ──
            if self._last_level_progress_at is None:
                self._last_level_progress_at = now
                self._last_levels_seen = levels
            elif levels > self._last_levels_seen:
                self._last_levels_seen = levels
                self._last_level_progress_at = now
            elif (now - self._last_level_progress_at) >= self._stagnation_seconds:
                self._stop_reason = "stagnation"
                return True

            # ── Channel 2: no LP/SP/TP improvement for the window ──
            # This catches attempts that reach a level mid-run then go
            # internally flat — the level-based channel alone would
            # otherwise let them live until the ceiling.
            try:
                lp, sp, tp = self.v5.progress.scores()
            except Exception:
                lp = sp = tp = 0.0
            improved = (
                lp > self._best_lp + 0.01
                or sp > self._best_sp + 0.01
                or tp > self._best_tp + 0.01
            )
            if self._last_score_progress_at is None or improved:
                self._last_score_progress_at = now
                self._best_lp = max(self._best_lp, lp)
                self._best_sp = max(self._best_sp, sp)
                self._best_tp = max(self._best_tp, tp)
            elif (now - self._last_score_progress_at) >= self._stagnation_seconds:
                self._stop_reason = "stagnation_flat"
                return True
        return False

    @trace_agent_session
    def main(self) -> None:
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
                self._stop_reason = "time_ceiling"
                break

            if frame := self.take_action(action):
                self.append_frame(frame)
            self.action_counter += 1

        self.cleanup()

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if self._deadline_reached():
            return GameAction.RESET
        try:
            return self._choose_action_impl(frames, latest_frame)
        except Exception as exc:
            logger.error(f"[{self.game_id}] V5 error: {exc}", exc_info=True)
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return GameAction.RESET
            return GameAction.ACTION1

    def _choose_action_impl(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last_action_name = "RESET"
            return GameAction.RESET

        if latest_frame.full_reset and self._last_action_name != "RESET":
            self._last_action_name = "RESET"
            return GameAction.RESET

        grid = self._frame_to_grid(latest_frame)
        if latest_frame.available_actions:
            self._available_action_names = [
                _normalize_action_name(item)
                for item in latest_frame.available_actions
                if _normalize_action_name(item) != "RESET"
            ]

        result = self.v5.choose_action(
            frames=[grid],
            available_actions=self._available_action_names,
            game_state=latest_frame.state.name,
            levels_completed=latest_frame.levels_completed,
        )
        action_name = result["action"]
        self._last_action_name = action_name
        game_action = _NAME_TO_ACTION.get(action_name, GameAction.ACTION1)
        if action_name == "ACTION6" and result.get("x") is not None:
            game_action = GameAction.ACTION6
            game_action.set_data({
                "game_id": self.game_id,
                "x": int(result["x"]),
                "y": int(result["y"]),
            })
        else:
            game_action.set_data({"game_id": self.game_id})
        return game_action

    def _frame_to_grid(self, frame: FrameData) -> np.ndarray:
        if frame.frame:
            grid = frame.frame[-1] if isinstance(frame.frame, list) else frame.frame
            return np.array(grid, dtype=np.int32)
        return np.zeros((10, 10), dtype=np.int32)

    def end_game(self) -> Dict[str, Any]:
        won = self.frames[-1].state == GameState.WIN if self.frames else False
        summary = self.v5.end_game(won=won, game_id=self.game_id)
        summary["wrapper"] = {
            "elapsed_seconds": round(self._elapsed(), 2),
            "stop_reason": self._stop_reason,
        }
        return summary
