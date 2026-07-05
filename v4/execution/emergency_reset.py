"""Emergency reset policy for V4."""

from __future__ import annotations


class EmergencyReset:
    """Decide when the current internal branch should be abandoned."""

    def __init__(self) -> None:
        self._last_branch_reset = -1

    def should_reset(self, memory) -> bool:
        phase = memory.fast.current_phase
        state = memory.game.progress.state
        sterility_risk = (
            memory.game.learning.sterility_predictor.predict(memory)
            if getattr(memory.game, "learning", None) is not None else 0.0
        )
        world_reliability = (
            memory.game.learning.world_reliability.estimate(memory)
            if getattr(memory.game, "learning", None) is not None else 0.5
        )
        if memory.game.progress.should_kill_branch():
            return True
        if memory.game.total_actions >= 24 and sterility_risk > 0.82:
            return True
        if phase == "closure_pressure" and state.terminal_stall_steps >= 80:
            return True
        if phase == "closure_pressure" and state.terminal_stall_steps >= 40 and world_reliability < 0.28:
            return True
        if memory.game.total_actions >= 40 and world_reliability < 0.12 and state.sp < 0.08:
            return True
        if phase == "crisis" and state.branch_id != self._last_branch_reset:
            return True
        return False

    def mark_reset(self, memory) -> None:
        self._last_branch_reset = memory.game.progress.state.branch_id
