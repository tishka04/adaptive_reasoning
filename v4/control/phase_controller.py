"""Phase selection for V4."""

from __future__ import annotations

from typing import Any

from ..schemas import ObservationV4


class PhaseController:
    """Select epistemic phase based on current world understanding."""

    def select_phase(self, obs: ObservationV4, memory: Any) -> str:
        progress = memory.game.progress
        lp, sp, tp = progress.scores()

        if progress.should_kill_branch():
            return "crisis"
        if memory.fast.just_completed_level:
            return "compression"
        if tp > 0.20:
            return "closure_pressure"
        if sp > 0.15:
            return "project_emergence"
        if memory.game.avg_operator_confidence() > 0.45:
            return "mechanical_stabilization"
        return "sensory_ignorance"
