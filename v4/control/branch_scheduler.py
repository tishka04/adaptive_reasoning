"""Branch scheduling and reset bookkeeping for V4."""

from __future__ import annotations

from typing import Any


class BranchScheduler:
    """Coordinate branch changes, project starvation, and reset pressure."""

    def __init__(self) -> None:
        self.branch_switches: int = 0
        self.last_reset_step: int = 0

    def on_branch_kill(self, memory: Any) -> None:
        self.branch_switches += 1
        self.last_reset_step = memory.game.total_actions
        memory.game.progress.start_new_branch()
        memory.fast.clear_plan()
        memory.fast.current_project_id = None

    def should_force_diversification(self, memory: Any) -> bool:
        recent = memory.fast.recent_project_ids
        if len(recent) < 8:
            return False
        dominant = max(recent.count(pid) for pid in set(recent))
        return dominant / max(len(recent), 1) > 0.65
