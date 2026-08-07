"""T8.6j-r3 globally budgeted Repair V2 for long-horizon shadow runs."""

from __future__ import annotations

from typing import Any

from .contracts import ObservedTransition
from .posterior_v2 import CalibratedProgramParticle
from .posterior_v10 import ContextMemoizedRepairProgramPosterior


class BudgetedRepairProgramPosterior(ContextMemoizedRepairProgramPosterior):
    """Bound novel repair contexts while retaining exact posterior updates."""

    def __init__(
        self,
        *args: Any,
        maximum_repair_contexts: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(maximum_repair_contexts) < 0:
            raise ValueError("maximum repair contexts must be nonnegative")
        self.maximum_repair_contexts = int(maximum_repair_contexts)
        self._repair_budget_skips = 0

    def repair(
        self,
        evidence: ObservedTransition | None = None,
    ) -> tuple[CalibratedProgramParticle, ...]:
        if not self._history:
            return ()
        target = evidence or self._history[-1]
        context = self._repair_context_key(target)
        if (
            context not in self._repaired_contexts
            and len(self._repaired_contexts) >= self.maximum_repair_contexts
        ):
            self._repair_budget_skips += 1
            return ()
        return super().repair(target)

    def performance_snapshot(self) -> dict[str, int]:
        payload = dict(super().performance_snapshot())
        payload.update(
            {
                "maximum_repair_contexts": self.maximum_repair_contexts,
                "repair_budget_skips": self._repair_budget_skips,
            }
        )
        return payload


__all__ = ["BudgetedRepairProgramPosterior"]
