"""T8.6j-r2 context-memoized Repair V2 over the exact incremental posterior."""

from __future__ import annotations

from typing import Any

from .contracts import ObservedTransition
from .posterior_v2 import CalibratedProgramParticle
from .posterior_v9 import IncrementalMinimumKLProgramPosterior


class ContextMemoizedRepairProgramPosterior(
    IncrementalMinimumKLProgramPosterior
):
    """Attempt a falsified repair context once across reset branches."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._repaired_contexts: set[str] = set()
        self._repair_context_skips = 0

    @classmethod
    def _repair_context_key(cls, evidence: ObservedTransition) -> str:
        return "|".join(
            (
                cls._context_signature(evidence),
                repr(evidence.observation.full_signature),
                ",".join(sorted(evidence.events)),
            )
        )

    def repair(
        self,
        evidence: ObservedTransition | None = None,
    ) -> tuple[CalibratedProgramParticle, ...]:
        if not self._history:
            return ()
        target = evidence or self._history[-1]
        context = self._repair_context_key(target)
        if context in self._repaired_contexts:
            self._repair_context_skips += 1
            return ()
        self._repaired_contexts.add(context)
        return super().repair(target)

    def performance_snapshot(self) -> dict[str, int]:
        payload = dict(super().performance_snapshot())
        payload.update(
            {
                "unique_repair_contexts": len(self._repaired_contexts),
                "repair_context_skips": self._repair_context_skips,
            }
        )
        return payload


__all__ = ["ContextMemoizedRepairProgramPosterior"]
