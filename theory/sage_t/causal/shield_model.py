"""Progress-protected multi-step terminal shield for SAGE.T12.3b."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import GroundedAction
from .terminal_shield import MultiStepTerminalShield

PROGRESS_PROTECTED_SHIELD_FORMAT = "sage-t12.3b-progress-protected-shield-v1"


class ProgressProtectedTerminalShield:
    """Wrap the T12.1 shield while preserving replay-confirmed progress routes.

    Terminal evidence is allowed to propagate for up to 64 steps.  A
    state/action pair that belongs to either exact T12.3a progress witness is
    nevertheless always allowed.  This implements the preregistered
    lexicographic rule: observed progress outranks terminal-risk induction.
    """

    def __init__(
        self,
        *,
        base: MultiStepTerminalShield | None = None,
        protected_pairs: Sequence[tuple[str, str]] = (),
        witness_ids: Sequence[str] = (),
    ) -> None:
        self.base = base or MultiStepTerminalShield(horizon=64, minimum_support=2)
        self.protected_pairs = frozenset(
            (str(cell_id), str(action_key)) for cell_id, action_key in protected_pairs
        )
        self.witness_ids = tuple(sorted({str(value) for value in witness_ids}))
        self.protected_decisions = 0
        self.protected_conflict_overrides = 0

    def is_protected(self, cell_id: str, action: GroundedAction) -> bool:
        return (str(cell_id), action.key) in self.protected_pairs

    def allows(self, cell_id: str, action: GroundedAction) -> bool:
        if self.is_protected(cell_id, action):
            self.protected_decisions += 1
            if self.base.support(cell_id, action).confirmed_unsafe:
                self.protected_conflict_overrides += 1
            return True
        return self.base.allows(cell_id, action)

    def metrics(self) -> dict[str, Any]:
        base_metrics = self.base.metrics()
        unsafe_pairs = {
            (str(item["cell_id"]), str(item["action_key"]))
            for item in self.base.to_dict().get("support", ())
            if bool(item.get("confirmed_unsafe"))
        }
        protected_conflicts = sum(pair in unsafe_pairs for pair in self.protected_pairs)
        return {
            **base_metrics,
            "protected_action_pairs": len(self.protected_pairs),
            "protected_witnesses": len(self.witness_ids),
            "protected_decisions": self.protected_decisions,
            "protected_conflicts": protected_conflicts,
            "protected_conflict_overrides": self.protected_conflict_overrides,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PROGRESS_PROTECTED_SHIELD_FORMAT,
            "base_shield": self.base.to_dict(),
            "protected_pairs": [
                {"cell_id": cell_id, "action_key": action_key}
                for cell_id, action_key in sorted(self.protected_pairs)
            ],
            "witness_ids": list(self.witness_ids),
            "protected_decisions": self.protected_decisions,
            "protected_conflict_overrides": self.protected_conflict_overrides,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProgressProtectedTerminalShield:
        if payload.get("format_version") != PROGRESS_PROTECTED_SHIELD_FORMAT:
            raise ValueError("unsupported progress-protected shield payload")
        shield = cls(
            base=MultiStepTerminalShield.from_dict(dict(payload["base_shield"])),
            protected_pairs=tuple(
                (str(item["cell_id"]), str(item["action_key"]))
                for item in payload.get("protected_pairs", ())
            ),
            witness_ids=tuple(str(value) for value in payload.get("witness_ids", ())),
        )
        shield.protected_decisions = int(payload.get("protected_decisions", 0))
        shield.protected_conflict_overrides = int(
            payload.get("protected_conflict_overrides", 0)
        )
        return shield


__all__ = [
    "PROGRESS_PROTECTED_SHIELD_FORMAT",
    "ProgressProtectedTerminalShield",
]
