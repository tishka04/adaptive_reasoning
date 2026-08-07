"""Gate-controlled retirement registry for legacy independent arbiters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class LegacyArbiter(str, Enum):
    GOAL_GENERATOR = "goal_generator"
    SUBGOAL_CONFIDENCE = "subgoal_confidence"
    FRONTIER_SCORE = "frontier_score"
    TERMINAL_PREDICTOR = "terminal_predictor"


@dataclass(frozen=True)
class ConsolidationEntry:
    arbiter: LegacyArbiter
    retired: bool = False
    active_gate_passed: bool = False
    paired_ablation_passed: bool = False
    rollback_enabled: bool = True
    adapter: str = ""


class ConsolidationRegistry:
    """T6 state machine.

    It deliberately has no automatic promotion path: callers must provide
    evidence that the active gate and the arbiter-specific paired ablation
    passed.  Rollback remains enabled for every retirement.
    """

    A32_A40_ADAPTERS: Mapping[str, str] = {
        "A32": "provenance",
        "A33": "contradictions",
        "A34": "typed_assembly",
        "A35": "canonical_execution",
        "A36": "factorized_likelihood",
        "A37": "program_posterior",
        "A38": "discriminative_experimentation",
        "A39": "bayesian_decision",
        "A40": "consolidation",
    }

    def __init__(self) -> None:
        self._entries = {
            arbiter: ConsolidationEntry(
                arbiter=arbiter,
                adapter=f"sage_t_adapter:{arbiter.value}",
            )
            for arbiter in LegacyArbiter
        }

    def retire(
        self,
        arbiter: LegacyArbiter | str,
        *,
        active_gate_passed: bool,
        paired_ablation_passed: bool,
        rollback_enabled: bool = True,
    ) -> bool:
        key = LegacyArbiter(str(getattr(arbiter, "value", arbiter)))
        admitted = bool(
            active_gate_passed and paired_ablation_passed and rollback_enabled
        )
        self._entries[key] = ConsolidationEntry(
            arbiter=key,
            retired=admitted,
            active_gate_passed=bool(active_gate_passed),
            paired_ablation_passed=bool(paired_ablation_passed),
            rollback_enabled=bool(rollback_enabled),
            adapter=self._entries[key].adapter,
        )
        return admitted

    def rollback(self, arbiter: LegacyArbiter | str) -> None:
        key = LegacyArbiter(str(getattr(arbiter, "value", arbiter)))
        previous = self._entries[key]
        self._entries[key] = ConsolidationEntry(
            arbiter=key,
            retired=False,
            active_gate_passed=previous.active_gate_passed,
            paired_ablation_passed=previous.paired_ablation_passed,
            rollback_enabled=True,
            adapter=previous.adapter,
        )

    def is_authoritative(self, arbiter: LegacyArbiter | str) -> bool:
        key = LegacyArbiter(str(getattr(arbiter, "value", arbiter)))
        return not self._entries[key].retired

    def snapshot(self) -> Mapping[str, object]:
        return {
            "a32_a40_adapters": dict(self.A32_A40_ADAPTERS),
            "legacy_arbiters": {
                key.value: {
                    "retired": entry.retired,
                    "active_gate_passed": entry.active_gate_passed,
                    "paired_ablation_passed": (entry.paired_ablation_passed),
                    "rollback_enabled": entry.rollback_enabled,
                    "adapter": entry.adapter,
                }
                for key, entry in self._entries.items()
            },
        }


__all__ = [
    "ConsolidationEntry",
    "ConsolidationRegistry",
    "LegacyArbiter",
]
