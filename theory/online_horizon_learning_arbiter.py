"""Online action-budget arbitration for long-horizon causal learning.

The arbiter never decides which goal is true and never assigns terminal credit.
It only decides whether saturated operator planning should yield action budget to
already observable causal uncertainty or to a nearby terminally testable goal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class HorizonLearningSignals:
    """Auditable online evidence available at one decision point."""

    active_causal_option: bool = False
    productive_causal_edges: int = 0
    unresolved_opened_options: int = 0
    active_mediated_requests: int = 0
    compiled_policies: int = 0
    open_successor_states: int = 0
    supported_mediated_hyperedges: int = 0
    terminal_test_status: str = ""
    terminal_test_distance: float | None = None
    causal_target_distance: float | None = None


@dataclass(frozen=True)
class HorizonLearningAllocation:
    """One bounded reservation decision without goal-value semantics."""

    reserve_learning: bool
    operator_action_budget: int | None
    priority: int
    reasons: Tuple[str, ...] = ()
    terminal_test_near: bool = False
    causal_uncertainty_present: bool = False


class OnlineHorizonLearningArbiter:
    """Allocate extended-horizon actions from live epistemic demand only."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_priority: int = 2,
        maximum_terminal_test_distance: float = 2.0,
        base_operator_action_budget: int = 12,
    ) -> None:
        self.enabled = bool(enabled)
        self.minimum_priority = max(1, int(minimum_priority))
        self.maximum_terminal_test_distance = max(
            0.0,
            float(maximum_terminal_test_distance),
        )
        self.base_operator_action_budget = max(
            1,
            int(base_operator_action_budget),
        )
        self._evaluations = 0
        self._reservations = 0
        self._releases = 0
        self._causal_uncertainty_reservations = 0
        self._terminal_test_reservations = 0
        self._priority_peak = 0
        self._budget_allocations: Counter[int] = Counter()
        self._last_allocation = HorizonLearningAllocation(
            reserve_learning=False,
            operator_action_budget=None,
            priority=0,
            reasons=("not_evaluated",),
        )

    def allocate(
        self,
        signals: HorizonLearningSignals,
    ) -> HorizonLearningAllocation:
        """Return a reservation proportional to current epistemic demand."""
        self._evaluations += 1
        if not self.enabled:
            allocation = HorizonLearningAllocation(
                reserve_learning=False,
                operator_action_budget=None,
                priority=0,
                reasons=("arbiter_disabled",),
            )
            self._releases += 1
            self._last_allocation = allocation
            return allocation

        priority = 0
        reasons = []
        causal_uncertainty = False
        if signals.active_causal_option:
            priority += 3
            reasons.append("active_causal_option")
            causal_uncertainty = True
        if signals.productive_causal_edges > 0:
            priority += 3
            reasons.append("productive_causal_edge")
            causal_uncertainty = True
        if signals.unresolved_opened_options > 0:
            priority += 2
            reasons.append("unresolved_opened_option")
            causal_uncertainty = True
        if signals.active_mediated_requests > 0:
            priority += 4
            reasons.append("active_mediated_request")
            causal_uncertainty = True
        if signals.compiled_policies > 0:
            priority += 3
            reasons.append("compiled_online_policy")
            causal_uncertainty = True
        if signals.open_successor_states > 0:
            priority += 2
            reasons.append("open_successor_state")
            causal_uncertainty = True
        if signals.supported_mediated_hyperedges > 0:
            priority += 2
            reasons.append("supported_mediated_hyperedge")
            causal_uncertainty = True

        terminal_near = _near(
            signals.terminal_test_distance,
            self.maximum_terminal_test_distance,
        ) and signals.terminal_test_status in {
            "needs_contrast",
            "terminal_supported",
        }
        if terminal_near:
            priority += (
                6
                if signals.terminal_test_status == "terminal_supported"
                else 5
            )
            reasons.append(
                f"near_{signals.terminal_test_status}_objective"
            )
        if causal_uncertainty and _near(
            signals.causal_target_distance,
            self.maximum_terminal_test_distance,
        ):
            priority += 2
            reasons.append("near_causal_target")

        reserve = priority >= self.minimum_priority
        budget = self._operator_budget(
            terminal_near=terminal_near,
            active_demand=bool(
                signals.active_causal_option
                or signals.active_mediated_requests > 0
            ),
        ) if reserve else None
        allocation = HorizonLearningAllocation(
            reserve_learning=reserve,
            operator_action_budget=budget,
            priority=priority,
            reasons=tuple(reasons or ("no_online_learning_demand",)),
            terminal_test_near=terminal_near,
            causal_uncertainty_present=causal_uncertainty,
        )
        self._priority_peak = max(self._priority_peak, priority)
        if reserve:
            self._reservations += 1
            self._budget_allocations[int(budget or 0)] += 1
            if causal_uncertainty:
                self._causal_uncertainty_reservations += 1
            if terminal_near:
                self._terminal_test_reservations += 1
        else:
            self._releases += 1
        self._last_allocation = allocation
        return allocation

    def summary(self) -> Dict[str, Any]:
        """Return compact counters suitable for held-out A/B attribution."""
        return {
            "enabled": self.enabled,
            "minimum_priority": self.minimum_priority,
            "maximum_terminal_test_distance": (
                self.maximum_terminal_test_distance
            ),
            "base_operator_action_budget": self.base_operator_action_budget,
            "evaluations": self._evaluations,
            "reservations": self._reservations,
            "releases": self._releases,
            "causal_uncertainty_reservations": (
                self._causal_uncertainty_reservations
            ),
            "terminal_test_reservations": self._terminal_test_reservations,
            "priority_peak": self._priority_peak,
            "budget_allocations": {
                str(budget): count
                for budget, count in sorted(self._budget_allocations.items())
            },
            "last_allocation": {
                "reserve_learning": self._last_allocation.reserve_learning,
                "operator_action_budget": (
                    self._last_allocation.operator_action_budget
                ),
                "priority": self._last_allocation.priority,
                "reasons": list(self._last_allocation.reasons),
                "terminal_test_near": (
                    self._last_allocation.terminal_test_near
                ),
                "causal_uncertainty_present": (
                    self._last_allocation.causal_uncertainty_present
                ),
            },
        }

    def _operator_budget(
        self,
        *,
        terminal_near: bool,
        active_demand: bool,
    ) -> int:
        if terminal_near:
            return max(2, self.base_operator_action_budget // 2)
        if active_demand:
            return max(4, (2 * self.base_operator_action_budget) // 3)
        return self.base_operator_action_budget


def _near(distance: float | None, maximum: float) -> bool:
    return bool(
        distance is not None
        and 0.0 < float(distance) <= float(maximum)
    )


__all__ = [
    "HorizonLearningAllocation",
    "HorizonLearningSignals",
    "OnlineHorizonLearningArbiter",
]
