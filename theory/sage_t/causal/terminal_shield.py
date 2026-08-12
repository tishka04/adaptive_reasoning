"""Replay-confirmed multi-step terminal shield for symbolic exploration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .archive import ArchiveEdge
from .contracts import GroundedAction

SHIELD_FORMAT = "sage-t12.1-terminal-shield-v1"


@dataclass(frozen=True)
class TerminalRiskSupport:
    cell_id: str
    action_key: str
    trials: int = 0
    terminal_failures: int = 0
    progress_successes: int = 0
    minimum_failure_distance: int | None = None
    maximum_failure_distance: int | None = None

    @property
    def confirmed_unsafe(self) -> bool:
        return (
            self.trials >= 2
            and self.terminal_failures == self.trials
            and self.progress_successes == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "action_key": self.action_key,
            "trials": self.trials,
            "terminal_failures": self.terminal_failures,
            "progress_successes": self.progress_successes,
            "minimum_failure_distance": self.minimum_failure_distance,
            "maximum_failure_distance": self.maximum_failure_distance,
            "confirmed_unsafe": self.confirmed_unsafe,
        }


@dataclass(frozen=True)
class ShieldDecision:
    allowed: bool
    reason: str
    support: TerminalRiskSupport


class MultiStepTerminalShield:
    """Veto only replay-confirmed terminal continuations.

    One original terminal trace plus one exact reproduction provide the two
    independent observations required for authority. Unknown actions remain
    explorable, which prevents the shield from collapsing Go-Explore into a
    conservative no-op policy.
    """

    def __init__(self, *, horizon: int = 64, minimum_support: int = 2) -> None:
        self.horizon = max(2, min(64, int(horizon)))
        self.minimum_support = max(2, int(minimum_support))
        self._support: dict[tuple[str, str], TerminalRiskSupport] = {}
        self.confirmed_terminal_traces = 0
        self.unconfirmed_terminal_traces = 0
        self.progress_observations = 0
        self.vetoes = 0

    def support(self, cell_id: str, action: GroundedAction) -> TerminalRiskSupport:
        key = (str(cell_id), action.key)
        return self._support.get(
            key,
            TerminalRiskSupport(cell_id=str(cell_id), action_key=action.key),
        )

    def observe_progress(self, edge: ArchiveEdge) -> None:
        if edge.level_delta <= 0 and not edge.success:
            return
        key = (edge.source_cell_id, edge.action.key)
        current = self.support(edge.source_cell_id, edge.action)
        self._support[key] = TerminalRiskSupport(
            cell_id=current.cell_id,
            action_key=current.action_key,
            trials=current.trials,
            terminal_failures=current.terminal_failures,
            progress_successes=current.progress_successes + 1,
            minimum_failure_distance=current.minimum_failure_distance,
            maximum_failure_distance=current.maximum_failure_distance,
        )
        self.progress_observations += 1

    def record_terminal_trace(
        self,
        edges: Sequence[ArchiveEdge],
        *,
        exact_replay_confirmed: bool,
    ) -> tuple[TerminalRiskSupport, ...]:
        """Propagate terminal evidence backward until progress or horizon."""

        selected: list[ArchiveEdge] = []
        for edge in reversed(tuple(edges)):
            if edge.level_delta > 0 or edge.success:
                break
            selected.append(edge)
            if len(selected) >= self.horizon:
                break
        selected.reverse()
        if not selected or not selected[-1].terminal or selected[-1].success:
            raise ValueError("terminal shield needs a failing terminal suffix")
        observation_count = 2 if exact_replay_confirmed else 1
        if exact_replay_confirmed:
            self.confirmed_terminal_traces += 1
        else:
            self.unconfirmed_terminal_traces += 1
        updated = []
        total = len(selected)
        for index, edge in enumerate(selected):
            distance = total - index
            key = (edge.source_cell_id, edge.action.key)
            current = self.support(edge.source_cell_id, edge.action)
            minimum = (
                distance
                if current.minimum_failure_distance is None
                else min(current.minimum_failure_distance, distance)
            )
            maximum = (
                distance
                if current.maximum_failure_distance is None
                else max(current.maximum_failure_distance, distance)
            )
            next_support = TerminalRiskSupport(
                cell_id=current.cell_id,
                action_key=current.action_key,
                trials=current.trials + observation_count,
                terminal_failures=current.terminal_failures + observation_count,
                progress_successes=current.progress_successes,
                minimum_failure_distance=minimum,
                maximum_failure_distance=maximum,
            )
            self._support[key] = next_support
            updated.append(next_support)
        return tuple(updated)

    def evaluate(self, cell_id: str, action: GroundedAction) -> ShieldDecision:
        support = self.support(cell_id, action)
        unsafe = (
            support.trials >= self.minimum_support
            and support.terminal_failures == support.trials
            and support.progress_successes == 0
        )
        if unsafe:
            self.vetoes += 1
        return ShieldDecision(
            allowed=not unsafe,
            reason=(
                "confirmed_multi_step_terminal_veto"
                if unsafe
                else "unknown_or_not_confirmed_safe_to_explore"
            ),
            support=support,
        )

    def allows(self, cell_id: str, action: GroundedAction) -> bool:
        return self.evaluate(cell_id, action).allowed

    def metrics(self) -> dict[str, Any]:
        risks = tuple(self._support.values())
        distances = [
            item.maximum_failure_distance
            for item in risks
            if item.maximum_failure_distance is not None
        ]
        return {
            "horizon": self.horizon,
            "minimum_support": self.minimum_support,
            "risk_entries": len(risks),
            "confirmed_unsafe_actions": sum(
                item.trials >= self.minimum_support
                and item.terminal_failures == item.trials
                and item.progress_successes == 0
                for item in risks
            ),
            "confirmed_terminal_traces": self.confirmed_terminal_traces,
            "unconfirmed_terminal_traces": self.unconfirmed_terminal_traces,
            "progress_observations": self.progress_observations,
            "multi_step_hazard_observed": any(distance > 1 for distance in distances),
            "maximum_failure_distance": max(distances, default=0),
            "vetoes": self.vetoes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": SHIELD_FORMAT,
            "horizon": self.horizon,
            "minimum_support": self.minimum_support,
            "support": [
                self._support[key].to_dict() for key in sorted(self._support)
            ],
            "confirmed_terminal_traces": self.confirmed_terminal_traces,
            "unconfirmed_terminal_traces": self.unconfirmed_terminal_traces,
            "progress_observations": self.progress_observations,
            "vetoes": self.vetoes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MultiStepTerminalShield:
        if payload.get("format_version") != SHIELD_FORMAT:
            raise ValueError("unsupported terminal-shield payload")
        shield = cls(
            horizon=int(payload.get("horizon", 64)),
            minimum_support=int(payload.get("minimum_support", 2)),
        )
        for row in payload.get("support", ()):
            item = TerminalRiskSupport(
                cell_id=str(row["cell_id"]),
                action_key=str(row["action_key"]),
                trials=int(row.get("trials", 0)),
                terminal_failures=int(row.get("terminal_failures", 0)),
                progress_successes=int(row.get("progress_successes", 0)),
                minimum_failure_distance=(
                    None
                    if row.get("minimum_failure_distance") is None
                    else int(row["minimum_failure_distance"])
                ),
                maximum_failure_distance=(
                    None
                    if row.get("maximum_failure_distance") is None
                    else int(row["maximum_failure_distance"])
                ),
            )
            shield._support[(item.cell_id, item.action_key)] = item
        shield.confirmed_terminal_traces = int(
            payload.get("confirmed_terminal_traces", 0)
        )
        shield.unconfirmed_terminal_traces = int(
            payload.get("unconfirmed_terminal_traces", 0)
        )
        shield.progress_observations = int(payload.get("progress_observations", 0))
        shield.vetoes = int(payload.get("vetoes", 0))
        return shield


__all__ = [
    "SHIELD_FORMAT",
    "MultiStepTerminalShield",
    "ShieldDecision",
    "TerminalRiskSupport",
]
