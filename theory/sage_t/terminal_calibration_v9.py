"""Observed-safety terminal calibration for SAGE.T9.1 and later controllers."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .compiler import compile_transition_record
from .contracts import ActionCandidate, ObservedTransition
from .decision import SequenceAssessment
from .live_shadow_pilot_v4 import BaselineInclusiveDecisionEngine
from .live_shadow_pilot_v5 import MaterializedActionController


@dataclass(frozen=True)
class TerminalCalibrationPolicy:
    name: str
    minimum_safe_observations: int = 3
    prior_terminal: float = 1.0
    prior_safe: float = 1.0
    probability_floor: float = 0.05
    high_probability_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("terminal calibration policy needs a name")
        if int(self.minimum_safe_observations) < 1:
            raise ValueError("minimum safe observations must be positive")
        if float(self.prior_terminal) <= 0.0 or float(self.prior_safe) <= 0.0:
            raise ValueError("beta prior parameters must be positive")
        if not 0.0 <= float(self.probability_floor) < 0.5:
            raise ValueError("terminal floor must be in [0, 0.5)")
        if not 0.5 <= float(self.high_probability_threshold) <= 1.0:
            raise ValueError("high terminal threshold must be in [0.5, 1]")


T9_1_POLICIES: Mapping[str, TerminalCalibrationPolicy] = {
    policy.name: policy
    for policy in (
        TerminalCalibrationPolicy("safe_after_3", minimum_safe_observations=3),
        TerminalCalibrationPolicy("safe_after_5", minimum_safe_observations=5),
        TerminalCalibrationPolicy("safe_after_8", minimum_safe_observations=8),
    )
}


class ObservedSafetyCalibrator:
    """Cap unsupported high risk while latching every observed danger.

    Safe evidence pools by abstract action name inside a regime, so a new
    coordinate can benefit from repeated safe use of the same local operator.
    A terminal observation latches the exact materialized action forever in
    that regime; predictions for that action are then never reduced.
    """

    def __init__(self, policy: TerminalCalibrationPolicy) -> None:
        self.policy = policy
        self._safe: dict[tuple[int, str], int] = defaultdict(int)
        self._terminal: dict[tuple[int, str], int] = defaultdict(int)
        self._danger: set[tuple[int, str]] = set()
        self._calibrations = 0
        self._preserved_danger_predictions = 0

    @staticmethod
    def action_from_key(key: str) -> ActionCandidate:
        name, separator, raw = str(key).partition(":")
        if not separator:
            return ActionCandidate(name)
        return ActionCandidate(name, json.loads(raw))

    def calibrate(
        self,
        action: ActionCandidate,
        probability: float,
        *,
        regime_index: int = 0,
    ) -> float:
        raw = min(1.0, max(0.0, float(probability)))
        regime = max(0, int(regime_index))
        danger_key = (regime, action.key)
        if danger_key in self._danger:
            self._preserved_danger_predictions += int(
                raw >= self.policy.high_probability_threshold
            )
            return raw
        family_key = (regime, action.action_name)
        safe = self._safe[family_key]
        terminal = self._terminal[family_key]
        if (
            raw < self.policy.high_probability_threshold
            or safe < self.policy.minimum_safe_observations
        ):
            return raw
        empirical = (
            terminal + self.policy.prior_terminal
        ) / (
            safe
            + terminal
            + self.policy.prior_terminal
            + self.policy.prior_safe
        )
        cap = max(self.policy.probability_floor, empirical)
        calibrated = min(raw, cap)
        self._calibrations += int(calibrated < raw)
        return calibrated

    def sequence_cap(
        self,
        actions: Sequence[ActionCandidate],
        *,
        regime_index: int = 0,
    ) -> float:
        per_action = [
            self.calibrate(action, 1.0, regime_index=regime_index)
            for action in actions
        ]
        return 1.0 - math.prod(1.0 - value for value in per_action)

    def observe(self, evidence: ObservedTransition) -> None:
        if evidence.reset:
            return
        probability = evidence.observation.terminal_probability
        if probability is None or "terminal" not in evidence.observation.known_channels:
            return
        regime = max(0, int(evidence.state_before.regime_index))
        family_key = (regime, evidence.action.action_name)
        if float(probability) >= 0.5:
            self._terminal[family_key] += 1
            self._danger.add((regime, evidence.action.key))
        else:
            self._safe[family_key] += 1

    def observe_outcome(
        self,
        action: ActionCandidate,
        terminal: bool,
        *,
        regime_index: int = 0,
    ) -> None:
        regime = max(0, int(regime_index))
        family_key = (regime, action.action_name)
        if terminal:
            self._terminal[family_key] += 1
            self._danger.add((regime, action.key))
        else:
            self._safe[family_key] += 1

    def is_observed_danger(
        self,
        action: ActionCandidate,
        *,
        regime_index: int = 0,
    ) -> bool:
        return (max(0, int(regime_index)), action.key) in self._danger

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy.name,
            "safe_observations": sum(self._safe.values()),
            "terminal_observations": sum(self._terminal.values()),
            "danger_actions": len(self._danger),
            "calibrations": self._calibrations,
            "preserved_danger_predictions": self._preserved_danger_predictions,
            "regimes": len({regime for regime, _ in (*self._safe, *self._terminal)}),
        }


@dataclass
class TerminalCalibratedDecisionEngine(BaselineInclusiveDecisionEngine):
    calibrator: ObservedSafetyCalibrator | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def assess(self, *args: Any, **kwargs: Any) -> SequenceAssessment:
        assessment = super().assess(*args, **kwargs)
        if assessment.veto or self.calibrator is None:
            return assessment
        state = kwargs.get("state")
        regime = int(getattr(state, "regime_index", 0))
        cap = self.calibrator.sequence_cap(
            assessment.candidate.actions,
            regime_index=regime,
        )
        calibrated_risk = min(assessment.terminal_risk, cap)
        utility = (
            assessment.utility
            + 5.0 * assessment.terminal_risk
            - 5.0 * calibrated_risk
        )
        return replace(
            assessment,
            terminal_risk=calibrated_risk,
            utility=utility,
        )


class TerminalCalibratedMaterializedController(MaterializedActionController):
    """Materialized controller with post-posterior terminal calibration."""

    def __init__(
        self,
        *args: Any,
        terminal_policy: TerminalCalibrationPolicy,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.terminal_calibrator = ObservedSafetyCalibrator(terminal_policy)
        self.decision_engine = TerminalCalibratedDecisionEngine(
            executor=self.executor,
            maximum_sequences=self.config.maximum_sequences,
            maximum_particles=self.config.maximum_particles_per_decision,
            ordinary_horizon=self.config.ordinary_horizon,
            calibrator=self.terminal_calibrator,
        )

    def observe_transition(self, record: Any) -> None:
        try:
            evidence = compile_transition_record(
                record,
                regime_index=self._regime_index,
            )
        except (ArithmeticError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
            evidence = None
        super().observe_transition(record)
        if evidence is not None:
            self.terminal_calibrator.observe(evidence)

    def summary(self) -> Mapping[str, Any]:
        payload = dict(super().summary())
        payload["terminal_calibration"] = self.terminal_calibrator.snapshot()
        return payload


__all__ = [
    "T9_1_POLICIES",
    "ObservedSafetyCalibrator",
    "TerminalCalibratedDecisionEngine",
    "TerminalCalibratedMaterializedController",
    "TerminalCalibrationPolicy",
]
