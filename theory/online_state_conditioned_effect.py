"""State-conditioned online prediction of objective-direction effects.

Primitive actions in ARC-AGI-3 are often modal: the same action can advance an
objective in one visual mode and undo it in another.  This module learns that
direction only from live transitions.  Its latent modes are position-invariant
summaries of the current grid and objective deficit; terminal truth is never
inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from v3.schemas import GameObservation

from .online_terminal_objective import TerminalObjectiveHypothesis


class DirectionalEffectStatus(str, Enum):
    UNKNOWN = "unknown"
    NEEDS_MODE_CONTRAST = "needs_mode_contrast"
    PROGRESSIVE = "progressive"
    REGRESSIVE = "regressive"
    NEUTRAL = "neutral"
    UNSTABLE = "unstable"


@dataclass
class StateConditionedActionEvidence:
    """Observed objective-direction evidence in one latent mode."""

    option_id: str
    objective_id: str
    mode_signature: str
    action_signature: str
    attempts: int = 0
    progress_events: int = 0
    regression_events: int = 0
    stalls: int = 0
    unsafe_failures: int = 0
    total_progress: float = 0.0
    total_regression: float = 0.0
    trigger_observations: int = 0
    pursuit_observations: int = 0
    branches: set[int] = field(default_factory=set)
    contexts: set[str] = field(default_factory=set)
    effect_signatures: set[str] = field(default_factory=set)
    next_mode_signatures: set[str] = field(default_factory=set)

    @property
    def status(self) -> DirectionalEffectStatus:
        if self.progress_events and self.regression_events:
            return DirectionalEffectStatus.UNSTABLE
        if self.progress_events:
            return DirectionalEffectStatus.PROGRESSIVE
        if self.regression_events:
            return DirectionalEffectStatus.REGRESSIVE
        if self.attempts >= 2:
            return DirectionalEffectStatus.NEUTRAL
        return DirectionalEffectStatus.UNKNOWN

    @property
    def expected_gain(self) -> float:
        return (
            self.total_progress
            - self.total_regression
            - 2.0 * self.unsafe_failures
        ) / (self.attempts + 1.0)

    @property
    def confidence(self) -> float:
        directional = self.progress_events + self.regression_events
        return directional / (self.attempts + 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "objective_id": self.objective_id,
            "mode_signature": self.mode_signature,
            "action_signature": self.action_signature,
            "status": self.status.value,
            "attempts": self.attempts,
            "progress_events": self.progress_events,
            "regression_events": self.regression_events,
            "stalls": self.stalls,
            "unsafe_failures": self.unsafe_failures,
            "total_progress": round(float(self.total_progress), 4),
            "total_regression": round(float(self.total_regression), 4),
            "expected_gain": round(float(self.expected_gain), 4),
            "confidence": round(float(self.confidence), 4),
            "trigger_observations": self.trigger_observations,
            "pursuit_observations": self.pursuit_observations,
            "branches": sorted(self.branches),
            "contexts": sorted(self.contexts),
            "effect_signatures": sorted(self.effect_signatures),
            "next_mode_signatures": sorted(self.next_mode_signatures),
        }


@dataclass(frozen=True)
class DirectionalActionPrediction:
    option_id: str
    objective_id: str
    mode_signature: str
    action_signature: str
    status: DirectionalEffectStatus
    expected_gain: float
    confidence: float
    compatible: bool
    exact_mode_evidence: bool
    reversible_across_modes: bool
    reason: str

    @property
    def selection_rank(self) -> int:
        return {
            DirectionalEffectStatus.PROGRESSIVE: 4,
            DirectionalEffectStatus.NEEDS_MODE_CONTRAST: 3,
            DirectionalEffectStatus.UNKNOWN: 2,
            DirectionalEffectStatus.UNSTABLE: 1,
            DirectionalEffectStatus.NEUTRAL: 0,
            DirectionalEffectStatus.REGRESSIVE: -1,
        }[self.status]


class OnlineStateConditionedEffectModel:
    """Learn action direction per objective and recurring latent visual mode."""

    def __init__(self) -> None:
        self._evidence: Dict[
            Tuple[str, str, str, str],
            StateConditionedActionEvidence,
        ] = {}
        self._observations = 0
        self._trigger_observations = 0
        self._pursuit_observations = 0
        self._progress_events = 0
        self._regression_events = 0
        self._stall_events = 0
        self._predictions = 0
        self._exact_predictions = 0
        self._contrast_predictions = 0
        self._progressive_selections = 0
        self._contrast_selections = 0
        self._blocked_regressive_actions = 0

    def observe(
        self,
        *,
        option_id: str,
        objective: TerminalObjectiveHypothesis,
        observation_before: GameObservation,
        observation_after: GameObservation,
        action_signature: str,
        effect_signature: str,
        branch_index: int,
        context_signature: str,
        source: str,
        unsafe: bool = False,
    ) -> Dict[str, Any]:
        """Record the signed objective delta caused in the current latent mode."""
        before = objective.distance(observation_before)
        after = objective.distance(observation_after)
        if before is None or after is None:
            return {
                "observed": False,
                "mode_signature": "",
                "status": DirectionalEffectStatus.UNKNOWN.value,
                "gain": 0.0,
                "reversible_across_modes": False,
            }
        mode = latent_mode_signature(observation_before, objective)
        next_mode = latent_mode_signature(observation_after, objective)
        key = (
            str(option_id),
            objective.objective_id,
            mode,
            str(action_signature),
        )
        evidence = self._evidence.get(key)
        if evidence is None:
            evidence = StateConditionedActionEvidence(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                action_signature=str(action_signature),
            )
            self._evidence[key] = evidence
        evidence.attempts += 1
        evidence.unsafe_failures += int(bool(unsafe))
        evidence.branches.add(int(branch_index))
        evidence.contexts.add(str(context_signature))
        if effect_signature:
            evidence.effect_signatures.add(str(effect_signature))
        evidence.next_mode_signatures.add(next_mode)
        if str(source) == "trigger":
            evidence.trigger_observations += 1
            self._trigger_observations += 1
        else:
            evidence.pursuit_observations += 1
            self._pursuit_observations += 1

        gain = float(before) - float(after)
        if gain > 0.0:
            evidence.progress_events += 1
            evidence.total_progress += gain
            self._progress_events += 1
        elif gain < 0.0:
            evidence.regression_events += 1
            evidence.total_regression += abs(gain)
            self._regression_events += 1
        else:
            evidence.stalls += 1
            self._stall_events += 1
        self._observations += 1
        return {
            "observed": True,
            "mode_signature": mode,
            "next_mode_signature": next_mode,
            "status": evidence.status.value,
            "gain": gain,
            "reversible_across_modes": self.is_reversible(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                action_signature=str(action_signature),
            ),
        }

    def predict(
        self,
        *,
        option_id: str,
        objective: TerminalObjectiveHypothesis,
        observation: GameObservation,
        action_signature: str,
    ) -> DirectionalActionPrediction:
        mode = latent_mode_signature(observation, objective)
        exact = self._evidence.get((
            str(option_id),
            objective.objective_id,
            mode,
            str(action_signature),
        ))
        related = self._related_evidence(
            option_id=str(option_id),
            objective_id=objective.objective_id,
            action_signature=str(action_signature),
        )
        directional_related = [
            item
            for item in related
            if item.progress_events > 0 or item.regression_events > 0
        ]
        reversible = self._is_reversible_evidence(related)
        self._predictions += 1
        if exact is not None:
            self._exact_predictions += 1
            status = exact.status
            other_directional_modes = [
                item
                for item in directional_related
                if item.mode_signature != mode
            ]
            contrast_stalled = bool(
                status == DirectionalEffectStatus.UNKNOWN
                and exact.attempts > 0
                and other_directional_modes
            )
            if contrast_stalled:
                status = DirectionalEffectStatus.NEUTRAL
            compatible = status not in {
                DirectionalEffectStatus.REGRESSIVE,
                DirectionalEffectStatus.NEUTRAL,
            }
            return DirectionalActionPrediction(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                action_signature=str(action_signature),
                status=status,
                expected_gain=exact.expected_gain,
                confidence=exact.confidence,
                compatible=compatible,
                exact_mode_evidence=True,
                reversible_across_modes=reversible,
                reason=(
                    "exact latent mode previously reduced the objective"
                    if status == DirectionalEffectStatus.PROGRESSIVE
                    else (
                        "exact latent mode previously increased the objective"
                        if status == DirectionalEffectStatus.REGRESSIVE
                        else (
                            "one cross-mode contrast stalled in this latent mode"
                            if contrast_stalled
                            else "exact latent-mode evidence is non-progressive"
                        )
                    )
                ),
            )
        if directional_related:
            self._contrast_predictions += 1
            directional_gain = sum(
                item.expected_gain for item in directional_related
            ) / len(directional_related)
            return DirectionalActionPrediction(
                option_id=str(option_id),
                objective_id=objective.objective_id,
                mode_signature=mode,
                action_signature=str(action_signature),
                status=DirectionalEffectStatus.NEEDS_MODE_CONTRAST,
                expected_gain=directional_gain,
                confidence=0.0,
                compatible=True,
                exact_mode_evidence=False,
                reversible_across_modes=reversible,
                reason=(
                    "same action has directional evidence in another latent mode; "
                    "test this mode once"
                ),
            )
        return DirectionalActionPrediction(
            option_id=str(option_id),
            objective_id=objective.objective_id,
            mode_signature=mode,
            action_signature=str(action_signature),
            status=DirectionalEffectStatus.UNKNOWN,
            expected_gain=0.0,
            confidence=0.0,
            compatible=True,
            exact_mode_evidence=False,
            reversible_across_modes=False,
            reason="no directional evidence for this action and latent mode",
        )

    def note_selection(self, prediction: DirectionalActionPrediction) -> None:
        if prediction.status == DirectionalEffectStatus.PROGRESSIVE:
            self._progressive_selections += 1
        elif prediction.status == DirectionalEffectStatus.NEEDS_MODE_CONTRAST:
            self._contrast_selections += 1

    def note_blocked(self, prediction: DirectionalActionPrediction) -> None:
        if prediction.status == DirectionalEffectStatus.REGRESSIVE:
            self._blocked_regressive_actions += 1

    def is_reversible(
        self,
        *,
        option_id: str,
        objective_id: str,
        action_signature: str,
    ) -> bool:
        return self._is_reversible_evidence(self._related_evidence(
            option_id=str(option_id),
            objective_id=str(objective_id),
            action_signature=str(action_signature),
        ))

    def summary(self) -> Dict[str, Any]:
        evidence = sorted(
            self._evidence.values(),
            key=lambda item: (
                item.option_id,
                item.objective_id,
                item.mode_signature,
                item.action_signature,
            ),
        )
        action_keys = {
            (item.option_id, item.objective_id, item.action_signature)
            for item in evidence
        }
        statuses = {
            status.value: sum(item.status == status for item in evidence)
            for status in DirectionalEffectStatus
            if status != DirectionalEffectStatus.NEEDS_MODE_CONTRAST
        }
        return {
            "observations": self._observations,
            "trigger_observations": self._trigger_observations,
            "pursuit_observations": self._pursuit_observations,
            "progress_events": self._progress_events,
            "regression_events": self._regression_events,
            "stall_events": self._stall_events,
            "latent_modes": len({item.mode_signature for item in evidence}),
            "mode_action_models": len(evidence),
            "action_objective_models": len(action_keys),
            "statuses": statuses,
            "reversible_action_objectives": sum(
                self.is_reversible(
                    option_id=option_id,
                    objective_id=objective_id,
                    action_signature=action_signature,
                )
                for option_id, objective_id, action_signature in action_keys
            ),
            "predictions": self._predictions,
            "exact_mode_predictions": self._exact_predictions,
            "mode_contrast_predictions": self._contrast_predictions,
            "progressive_selections": self._progressive_selections,
            "mode_contrast_selections": self._contrast_selections,
            "blocked_regressive_actions": self._blocked_regressive_actions,
            "hypotheses": [item.to_dict() for item in evidence],
        }

    def _related_evidence(
        self,
        *,
        option_id: str,
        objective_id: str,
        action_signature: str,
    ) -> list[StateConditionedActionEvidence]:
        return [
            item for item in self._evidence.values()
            if item.option_id == str(option_id)
            and item.objective_id == str(objective_id)
            and item.action_signature == str(action_signature)
        ]

    @staticmethod
    def _is_reversible_evidence(
        evidence: Sequence[StateConditionedActionEvidence],
    ) -> bool:
        progressive_modes = {
            item.mode_signature for item in evidence if item.progress_events > 0
        }
        regressive_modes = {
            item.mode_signature for item in evidence if item.regression_events > 0
        }
        return bool(progressive_modes and regressive_modes)


def latent_mode_signature(
    observation: GameObservation,
    objective: TerminalObjectiveHypothesis,
) -> str:
    """Return a position-invariant objective-conditioned visual mode."""
    grid = np.asarray(observation.raw_grid, dtype=np.int32)
    values, counts = np.unique(grid, return_counts=True)
    color_counts = {
        int(value): int(count)
        for value, count in zip(values.tolist(), counts.tolist())
    }
    ranked_colors = sorted(
        color_counts.items(),
        key=lambda item: (item[1], -item[0]),
        reverse=True,
    )[:5]
    dominant_color, dominant_count = ranked_colors[0] if ranked_colors else (-1, 0)
    distance = objective.distance(observation)
    source = objective.source_color
    target = objective.target_color
    source_pixels = 0 if source is None else color_counts.get(int(source), 0)
    target_pixels = 0 if target is None else color_counts.get(int(target), 0)
    source_objects = sum(
        obj.value == int(source) for obj in observation.objects
    ) if source is not None else 0
    target_objects = sum(
        obj.value == int(target) for obj in observation.objects
    ) if target is not None else 0
    color_profile = ",".join(
        f"{color}:{_count_bucket(count)}" for color, count in ranked_colors
    )
    return "|".join((
        f"latent-mode:{objective.family}",
        f"shape:{grid.shape[0]}x{grid.shape[1]}",
        f"distance:{_distance_bucket(distance)}",
        f"source:{source}:{_count_bucket(source_pixels)}:{source_objects}",
        f"target:{target}:{_count_bucket(target_pixels)}:{target_objects}",
        f"dominant:{dominant_color}:{_ratio_bucket(dominant_count, grid.size)}",
        f"colors:{color_profile}",
    ))


def _count_bucket(count: int) -> str:
    value = max(0, int(count))
    if value <= 2:
        return str(value)
    if value <= 4:
        return "3-4"
    if value <= 16:
        return "5-16"
    if value <= 64:
        return "17-64"
    return "65+"


def _distance_bucket(distance: float | None) -> str:
    if distance is None:
        return "unknown"
    value = float(distance)
    if value <= 2.0:
        return f"{int(value)}"
    if value <= 4.0:
        return "3-4"
    if value <= 8.0:
        return "5-8"
    if value <= 32.0:
        return "9-32"
    return "33+"


def _ratio_bucket(count: int, total: int) -> str:
    ratio = 0.0 if total <= 0 else float(count) / float(total)
    if ratio < 0.25:
        return "sparse"
    if ratio < 0.5:
        return "minority"
    if ratio < 0.8:
        return "majority"
    return "dominant"


__all__ = [
    "DirectionalActionPrediction",
    "DirectionalEffectStatus",
    "OnlineStateConditionedEffectModel",
    "StateConditionedActionEvidence",
    "latent_mode_signature",
]
