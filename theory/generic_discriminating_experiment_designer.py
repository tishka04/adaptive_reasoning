"""Generic active experiment designer for observable divergent predictions.

A15 extracts the experiment-selection core from the ft09 A14 harness. The
designer is game-agnostic: it receives live state, available concrete actions,
and unresolved hypotheses with predicted observable effects, then picks the
action/coordinates whose predictions disagree the most.

Trace-derived support may rank otherwise equivalent hypotheses, but it is never
counted as proof; revision still belongs to the live transition observer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from .epistemic_metrics import HypothesisStatus


@dataclass(frozen=True)
class DiscriminatingPrediction:
    """One unresolved predicted effect that can compete in an experiment."""

    key: str
    action: str
    source_color: int
    target_color: int | None = None
    family: str = "color_transform"
    predicate: str = ""
    predicted_outcome: str = ""
    relation: str = "modifies"
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED
    support: int = 0
    transition_support: int = 0
    epistemic_prior: float = 0.0
    prior_source_keys: Tuple[str, ...] = ()
    prior_counted_as_proof: bool = False

    @property
    def pair_colors(self) -> Tuple[int, int] | None:
        if self.target_color is None:
            return None
        return (self.source_color, self.target_color)

    @property
    def normalized_family(self) -> str:
        return _normalize_family(self.family)

    @property
    def predicate_name(self) -> str:
        if self.predicate:
            return str(self.predicate)
        if self.normalized_family == "color_transform":
            return "source_target_color_transform"
        return self.normalized_family

    @property
    def outcome(self) -> str:
        if self.predicted_outcome:
            return str(self.predicted_outcome)
        if self.normalized_family == "color_transform" and self.target_color is not None:
            return f"{self.source_color}->{self.target_color}"
        return "unknown"

    @property
    def divergence_group(self) -> Tuple[str, int, str, str]:
        return (
            self.action,
            int(self.source_color),
            self.normalized_family,
            self.predicate_name,
        )

    @property
    def outcome_signature(self) -> Tuple[str, str, str]:
        return (self.normalized_family, self.predicate_name, self.outcome)

    @classmethod
    def from_hypothesis(cls, hypothesis: Any) -> "DiscriminatingPrediction":
        """Create a prediction from a correspondence-like hypothesis object."""
        if isinstance(hypothesis, cls):
            return hypothesis

        pair = getattr(hypothesis, "pair_colors", None)
        source = getattr(hypothesis, "source_color", None)
        target = getattr(hypothesis, "target_color", None)
        if pair is not None:
            source, target = pair
        family = _normalize_family(getattr(hypothesis, "family", "color_transform"))
        if source is None:
            raise ValueError(f"hypothesis has no source color: {hypothesis!r}")
        if family == "color_transform" and target is None:
            raise ValueError(f"hypothesis has no target color: {hypothesis!r}")

        status = getattr(hypothesis, "status", None)
        if status is None and hasattr(hypothesis, "to_hypothesis"):
            status = hypothesis.to_hypothesis().status
        return cls(
            key=str(getattr(hypothesis, "key")),
            action=_normalize_action_name(getattr(hypothesis, "action", "")),
            source_color=int(source),
            target_color=None if target is None else int(target),
            family=family,
            predicate=str(getattr(hypothesis, "predicate", "")),
            predicted_outcome=str(getattr(hypothesis, "predicted_outcome", "")),
            relation=str(getattr(hypothesis, "relation", "modifies")),
            status=_normalize_status(status),
            support=int(getattr(hypothesis, "support", 0) or 0),
            transition_support=int(
                getattr(hypothesis, "transition_support", 0) or 0
            ),
            epistemic_prior=float(getattr(hypothesis, "epistemic_prior", 0.0) or 0.0),
            prior_source_keys=tuple(getattr(hypothesis, "prior_source_keys", ()) or ()),
            prior_counted_as_proof=bool(
                getattr(hypothesis, "prior_counted_as_proof", False)
            ),
        )


@dataclass(frozen=True)
class DesignedExperimentAction:
    """Concrete action selected by the generic designer."""

    name: str
    raw_action: Any
    action_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_args": dict(self.action_args),
        }


@dataclass(frozen=True)
class GenericDiscriminatingExperimentChoice:
    """A designed experiment that separates competing predictions."""

    action: DesignedExperimentAction
    competing_keys: Tuple[str, ...]
    predicted_pairs: Tuple[Tuple[int, int], ...]
    prediction_families: Tuple[str, ...]
    predicted_outcomes: Tuple[str, ...]
    observed_source_color: int
    expected_divergence: float
    epistemic_prior: float
    candidate_pool_size: int
    divergence_reason: str
    prior_source_keys: Tuple[str, ...] = ()
    selection_reason: str = "generic_designer:max_observable_prediction_divergence"
    trace_support_counted_as_proof: bool = False

    @property
    def has_divergent_predictions(self) -> bool:
        return len(set(self.predicted_outcomes)) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "competing_keys": list(self.competing_keys),
            "predicted_pairs": [list(pair) for pair in self.predicted_pairs],
            "prediction_families": list(self.prediction_families),
            "predicted_outcomes": list(self.predicted_outcomes),
            "observed_source_color": self.observed_source_color,
            "expected_divergence": self.expected_divergence,
            "epistemic_prior": self.epistemic_prior,
            "prior_source_keys": list(self.prior_source_keys),
            "candidate_pool_size": self.candidate_pool_size,
            "divergence_reason": self.divergence_reason,
            "selection_reason": self.selection_reason,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "has_divergent_predictions": self.has_divergent_predictions,
        }


class GenericDiscriminatingExperimentDesigner:
    """Pick the concrete action whose observable predictions diverge most."""

    def __init__(self, *, max_competing_hypotheses: int = 2) -> None:
        self.max_competing_hypotheses = max(2, int(max_competing_hypotheses))

    def design(
        self,
        *,
        hypotheses: Sequence[Any],
        live_grid: Any,
        available_actions: Sequence[Any],
        preferred_source_color: int | None = None,
        preferred_family: str | None = None,
    ) -> GenericDiscriminatingExperimentChoice | None:
        """Return the most discriminating concrete experiment, if one exists."""
        grid = np.asarray(live_grid, dtype=np.int32)
        predictions = [
            DiscriminatingPrediction.from_hypothesis(hypothesis)
            for hypothesis in hypotheses
        ]
        predictions = [
            prediction
            for prediction in predictions
            if prediction.status == HypothesisStatus.UNRESOLVED
        ]
        if preferred_family is not None:
            family = _normalize_family(preferred_family)
            predictions = [
                prediction
                for prediction in predictions
                if prediction.normalized_family == family
            ]
        if not predictions:
            return None

        choices: List[
            tuple[
                tuple[float, float, int, int, int],
                GenericDiscriminatingExperimentChoice,
            ]
        ] = []
        for index, raw_action in enumerate(available_actions):
            action = _action_view(raw_action)
            action_sources = _candidate_source_colors(
                grid,
                action,
                predictions,
                preferred_source_color=preferred_source_color,
            )
            for source_color in action_sources:
                compatible = [
                    prediction
                    for prediction in predictions
                    if prediction.action == action.name
                    and prediction.source_color == source_color
                ]
                by_group = _group_by_divergence(compatible)
                for group_predictions in by_group.values():
                    choice = _choice_for_group(
                        grid=grid,
                        action=action,
                        group_predictions=group_predictions,
                        source_color=source_color,
                        max_competing_hypotheses=self.max_competing_hypotheses,
                    )
                    if choice is None:
                        continue
                    choices.append(
                        (
                            (
                                choice.expected_divergence,
                                choice.epistemic_prior,
                                sum(
                                    prediction.support
                                    for prediction in group_predictions
                                    if prediction.key in choice.competing_keys
                                ),
                                int(np.sum(grid == source_color)),
                                -index,
                            ),
                            choice,
                        )
                    )
        if not choices:
            return None
        return max(choices, key=lambda item: item[0])[1]


def _choice_for_group(
    *,
    grid: np.ndarray,
    action: DesignedExperimentAction,
    group_predictions: Sequence[DiscriminatingPrediction],
    source_color: int,
    max_competing_hypotheses: int,
) -> GenericDiscriminatingExperimentChoice | None:
    unique = _unique_by_outcome(group_predictions)
    if len(unique) < 2:
        return None
    ranked = _rank_predictions(unique)
    selected = tuple(ranked[: max_competing_hypotheses])
    distinct_outcomes = {prediction.outcome_signature for prediction in unique}
    expected_divergence = float(len(distinct_outcomes))
    prior_total = sum(prediction.epistemic_prior for prediction in selected)
    prior_source_keys = _dedupe(
        source_key
        for prediction in selected
        for source_key in prediction.prior_source_keys
    )
    return GenericDiscriminatingExperimentChoice(
        action=action,
        competing_keys=tuple(prediction.key for prediction in selected),
        predicted_pairs=tuple(
            pair
            for pair in (prediction.pair_colors for prediction in selected)
            if pair is not None
        ),
        prediction_families=tuple(
            prediction.normalized_family for prediction in selected
        ),
        predicted_outcomes=tuple(prediction.outcome for prediction in selected),
        observed_source_color=source_color,
        expected_divergence=expected_divergence,
        epistemic_prior=prior_total,
        prior_source_keys=tuple(prior_source_keys),
        candidate_pool_size=len(unique),
        divergence_reason=_divergence_reason(selected[0]),
    )


def _action_view(action: Any) -> DesignedExperimentAction:
    name = getattr(action, "name", None)
    raw_action = getattr(action, "raw_action", getattr(action, "id", action))
    action_args = (
        getattr(action, "action_args", None)
        or getattr(action, "data", None)
        or {}
    )
    return DesignedExperimentAction(
        name=_normalize_action_name(name if name is not None else raw_action),
        raw_action=raw_action,
        action_args=dict(action_args),
    )


def _candidate_source_colors(
    grid: np.ndarray,
    action: DesignedExperimentAction,
    predictions: Sequence[DiscriminatingPrediction],
    *,
    preferred_source_color: int | None,
) -> List[int]:
    clicked_color = _action_pixel_color(grid, action.action_args)
    if clicked_color is not None:
        colors = [clicked_color]
    else:
        colors = _dedupe(
            int(prediction.source_color)
            for prediction in predictions
            if prediction.action == action.name
            and bool(np.any(grid == int(prediction.source_color)))
        )
    if preferred_source_color is not None:
        wanted = int(preferred_source_color)
        colors = [color for color in colors if color == wanted]
    return colors


def _group_by_divergence(
    predictions: Iterable[DiscriminatingPrediction],
) -> dict[Tuple[str, int, str, str], List[DiscriminatingPrediction]]:
    groups: dict[Tuple[str, int, str, str], List[DiscriminatingPrediction]] = {}
    for prediction in predictions:
        groups.setdefault(prediction.divergence_group, []).append(prediction)
    return groups


def _unique_by_outcome(
    predictions: Iterable[DiscriminatingPrediction],
) -> List[DiscriminatingPrediction]:
    seen: set[Tuple[str, str, str]] = set()
    result: List[DiscriminatingPrediction] = []
    for prediction in _rank_predictions(predictions):
        if prediction.outcome_signature in seen:
            continue
        seen.add(prediction.outcome_signature)
        result.append(prediction)
    return result


def _rank_predictions(
    predictions: Iterable[DiscriminatingPrediction],
) -> List[DiscriminatingPrediction]:
    return sorted(
        predictions,
        key=lambda prediction: (
            prediction.epistemic_prior,
            prediction.support,
            prediction.transition_support,
            prediction.key,
        ),
        reverse=True,
    )


def _action_pixel_color(
    grid: np.ndarray,
    action_args: dict[str, Any],
) -> int | None:
    if "x" not in action_args or "y" not in action_args:
        return None
    try:
        x = int(action_args["x"])
        y = int(action_args["y"])
    except (TypeError, ValueError):
        return None
    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
        return None
    return int(grid[y, x])


def _normalize_status(status: Any) -> HypothesisStatus:
    if isinstance(status, HypothesisStatus):
        return status
    if status is None:
        return HypothesisStatus.UNRESOLVED
    return HypothesisStatus(str(status))


def _normalize_family(family: Any) -> str:
    raw = str(family or "color_transform").strip().lower()
    aliases = {
        "color": "color_transform",
        "source_target_color_transform": "color_transform",
        "effect": "effect_scope",
        "scope": "effect_scope",
        "object_count_changes": "object_count",
        "object_count_delta": "object_count",
    }
    return aliases.get(raw, raw)


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _divergence_reason(prediction: DiscriminatingPrediction) -> str:
    family = prediction.normalized_family
    if family == "color_transform":
        return "same_source_different_target"
    if family == "effect_scope":
        return "same_source_different_effect_scope"
    if family == "object_count":
        return "same_source_different_object_count"
    if family == "relation":
        return f"same_source_different_{prediction.predicate_name}"
    return f"same_source_different_{family}"


def _normalize_action_name(action: Any) -> str:
    if isinstance(action, (int, np.integer)):
        if int(action) == 0:
            return "RESET"
        return f"ACTION{int(action)}"
    raw = str(action or "").strip().upper()
    if "." in raw:
        raw = raw.split(".")[-1]
    if raw.isdigit():
        if int(raw) == 0:
            return "RESET"
        return f"ACTION{raw}"
    return raw
