"""SAGE.T.A38: structured causal prediction/observation comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .contracts import PredictionDistribution, TransitionEvidence
from .executor import CausalExecutor

DEFAULT_CHANNEL_WEIGHTS: Mapping[str, float] = {
    "variables": 1.0,
    "objects": 1.0,
    "relations": 1.0,
    "patch": 0.5,
    "progress": 2.0,
    "terminal": 4.0,
    "goal": 2.0,
    "level": 2.0,
}


@dataclass(frozen=True)
class ParticleComparison:
    program_hash: str
    prediction: PredictionDistribution
    log_likelihood: float
    channel_errors: Mapping[str, float]
    channel_responsibility: Mapping[str, float]
    responsibility: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_errors", MappingProxyType(dict(self.channel_errors)))
        object.__setattr__(
            self,
            "channel_responsibility",
            MappingProxyType(dict(self.channel_responsibility)),
        )
        object.__setattr__(self, "responsibility", MappingProxyType(dict(self.responsibility)))


def compare_particle(
    *,
    program: object,
    evidence: TransitionEvidence,
    executor: CausalExecutor,
) -> ParticleComparison:
    prediction = executor.predict_step(program, evidence.state_before, evidence.action)
    weights = dict(DEFAULT_CHANNEL_WEIGHTS)
    weights.update(program.observation_model.channel_weights)
    declared_variables = tuple(item.variable_id for item in program.variables)
    available_errors = _channel_errors(
        prediction,
        evidence,
        declared_variables=declared_variables,
    )
    declared_channels = set(program.observation_model.channels)
    if "goal" in declared_channels:
        declared_channels.add("level")
    errors = {
        channel: error
        for channel, error in available_errors.items()
        if channel in declared_channels
    }
    floor = float(program.observation_model.noise_floor)
    log_likelihood = 0.0
    weighted_errors: dict[str, float] = {}
    for channel, error in errors.items():
        weight = max(0.0, float(weights.get(channel, 1.0)))
        clipped = min(1.0, max(0.0, float(error)))
        log_likelihood += weight * math.log(max(floor, 1.0 - clipped))
        weighted_errors[channel] = weight * clipped
    observation_module = program.observation_model.neural_module_id
    if observation_module is not None:
        log_likelihood += executor.mechanism_registry.observation_log_likelihood(
            observation_module,
            prediction,
            evidence,
        )
    total_error = sum(weighted_errors.values())
    channel_responsibility = {
        channel: (value / total_error if total_error > 0.0 else 0.0)
        for channel, value in weighted_errors.items()
    }
    components = {
        "binding": weighted_errors.get("objects", 0.0),
        "dynamics": sum(
            weighted_errors.get(channel, 0.0)
            for channel in ("variables", "relations", "progress")
        ),
        "goal": sum(
            weighted_errors.get(channel, 0.0)
            for channel in ("goal", "level", "terminal")
        ),
        "observation_model": weighted_errors.get("patch", 0.0),
    }
    component_total = sum(components.values())
    responsibility = {
        component: (value / component_total if component_total > 0.0 else 0.0)
        for component, value in components.items()
    }
    return ParticleComparison(
        program_hash=program.canonical_hash,
        prediction=prediction,
        log_likelihood=log_likelihood,
        channel_errors=errors,
        channel_responsibility=channel_responsibility,
        responsibility=responsibility,
    )


def _channel_errors(
    prediction: PredictionDistribution,
    evidence: TransitionEvidence,
    *,
    declared_variables: Sequence[str],
) -> dict[str, float]:
    variable_errors = []
    for key in declared_variables:
        predicted_value = prediction.state_after.value(key)
        observed_value = evidence.state_after.value(key)
        variable_errors.append(predicted_value.total_variation(observed_value))
    errors = {
        "variables": sum(variable_errors) / len(variable_errors) if variable_errors else 0.0,
        "objects": _set_distance(
            prediction.delta.affected_objects,
            evidence.observed_delta.affected_objects,
        ),
        "relations": _set_distance(
            prediction.delta.relation_changes,
            evidence.observed_delta.relation_changes,
        ),
        "progress": abs(
            float(prediction.progress_probability)
            - min(1.0, max(0.0, float(evidence.observed_delta.progress)))
        ),
        "terminal": abs(float(prediction.terminal_probability) - float(evidence.terminal)),
        "level": abs(float(prediction.goal_probability) - float(evidence.level_change > 0)),
    }
    if evidence.success is not None:
        errors["goal"] = abs(float(prediction.goal_probability) - float(evidence.success))
    if evidence.observed_delta.patch_digest:
        errors["patch"] = float(
            prediction.delta.patch_digest != evidence.observed_delta.patch_digest
        )
    return errors


def _set_distance(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


__all__ = ["DEFAULT_CHANNEL_WEIGHTS", "ParticleComparison", "compare_particle"]
