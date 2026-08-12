"""Shared symbolic and neural mechanism registry for causal programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import GroundedAction, MechanismSpec, ValueDistribution


@dataclass(frozen=True)
class MechanismContext:
    action: GroundedAction
    current_output: ValueDistribution


class NeuralMechanism(Protocol):
    def predict(
        self,
        parents: Sequence[ValueDistribution],
        action: GroundedAction,
        parameters: Mapping[str, Any],
    ) -> ValueDistribution:
        ...


class ObservationLikelihoodModel(Protocol):
    def log_likelihood(self, prediction: Any, evidence: Any) -> float:
        ...


SymbolicMechanism = Callable[
    [Sequence[ValueDistribution], Mapping[str, Any], MechanismContext],
    ValueDistribution,
]


class MechanismRegistry:
    """One content-addressable registry shared by every particle."""

    def __init__(self) -> None:
        self._symbolic: dict[str, SymbolicMechanism] = {}
        self._neural: dict[str, NeuralMechanism] = {}
        self._observation: dict[str, ObservationLikelihoodModel] = {}
        self._register_defaults()

    def register_symbolic(self, operator_type: str, mechanism: SymbolicMechanism) -> None:
        key = str(operator_type).lower()
        if key in self._symbolic:
            raise ValueError(f"symbolic mechanism already registered: {key}")
        self._symbolic[key] = mechanism

    def register_neural(self, module_id: str, mechanism: NeuralMechanism) -> None:
        key = str(module_id)
        if key in self._neural:
            raise ValueError(f"neural mechanism already registered: {key}")
        self._neural[key] = mechanism

    def register_observation_likelihood(
        self, module_id: str, model: ObservationLikelihoodModel
    ) -> None:
        key = str(module_id)
        if key in self._observation:
            raise ValueError(f"observation model already registered: {key}")
        self._observation[key] = model

    def observation_log_likelihood(
        self, module_id: str, prediction: Any, evidence: Any
    ) -> float:
        model = self._observation.get(str(module_id))
        if model is None:
            raise RuntimeError(f"unresolved observation likelihood: {module_id}")
        return float(model.log_likelihood(prediction, evidence))

    def can_resolve(self, spec: MechanismSpec) -> bool:
        if spec.neural_module_id in self._neural:
            return True
        fallback = spec.symbolic_fallback or spec.operator_type
        return str(fallback).lower() in self._symbolic

    def evaluate(
        self,
        spec: MechanismSpec,
        parents: Sequence[ValueDistribution],
        *,
        action: GroundedAction,
        current_output: ValueDistribution,
    ) -> ValueDistribution:
        required_action = spec.parameters.get("action_name")
        if required_action is not None and str(required_action) != action.action_name:
            return current_output
        if spec.neural_module_id is not None and spec.neural_module_id in self._neural:
            return self._neural[spec.neural_module_id].predict(
                parents, action, spec.parameters
            )
        operator = str(spec.symbolic_fallback or spec.operator_type).lower()
        mechanism = self._symbolic.get(operator)
        if mechanism is None:
            raise RuntimeError(
                f"unresolved mechanism {spec.mechanism_id}: {spec.operator_type}"
            )
        return mechanism(
            parents,
            spec.parameters,
            MechanismContext(action=action, current_output=current_output),
        )

    @property
    def symbolic_operators(self) -> tuple[str, ...]:
        return tuple(sorted(self._symbolic))

    @property
    def neural_modules(self) -> tuple[str, ...]:
        return tuple(sorted(self._neural))

    def _register_defaults(self) -> None:
        for name, implementation in {
            "identity": _identity,
            "set": _set,
            "copy_attribute": _copy,
            "cycle_attribute": _cycle,
            "toggle": _toggle,
            "move": _move,
            "swap": _copy,
            "spawn": _spawn,
            "destroy": _destroy,
            "align_relation": _align,
            "all_predicate": _all_predicate,
            "any_predicate": _any_predicate,
            "count_threshold": _count_threshold,
            "terminal_predicate": _all_predicate,
            "translate_patch": _translate_patch,
            "replace_patch": _set,
            "collision": _collision,
            "action_position": _action_position,
        }.items():
            self.register_symbolic(name, implementation)


def _first(
    parents: Sequence[ValueDistribution], context: MechanismContext
) -> ValueDistribution:
    return parents[0] if parents else context.current_output


def _identity(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return _first(parents, context)


def _set(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(parameters.get("value"))


def _copy(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return _first(parents, context)


def _cycle(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    values = tuple(parameters.get("values", ()))
    if not values:
        return context.current_output
    current = _first(parents, context).mode
    try:
        index = values.index(current)
    except ValueError:
        index = -1
    return ValueDistribution.deterministic(values[(index + 1) % len(values)])


def _toggle(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(not bool(_first(parents, context).mode))


def _move(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    raw = _first(parents, context).mode
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        return context.current_output
    dx = float(context.action.action_data.get("dx", parameters.get("dx", 0.0)))
    dy = float(context.action.action_data.get("dy", parameters.get("dy", 0.0)))
    return ValueDistribution.deterministic([float(raw[0]) + dx, float(raw[1]) + dy])


def _action_position(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    """Apply one complete action-conditioned transition to a 2-D role center."""

    raw = _first(parents, context).mode
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        return context.current_output
    action_name = context.action.action_name
    row = float(raw[0])
    column = float(raw[1])
    positions = parameters.get("positions_by_action", {})
    if isinstance(positions, Mapping) and action_name in positions:
        target = positions[action_name]
        if isinstance(target, (tuple, list)) and len(target) == 2:
            return ValueDistribution.deterministic(
                [float(target[0]), float(target[1])]
            )
    columns = parameters.get("columns_by_action", {})
    if isinstance(columns, Mapping) and action_name in columns:
        return ValueDistribution.deterministic([row, float(columns[action_name])])
    deltas = parameters.get("deltas_by_action", {})
    if isinstance(deltas, Mapping) and action_name in deltas:
        delta = deltas[action_name]
        if isinstance(delta, (tuple, list)) and len(delta) == 2:
            return ValueDistribution.deterministic(
                [row + float(delta[0]), column + float(delta[1])]
            )
    if action_name == str(parameters.get("ground_action", "")):
        row_key = str(parameters.get("row_key", "y"))
        column_key = str(parameters.get("column_key", "x"))
        if (
            row_key in context.action.action_data
            and column_key in context.action.action_data
        ):
            return ValueDistribution.deterministic(
                [
                    float(context.action.action_data[row_key]),
                    float(context.action.action_data[column_key]),
                ]
            )
    return context.current_output


def _spawn(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(True)


def _destroy(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(False)


def _align(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(
        len(parents) >= 2 and parents[0].mode_key == parents[1].mode_key
    )


def _all_predicate(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(all(bool(parent.mode) for parent in parents))


def _any_predicate(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(any(bool(parent.mode) for parent in parents))


def _count_threshold(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    threshold = int(parameters.get("threshold", 1))
    return ValueDistribution.deterministic(
        sum(bool(parent.mode) for parent in parents) >= threshold
    )


def _translate_patch(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    patch = _first(parents, context).mode
    return ValueDistribution.deterministic(
        {
            "patch": patch,
            "dx": context.action.action_data.get("dx", parameters.get("dx", 0)),
            "dy": context.action.action_data.get("dy", parameters.get("dy", 0)),
        }
    )


def _collision(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    return ValueDistribution.deterministic(
        len(parents) >= 2 and parents[0].mode_key == parents[1].mode_key
    )


__all__ = [
    "MechanismContext", "MechanismRegistry", "NeuralMechanism",
    "ObservationLikelihoodModel",
]
