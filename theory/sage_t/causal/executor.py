"""The single causal executor for symbolic, hybrid, and neural particles."""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Mapping
from typing import Any

from .compiler import CompiledCausalProgram, ProgramCompiler
from .contracts import (
    ActionProgram,
    CausalProgram,
    CausalState,
    GroundedAction,
    Intervention,
    PredictedTrace,
    PredictionDistribution,
    StructuredDelta,
    ValueDistribution,
)
from .mechanisms import MechanismRegistry

_COMPARISON = re.compile(r"^(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+)$")


class CausalExecutor:
    """Pure two-slice SCM interpreter shared by posterior and planner."""

    def __init__(
        self,
        *,
        mechanism_registry: MechanismRegistry | None = None,
        maximum_cache_entries: int = 4096,
    ) -> None:
        self.mechanism_registry = mechanism_registry or MechanismRegistry()
        self.compiler = ProgramCompiler(self.mechanism_registry)
        self.maximum_cache_entries = max(0, int(maximum_cache_entries))
        self._compiled: dict[str, CompiledCausalProgram] = {}
        self._step_cache: dict[tuple[str, str, str], PredictionDistribution] = {}

    def compile(
        self,
        program: CausalProgram,
        *,
        action_catalog: tuple[str, ...] | None = None,
    ) -> CompiledCausalProgram:
        catalog = action_catalog or tuple(item.action_name for item in program.action_model)
        key = f"{program.canonical_hash}:{','.join(sorted(catalog))}"
        compiled = self._compiled.get(key)
        if compiled is None:
            compiled = self.compiler.compile(program, action_catalog=catalog)
            self._remember(self._compiled, key, compiled)
        return compiled

    def predict_step(
        self,
        program: CausalProgram | CompiledCausalProgram,
        state: CausalState,
        action: GroundedAction,
    ) -> PredictionDistribution:
        compiled = self._as_compiled(program)
        if action.action_name not in compiled.available_actions:
            raise ValueError(f"action {action.action_name} is unavailable for the causal program")
        key = (compiled.canonical_hash, state.abstract_signature, action.key)
        cached = self._step_cache.get(key)
        if cached is not None:
            return cached
        prediction = self._execute(compiled, state, action, interventions={})
        self._remember(self._step_cache, key, prediction)
        return prediction

    def rollout(
        self,
        program: CausalProgram | CompiledCausalProgram,
        state: CausalState,
        action_program: ActionProgram,
        horizon: int,
    ) -> PredictedTrace:
        maximum = min(len(action_program.actions), max(0, int(horizon)), 8)
        if maximum <= 0:
            raise ValueError("rollout horizon must include at least one action")
        predictions = []
        current = state
        for action in action_program.actions[:maximum]:
            prediction = self.predict_step(program, current, action)
            predictions.append(prediction)
            current = prediction.state_after
        return PredictedTrace(action_program=action_program, predictions=tuple(predictions))

    def intervene(
        self,
        program: CausalProgram | CompiledCausalProgram,
        state: CausalState,
        intervention: Intervention,
    ) -> PredictionDistribution:
        compiled = self._as_compiled(program)
        if intervention.variable_id not in compiled.variables:
            raise ValueError(f"unknown intervention variable: {intervention.variable_id}")
        return self._execute(
            compiled,
            state,
            GroundedAction("INTERVENE"),
            interventions={intervention.variable_id: intervention.value},
        )

    def clear_cache(self) -> None:
        self._compiled.clear()
        self._step_cache.clear()

    def _as_compiled(
        self, program: CausalProgram | CompiledCausalProgram
    ) -> CompiledCausalProgram:
        return program if isinstance(program, CompiledCausalProgram) else self.compile(program)

    def _execute(
        self,
        compiled: CompiledCausalProgram,
        state: CausalState,
        action: GroundedAction,
        *,
        interventions: Mapping[str, ValueDistribution],
    ) -> PredictionDistribution:
        next_values: dict[str, ValueDistribution] = {}
        for variable_id in compiled.topological_order:
            if variable_id in interventions:
                next_values[variable_id] = interventions[variable_id]
                continue
            mechanism = compiled.mechanisms[variable_id]
            parents = [
                (
                    next_values[parent.variable_id]
                    if parent.time_slice == "next"
                    else state.value(parent.variable_id)
                )
                for parent in mechanism.parent_variables
            ]
            output = self.mechanism_registry.evaluate(
                mechanism,
                parents,
                action=action,
                current_output=state.value(variable_id),
            )
            domain = compiled.variables[variable_id].domain
            if domain and output.mode not in domain:
                raise ValueError(
                    f"mechanism {mechanism.mechanism_id} emitted an out-of-domain value"
                )
            next_values[variable_id] = output
        state_after = CausalState(
            variables=next_values,
            entities=state.entities,
            relations=state.relations,
            confidence=state.confidence,
        )
        changes = {
            variable_id: value
            for variable_id, value in next_values.items()
            if value.total_variation(state.value(variable_id)) > 1e-12
        }
        goal_probability = float(_predicate(compiled.program.goal.success_predicate, state_after))
        progress_values = [
            float(_predicate(predicate, state_after))
            for predicate in compiled.program.goal.progress_predicates
        ]
        progress_probability = (
            sum(progress_values) / len(progress_values) if progress_values else goal_probability
        )
        failure_probability = (
            0.0
            if compiled.program.goal.failure_predicate is None
            else float(_predicate(compiled.program.goal.failure_predicate, state_after))
        )
        terminal_probability = max(goal_probability, failure_probability)
        affected_objects = tuple(
            sorted({variable_id.split(".", 1)[0] for variable_id in changes if "." in variable_id})
        )
        relation_changes = tuple(
            sorted(variable_id for variable_id in changes if compiled.variables[variable_id].variable_type == "relation")
        )
        delta = StructuredDelta(
            variable_changes=changes,
            affected_objects=affected_objects,
            relation_changes=relation_changes,
            progress=progress_probability,
        )
        return PredictionDistribution(
            state_after=state_after,
            delta=delta,
            terminal_probability=terminal_probability,
            goal_probability=goal_probability,
            progress_probability=progress_probability,
            known_variables=frozenset(next_values),
        )

    def _remember(self, cache: dict[Any, Any], key: Any, value: Any) -> None:
        if self.maximum_cache_entries <= 0:
            return
        if len(cache) >= self.maximum_cache_entries:
            cache.pop(next(iter(cache)), None)
        cache[key] = value


def _predicate(expression: str, state: CausalState) -> bool:
    text = str(expression).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered.startswith("not "):
        return not _predicate(text[4:], state)
    if lowered.startswith("all(") and text.endswith(")"):
        return all(_predicate(item, state) for item in _split_arguments(text[4:-1]))
    if lowered.startswith("any(") and text.endswith(")"):
        return any(_predicate(item, state) for item in _split_arguments(text[4:-1]))
    match = _COMPARISON.fullmatch(text)
    if match:
        left = _operand(match.group(1), state)
        right = _operand(match.group(3), state)
        operation = {
            "==": operator.eq,
            "!=": operator.ne,
            ">=": operator.ge,
            "<=": operator.le,
            ">": operator.gt,
            "<": operator.lt,
        }[match.group(2)]
        try:
            return bool(operation(left, right))
        except TypeError:
            return False
    return bool(state.value(text).mode)


def _split_arguments(payload: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in payload.split(",") if item.strip())


def _operand(payload: str, state: CausalState) -> Any:
    value = payload.strip()
    if value in state.variables:
        return state.value(value).mode
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip("'\"")


__all__ = ["CausalExecutor"]
