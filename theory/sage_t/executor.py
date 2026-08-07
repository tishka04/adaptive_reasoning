"""Canonical deterministic executor for every SAGE.T program particle."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    OBJECT_EVENT_PREDICATES,
    RELATION_PREDICATES,
    TOPOLOGY_PREDICATES,
    AbstractState,
    ActionBinding,
    ActionCandidate,
    Effect,
    Expression,
    GroundFact,
    JointProgramHypothesis,
    PredictionPacket,
    RolloutPrediction,
    TruthValue,
)

PREDICTED_TRUE = 0.95
PREDICTED_FALSE = 0.05


@dataclass(frozen=True)
class _ExecutionContext:
    action: ActionCandidate
    binding: ActionBinding
    target_id: str = ""
    actor_id: str = ""
    selected_id: str = ""


class ProgramExecutor:
    """Pure interpreter with a bounded memoization cache."""

    def __init__(self, *, maximum_cache_entries: int = 16_384) -> None:
        self.maximum_cache_entries = max(0, int(maximum_cache_entries))
        self._step_cache: dict[tuple[str, str, str], PredictionPacket] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def step(
        self,
        program: JointProgramHypothesis,
        state: AbstractState,
        action: ActionCandidate,
    ) -> PredictionPacket:
        cache_key = (
            program.canonical_hash,
            state.execution_signature,
            action.key,
        )
        cached = self._step_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        packet = self._execute_step(program, state, action)
        if self.maximum_cache_entries > 0:
            if len(self._step_cache) >= self.maximum_cache_entries:
                # Deterministic insertion-order eviction is sufficient for the
                # small receding-horizon cache and avoids another dependency.
                oldest = next(iter(self._step_cache))
                self._step_cache.pop(oldest, None)
            self._step_cache[cache_key] = packet
        return packet

    def rollout(
        self,
        program: JointProgramHypothesis,
        state: AbstractState,
        action_sequence: Sequence[ActionCandidate],
        *,
        maximum_actions: int = 8,
    ) -> RolloutPrediction:
        sequence = tuple(action_sequence[: max(0, int(maximum_actions))])
        packets = []
        current = state
        for action in sequence:
            packet = self.step(program, current, action)
            packets.append(packet)
            current = packet.state_after or current
        return RolloutPrediction(
            sequence=sequence,
            packets=tuple(packets),
            final_state=current,
        )

    def clear_cache(self) -> None:
        self._step_cache.clear()

    def summary(self) -> Mapping[str, int]:
        return {
            "cache_entries": len(self._step_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def _execute_step(
        self,
        program: JointProgramHypothesis,
        state: AbstractState,
        action: ActionCandidate,
    ) -> PredictionPacket:
        binding = next(
            (
                item
                for item in program.action_bindings
                if item.action_name == action.action_name
            ),
            None,
        )
        if binding is None:
            return PredictionPacket(
                known_channels=frozenset(),
                state_after=state,
            )
        context = _context(state, action, binding)
        asserted: list[GroundFact] = []
        retracted: list[GroundFact] = []
        registers: dict[str, str | None] = {}
        counters = dict(state.counters)
        applied_rule = False
        unresolved_effect = False

        if binding.operator == "select":
            if context.target_id:
                registers["selected"] = context.target_id
            else:
                unresolved_effect = True

        for rule in program.transition_rules:
            if rule.action_operator != binding.operator:
                continue
            condition = evaluate_expression(
                rule.condition,
                state,
                context=context,
            )
            if not _as_bool(condition):
                continue
            applied_rule = True
            for effect in rule.effects:
                resolved = _apply_effect(
                    effect,
                    state=state,
                    context=context,
                    asserted=asserted,
                    retracted=retracted,
                    counters=counters,
                    registers=registers,
                )
                unresolved_effect = unresolved_effect or not resolved

        if not applied_rule and binding.operator != "select":
            asserted.append(GroundFact("no_effect"))

        next_state = state.with_updates(
            asserted=asserted,
            retracted=retracted,
            counters=counters,
            registers=registers,
        )
        object_deltas, relation_deltas, topology_deltas = _packet_events(
            asserted,
            retracted,
        )
        known_channels = set()
        if applied_rule or binding.operator == "select":
            if object_deltas or not unresolved_effect:
                known_channels.add("objects")
            if relation_deltas:
                known_channels.add("relations")
            if topology_deltas:
                known_channels.add("topology")
        elif asserted:
            known_channels.add("objects")

        before_progress = _numeric(
            evaluate_expression(
                program.progress_rule.expression,
                state,
                context=context,
            )
        )
        after_progress = _numeric(
            evaluate_expression(
                program.progress_rule.expression,
                next_state,
                context=context,
            )
        )
        progress_mean = None
        if before_progress is not None and after_progress is not None:
            progress_mean = max(-1.0, min(1.0, after_progress - before_progress))
            known_channels.add("progress")

        goal_value = evaluate_expression(
            program.goal_rule.expression,
            next_state,
            context=context,
        )
        goal_probability = _smoothed_boolean(goal_value)
        if goal_probability is not None:
            known_channels.add("goal")

        game_over_values = [
            evaluate_expression(rule.expression, next_state, context=context)
            for rule in program.terminal_rules
            if rule.outcome == "game_over"
        ]
        terminal_probability = (
            max(
                (_smoothed_boolean(value) or PREDICTED_FALSE)
                for value in game_over_values
            )
            if game_over_values
            else None
        )
        if terminal_probability is not None:
            known_channels.add("terminal")

        return PredictionPacket(
            object_deltas=object_deltas,
            relation_deltas=relation_deltas,
            topology_deltas=topology_deltas,
            progress_mean=progress_mean,
            progress_distribution=(
                {}
                if progress_mean is None
                else {
                    f"value:{progress_mean:.6g}": PREDICTED_TRUE,
                    "other": PREDICTED_FALSE,
                }
            ),
            terminal_probability=terminal_probability,
            goal_probability=goal_probability,
            known_channels=frozenset(known_channels),
            state_after=next_state,
        )


def _context(
    state: AbstractState,
    action: ActionCandidate,
    binding: ActionBinding,
) -> _ExecutionContext:
    actors = state.entities_for_role("player")
    actor_id = actors[0].entity_id if actors else ""
    target_id = _ground_action_target(state, action, role=binding.target_role)
    return _ExecutionContext(
        action=action,
        binding=binding,
        target_id=target_id,
        actor_id=actor_id,
        selected_id=state.register("selected"),
    )


def _ground_action_target(
    state: AbstractState,
    action: ActionCandidate,
    *,
    role: str = "",
) -> str:
    payload = dict(action.action_data)
    explicit = str(
        payload.get(
            "entity_id",
            payload.get("target_id", payload.get("object_id", "")),
        )
    )
    if explicit and any(entity.entity_id == explicit for entity in state.entities):
        return explicit
    candidates = state.entities_for_role(role) if role else state.entities
    if not candidates:
        return ""
    x = payload.get("x", payload.get("col"))
    y = payload.get("y", payload.get("row"))
    if x is None or y is None:
        return candidates[0].entity_id if len(candidates) == 1 else ""
    try:
        column = float(x)
        row = float(y)
    except (TypeError, ValueError):
        return ""
    positioned = [entity for entity in candidates if entity.center is not None]
    if not positioned:
        return ""
    nearest = min(
        positioned,
        key=lambda entity: math.dist(
            (row, column),
            entity.center or (row, column),
        ),
    )
    return nearest.entity_id


def _resolve_term(
    term: str,
    *,
    variables: Mapping[str, str],
    context: _ExecutionContext | None,
) -> str:
    if term.startswith("?"):
        return str(variables.get(term, ""))
    if context is None:
        return term
    return {
        "$target": context.target_id,
        "$actor": context.actor_id,
        "$selected": context.selected_id,
        "$action": context.action.action_name.lower(),
    }.get(term, term)


def evaluate_expression(
    expression: Expression,
    state: AbstractState,
    *,
    context: _ExecutionContext | None = None,
    variables: Mapping[str, str] | None = None,
) -> TruthValue | bool | float | str | None:
    """Evaluate one DSL expression with three-valued fact semantics."""

    variables = dict(variables or {})
    op = expression.op
    if op == "const":
        return expression.value
    if op == "counter":
        return state.counter(str(expression.value))
    if op == "fact":
        terms = tuple(
            _resolve_term(term, variables=variables, context=context)
            for term in expression.terms
        )
        if any(not term for term in terms):
            return TruthValue.UNKNOWN
        if expression.predicate == "role" and len(terms) == 2:
            entity = next(
                (item for item in state.entities if item.entity_id == terms[0]),
                None,
            )
            if entity is None:
                return TruthValue.UNKNOWN
            return TruthValue.TRUE if entity.has_role(terms[1]) else TruthValue.FALSE
        return state.truth(GroundFact(expression.predicate, terms))
    if op == "not":
        value = evaluate_expression(
            expression.args[0],
            state,
            context=context,
            variables=variables,
        )
        if value == TruthValue.UNKNOWN or value is None:
            return TruthValue.UNKNOWN
        return not _as_bool(value)
    if op in {"and", "or"}:
        values = [
            evaluate_expression(
                arg,
                state,
                context=context,
                variables=variables,
            )
            for arg in expression.args
        ]
        if op == "and":
            if any(
                not _as_bool(value)
                for value in values
                if value not in {TruthValue.UNKNOWN, None}
            ):
                return False
            if any(value == TruthValue.UNKNOWN or value is None for value in values):
                return TruthValue.UNKNOWN
            return True
        if any(_as_bool(value) for value in values):
            return True
        if any(value == TruthValue.UNKNOWN or value is None for value in values):
            return TruthValue.UNKNOWN
        return False
    if op in {"exists", "forall", "count"}:
        entities = state.entities_for_role(expression.role)
        if op == "count" and not expression.args:
            return float(len(entities))
        if op == "exists" and not expression.args:
            return bool(entities)
        if op == "forall" and not expression.args:
            return bool(entities)
        results = []
        for entity in entities:
            local = dict(variables)
            local[expression.variable] = entity.entity_id
            results.append(
                evaluate_expression(
                    expression.args[0],
                    state,
                    context=context,
                    variables=local,
                )
            )
        if op == "count":
            if any(value == TruthValue.UNKNOWN or value is None for value in results):
                return None
            return float(sum(_as_bool(value) for value in results))
        if op == "exists":
            if any(_as_bool(value) for value in results):
                return True
            if any(value == TruthValue.UNKNOWN or value is None for value in results):
                return TruthValue.UNKNOWN
            return False
        if not entities:
            return False
        if any(
            not _as_bool(value)
            for value in results
            if value not in {TruthValue.UNKNOWN, None}
        ):
            return False
        if any(value == TruthValue.UNKNOWN or value is None for value in results):
            return TruthValue.UNKNOWN
        return True
    if op == "ratio":
        numerator = _numeric(
            evaluate_expression(
                expression.args[0],
                state,
                context=context,
                variables=variables,
            )
        )
        denominator = _numeric(
            evaluate_expression(
                expression.args[1],
                state,
                context=context,
                variables=variables,
            )
        )
        if numerator is None or denominator is None or denominator <= 0.0:
            return None
        return numerator / denominator
    if op in {"eq", "gt", "ge", "lt", "le"}:
        left = evaluate_expression(
            expression.args[0],
            state,
            context=context,
            variables=variables,
        )
        right = evaluate_expression(
            expression.args[1],
            state,
            context=context,
            variables=variables,
        )
        if left in {TruthValue.UNKNOWN, None} or right in {TruthValue.UNKNOWN, None}:
            return TruthValue.UNKNOWN
        lhs = _numeric(left)
        rhs = _numeric(right)
        if lhs is None or rhs is None:
            if op == "eq":
                return left == right
            return TruthValue.UNKNOWN
        return {
            "eq": lhs == rhs,
            "gt": lhs > rhs,
            "ge": lhs >= rhs,
            "lt": lhs < rhs,
            "le": lhs <= rhs,
        }[op]
    raise ValueError(f"unsupported expression op: {op}")


def _apply_effect(
    effect: Effect,
    *,
    state: AbstractState,
    context: _ExecutionContext,
    asserted: list[GroundFact],
    retracted: list[GroundFact],
    counters: dict[str, float],
    registers: dict[str, str | None],
) -> bool:
    if effect.operation in {"assert", "retract"}:
        terms = tuple(
            _resolve_term(term, variables={}, context=context) for term in effect.terms
        )
        if any(not term for term in terms):
            return False
        fact = GroundFact(effect.predicate, terms)
        (asserted if effect.operation == "assert" else retracted).append(fact)
        return True
    if effect.operation in {"move_relative", "change_morphology"}:
        terms = tuple(
            _resolve_term(term, variables={}, context=context) for term in effect.terms
        )
        if any(not term for term in terms):
            return False
        predicate = (
            "moved" if effect.operation == "move_relative" else "morphology_changed"
        )
        asserted.append(
            GroundFact(
                predicate,
                terms,
                value=str(effect.value or ""),
            )
        )
        return True
    if effect.operation == "progress":
        try:
            delta = float(effect.value or 0.0)
        except (TypeError, ValueError):
            return False
        counters["progress"] = counters.get("progress", 0.0) + delta
        asserted.append(GroundFact("progress"))
        return True
    if effect.operation == "win":
        asserted.append(GroundFact("level_complete"))
        return True
    if effect.operation == "fail":
        asserted.append(GroundFact("game_over"))
        return True
    if effect.operation == "set_register":
        raw = str(effect.value or "")
        value = _resolve_term(raw, variables={}, context=context)
        if not value:
            return False
        registers[effect.key] = value
        return True
    if effect.operation == "clear_register":
        registers[effect.key] = None
        return True
    if effect.operation == "set_counter":
        try:
            counters[effect.key] = float(effect.value or 0.0)
        except (TypeError, ValueError):
            return False
        return True
    if effect.operation == "increment_counter":
        try:
            delta = float(effect.value or 0.0)
        except (TypeError, ValueError):
            return False
        counters[effect.key] = float(counters.get(effect.key, 0.0)) + delta
        return True
    return False


def _packet_events(
    asserted: Sequence[GroundFact],
    retracted: Sequence[GroundFact],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    objects: dict[str, float] = {}
    relations: dict[str, float] = {}
    topology: dict[str, float] = {}
    for fact, operation in (
        *((fact, "added") for fact in asserted),
        *((fact, "removed") for fact in retracted),
    ):
        if fact.predicate in RELATION_PREDICATES:
            relation_operation = (
                "relation_added" if operation == "added" else "relation_removed"
            )
            relations[f"{relation_operation}:{fact.predicate}"] = PREDICTED_TRUE
        elif fact.predicate in TOPOLOGY_PREDICATES:
            topology[fact.predicate] = PREDICTED_TRUE
        elif fact.predicate in OBJECT_EVENT_PREDICATES:
            objects[fact.predicate] = PREDICTED_TRUE
    return objects, relations, topology


def _as_bool(value: Any) -> bool:
    if value == TruthValue.TRUE:
        return True
    if value in {TruthValue.FALSE, TruthValue.UNKNOWN, None}:
        return False
    return bool(value)


def _numeric(value: Any) -> float | None:
    if value == TruthValue.TRUE:
        return 1.0
    if value in {TruthValue.FALSE}:
        return 0.0
    if value in {TruthValue.UNKNOWN, None}:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _smoothed_boolean(value: Any) -> float | None:
    if value == TruthValue.UNKNOWN or value is None:
        return None
    return PREDICTED_TRUE if _as_bool(value) else PREDICTED_FALSE


__all__ = [
    "PREDICTED_FALSE",
    "PREDICTED_TRUE",
    "ProgramExecutor",
    "evaluate_expression",
]
