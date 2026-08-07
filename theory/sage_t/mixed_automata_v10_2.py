"""Bounded mixed-sequence option automata for SAGE.T10.2.

The transferable object in this module is an immutable finite-state option.
Grounded action names and arguments are supplied only when one action is
issued; they never enter the option's canonical payload or hash.

Execution is deliberately transactional.  ``execute_one`` can issue at most
one grounded action and returns an execution state that refuses another action
until ``observe`` (or ``observe_update_replan``) consumes the resulting event.
This keeps active control on the SAGE invariant of observe/update/replan after
every real transition.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

from .contracts import ALLOWED_PREDICATES, AbstractState

MAX_OPTION_STATES = 4
MAX_ACTION_SCHEMAS = 2
MAX_OPTION_HORIZON = 16
AST_NODE_LOG_COST = -0.05

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PREDICATE = re.compile(r"^[a-z][a-z0-9_:.-]{0,95}$")
_GROUND_VALUE = (str, int, float, bool, type(None))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _identifier(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{label} must be a bounded snake_case identifier")
    return normalized


def _predicate(value: str, *, label: str, allow_empty: bool = False) -> str:
    normalized = str(value).strip().lower()
    if allow_empty and not normalized:
        return ""
    if not _PREDICATE.fullmatch(normalized):
        raise ValueError(f"{label} must be a bounded event predicate")
    return normalized


def ast_node_log_cost(node_count: int) -> float:
    """Return the frozen T10.2 MDL contribution for ``node_count`` nodes."""

    count = int(node_count)
    if count < 0:
        raise ValueError("node_count cannot be negative")
    return AST_NODE_LOG_COST * count


# A concise alias for posterior code that treats the value as a log-cost.
ast_node_cost = ast_node_log_cost


@dataclass(frozen=True, order=True)
class OptionState:
    """One immutable control state in a transferable option automaton."""

    state_id: str
    terminal: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_id",
            _identifier(self.state_id, label="option state id"),
        )
        object.__setattr__(self, "terminal", bool(self.terminal))

    @property
    def id(self) -> str:
        return self.state_id


@dataclass(frozen=True, order=True)
class OptionTransition:
    """One action-labelled, post-observation transition.

    An empty ``predicate`` is the deterministic fallback.  A non-empty guard
    is evaluated against the event/predicate set after the action.  The
    ``minimum_visits`` counter permits compact priming without unbounded or
    coordinate-bearing state.
    """

    source_state_id: str
    target_state_id: str
    action_schema: str
    predicate: str = ""
    predicate_present: bool = True
    minimum_visits: int = 1
    relation: str = "identity"
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_state_id",
            _identifier(self.source_state_id, label="transition source"),
        )
        object.__setattr__(
            self,
            "target_state_id",
            _identifier(self.target_state_id, label="transition target"),
        )
        object.__setattr__(
            self,
            "action_schema",
            _identifier(self.action_schema, label="action schema"),
        )
        predicate = _predicate(
            self.predicate,
            label="transition predicate",
            allow_empty=True,
        )
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "predicate_present", bool(self.predicate_present))
        visits = int(self.minimum_visits)
        if visits < 1 or visits > MAX_OPTION_HORIZON:
            raise ValueError(f"minimum_visits must be in [1, {MAX_OPTION_HORIZON}]")
        object.__setattr__(self, "minimum_visits", visits)
        object.__setattr__(
            self,
            "relation",
            _predicate(self.relation, label="transition relation"),
        )
        object.__setattr__(self, "priority", int(self.priority))
        if not predicate and not self.predicate_present:
            raise ValueError("an unconditional transition cannot require absence")

    @property
    def source(self) -> str:
        return self.source_state_id

    @property
    def target(self) -> str:
        return self.target_state_id

    def matches(self, evidence: frozenset[str], *, visits: int) -> bool:
        if int(visits) < self.minimum_visits:
            return False
        if not self.predicate:
            return True
        present = self.predicate in evidence
        return present if self.predicate_present else not present


@dataclass(frozen=True)
class OptionInitiationCondition:
    """Closed structural condition required before an option can execute."""

    kind: str
    predicate: str = ""
    role: str = ""
    predicate_present: bool = True

    def __post_init__(self) -> None:
        kind = _identifier(self.kind, label="initiation condition kind")
        if kind not in {"registered_true", "fact", "role"}:
            raise ValueError("unsupported option initiation condition kind")
        predicate = _predicate(
            self.predicate,
            label="initiation predicate",
            allow_empty=True,
        )
        role = (
            _identifier(self.role, label="initiation role")
            if str(self.role).strip()
            else ""
        )
        present = bool(self.predicate_present)
        if kind == "registered_true":
            if predicate or role or not present:
                raise ValueError(
                    "registered_true initiation cannot carry structural guards"
                )
        elif kind == "fact":
            if not predicate:
                raise ValueError("fact initiation requires a predicate")
            if predicate not in ALLOWED_PREDICATES:
                raise ValueError(
                    "fact initiation predicate is outside the state schema"
                )
        elif not role or predicate:
            raise ValueError("role initiation requires exactly one role")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "predicate_present", present)

    @classmethod
    def registered_true(cls) -> OptionInitiationCondition:
        return cls("registered_true")

    @classmethod
    def fact(
        cls,
        predicate: str,
        *,
        role: str = "",
        present: bool = True,
    ) -> OptionInitiationCondition:
        return cls("fact", predicate, role, present)

    @classmethod
    def entity_role(
        cls,
        role: str,
        *,
        present: bool = True,
    ) -> OptionInitiationCondition:
        return cls("role", role=role, predicate_present=present)

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "predicate": self.predicate,
            "predicate_present": self.predicate_present,
            "role": self.role,
        }

    def satisfied_by(self, state: AbstractState | None) -> bool:
        if self.kind == "registered_true":
            return True
        if state is None:
            return False
        if not isinstance(state, AbstractState):
            raise TypeError("option initiation requires AbstractState")
        if self.kind == "role":
            present = bool(state.entities_for_role(self.role))
            return present if self.predicate_present else not present
        eligible_ids = (
            {entity.entity_id for entity in state.entities_for_role(self.role)}
            if self.role
            else set()
        )
        evidence = state.true_facts if self.predicate_present else state.false_facts
        return any(
            fact.predicate == self.predicate
            and (not self.role or bool(eligible_ids & set(fact.terms)))
            for fact in evidence
        )


REGISTERED_TRUE_INITIATION = OptionInitiationCondition.registered_true()


@dataclass(frozen=True)
class OptionExecutionState:
    """Immutable runtime cursor for one option execution."""

    automaton_hash: str
    state_id: str
    steps: int = 0
    state_visits: int = 0
    maximum_horizon: int = MAX_OPTION_HORIZON
    awaiting_observation: bool = False
    pending_action_schema: str = ""
    terminated: bool = False

    def __post_init__(self) -> None:
        digest = str(self.automaton_hash).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("automaton_hash must be a SHA-256 digest")
        object.__setattr__(self, "automaton_hash", digest)
        object.__setattr__(
            self,
            "state_id",
            _identifier(self.state_id, label="execution state id"),
        )
        steps = int(self.steps)
        visits = int(self.state_visits)
        horizon = int(self.maximum_horizon)
        if not 1 <= horizon <= MAX_OPTION_HORIZON:
            raise ValueError(f"execution horizon must be in [1, {MAX_OPTION_HORIZON}]")
        if not 0 <= visits <= steps <= horizon:
            raise ValueError(
                "execution counters require 0 <= state_visits <= steps <= horizon"
            )
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "state_visits", visits)
        object.__setattr__(self, "maximum_horizon", horizon)
        object.__setattr__(
            self,
            "awaiting_observation",
            bool(self.awaiting_observation),
        )
        pending = str(self.pending_action_schema).strip().lower()
        if pending:
            pending = _identifier(pending, label="pending action schema")
        object.__setattr__(self, "pending_action_schema", pending)
        object.__setattr__(self, "terminated", bool(self.terminated))
        if self.awaiting_observation != bool(pending):
            raise ValueError(
                "pending_action_schema is required exactly while awaiting observation"
            )
        if self.awaiting_observation and steps == 0:
            raise ValueError("an initial execution cannot await observation")
        if self.awaiting_observation and visits == 0:
            raise ValueError("a pending action requires a positive state visit count")
        if self.terminated and self.awaiting_observation:
            raise ValueError("a terminated execution cannot have a pending action")


@dataclass(frozen=True)
class GroundedExecutionResult:
    """Result of issuing zero (veto) or exactly one grounded action."""

    execution_before: OptionExecutionState
    execution_after_action: OptionExecutionState
    action_schema: str = ""
    action_name: str = ""
    action_data: tuple[tuple[str, str | int | float | bool | None], ...] = ()
    executed: bool = False
    danger_vetoed: bool = False
    requires_observation: bool = False
    requires_update: bool = False
    requires_replan: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        schema = str(self.action_schema).strip().lower()
        if schema:
            schema = _identifier(schema, label="grounded action schema")
        object.__setattr__(self, "action_schema", schema)
        object.__setattr__(self, "action_name", str(self.action_name).strip().upper())
        data = tuple(sorted((str(key), value) for key, value in self.action_data))
        for key, value in data:
            if not key:
                raise ValueError("grounded action argument keys cannot be empty")
            if not isinstance(value, _GROUND_VALUE):
                raise TypeError("grounded action arguments must be JSON scalars")
        object.__setattr__(self, "action_data", data)
        for field_name in (
            "executed",
            "danger_vetoed",
            "requires_observation",
            "requires_update",
            "requires_replan",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        object.__setattr__(self, "reason", str(self.reason))
        if self.executed:
            if not schema or not self.action_name:
                raise ValueError("an executed action needs schema and grounded name")
            if not self.requires_observation or not self.requires_update:
                raise ValueError("executed actions must await observation and update")
        elif self.requires_observation or self.requires_update:
            raise ValueError("a non-executed result cannot await action evidence")
        if self.danger_vetoed and self.executed:
            raise ValueError("a danger-vetoed action cannot be executed")

    @property
    def data(self) -> dict[str, str | int | float | bool | None]:
        return dict(self.action_data)

    @property
    def issued_actions(self) -> int:
        return int(self.executed)


StateLike = OptionState | OptionExecutionState | str
DangerVeto = Callable[[str, str, Mapping[str, Any]], bool]


@dataclass(frozen=True)
class OptionAutomaton:
    """A bounded immutable option whose hash excludes grounded identities."""

    schema: str
    states: tuple[OptionState, ...]
    transitions: tuple[OptionTransition, ...]
    initial_state_id: str
    maximum_horizon: int = MAX_OPTION_HORIZON
    initiation_condition: OptionInitiationCondition | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema", _identifier(self.schema, label="option schema")
        )
        states = tuple(self.states)
        transitions = tuple(self.transitions)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "transitions", transitions)
        initial = _identifier(self.initial_state_id, label="initial state id")
        object.__setattr__(self, "initial_state_id", initial)
        horizon = int(self.maximum_horizon)
        if not 1 <= horizon <= MAX_OPTION_HORIZON:
            raise ValueError(f"maximum_horizon must be in [1, {MAX_OPTION_HORIZON}]")
        object.__setattr__(self, "maximum_horizon", horizon)
        if not isinstance(self.initiation_condition, OptionInitiationCondition):
            raise TypeError(
                "option automaton requires an explicit initiation condition"
            )
        if not states:
            raise ValueError("an option automaton needs at least one state")
        if len(states) > MAX_OPTION_STATES:
            raise ValueError(
                f"option automata are limited to {MAX_OPTION_STATES} states"
            )
        state_ids = tuple(state.state_id for state in states)
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("option state ids must be unique")
        if initial not in set(state_ids):
            raise ValueError("initial_state_id must reference an option state")
        state_map = {state.state_id: state for state in states}
        action_schemas = {transition.action_schema for transition in transitions}
        if len(action_schemas) > MAX_ACTION_SCHEMAS:
            raise ValueError(
                f"option automata are limited to {MAX_ACTION_SCHEMAS} action schemas"
            )
        seen_guards: set[tuple[Any, ...]] = set()
        for transition in transitions:
            if transition.source_state_id not in state_map:
                raise ValueError("transition source does not exist")
            if transition.target_state_id not in state_map:
                raise ValueError("transition target does not exist")
            if state_map[transition.source_state_id].terminal:
                raise ValueError("terminal option states cannot have transitions")
            if transition.minimum_visits > horizon:
                raise ValueError("transition minimum_visits exceeds option horizon")
            key = (
                transition.source_state_id,
                transition.action_schema,
                transition.predicate,
                transition.predicate_present,
                transition.minimum_visits,
                transition.priority,
            )
            if key in seen_guards:
                raise ValueError("duplicate option transition guard")
            seen_guards.add(key)
        for state in states:
            outgoing = [
                transition
                for transition in transitions
                if transition.source_state_id == state.state_id
            ]
            if state.terminal:
                continue
            if not outgoing:
                raise ValueError("every nonterminal state needs an outgoing action")
            if len({transition.action_schema for transition in outgoing}) != 1:
                raise ValueError(
                    "one state must issue exactly one action schema before observation"
                )
        reachable = {initial}
        changed = True
        while changed:
            changed = False
            for transition in transitions:
                if (
                    transition.source_state_id in reachable
                    and transition.target_state_id not in reachable
                ):
                    reachable.add(transition.target_state_id)
                    changed = True
        if reachable != set(state_ids):
            raise ValueError(
                "all option states must be reachable from the initial state"
            )

    @property
    def initial_state(self) -> OptionState:
        return self.state(self.initial_state_id)

    @property
    def action_schemas(self) -> tuple[str, ...]:
        return tuple(sorted({item.action_schema for item in self.transitions}))

    @property
    def node_count(self) -> int:
        return 2 + len(self.states) + len(self.transitions)

    @property
    def log_ast_cost(self) -> float:
        return ast_node_log_cost(self.node_count)

    @property
    def canonical_payload(self) -> dict[str, Any]:
        """Alpha-normalize state ids while retaining abstract action roles."""

        return min(
            (payload for payload, _ in self._canonical_forms()),
            key=_canonical_json,
        )

    def _canonical_forms(self) -> tuple[tuple[dict[str, Any], dict[str, str]], ...]:
        other_ids = sorted(
            state.state_id
            for state in self.states
            if state.state_id != self.initial_state_id
        )
        candidates: list[tuple[dict[str, Any], dict[str, str]]] = []
        for order in itertools.permutations(other_ids):
            state_order = (self.initial_state_id, *order)
            normalized = {
                state_id: f"q{index}" for index, state_id in enumerate(state_order)
            }
            payload = {
                "schema": self.schema,
                "maximum_horizon": self.maximum_horizon,
                "initiation": self.initiation_condition.canonical_payload,
                "initial_state": "q0",
                "states": [
                    {
                        "state": normalized[state_id],
                        "terminal": self.state(state_id).terminal,
                    }
                    for state_id in state_order
                ],
                "transitions": sorted(
                    (
                        {
                            "source": normalized[item.source_state_id],
                            "target": normalized[item.target_state_id],
                            "action_schema": item.action_schema,
                            "predicate": item.predicate,
                            "predicate_present": item.predicate_present,
                            "minimum_visits": item.minimum_visits,
                            "relation": item.relation,
                            "priority": item.priority,
                        }
                        for item in self.transitions
                    ),
                    key=_canonical_json,
                ),
            }
            candidates.append((payload, normalized))
        return tuple(candidates)

    def _canonical_state_label(self, state_id: str) -> str:
        forms = self._canonical_forms()
        canonical_text = min(_canonical_json(payload) for payload, _ in forms)
        labels = [
            mapping[state_id]
            for payload, mapping in forms
            if _canonical_json(payload) == canonical_text
        ]
        return min(labels)

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.canonical_payload).encode("utf-8")
        ).hexdigest()

    def state(self, value: StateLike) -> OptionState:
        if isinstance(value, OptionState):
            state_id = value.state_id
        elif isinstance(value, OptionExecutionState):
            self._validate_execution(value)
            state_id = value.state_id
        else:
            state_id = _identifier(value, label="option state id")
        for state in self.states:
            if state.state_id == state_id:
                return state
        raise KeyError(f"unknown option state: {state_id}")

    def allowed_action_schemas(self, state: StateLike) -> tuple[str, ...]:
        if isinstance(state, OptionExecutionState) and state.terminated:
            self._validate_execution(state)
            return ()
        resolved = self.state(state)
        if resolved.terminal:
            return ()
        return tuple(
            sorted(
                {
                    transition.action_schema
                    for transition in self.transitions
                    if transition.source_state_id == resolved.state_id
                }
            )
        )

    def can_initiate(self, state: AbstractState | None = None) -> bool:
        condition = self.initiation_condition
        if condition is None:  # protected by construction, retained fail-closed
            return False
        return condition.satisfied_by(state)

    def _require_initiation(self, state: AbstractState | None) -> None:
        if not self.can_initiate(state):
            raise ValueError("option initiation condition is not satisfied")

    def new_execution(
        self,
        state: AbstractState | None = None,
    ) -> OptionExecutionState:
        self._require_initiation(state)
        return OptionExecutionState(
            automaton_hash=self.canonical_hash,
            state_id=self.initial_state_id,
            maximum_horizon=self.maximum_horizon,
            terminated=self.initial_state.terminal,
        )

    def stateful_signature(self, execution: OptionExecutionState) -> str:
        """Hash both current cursor and future option semantics.

        Including the canonical option hash is intentional: two options that
        observed the same no-op prefix remain distinct when their future
        termination guards differ.
        """

        self._validate_execution(execution)
        payload = {
            "automaton_hash": self.canonical_hash,
            "state": self._canonical_state_label(execution.state_id),
            "steps": execution.steps,
            "state_visits": execution.state_visits,
            "maximum_horizon": execution.maximum_horizon,
            "awaiting_observation": execution.awaiting_observation,
            "pending_action_schema": execution.pending_action_schema,
            "terminated": execution.terminated,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def execute_one(
        self,
        execution: OptionExecutionState,
        groundings: Mapping[str, Any],
        *,
        state: AbstractState | None = None,
        danger: bool = False,
        danger_veto: DangerVeto | None = None,
    ) -> GroundedExecutionResult:
        """Issue at most one action and require observation before continuation."""

        self._validate_execution(execution)
        self._require_initiation(state)
        if execution.awaiting_observation:
            raise RuntimeError(
                "observe/update/replan is required before another action"
            )
        current = self.state(execution)
        if (
            execution.terminated
            or current.terminal
            or execution.steps >= self.maximum_horizon
        ):
            return GroundedExecutionResult(
                execution_before=execution,
                execution_after_action=execution,
                executed=False,
                requires_replan=False,
                reason="option_terminated",
            )
        schemas = self.allowed_action_schemas(current)
        if len(schemas) != 1:  # protected by construction, retained fail-closed
            raise RuntimeError("option state does not select exactly one action schema")
        schema = schemas[0]
        grounding = _grounding_for_schema(groundings, schema)
        required_relations = {
            transition.relation
            for transition in self.transitions
            if transition.source_state_id == current.state_id
            and transition.action_schema == schema
            and transition.relation != "identity"
        }
        if required_relations:
            grounded_relation = _grounding_relation(grounding)
            if not grounded_relation:
                raise ValueError(
                    "relation-grounded option action requires relation provenance"
                )
            if grounded_relation not in required_relations:
                raise ValueError(
                    "grounded relation does not match the option transition"
                )
        action_name, action_data = _ground_action(grounding)
        vetoed = bool(danger)
        if danger_veto is not None:
            vetoed = vetoed or bool(danger_veto(schema, action_name, dict(action_data)))
        if vetoed:
            return GroundedExecutionResult(
                execution_before=execution,
                execution_after_action=execution,
                action_schema=schema,
                action_name=action_name,
                action_data=action_data,
                executed=False,
                danger_vetoed=True,
                requires_replan=True,
                reason="danger_veto",
            )
        after = OptionExecutionState(
            automaton_hash=self.canonical_hash,
            state_id=execution.state_id,
            steps=execution.steps + 1,
            state_visits=execution.state_visits + 1,
            maximum_horizon=self.maximum_horizon,
            awaiting_observation=True,
            pending_action_schema=schema,
            terminated=False,
        )
        return GroundedExecutionResult(
            execution_before=execution,
            execution_after_action=after,
            action_schema=schema,
            action_name=action_name,
            action_data=action_data,
            executed=True,
            requires_observation=True,
            requires_update=True,
            requires_replan=True,
            reason="action_issued_waiting_for_observation",
        )

    @overload
    def observe(
        self,
        state: OptionExecutionState,
        action: str | GroundedExecutionResult,
        events: Iterable[str] = (),
        predicates: Iterable[str] = (),
    ) -> OptionExecutionState: ...

    @overload
    def observe(
        self,
        state: OptionState | str,
        action: str | GroundedExecutionResult,
        events: Iterable[str] = (),
        predicates: Iterable[str] = (),
    ) -> OptionState: ...

    def observe(
        self,
        state: StateLike,
        action: str | GroundedExecutionResult,
        events: Iterable[str] = (),
        predicates: Iterable[str] = (),
    ) -> OptionExecutionState | OptionState:
        """Consume post-action evidence and return the replanned next state."""

        action_schema = _action_schema(action)
        evidence = _evidence(events, predicates)
        if isinstance(state, OptionExecutionState):
            self._validate_execution(state)
            if not state.awaiting_observation:
                raise RuntimeError("no issued action is awaiting observation")
            if action_schema != state.pending_action_schema:
                raise ValueError(
                    "observed action does not match the pending option action"
                )
            target = self._next_state_id(
                state.state_id,
                action_schema,
                evidence,
                visits=state.state_visits,
            )
            visits = state.state_visits if target == state.state_id else 0
            target_state = self.state(target)
            return OptionExecutionState(
                automaton_hash=self.canonical_hash,
                state_id=target,
                steps=state.steps,
                state_visits=visits,
                maximum_horizon=self.maximum_horizon,
                awaiting_observation=False,
                pending_action_schema="",
                terminated=target_state.terminal or state.steps >= self.maximum_horizon,
            )
        resolved = self.state(state)
        if action_schema not in self.allowed_action_schemas(resolved):
            raise ValueError("action schema is not allowed from this option state")
        return self.state(
            self._next_state_id(
                resolved.state_id,
                action_schema,
                evidence,
                visits=1,
            )
        )

    def observe_update_replan(
        self,
        execution: OptionExecutionState,
        *,
        events: Iterable[str] = (),
        predicates: Iterable[str] = (),
    ) -> OptionExecutionState:
        if not execution.pending_action_schema:
            raise RuntimeError("no option action is awaiting observation")
        return self.observe(
            execution,
            execution.pending_action_schema,
            events,
            predicates,
        )

    def _next_state_id(
        self,
        state_id: str,
        action_schema: str,
        evidence: frozenset[str],
        *,
        visits: int,
    ) -> str:
        candidates = [
            transition
            for transition in self.transitions
            if transition.source_state_id == state_id
            and transition.action_schema == action_schema
            and transition.matches(evidence, visits=visits)
        ]
        guarded = [item for item in candidates if item.predicate]
        eligible = guarded or [item for item in candidates if not item.predicate]
        if not eligible:
            return state_id
        selected = min(
            eligible,
            key=lambda item: (
                -item.priority,
                item.predicate,
                not item.predicate_present,
                item.target_state_id,
            ),
        )
        return selected.target_state_id

    def _validate_execution(self, execution: OptionExecutionState) -> None:
        if execution.automaton_hash != self.canonical_hash:
            raise ValueError("execution state belongs to a different option automaton")
        if execution.maximum_horizon != self.maximum_horizon:
            raise ValueError("execution state has a mismatched option horizon")
        current = self.state(execution.state_id)
        if execution.awaiting_observation:
            schemas = self.allowed_action_schemas(current)
            if (
                current.terminal
                or execution.terminated
                or execution.pending_action_schema not in schemas
            ):
                raise ValueError(
                    "pending action schema is incoherent with the execution state"
                )
            return
        should_terminate = current.terminal or (execution.steps >= self.maximum_horizon)
        if execution.terminated != should_terminate:
            raise ValueError("terminated flag is incoherent with state and horizon")


def _action_schema(action: str | GroundedExecutionResult) -> str:
    if isinstance(action, GroundedExecutionResult):
        if not action.action_schema:
            raise ValueError("grounded result does not carry an action schema")
        return action.action_schema
    return _identifier(action, label="observed action schema")


def _evidence(
    events: Iterable[str],
    predicates: Iterable[str],
) -> frozenset[str]:
    return frozenset(
        _predicate(item, label="observed event")
        for item in itertools.chain(events, predicates)
    )


def _grounding_for_schema(groundings: Mapping[str, Any], schema: str) -> Any:
    normalized = {
        _identifier(str(key), label="grounding schema"): value
        for key, value in groundings.items()
    }
    if schema not in normalized:
        raise KeyError(f"missing grounding for action schema: {schema}")
    return normalized[schema]


def _ground_action(
    value: Any,
) -> tuple[str, tuple[tuple[str, str | int | float | bool | None], ...]]:
    if isinstance(value, str):
        name = value
        data: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        name = str(value.get("action_name", value.get("name", "")))
        raw_data = value.get(
            "action_data",
            value.get("action_args", value.get("data", {})),
        )
        data = dict(raw_data or {})
    elif isinstance(value, tuple) and len(value) == 2:
        name = str(value[0])
        data = dict(value[1] or {})
    else:
        name = str(getattr(value, "action_name", getattr(value, "name", "")))
        raw_data = getattr(
            value,
            "data",
            getattr(value, "action_args", getattr(value, "action_data", {})),
        )
        data = dict(raw_data or {})
    normalized_name = name.strip().upper()
    if not normalized_name:
        raise ValueError("grounded action needs a name")
    normalized_data = []
    for key, item in data.items():
        if not isinstance(item, _GROUND_VALUE):
            raise TypeError("grounded action arguments must be JSON scalars")
        normalized_data.append((str(key), item))
    return normalized_name, tuple(sorted(normalized_data))


def _grounding_relation(value: Any) -> str:
    """Extract the structural relation used to ground an option action."""

    if not isinstance(value, Mapping):
        return ""
    relation = value.get("relation", value.get("transport_relation", ""))
    if not str(relation).strip():
        return ""
    return _predicate(str(relation), label="grounded relation")


def _states(*state_ids: str) -> tuple[OptionState, ...]:
    return tuple(
        OptionState(state_id, terminal=state_id == "done") for state_id in state_ids
    )


def _transition(
    source: str,
    target: str,
    action: str,
    *,
    predicate: str = "",
    minimum_visits: int = 1,
    relation: str = "identity",
    priority: int = 0,
) -> OptionTransition:
    return OptionTransition(
        source_state_id=source,
        target_state_id=target,
        action_schema=action,
        predicate=predicate,
        minimum_visits=minimum_visits,
        relation=relation,
        priority=priority,
    )


def repeat(
    action_schema: str,
    *,
    termination_predicate: str = "level_complete",
    maximum_horizon: int = MAX_OPTION_HORIZON,
) -> OptionAutomaton:
    """Preserve the T10 repeat macro as a bounded option."""

    return OptionAutomaton(
        schema="repeat",
        states=_states("active", "done"),
        transitions=(
            _transition(
                "active",
                "done",
                action_schema,
                predicate=termination_predicate,
                priority=100,
            ),
            _transition("active", "active", action_schema),
        ),
        initial_state_id="active",
        maximum_horizon=maximum_horizon,
        initiation_condition=REGISTERED_TRUE_INITIATION,
    )


repeat_macro = repeat


def alternate(
    action_a: str,
    action_b: str,
    *,
    termination_predicate: str = "level_complete",
    maximum_horizon: int = MAX_OPTION_HORIZON,
) -> OptionAutomaton:
    """Alternate ``A, B, A, B, ...`` until progress or the horizon."""

    _require_distinct(action_a, action_b)
    return OptionAutomaton(
        schema="alternate",
        states=_states("first", "second", "done"),
        transitions=(
            _transition(
                "first",
                "done",
                action_a,
                predicate=termination_predicate,
                priority=100,
            ),
            _transition("first", "second", action_a),
            _transition(
                "second",
                "done",
                action_b,
                predicate=termination_predicate,
                priority=100,
            ),
            _transition("second", "first", action_b),
        ),
        initial_state_id="first",
        maximum_horizon=maximum_horizon,
        initiation_condition=REGISTERED_TRUE_INITIATION,
    )


def prime_then_repeat(
    action_a: str,
    prime_count: int,
    action_b: str,
    *,
    termination_predicate: str = "level_complete",
    maximum_horizon: int = MAX_OPTION_HORIZON,
) -> OptionAutomaton:
    """Apply ``A`` ``prime_count`` times, then repeat ``B``."""

    _require_distinct(action_a, action_b)
    count = int(prime_count)
    if count < 1 or count >= int(maximum_horizon):
        raise ValueError("prime_count must leave at least one action for B")
    return OptionAutomaton(
        schema="prime_then_repeat",
        states=_states("prime", "repeated", "done"),
        transitions=(
            _transition(
                "prime",
                "done",
                action_a,
                predicate=termination_predicate,
                priority=200,
            ),
            _transition(
                "prime",
                "repeated",
                action_a,
                minimum_visits=count,
            ),
            _transition(
                "repeated",
                "done",
                action_b,
                predicate=termination_predicate,
                priority=100,
            ),
            _transition("repeated", "repeated", action_b),
        ),
        initial_state_id="prime",
        maximum_horizon=maximum_horizon,
        initiation_condition=REGISTERED_TRUE_INITIATION,
    )


def until_then(
    action_a: str,
    predicate: str,
    action_b: str,
    *,
    termination_predicate: str = "level_complete",
    maximum_horizon: int = MAX_OPTION_HORIZON,
) -> OptionAutomaton:
    """Apply ``A`` until ``predicate`` is observed, then repeat ``B``."""

    _require_distinct(action_a, action_b)
    gate = _predicate(predicate, label="until predicate")
    terminal = _predicate(
        termination_predicate,
        label="termination predicate",
    )
    transitions = []
    if gate != terminal:
        transitions.append(
            _transition(
                "until",
                "done",
                action_a,
                predicate=terminal,
                priority=200,
            )
        )
    transitions.extend(
        (
            _transition(
                "until",
                "apply",
                action_a,
                predicate=gate,
                priority=100,
            ),
            _transition("until", "until", action_a),
            _transition(
                "apply",
                "done",
                action_b,
                predicate=terminal,
                priority=100,
            ),
            _transition("apply", "apply", action_b),
        )
    )
    return OptionAutomaton(
        schema="until_then",
        states=_states("until", "apply", "done"),
        transitions=tuple(transitions),
        initial_state_id="until",
        maximum_horizon=maximum_horizon,
        initiation_condition=REGISTERED_TRUE_INITIATION,
    )


a_until_predicate_then_b = until_then


def follow_relation_then_apply(
    action_a: str,
    action_b: str,
    *,
    relation: str = "successor_toward_enclosure",
    termination_predicate: str = "level_complete",
    maximum_horizon: int = MAX_OPTION_HORIZON,
) -> OptionAutomaton:
    """Follow one relation-grounded ``A`` edge, then repeatedly apply ``B``."""

    _require_distinct(action_a, action_b)
    return OptionAutomaton(
        schema="follow_relation_then_apply",
        states=_states("follow", "apply", "done"),
        transitions=(
            _transition(
                "follow",
                "done",
                action_a,
                predicate=termination_predicate,
                relation=relation,
                priority=200,
            ),
            _transition(
                "follow",
                "apply",
                action_a,
                relation=relation,
            ),
            _transition(
                "apply",
                "done",
                action_b,
                predicate=termination_predicate,
                priority=100,
            ),
            _transition("apply", "apply", action_b),
        ),
        initial_state_id="follow",
        maximum_horizon=maximum_horizon,
        initiation_condition=REGISTERED_TRUE_INITIATION,
    )


def distinct_noop_probe(
    action_a: str,
    action_b: str,
    *,
    termination_predicate: str,
    maximum_horizon: int = MAX_OPTION_HORIZON,
) -> OptionAutomaton:
    """Retain a one-step probe even when its first observation is ``no_effect``."""

    _require_distinct(action_a, action_b)
    return OptionAutomaton(
        schema="distinct_noop_probe",
        states=_states("probe", "followup", "done"),
        transitions=(
            _transition(
                "probe",
                "done",
                action_a,
                predicate=termination_predicate,
                priority=200,
            ),
            _transition("probe", "followup", action_a),
            _transition(
                "followup",
                "done",
                action_b,
                predicate=termination_predicate,
                priority=100,
            ),
            _transition("followup", "followup", action_b),
        ),
        initial_state_id="probe",
        maximum_horizon=maximum_horizon,
        initiation_condition=REGISTERED_TRUE_INITIATION,
    )


noop_probe = distinct_noop_probe


def generate_mixed_grammar(
    action_schemas: Sequence[str],
    *,
    predicates: Sequence[str] = ("state_change",),
    prime_counts: Sequence[int] = (1,),
    relation: str = "successor_toward_enclosure",
    termination_predicate: str = "level_complete",
    noop_termination_predicates: Sequence[str] = (
        "state_change",
        "level_complete",
    ),
    maximum_horizon: int = MAX_OPTION_HORIZON,
) -> tuple[OptionAutomaton, ...]:
    """Generate the bounded registered T10.2 mixed-sequence grammar."""

    actions = tuple(
        dict.fromkeys(
            _identifier(item, label="grammar action schema") for item in action_schemas
        )
    )
    if not actions:
        return ()
    options = [
        repeat(
            action,
            termination_predicate=termination_predicate,
            maximum_horizon=maximum_horizon,
        )
        for action in actions
    ]
    for first, second in itertools.permutations(actions, 2):
        options.append(
            alternate(
                first,
                second,
                termination_predicate=termination_predicate,
                maximum_horizon=maximum_horizon,
            )
        )
        for count in prime_counts:
            options.append(
                prime_then_repeat(
                    first,
                    count,
                    second,
                    termination_predicate=termination_predicate,
                    maximum_horizon=maximum_horizon,
                )
            )
        for predicate in predicates:
            options.append(
                until_then(
                    first,
                    predicate,
                    second,
                    termination_predicate=termination_predicate,
                    maximum_horizon=maximum_horizon,
                )
            )
        options.append(
            follow_relation_then_apply(
                first,
                second,
                relation=relation,
                termination_predicate=termination_predicate,
                maximum_horizon=maximum_horizon,
            )
        )
        for predicate in noop_termination_predicates:
            options.append(
                distinct_noop_probe(
                    first,
                    second,
                    termination_predicate=predicate,
                    maximum_horizon=maximum_horizon,
                )
            )
    unique: dict[str, OptionAutomaton] = {}
    for option in options:
        unique.setdefault(option.canonical_hash, option)
    return tuple(unique[key] for key in sorted(unique))


def _require_distinct(action_a: str, action_b: str) -> None:
    first = _identifier(action_a, label="first action schema")
    second = _identifier(action_b, label="second action schema")
    if first == second:
        raise ValueError("mixed options require two distinct action schemas")


__all__ = [
    "AST_NODE_LOG_COST",
    "MAX_ACTION_SCHEMAS",
    "MAX_OPTION_HORIZON",
    "MAX_OPTION_STATES",
    "REGISTERED_TRUE_INITIATION",
    "GroundedExecutionResult",
    "OptionAutomaton",
    "OptionExecutionState",
    "OptionInitiationCondition",
    "OptionState",
    "OptionTransition",
    "a_until_predicate_then_b",
    "alternate",
    "ast_node_cost",
    "ast_node_log_cost",
    "distinct_noop_probe",
    "follow_relation_then_apply",
    "generate_mixed_grammar",
    "noop_probe",
    "prime_then_repeat",
    "repeat",
    "repeat_macro",
    "until_then",
]
