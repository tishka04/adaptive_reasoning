"""Typed contracts for the SAGE.T joint program posterior.

The runtime deliberately separates three kinds of objects:

* :class:`ProgramFragment` values are proposals.  They never carry empirical
  support and cannot directly select an action.
* :class:`JointProgramHypothesis` values are complete, executable programs.
* :class:`PredictionPacket` values are comparable counterfactual predictions
  emitted by the single canonical executor.

Programs may bind local action names, but their executable clauses cannot
contain game ids, pixel values, absolute coordinates, or grounded entity ids.
Those values belong to :class:`AbstractState`, never to transferable theory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VARIABLE = re.compile(r"^\?[a-z][a-z0-9_]{0,31}$")
_SPECIAL_TERM = re.compile(r"^\$(actor|target|selected|action)$")
_LOCAL_ACTION = re.compile(r"^(ACTION[1-9][0-9]*|[A-Z][A-Z0-9_]{0,31})$")
_GROUNDED_ENTITY = re.compile(r"^e(?:_player|[0-9]+(?::.*)?)$")
_COORDINATE = re.compile(r"^-?[0-9]+(?:,-?[0-9]+)?$")


ALLOWED_PREDICATES = frozenset(
    {
        "exists",
        "role",
        "selected",
        "same_attribute",
        "same_color",
        "same_shape",
        "contact",
        "adjacent",
        "aligned",
        "inside",
        "encloses",
        "reachable",
        "near",
        "north_of",
        "south_of",
        "east_of",
        "west_of",
        "solved",
        "changed",
        "moved",
        "created",
        "removed",
        "attached",
        "detached",
        "morphology_changed",
        "component_count_changed",
        "hole_count_changed",
        "relation_changed",
        "progress",
        "level_complete",
        "game_over",
        "no_effect",
    }
)

ALLOWED_EXPRESSION_OPS = frozenset(
    {
        "const",
        "fact",
        "counter",
        "not",
        "and",
        "or",
        "exists",
        "forall",
        "count",
        "ratio",
        "eq",
        "gt",
        "ge",
        "lt",
        "le",
    }
)

ALLOWED_ACTION_OPERATORS = frozenset(
    {
        "select",
        "apply",
        "move",
        "transform",
        "create",
        "remove",
        "attach",
        "detach",
        "mark_solved",
        "toggle",
        "noop",
    }
)

ALLOWED_EFFECT_OPERATIONS = frozenset(
    {
        "assert",
        "retract",
        "set_register",
        "clear_register",
        "set_counter",
        "increment_counter",
        "move_relative",
        "change_morphology",
        "progress",
        "win",
        "fail",
    }
)

RELATION_PREDICATES = frozenset(
    {
        "contact",
        "adjacent",
        "aligned",
        "inside",
        "encloses",
        "reachable",
        "near",
        "north_of",
        "south_of",
        "east_of",
        "west_of",
        "same_attribute",
        "same_color",
        "same_shape",
        "attached",
        "detached",
        "relation_changed",
    }
)

TOPOLOGY_PREDICATES = frozenset(
    {
        "component_count_changed",
        "hole_count_changed",
        "morphology_changed",
    }
)

OBJECT_EVENT_PREDICATES = frozenset(
    {
        "changed",
        "moved",
        "created",
        "removed",
        "solved",
        "no_effect",
    }
)


def _safe_identifier(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{label} must be a bounded snake_case identifier")
    return normalized


def _safe_program_identifier(value: str, *, label: str) -> str:
    normalized = _safe_identifier(value, label=label)
    if normalized.startswith(("game_", "color_", "pixel_", "value_")):
        raise ValueError(f"forbidden local constant in {label}: {value}")
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


class TruthValue(str, Enum):
    """Three-valued state truth used by partial symbolic predictions."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True)
class GroundFact:
    """One grounded predicate in an observed or simulated abstract state."""

    predicate: str
    terms: tuple[str, ...] = ()
    value: str = ""

    def __post_init__(self) -> None:
        predicate = _safe_identifier(self.predicate, label="predicate")
        if predicate not in ALLOWED_PREDICATES:
            raise ValueError(f"unsupported predicate: {predicate}")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(
            self,
            "terms",
            tuple(str(term)[:96] for term in self.terms),
        )
        object.__setattr__(self, "value", str(self.value)[:96])

    @property
    def key(self) -> str:
        terms = "|".join(self.terms) if self.terms else "-"
        return f"{self.predicate}|{terms}|{self.value}"

    @classmethod
    def from_key(cls, key: str) -> GroundFact:
        parts = str(key).split("|")
        if len(parts) >= 4:
            predicate = parts[0]
            terms = tuple(part for part in parts[1:-1] if part != "-")
            value = parts[-1]
        elif len(parts) == 3:
            predicate, subject, target = parts
            terms = tuple(part for part in (subject, target) if part != "-")
            value = ""
        else:
            predicate = parts[0]
            terms = tuple(parts[1:])
            value = ""
        return cls(predicate=predicate, terms=terms, value=value)


@dataclass(frozen=True)
class AbstractEntity:
    """Grounded entity available to a program through structural roles."""

    entity_id: str
    roles: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...] = ()
    center: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("abstract entity needs an id")
        object.__setattr__(
            self,
            "roles",
            tuple(
                sorted({_safe_identifier(role, label="role") for role in self.roles})
            ),
        )
        object.__setattr__(
            self,
            "attributes",
            tuple(
                sorted(
                    (
                        _safe_identifier(str(key), label="attribute"),
                        str(value)[:64],
                    )
                    for key, value in self.attributes
                )
            ),
        )
        if self.center is not None:
            object.__setattr__(
                self,
                "center",
                (float(self.center[0]), float(self.center[1])),
            )

    def has_role(self, role: str) -> bool:
        return str(role).lower() in self.roles


@dataclass(frozen=True)
class AbstractState:
    """Identity-free public facts plus grounded branch-local entity bindings."""

    entities: tuple[AbstractEntity, ...] = ()
    true_facts: frozenset[GroundFact] = frozenset()
    false_facts: frozenset[GroundFact] = frozenset()
    counters: tuple[tuple[str, float], ...] = ()
    registers: tuple[tuple[str, str], ...] = ()
    topology: tuple[tuple[str, int], ...] = ()
    regime_index: int = 0

    def __post_init__(self) -> None:
        overlap = self.true_facts & self.false_facts
        if overlap:
            raise ValueError(f"facts cannot be both true and false: {overlap}")
        object.__setattr__(
            self,
            "counters",
            tuple(
                sorted(
                    (
                        _safe_identifier(key, label="counter"),
                        float(value),
                    )
                    for key, value in self.counters
                )
            ),
        )
        object.__setattr__(
            self,
            "registers",
            tuple(
                sorted(
                    (
                        _safe_identifier(key, label="register"),
                        str(value)[:96],
                    )
                    for key, value in self.registers
                )
            ),
        )
        object.__setattr__(
            self,
            "topology",
            tuple(
                sorted(
                    (
                        _safe_identifier(key, label="topology key"),
                        int(value),
                    )
                    for key, value in self.topology
                )
            ),
        )
        object.__setattr__(self, "regime_index", max(0, int(self.regime_index)))

    def truth(self, fact: GroundFact) -> TruthValue:
        if fact in self.true_facts:
            return TruthValue.TRUE
        if fact in self.false_facts:
            return TruthValue.FALSE
        return TruthValue.UNKNOWN

    def entities_for_role(self, role: str) -> tuple[AbstractEntity, ...]:
        normalized = str(role).lower()
        return tuple(entity for entity in self.entities if normalized in entity.roles)

    def counter(self, key: str, default: float = 0.0) -> float:
        return float(dict(self.counters).get(str(key).lower(), default))

    def register(self, key: str, default: str = "") -> str:
        return str(dict(self.registers).get(str(key).lower(), default))

    def with_updates(
        self,
        *,
        asserted: Iterable[GroundFact] = (),
        retracted: Iterable[GroundFact] = (),
        counters: Mapping[str, float] | None = None,
        registers: Mapping[str, str | None] | None = None,
        regime_index: int | None = None,
    ) -> AbstractState:
        true_facts = set(self.true_facts)
        false_facts = set(self.false_facts)
        for fact in asserted:
            true_facts.add(fact)
            false_facts.discard(fact)
        for fact in retracted:
            true_facts.discard(fact)
            false_facts.add(fact)
        next_counters = dict(self.counters)
        next_counters.update(
            {str(key).lower(): float(value) for key, value in (counters or {}).items()}
        )
        next_registers = dict(self.registers)
        for key, value in (registers or {}).items():
            normalized = str(key).lower()
            if value is None:
                next_registers.pop(normalized, None)
            else:
                next_registers[normalized] = str(value)
        return AbstractState(
            entities=self.entities,
            true_facts=frozenset(true_facts),
            false_facts=frozenset(false_facts),
            counters=tuple(next_counters.items()),
            registers=tuple(next_registers.items()),
            topology=self.topology,
            regime_index=(
                self.regime_index if regime_index is None else int(regime_index)
            ),
        )

    def merge_observation(self, observed: AbstractState) -> AbstractState:
        """Use observed public state while preserving hidden program registers."""

        true_facts = set(observed.true_facts)
        false_facts = set(observed.false_facts)
        for fact in self.true_facts:
            if (
                fact.predicate in {"solved", "attached", "detached"}
                and observed.truth(fact) is TruthValue.UNKNOWN
            ):
                true_facts.add(fact)
        for fact in self.false_facts:
            if (
                fact.predicate in {"solved", "attached", "detached"}
                and observed.truth(fact) is TruthValue.UNKNOWN
            ):
                false_facts.add(fact)
        counters = dict(self.counters)
        counters.update(observed.counters)
        return AbstractState(
            entities=observed.entities,
            true_facts=frozenset(true_facts),
            false_facts=frozenset(false_facts),
            counters=tuple(counters.items()),
            registers=self.registers,
            topology=observed.topology,
            regime_index=observed.regime_index,
        )

    @property
    def signature(self) -> str:
        public = {
            "entities": [
                {
                    "roles": entity.roles,
                    "attributes": entity.attributes,
                }
                for entity in self.entities
            ],
            "true": sorted(fact.key for fact in self.true_facts),
            "false": sorted(fact.key for fact in self.false_facts),
            "counters": self.counters,
            "registers": self.registers,
            "topology": self.topology,
            "regime": self.regime_index,
        }
        return hashlib.sha256(_canonical_json(public).encode("utf-8")).hexdigest()[:20]

    @property
    def execution_signature(self) -> str:
        """Cache identity including branch-local grounding coordinates."""

        grounded = {
            "abstract": self.signature,
            "entities": [
                {
                    "id": entity.entity_id,
                    "center": entity.center,
                }
                for entity in self.entities
            ],
        }
        return hashlib.sha256(_canonical_json(grounded).encode("utf-8")).hexdigest()[
            :20
        ]


@dataclass(frozen=True)
class Expression:
    """Small typed expression tree shared by conditions, progress and goals."""

    op: str
    args: tuple[Expression, ...] = ()
    predicate: str = ""
    terms: tuple[str, ...] = ()
    value: bool | float | str | None = None
    variable: str = ""
    role: str = ""

    def __post_init__(self) -> None:
        op = _safe_identifier(self.op, label="expression op")
        if op not in ALLOWED_EXPRESSION_OPS:
            raise ValueError(f"unsupported expression op: {op}")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "args", tuple(self.args))
        if self.predicate:
            predicate = _safe_identifier(self.predicate, label="predicate")
            if predicate not in ALLOWED_PREDICATES:
                raise ValueError(f"unsupported predicate: {predicate}")
            object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "terms", tuple(str(term) for term in self.terms))
        if self.variable:
            variable = str(self.variable)
            if not _VARIABLE.fullmatch(variable):
                raise ValueError("quantifier variable must look like ?entity")
            object.__setattr__(self, "variable", variable)
        if self.role:
            object.__setattr__(
                self,
                "role",
                _safe_program_identifier(self.role, label="role"),
            )
        self._validate_shape()
        for term in self.terms:
            _validate_program_term(term)

    def _validate_shape(self) -> None:
        if self.op == "fact" and not self.predicate:
            raise ValueError("fact expression needs a predicate")
        if self.op in {"not"} and len(self.args) != 1:
            raise ValueError(f"{self.op} expression needs one argument")
        if self.op in {"and", "or"} and not self.args:
            raise ValueError(f"{self.op} expression needs arguments")
        if self.op in {"exists", "forall", "count"}:
            if len(self.args) > 1 or not self.role:
                raise ValueError(f"{self.op} needs a role and at most one body")
            if self.args and not self.variable:
                raise ValueError(f"{self.op} body needs a variable")
        if self.op == "ratio" and len(self.args) != 2:
            raise ValueError("ratio needs numerator and denominator")
        if self.op in {"eq", "gt", "ge", "lt", "le"} and len(self.args) != 2:
            raise ValueError(f"{self.op} needs two arguments")
        if self.op == "counter" and not isinstance(self.value, str):
            raise ValueError("counter expression value must name a counter")
        if self.op == "counter":
            _safe_program_identifier(str(self.value), label="counter")
        if self.op == "const":
            try:
                finite_constant = (
                    not isinstance(self.value, str)
                    and self.value is not None
                    and math.isfinite(float(self.value))
                )
            except (TypeError, ValueError):
                finite_constant = False
            if not finite_constant:
                raise ValueError(
                    "const expression must contain a finite boolean or number"
                )

    @property
    def node_count(self) -> int:
        return 1 + sum(arg.node_count for arg in self.args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": [arg.to_dict() for arg in self.args],
            "predicate": self.predicate,
            "terms": list(self.terms),
            "value": self.value,
            "variable": self.variable,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Expression:
        return cls(
            op=str(payload["op"]),
            args=tuple(
                cls.from_dict(item)
                for item in payload.get("args", ())
                if isinstance(item, Mapping)
            ),
            predicate=str(payload.get("predicate", "")),
            terms=tuple(str(item) for item in payload.get("terms", ())),
            value=payload.get("value"),
            variable=str(payload.get("variable", "")),
            role=str(payload.get("role", "")),
        )

    @classmethod
    def constant(cls, value: bool | float) -> Expression:
        return cls(op="const", value=value)

    @classmethod
    def fact(cls, predicate: str, *terms: str) -> Expression:
        return cls(op="fact", predicate=predicate, terms=tuple(terms))


def _validate_program_term(term: str) -> None:
    text = str(term)
    if _VARIABLE.fullmatch(text) or _SPECIAL_TERM.fullmatch(text):
        return
    normalized = text.lower()
    if _IDENTIFIER.fullmatch(normalized):
        if (
            _GROUNDED_ENTITY.fullmatch(text)
            or _COORDINATE.fullmatch(text)
            or normalized.startswith(("game_", "color_", "pixel_", "value_"))
        ):
            raise ValueError(f"forbidden grounded program constant: {term}")
        return
    raise ValueError(f"invalid or grounded program term: {term}")


@dataclass(frozen=True)
class Effect:
    """One state edit emitted by a transition rule."""

    operation: str
    predicate: str = ""
    terms: tuple[str, ...] = ()
    key: str = ""
    value: str | float | None = None

    def __post_init__(self) -> None:
        operation = _safe_identifier(self.operation, label="effect operation")
        if operation not in ALLOWED_EFFECT_OPERATIONS:
            raise ValueError(f"unsupported effect operation: {operation}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "terms", tuple(str(term) for term in self.terms))
        if operation in {"assert", "retract"}:
            predicate = _safe_identifier(self.predicate, label="predicate")
            if predicate not in ALLOWED_PREDICATES:
                raise ValueError(f"unsupported predicate: {predicate}")
            object.__setattr__(self, "predicate", predicate)
            for term in self.terms:
                _validate_program_term(term)
        elif operation in {"move_relative", "change_morphology"}:
            for term in self.terms:
                _validate_program_term(term)
            if self.value is not None:
                _safe_program_identifier(
                    str(self.value),
                    label=f"{operation} value",
                )
        elif operation == "progress":
            try:
                numeric = float(self.value or 0.0)
            except (TypeError, ValueError) as exc:
                raise ValueError("progress effects need a numeric value") from exc
            if not math.isfinite(numeric):
                raise ValueError("progress effects need a finite value")
        elif operation in {"win", "fail"}:
            if self.terms or self.key or self.value is not None:
                raise ValueError(f"{operation} effects take no arguments")
        else:
            object.__setattr__(
                self,
                "key",
                _safe_program_identifier(self.key, label="key"),
            )
            if operation == "set_register" and self.value is not None:
                _validate_program_term(str(self.value))
            if operation in {"set_counter", "increment_counter"}:
                try:
                    numeric = float(self.value or 0.0)
                except (TypeError, ValueError) as exc:
                    raise ValueError("counter effects need a numeric value") from exc
                if not math.isfinite(numeric):
                    raise ValueError("counter effects need a finite value")

    @property
    def node_count(self) -> int:
        return 1


@dataclass(frozen=True)
class ObjectSchema:
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        roles = tuple(
            sorted(
                {_safe_program_identifier(role, label="role") for role in self.roles}
            )
        )
        if not roles:
            raise ValueError("object schema needs at least one role")
        object.__setattr__(self, "roles", roles)


@dataclass(frozen=True)
class ActionBinding:
    """Local primitive action mapped onto a transferable semantic operator."""

    action_name: str
    operator: str
    target_role: str = ""

    def __post_init__(self) -> None:
        action = str(self.action_name).strip().upper()
        if action == "RESET" or not _LOCAL_ACTION.fullmatch(action):
            raise ValueError(f"invalid local action binding: {action}")
        operator = _safe_identifier(self.operator, label="action operator")
        if operator not in ALLOWED_ACTION_OPERATORS:
            raise ValueError(f"unsupported action operator: {operator}")
        object.__setattr__(self, "action_name", action)
        object.__setattr__(self, "operator", operator)
        if self.target_role:
            object.__setattr__(
                self,
                "target_role",
                _safe_program_identifier(
                    self.target_role,
                    label="target role",
                ),
            )


@dataclass(frozen=True)
class TransitionRule:
    rule_id: str
    action_operator: str
    condition: Expression
    effects: tuple[Effect, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rule_id",
            _safe_identifier(self.rule_id, label="rule id"),
        )
        operator = _safe_identifier(self.action_operator, label="action operator")
        if operator not in ALLOWED_ACTION_OPERATORS:
            raise ValueError(f"unsupported action operator: {operator}")
        object.__setattr__(self, "action_operator", operator)
        object.__setattr__(self, "effects", tuple(self.effects))
        if not self.effects:
            raise ValueError("transition rule needs at least one effect")

    @property
    def node_count(self) -> int:
        return 1 + self.condition.node_count + len(self.effects)


@dataclass(frozen=True)
class ProgressRule:
    expression: Expression

    @property
    def node_count(self) -> int:
        return 1 + self.expression.node_count


@dataclass(frozen=True)
class GoalRule:
    expression: Expression
    family: str = "generic"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _safe_program_identifier(
                self.family,
                label="goal family",
            ),
        )

    @property
    def node_count(self) -> int:
        return 1 + self.expression.node_count


@dataclass(frozen=True)
class TerminalRule:
    expression: Expression
    outcome: str

    def __post_init__(self) -> None:
        outcome = _safe_identifier(self.outcome, label="terminal outcome")
        if outcome not in {"win", "game_over"}:
            raise ValueError("terminal outcome must be win or game_over")
        object.__setattr__(self, "outcome", outcome)

    @property
    def node_count(self) -> int:
        return 1 + self.expression.node_count


@dataclass(frozen=True)
class ProgramFragment:
    """Typed, non-authoritative input to the program assembler."""

    fragment_id: str
    kind: str
    payload: Any
    roles: tuple[str, ...] = ()
    predicted_events: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    prior_logprob: float = 0.0
    support: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fragment_id",
            _safe_identifier(self.fragment_id, label="fragment id"),
        )
        kind = _safe_identifier(self.kind, label="fragment kind")
        if kind not in {"schema", "dynamics", "goal_bundle", "terminal", "plan"}:
            raise ValueError(f"unsupported fragment kind: {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "roles",
            tuple(
                sorted({_safe_identifier(role, label="role") for role in self.roles})
            ),
        )
        object.__setattr__(
            self,
            "predicted_events",
            tuple(sorted({str(event)[:96] for event in self.predicted_events})),
        )
        object.__setattr__(
            self,
            "provenance",
            tuple(sorted({str(item)[:160] for item in self.provenance})),
        )
        if int(self.support) != 0:
            raise ValueError("program fragments must enter with support=0")
        if not math.isfinite(float(self.prior_logprob)):
            raise ValueError("fragment prior must be finite")


@dataclass(frozen=True)
class JointProgramHypothesis:
    """Complete executable particle in the common program posterior."""

    program_id: str
    object_schema: ObjectSchema
    action_bindings: tuple[ActionBinding, ...]
    transition_rules: tuple[TransitionRule, ...]
    progress_rule: ProgressRule
    terminal_rules: tuple[TerminalRule, ...]
    goal_rule: GoalRule
    provenance: tuple[str, ...] = ()
    parent_hash: str = ""
    edit_distance: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "program_id",
            _safe_identifier(self.program_id, label="program id"),
        )
        object.__setattr__(self, "action_bindings", tuple(self.action_bindings))
        object.__setattr__(self, "transition_rules", tuple(self.transition_rules))
        object.__setattr__(self, "terminal_rules", tuple(self.terminal_rules))
        object.__setattr__(
            self,
            "provenance",
            tuple(sorted({str(item)[:160] for item in self.provenance})),
        )
        object.__setattr__(self, "edit_distance", max(0, int(self.edit_distance)))
        if not self.action_bindings:
            raise ValueError("joint program needs action semantics")
        if not self.transition_rules:
            raise ValueError("joint program needs transition rules")
        if not self.terminal_rules:
            raise ValueError("joint program needs terminal rules")
        bound = {binding.operator for binding in self.action_bindings}
        unbound = {
            rule.action_operator
            for rule in self.transition_rules
            if rule.action_operator not in bound
        }
        if unbound:
            raise ValueError(
                f"transition rules use unbound operators: {sorted(unbound)}"
            )

    @property
    def node_count(self) -> int:
        return (
            1
            + len(self.object_schema.roles)
            + len(self.action_bindings)
            + sum(rule.node_count for rule in self.transition_rules)
            + self.progress_rule.node_count
            + sum(rule.node_count for rule in self.terminal_rules)
            + self.goal_rule.node_count
        )

    @property
    def local_constant_count(self) -> int:
        return len({binding.action_name for binding in self.action_bindings})

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "object_schema": asdict(self.object_schema),
            "action_bindings": [
                asdict(item)
                for item in sorted(
                    self.action_bindings,
                    key=lambda item: (
                        item.action_name,
                        item.operator,
                        item.target_role,
                    ),
                )
            ],
            "transition_rules": _canonical_transition_rules(self.transition_rules),
            "progress": _alpha_normalize_expression(self.progress_rule.expression),
            "terminal": [
                {
                    "outcome": rule.outcome,
                    "expression": _alpha_normalize_expression(rule.expression),
                }
                for rule in sorted(self.terminal_rules, key=lambda item: item.outcome)
            ],
            "goal": {
                "family": self.goal_rule.family,
                "expression": _alpha_normalize_expression(self.goal_rule.expression),
            },
        }

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.canonical_payload).encode("utf-8")
        ).hexdigest()

    @property
    def semantic_family(self) -> tuple[str, str]:
        dynamics = ",".join(
            sorted({binding.operator for binding in self.action_bindings})
        )
        return dynamics, self.goal_rule.family


def _alpha_normalize_transition_rule(
    rule: TransitionRule,
) -> dict[str, Any]:
    variables: dict[str, str] = {}
    return {
        "action_operator": rule.action_operator,
        "condition": _alpha_normalize_expression(
            rule.condition,
            variables=variables,
        ),
        "effects": [
            _alpha_normalize_effect(effect, variables=variables)
            for effect in rule.effects
        ],
    }


def _canonical_transition_rules(
    rules: Sequence[TransitionRule],
) -> list[dict[str, Any]]:
    normalized = [_alpha_normalize_transition_rule(rule) for rule in rules]
    return sorted(normalized, key=_canonical_json)


def _alpha_normalize_expression(
    expression: Expression,
    *,
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    variables = {} if variables is None else variables

    def normalize(item: Expression) -> dict[str, Any]:
        if item.variable and item.variable not in variables:
            variables[item.variable] = f"?v{len(variables)}"
        terms = []
        for term in item.terms:
            if term.startswith("?"):
                variables.setdefault(term, f"?v{len(variables)}")
                terms.append(variables[term])
            else:
                terms.append(term)
        return {
            "op": item.op,
            "args": [normalize(arg) for arg in item.args],
            "predicate": item.predicate,
            "terms": terms,
            "value": item.value,
            "variable": variables.get(item.variable, item.variable),
            "role": item.role,
        }

    return normalize(expression)


def _alpha_normalize_effect(
    effect: Effect,
    *,
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    variables = {} if variables is None else variables
    terms = []
    for term in effect.terms:
        if term.startswith("?"):
            variables.setdefault(term, f"?v{len(variables)}")
            terms.append(variables[term])
        else:
            terms.append(term)
    value = effect.value
    if isinstance(value, str) and value.startswith("?"):
        variables.setdefault(value, f"?v{len(variables)}")
        value = variables[value]
    return {
        "operation": effect.operation,
        "predicate": effect.predicate,
        "terms": terms,
        "key": effect.key,
        "value": value,
    }


@dataclass(frozen=True)
class ActionCandidate:
    action_name: str
    action_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = str(self.action_name).strip().upper()
        if not _LOCAL_ACTION.fullmatch(action):
            raise ValueError(f"invalid action candidate: {action}")
        object.__setattr__(self, "action_name", action)
        object.__setattr__(
            self,
            "action_data",
            MappingProxyType(dict(self.action_data)),
        )

    @property
    def key(self) -> str:
        return f"{self.action_name}:{_canonical_json(dict(self.action_data))}"


@dataclass(frozen=True)
class PredictionPacket:
    """Comparable, partially-known prediction emitted by every program."""

    object_deltas: Mapping[str, float] = field(default_factory=dict)
    relation_deltas: Mapping[str, float] = field(default_factory=dict)
    topology_deltas: Mapping[str, float] = field(default_factory=dict)
    progress_mean: float | None = None
    progress_distribution: Mapping[str, float] = field(default_factory=dict)
    terminal_probability: float | None = None
    goal_probability: float | None = None
    known_channels: frozenset[str] = frozenset()
    residual: tuple[float, ...] = ()
    state_after: AbstractState | None = None

    def __post_init__(self) -> None:
        allowed = {"objects", "relations", "topology", "progress", "terminal", "goal"}
        unknown = set(self.known_channels) - allowed
        if unknown:
            raise ValueError(f"unknown prediction channels: {sorted(unknown)}")
        for mapping in (
            self.object_deltas,
            self.relation_deltas,
            self.topology_deltas,
        ):
            if any(not 0.0 <= float(value) <= 1.0 for value in mapping.values()):
                raise ValueError("event probabilities must be in [0, 1]")
        for value in (self.terminal_probability, self.goal_probability):
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError("Bernoulli probability must be in [0, 1]")
        if self.progress_mean is not None and not math.isfinite(
            float(self.progress_mean)
        ):
            raise ValueError("progress prediction must be finite")
        distribution = dict(self.progress_distribution)
        if not distribution and self.progress_mean is not None:
            distribution = {
                f"value:{float(self.progress_mean):.6g}": 1.0,
            }
        if distribution:
            if any(
                not 0.0 <= float(probability) <= 1.0
                for probability in distribution.values()
            ):
                raise ValueError(
                    "progress distribution probabilities must be in [0, 1]"
                )
            if not math.isclose(
                sum(float(value) for value in distribution.values()),
                1.0,
                abs_tol=1e-6,
            ):
                raise ValueError("progress distribution must sum to one")
        object.__setattr__(
            self,
            "progress_distribution",
            MappingProxyType(distribution),
        )
        for attribute in (
            "object_deltas",
            "relation_deltas",
            "topology_deltas",
        ):
            object.__setattr__(
                self,
                attribute,
                MappingProxyType(dict(getattr(self, attribute))),
            )

    @property
    def coverage(self) -> float:
        return len(self.known_channels) / 6.0

    @property
    def unknown_channels(self) -> frozenset[str]:
        return frozenset(
            {
                "objects",
                "relations",
                "topology",
                "progress",
                "terminal",
                "goal",
            }
            - set(self.known_channels)
        )

    def channel_signature(self, channel: str) -> tuple[Any, ...]:
        if channel == "objects":
            return tuple(
                sorted(
                    key
                    for key, value in self.object_deltas.items()
                    if float(value) >= 0.5
                )
            )
        if channel == "relations":
            return tuple(
                sorted(
                    key
                    for key, value in self.relation_deltas.items()
                    if float(value) >= 0.5
                )
            )
        if channel == "topology":
            return tuple(
                sorted(
                    key
                    for key, value in self.topology_deltas.items()
                    if float(value) >= 0.5
                )
            )
        if channel == "progress":
            if self.progress_distribution:
                return tuple(
                    sorted(
                        (
                            key,
                            round(float(probability), 4),
                        )
                        for key, probability in (self.progress_distribution.items())
                    )
                )
            return (
                None
                if self.progress_mean is None
                else round(float(self.progress_mean), 2),
            )
        if channel == "terminal":
            return (
                None
                if self.terminal_probability is None
                else int(float(self.terminal_probability) >= 0.5),
            )
        if channel == "goal":
            return (
                None
                if self.goal_probability is None
                else int(float(self.goal_probability) >= 0.5),
            )
        raise ValueError(f"unknown channel: {channel}")

    @property
    def full_signature(self) -> tuple[Any, ...]:
        return tuple(
            (channel, self.channel_signature(channel))
            for channel in (
                "objects",
                "relations",
                "topology",
                "progress",
                "terminal",
                "goal",
            )
            if channel in self.known_channels
        )


@dataclass(frozen=True)
class ObservedTransition:
    """Canonical evidence packet compiled from one real transition."""

    state_before: AbstractState
    action: ActionCandidate
    state_after: AbstractState
    observation: PredictionPacket
    events: tuple[str, ...] = ()
    reset: bool = False


@dataclass(frozen=True)
class RolloutPrediction:
    sequence: tuple[ActionCandidate, ...]
    packets: tuple[PredictionPacket, ...]
    final_state: AbstractState

    @property
    def final_packet(self) -> PredictionPacket:
        if not self.packets:
            return PredictionPacket(state_after=self.final_state)
        return self.packets[-1]


def program_from_dict(payload: Mapping[str, Any]) -> JointProgramHypothesis:
    """Deserialize a complete program using the same validating constructors."""

    def expression(raw: Mapping[str, Any]) -> Expression:
        return Expression.from_dict(raw)

    return JointProgramHypothesis(
        program_id=str(payload["program_id"]),
        object_schema=ObjectSchema(roles=tuple(payload["object_schema"]["roles"])),
        action_bindings=tuple(
            ActionBinding(**dict(item)) for item in payload.get("action_bindings", ())
        ),
        transition_rules=tuple(
            TransitionRule(
                rule_id=str(item["rule_id"]),
                action_operator=str(item["action_operator"]),
                condition=expression(item["condition"]),
                effects=tuple(
                    Effect(**dict(effect)) for effect in item.get("effects", ())
                ),
            )
            for item in payload.get("transition_rules", ())
        ),
        progress_rule=ProgressRule(
            expression=expression(payload["progress_rule"]["expression"])
        ),
        terminal_rules=tuple(
            TerminalRule(
                expression=expression(item["expression"]),
                outcome=str(item["outcome"]),
            )
            for item in payload.get("terminal_rules", ())
        ),
        goal_rule=GoalRule(
            expression=expression(payload["goal_rule"]["expression"]),
            family=str(payload["goal_rule"].get("family", "generic")),
        ),
        provenance=tuple(payload.get("provenance", ())),
        parent_hash=str(payload.get("parent_hash", "")),
        edit_distance=int(payload.get("edit_distance", 0)),
    )


def program_to_dict(program: JointProgramHypothesis) -> dict[str, Any]:
    return {
        "program_id": program.program_id,
        "object_schema": asdict(program.object_schema),
        "action_bindings": [asdict(item) for item in program.action_bindings],
        "transition_rules": [
            {
                "rule_id": rule.rule_id,
                "action_operator": rule.action_operator,
                "condition": rule.condition.to_dict(),
                "effects": [asdict(effect) for effect in rule.effects],
            }
            for rule in program.transition_rules
        ],
        "progress_rule": {"expression": program.progress_rule.expression.to_dict()},
        "terminal_rules": [
            {
                "expression": rule.expression.to_dict(),
                "outcome": rule.outcome,
            }
            for rule in program.terminal_rules
        ],
        "goal_rule": {
            "expression": program.goal_rule.expression.to_dict(),
            "family": program.goal_rule.family,
        },
        "provenance": list(program.provenance),
        "parent_hash": program.parent_hash,
        "edit_distance": program.edit_distance,
    }


def normalized_action_candidates(values: Sequence[Any]) -> tuple[ActionCandidate, ...]:
    output: dict[str, ActionCandidate] = {}
    for item in values:
        if isinstance(item, ActionCandidate):
            candidate = item
        else:
            action_name = getattr(item, "action_name", getattr(item, "name", ""))
            action_data = getattr(
                item,
                "action_data",
                getattr(item, "action_args", {}),
            )
            if isinstance(item, Mapping):
                action_name = item.get("action_name", item.get("name", action_name))
                action_data = item.get(
                    "action_data",
                    item.get("action_args", action_data),
                )
            if str(action_name).strip().upper() == "RESET":
                continue
            candidate = ActionCandidate(
                action_name=str(action_name),
                action_data=dict(action_data or {}),
            )
        if candidate.action_name == "RESET":
            continue
        output[candidate.key] = candidate
    return tuple(output.values())


__all__ = [
    "ALLOWED_ACTION_OPERATORS",
    "ALLOWED_PREDICATES",
    "RELATION_PREDICATES",
    "TOPOLOGY_PREDICATES",
    "AbstractEntity",
    "AbstractState",
    "ActionBinding",
    "ActionCandidate",
    "Effect",
    "Expression",
    "GoalRule",
    "GroundFact",
    "JointProgramHypothesis",
    "ObjectSchema",
    "ObservedTransition",
    "PredictionPacket",
    "ProgramFragment",
    "ProgressRule",
    "RolloutPrediction",
    "TerminalRule",
    "TransitionRule",
    "TruthValue",
    "normalized_action_candidates",
    "program_from_dict",
    "program_to_dict",
]
