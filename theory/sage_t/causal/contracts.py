"""Versioned contracts for the SAGE.T11 dynamic causal-program vertical."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

FORMAT_VERSION = "sage-t-causal-program-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


def _identifier(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"invalid {label}: {value!r}")
    return normalized


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): item for key, item in value.items()})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


@dataclass(frozen=True)
class ValueDistribution:
    """Small categorical distribution used for both discrete and encoded values."""

    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        raw = {str(key): float(value) for key, value in self.probabilities.items()}
        if not raw or any(value < 0.0 or not math.isfinite(value) for value in raw.values()):
            raise ValueError("a value distribution needs finite non-negative mass")
        total = sum(raw.values())
        if total <= 0.0:
            raise ValueError("a value distribution needs positive mass")
        normalized = {key: value / total for key, value in sorted(raw.items())}
        object.__setattr__(self, "probabilities", MappingProxyType(normalized))

    @classmethod
    def deterministic(cls, value: Any) -> ValueDistribution:
        return cls({encode_value(value): 1.0})

    @classmethod
    def unknown(cls) -> ValueDistribution:
        return cls({"__unknown__": 1.0})

    @property
    def mode_key(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def mode(self) -> Any:
        return decode_value(self.mode_key)

    def probability_of(self, value: Any) -> float:
        return float(self.probabilities.get(encode_value(value), 0.0))

    def total_variation(self, other: ValueDistribution) -> float:
        keys = set(self.probabilities) | set(other.probabilities)
        return 0.5 * sum(
            abs(float(self.probabilities.get(key, 0.0)) - float(other.probabilities.get(key, 0.0)))
            for key in keys
        )

    def to_dict(self) -> dict[str, Any]:
        return {"probabilities": dict(self.probabilities)}


def encode_value(value: Any) -> str:
    return _canonical_json(value)


def decode_value(value: str) -> Any:
    if value == "__unknown__":
        return None
    return json.loads(value)


@dataclass(frozen=True)
class CausalVariableSpec:
    variable_id: str
    variable_type: str
    domain: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_id", _identifier(self.variable_id, label="variable id"))
        object.__setattr__(self, "variable_type", _identifier(self.variable_type, label="variable type"))
        object.__setattr__(self, "domain", tuple(self.domain))


@dataclass(frozen=True)
class ParentRef:
    variable_id: str
    time_slice: str = "current"

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_id", _identifier(self.variable_id, label="parent variable"))
        normalized = str(self.time_slice).lower()
        if normalized not in {"current", "next"}:
            raise ValueError("parent time_slice must be current or next")
        object.__setattr__(self, "time_slice", normalized)


@dataclass(frozen=True)
class BindingSpec:
    role_to_entity: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role_to_entity",
            MappingProxyType(
                {
                    _identifier(role, label="binding role"): str(entity)
                    for role, entity in sorted(self.role_to_entity.items())
                }
            ),
        )


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    output_variable: str
    parent_variables: tuple[ParentRef, ...]
    operator_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    neural_module_id: str | None = None
    symbolic_fallback: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanism_id", _identifier(self.mechanism_id, label="mechanism id"))
        object.__setattr__(self, "output_variable", _identifier(self.output_variable, label="output variable"))
        object.__setattr__(self, "parent_variables", tuple(self.parent_variables))
        object.__setattr__(self, "operator_type", _identifier(self.operator_type, label="operator type"))
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))
        if self.neural_module_id is not None:
            object.__setattr__(self, "neural_module_id", _identifier(self.neural_module_id, label="neural module id"))
        if self.symbolic_fallback is not None:
            object.__setattr__(self, "symbolic_fallback", _identifier(self.symbolic_fallback, label="symbolic fallback"))


@dataclass(frozen=True)
class ActionInterventionSpec:
    action_name: str
    intervention_type: str = "do_action"
    target_variable: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_name", _identifier(self.action_name, label="action name"))
        object.__setattr__(self, "intervention_type", _identifier(self.intervention_type, label="intervention type"))
        if self.target_variable is not None:
            object.__setattr__(self, "target_variable", _identifier(self.target_variable, label="target variable"))
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))


@dataclass(frozen=True)
class GoalSpec:
    success_predicate: str
    progress_predicates: tuple[str, ...] = ()
    failure_predicate: str | None = None

    def __post_init__(self) -> None:
        if not str(self.success_predicate).strip():
            raise ValueError("a causal program needs a success predicate")
        object.__setattr__(self, "success_predicate", str(self.success_predicate).strip())
        object.__setattr__(self, "progress_predicates", tuple(str(item).strip() for item in self.progress_predicates if str(item).strip()))
        if self.failure_predicate is not None:
            object.__setattr__(self, "failure_predicate", str(self.failure_predicate).strip() or None)


@dataclass(frozen=True)
class ObservationModelSpec:
    channels: tuple[str, ...] = ("variables", "terminal", "goal", "progress")
    channel_weights: Mapping[str, float] = field(default_factory=dict)
    noise_floor: float = 0.02
    neural_module_id: str | None = None

    def __post_init__(self) -> None:
        channels = tuple(dict.fromkeys(_identifier(item, label="observation channel") for item in self.channels))
        if not channels:
            raise ValueError("an observation model needs at least one channel")
        if not 0.0 < float(self.noise_floor) < 0.5:
            raise ValueError("noise_floor must be in (0, 0.5)")
        weights = {str(key): float(value) for key, value in self.channel_weights.items()}
        if any(value < 0.0 or not math.isfinite(value) for value in weights.values()):
            raise ValueError("observation weights must be finite and non-negative")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "channel_weights", MappingProxyType(weights))
        if self.neural_module_id is not None:
            object.__setattr__(self, "neural_module_id", _identifier(self.neural_module_id, label="observation module id"))


@dataclass(frozen=True)
class CausalProgram:
    program_id: str
    bindings: BindingSpec
    variables: tuple[CausalVariableSpec, ...]
    mechanisms: tuple[MechanismSpec, ...]
    action_model: tuple[ActionInterventionSpec, ...]
    goal: GoalSpec
    observation_model: ObservationModelSpec = field(default_factory=ObservationModelSpec)
    description_length: float = 0.0
    provenance: tuple[str, ...] = ()
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", _identifier(self.program_id, label="program id"))
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        object.__setattr__(self, "action_model", tuple(self.action_model))
        object.__setattr__(self, "provenance", tuple(sorted({str(item)[:200] for item in self.provenance})))
        if self.format_version != FORMAT_VERSION:
            raise ValueError(f"unsupported causal-program format: {self.format_version}")
        if not self.variables or not self.mechanisms or not self.action_model:
            raise ValueError("a causal program needs variables, mechanisms, and actions")
        if not math.isfinite(float(self.description_length)) or float(self.description_length) < 0.0:
            raise ValueError("description_length must be finite and non-negative")

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "bindings": dict(self.bindings.role_to_entity),
            "variables": [_jsonable(asdict(item)) for item in sorted(self.variables, key=lambda item: item.variable_id)],
            "mechanisms": [_mechanism_payload(item) for item in sorted(self.mechanisms, key=lambda item: item.output_variable)],
            "action_model": [
                {
                    "action_name": item.action_name,
                    "intervention_type": item.intervention_type,
                    "target_variable": item.target_variable,
                    "parameters": _jsonable(item.parameters),
                }
                for item in sorted(self.action_model, key=lambda item: item.action_name)
            ],
            "goal": _jsonable(asdict(self.goal)),
            "observation_model": {
                "channels": list(self.observation_model.channels),
                "channel_weights": _jsonable(self.observation_model.channel_weights),
                "noise_floor": float(self.observation_model.noise_floor),
                "neural_module_id": self.observation_model.neural_module_id,
            },
            "description_length": float(self.description_length),
        }

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_payload).encode("utf-8")).hexdigest()

    @property
    def structural_family(self) -> tuple[tuple[str, ...], str]:
        return (
            tuple(sorted(mechanism.operator_type for mechanism in self.mechanisms)),
            self.goal.success_predicate,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.canonical_payload)
        payload.update({"program_id": self.program_id, "provenance": list(self.provenance)})
        return payload


def _mechanism_payload(mechanism: MechanismSpec) -> dict[str, Any]:
    return {
        "mechanism_id": mechanism.mechanism_id,
        "output_variable": mechanism.output_variable,
        "parent_variables": [asdict(parent) for parent in mechanism.parent_variables],
        "operator_type": mechanism.operator_type,
        "parameters": _jsonable(mechanism.parameters),
        "neural_module_id": mechanism.neural_module_id,
        "symbolic_fallback": mechanism.symbolic_fallback,
    }


@dataclass(frozen=True)
class GroundedAction:
    action_name: str
    action_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_name", _identifier(self.action_name, label="grounded action"))
        object.__setattr__(self, "action_data", _frozen_mapping(self.action_data))

    @property
    def key(self) -> str:
        return f"{self.action_name}:{_canonical_json(self.action_data)}"


@dataclass(frozen=True)
class ActionProgram:
    actions: tuple[GroundedAction, ...]
    source: str = "generic"

    def __post_init__(self) -> None:
        if not self.actions or len(self.actions) > 8:
            raise ValueError("an action program needs one to eight actions")
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "source", _identifier(self.source, label="action-program source"))

    @property
    def key(self) -> str:
        return "->".join(action.key for action in self.actions)


@dataclass(frozen=True)
class CausalState:
    variables: Mapping[str, ValueDistribution]
    entities: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    observation_hash: str = ""
    confidence: float = 1.0
    abstract_signature: str = ""

    def __post_init__(self) -> None:
        values = {_identifier(key, label="state variable"): value for key, value in self.variables.items()}
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("state confidence must be in [0, 1]")
        object.__setattr__(self, "variables", MappingProxyType(dict(sorted(values.items()))))
        object.__setattr__(self, "entities", tuple(str(item) for item in self.entities))
        object.__setattr__(self, "relations", tuple(str(item) for item in self.relations))
        if not self.abstract_signature:
            payload = {key: dict(value.probabilities) for key, value in sorted(values.items())}
            object.__setattr__(self, "abstract_signature", hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest())

    def value(self, variable_id: str) -> ValueDistribution:
        return self.variables.get(variable_id, ValueDistribution.unknown())


@dataclass(frozen=True)
class StructuredDelta:
    variable_changes: Mapping[str, ValueDistribution] = field(default_factory=dict)
    affected_objects: tuple[str, ...] = ()
    relation_changes: tuple[str, ...] = ()
    patch_digest: str = ""
    progress: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_changes", MappingProxyType(dict(sorted(self.variable_changes.items()))))
        object.__setattr__(self, "affected_objects", tuple(sorted({str(item) for item in self.affected_objects})))
        object.__setattr__(self, "relation_changes", tuple(sorted({str(item) for item in self.relation_changes})))
        if not math.isfinite(float(self.progress)):
            raise ValueError("delta progress must be finite")


@dataclass(frozen=True)
class TransitionEvidence:
    evidence_id: str
    state_before: CausalState
    action: GroundedAction
    state_after: CausalState
    observed_delta: StructuredDelta
    terminal: bool = False
    success: bool | None = None
    level_change: int = 0
    prefix_hash: str = ""
    game_id: str = ""
    context_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, label="evidence id"))
        object.__setattr__(self, "game_id", str(self.game_id))
        if not self.context_id:
            object.__setattr__(self, "context_id", self.state_before.abstract_signature)


@dataclass(frozen=True)
class Intervention:
    variable_id: str
    value: ValueDistribution

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_id", _identifier(self.variable_id, label="intervention variable"))


@dataclass(frozen=True)
class PredictionDistribution:
    state_after: CausalState
    delta: StructuredDelta
    terminal_probability: float
    goal_probability: float
    progress_probability: float
    known_variables: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for value in (self.terminal_probability, self.goal_probability, self.progress_probability):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("prediction probabilities must be in [0, 1]")

    @property
    def structured_signature(self) -> tuple[Any, ...]:
        return (
            tuple((key, value.mode_key) for key, value in self.delta.variable_changes.items()),
            self.delta.affected_objects,
            self.delta.relation_changes,
            round(float(self.progress_probability), 6),
            round(float(self.terminal_probability), 6),
            round(float(self.goal_probability), 6),
        )


@dataclass(frozen=True)
class PredictedTrace:
    action_program: ActionProgram
    predictions: tuple[PredictionDistribution, ...]

    @property
    def final_prediction(self) -> PredictionDistribution:
        return self.predictions[-1]


@dataclass(frozen=True)
class InterventionBranch:
    action: GroundedAction
    predicted_signatures: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicted_signatures", _frozen_mapping(self.predicted_signatures))


@dataclass(frozen=True)
class InterventionBundle:
    prefix: ActionProgram
    prefix_hash: str
    branches: tuple[InterventionBranch, ...]

    def __post_init__(self) -> None:
        if len(self.branches) < 2:
            raise ValueError("an intervention bundle needs at least two branches")
        if not self.prefix_hash:
            raise ValueError("an intervention bundle needs an exact prefix hash")


def causal_program_from_dict(payload: Mapping[str, Any]) -> CausalProgram:
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported causal-program payload")
    observation = dict(payload.get("observation_model", {}) or {})
    return CausalProgram(
        program_id=str(payload["program_id"]),
        bindings=BindingSpec(dict(payload.get("bindings", {}) or {})),
        variables=tuple(CausalVariableSpec(**dict(item)) for item in payload.get("variables", ())),
        mechanisms=tuple(
            MechanismSpec(
                mechanism_id=str(item["mechanism_id"]),
                output_variable=str(item["output_variable"]),
                parent_variables=tuple(ParentRef(**dict(parent)) for parent in item.get("parent_variables", ())),
                operator_type=str(item["operator_type"]),
                parameters=dict(item.get("parameters", {}) or {}),
                neural_module_id=item.get("neural_module_id"),
                symbolic_fallback=item.get("symbolic_fallback"),
            )
            for item in payload.get("mechanisms", ())
        ),
        action_model=tuple(ActionInterventionSpec(**dict(item)) for item in payload.get("action_model", ())),
        goal=GoalSpec(**dict(payload["goal"])),
        observation_model=ObservationModelSpec(
            channels=tuple(observation.get("channels", ("variables", "terminal", "goal", "progress"))),
            channel_weights=dict(observation.get("channel_weights", {}) or {}),
            noise_floor=float(observation.get("noise_floor", 0.02)),
            neural_module_id=observation.get("neural_module_id"),
        ),
        description_length=float(payload.get("description_length", 0.0)),
        provenance=tuple(payload.get("provenance", ())),
        format_version=str(payload["format_version"]),
    )


def causal_program_to_json(program: CausalProgram) -> str:
    return _canonical_json(program.to_dict())


def causal_program_from_json(payload: str) -> CausalProgram:
    return causal_program_from_dict(json.loads(payload))


def causal_state_to_dict(state: CausalState) -> dict[str, Any]:
    return {
        "variables": {
            key: value.to_dict() for key, value in state.variables.items()
        },
        "entities": list(state.entities),
        "relations": list(state.relations),
        "observation_hash": state.observation_hash,
        "confidence": float(state.confidence),
        "abstract_signature": state.abstract_signature,
    }


def causal_state_from_dict(payload: Mapping[str, Any]) -> CausalState:
    return CausalState(
        variables={
            str(key): ValueDistribution(dict(value["probabilities"]))
            for key, value in dict(payload.get("variables", {}) or {}).items()
        },
        entities=tuple(payload.get("entities", ())),
        relations=tuple(payload.get("relations", ())),
        observation_hash=str(payload.get("observation_hash", "")),
        confidence=float(payload.get("confidence", 1.0)),
        abstract_signature=str(payload.get("abstract_signature", "")),
    )


def transition_evidence_to_dict(evidence: TransitionEvidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "state_before": causal_state_to_dict(evidence.state_before),
        "action": {
            "action_name": evidence.action.action_name,
            "action_data": _jsonable(evidence.action.action_data),
        },
        "state_after": causal_state_to_dict(evidence.state_after),
        "observed_delta": {
            "variable_changes": {
                key: value.to_dict()
                for key, value in evidence.observed_delta.variable_changes.items()
            },
            "affected_objects": list(evidence.observed_delta.affected_objects),
            "relation_changes": list(evidence.observed_delta.relation_changes),
            "patch_digest": evidence.observed_delta.patch_digest,
            "progress": float(evidence.observed_delta.progress),
        },
        "terminal": evidence.terminal,
        "success": evidence.success,
        "level_change": evidence.level_change,
        "prefix_hash": evidence.prefix_hash,
        "game_id": evidence.game_id,
        "context_id": evidence.context_id,
    }


def transition_evidence_from_dict(
    payload: Mapping[str, Any],
) -> TransitionEvidence:
    delta = dict(payload.get("observed_delta", {}) or {})
    return TransitionEvidence(
        evidence_id=str(payload["evidence_id"]),
        state_before=causal_state_from_dict(payload["state_before"]),
        action=GroundedAction(
            str(payload["action"]["action_name"]),
            dict(payload["action"].get("action_data", {}) or {}),
        ),
        state_after=causal_state_from_dict(payload["state_after"]),
        observed_delta=StructuredDelta(
            variable_changes={
                str(key): ValueDistribution(dict(value["probabilities"]))
                for key, value in dict(
                    delta.get("variable_changes", {}) or {}
                ).items()
            },
            affected_objects=tuple(delta.get("affected_objects", ())),
            relation_changes=tuple(delta.get("relation_changes", ())),
            patch_digest=str(delta.get("patch_digest", "")),
            progress=float(delta.get("progress", 0.0)),
        ),
        terminal=bool(payload.get("terminal", False)),
        success=payload.get("success"),
        level_change=int(payload.get("level_change", 0)),
        prefix_hash=str(payload.get("prefix_hash", "")),
        game_id=str(payload.get("game_id", "")),
        context_id=str(payload.get("context_id", "")),
    )


__all__ = [
    "FORMAT_VERSION",
    "ActionInterventionSpec",
    "ActionProgram",
    "BindingSpec",
    "CausalProgram",
    "CausalState",
    "CausalVariableSpec",
    "GoalSpec",
    "GroundedAction",
    "Intervention",
    "InterventionBranch",
    "InterventionBundle",
    "MechanismSpec",
    "ObservationModelSpec",
    "ParentRef",
    "PredictedTrace",
    "PredictionDistribution",
    "StructuredDelta",
    "TransitionEvidence",
    "ValueDistribution",
    "causal_program_from_dict",
    "causal_program_from_json",
    "causal_program_to_json",
    "causal_state_from_dict",
    "causal_state_to_dict",
    "decode_value",
    "encode_value",
    "transition_evidence_from_dict",
    "transition_evidence_to_dict",
]
