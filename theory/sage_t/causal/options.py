"""Minimal option extraction and compilation into causal-program particles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from theory.sage_t.contracts import AbstractEntity, AbstractState

from .contracts import (
    ActionInterventionSpec,
    CausalProgram,
    CausalVariableSpec,
    GroundedAction,
    MechanismSpec,
    ParentRef,
    ValueDistribution,
)
from .mechanisms import MechanismContext, MechanismRegistry

OPTION_FORMAT = "sage-t12.1-minimal-causal-option-v1"
COMPILED_OPTION_FORMAT = "sage-t12.1-compiled-option-registry-v1"
MAXIMUM_OPTION_HORIZON = 32


def _finite_state_option(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    sequence = tuple(str(value) for value in parameters.get("sequence", ()))
    if not sequence:
        return ValueDistribution.deterministic(0)
    raw_phase = parents[0].mode if parents else context.current_output.mode
    try:
        phase = max(0, min(len(sequence), int(raw_phase or 0)))
    except (TypeError, ValueError):
        phase = 0
    if phase >= len(sequence):
        return ValueDistribution.deterministic(phase)
    if context.action.action_name == sequence[phase]:
        return ValueDistribution.deterministic(phase + 1)
    return ValueDistribution.deterministic(
        1 if context.action.action_name == sequence[0] else 0
    )


def _option_completion(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    raw_phase = parents[0].mode if parents else context.current_output.mode
    try:
        phase = int(raw_phase or 0)
    except (TypeError, ValueError):
        phase = 0
    return ValueDistribution.deterministic(
        phase >= max(1, int(parameters.get("length", 1)))
    )


class OptionMechanismRegistry(MechanismRegistry):
    """T12.1 extension that leaves the frozen T11 registry unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.register_symbolic("finite_state_option", _finite_state_option)
        self.register_symbolic("option_completion", _option_completion)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _entity_signature(entity: AbstractEntity) -> str:
    return _canonical(
        {
            "roles": list(entity.roles),
            "attributes": [list(item) for item in entity.attributes],
        }
    )


@dataclass(frozen=True)
class MinimalOptionStep:
    action_name: str
    static_action_data: Mapping[str, Any]
    binding_method: str = "unique_action_schema"
    structural_signature: str | None = None
    relative_offset: tuple[float, float] = (0.0, 0.0)
    expected_effect: str = "unknown"

    def __post_init__(self) -> None:
        name = str(self.action_name).strip().upper()
        if not name.startswith("ACTION"):
            raise ValueError("minimal option steps require an ACTION schema")
        if self.binding_method not in {
            "unique_action_schema",
            "structural_entity_center",
        }:
            raise ValueError("unsupported transfer-safe option binding")
        if self.binding_method == "structural_entity_center" and not self.structural_signature:
            raise ValueError("structural binding needs an entity signature")
        object.__setattr__(self, "action_name", name)
        object.__setattr__(
            self,
            "static_action_data",
            {
                str(key): value
                for key, value in dict(self.static_action_data).items()
                if str(key) not in {"x", "y"}
            },
        )
        object.__setattr__(
            self,
            "relative_offset",
            (float(self.relative_offset[0]), float(self.relative_offset[1])),
        )

    def materialize(self, state: AbstractState) -> GroundedAction:
        data = dict(self.static_action_data)
        if self.binding_method == "structural_entity_center":
            matches = [
                entity
                for entity in state.entities
                if entity.center is not None
                and _entity_signature(entity) == self.structural_signature
            ]
            if len(matches) != 1:
                raise ValueError(
                    "option structural binding is not unique in the target state"
                )
            row, column = matches[0].center or (0.0, 0.0)
            dx, dy = self.relative_offset
            data.update({"x": column + dx, "y": row + dy})
        return GroundedAction(self.action_name, data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "static_action_data": dict(self.static_action_data),
            "binding_method": self.binding_method,
            "structural_signature": self.structural_signature,
            "relative_offset": list(self.relative_offset),
            "expected_effect": self.expected_effect,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MinimalOptionStep:
        return cls(
            action_name=str(payload["action_name"]),
            static_action_data=dict(payload.get("static_action_data", {}) or {}),
            binding_method=str(payload.get("binding_method", "unique_action_schema")),
            structural_signature=(
                None
                if payload.get("structural_signature") is None
                else str(payload["structural_signature"])
            ),
            relative_offset=tuple(
                float(value) for value in payload.get("relative_offset", (0.0, 0.0))
            ),
            expected_effect=str(payload.get("expected_effect", "unknown")),
        )


@dataclass(frozen=True)
class MinimalCausalOption:
    initiation_signature: str
    initiation_exact_hash: str
    steps: tuple[MinimalOptionStep, ...]
    source_evidence_ids: tuple[str, ...]
    termination_predicate: str = "counter.levels_completed increased"
    source: str = "go_explore_first_progress"
    minimization_evaluations: int = 0
    reproduction_count: int = 2
    format_version: str = OPTION_FORMAT

    def __post_init__(self) -> None:
        if self.format_version != OPTION_FORMAT:
            raise ValueError("unsupported minimal-option format")
        if not self.initiation_signature or not self.initiation_exact_hash:
            raise ValueError("minimal option needs symbolic and exact initiation")
        if not 1 <= len(self.steps) <= MAXIMUM_OPTION_HORIZON:
            raise ValueError("minimal option needs one to 32 steps")
        if int(self.reproduction_count) < 2:
            raise ValueError("minimal option needs two exact reproductions")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(
            self,
            "source_evidence_ids",
            tuple(str(value) for value in self.source_evidence_ids),
        )

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "initiation_signature": self.initiation_signature,
            "initiation_exact_hash": self.initiation_exact_hash,
            "steps": [step.to_dict() for step in self.steps],
            "source_evidence_ids": list(self.source_evidence_ids),
            "termination_predicate": self.termination_predicate,
            "source": self.source,
            "minimization_evaluations": self.minimization_evaluations,
            "reproduction_count": self.reproduction_count,
        }

    @property
    def checksum(self) -> str:
        return _sha(self.safe_payload)

    @property
    def option_id(self) -> str:
        return "opt_" + self.checksum[:20]

    def materialize(self, state: AbstractState) -> tuple[GroundedAction, ...]:
        return tuple(step.materialize(state) for step in self.steps)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MinimalCausalOption:
        return cls(
            initiation_signature=str(payload["initiation_signature"]),
            initiation_exact_hash=str(payload["initiation_exact_hash"]),
            steps=tuple(
                MinimalOptionStep.from_dict(dict(item))
                for item in payload.get("steps", ())
            ),
            source_evidence_ids=tuple(
                str(value) for value in payload.get("source_evidence_ids", ())
            ),
            termination_predicate=str(
                payload.get(
                    "termination_predicate", "counter.levels_completed increased"
                )
            ),
            source=str(payload.get("source", "go_explore_first_progress")),
            minimization_evaluations=int(payload.get("minimization_evaluations", 0)),
            reproduction_count=int(payload.get("reproduction_count", 2)),
            format_version=str(payload.get("format_version", OPTION_FORMAT)),
        )


ReplayProgress = Callable[[tuple[GroundedAction, ...]], bool]


class MinimalOptionExtractor:
    def __init__(self, *, maximum_horizon: int = MAXIMUM_OPTION_HORIZON) -> None:
        self.maximum_horizon = max(1, min(MAXIMUM_OPTION_HORIZON, int(maximum_horizon)))

    def extract(
        self,
        *,
        initiation_state: AbstractState,
        initiation_exact_hash: str,
        actions: Sequence[GroundedAction],
        states_before: Sequence[AbstractState],
        replay_progress: ReplayProgress,
        expected_effects: Sequence[str] = (),
        source_evidence_ids: Sequence[str] = (),
    ) -> MinimalCausalOption:
        if len(actions) != len(states_before):
            raise ValueError("option extraction needs one pre-state per action")
        if not actions:
            raise ValueError("option extraction needs a successful action sequence")
        selected_actions = tuple(actions)[-self.maximum_horizon :]
        selected_states = tuple(states_before)[-self.maximum_horizon :]
        selected_effects = tuple(expected_effects)[-self.maximum_horizon :]
        if not replay_progress(selected_actions):
            raise ValueError("candidate suffix does not reproduce level progression")
        evaluations = 1
        indices = list(range(len(selected_actions)))
        granularity = 2
        while len(indices) >= 2:
            chunk_size = max(1, math.ceil(len(indices) / granularity))
            reduced = False
            for start in range(0, len(indices), chunk_size):
                candidate = indices[:start] + indices[start + chunk_size :]
                if not candidate:
                    continue
                evaluations += 1
                if replay_progress(tuple(selected_actions[index] for index in candidate)):
                    indices = candidate
                    granularity = max(2, granularity - 1)
                    reduced = True
                    break
            if reduced:
                continue
            if granularity >= len(indices):
                break
            granularity = min(len(indices), granularity * 2)
        changed = True
        while changed and len(indices) > 1:
            changed = False
            for position in range(len(indices)):
                candidate = indices[:position] + indices[position + 1 :]
                evaluations += 1
                if candidate and replay_progress(
                    tuple(selected_actions[index] for index in candidate)
                ):
                    indices = candidate
                    changed = True
                    break
        minimized_actions = tuple(selected_actions[index] for index in indices)
        minimized_states = tuple(selected_states[index] for index in indices)
        for position in range(len(minimized_actions)):
            candidate = minimized_actions[:position] + minimized_actions[position + 1 :]
            if not candidate:
                continue
            evaluations += 1
            if replay_progress(candidate):
                raise RuntimeError("option minimizer failed its one-step minimality check")
        reproduction_count = 0
        for _ in range(2):
            evaluations += 1
            reproduction_count += int(replay_progress(minimized_actions))
        if reproduction_count != 2:
            raise ValueError("minimal option failed exact reproduction")
        steps = tuple(
            self._transfer_safe_step(
                action,
                state,
                expected_effect=(
                    selected_effects[index]
                    if index < len(selected_effects)
                    else "unknown"
                ),
            )
            for index, action, state in zip(indices, minimized_actions, minimized_states)
        )
        return MinimalCausalOption(
            initiation_signature=initiation_state.signature,
            initiation_exact_hash=str(initiation_exact_hash),
            steps=steps,
            source_evidence_ids=tuple(source_evidence_ids),
            minimization_evaluations=evaluations,
            reproduction_count=reproduction_count,
        )

    @staticmethod
    def _transfer_safe_step(
        action: GroundedAction,
        state: AbstractState,
        *,
        expected_effect: str,
    ) -> MinimalOptionStep:
        data = dict(action.action_data)
        if "x" not in data and "y" not in data:
            return MinimalOptionStep(
                action_name=action.action_name,
                static_action_data=data,
                expected_effect=expected_effect,
            )
        if "x" not in data or "y" not in data:
            raise ValueError("partial absolute coordinates cannot be transferred")
        try:
            x = float(data["x"])
            y = float(data["y"])
        except (TypeError, ValueError) as exc:
            raise ValueError("option coordinates must be numeric") from exc
        candidates = [entity for entity in state.entities if entity.center is not None]
        if not candidates:
            raise ValueError("parameterized option lacks a structural entity anchor")
        nearest = min(
            candidates,
            key=lambda entity: (
                (entity.center[1] - x) ** 2 + (entity.center[0] - y) ** 2,
                entity.entity_id,
            ),
        )
        signature = _entity_signature(nearest)
        if sum(_entity_signature(entity) == signature for entity in candidates) != 1:
            raise ValueError("parameterized option anchor is structurally ambiguous")
        row, column = nearest.center or (0.0, 0.0)
        return MinimalOptionStep(
            action_name=action.action_name,
            static_action_data=data,
            binding_method="structural_entity_center",
            structural_signature=signature,
            relative_offset=(x - column, y - row),
            expected_effect=expected_effect,
        )


@dataclass(frozen=True)
class CompiledCausalOption:
    option: MinimalCausalOption
    owner_program_hashes: tuple[str, ...]
    child_program_ids: tuple[str, ...]
    format_version: str = COMPILED_OPTION_FORMAT

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format_version": self.format_version,
            "option": self.option.safe_payload,
            "owner_program_hashes": list(self.owner_program_hashes),
            "child_program_ids": list(self.child_program_ids),
        }
        payload["registry_checksum"] = _sha(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompiledCausalOption:
        if payload.get("format_version") != COMPILED_OPTION_FORMAT:
            raise ValueError("unsupported compiled-option registry")
        unsigned = dict(payload)
        checksum = str(unsigned.pop("registry_checksum", ""))
        if not checksum or checksum != _sha(unsigned):
            raise ValueError("compiled-option registry checksum mismatch")
        return cls(
            option=MinimalCausalOption.from_dict(dict(payload["option"])),
            owner_program_hashes=tuple(
                str(value) for value in payload.get("owner_program_hashes", ())
            ),
            child_program_ids=tuple(
                str(value) for value in payload.get("child_program_ids", ())
            ),
        )


class CausalOptionCompiler:
    def compile(
        self,
        option: MinimalCausalOption,
        parent_programs: Sequence[CausalProgram],
    ) -> tuple[tuple[CausalProgram, ...], CompiledCausalOption]:
        if not parent_programs:
            raise ValueError("causal option compiler needs parent programs")
        children = tuple(self._compile_child(option, parent) for parent in parent_programs)
        registry = CompiledCausalOption(
            option=option,
            owner_program_hashes=tuple(child.canonical_hash for child in children),
            child_program_ids=tuple(child.program_id for child in children),
        )
        return children, registry

    @staticmethod
    def _compile_child(
        option: MinimalCausalOption,
        parent: CausalProgram,
    ) -> CausalProgram:
        token = option.option_id
        phase_variable = f"option.{token}.phase"
        complete_variable = f"option.{token}.complete"
        length = len(option.steps)
        variables = parent.variables + (
            CausalVariableSpec(phase_variable, "option_phase", tuple(range(length + 1))),
            CausalVariableSpec(complete_variable, "boolean", (False, True)),
        )
        sequence = [step.action_name for step in option.steps]
        mechanisms = parent.mechanisms + (
            MechanismSpec(
                mechanism_id=f"{token}.phase_transition",
                output_variable=phase_variable,
                parent_variables=(ParentRef(phase_variable),),
                operator_type="finite_state_option",
                parameters={"sequence": sequence},
            ),
            MechanismSpec(
                mechanism_id=f"{token}.completion",
                output_variable=complete_variable,
                parent_variables=(ParentRef(phase_variable, "next"),),
                operator_type="option_completion",
                parameters={"length": length},
            ),
        )
        by_name = {item.action_name: item for item in parent.action_model}
        for action_name in sequence:
            by_name.setdefault(action_name, ActionInterventionSpec(action_name))
        progress_predicate = f"{complete_variable} == true"
        goal = replace(
            parent.goal,
            progress_predicates=tuple(
                dict.fromkeys((*parent.goal.progress_predicates, progress_predicate))
            ),
        )
        child_id = f"{parent.program_id}.opt.{option.checksum[:12]}"
        return CausalProgram(
            program_id=child_id[:128],
            bindings=parent.bindings,
            variables=variables,
            mechanisms=mechanisms,
            action_model=tuple(by_name[key] for key in sorted(by_name)),
            goal=goal,
            observation_model=parent.observation_model,
            description_length=(
                float(parent.description_length)
                + float(length)
                + sum(
                    0.5
                    for step in option.steps
                    if step.binding_method == "structural_entity_center"
                )
            ),
            provenance=tuple(
                dict.fromkeys(
                    (*parent.provenance, f"option:{option.option_id}:{option.checksum}")
                )
            ),
        )


class PosteriorOptionProvider:
    """Materialize an option only from posterior mass over its owner programs."""

    def __init__(
        self,
        compiled: CompiledCausalOption,
        *,
        minimum_posterior_mass: float = 0.8,
    ) -> None:
        self.compiled = compiled
        self.minimum_posterior_mass = max(
            0.0, min(1.0, float(minimum_posterior_mass))
        )

    def owner_mass(self, posterior: Any) -> float:
        owners = set(self.compiled.owner_program_hashes)
        return sum(
            particle.probability
            for particle in posterior.particles
            if particle.program.canonical_hash in owners
        )

    def materialize(
        self,
        state: AbstractState,
        posterior: Any,
    ) -> tuple[GroundedAction, ...]:
        if self.owner_mass(posterior) + 1e-12 < self.minimum_posterior_mass:
            return ()
        return self.compiled.option.materialize(state)


__all__ = [
    "COMPILED_OPTION_FORMAT",
    "MAXIMUM_OPTION_HORIZON",
    "OPTION_FORMAT",
    "CausalOptionCompiler",
    "CompiledCausalOption",
    "MinimalCausalOption",
    "MinimalOptionExtractor",
    "MinimalOptionStep",
    "OptionMechanismRegistry",
    "PosteriorOptionProvider",
]
