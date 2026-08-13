"""Local initiation/effect contracts for causal option particles.

T12.4a.4c deliberately lives beside the original T12.1 option contract.  This
keeps the sealed parent implementation immutable while making applicability a
first-class part of each complete causal-program particle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from theory.sage_t.contracts import AbstractState

from .contracts import (
    CausalProgram,
    CausalState,
    CausalVariableSpec,
    MechanismSpec,
    ParentRef,
    ValueDistribution,
)
from .mechanisms import MechanismContext
from .options import MinimalCausalOption, OptionMechanismRegistry

OPTION_CONTRACT_FORMAT = "sage-t12.4a.4c-option-contract-v1"
OPTION_CONTRACT_REGISTRY_FORMAT = "sage-t12.4a.4c-option-contract-registry-v1"
_SAFE = re.compile(r"[^a-z0-9_]+")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _feature_value(descriptor: Mapping[str, Any], family: str, key: str) -> int:
    mechanism = dict(descriptor.get("mechanism", {}))
    values = dict(mechanism.get(family, {}))
    return int(values.get(key, 0))


@dataclass(frozen=True)
class InitiationAtom:
    """One identity-free aggregate predicate used by an initiation particle."""

    family: str
    key: str
    expected_value: int

    def __post_init__(self) -> None:
        if self.family not in {"role_counts", "predicate_counts"}:
            raise ValueError("option initiation atoms require a local typed family")
        if not self.key or any(
            forbidden in self.key.lower()
            for forbidden in ("hash", "level", "pixel", "coord", "game_id")
        ):
            raise ValueError("option initiation atom contains a forbidden field")
        object.__setattr__(self, "expected_value", int(self.expected_value))

    @property
    def feature_id(self) -> str:
        raw = _SAFE.sub("_", f"{self.family}_{self.key}".lower()).strip("_")
        return f"context.{raw[:72]}"

    def observed_value(self, descriptor: Mapping[str, Any]) -> int:
        return _feature_value(descriptor, self.family, self.key)

    def matches(self, descriptor: Mapping[str, Any]) -> bool:
        return self.observed_value(descriptor) == self.expected_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_value": self.expected_value,
            "family": self.family,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InitiationAtom:
        return cls(
            family=str(payload["family"]),
            key=str(payload["key"]),
            expected_value=int(payload["expected_value"]),
        )


@dataclass(frozen=True)
class InitiationSpec:
    """A sparse guard kept as one explicit rival posterior hypothesis."""

    atoms: tuple[InitiationAtom, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(
            self, "evidence_ids", tuple(str(item) for item in self.evidence_ids)
        )
        if len(self.atoms) != 1:
            raise ValueError("T12.4a.4c initiation particles contain exactly one atom")
        if len(self.evidence_ids) < 2:
            raise ValueError("initiation spec needs two independent evidence lineages")

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def checksum(self) -> str:
        return _checksum(self.safe_payload)

    @property
    def spec_id(self) -> str:
        return f"guard_{self.checksum[:16]}"

    def matches(self, descriptor: Mapping[str, Any]) -> bool:
        return all(atom.matches(descriptor) for atom in self.atoms)

    def to_dict(self) -> dict[str, Any]:
        return {**self.safe_payload, "spec_id": self.spec_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InitiationSpec:
        spec = cls(
            atoms=tuple(
                InitiationAtom.from_dict(dict(item))
                for item in payload.get("atoms", ())
            ),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
        )
        if payload.get("spec_id") not in {None, spec.spec_id}:
            raise ValueError("initiation spec checksum mismatch")
        return spec


@dataclass(frozen=True)
class EffectAtom:
    family: str
    key: str
    expected_delta: int

    def __post_init__(self) -> None:
        if self.family not in {"role_counts", "predicate_counts"}:
            raise ValueError("option effect atoms require a local typed family")
        if not self.key:
            raise ValueError("option effect atom needs a key")
        object.__setattr__(self, "expected_delta", int(self.expected_delta))
        if self.expected_delta == 0:
            raise ValueError("option effect atoms must describe a change")

    def observed_delta(self, step: Mapping[str, Any]) -> int:
        mechanism = dict(dict(step.get("delta", {})).get("mechanism", {}))
        item = dict(dict(mechanism.get(self.family, {})).get(self.key, {}))
        return int(item.get("after", 0)) - int(item.get("before", 0))

    def matches(self, step: Mapping[str, Any]) -> bool:
        return self.observed_delta(step) == self.expected_delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_delta": self.expected_delta,
            "family": self.family,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EffectAtom:
        return cls(
            family=str(payload["family"]),
            key=str(payload["key"]),
            expected_delta=int(payload["expected_delta"]),
        )


@dataclass(frozen=True)
class StepEffectContract:
    position: int
    action_name: str
    atoms: tuple[EffectAtom, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", int(self.position))
        object.__setattr__(self, "action_name", str(self.action_name).upper())
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if self.position < 0 or not self.action_name.startswith("ACTION"):
            raise ValueError("invalid option effect-contract position/action")
        if not 1 <= len(self.atoms) <= 3:
            raise ValueError("each option step needs one to three effect atoms")

    @property
    def checksum(self) -> str:
        return _checksum(self.to_dict())

    def matches(self, step: Mapping[str, Any]) -> bool:
        return (
            int(step.get("position", -1)) == self.position
            and str(step.get("action_name", "")).upper() == self.action_name
            and all(atom.matches(step) for atom in self.atoms)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StepEffectContract:
        return cls(
            position=int(payload["position"]),
            action_name=str(payload["action_name"]),
            atoms=tuple(
                EffectAtom.from_dict(dict(item)) for item in payload.get("atoms", ())
            ),
        )


def effect_trace_matches(
    contracts: Sequence[StepEffectContract],
    trace: Sequence[Mapping[str, Any]],
) -> bool:
    if len(contracts) != len(trace):
        return False
    return all(contract.matches(step) for contract, step in zip(contracts, trace))


@dataclass(frozen=True)
class ContractedCausalOption:
    option: MinimalCausalOption
    initiation_specs: tuple[InitiationSpec, ...]
    effect_contracts: tuple[StepEffectContract, ...]
    parent_program_hashes: tuple[str, ...]
    owner_program_hashes: tuple[str, ...]
    owner_parent_hashes: tuple[tuple[str, str], ...]
    owner_spec_ids: tuple[tuple[str, str], ...]
    child_program_ids: tuple[str, ...]
    format_version: str = OPTION_CONTRACT_REGISTRY_FORMAT

    def __post_init__(self) -> None:
        if self.format_version != OPTION_CONTRACT_REGISTRY_FORMAT:
            raise ValueError("unsupported contracted-option registry")
        object.__setattr__(self, "initiation_specs", tuple(self.initiation_specs))
        object.__setattr__(self, "effect_contracts", tuple(self.effect_contracts))
        object.__setattr__(
            self, "parent_program_hashes", tuple(self.parent_program_hashes)
        )
        object.__setattr__(self, "owner_program_hashes", tuple(self.owner_program_hashes))
        object.__setattr__(self, "owner_parent_hashes", tuple(self.owner_parent_hashes))
        object.__setattr__(self, "owner_spec_ids", tuple(self.owner_spec_ids))
        object.__setattr__(self, "child_program_ids", tuple(self.child_program_ids))
        if len(self.effect_contracts) != len(self.option.steps):
            raise ValueError("contracted option needs one effect contract per step")
        if not (
            len(self.owner_program_hashes)
            == len(self.owner_parent_hashes)
            == len(self.owner_spec_ids)
        ):
            raise ValueError("contracted option owner/spec mapping is incomplete")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "child_program_ids": list(self.child_program_ids),
            "effect_contracts": [item.to_dict() for item in self.effect_contracts],
            "format_version": self.format_version,
            "initiation_specs": [item.to_dict() for item in self.initiation_specs],
            "option": self.option.safe_payload,
            "owner_parent_hashes": [list(item) for item in self.owner_parent_hashes],
            "owner_program_hashes": list(self.owner_program_hashes),
            "owner_spec_ids": [list(item) for item in self.owner_spec_ids],
            "parent_program_hashes": list(self.parent_program_hashes),
        }
        payload["registry_checksum"] = _checksum(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContractedCausalOption:
        unsigned = dict(payload)
        checksum = str(unsigned.pop("registry_checksum", ""))
        if not checksum or checksum != _checksum(unsigned):
            raise ValueError("contracted-option registry checksum mismatch")
        return cls(
            option=MinimalCausalOption.from_dict(dict(payload["option"])),
            initiation_specs=tuple(
                InitiationSpec.from_dict(dict(item))
                for item in payload.get("initiation_specs", ())
            ),
            effect_contracts=tuple(
                StepEffectContract.from_dict(dict(item))
                for item in payload.get("effect_contracts", ())
            ),
            parent_program_hashes=tuple(
                str(item) for item in payload.get("parent_program_hashes", ())
            ),
            owner_program_hashes=tuple(
                str(item) for item in payload.get("owner_program_hashes", ())
            ),
            owner_parent_hashes=tuple(
                (str(item[0]), str(item[1]))
                for item in payload.get("owner_parent_hashes", ())
            ),
            owner_spec_ids=tuple(
                (str(item[0]), str(item[1]))
                for item in payload.get("owner_spec_ids", ())
            ),
            child_program_ids=tuple(
                str(item) for item in payload.get("child_program_ids", ())
            ),
            format_version=str(payload.get("format_version", "")),
        )


def _range_guard(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    del context
    observed = parents[0].mode if parents else None
    try:
        value = int(observed)
    except (TypeError, ValueError):
        return ValueDistribution.deterministic(False)
    return ValueDistribution.deterministic(
        int(parameters["minimum"]) <= value <= int(parameters["maximum"])
    )


def _guarded_finite_state_option(
    parents: Sequence[ValueDistribution],
    parameters: Mapping[str, Any],
    context: MechanismContext,
) -> ValueDistribution:
    sequence = tuple(str(value) for value in parameters.get("sequence", ()))
    raw_phase = parents[0].mode if parents else 0
    applicable = bool(parents[1].mode) if len(parents) > 1 else False
    try:
        phase = max(0, min(len(sequence), int(raw_phase or 0)))
    except (TypeError, ValueError):
        phase = 0
    if not applicable or not sequence:
        return ValueDistribution.deterministic(0)
    if phase >= len(sequence):
        return ValueDistribution.deterministic(phase)
    if context.action.action_name == sequence[phase]:
        return ValueDistribution.deterministic(phase + 1)
    return ValueDistribution.deterministic(
        1 if context.action.action_name == sequence[0] else 0
    )


class ContractOptionMechanismRegistry(OptionMechanismRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.register_symbolic("range_guard", _range_guard)
        self.register_symbolic(
            "guarded_finite_state_option", _guarded_finite_state_option
        )


class ContractOptionCompiler:
    """Create one complete program for every parent x initiation hypothesis."""

    def compile(
        self,
        *,
        option: MinimalCausalOption,
        initiation_specs: Sequence[InitiationSpec],
        effect_contracts: Sequence[StepEffectContract],
        parent_programs: Sequence[CausalProgram],
    ) -> tuple[tuple[CausalProgram, ...], ContractedCausalOption]:
        specs = tuple(initiation_specs)
        effects = tuple(effect_contracts)
        parents = tuple(parent_programs)
        if not specs or not parents:
            raise ValueError("contract compiler needs guards and parent programs")
        if len(effects) != len(option.steps):
            raise ValueError("contract compiler needs one effect contract per step")
        children: list[CausalProgram] = []
        owner_parent_hashes: list[tuple[str, str]] = []
        owner_spec_ids: list[tuple[str, str]] = []
        for parent in parents:
            for spec in specs:
                child = self._compile_child(option, spec, effects, parent)
                children.append(child)
                owner_parent_hashes.append(
                    (child.canonical_hash, parent.canonical_hash)
                )
                owner_spec_ids.append((child.canonical_hash, spec.spec_id))
        registry = ContractedCausalOption(
            option=option,
            initiation_specs=specs,
            effect_contracts=effects,
            parent_program_hashes=tuple(item.canonical_hash for item in parents),
            owner_program_hashes=tuple(item.canonical_hash for item in children),
            owner_parent_hashes=tuple(owner_parent_hashes),
            owner_spec_ids=tuple(owner_spec_ids),
            child_program_ids=tuple(item.program_id for item in children),
        )
        return tuple(children), registry

    @staticmethod
    def _compile_child(
        option: MinimalCausalOption,
        spec: InitiationSpec,
        effects: Sequence[StepEffectContract],
        parent: CausalProgram,
    ) -> CausalProgram:
        token = option.option_id
        phase_variable = f"option.{token}.phase"
        phase = next(
            (item for item in parent.mechanisms if item.output_variable == phase_variable),
            None,
        )
        if phase is None or phase.operator_type != "finite_state_option":
            raise ValueError("parent program lacks the sealed option phase mechanism")
        atom = spec.atoms[0]
        feature_variable = atom.feature_id
        applicable_variable = f"option.{token}.{spec.spec_id}.applicable"
        if any(item.variable_id == feature_variable for item in parent.variables):
            raise ValueError("parent already declares the contract feature variable")
        variables = parent.variables + (
            CausalVariableSpec(feature_variable, "count"),
            CausalVariableSpec(applicable_variable, "boolean", (False, True)),
        )
        effect_payload = [item.to_dict() for item in effects]
        mechanisms = []
        for mechanism in parent.mechanisms:
            if mechanism.output_variable != phase_variable:
                mechanisms.append(mechanism)
                continue
            mechanisms.append(
                MechanismSpec(
                    mechanism_id=f"{token}.{spec.spec_id}.phase",
                    output_variable=phase_variable,
                    parent_variables=(
                        ParentRef(phase_variable),
                        ParentRef(applicable_variable, "next"),
                    ),
                    operator_type="guarded_finite_state_option",
                    parameters={
                        "effect_contract_checksum": _checksum(effect_payload),
                        "effect_contracts": effect_payload,
                        "initiation_spec": spec.to_dict(),
                        "sequence": [step.action_name for step in option.steps],
                    },
                )
            )
        mechanisms.extend(
            (
                MechanismSpec(
                    mechanism_id=f"{token}.{spec.spec_id}.feature",
                    output_variable=feature_variable,
                    parent_variables=(ParentRef(feature_variable),),
                    operator_type="identity",
                ),
                MechanismSpec(
                    mechanism_id=f"{token}.{spec.spec_id}.guard",
                    output_variable=applicable_variable,
                    parent_variables=(ParentRef(feature_variable, "next"),),
                    operator_type="range_guard",
                    parameters={
                        "family": atom.family,
                        "key": atom.key,
                        "maximum": atom.expected_value,
                        "minimum": atom.expected_value,
                    },
                ),
            )
        )
        suffix = f".contract.{spec.checksum[:10]}"
        return CausalProgram(
            program_id=f"{parent.program_id}{suffix}"[:128],
            bindings=parent.bindings,
            variables=variables,
            mechanisms=tuple(mechanisms),
            action_model=parent.action_model,
            goal=parent.goal,
            observation_model=parent.observation_model,
            description_length=(
                float(parent.description_length)
                + 1.0
                + 0.1 * sum(len(item.atoms) for item in effects)
            ),
            provenance=tuple(
                dict.fromkeys(
                    (
                        *parent.provenance,
                        f"option_contract:{spec.spec_id}:{spec.checksum}",
                    )
                )
            ),
        )


class ContractedOptionProvider:
    """Expose the option only when applicable posterior owner mass is high."""

    def __init__(
        self,
        registry: ContractedCausalOption,
        *,
        minimum_applicable_mass: float = 0.8,
    ) -> None:
        self.registry = registry
        self.minimum_applicable_mass = max(
            0.0, min(1.0, float(minimum_applicable_mass))
        )
        self._specs = {item.spec_id: item for item in registry.initiation_specs}
        self._owners = dict(registry.owner_spec_ids)

    def applicable_mass(
        self,
        descriptor: Mapping[str, Any],
        posterior: Any,
    ) -> float:
        return sum(
            particle.probability
            for particle in posterior.particles
            if particle.program.canonical_hash in self._owners
            and self._specs[self._owners[particle.program.canonical_hash]].matches(
                descriptor
            )
        )

    def materialize(
        self,
        descriptor: Mapping[str, Any],
        state: AbstractState,
        posterior: Any,
    ) -> tuple[Any, ...]:
        if (
            self.applicable_mass(descriptor, posterior) + 1e-12
            < self.minimum_applicable_mass
        ):
            return ()
        return self.registry.option.materialize(state)


def causal_state_for_contract(
    program: CausalProgram,
    spec: InitiationSpec,
    descriptor: Mapping[str, Any],
) -> CausalState:
    values = {}
    for variable in program.variables:
        default = variable.domain[0] if variable.domain else None
        values[variable.variable_id] = ValueDistribution.deterministic(default)
    atom = spec.atoms[0]
    values[atom.feature_id] = ValueDistribution.deterministic(
        atom.observed_value(descriptor)
    )
    return CausalState(variables=values)


__all__ = [
    "ContractOptionCompiler",
    "ContractOptionMechanismRegistry",
    "ContractedCausalOption",
    "ContractedOptionProvider",
    "EffectAtom",
    "InitiationAtom",
    "InitiationSpec",
    "OPTION_CONTRACT_FORMAT",
    "OPTION_CONTRACT_REGISTRY_FORMAT",
    "StepEffectContract",
    "causal_state_for_contract",
    "effect_trace_matches",
]
