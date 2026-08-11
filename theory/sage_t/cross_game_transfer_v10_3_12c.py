"""Factor-isolated cross-game transfer machinery for SAGE.T10.3.12c.

The module deliberately separates transferable descriptions from reset-local
grounding.  No game identifier, coordinate, action argument, observed target
identity, or active support is serialised in a transfer program.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import AbstractState, ActionCandidate, normalized_action_candidates
from .factorial_invariants_v10_3_12b import FactorRegistry, assert_transfer_safe
from .progress_witness_v10 import GroundedAction, SearchConfig, chain_successor_macro
from .relational_program_v10_3_12 import boundary_distance

FORMAT_VERSION = "sage-t10.3.12c-cross-game-factor-program-v1"
REGISTRY_FORMAT_VERSION = "sage-t10.3.12c-cross-game-factor-registry-v1"

CONTEXTS = ("repeat_context", "path_context")
FACTORS = ("operator", "role_binding", "transition", "termination")
ARMS = (
    "factorized_source",
    "generic_source_free",
    "operator_ablation",
    "role_binding_ablation",
    "transition_ablation",
    "termination_ablation",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def verify_signed(payload: Mapping[str, Any], checksum_field: str) -> None:
    expected = str(payload.get(checksum_field, ""))
    core = {key: value for key, value in payload.items() if key != checksum_field}
    if not expected or sha256_payload(core) != expected:
        raise ValueError(f"invalid {checksum_field}")


@dataclass(frozen=True)
class CrossGameFactorProgram:
    """One safe factor bundle or exactly-one-factor ablation."""

    arm: str
    context: str
    operator: str
    role_binding: str
    transition: str
    termination: str
    safety_horizon: int
    source_kind: str
    ablated_factor: str | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unsupported T10.3.12c arm: {self.arm}")
        if self.context not in CONTEXTS:
            raise ValueError(f"unsupported transfer context: {self.context}")
        if self.ablated_factor is not None and self.ablated_factor not in FACTORS:
            raise ValueError(f"unsupported ablated factor: {self.ablated_factor}")
        if not 1 <= int(self.safety_horizon) <= 16:
            raise ValueError("safety horizon must be in [1, 16]")
        assert_transfer_safe(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "arm": self.arm,
            "context": self.context,
            "operator": self.operator,
            "role_binding": self.role_binding,
            "transition": self.transition,
            "termination": self.termination,
            "safety_horizon": int(self.safety_horizon),
            "source_kind": self.source_kind,
            "ablated_factor": self.ablated_factor,
        }

    @property
    def program_hash(self) -> str:
        return sha256_payload(self.safe_payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CrossGameFactorProgram":
        assert_transfer_safe(payload)
        return cls(
            arm=str(payload["arm"]),
            context=str(payload["context"]),
            operator=str(payload["operator"]),
            role_binding=str(payload["role_binding"]),
            transition=str(payload["transition"]),
            termination=str(payload["termination"]),
            safety_horizon=int(payload["safety_horizon"]),
            source_kind=str(payload["source_kind"]),
            ablated_factor=(
                None
                if payload.get("ablated_factor") is None
                else str(payload["ablated_factor"])
            ),
        )


class CrossGameFactorRegistry:
    """Immutable support-zero registry compiled before target outcomes exist."""

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self.parent_factor_registry_checksum = ""
        self._programs: dict[tuple[str, str], CrossGameFactorProgram] = {}
        if payload is None:
            return
        if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
            raise ValueError("cross-game factor registry format drifted")
        if int(payload.get("local_support_total", -1)) != 0:
            raise ValueError("cross-game registry imported active support")
        self.parent_factor_registry_checksum = str(
            payload.get("parent_factor_registry_checksum", "")
        )
        for row in payload.get("programs", ()):
            program = CrossGameFactorProgram.from_payload(row["program"])
            if row.get("program_hash") != program.program_hash:
                raise ValueError("cross-game program hash drifted")
            if int(row.get("local_support", -1)) != 0:
                raise ValueError("cross-game program contains local support")
            self.register(program)

    def register(self, program: CrossGameFactorProgram) -> None:
        key = (program.arm, program.context)
        if key in self._programs:
            raise ValueError(f"duplicate cross-game factor program: {key}")
        self._programs[key] = program

    def program_for(self, arm: str, context: str) -> CrossGameFactorProgram:
        try:
            return self._programs[(arm, context)]
        except KeyError as exc:
            raise KeyError(f"missing cross-game program for {arm}/{context}") from exc

    def snapshot(self) -> dict[str, Any]:
        rows = []
        for key in sorted(self._programs):
            program = self._programs[key]
            rows.append(
                {
                    "arm": program.arm,
                    "context": program.context,
                    "program_hash": program.program_hash,
                    "program": program.safe_payload,
                    "local_support": 0,
                    "promoted": False,
                }
            )
        core = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "parent_factor_registry_checksum": self.parent_factor_registry_checksum,
            "programs": rows,
            "local_support_total": 0,
            "promotion_count": 0,
        }
        return signed(core, "registry_checksum")


def compile_cross_game_registry(
    parent_payload: Mapping[str, Any],
) -> CrossGameFactorRegistry:
    """Freeze T10.3.12b's source bundle and four isolated ablations."""

    parent = FactorRegistry(parent_payload)
    if int(parent_payload.get("local_support_total", -1)) != 0:
        raise ValueError("T10.3.12b factor registry does not have support zero")
    registry = CrossGameFactorRegistry()
    registry.parent_factor_registry_checksum = str(
        parent_payload.get("registry_checksum", "")
    )
    for context in CONTEXTS:
        source = parent.program_for("factorized_source", context)
        generic = parent.program_for("generic_source_free", context)
        base = {
            "operator": source.operator,
            "role_binding": source.role_binding,
            "transition": source.transition,
            "termination": source.termination,
            "safety_horizon": source.safety_horizon,
        }
        programs = {
            "factorized_source": {
                **base,
                "source_kind": "frozen_t10_3_12b_factor_bundle",
                "ablated_factor": None,
            },
            "generic_source_free": {
                "operator": "infer_legal_operator",
                "role_binding": generic.role_binding,
                "transition": "bounded_schema_and_binding_search",
                "termination": "stop_on_progress_or_repeated_state",
                "safety_horizon": source.safety_horizon,
                "source_kind": "source_free_generic_grammar",
                "ablated_factor": None,
            },
            "operator_ablation": {
                **base,
                "operator": "unparameterized_apply",
                "source_kind": "single_factor_ablation",
                "ablated_factor": "operator",
            },
            "role_binding_ablation": {
                **base,
                "role_binding": "lexicographic_local_binding",
                "source_kind": "single_factor_ablation",
                "ablated_factor": "role_binding",
            },
            "transition_ablation": {
                **base,
                "transition": "break_source_transition",
                "source_kind": "single_factor_ablation",
                "ablated_factor": "transition",
            },
            "termination_ablation": {
                **base,
                "termination": "fixed_two_steps",
                "source_kind": "single_factor_ablation",
                "ablated_factor": "termination",
            },
        }
        for arm in ARMS:
            registry.register(
                CrossGameFactorProgram(arm=arm, context=context, **programs[arm])
            )
    return registry


@dataclass(frozen=True)
class GroundingDecision:
    candidate: ActionCandidate | None
    context: str
    reason: str
    inspections: int
    program_hash: str | None
    ablated_factor: str | None

    @property
    def abstained(self) -> bool:
        return self.candidate is None


def _candidate_key(candidate: ActionCandidate) -> str:
    return f"{candidate.action_name}:{canonical_json(dict(candidate.action_data))}"


def _parameterized(candidates: Sequence[ActionCandidate]) -> tuple[ActionCandidate, ...]:
    return tuple(candidate for candidate in candidates if candidate.action_data)


def _unparameterized(candidates: Sequence[ActionCandidate]) -> tuple[ActionCandidate, ...]:
    return tuple(candidate for candidate in candidates if not candidate.action_data)


def _grounded(candidates: Sequence[ActionCandidate]) -> tuple[GroundedAction, ...]:
    return tuple(
        GroundedAction(candidate.action_name, tuple(dict(candidate.action_data).items()))
        for candidate in candidates
    )


def recognize_context(
    state: AbstractState | None,
    candidates: Sequence[ActionCandidate],
) -> tuple[str, tuple[GroundedAction, ...]]:
    parameterized = _parameterized(candidates)
    if not parameterized:
        return "", ()
    if state is not None:
        macro = chain_successor_macro(
            state,
            _grounded(parameterized),
            config=SearchConfig(maximum_horizon=16),
        )
        if macro is not None and macro.actions:
            return "path_context", tuple(macro.actions)
    return "repeat_context", ()


def _same_candidate(left: ActionCandidate, right: GroundedAction) -> bool:
    return left.action_name == right.action_name and dict(left.action_data) == right.data


def select_grounding(
    registry: CrossGameFactorRegistry,
    *,
    arm: str,
    candidates: Sequence[Any],
    shape: tuple[int, int],
    step_index: int,
    state: AbstractState | None = None,
    forced_context: str | None = None,
    forced_path: Sequence[GroundedAction] = (),
) -> GroundingDecision:
    """Select a legal action without any legacy fallback.

    ``forced_context`` and ``forced_path`` exist only for deterministic
    synthetic tests.  Active execution derives both from the current frame.
    """

    legal = tuple(sorted(normalized_action_candidates(candidates), key=_candidate_key))
    inspections = len(legal)
    context, path = recognize_context(state, legal)
    if forced_context is not None:
        context = forced_context
        path = tuple(forced_path)
    lookup_context = context or "repeat_context"
    program = registry.program_for(arm, lookup_context)

    if program.termination == "fixed_two_steps" and step_index >= 2:
        return GroundingDecision(
            None, context, "termination_ablation_fixed_stop", inspections,
            program.program_hash, program.ablated_factor,
        )

    if arm == "generic_source_free":
        if not legal:
            return GroundingDecision(
                None, context, "no_legal_candidate", inspections,
                program.program_hash, None,
            )
        return GroundingDecision(
            legal[step_index % len(legal)], context or "schema_context",
            "source_free_bounded_enumeration", inspections,
            program.program_hash, None,
        )

    if program.operator == "unparameterized_apply":
        matches = _unparameterized(legal)
        if not matches:
            return GroundingDecision(
                None, context, "operator_ablation_schema_miss", inspections,
                program.program_hash, program.ablated_factor,
            )
        return GroundingDecision(
            matches[0], context or "schema_context", "unparameterized_operator_ablation",
            inspections, program.program_hash, program.ablated_factor,
        )

    parameterized = _parameterized(legal)
    if not parameterized or not context:
        return GroundingDecision(
            None, "", "parameterized_operator_schema_miss", inspections,
            program.program_hash, program.ablated_factor,
        )

    if context == "path_context":
        matches = []
        for wanted in path:
            grounded = [
                candidate for candidate in parameterized
                if _same_candidate(candidate, wanted)
            ]
            if len(grounded) == 1 and grounded[0] not in matches:
                matches.append(grounded[0])
        if not matches:
            return GroundingDecision(
                None, context, "path_grounding_miss", inspections,
                program.program_hash, program.ablated_factor,
            )
        if arm == "role_binding_ablation":
            selected = matches[-1]
            reason = "path_orientation_ablation"
        elif arm == "transition_ablation":
            selected = matches[0] if step_index == 0 else matches[-1]
            reason = "path_transition_ablation"
        else:
            selected = matches[0]
            reason = "successor_toward_salient_end"
        return GroundingDecision(
            selected, context, reason, inspections,
            program.program_hash, program.ablated_factor,
        )

    if arm == "role_binding_ablation":
        return GroundingDecision(
            parameterized[0], context, "lexicographic_role_ablation", inspections,
            program.program_hash, program.ablated_factor,
        )
    distances = [
        (boundary_distance(candidate.action_data, shape), candidate)
        for candidate in parameterized
    ]
    scored = [(distance, candidate) for distance, candidate in distances if distance is not None]
    if not scored:
        return GroundingDecision(
            None, context, "relative_role_unavailable", inspections,
            program.program_hash, program.ablated_factor,
        )
    best = min(float(distance) for distance, _ in scored)
    winners = [candidate for distance, candidate in scored if float(distance) == best]
    if len(winners) != 1:
        return GroundingDecision(
            None, context, "ambiguous_relative_role", inspections,
            program.program_hash, program.ablated_factor,
        )
    if arm == "transition_ablation" and step_index > 0:
        alternatives = [candidate for candidate in parameterized if candidate != winners[0]]
        if alternatives:
            return GroundingDecision(
                alternatives[(step_index - 1) % len(alternatives)], context,
                "repeat_transition_ablation", inspections,
                program.program_hash, program.ablated_factor,
            )
    return GroundingDecision(
        winners[0], context, "relative_boundary_role", inspections,
        program.program_hash, program.ablated_factor,
    )


__all__ = [
    "ARMS",
    "CONTEXTS",
    "FACTORS",
    "FORMAT_VERSION",
    "REGISTRY_FORMAT_VERSION",
    "CrossGameFactorProgram",
    "CrossGameFactorRegistry",
    "GroundingDecision",
    "canonical_json",
    "compile_cross_game_registry",
    "recognize_context",
    "select_grounding",
    "sha256_payload",
    "signed",
    "verify_signed",
]
