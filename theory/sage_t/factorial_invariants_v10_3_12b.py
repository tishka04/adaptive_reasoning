"""Factorial invariant identification for SAGE.T10.3.12b.

T10.3.12 established that the source-conditioned and source-free controllers
produce the same physical trajectories on the two deterministic core states.
This module therefore does not create another live controller.  It exposes a
compact counterfactual laboratory in which four program factors can be
intervened on independently:

* operator schema;
* causal role binding;
* transition relation;
* terminal stopping rule.

The laboratory is deliberately symbolic and source-identity free.  It can
identify transportable *candidates*, but it cannot establish transfer to an
unseen game.  That distinction is part of the frozen T10.3.12b contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .relational_program_v10_3_12 import (
    ARMS as PARENT_ARMS,
    RelationalProgramRegistry,
    assert_transfer_safe,
)

FORMAT_VERSION = "sage-t10.3.12b-factorial-invariant-v1"
REGISTRY_FORMAT_VERSION = "sage-t10.3.12b-factor-registry-v1"
VARIANT_FORMAT_VERSION = "sage-t10.3.12b-counterfactual-recipe-v1"

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
TRANSFORMS = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "mirror_x",
    "mirror_y",
    "main_diagonal",
    "anti_diagonal",
)
ORDERS = ("canonical", "reverse")
CHALLENGES = (
    "short_positive",
    "long_positive",
    "ambiguous_role",
    "relation_decoupled",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = sha256_payload(result)
    return result


def verify_signed(payload: Mapping[str, Any], checksum_field: str) -> None:
    expected = str(payload.get(checksum_field, ""))
    core = {key: value for key, value in payload.items() if key != checksum_field}
    if not expected or sha256_payload(core) != expected:
        raise ValueError(f"invalid {checksum_field}")


@dataclass(frozen=True)
class FactorProgram:
    """Transfer-safe factorization; no grounded token or local coordinate."""

    context: str
    arm: str
    operator: str
    role_binding: str
    transition: str
    termination: str
    safety_horizon: int
    source_kind: str

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS:
            raise ValueError(f"unsupported context: {self.context}")
        if self.arm not in ARMS:
            raise ValueError(f"unsupported factor arm: {self.arm}")
        if not 1 <= int(self.safety_horizon) <= 16:
            raise ValueError("safety horizon must be in [1, 16]")
        assert_transfer_safe(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "context": self.context,
            "arm": self.arm,
            "operator": self.operator,
            "role_binding": self.role_binding,
            "transition": self.transition,
            "termination": self.termination,
            "safety_horizon": int(self.safety_horizon),
            "source_kind": self.source_kind,
        }

    @property
    def program_hash(self) -> str:
        return sha256_payload(self.safe_payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FactorProgram:
        assert_transfer_safe(payload)
        return cls(
            context=str(payload["context"]),
            arm=str(payload["arm"]),
            operator=str(payload["operator"]),
            role_binding=str(payload["role_binding"]),
            transition=str(payload["transition"]),
            termination=str(payload["termination"]),
            safety_horizon=int(payload["safety_horizon"]),
            source_kind=str(payload["source_kind"]),
        )


class FactorRegistry:
    """Two context programs per arm, always with zero active support."""

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self._programs: dict[tuple[str, str], FactorProgram] = {}
        self.parent_registry_checksum = ""
        if payload is None:
            return
        if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
            raise ValueError("factor registry format drifted")
        self.parent_registry_checksum = str(payload.get("parent_registry_checksum", ""))
        if int(payload.get("local_support_total", -1)) != 0:
            raise ValueError("factor registry must start with support zero")
        for row in payload.get("programs", ()):
            program = FactorProgram.from_payload(row["program"])
            if str(row.get("program_hash")) != program.program_hash:
                raise ValueError("factor program hash drifted")
            if int(row.get("local_support", -1)) != 0:
                raise ValueError("factor program contains active support")
            self.register(program)

    def register(self, program: FactorProgram) -> None:
        key = (program.arm, program.context)
        if key in self._programs:
            raise ValueError(f"duplicate factor program: {key}")
        self._programs[key] = program

    def program_for(self, arm: str, context: str) -> FactorProgram:
        try:
            return self._programs[(arm, context)]
        except KeyError as exc:
            raise KeyError(f"missing factor program for {arm}/{context}") from exc

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
        return {
            "format_version": REGISTRY_FORMAT_VERSION,
            "parent_registry_checksum": self.parent_registry_checksum,
            "programs": rows,
            "local_support_total": 0,
            "promotion_count": 0,
        }


def _program(
    context: str,
    arm: str,
    *,
    operator: str,
    role: str,
    transition: str,
    termination: str,
    source_kind: str,
) -> FactorProgram:
    return FactorProgram(
        context=context,
        arm=arm,
        operator=operator,
        role_binding=role,
        transition=transition,
        termination=termination,
        safety_horizon=8 if context == "repeat_context" else 16,
        source_kind=source_kind,
    )


def compile_factor_registry(parent_payload: Mapping[str, Any]) -> FactorRegistry:
    """Factor T10.3.12 source programs without importing grounded evidence."""

    parent = RelationalProgramRegistry(parent_payload)
    if int(parent_payload.get("local_support_total", -1)) != 0:
        raise ValueError("T10.3.12 source registry does not have support zero")
    parent_repeat = parent.program_for(PARENT_ARMS[0], "repeat_context")
    parent_path = parent.program_for(PARENT_ARMS[0], "path_context")
    if parent_repeat.mechanism != "repeat_causal_role":
        raise ValueError("T10.3.12 repeat source mechanism drifted")
    if parent_path.mechanism != "salient_path_successor":
        raise ValueError("T10.3.12 path source mechanism drifted")

    registry = FactorRegistry()
    registry.parent_registry_checksum = str(parent_payload.get("registry_checksum", ""))
    for context in CONTEXTS:
        if context == "repeat_context":
            correct = {
                "operator": "parameterized_apply",
                "role": "boundary_prior_with_causal_verification",
                "transition": "same_role_until_progress",
            }
            generic_role = "probe_relations_then_bind_productive_role"
        else:
            correct = {
                "operator": "parameterized_apply",
                "role": "salient_end_prior_with_causal_verification",
                "transition": "successor_toward_goal_end",
            }
            generic_role = "probe_orientations_then_bind_goal_end"
        registry.register(
            _program(
                context,
                ARMS[0],
                operator=correct["operator"],
                role=correct["role"],
                transition=correct["transition"],
                termination="stop_on_progress_or_ambiguity",
                source_kind="t10_3_12_factorized_projection",
            )
        )
        registry.register(
            _program(
                context,
                ARMS[1],
                operator=correct["operator"],
                role=generic_role,
                transition=correct["transition"],
                termination="stop_on_progress_or_ambiguity",
                source_kind="generic_grammar_a_priori",
            )
        )
        registry.register(
            _program(
                context,
                ARMS[2],
                operator="unrelated_operator",
                role=correct["role"],
                transition=correct["transition"],
                termination="stop_on_progress_or_ambiguity",
                source_kind="single_factor_ablation",
            )
        )
        registry.register(
            _program(
                context,
                ARMS[3],
                operator=correct["operator"],
                role="candidate_order_role",
                transition=correct["transition"],
                termination="stop_on_progress_or_ambiguity",
                source_kind="single_factor_ablation",
            )
        )
        registry.register(
            _program(
                context,
                ARMS[4],
                operator=correct["operator"],
                role=correct["role"],
                transition=(
                    "rotate_roles_without_accumulation"
                    if context == "repeat_context"
                    else "repeat_node_without_successor"
                ),
                termination="stop_on_progress_or_ambiguity",
                source_kind="single_factor_ablation",
            )
        )
        registry.register(
            _program(
                context,
                ARMS[5],
                operator=correct["operator"],
                role=correct["role"],
                transition=correct["transition"],
                termination="fixed_horizon_ignore_progress",
                source_kind="single_factor_ablation",
            )
        )
    return registry


@dataclass(frozen=True)
class VariantRecipe:
    variant_id: str
    context: str
    split: str
    transform: str
    order: str
    challenge: str
    size: int

    @property
    def core(self) -> dict[str, Any]:
        return {
            "format_version": VARIANT_FORMAT_VERSION,
            "context": self.context,
            "split": self.split,
            "transform": self.transform,
            "order": self.order,
            "challenge": self.challenge,
            "size": int(self.size),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.core, "variant_id": self.variant_id}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VariantRecipe:
        core = {
            key: payload[key]
            for key in (
                "format_version",
                "context",
                "split",
                "transform",
                "order",
                "challenge",
                "size",
            )
        }
        if core["format_version"] != VARIANT_FORMAT_VERSION:
            raise ValueError("counterfactual recipe format drifted")
        variant_id = str(payload.get("variant_id", ""))
        if variant_id != sha256_payload(core):
            raise ValueError("counterfactual recipe checksum drifted")
        return cls(
            variant_id=variant_id,
            context=str(core["context"]),
            split=str(core["split"]),
            transform=str(core["transform"]),
            order=str(core["order"]),
            challenge=str(core["challenge"]),
            size=int(core["size"]),
        )


def variant_recipes() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for context in CONTEXTS:
        for transform_index, transform in enumerate(TRANSFORMS):
            split = "identification" if transform_index < 4 else "challenge"
            for order in ORDERS:
                for challenge in CHALLENGES:
                    if context == "repeat_context":
                        size = 3 if challenge == "short_positive" else 7
                    else:
                        size = 6 if challenge == "short_positive" else 14
                    core = {
                        "format_version": VARIANT_FORMAT_VERSION,
                        "context": context,
                        "split": split,
                        "transform": transform,
                        "order": order,
                        "challenge": challenge,
                        "size": size,
                    }
                    rows.append({**core, "variant_id": sha256_payload(core)})
    if len(rows) != 128 or len({row["variant_id"] for row in rows}) != 128:
        raise AssertionError("T10.3.12b requires 128 unique variants")
    return tuple(rows)


@dataclass(frozen=True)
class CounterfactualWorld:
    recipe: VariantRecipe
    state_hash: str
    candidate_count: int
    boundary_position: int
    productive_position: int
    goal_at_start: bool
    causal_goal_at_start: bool
    ambiguous: bool
    expected_abstain: bool
    expected_steps: int


def materialize_variant(payload: Mapping[str, Any]) -> CounterfactualWorld:
    recipe = VariantRecipe.from_payload(payload)
    transform_index = TRANSFORMS.index(recipe.transform)
    reverse = recipe.order == "reverse"
    ambiguous = recipe.challenge == "ambiguous_role"
    decoupled = recipe.challenge == "relation_decoupled"
    if recipe.context == "repeat_context":
        candidate_count = 2 + (transform_index % 4)
        if recipe.challenge != "short_positive":
            candidate_count += 2
        boundary_position = (transform_index + int(reverse)) % candidate_count
        productive_position = (
            (
                boundary_position
                + 1
                + (transform_index % (candidate_count - 1))
            )
            % candidate_count
            if decoupled
            else boundary_position
        )
        goal_at_start = False
        causal_goal_at_start = False
        expected_steps = recipe.size
        observable = {
            "context": recipe.context,
            "transform": recipe.transform,
            "order": recipe.order,
            "candidate_count": candidate_count,
            "boundary_relation_positions": (
                [boundary_position, (boundary_position + 1) % candidate_count]
                if ambiguous
                else [boundary_position]
            ),
        }
    else:
        candidate_count = recipe.size + (transform_index % 3)
        boundary_position = -1
        productive_position = -1
        goal_at_start = bool((transform_index + int(reverse)) % 2)
        causal_goal_at_start = not goal_at_start if decoupled else goal_at_start
        expected_steps = recipe.size - 1
        observable = {
            "context": recipe.context,
            "transform": recipe.transform,
            "order": recipe.order,
            "path_nodes": recipe.size,
            "branch_distractors": transform_index % 3,
            "salient_goal_ends": [0, recipe.size - 1] if ambiguous else [0 if goal_at_start else recipe.size - 1],
        }
    state_hash = sha256_payload(observable)
    return CounterfactualWorld(
        recipe=recipe,
        state_hash=state_hash,
        candidate_count=candidate_count,
        boundary_position=boundary_position,
        productive_position=productive_position,
        goal_at_start=goal_at_start,
        causal_goal_at_start=causal_goal_at_start,
        ambiguous=ambiguous,
        expected_abstain=ambiguous,
        expected_steps=expected_steps,
    )


@dataclass(frozen=True)
class TrialOutcome:
    correct: bool
    success: bool
    abstained: bool
    virtual_actions: int
    probes: int
    first_decision_class: str
    stop_reason: str
    factor_failure: str

    def compact(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "success": self.success,
            "abstained": self.abstained,
            "virtual_actions": self.virtual_actions,
            "probes": self.probes,
            "first_decision_class": self.first_decision_class,
            "stop_reason": self.stop_reason,
            "factor_failure": self.factor_failure,
        }


def evaluate_trial(program: FactorProgram, world: CounterfactualWorld) -> TrialOutcome:
    """Run a compact interactive intervention with probes charged as actions."""

    if program.context != world.recipe.context:
        return TrialOutcome(False, False, True, 0, 0, "abstain", "context_miss", "operator")
    generic = program.arm == ARMS[1]
    if world.ambiguous:
        if program.role_binding == "candidate_order_role":
            return TrialOutcome(False, False, False, 1, 0, "apply", "ambiguity_ignored", "role_binding")
        probes = world.candidate_count if generic else 2
        return TrialOutcome(
            True,
            False,
            True,
            probes,
            probes,
            "exhaustive_probe" if generic else "verify_source_prior",
            "ambiguity",
            "",
        )

    if program.operator != "parameterized_apply":
        return TrialOutcome(False, False, False, 1, 0, "unrelated_operator", "operator_miss", "operator")

    probes = 0
    first = "apply" if world.recipe.context == "repeat_context" else "successor"
    if generic:
        probes = world.candidate_count
        first = "exhaustive_probe"

    if world.recipe.context == "repeat_context":
        if generic:
            selected_position = world.productive_position
        elif program.role_binding == "candidate_order_role":
            selected_position = 0 if world.recipe.order == "canonical" else world.candidate_count - 1
        else:
            if world.boundary_position == world.productive_position:
                probes = 1
            else:
                order = list(range(world.candidate_count))
                if world.recipe.order == "reverse":
                    order.reverse()
                alternatives = [
                    position
                    for position in order
                    if position != world.boundary_position
                ]
                probes = 2 + alternatives.index(world.productive_position)
            first = "verify_source_prior"
            selected_position = world.productive_position
        if selected_position != world.productive_position:
            return TrialOutcome(False, False, False, probes + 1, probes, first, "wrong_role", "role_binding")
    else:
        if generic:
            selected_start = world.causal_goal_at_start
        elif program.role_binding == "candidate_order_role":
            selected_start = world.recipe.order == "canonical"
        else:
            probes = 1 if world.goal_at_start == world.causal_goal_at_start else 2
            first = "verify_source_prior"
            selected_start = world.causal_goal_at_start
        if selected_start != world.causal_goal_at_start:
            return TrialOutcome(False, False, False, probes + 1, probes, first, "wrong_goal_end", "role_binding")

    correct_transition = (
        "same_role_until_progress"
        if world.recipe.context == "repeat_context"
        else "successor_toward_goal_end"
    )
    if program.transition != correct_transition:
        return TrialOutcome(
            False,
            False,
            False,
            probes + min(world.expected_steps, 2),
            probes,
            first,
            "transition_stall",
            "transition",
        )

    actions = probes + world.expected_steps
    if program.termination != "stop_on_progress_or_ambiguity":
        return TrialOutcome(
            False,
            True,
            False,
            actions + 1,
            probes,
            first,
            "post_progress_overrun",
            "termination",
        )
    return TrialOutcome(True, True, False, actions, probes, first, "progress_stop", "")


def median(values: Sequence[int]) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


__all__ = [
    "ARMS",
    "CHALLENGES",
    "CONTEXTS",
    "CounterfactualWorld",
    "FACTORS",
    "FORMAT_VERSION",
    "FactorProgram",
    "FactorRegistry",
    "ORDERS",
    "REGISTRY_FORMAT_VERSION",
    "TRANSFORMS",
    "TrialOutcome",
    "VARIANT_FORMAT_VERSION",
    "canonical_json",
    "compile_factor_registry",
    "evaluate_trial",
    "materialize_variant",
    "median",
    "sha256_payload",
    "signed",
    "variant_recipes",
    "verify_signed",
]
