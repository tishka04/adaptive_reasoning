"""Factorised, transfer-safe relational programs for SAGE.T10.3.12.

The module deliberately separates a transferable program from its reset-local
grounding.  Persisted programs describe an operator, a causal role, a
transition relation and a stop condition; concrete action arguments and
coordinates only exist inside an in-memory fixture or live controller.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

FORMAT_VERSION = "sage-t10.3.12-relational-program-v1"
REGISTRY_FORMAT_VERSION = "sage-t10.3.12-relational-registry-v1"
FIXTURE_FORMAT_VERSION = "sage-t10.3.12-offline-fixture-recipe-v1"

ARMS = (
    "factorized_relational_source",
    "generic_grammar_source_free",
    "schema_swap_wrong_source",
    "relation_ablation",
)
CONTEXTS = ("repeat_context", "path_context")
D4_TRANSFORMS = (
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
PALETTES = ("identity", "cycle_nonmodal")

_FORBIDDEN_KEYS = frozenset(
    {
        "game_id",
        "seed",
        "x",
        "y",
        "color",
        "raw_grid",
        "grid",
        "entity_id",
        "action_data",
        "argument_checksum",
        "grounded_evidence",
    }
)
_FORBIDDEN_STRING_TOKENS = (
    "lp85",
    "su15",
    "re86",
    "ls20",
    "sc25",
    "ar25",
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


def assert_transfer_safe(value: Any, *, path: str = "payload") -> None:
    """Reject local identities, raw observations and grounded arguments."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden transfer field at {path}.{key}")
            assert_transfer_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            assert_transfer_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in _FORBIDDEN_STRING_TOKENS):
            raise ValueError(f"forbidden local identity at {path}")


@dataclass(frozen=True)
class RoleBindingSpec:
    """Coordinate-free description of the target role and ambiguity policy."""

    selector: str
    evidence_relation: str
    ambiguity_policy: str = "abstain"

    def __post_init__(self) -> None:
        if self.ambiguity_policy != "abstain":
            raise ValueError("relational bindings must abstain on ambiguity")
        if not self.selector or not self.evidence_relation:
            raise ValueError("role binding needs selector and evidence relation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "evidence_relation": self.evidence_relation,
            "ambiguity_policy": self.ambiguity_policy,
        }


@dataclass(frozen=True)
class RelationalProgram:
    """A compact option template with no grounded trajectory."""

    context: str
    mechanism: str
    operator_schema: str
    role_binding: RoleBindingSpec
    transition_relation: str
    stop_conditions: tuple[str, ...]
    safety_horizon: int
    source_kind: str

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS:
            raise ValueError(f"unsupported relational context: {self.context}")
        if not 1 <= int(self.safety_horizon) <= 16:
            raise ValueError("relational program horizon must be in [1, 16]")
        required = {"level_increment", "game_over", "state_cycle", "ambiguity"}
        if not required.issubset(self.stop_conditions):
            raise ValueError("relational program lacks fail-closed stop conditions")
        assert_transfer_safe(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "context": self.context,
            "mechanism": self.mechanism,
            "operator_schema": self.operator_schema,
            "role_binding": self.role_binding.as_dict(),
            "transition_relation": self.transition_relation,
            "stop_conditions": list(self.stop_conditions),
            "safety_horizon": int(self.safety_horizon),
            "source_kind": self.source_kind,
        }

    @property
    def program_hash(self) -> str:
        return sha256_payload(self.safe_payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RelationalProgram:
        assert_transfer_safe(payload)
        role = payload["role_binding"]
        if not isinstance(role, Mapping):
            raise ValueError("role_binding must be a mapping")
        return cls(
            context=str(payload["context"]),
            mechanism=str(payload["mechanism"]),
            operator_schema=str(payload["operator_schema"]),
            role_binding=RoleBindingSpec(
                selector=str(role["selector"]),
                evidence_relation=str(role["evidence_relation"]),
                ambiguity_policy=str(role.get("ambiguity_policy", "abstain")),
            ),
            transition_relation=str(payload["transition_relation"]),
            stop_conditions=tuple(str(item) for item in payload["stop_conditions"]),
            safety_horizon=int(payload["safety_horizon"]),
            source_kind=str(payload["source_kind"]),
        )


@dataclass
class _ProgramEvidence:
    program: RelationalProgram
    support_scopes: set[str] = field(default_factory=set)
    contradictions: set[str] = field(default_factory=set)
    controls: set[str] = field(default_factory=set)

    @property
    def promoted(self) -> bool:
        return (
            len(self.support_scopes) >= 4
            and {"binding_swap", "order_permutation", "relation_ablation"}.issubset(
                self.controls
            )
        )


class RelationalProgramRegistry:
    """Arm/context registry with support-zero transfer and explicit promotion."""

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self._rows: dict[tuple[str, str], _ProgramEvidence] = {}
        self._local_support: dict[str, int] = {}
        self.loaded_checksum: str | None = None
        if payload is not None:
            self._load(payload)

    def register(self, arm: str, program: RelationalProgram) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown T10.3.12 arm: {arm}")
        key = (arm, program.context)
        current = self._rows.get(key)
        if current is not None and current.program.program_hash != program.program_hash:
            raise ValueError("arm/context program collision")
        self._rows.setdefault(key, _ProgramEvidence(program=program))

    def program_for(self, arm: str, context: str) -> RelationalProgram:
        try:
            return self._rows[(str(arm), str(context))].program
        except KeyError as exc:
            raise ValueError(f"missing relational program for {arm}/{context}") from exc

    def note_success(self, arm: str, context: str, scope: str) -> None:
        row = self._rows[(arm, context)]
        token = sha256_payload({"program": row.program.program_hash, "scope": str(scope)})
        row.support_scopes.add(token)
        self._local_support[row.program.program_hash] = (
            self._local_support.get(row.program.program_hash, 0) + 1
        )

    def note_contradiction(self, arm: str, context: str, receipt: str) -> None:
        row = self._rows[(arm, context)]
        row.contradictions.add(
            sha256_payload({"program": row.program.program_hash, "receipt": str(receipt)})
        )

    def note_controls(self, arm: str, context: str, controls: Sequence[str]) -> None:
        self._rows[(arm, context)].controls.update(str(item) for item in controls)

    def local_support(self, program_hash: str) -> int:
        return int(self._local_support.get(str(program_hash), 0))

    def snapshot(self, *, promoted_only: bool = False) -> dict[str, Any]:
        programs = []
        for (arm, context), row in sorted(self._rows.items()):
            if promoted_only and not row.promoted:
                continue
            programs.append(
                {
                    "arm": arm,
                    "context": context,
                    "program_hash": row.program.program_hash,
                    "program": row.program.safe_payload,
                    "support_scopes": sorted(row.support_scopes),
                    "contradictions": sorted(row.contradictions),
                    "controls": sorted(row.controls),
                    "promoted": row.promoted,
                }
            )
        core = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "transfer_support_policy": "support_zero_until_active_level_increment",
            "programs": programs,
        }
        assert_transfer_safe(core)
        return signed(core, "registry_checksum")

    def _load(self, payload: Mapping[str, Any]) -> None:
        verify_signed(payload, "registry_checksum")
        if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
            raise ValueError("unsupported relational registry format")
        assert_transfer_safe(
            {key: value for key, value in payload.items() if key != "registry_checksum"}
        )
        for item in payload.get("programs", ()):
            if not isinstance(item, Mapping):
                raise ValueError("invalid relational registry row")
            program = RelationalProgram.from_payload(item["program"])
            if str(item.get("program_hash")) != program.program_hash:
                raise ValueError("relational program hash mismatch")
            arm = str(item["arm"])
            self.register(arm, program)
            row = self._rows[(arm, program.context)]
            row.support_scopes.update(str(value) for value in item.get("support_scopes", ()))
            row.contradictions.update(str(value) for value in item.get("contradictions", ()))
            row.controls.update(str(value) for value in item.get("controls", ()))
        self.loaded_checksum = str(payload["registry_checksum"])


def _program(
    context: str,
    mechanism: str,
    selector: str,
    relation: str,
    *,
    horizon: int,
    source_kind: str,
) -> RelationalProgram:
    return RelationalProgram(
        context=context,
        mechanism=mechanism,
        operator_schema="parameterized_apply",
        role_binding=RoleBindingSpec(
            selector=selector,
            evidence_relation=(
                "productive_effect_equivalence" if context == "repeat_context"
                else "path_component_membership"
            ),
        ),
        transition_relation=relation,
        stop_conditions=("level_increment", "game_over", "state_cycle", "ambiguity"),
        safety_horizon=horizon,
        source_kind=source_kind,
    )


def compile_candidate_registry(
    canonical_descriptors: Mapping[str, Any],
    *,
    repeat_projection_identified: bool,
) -> RelationalProgramRegistry:
    """Compile all preregistered arms from an allowlisted abstract projection.

    The input may contain source dictionary keys, but no value carrying
    grounded evidence is accepted and no source identity reaches the output.
    """

    if "grounded_evidence" in canonical_json(canonical_descriptors):
        raise ValueError("grounded evidence is forbidden in T10.3.12 compilation")
    descriptors = list(canonical_descriptors.values())
    schemas = {str(row.get("macro_schema")) for row in descriptors if isinstance(row, Mapping)}
    selectors = {str(row.get("target_selector")) for row in descriptors if isinstance(row, Mapping)}
    if schemas != {"repeat_target", "path_successor"}:
        raise ValueError("canonical source projection lacks both mechanisms")
    if not {"same_effect_distinct_target", "successor_toward_enclosure"}.issubset(selectors):
        raise ValueError("canonical source projection selectors drifted")
    if not repeat_projection_identified:
        raise ValueError("repeat causal role was not identified in the authorized source shard")

    registry = RelationalProgramRegistry()
    registry.register(
        ARMS[0],
        _program(
            "repeat_context",
            "repeat_causal_role",
            "unique_productive_effect_role_then_boundary_relation",
            "same_effect_same_role_until_terminal",
            horizon=8,
            source_kind="factorized_source_projection",
        ),
    )
    registry.register(
        ARMS[0],
        _program(
            "path_context",
            "salient_path_successor",
            "unique_salient_enclosure_end",
            "successor_toward_salient_enclosure",
            horizon=16,
            source_kind="factorized_source_projection",
        ),
    )
    registry.register(
        ARMS[1],
        _program(
            "repeat_context",
            "generic_repeat_search",
            "enumerate_all_candidate_roles",
            "repeat_first_consistent_role",
            horizon=8,
            source_kind="generic_grammar_a_priori",
        ),
    )
    registry.register(
        ARMS[1],
        _program(
            "path_context",
            "generic_path_search",
            "enumerate_all_path_orientations",
            "successor_toward_unique_salient_end",
            horizon=16,
            source_kind="generic_grammar_a_priori",
        ),
    )
    registry.register(
        ARMS[2],
        _program(
            "repeat_context",
            "path_schema_on_repeat_context",
            "distinct_successor_order",
            "advance_instead_of_repeat",
            horizon=8,
            source_kind="wrong_source_schema_swap",
        ),
    )
    registry.register(
        ARMS[2],
        _program(
            "path_context",
            "repeat_schema_on_path_context",
            "single_repeated_role",
            "repeat_instead_of_advance",
            horizon=16,
            source_kind="wrong_source_schema_swap",
        ),
    )
    registry.register(
        ARMS[3],
        _program(
            "repeat_context",
            "hash_offset_repeat_ablation",
            "ephemeral_hash_offset",
            "repeat_hash_selected_role",
            horizon=8,
            source_kind="relation_ablation",
        ),
    )
    registry.register(
        ARMS[3],
        _program(
            "path_context",
            "lexicographic_path_ablation",
            "lexicographic_endpoint_without_salience",
            "successor_toward_lexicographic_end",
            horizon=16,
            source_kind="relation_ablation",
        ),
    )
    return registry


def transform_point(point: tuple[int, int], transform: str, *, size: int = 64) -> tuple[int, int]:
    """Apply one element of D4 to a row/column point."""

    row, col = point
    maximum = size - 1
    operations = {
        "identity": (row, col),
        "rotate_90": (col, maximum - row),
        "rotate_180": (maximum - row, maximum - col),
        "rotate_270": (maximum - col, row),
        "mirror_x": (maximum - row, col),
        "mirror_y": (row, maximum - col),
        "main_diagonal": (col, row),
        "anti_diagonal": (maximum - col, maximum - row),
    }
    if transform not in operations:
        raise ValueError(f"unknown D4 transform: {transform}")
    return operations[transform]


def inverse_transform(transform: str) -> str:
    return {
        "identity": "identity",
        "rotate_90": "rotate_270",
        "rotate_180": "rotate_180",
        "rotate_270": "rotate_90",
        "mirror_x": "mirror_x",
        "mirror_y": "mirror_y",
        "main_diagonal": "main_diagonal",
        "anti_diagonal": "anti_diagonal",
    }[transform]


@dataclass(frozen=True)
class OfflineCandidate:
    token: str
    point: tuple[int, int]
    effect_role: str = "unknown"
    path_index: int = -1
    salient_end: bool = False


@dataclass(frozen=True)
class RelationalFixture:
    fixture_id: str
    context: str
    transform: str
    order: str
    palette: str
    control: str
    candidates: tuple[OfflineCandidate, ...]
    expected_tokens: tuple[str, ...]
    expected_abstain: bool


def fixture_recipes() -> tuple[dict[str, Any], ...]:
    recipes: list[dict[str, Any]] = []
    for context in CONTEXTS:
        for transform in D4_TRANSFORMS:
            for order in ORDERS:
                for palette in PALETTES:
                    core = {
                        "format_version": FIXTURE_FORMAT_VERSION,
                        "context": context,
                        "transform": transform,
                        "order": order,
                        "palette": palette,
                        "control": "positive",
                    }
                    recipes.append({**core, "fixture_id": sha256_payload(core)})
            controls = (
                ("binding_conflict", "ambiguous_effect")
                if context == "repeat_context"
                else ("broken_bridge", "ambiguous_orientation")
            )
            for control in controls:
                core = {
                    "format_version": FIXTURE_FORMAT_VERSION,
                    "context": context,
                    "transform": transform,
                    "order": "canonical",
                    "palette": "identity",
                    "control": control,
                }
                recipes.append({**core, "fixture_id": sha256_payload(core)})
    if len(recipes) != 96 or len({row["fixture_id"] for row in recipes}) != 96:
        raise AssertionError("T10.3.12 fixture matrix must contain 96 unique recipes")
    return tuple(recipes)


def materialize_fixture(recipe: Mapping[str, Any]) -> RelationalFixture:
    core = {key: recipe[key] for key in ("format_version", "context", "transform", "order", "palette", "control")}
    if core["format_version"] != FIXTURE_FORMAT_VERSION:
        raise ValueError("fixture recipe format drifted")
    if str(recipe.get("fixture_id")) != sha256_payload(core):
        raise ValueError("fixture recipe checksum drifted")
    context = str(core["context"])
    transform = str(core["transform"])
    order = str(core["order"])
    control = str(core["control"])
    if context == "repeat_context":
        base = (
            OfflineCandidate("productive_role", (31, 4), "productive"),
            OfflineCandidate("distractor_role", (31, 57), "sterile"),
        )
        if control == "binding_conflict":
            base = (
                OfflineCandidate("distractor_role", (31, 4), "sterile"),
                OfflineCandidate("productive_role", (31, 57), "productive"),
            )
        elif control == "ambiguous_effect":
            base = (
                OfflineCandidate("productive_role", (31, 4), "productive"),
                OfflineCandidate("distractor_role", (31, 57), "productive"),
            )
        candidates = tuple(
            OfflineCandidate(
                item.token,
                transform_point(item.point, transform),
                item.effect_role,
                item.path_index,
                item.salient_end,
            )
            for item in base
        )
        if order == "reverse":
            candidates = tuple(reversed(candidates))
        expected_abstain = control == "ambiguous_effect"
        expected = () if expected_abstain else ("productive_role",)
    else:
        points = tuple((54 - 4 * index, 8 + 4 * index) for index in range(11))
        base = tuple(
            OfflineCandidate(
                f"path_{index:02d}",
                point,
                path_index=index,
                salient_end=index == len(points) - 1,
            )
            for index, point in enumerate(points)
        )
        if control == "ambiguous_orientation":
            base = tuple(
                OfflineCandidate(
                    item.token,
                    item.point,
                    path_index=item.path_index,
                    salient_end=item.path_index in {0, len(points) - 1},
                )
                for item in base
            )
        candidates = tuple(
            OfflineCandidate(
                item.token,
                transform_point(item.point, transform),
                item.effect_role,
                item.path_index,
                item.salient_end,
            )
            for item in base
        )
        if order == "reverse":
            candidates = tuple(reversed(candidates))
        expected_abstain = control in {"broken_bridge", "ambiguous_orientation"}
        expected = () if expected_abstain else tuple(f"path_{index:02d}" for index in range(1, 11))
    return RelationalFixture(
        fixture_id=str(recipe["fixture_id"]),
        context=context,
        transform=transform,
        order=order,
        palette=str(core["palette"]),
        control=control,
        candidates=candidates,
        expected_tokens=expected,
        expected_abstain=expected_abstain,
    )


@dataclass(frozen=True)
class GroundingOutcome:
    tokens: tuple[str, ...]
    points: tuple[tuple[int, int], ...]
    abstained: bool
    inspections: int
    reason: str


def _ordered_path(fixture: RelationalFixture) -> tuple[OfflineCandidate, ...]:
    return tuple(sorted(fixture.candidates, key=lambda item: item.path_index))


def evaluate_fixture(program: RelationalProgram, fixture: RelationalFixture) -> GroundingOutcome:
    """Evaluate a preregistered program without using source identities."""

    if program.context != fixture.context:
        return GroundingOutcome((), (), True, 1, "context_miss")
    if fixture.context == "repeat_context":
        candidates = fixture.candidates
        if program.mechanism == "repeat_causal_role":
            productive = [item for item in candidates if item.effect_role == "productive"]
            if len(productive) != 1:
                return GroundingOutcome((), (), True, len(candidates), "ambiguous_effect_role")
            selected = productive[0]
            inspections = len(candidates)
        elif program.mechanism == "generic_repeat_search":
            productive = [item for item in candidates if item.effect_role == "productive"]
            if len(productive) != 1:
                return GroundingOutcome((), (), True, 8, "generic_ambiguity")
            selected = productive[0]
            inspections = 8
        elif program.mechanism == "path_schema_on_repeat_context":
            ordered = sorted(candidates, key=lambda item: item.point)
            tokens = tuple(item.token for item in ordered)
            return GroundingOutcome(tokens, tuple(item.point for item in ordered), False, len(ordered), "wrong_successor")
        else:
            selected = candidates[0]
            inspections = 1
        return GroundingOutcome(
            (selected.token,),
            (selected.point,),
            False,
            inspections,
            "selected_repeat_role",
        )

    ordered = _ordered_path(fixture)
    salient = [item for item in ordered if item.salient_end]
    if fixture.control == "broken_bridge":
        if program.mechanism in {"salient_path_successor", "generic_path_search"}:
            return GroundingOutcome((), (), True, len(ordered), "broken_bridge")
    if program.mechanism == "salient_path_successor":
        if len(salient) != 1 or salient[0].path_index not in {0, len(ordered) - 1}:
            return GroundingOutcome((), (), True, len(ordered), "ambiguous_orientation")
        path = ordered if salient[0].path_index == len(ordered) - 1 else tuple(reversed(ordered))
        inspections = len(ordered)
    elif program.mechanism == "generic_path_search":
        if len(salient) != 1:
            return GroundingOutcome((), (), True, 32, "generic_ambiguity")
        path = ordered if salient[0].path_index == len(ordered) - 1 else tuple(reversed(ordered))
        inspections = 32
    elif program.mechanism == "repeat_schema_on_path_context":
        selected = ordered[0]
        path = tuple(selected for _ in range(min(10, program.safety_horizon)))
        inspections = 1
    else:
        target = min((ordered[0], ordered[-1]), key=lambda item: item.point)
        path = ordered if target.path_index == len(ordered) - 1 else tuple(reversed(ordered))
        inspections = 2
    emitted = tuple(path[1 : 1 + program.safety_horizon])
    return GroundingOutcome(
        tuple(item.token for item in emitted),
        tuple(item.point for item in emitted),
        False,
        inspections,
        "selected_path_successors",
    )


def fixture_correct(fixture: RelationalFixture, outcome: GroundingOutcome) -> bool:
    if fixture.expected_abstain:
        return outcome.abstained
    return not outcome.abstained and outcome.tokens == fixture.expected_tokens


def boundary_distance(
    action_data: Mapping[str, Any], shape: tuple[int, int]
) -> float | None:
    """Reset-local D4-equivariant realization of the repeat target role."""

    try:
        row = float(action_data["y"])
        col = float(action_data["x"])
        height, width = shape
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(row) and math.isfinite(col) and height > 0 and width > 0):
        return None
    return min(row, col, float(height - 1) - row, float(width - 1) - col)


__all__ = [
    "ARMS",
    "CONTEXTS",
    "D4_TRANSFORMS",
    "FIXTURE_FORMAT_VERSION",
    "FORMAT_VERSION",
    "GroundingOutcome",
    "OfflineCandidate",
    "RelationalFixture",
    "RelationalProgram",
    "RelationalProgramRegistry",
    "RoleBindingSpec",
    "assert_transfer_safe",
    "boundary_distance",
    "canonical_json",
    "compile_candidate_registry",
    "evaluate_fixture",
    "fixture_correct",
    "fixture_recipes",
    "inverse_transform",
    "materialize_fixture",
    "sha256_payload",
    "signed",
    "transform_point",
    "verify_signed",
]
