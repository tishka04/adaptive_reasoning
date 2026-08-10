"""Corrected action-root continuation and shared-quotient transport for T10.3.1.

T10.3.1 keeps pre-action rooting independent of outcomes.  A successful
pre-action binding remains valid evidence even when the rooted object
disappears during a level transition.  After-state continuation follows only
predeclared rules: same branch-local id, movement actor, the same explicit or
transient action binding, then an exact unique structural-signature match.

The only non-identity transport declared comparable is the common structural
quotient between the allocentric and action-aligned relational frames.
Orientation-specific predicates are deliberately outside that quotient.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import AbstractState, ActionCandidate, ObservedTransition
from .frame_adapters_v10_2 import project_transition_with_frozen_frames
from .frame_adapters_v10_3 import (
    MOVEMENT_ACTIONS,
    resolve_pre_action_root,
    structural_signature,
)
from .observer_frames_v10_2 import (
    OBSERVER_FRAME_SPECS,
    ObserverFrameSpec,
    PhysicalEventBundle,
)

FORMAT_VERSION = "sage-t10.3.1-frame-adapter-v1"
COMPARABLE_SOURCE_FRAME = "allocentric_object_relative"
COMPARABLE_TARGET_FRAME = "action_aligned_relational"
ORIENTATION_PREDICATES = frozenset(
    {
        "north_of",
        "south_of",
        "east_of",
        "west_of",
        "ahead_of",
        "behind",
        "left_of_axis",
        "right_of_axis",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorrectedRootBinding:
    method: str
    structural_signature: str | None
    unique: bool
    pre_action_complete: bool
    after_root_available: bool
    after_method: str
    missing: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Rooting QA concerns the pre-action causal binding only."""

        return self.pre_action_complete

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "method": self.method,
            "structural_signature": self.structural_signature,
            "unique": self.unique,
            "complete": self.complete,
            "pre_action_complete": self.pre_action_complete,
            "after_root_available": self.after_root_available,
            "after_method": self.after_method,
            "missing": list(self.missing),
            "raw_identifier_retained": False,
            "spatial_anchor_retained": False,
            "post_action_effect_inference_used": False,
        }


@dataclass(frozen=True)
class CorrectedGoalProjection:
    bundle: PhysicalEventBundle
    binding: CorrectedRootBinding


def _with_root(state: AbstractState, entity_id: str | None) -> AbstractState:
    registers = {key: value for key, value in state.registers if key != "action_root"}
    if entity_id is not None:
        registers["action_root"] = entity_id
    return replace(state, registers=tuple(registers.items()))


def _after_continuation(
    before: AbstractState,
    after: AbstractState,
    action: ActionCandidate,
    root_id: str,
) -> tuple[str | None, str]:
    after_ids = {entity.entity_id for entity in after.entities}
    if root_id in after_ids:
        return root_id, "same_branch_local_binding"
    if action.action_name in MOVEMENT_ACTIONS:
        actors = tuple(
            entity.entity_id
            for entity in after.entities
            if set(entity.roles) & {"actor", "player"}
        )
        if len(actors) == 1:
            return actors[0], "movement_actor_continuation"
    rebound = resolve_pre_action_root(after, action)
    if rebound.entity_id is not None and rebound.unique:
        return rebound.entity_id, f"predeclared_{rebound.method}"
    signature = structural_signature(before, root_id)
    matches = tuple(
        entity.entity_id
        for entity in after.entities
        if structural_signature(after, entity.entity_id) == signature
    )
    if len(matches) == 1:
        return matches[0], "exact_structural_signature"
    return None, "unavailable"


def project_goal_transition(
    evidence: ObservedTransition,
    *,
    frames: Sequence[ObserverFrameSpec] = OBSERVER_FRAME_SPECS,
    event_id: str | None = None,
    event_nonce: str = "",
) -> CorrectedGoalProjection:
    resolved = resolve_pre_action_root(evidence.state_before, evidence.action)
    signature = (
        structural_signature(evidence.state_before, resolved.entity_id)
        if resolved.entity_id is not None
        else None
    )
    after_root, after_method = (
        _after_continuation(
            evidence.state_before,
            evidence.state_after,
            evidence.action,
            resolved.entity_id,
        )
        if resolved.entity_id is not None
        else (None, "unavailable")
    )
    missing = list(resolved.missing)
    if resolved.entity_id is not None and after_root is None:
        missing.append("after_root_unavailable")
    rooted = replace(
        evidence,
        state_before=_with_root(evidence.state_before, resolved.entity_id),
        state_after=_with_root(evidence.state_after, after_root),
    )
    bundle = project_transition_with_frozen_frames(
        rooted,
        frames=frames,
        event_id=event_id,
        event_nonce=event_nonce,
    )
    return CorrectedGoalProjection(
        bundle=bundle,
        binding=CorrectedRootBinding(
            method=resolved.method,
            structural_signature=signature,
            unique=resolved.unique,
            pre_action_complete=resolved.entity_id is not None and resolved.unique,
            after_root_available=after_root is not None,
            after_method=after_method,
            missing=tuple(sorted(set(missing))),
        ),
    )


def common_transport_quotient(frame_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a compact frame onto the preregistered shared relation quotient."""

    def endpoint(stage: str) -> dict[str, Any]:
        raw = frame_payload.get(stage, {})
        if not isinstance(raw, Mapping):
            raise ValueError("frame endpoint quotient is absent")
        facts = [
            dict(row)
            for row in raw.get("fact_rows", ())
            if isinstance(row, Mapping)
            and str(row.get("predicate", "")) not in ORIENTATION_PREDICATES
        ]
        return {
            "entity_count": raw.get("entity_count"),
            "regime_index": raw.get("regime_index"),
            "role_rows": raw.get("role_rows", []),
            "fact_rows": sorted(facts, key=_canonical),
            "counter_rows": raw.get("counter_rows", []),
            "register_rows": raw.get("register_rows", []),
            "topology_rows": raw.get("topology_rows", []),
        }

    return {"before": endpoint("before"), "after": endpoint("after")}


def shared_quotient_transport(
    frames: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Certify exact non-identity transport only on the declared common quotient."""

    source = frames.get(COMPARABLE_SOURCE_FRAME)
    target = frames.get(COMPARABLE_TARGET_FRAME)
    complete = bool(
        isinstance(source, Mapping)
        and isinstance(target, Mapping)
        and source.get("complete")
        and target.get("complete")
    )
    source_quotient = common_transport_quotient(source) if complete else None
    target_quotient = common_transport_quotient(target) if complete else None
    exact = bool(complete and source_quotient == target_quotient)
    before_hash = _sha(source_quotient["before"]) if source_quotient else None
    after_hash = _sha(source_quotient["after"]) if source_quotient else None
    delta_count = 0
    if source_quotient:
        for key in (
            "role_rows",
            "fact_rows",
            "counter_rows",
            "register_rows",
            "topology_rows",
        ):
            before_rows = {_canonical(row) for row in source_quotient["before"][key]}
            after_rows = {_canonical(row) for row in source_quotient["after"][key]}
            delta_count += len(before_rows ^ after_rows)
    certificate = {
        "source_frame": COMPARABLE_SOURCE_FRAME,
        "target_frame": COMPARABLE_TARGET_FRAME,
        "mapping_kind": "exact_common_quotient" if exact else "noncomparable",
        "comparable": exact,
        "exact": exact,
        "projection_complete": complete,
        "round_trip_exact": exact,
        "certifies_gauge_equivalence": exact,
        "commutativity": {"exact": exact},
        "common_quotient_before_hash": before_hash,
        "common_quotient_after_hash": after_hash,
        "transport_hash": _sha(
            {
                "source": COMPARABLE_SOURCE_FRAME,
                "target": COMPARABLE_TARGET_FRAME,
                "quotient": "orientation_erased_structural_relational_v1",
            }
        ),
    }
    certificate["certificate_hash"] = _sha(certificate)
    visible_noncomparable = [
        {
            "source_frame": "root_only",
            "target_frame": frame,
            "mapping_kind": "visible_noncomparable",
            "comparable": False,
            "exact": False,
            "projection_complete": bool(
                frames.get("root_only", {}).get("complete")
                and frames.get(frame, {}).get("complete")
            ),
            "round_trip_exact": False,
            "certifies_gauge_equivalence": False,
            "commutativity": {"exact": False},
            "reason": "FRAME_VOCABULARY_NOT_ISOMORPHIC",
        }
        for frame in (
            "allocentric_object_relative",
            "action_aligned_relational",
            "action_rooted_topological",
        )
    ]
    certificates = [certificate, *visible_noncomparable]
    summary = {
        "declared_comparable_certificate_count": int(exact),
        "noncomparable_certificate_count": len(certificates) - int(exact),
        "exact_certificate_count": int(exact),
        "exact_nonidentity_certificate_count": int(exact),
        "commutative_exact": exact,
        "round_trip_exact": exact,
        "multiframe_exact_nonidentity": exact,
        "identity_only_control": False,
        "incomplete_projections_attested_exact": False,
        "common_quotient": "orientation_erased_structural_relational_v1",
        "common_quotient_changed": bool(before_hash and before_hash != after_hash),
        "common_quotient_delta_count": delta_count,
    }
    return summary, certificates


__all__ = [
    "COMPARABLE_SOURCE_FRAME",
    "COMPARABLE_TARGET_FRAME",
    "CorrectedGoalProjection",
    "CorrectedRootBinding",
    "FORMAT_VERSION",
    "common_transport_quotient",
    "project_goal_transition",
    "shared_quotient_transport",
]
