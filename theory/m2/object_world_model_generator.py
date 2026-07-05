"""Deterministic object/world-model context packet for M2.14a.

This module does not run a world model and does not produce evidence. It emits
candidate semantic context for ARC-LeWM and later fusion/handoff modules.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .schema import M2_HYPOTHESIS_STATUS, M2_TRUTH_STATUS


DEFAULT_OBJECT_WORLD_MODEL_PACKET_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "object_world_model_invariant_packet.json"
)
OBJECT_WORLD_MODEL_SCHEMA_VERSION = "m2.object_world_model_invariant_packet.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
UNLOCK_ONLY_ACTIONS = ("ACTION1", "ACTION2", "ACTION5")


def candidate_only_guard() -> Dict[str, Any]:
    return {
        "status": M2_HYPOTHESIS_STATUS,
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "controlled_test_required": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "world_model_prediction_counted_as_evidence": False,
        "world_model_score_counted_as_support": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def build_object_world_model_invariant_packet(
    *,
    raw_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    raw_context = dict(raw_context or {})
    candidates = list(_base_mechanistic_context_candidates())
    candidates.extend(
        _sanitize_candidate(candidate)
        for candidate in raw_context.get("mechanistic_context_candidates", []) or []
        if isinstance(candidate, Mapping)
    )
    candidates = [_sanitize_candidate(candidate) for candidate in candidates]
    guard = candidate_only_guard()
    return {
        "config": {
            "schema_version": OBJECT_WORLD_MODEL_SCHEMA_VERSION,
            "generator_mode": "deterministic_semantic_side_channel",
            "scope": "m2_14a_foundation_only",
            "not_a_world_model": True,
            "rollout_performed": False,
            "reward_read": False,
            "reconstruction_target_built": False,
        },
        "mechanistic_context_candidates": candidates,
        "summary": {
            "mechanistic_context_candidates": len(candidates),
            "world_model_candidates_section_present": False,
            "relation_progress_completion_signal": False,
            "hud_counted_as_objective_signal": False,
            "unavailable_actions_policy": "unlock_only",
            "unlock_target_actions": list(UNLOCK_ONLY_ACTIONS),
            "entity_identifier_policy": "role_candidate_only",
            **guard,
        },
        **guard,
    }


def write_object_world_model_invariant_packet(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OBJECT_WORLD_MODEL_PACKET_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_mechanistic_context_candidates() -> tuple[Dict[str, Any], ...]:
    guard = candidate_only_guard()
    return (
        {
            "candidate_id": "m2_14a::role_context::actor_target_locality",
            "candidate_type": "object_role_context",
            "description": (
                "Candidate actor, target and terrain roles derived as semantic "
                "context for latent transition modeling."
            ),
            "entities": [
                {
                    "role_candidate": "actor",
                    "entity_id": None,
                    "hard_coded_entity_id_used": False,
                },
                {
                    "role_candidate": "target_or_transformable_patch",
                    "entity_id": None,
                    "hard_coded_entity_id_used": False,
                },
                {
                    "role_candidate": "terrain_or_boundary",
                    "entity_id": None,
                    "hard_coded_entity_id_used": False,
                },
            ],
            "relation_policy": "candidate_only_object_roles",
            **guard,
        },
        {
            "candidate_id": "m2_14a::relation_context::progress_non_completion",
            "candidate_type": "relation_progress_context",
            "relation_candidate": "progress",
            "relation_interpretation": "non_completion_signal",
            "completion_signal": False,
            "progress_counted_as_completion": False,
            "hud_counted_as_objective_signal": False,
            "description": (
                "Progress-like relation changes may prioritize candidate actions, "
                "but are not objective completion evidence."
            ),
            **guard,
        },
        {
            "candidate_id": "m2_14a::hud_context::terminal_horizon_only",
            "candidate_type": "hud_horizon_context",
            "hud_policy": "terminal_avoidance_only",
            "hud_counted_as_objective_signal": False,
            "hud_counted_as_completion_signal": False,
            "use_hud_for_foundation_training": False,
            "description": (
                "HUD and horizon cues are preserved only as terminal-avoidance "
                "context; M2.14 foundation training disables HUD learning."
            ),
            **guard,
        },
        {
            "candidate_id": "m2_14a::precondition_context::unlock_only",
            "candidate_type": "unavailable_action_precondition_context",
            "unavailable_action_policy": "unlock_only",
            "unlock_target_actions": list(UNLOCK_ONLY_ACTIONS),
            "unavailable_actions_counted_as_objective_signal": False,
            "unavailable_actions_counted_as_support": False,
            "description": (
                "Unavailable ACTION1/ACTION2/ACTION5 are tracked only as possible "
                "unlock targets, never as confirmed objectives or support."
            ),
            **guard,
        },
        {
            "candidate_id": "m2_14a::proxy_gap_context::completion_gap",
            "candidate_type": "proxy_completion_gap_context",
            "proxy_completion_gap_policy": "candidate_context_only",
            "proxy_progress_counted_as_completion": False,
            "world_model_score_counted_as_support": False,
            "description": (
                "M3.G6/M3.G7 proxy-completion divergence is represented as a "
                "candidate context gap for future controlled tests."
            ),
            **guard,
        },
    )


def _sanitize_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    sanitized = deepcopy(dict(candidate))
    sanitized.update(candidate_only_guard())
    sanitized["candidate_id"] = str(
        sanitized.get("candidate_id") or "m2_14a::raw_context::candidate"
    )
    sanitized["candidate_type"] = str(
        sanitized.get("candidate_type") or "raw_context_candidate"
    )
    if sanitized.get("candidate_type") == "relation_progress_context":
        sanitized["relation_interpretation"] = "non_completion_signal"
        sanitized["completion_signal"] = False
        sanitized["progress_counted_as_completion"] = False
    if "hud" in sanitized["candidate_type"] or "hud_policy" in sanitized:
        sanitized["hud_policy"] = "terminal_avoidance_only"
        sanitized["hud_counted_as_objective_signal"] = False
        sanitized["hud_counted_as_completion_signal"] = False
    if "unlock_target_actions" in sanitized or "unavailable_action" in sanitized["candidate_type"]:
        sanitized["unavailable_action_policy"] = "unlock_only"
        sanitized["unlock_target_actions"] = list(UNLOCK_ONLY_ACTIONS)
        sanitized["unavailable_actions_counted_as_support"] = False
    sanitized["world_model_prediction_counted_as_evidence"] = False
    sanitized["world_model_score_counted_as_support"] = False
    _sanitize_entities(sanitized)
    return sanitized


def _sanitize_entities(candidate: Dict[str, Any]) -> None:
    entities = candidate.get("entities")
    if not isinstance(entities, list):
        return
    sanitized_entities = []
    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, Mapping):
            continue
        role = str(entity.get("role_candidate") or entity.get("role") or f"role_{index}")
        sanitized_entities.append(
            {
                "role_candidate": role,
                "entity_id": None,
                "hard_coded_entity_id_used": False,
            }
        )
    candidate["entities"] = sanitized_entities


def _candidate_only_errors(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        if value.get("support", 0) != 0:
            errors.append(f"{path}.support")
        if value.get("truth_status") not in {None, M2_TRUTH_STATUS}:
            errors.append(f"{path}.truth_status")
        if value.get("revision_status") not in {None, CANDIDATE_REVISION_STATUS}:
            errors.append(f"{path}.revision_status")
        for key in (
            "a32_write_performed",
            "a33_write_performed",
            "world_model_prediction_counted_as_evidence",
            "world_model_score_counted_as_support",
        ):
            if value.get(key) is True:
                errors.append(f"{path}.{key}")
        for child_key, child_value in value.items():
            errors.extend(_candidate_only_errors(child_value, f"{path}.{child_key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_candidate_only_errors(child, f"{path}[{index}]"))
    return errors


def validate_object_world_model_packet(payload: Mapping[str, Any]) -> None:
    if "world_model_candidates" in payload:
        raise ValueError("packet must use mechanistic_context_candidates, not world_model_candidates")
    if "mechanistic_context_candidates" not in payload:
        raise ValueError("missing mechanistic_context_candidates")
    errors = _candidate_only_errors(payload)
    if errors:
        raise ValueError(f"candidate-only guard failed: {errors}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic M2.14a object/world-model context packet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECT_WORLD_MODEL_PACKET_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_object_world_model_invariant_packet()
    validate_object_world_model_packet(payload)
    write_object_world_model_invariant_packet(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "support": 0,
                "revision_status": CANDIDATE_REVISION_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
