"""M3.7f feasibility probe for offline-frame counterfactual replay.

This module does not replay an alternative action. It checks whether an
offline trace row contains enough executable state to justify a later
counterfactual executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m2.schema import M2_TRUTH_STATUS, M3CandidateExperimentRequest
from theory.m2.testability_compiler import load_m3_requests_payload, requests_from_payload

from .m2_candidate_experiment_runner import (
    DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    find_offline_source_transition,
    load_offline_trace_rows,
    source_transition_id_for_row,
)


DEFAULT_COUNTERFACTUAL_M2_REQUESTS_PATH = (
    Path("diagnostics") / "m2" / "fused_llm_wm_m3_candidate_requests.json"
)
DEFAULT_COUNTERFACTUAL_FEASIBILITY_PATH = (
    Path("diagnostics") / "m3" / "offline_frame_counterfactual_feasibility.json"
)
COUNTERFACTUAL_PROBE_SCHEMA_VERSION = (
    "m3.offline_frame_counterfactual_feasibility.v1"
)
COUNTERFACTUAL_BLOCKING_REASON = (
    "BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME"
)
ENV_RESTORE_OR_COLLECTION_BLOCK = (
    "BLOCKED_REQUIRES_ENV_STATE_RESTORE_OR_ACTIVE_COLLECTION"
)
ACTION_UNAVAILABLE_BLOCK = "BLOCKED_ALTERNATIVE_ACTION_NOT_AVAILABLE_IN_FRAME"
SOURCE_MISSING_BLOCK = "BLOCKED_SOURCE_TRANSITION_NOT_FOUND"
FEASIBLE_FOR_EXECUTOR = "FEASIBLE_FOR_M3_7G_COUNTERFACTUAL_EXECUTOR"
FRONTIER_TYPE = "NEED_ACTIVE_COUNTERFACTUAL_COLLECTION_FROM_TRACE_CONTEXT"
FULL_ENV_STATE_KEYS = (
    "env_state",
    "serialized_env_state",
    "full_env_state",
    "env_snapshot",
    "state_snapshot",
    "frame_before_state",
    "restore_state",
    "state_payload",
)
RESTORE_CONTRACT_KEYS = (
    "env_state_restore_contract",
    "restore_contract",
    "restore_api",
    "restore_state_contract",
)
PAYLOAD_CONTRACT_HINTS = (
    "format",
    "state_hash",
    "hash",
    "state_id",
    "restore_api",
    "restore_contract",
)


def run_offline_frame_counterfactual_probe(
    *,
    m2_requests_path: str | Path = DEFAULT_COUNTERFACTUAL_M2_REQUESTS_PATH,
    offline_trace_dataset_path: str | Path = DEFAULT_OFFLINE_TRACE_DATASET_PATH,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    request_payload = load_m3_requests_payload(m2_requests_path)
    requests = [
        request
        for request in requests_from_payload(request_payload)
        if is_counterfactual_offline_frame_request(request)
    ]
    rows = load_offline_trace_rows(offline_trace_dataset_path)
    probe_results = [
        probe_counterfactual_request(request, rows)
        for request in requests
    ]
    payload = build_counterfactual_probe_payload(
        m2_requests_path=str(m2_requests_path),
        offline_trace_dataset_path=str(offline_trace_dataset_path),
        request_payload=request_payload,
        probe_results=probe_results,
    )
    if output_path is not None:
        write_offline_frame_counterfactual_feasibility(payload, output_path)
    return payload


def is_counterfactual_offline_frame_request(
    request: M3CandidateExperimentRequest,
) -> bool:
    reason = str(request.blocking_reason or "")
    request_id = str(request.request_id or "")
    expected_signal = str(request.expected_signal or "")
    return (
        request.status == "BLOCKED_NOT_TESTABLE"
        and request.replayability == "OFFLINE_TRACE_CONTEXT_ONLY"
        and request.context_state_origin == "human_trace_frame_before"
        and (
            "COUNTERFACTUAL" in reason
            or "terminal_safe_alternative_action" in request_id
            or "alternative" in expected_signal
        )
    )


def probe_counterfactual_request(
    request: M3CandidateExperimentRequest,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    try:
        source = find_offline_source_transition(request, rows)
    except ValueError:
        return _missing_source_result(request)

    available_actions = [
        str(action)
        for action in source.get("available_actions_t", []) or []
        if str(action)
    ]
    target_action_available = request.target_action in set(available_actions)
    payload_key, restore_payload = find_full_env_state_payload(source)
    payload_hash = stable_payload_hash(restore_payload)
    explicit_contract = find_restore_contract(source)
    restore_contract_detected = bool(
        explicit_contract
        or payload_contract_detected(restore_payload)
    )
    can_reconstruct = bool(restore_payload) and restore_contract_detected
    replay_exact_hashable = can_reconstruct and bool(payload_hash)
    can_replay = can_reconstruct and target_action_available

    if not target_action_available:
        feasibility = ACTION_UNAVAILABLE_BLOCK
    elif can_replay and replay_exact_hashable:
        feasibility = FEASIBLE_FOR_EXECUTOR
    else:
        feasibility = ENV_RESTORE_OR_COLLECTION_BLOCK

    source_transition_id = source_transition_id_for_row(source)
    result: Dict[str, Any] = {
        "request_id": request.request_id,
        "source_hypothesis_id": request.source_hypothesis_id,
        "game_id": request.game_id,
        "source_transition_id": source_transition_id,
        "source_episode_id": request.source_episode_id or source.get("episode_id"),
        "source_step": (
            request.source_step
            if request.source_step is not None
            else int(source.get("step", 0) or 0)
        ),
        "source_transition_found": True,
        "context_state_origin": request.context_state_origin,
        "replayability": request.replayability,
        "original_blocking_reason": request.blocking_reason,
        "target_action": request.target_action,
        "target_action_args": (
            dict(request.target_action_args)
            if request.target_action_args is not None
            else None
        ),
        "observed_trace_action": str(source.get("action", "")),
        "observed_trace_action_args": dict(source.get("action_args", {}) or {}),
        "target_action_would_be_counterfactual": (
            str(source.get("action", "")) != request.target_action
            or dict(source.get("action_args", {}) or {})
            != dict(request.target_action_args or {})
        ),
        "available_actions": available_actions,
        "available_actions_present": bool(available_actions),
        "target_action_available_in_frame": target_action_available,
        "frame_before_grid_present": source.get("grid_t") is not None,
        "frame_after_grid_present": source.get("grid_t1") is not None,
        "frame_before_is_visual_observation_only": not bool(restore_payload),
        "source_trace_locator_present": source_trace_locator_present(source),
        "source_trace_path": _source_value(source, "source_path"),
        "source_trace_line_number": _source_value(source, "line_number"),
        "full_env_state_payload_present": bool(restore_payload),
        "full_env_state_payload_key": payload_key,
        "restore_contract_detected": restore_contract_detected,
        "explicit_restore_contract_present": bool(explicit_contract),
        "env_state_payload_hash": payload_hash,
        "can_reconstruct_env_state": can_reconstruct,
        "can_replay_alternative_action": can_replay,
        "replay_exact_hashable": replay_exact_hashable,
        "comparable_result_expected": replay_exact_hashable,
        "counterfactual_feasibility": feasibility,
        "active_collection_required": feasibility
        == ENV_RESTORE_OR_COLLECTION_BLOCK,
        "recommended_frontier": (
            build_frontier_recommendation(request, source_transition_id, feasibility)
            if feasibility != FEASIBLE_FOR_EXECUTOR
            else None
        ),
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    return result


def find_full_env_state_payload(
    source: Mapping[str, Any],
) -> tuple[str | None, Any | None]:
    for key in FULL_ENV_STATE_KEYS:
        value = source.get(key)
        if value not in (None, "", {}, []):
            return key, value
    return None, None


def find_restore_contract(source: Mapping[str, Any]) -> Any | None:
    for key in RESTORE_CONTRACT_KEYS:
        value = source.get(key)
        if value not in (None, "", {}, []):
            return value
    return None


def payload_contract_detected(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        keys = {str(key) for key in payload.keys()}
        return bool(keys.intersection(PAYLOAD_CONTRACT_HINTS))
    return isinstance(payload, str) and bool(payload.strip())


def stable_payload_hash(payload: Any) -> str | None:
    if payload in (None, "", {}, []):
        return None
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    except TypeError:
        return None
    return hashlib.sha256(encoded).hexdigest()


def source_trace_locator_present(source: Mapping[str, Any]) -> bool:
    source_obj = source.get("source", {}) or {}
    if not isinstance(source_obj, Mapping):
        source_obj = {}
    return bool(
        source.get("source_path")
        or source_obj.get("source_path")
        or source.get("line_number")
        or source_obj.get("line_number")
    )


def build_frontier_recommendation(
    request: M3CandidateExperimentRequest,
    source_transition_id: str,
    feasibility: str,
) -> Dict[str, Any]:
    return {
        "frontier_type": FRONTIER_TYPE,
        "target_queue": "A40_OR_P2",
        "blocked_capability": "counterfactual_action_from_offline_trace_frame",
        "blocking_reason": feasibility,
        "source_request_id": request.request_id,
        "source_hypothesis_id": request.source_hypothesis_id,
        "source_transition_id": source_transition_id,
        "game_id": request.game_id,
        "target_action": request.target_action,
        "target_action_args": (
            dict(request.target_action_args)
            if request.target_action_args is not None
            else None
        ),
        "required_capability": (
            "restore_full_env_state_from_trace_frame_before_or_collect_active_"
            "counterfactual_from_same_context"
        ),
        "frontier_write_performed": False,
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
    }


def build_counterfactual_probe_payload(
    *,
    m2_requests_path: str,
    offline_trace_dataset_path: str,
    request_payload: Mapping[str, Any],
    probe_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = summarize_probe_results(probe_results)
    return {
        "config": {
            "schema_version": COUNTERFACTUAL_PROBE_SCHEMA_VERSION,
            "m2_requests_path": m2_requests_path,
            "offline_trace_dataset_path": offline_trace_dataset_path,
            "inputs_read": ["M2.15", "M2.14 offline trace dataset"],
            "artifacts_not_modified": ["M2", "A32", "A33", "A40", "P2"],
            "probe_policy": (
                "inspect_restore_feasibility_only_never_simulate_from_visual_grid"
            ),
        },
        "source_m2_summary": dict(request_payload.get("summary", {}) or {}),
        "counterfactual_probe_results": [dict(row) for row in probe_results],
        "summary": {
            **summary,
            "support": 0,
            "truth_status": M2_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "frontier_write_performed": False,
            "a32_remains_only_verdict_location": True,
        },
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
    }


def summarize_probe_results(
    probe_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    blocked_by_reason = Counter(
        str(row.get("counterfactual_feasibility", ""))
        for row in probe_results
        if str(row.get("counterfactual_feasibility", "")) != FEASIBLE_FOR_EXECUTOR
    )
    return {
        "counterfactual_requests_seen": len(probe_results),
        "source_transitions_found": sum(
            1 for row in probe_results if bool(row.get("source_transition_found"))
        ),
        "frame_before_visual_contexts_present": sum(
            1 for row in probe_results if bool(row.get("frame_before_grid_present"))
        ),
        "full_env_state_payloads_present": sum(
            1 for row in probe_results if bool(row.get("full_env_state_payload_present"))
        ),
        "restore_contracts_detected": sum(
            1 for row in probe_results if bool(row.get("restore_contract_detected"))
        ),
        "feasible_counterfactual_requests": sum(
            1
            for row in probe_results
            if str(row.get("counterfactual_feasibility", "")) == FEASIBLE_FOR_EXECUTOR
        ),
        "blocked_counterfactual_requests": sum(
            1
            for row in probe_results
            if str(row.get("counterfactual_feasibility", "")) != FEASIBLE_FOR_EXECUTOR
        ),
        "blocked_by_reason": dict(sorted(blocked_by_reason.items())),
        "can_reconstruct_env_state": bool(probe_results)
        and all(bool(row.get("can_reconstruct_env_state")) for row in probe_results),
        "can_replay_alternative_action": bool(probe_results)
        and all(bool(row.get("can_replay_alternative_action")) for row in probe_results),
        "replay_exact_hashable": bool(probe_results)
        and all(bool(row.get("replay_exact_hashable")) for row in probe_results),
        "active_collection_required": any(
            bool(row.get("active_collection_required")) for row in probe_results
        ),
        "frontier_recommendations": sum(
            1 for row in probe_results if row.get("recommended_frontier") is not None
        ),
        "recommended_frontier_type": FRONTIER_TYPE
        if any(row.get("recommended_frontier") is not None for row in probe_results)
        else None,
        "blocked_capability": (
            "counterfactual_action_from_offline_trace_frame"
            if probe_results
            else None
        ),
    }


def write_offline_frame_counterfactual_feasibility(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_COUNTERFACTUAL_FEASIBILITY_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _missing_source_result(request: M3CandidateExperimentRequest) -> Dict[str, Any]:
    source_transition_id = str(request.source_transition_id or "")
    return {
        "request_id": request.request_id,
        "source_hypothesis_id": request.source_hypothesis_id,
        "game_id": request.game_id,
        "source_transition_id": source_transition_id,
        "source_episode_id": request.source_episode_id,
        "source_step": request.source_step,
        "source_transition_found": False,
        "context_state_origin": request.context_state_origin,
        "replayability": request.replayability,
        "original_blocking_reason": request.blocking_reason,
        "target_action": request.target_action,
        "target_action_args": (
            dict(request.target_action_args)
            if request.target_action_args is not None
            else None
        ),
        "available_actions": [],
        "available_actions_present": False,
        "target_action_available_in_frame": False,
        "frame_before_grid_present": False,
        "frame_after_grid_present": False,
        "frame_before_is_visual_observation_only": True,
        "source_trace_locator_present": False,
        "full_env_state_payload_present": False,
        "full_env_state_payload_key": None,
        "restore_contract_detected": False,
        "explicit_restore_contract_present": False,
        "env_state_payload_hash": None,
        "can_reconstruct_env_state": False,
        "can_replay_alternative_action": False,
        "replay_exact_hashable": False,
        "comparable_result_expected": False,
        "counterfactual_feasibility": SOURCE_MISSING_BLOCK,
        "active_collection_required": True,
        "recommended_frontier": build_frontier_recommendation(
            request,
            source_transition_id,
            SOURCE_MISSING_BLOCK,
        ),
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _source_value(source: Mapping[str, Any], key: str) -> Any:
    source_obj = source.get("source", {}) or {}
    if isinstance(source_obj, Mapping) and source_obj.get(key) is not None:
        return source_obj.get(key)
    return source.get(key)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe offline-frame counterfactual replay feasibility.",
    )
    parser.add_argument(
        "--m2-requests",
        default=str(DEFAULT_COUNTERFACTUAL_M2_REQUESTS_PATH),
        help="Path to fused M2.15 -> M3 candidate requests.",
    )
    parser.add_argument(
        "--offline-trace-dataset",
        default=str(DEFAULT_OFFLINE_TRACE_DATASET_PATH),
        help="Path to M2.14 offline trace transition dataset.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_COUNTERFACTUAL_FEASIBILITY_PATH),
        help="Output diagnostics JSON path.",
    )
    args = parser.parse_args(argv)
    run_offline_frame_counterfactual_probe(
        m2_requests_path=args.m2_requests,
        offline_trace_dataset_path=args.offline_trace_dataset,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
