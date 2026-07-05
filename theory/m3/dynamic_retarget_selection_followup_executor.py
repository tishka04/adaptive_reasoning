"""M3.16 executor for dynamic retarget selection-rule follow-ups."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from theory.m2.m3_execution_smoke import _execute_named_action, _make_env, _reset_env
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir
from theory.real_env_option_adapter import snapshot_frame

from .dynamic_retarget_followup_planner import available_target_args_after_replay
from .dynamic_retarget_selection_followup_planner import (
    DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    EXPLICIT_RETARGET_ARG_POLICY,
    LOCAL_PATCH_SIMILARITY_POLICY,
    READY_FOR_M3_SELECTION_FOLLOWUP,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .refined_followup_executor import (
    available_followup_controls,
    blocked_followup_row,
    execute_metric_followup_experiment,
)


DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "dynamic_retarget_selection_followup_results.json"
)
DEFAULT_MAX_DYNAMIC_ARGS_PER_REQUEST = 1


def run_dynamic_retarget_selection_followup_execution(
    *,
    selection_followup_requests_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    max_dynamic_args_per_request: int = DEFAULT_MAX_DYNAMIC_ARGS_PER_REQUEST,
) -> Dict[str, Any]:
    payload = _load_json(selection_followup_requests_path)
    requests = [
        dict(request)
        for request in payload.get("followup_experiment_requests", []) or []
        if str(request.get("status", "")) == READY_FOR_M3_SELECTION_FOLLOWUP
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    experiments: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    resolutions: list[Dict[str, Any]] = []
    for request in requests:
        target_args_list, resolution_rows = resolve_selection_followup_target_args(
            request,
            environments_dir=env_dir,
            max_dynamic_args_per_request=max_dynamic_args_per_request,
        )
        resolutions.extend(resolution_rows)
        if not target_args_list:
            blocked.append(
                blocked_selection_row(
                    request,
                    reason="no_target_args_resolved",
                    resolution_rows=resolution_rows,
                )
            )
            continue
        controls = available_followup_controls(request, environments_dir=env_dir)
        if not controls:
            for target_args in target_args_list:
                blocked.append(
                    blocked_selection_row(
                        request,
                        reason="no_dynamic_control_available",
                        target_action_args=target_args,
                        resolution_rows=resolution_rows,
                    )
                )
            continue
        for target_args in target_args_list:
            for metric in request.get("metrics", []) or []:
                for control_action in controls:
                    try:
                        row = execute_metric_followup_experiment(
                            request,
                            metric=str(metric),
                            control_action=str(control_action),
                            target_action_args=target_args,
                            target_action_arg_policy=str(
                                request.get("target_action_arg_policy", "")
                            ),
                            environments_dir=env_dir,
                        )
                    except Exception as exc:
                        blocked.append(
                            blocked_selection_row(
                                request,
                                reason=f"execution_failed:{exc}",
                                metric=str(metric),
                                control_action=str(control_action),
                                target_action_args=target_args,
                                resolution_rows=resolution_rows,
                            )
                        )
                        continue
                    row.update(selection_experiment_metadata(request, target_args, metric))
                    experiments.append(row)

    per_request = per_request_results(experiments, blocked)
    rule_summary = rule_family_summary(per_request)
    return build_selection_followup_results_payload(
        selection_followup_requests_path=selection_followup_requests_path,
        environments_dir=env_dir,
        requests=requests,
        experiments=experiments,
        blocked=blocked,
        resolutions=resolutions,
        per_request=per_request,
        rule_summary=rule_summary,
        max_dynamic_args_per_request=max_dynamic_args_per_request,
    )


def resolve_selection_followup_target_args(
    request: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    max_dynamic_args_per_request: int = DEFAULT_MAX_DYNAMIC_ARGS_PER_REQUEST,
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    explicit = request.get("target_action_args")
    policy = str(request.get("target_action_arg_policy", ""))
    available_args = available_args_for_request(request, environments_dir=environments_dir)
    available_keys = {_args_key(args) for args in available_args}
    excluded_keys = {_args_key(args) for args in effective_excluded_args(request)}
    if isinstance(explicit, Mapping):
        args = dict(explicit)
        available = _args_key(args) in available_keys
        excluded = _args_key(args) in excluded_keys
        resolution = resolution_record(
            request,
            policy=EXPLICIT_RETARGET_ARG_POLICY,
            resolution_basis="explicit_selection_rule_probe_args",
            resolved_args=[args] if available and not excluded else [],
            candidate_args=[args],
            excluded_known_args_respected=not excluded,
            exact_explicit_args_available=available,
            blocked_reason=None
            if available and not excluded
            else (
                "explicit_args_excluded_known_arg"
                if excluded
                else "explicit_args_not_available_after_replay"
            ),
        )
        return ((args,), (resolution,)) if available and not excluded else ((), (resolution,))

    if policy != LOCAL_PATCH_SIMILARITY_POLICY:
        resolution = resolution_record(
            request,
            policy=policy,
            resolution_basis="unsupported_dynamic_arg_policy",
            resolved_args=[],
            candidate_args=[],
            excluded_known_args_respected=True,
            blocked_reason=f"unsupported_dynamic_arg_policy:{policy}",
        )
        return (), (resolution,)

    selected, candidates = resolve_local_patch_similarity_args(
        request,
        available_args=available_args,
        environments_dir=environments_dir,
        max_dynamic_args_per_request=max_dynamic_args_per_request,
    )
    resolution = resolution_record(
        request,
        policy=LOCAL_PATCH_SIMILARITY_POLICY,
        resolution_basis=patch_resolution_basis(request),
        resolved_args=list(selected),
        candidate_args=list(candidates),
        excluded_known_args_respected=True,
        exact_explicit_args_available=None,
        blocked_reason=None if selected else "no_patch_similar_available_args",
    )
    return selected, (resolution,)


def resolve_local_patch_similarity_args(
    request: Mapping[str, Any],
    *,
    available_args: Sequence[Mapping[str, Any]],
    environments_dir: str | Path,
    max_dynamic_args_per_request: int,
) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
    excluded_keys = {_args_key(args) for args in effective_excluded_args(request)}
    seeds = list(request.get("seed_successful_args", []) or []) or list(
        request.get("seed_failed_args", []) or []
    )
    if not seeds:
        return (), ()
    before_grid = grid_after_replay(request, environments_dir=environments_dir)
    seed_signatures = [
        local_patch_signature(before_grid, dict(seed)) for seed in seeds
    ]
    candidates: list[Dict[str, Any]] = []
    for args in available_args:
        args_dict = dict(args)
        if _args_key(args_dict) in excluded_keys:
            continue
        signature = local_patch_signature(before_grid, args_dict)
        score = min(
            patch_distance(signature, seed_signature)
            for seed_signature in seed_signatures
        )
        candidates.append(
            {
                "action_args": args_dict,
                "patch_similarity_score": float(score),
                "resolution_basis": patch_resolution_basis(request),
            }
        )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda row: (
                float(row.get("patch_similarity_score", 0.0) or 0.0),
                _args_key(row.get("action_args", {}) or {}),
            ),
        )
    )
    selected = tuple(
        dict(row.get("action_args", {}) or {})
        for row in ordered[: max(0, int(max_dynamic_args_per_request))]
    )
    return selected, ordered


def available_args_for_request(
    request: Mapping[str, Any],
    *,
    environments_dir: str | Path,
) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        dict(args)
        for args in available_target_args_after_replay(
            game_id=str(request.get("game_id", "")),
            context_replay=request.get("context_replay", []) or [],
            context_replay_args=request.get("context_replay_args"),
            target_action=str(request.get("target_action", "")),
            environments_dir=environments_dir,
        )
    )


def grid_after_replay(
    request: Mapping[str, Any],
    *,
    environments_dir: str | Path,
) -> np.ndarray:
    env = _make_env(str(request.get("game_id", "")), environments_dir)
    frame = _reset_env(env)
    replay = list(request.get("context_replay", []) or [])
    replay_args = list(request.get("context_replay_args", []) or [])
    for index, action_name in enumerate(replay):
        action_args = replay_args[index] if index < len(replay_args) else None
        frame = _execute_named_action(
            env,
            str(action_name),
            required_observation="",
            action_args=action_args if isinstance(action_args, Mapping) else None,
        )
    return np.asarray(snapshot_frame(frame).grid, dtype=np.int32)


def local_patch_signature(
    grid: np.ndarray,
    action_args: Mapping[str, Any],
    *,
    radius: int = 1,
) -> Tuple[Tuple[int, ...], ...]:
    x = _safe_int(action_args.get("x"))
    y = _safe_int(action_args.get("y"))
    if x is None or y is None or grid.ndim != 2:
        return ()
    y0 = max(0, y - radius)
    y1 = min(grid.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(grid.shape[1], x + radius + 1)
    return tuple(tuple(int(value) for value in row) for row in grid[y0:y1, x0:x1])


def patch_distance(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> float:
    left_rows = [list(row) for row in left]
    right_rows = [list(row) for row in right]
    height = max(len(left_rows), len(right_rows))
    width = max(
        [len(row) for row in left_rows] + [len(row) for row in right_rows] + [0]
    )
    distance = 0.0
    for y in range(height):
        for x in range(width):
            left_value = left_rows[y][x] if y < len(left_rows) and x < len(left_rows[y]) else -999
            right_value = right_rows[y][x] if y < len(right_rows) and x < len(right_rows[y]) else -999
            if left_value != right_value:
                distance += 1.0
    return distance


def selection_experiment_metadata(
    request: Mapping[str, Any],
    target_args: Mapping[str, Any],
    metric: str,
) -> Dict[str, Any]:
    success_metrics = {str(item) for item in request.get("success_metrics", []) or []}
    diagnostic_metrics = {
        str(item) for item in request.get("diagnostic_metrics", []) or []
    }
    metric_role = "success_metric" if metric in success_metrics else "diagnostic_metric"
    if metric not in success_metrics and metric not in diagnostic_metrics:
        metric_role = "unclassified_metric"
    return {
        "source_selection_rule_candidate_id": str(
            request.get("source_selection_rule_candidate_id", "")
        ),
        "source_mechanism_candidate_id": str(
            request.get("source_mechanism_candidate_id", "")
        ),
        "source_rule_id": str(request.get("source_rule_id", "")),
        "rule_family": str(request.get("rule_family", "")),
        "probe_family": str(request.get("probe_family", "")),
        "metric_role": metric_role,
        "success_metrics": list(request.get("success_metrics", []) or []),
        "diagnostic_metrics": list(request.get("diagnostic_metrics", []) or []),
        "known_successful_retargets": [
            dict(item) for item in request.get("known_successful_retargets", []) or []
        ],
        "known_failed_retargets": [
            dict(item) for item in request.get("known_failed_retargets", []) or []
        ],
        "excluded_args": [dict(item) for item in effective_excluded_args(request)],
        "resolved_target_action_args": [dict(target_args)],
        "target_action_arg_policy": str(request.get("target_action_arg_policy", "")),
        "rule_counted_as_confirmation": False,
        "followup_request_counted_as_support": False,
    }


def per_request_results(
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_request: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in experiments:
        by_request[str(row.get("request_id", ""))].append(dict(row))

    results: list[Dict[str, Any]] = []
    for request_id, rows in sorted(by_request.items()):
        first = rows[0]
        success_rows = [row for row in rows if row.get("metric_role") == "success_metric"]
        diagnostic_rows = [
            row for row in rows if row.get("metric_role") == "diagnostic_metric"
        ]
        results.append(
            {
                "request_id": request_id,
                "source_selection_rule_candidate_id": str(
                    first.get("source_selection_rule_candidate_id", "")
                ),
                "source_mechanism_candidate_id": str(
                    first.get("source_mechanism_candidate_id", "")
                ),
                "source_rule_id": str(first.get("source_rule_id", "")),
                "rule_family": str(first.get("rule_family", "")),
                "probe_family": str(first.get("probe_family", "")),
                "target_action": str(first.get("target_action", "")),
                "resolved_target_action_args": sorted_args(
                    [row.get("target_action_args", {}) or {} for row in rows]
                ),
                "target_action_arg_policy": str(
                    first.get("target_action_arg_policy", "")
                ),
                "metrics_tested": sorted(
                    {str(row.get("metric", "")) for row in rows if row.get("metric")}
                ),
                "controls_tested": sorted(
                    {
                        str(row.get("control_action", ""))
                        for row in rows
                        if row.get("control_action")
                    }
                ),
                "controlled_experiments_run": sum(
                    int(row.get("controlled_experiments_run", 0) or 0)
                    for row in rows
                ),
                "support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in rows
                ),
                "contradiction_events": sum(
                    int(row.get("contradiction_events", 0) or 0) for row in rows
                ),
                "neutral_events": sum(
                    int(row.get("neutral_events", 0) or 0) for row in rows
                ),
                "success_metric_support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in success_rows
                ),
                "success_metric_contradiction_events": sum(
                    int(row.get("contradiction_events", 0) or 0)
                    for row in success_rows
                ),
                "diagnostic_support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in diagnostic_rows
                ),
                "diagnostic_contradiction_events": sum(
                    int(row.get("contradiction_events", 0) or 0)
                    for row in diagnostic_rows
                ),
                "grounded_success_metrics": sorted(
                    {
                        str(row.get("metric", ""))
                        for row in success_rows
                        if int(row.get("support_events", 0) or 0) > 0
                        and not bool(row.get("diagnostic_only"))
                    }
                ),
                "request_has_success_metric_support": any(
                    int(row.get("support_events", 0) or 0) > 0
                    and row.get("metric_role") == "success_metric"
                    for row in rows
                ),
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "controlled_test_required": True,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                "rule_counted_as_confirmation": False,
                "followup_request_counted_as_support": False,
            }
        )
    for row in blocked:
        if str(row.get("blocked_reason", "")) == "no_target_args_resolved":
            results.append(blocked_request_result(row))
    return tuple(results)


def blocked_request_result(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "request_id": str(row.get("request_id", "")),
        "source_selection_rule_candidate_id": str(
            row.get("source_selection_rule_candidate_id", "")
        ),
        "source_mechanism_candidate_id": str(
            row.get("source_mechanism_candidate_id", "")
        ),
        "source_rule_id": str(row.get("source_rule_id", "")),
        "rule_family": str(row.get("rule_family", "")),
        "probe_family": str(row.get("probe_family", "")),
        "resolved_target_action_args": [],
        "blocked_reason": str(row.get("blocked_reason", "")),
        "controlled_experiments_run": 0,
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 0,
        "success_metric_support_events": 0,
        "success_metric_contradiction_events": 0,
        "diagnostic_support_events": 0,
        "diagnostic_contradiction_events": 0,
        "grounded_success_metrics": [],
        "request_has_success_metric_support": False,
        "status": "BLOCKED_NOT_EXECUTED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "rule_counted_as_confirmation": False,
        "followup_request_counted_as_support": False,
    }


def rule_family_summary(
    per_request: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_family: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in per_request:
        by_family[str(row.get("rule_family", ""))].append(dict(row))
    summaries: list[Dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        summaries.append(
            {
                "rule_family": family,
                "requests": len(rows),
                "requests_with_success_metric_support": len(
                    [row for row in rows if bool(row.get("request_has_success_metric_support"))]
                ),
                "blocked_requests": len(
                    [row for row in rows if str(row.get("status", "")) == "BLOCKED_NOT_EXECUTED"]
                ),
                "controlled_experiments_run": sum(
                    int(row.get("controlled_experiments_run", 0) or 0) for row in rows
                ),
                "success_metric_support_events": sum(
                    int(row.get("success_metric_support_events", 0) or 0)
                    for row in rows
                ),
                "success_metric_contradiction_events": sum(
                    int(row.get("success_metric_contradiction_events", 0) or 0)
                    for row in rows
                ),
                "diagnostic_support_events": sum(
                    int(row.get("diagnostic_support_events", 0) or 0)
                    for row in rows
                ),
                "diagnostic_contradiction_events": sum(
                    int(row.get("diagnostic_contradiction_events", 0) or 0)
                    for row in rows
                ),
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "controlled_test_required": True,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                "rule_counted_as_confirmation": False,
            }
        )
    return tuple(summaries)


def build_selection_followup_results_payload(
    *,
    selection_followup_requests_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    resolutions: Sequence[Mapping[str, Any]],
    per_request: Sequence[Mapping[str, Any]],
    rule_summary: Sequence[Mapping[str, Any]],
    max_dynamic_args_per_request: int,
) -> Dict[str, Any]:
    success_metric_support_events = sum(
        int(row.get("support_events", 0) or 0)
        for row in experiments
        if row.get("metric_role") == "success_metric"
    )
    success_metric_contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0)
        for row in experiments
        if row.get("metric_role") == "success_metric"
    )
    diagnostic_support_events = sum(
        int(row.get("support_events", 0) or 0)
        for row in experiments
        if row.get("metric_role") == "diagnostic_metric"
    )
    diagnostic_contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0)
        for row in experiments
        if row.get("metric_role") == "diagnostic_metric"
    )
    resolved_arg_occurrences = [
        dict(args)
        for row in resolutions
        for args in row.get("resolved_target_action_args", []) or []
    ]
    unique_resolved_arg_keys = {_args_key(args) for args in resolved_arg_occurrences}
    return {
        "config": {
            "selection_followup_requests_path": str(selection_followup_requests_path),
            "environments_dir": str(environments_dir),
            "schema_version": "m3.dynamic_retarget_selection_followup_results.v1",
            "inputs_read": ["M3.15"],
            "artifacts_not_modified": [
                "M2",
                "M3.8",
                "M3.9",
                "M3.10",
                "M3.11",
                "M3.12",
                "M3.13",
                "M3.14",
                "M3.15",
                "A32",
                "A33",
            ],
            "max_dynamic_args_per_request": int(max_dynamic_args_per_request),
        },
        "summary": {
            "selection_followup_requests_consumed": len(requests),
            "selection_followup_requests_executed": len(
                {str(row.get("request_id", "")) for row in experiments}
            ),
            "controlled_experiments_run": sum(
                int(row.get("controlled_experiments_run", 0) or 0)
                for row in experiments
            ),
            "explicit_requests": len(
                [
                    request
                    for request in requests
                    if str(request.get("target_action_arg_policy", ""))
                    == EXPLICIT_RETARGET_ARG_POLICY
                ]
            ),
            "explicit_requests_available": len(
                [
                    row
                    for row in resolutions
                    if str(row.get("target_action_arg_policy", ""))
                    == EXPLICIT_RETARGET_ARG_POLICY
                    and row.get("resolved_target_action_args")
                ]
            ),
            "explicit_requests_blocked_unavailable": len(
                [
                    row
                    for row in resolutions
                    if str(row.get("blocked_reason", ""))
                    == "explicit_args_not_available_after_replay"
                ]
            ),
            "explicit_requests_blocked_excluded": len(
                [
                    row
                    for row in resolutions
                    if str(row.get("blocked_reason", ""))
                    == "explicit_args_excluded_known_arg"
                ]
            ),
            "dynamic_arg_resolution_requests": len(
                [
                    request
                    for request in requests
                    if str(request.get("target_action_arg_policy", ""))
                    == LOCAL_PATCH_SIMILARITY_POLICY
                ]
            ),
            "dynamic_arg_resolution_requests_resolved": len(
                [
                    row
                    for row in resolutions
                    if str(row.get("target_action_arg_policy", ""))
                    == LOCAL_PATCH_SIMILARITY_POLICY
                    and row.get("resolved_target_action_args")
                ]
            ),
            "resolved_request_arg_pairs": len(resolved_arg_occurrences),
            "unique_resolved_target_arg_sets": len(unique_resolved_arg_keys),
            "resolved_target_arg_sets": len(
                unique_resolved_arg_keys
            ),
            "duplicate_resolved_target_arg_sets": max(
                0,
                len(resolved_arg_occurrences) - len(unique_resolved_arg_keys),
            ),
            "duplicate_resolved_target_arg_sets_counted_as_independent": False,
            "success_metric_support_events": success_metric_support_events,
            "success_metric_contradiction_events": success_metric_contradiction_events,
            "diagnostic_support_events": diagnostic_support_events,
            "diagnostic_contradiction_events": diagnostic_contradiction_events,
            "neutral_events": sum(
                int(row.get("neutral_events", 0) or 0) for row in experiments
            ),
            "blocked_experiments": len(blocked),
            "status": "UNRESOLVED",
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "controlled_test_required": True,
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "a32_remains_only_verdict_location": True,
        },
        "target_arg_resolutions": [dict(row) for row in resolutions],
        "controlled_experiments": [dict(row) for row in experiments],
        "per_request_results": [dict(row) for row in per_request],
        "rule_family_summary": [dict(row) for row in rule_summary],
        "blocked_experiments": [dict(row) for row in blocked],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
        "rule_counted_as_confirmation": False,
        "followup_request_counted_as_support": False,
    }


def blocked_selection_row(
    request: Mapping[str, Any],
    *,
    reason: str,
    resolution_rows: Sequence[Mapping[str, Any]],
    target_action_args: Mapping[str, Any] | None = None,
    metric: str = "",
    control_action: str = "",
) -> Dict[str, Any]:
    row = blocked_followup_row(
        request,
        reason=reason,
        target_action_args=target_action_args,
        target_action_arg_policy=str(request.get("target_action_arg_policy", "")),
        metric=metric,
        control_action=control_action,
    )
    row.update(
        {
            "source_selection_rule_candidate_id": str(
                request.get("source_selection_rule_candidate_id", "")
            ),
            "source_mechanism_candidate_id": str(
                request.get("source_mechanism_candidate_id", "")
            ),
            "source_rule_id": str(request.get("source_rule_id", "")),
            "rule_family": str(request.get("rule_family", "")),
            "probe_family": str(request.get("probe_family", "")),
            "target_arg_resolutions": [dict(item) for item in resolution_rows],
            "rule_counted_as_confirmation": False,
            "followup_request_counted_as_support": False,
        }
    )
    return row


def resolution_record(
    request: Mapping[str, Any],
    *,
    policy: str,
    resolution_basis: str,
    resolved_args: Sequence[Mapping[str, Any]],
    candidate_args: Sequence[Mapping[str, Any]],
    excluded_known_args_respected: bool,
    exact_explicit_args_available: bool | None = None,
    blocked_reason: str | None = None,
) -> Dict[str, Any]:
    excluded_guard_triggered = (
        str(blocked_reason or "") == "explicit_args_excluded_known_arg"
    )
    return {
        "request_id": str(request.get("request_id", "")),
        "source_selection_rule_candidate_id": str(
            request.get("source_selection_rule_candidate_id", "")
        ),
        "source_mechanism_candidate_id": str(
            request.get("source_mechanism_candidate_id", "")
        ),
        "source_rule_id": str(request.get("source_rule_id", "")),
        "rule_family": str(request.get("rule_family", "")),
        "probe_family": str(request.get("probe_family", "")),
        "target_action": str(request.get("target_action", "")),
        "target_action_arg_policy": policy,
        "resolution_basis": resolution_basis,
        "resolved_target_action_args": [dict(item) for item in resolved_args],
        "candidate_resolution_args": [dict(item) for item in candidate_args],
        "excluded_args": [dict(item) for item in effective_excluded_args(request)],
        "excluded_known_args_respected": bool(excluded_known_args_respected),
        "candidate_arg_was_excluded": bool(excluded_guard_triggered),
        "excluded_known_args_guard_triggered": bool(excluded_guard_triggered),
        "exact_explicit_args_available": exact_explicit_args_available,
        "blocked_reason": blocked_reason,
        "status": "RESOLVED" if resolved_args else "BLOCKED_NOT_RESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def patch_resolution_basis(request: Mapping[str, Any]) -> str:
    if request.get("seed_successful_args"):
        return "nearest_patch_signature_to_success_seeds"
    if request.get("seed_failed_args"):
        return "nearest_patch_signature_to_failure_seeds"
    return "nearest_patch_signature_to_available_seed"


def effective_excluded_args(request: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    values = [dict(item) for item in request.get("excluded_args", []) or []]
    target_action = str(request.get("target_action", ""))
    replay = list(request.get("context_replay", []) or [])
    replay_args = list(request.get("context_replay_args", []) or [])
    for index, action_name in enumerate(replay):
        if str(action_name) != target_action:
            continue
        if index < len(replay_args) and isinstance(replay_args[index], Mapping):
            values.append(dict(replay_args[index]))
    unique = {_args_key(args): args for args in values if args}
    return tuple(unique[key] for key in sorted(unique))


def sorted_args(values: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    unique = {_args_key(value): dict(value) for value in values}
    return [unique[key] for key in sorted(unique)]


def write_dynamic_retarget_selection_followup_results(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _args_key(args: Mapping[str, Any]) -> str:
    return json.dumps({str(key): args[key] for key in sorted(args)}, sort_keys=True)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute M3.16 dynamic retarget selection follow-ups.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--max-dynamic-args-per-request",
        type=int,
        default=DEFAULT_MAX_DYNAMIC_ARGS_PER_REQUEST,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_FOLLOWUP_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_selection_followup_execution(
        selection_followup_requests_path=args.requests,
        environments_dir=args.environments_dir,
        max_dynamic_args_per_request=args.max_dynamic_args_per_request,
    )
    write_dynamic_retarget_selection_followup_results(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
