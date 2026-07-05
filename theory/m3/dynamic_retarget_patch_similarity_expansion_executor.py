"""M3.19 executor for patch-similarity expansion requests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .dynamic_retarget_patch_similarity_expansion_planner import (
    DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH,
    PATCH_SIMILARITY_EXPANSION_POLICY,
    READY_FOR_M3_PATCH_EXPANSION,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .refined_followup_executor import (
    available_followup_controls,
    blocked_followup_row,
    execute_metric_followup_experiment,
)


DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "dynamic_retarget_patch_similarity_expansion_results.json"
)


def run_dynamic_retarget_patch_similarity_expansion_execution(
    *,
    expansion_requests_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
) -> Dict[str, Any]:
    payload = _load_json(expansion_requests_path)
    requests = [
        dict(request)
        for request in payload.get("expansion_experiment_requests", []) or []
        if str(request.get("status", "")) == READY_FOR_M3_PATCH_EXPANSION
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    signatures = unique_execution_signatures(requests)
    experiments: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    provenance = [provenance_rationale(row) for row in signatures]
    for signature in signatures:
        request = canonical_request_for_signature(signature)
        target_args = dict(signature.get("target_action_args", {}) or {})
        controls = available_followup_controls(request, environments_dir=env_dir)
        if not controls:
            blocked.append(
                blocked_patch_expansion_row(
                    signature,
                    reason="no_dynamic_control_available",
                    target_action_args=target_args,
                )
            )
            continue
        for metric in signature.get("metrics", []) or []:
            for control_action in controls:
                try:
                    row = execute_metric_followup_experiment(
                        request,
                        metric=str(metric),
                        control_action=str(control_action),
                        target_action_args=target_args,
                        target_action_arg_policy=PATCH_SIMILARITY_EXPANSION_POLICY,
                        environments_dir=env_dir,
                    )
                except Exception as exc:  # pragma: no cover - integration failure path
                    blocked.append(
                        blocked_patch_expansion_row(
                            signature,
                            reason=f"execution_failed:{exc}",
                            target_action_args=target_args,
                            metric=str(metric),
                            control_action=str(control_action),
                        )
                    )
                    continue
                row.update(patch_expansion_experiment_metadata(signature, metric))
                experiments.append(row)

    per_signature = per_signature_execution_results(experiments, blocked)
    expansion_summary = build_expansion_summary(per_signature)
    return build_patch_expansion_results_payload(
        expansion_requests_path=expansion_requests_path,
        environments_dir=env_dir,
        requests=requests,
        signatures=signatures,
        provenance=provenance,
        experiments=experiments,
        blocked=blocked,
        per_signature=per_signature,
        expansion_summary=expansion_summary,
    )


def unique_execution_signatures(
    requests: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_key: dict[str, Dict[str, Any]] = {}
    for request in requests:
        for target_args in resolved_args_for_request(request):
            key = execution_signature_key(request, target_args)
            if key not in by_key:
                by_key[key] = {
                    "execution_signature": key,
                    "game_id": str(request.get("game_id", "")),
                    "context_replay": list(request.get("context_replay", []) or []),
                    "context_replay_args": _context_args_list(
                        request.get("context_replay_args")
                    ),
                    "target_action": str(request.get("target_action", "")),
                    "target_action_args": dict(target_args),
                    "target_action_arg_policy": PATCH_SIMILARITY_EXPANSION_POLICY,
                    "suggested_control_actions": [],
                    "metrics": [],
                    "success_metrics": [],
                    "diagnostic_metrics": [],
                    "source_selection_rule_consolidation_id": str(
                        request.get("source_selection_rule_consolidation_id", "")
                    ),
                    "source_selection_rule_candidate_id": str(
                        request.get("source_selection_rule_candidate_id", "")
                    ),
                    "source_mechanism_candidate_id": str(
                        request.get("source_mechanism_candidate_id", "")
                    ),
                    "rule_family": str(request.get("rule_family", "")),
                    "request_rationales": [],
                    "candidate_resolution_args": [],
                    "excluded_args": [],
                    "seed_successful_args": [],
                    "seed_failed_args": [],
                }
            signature = by_key[key]
            extend_unique_strings(
                signature["suggested_control_actions"],
                request.get("suggested_control_actions", []) or [],
            )
            extend_unique_strings(signature["metrics"], request.get("metrics", []) or [])
            extend_unique_strings(
                signature["success_metrics"], request.get("success_metrics", []) or []
            )
            extend_unique_strings(
                signature["diagnostic_metrics"],
                request.get("diagnostic_metrics", []) or [],
            )
            signature["request_rationales"].append(request_rationale(request))
            extend_unique_args(
                signature["candidate_resolution_args"],
                [
                    dict(item)
                    for item in request.get("candidate_resolution_args", []) or []
                    if isinstance(item, Mapping)
                ],
            )
            extend_unique_args(
                signature["excluded_args"],
                [
                    dict(item)
                    for item in request.get("excluded_args", []) or []
                    if isinstance(item, Mapping)
                ],
            )
            extend_unique_args(
                signature["seed_successful_args"],
                [
                    dict(item)
                    for item in request.get("seed_successful_args", []) or []
                    if isinstance(item, Mapping)
                ],
            )
            extend_unique_args(
                signature["seed_failed_args"],
                [
                    dict(item)
                    for item in request.get("seed_failed_args", []) or []
                    if isinstance(item, Mapping)
                ],
            )
    signatures = []
    for key in sorted(by_key):
        value = by_key[key]
        value["duplicate_request_rationales_preserved"] = len(
            value["request_rationales"]
        )
        value["duplicate_request_rationales_counted_as_independent"] = False
        signatures.append(value)
    return tuple(signatures)


def resolved_args_for_request(request: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    resolved = [
        dict(item)
        for item in request.get("resolved_target_action_args", []) or []
        if isinstance(item, Mapping)
    ]
    if resolved:
        return tuple(dedupe_args(resolved))
    explicit = request.get("target_action_args")
    if isinstance(explicit, Mapping):
        return (dict(explicit),)
    return ()


def canonical_request_for_signature(signature: Mapping[str, Any]) -> Dict[str, Any]:
    rationales = list(signature.get("request_rationales", []) or [])
    first_request_id = str(rationales[0].get("request_id", "")) if rationales else ""
    return {
        "request_id": execution_request_id(signature),
        "source_selection_rule_consolidation_id": str(
            signature.get("source_selection_rule_consolidation_id", "")
        ),
        "source_selection_rule_candidate_id": str(
            signature.get("source_selection_rule_candidate_id", "")
        ),
        "source_mechanism_candidate_id": str(
            signature.get("source_mechanism_candidate_id", "")
        ),
        "source_request_ids": [
            str(row.get("request_id", "")) for row in rationales if row.get("request_id")
        ],
        "source_first_request_id": first_request_id,
        "game_id": str(signature.get("game_id", "")),
        "hypothesis_tested": (
            "patch-similarity expansion candidate tested once per unique "
            "experimental signature"
        ),
        "context_replay": list(signature.get("context_replay", []) or []),
        "context_replay_args": _context_args_list(signature.get("context_replay_args")),
        "target_action": str(signature.get("target_action", "")),
        "target_action_args": dict(signature.get("target_action_args", {}) or {}),
        "target_action_arg_policy": PATCH_SIMILARITY_EXPANSION_POLICY,
        "suggested_control_actions": list(
            signature.get("suggested_control_actions", []) or []
        ),
        "metrics": list(signature.get("metrics", []) or []),
        "success_metrics": list(signature.get("success_metrics", []) or []),
        "diagnostic_metrics": list(signature.get("diagnostic_metrics", []) or []),
        "rule_family": str(signature.get("rule_family", "")),
    }


def patch_expansion_experiment_metadata(
    signature: Mapping[str, Any],
    metric: str,
) -> Dict[str, Any]:
    success_metrics = {str(item) for item in signature.get("success_metrics", []) or []}
    diagnostic_metrics = {
        str(item) for item in signature.get("diagnostic_metrics", []) or []
    }
    metric_role = "success_metric" if metric in success_metrics else "diagnostic_metric"
    if metric not in success_metrics and metric not in diagnostic_metrics:
        metric_role = "unclassified_metric"
    return {
        "execution_signature": str(signature.get("execution_signature", "")),
        "source_selection_rule_consolidation_id": str(
            signature.get("source_selection_rule_consolidation_id", "")
        ),
        "source_selection_rule_candidate_id": str(
            signature.get("source_selection_rule_candidate_id", "")
        ),
        "source_mechanism_candidate_id": str(
            signature.get("source_mechanism_candidate_id", "")
        ),
        "rule_family": str(signature.get("rule_family", "")),
        "probe_families": sorted(
            {
                str(row.get("probe_family", ""))
                for row in signature.get("request_rationales", []) or []
                if row.get("probe_family")
            }
        ),
        "metric_role": metric_role,
        "success_metrics": list(signature.get("success_metrics", []) or []),
        "diagnostic_metrics": list(signature.get("diagnostic_metrics", []) or []),
        "candidate_resolution_args": [
            dict(item) for item in signature.get("candidate_resolution_args", []) or []
        ],
        "excluded_args": [
            dict(item) for item in signature.get("excluded_args", []) or []
        ],
        "seed_successful_args": [
            dict(item) for item in signature.get("seed_successful_args", []) or []
        ],
        "seed_failed_args": [
            dict(item) for item in signature.get("seed_failed_args", []) or []
        ],
        "provenance_request_ids": [
            str(row.get("request_id", ""))
            for row in signature.get("request_rationales", []) or []
            if row.get("request_id")
        ],
        "duplicate_request_rationales_preserved": int(
            signature.get("duplicate_request_rationales_preserved", 0) or 0
        ),
        "duplicate_request_rationales_counted_as_independent": False,
        "expansion_request_counted_as_support": False,
        "rule_counted_as_confirmation": False,
    }


def per_signature_execution_results(
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_key: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in experiments:
        by_key[str(row.get("execution_signature", ""))].append(dict(row))

    results: list[Dict[str, Any]] = []
    for key, rows in sorted(by_key.items()):
        first = rows[0]
        success_rows = [row for row in rows if row.get("metric_role") == "success_metric"]
        diagnostic_rows = [
            row for row in rows if row.get("metric_role") == "diagnostic_metric"
        ]
        results.append(
            {
                "execution_signature": key,
                "source_selection_rule_consolidation_id": str(
                    first.get("source_selection_rule_consolidation_id", "")
                ),
                "source_selection_rule_candidate_id": str(
                    first.get("source_selection_rule_candidate_id", "")
                ),
                "source_mechanism_candidate_id": str(
                    first.get("source_mechanism_candidate_id", "")
                ),
                "rule_family": str(first.get("rule_family", "")),
                "probe_families": list(first.get("probe_families", []) or []),
                "provenance_request_ids": list(
                    first.get("provenance_request_ids", []) or []
                ),
                "duplicate_request_rationales_preserved": int(
                    first.get("duplicate_request_rationales_preserved", 0) or 0
                ),
                "duplicate_request_rationales_counted_as_independent": False,
                "game_id": str(first.get("game_id", "")),
                "context_replay": list(first.get("context_replay", []) or []),
                "context_replay_args": _context_args_list(
                    first.get("context_replay_args")
                ),
                "target_action": str(first.get("target_action", "")),
                "target_action_args": dict(first.get("target_action_args", {}) or {}),
                "target_action_arg_policy": PATCH_SIMILARITY_EXPANSION_POLICY,
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
                "signature_has_success_metric_support": any(
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
                "observation_counted_as_confirmation": False,
                "rule_counted_as_confirmation": False,
                "expansion_request_counted_as_support": False,
            }
        )
    for row in blocked:
        results.append(blocked_signature_result(row))
    return tuple(results)


def blocked_signature_result(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "execution_signature": str(row.get("execution_signature", "")),
        "target_action": str(row.get("target_action", "")),
        "target_action_args": (
            dict(row.get("target_action_args", {}) or {})
            if row.get("target_action_args") is not None
            else None
        ),
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
        "signature_has_success_metric_support": False,
        "status": "BLOCKED_NOT_EXECUTED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "observation_counted_as_confirmation": False,
        "rule_counted_as_confirmation": False,
        "expansion_request_counted_as_support": False,
    }


def build_expansion_summary(
    per_signature: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    executed = [
        row
        for row in per_signature
        if str(row.get("status", "")) != "BLOCKED_NOT_EXECUTED"
    ]
    supported = [
        row for row in executed if bool(row.get("signature_has_success_metric_support"))
    ]
    return {
        "tested_unique_arg_sets": len(executed),
        "args_with_grounded_support": len(supported),
        "args_with_success_metric_contradictions": len(
            [
                row
                for row in executed
                if int(row.get("success_metric_contradiction_events", 0) or 0) > 0
            ]
        ),
        "args_neutral_only": len(
            [
                row
                for row in executed
                if int(row.get("support_events", 0) or 0) == 0
                and int(row.get("contradiction_events", 0) or 0) == 0
            ]
        ),
        "best_arg": (
            dict(supported[0].get("target_action_args", {}) or {})
            if supported
            else None
        ),
        "mechanism_support_events": 1 if supported else 0,
        "signature_level_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in executed
        ),
        "signature_level_support_events_counted_as_mechanism_support": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_remains_only_verdict_location": True,
    }


def build_patch_expansion_results_payload(
    *,
    expansion_requests_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    signatures: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    per_signature: Sequence[Mapping[str, Any]],
    expansion_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    success_rows = [row for row in experiments if row.get("metric_role") == "success_metric"]
    diagnostic_rows = [
        row for row in experiments if row.get("metric_role") == "diagnostic_metric"
    ]
    return {
        "config": {
            "expansion_requests_path": str(expansion_requests_path),
            "environments_dir": str(environments_dir),
            "schema_version": "m3.dynamic_retarget_patch_similarity_expansion_results.v1",
            "inputs_read": ["M3.18"],
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
                "M3.16",
                "M3.17",
                "M3.18",
                "A32",
                "A33",
            ],
            "execution_policy": "execute_once_per_unique_experimental_signature",
            "target_action_arg_policy": PATCH_SIMILARITY_EXPANSION_POLICY,
        },
        "summary": {
            "expansion_requests_consumed": len(requests),
            "unique_execution_signatures": len(signatures),
            "unique_target_arg_sets_executed": len(
                {
                    _args_key(row.get("target_action_args", {}) or {})
                    for row in per_signature
                    if str(row.get("status", "")) != "BLOCKED_NOT_EXECUTED"
                }
            ),
            "duplicate_request_rationales_preserved": sum(
                int(row.get("duplicate_request_rationales_preserved", 0) or 0)
                for row in signatures
            ),
            "duplicate_request_rationales_counted_as_independent": False,
            "controlled_experiments_run": sum(
                int(row.get("controlled_experiments_run", 0) or 0)
                for row in experiments
            ),
            "success_metric_support_events": sum(
                int(row.get("support_events", 0) or 0) for row in success_rows
            ),
            "success_metric_contradiction_events": sum(
                int(row.get("contradiction_events", 0) or 0) for row in success_rows
            ),
            "diagnostic_support_events": sum(
                int(row.get("support_events", 0) or 0) for row in diagnostic_rows
            ),
            "diagnostic_contradiction_events": sum(
                int(row.get("contradiction_events", 0) or 0)
                for row in diagnostic_rows
            ),
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
        "per_signature_execution": [dict(row) for row in per_signature],
        "provenance_rationales": [dict(row) for row in provenance],
        "expansion_summary": dict(expansion_summary),
        "controlled_experiments": [dict(row) for row in experiments],
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
        "expansion_request_counted_as_support": False,
    }


def provenance_rationale(signature: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "execution_signature": str(signature.get("execution_signature", "")),
        "target_action": str(signature.get("target_action", "")),
        "target_action_args": dict(signature.get("target_action_args", {}) or {}),
        "request_rationales": [
            dict(item) for item in signature.get("request_rationales", []) or []
        ],
        "probe_families": sorted(
            {
                str(item.get("probe_family", ""))
                for item in signature.get("request_rationales", []) or []
                if item.get("probe_family")
            }
        ),
        "duplicate_request_rationales_preserved": int(
            signature.get("duplicate_request_rationales_preserved", 0) or 0
        ),
        "duplicate_request_rationales_counted_as_independent": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def request_rationale(request: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "request_id": str(request.get("request_id", "")),
        "probe_family": str(request.get("probe_family", "")),
        "hypothesis_tested": str(request.get("hypothesis_tested", "")),
        "planning_rationale": str(request.get("planning_rationale", "")),
        "resolution_basis": str(request.get("resolution_basis", "")),
        "falsification_criterion": str(request.get("falsification_criterion", "")),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def blocked_patch_expansion_row(
    signature: Mapping[str, Any],
    *,
    reason: str,
    target_action_args: Mapping[str, Any] | None,
    metric: str = "",
    control_action: str = "",
) -> Dict[str, Any]:
    request = canonical_request_for_signature(signature)
    row = blocked_followup_row(
        request,
        reason=reason,
        target_action_args=target_action_args,
        target_action_arg_policy=PATCH_SIMILARITY_EXPANSION_POLICY,
        metric=metric,
        control_action=control_action,
    )
    row.update(
        {
            "execution_signature": str(signature.get("execution_signature", "")),
            "source_selection_rule_consolidation_id": str(
                signature.get("source_selection_rule_consolidation_id", "")
            ),
            "source_selection_rule_candidate_id": str(
                signature.get("source_selection_rule_candidate_id", "")
            ),
            "source_mechanism_candidate_id": str(
                signature.get("source_mechanism_candidate_id", "")
            ),
            "rule_family": str(signature.get("rule_family", "")),
            "probe_families": sorted(
                {
                    str(item.get("probe_family", ""))
                    for item in signature.get("request_rationales", []) or []
                    if item.get("probe_family")
                }
            ),
            "duplicate_request_rationales_preserved": int(
                signature.get("duplicate_request_rationales_preserved", 0) or 0
            ),
            "duplicate_request_rationales_counted_as_independent": False,
            "rule_counted_as_confirmation": False,
            "expansion_request_counted_as_support": False,
        }
    )
    return row


def execution_signature_key(
    request: Mapping[str, Any],
    target_args: Mapping[str, Any],
) -> str:
    return _stable_json(
        {
            "game_id": str(request.get("game_id", "")),
            "context_replay": list(request.get("context_replay", []) or []),
            "context_replay_args": _context_args_list(
                request.get("context_replay_args")
            ),
            "target_action": str(request.get("target_action", "")),
            "target_action_args": dict(target_args),
        }
    )


def execution_request_id(signature: Mapping[str, Any]) -> str:
    args = dict(signature.get("target_action_args", {}) or {})
    args_token = "_".join(f"{key}{args[key]}" for key in sorted(args)) or "noargs"
    source = str(signature.get("source_selection_rule_consolidation_id", ""))
    source_token = source.replace("::", "_") or "unknown_consolidation"
    return f"m3_19::{source_token}::{signature.get('target_action', '')}_{args_token}"


def extend_unique_strings(target: list[str], values: Sequence[Any]) -> None:
    for value in values:
        text = str(value)
        if text and text not in target:
            target.append(text)


def extend_unique_args(target: list[Dict[str, Any]], values: Sequence[Mapping[str, Any]]) -> None:
    seen = {_args_key(item) for item in target}
    for value in values:
        args = dict(value)
        key = _args_key(args)
        if key in seen:
            continue
        seen.add(key)
        target.append(args)


def dedupe_args(values: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    by_key = {_args_key(dict(value)): dict(value) for value in values if value}
    return tuple(by_key[key] for key in sorted(by_key))


def _args_key(args: Mapping[str, Any]) -> str:
    return _stable_json({str(key): args[key] for key in sorted(args)})


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _context_args_list(raw: Any) -> list[Dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def write_dynamic_retarget_patch_similarity_expansion_results(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute M3.19 patch-similarity expansion requests.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_patch_similarity_expansion_execution(
        expansion_requests_path=args.requests,
        environments_dir=args.environments_dir,
    )
    write_dynamic_retarget_patch_similarity_expansion_results(payload, args.out)
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
