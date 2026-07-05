"""M3.23 executor for patch-similarity follow-ups requested by A32."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .a32_requested_patch_similarity_followup_planner import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    READY_FOR_M3_A32_PATCH_FOLLOWUP,
)
from .dynamic_retarget_patch_similarity_expansion_planner import (
    PATCH_SIMILARITY_EXPANSION_POLICY,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .refined_followup_executor import (
    available_followup_controls,
    blocked_followup_row,
    execute_metric_followup_experiment,
)


DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "a32_requested_patch_similarity_followup_results.json"
)

OUTSIDE_KNOWN_Y12_REGION_PROBE = "outside_known_y12_region_probe"
ALTERNATE_REPOSITIONING_CONTEXT_PROBE = "alternate_repositioning_context_probe"


def run_a32_requested_patch_similarity_followup_execution(
    *,
    a32_followup_requests_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
) -> Dict[str, Any]:
    payload = _load_json(a32_followup_requests_path)
    requests = [
        dict(request)
        for request in payload.get("a32_requested_followup_requests", []) or []
        if str(request.get("status", "")) == READY_FOR_M3_A32_PATCH_FOLLOWUP
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    signatures = unique_a32_followup_execution_signatures(requests)
    experiments: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    provenance = [provenance_rationale(signature) for signature in signatures]

    for signature in signatures:
        request = canonical_request_for_signature(signature)
        target_args = dict(signature.get("target_action_args", {}) or {})
        controls = available_followup_controls(request, environments_dir=env_dir)
        if not controls:
            blocked.append(
                blocked_a32_followup_row(
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
                        blocked_a32_followup_row(
                            signature,
                            reason=f"execution_failed:{exc}",
                            target_action_args=target_args,
                            metric=str(metric),
                            control_action=str(control_action),
                        )
                    )
                    continue
                row.update(a32_followup_experiment_metadata(signature, metric))
                experiments.append(row)

    per_signature = per_signature_execution_results(experiments, blocked)
    family_summary = build_family_summary(per_signature)
    return build_a32_followup_results_payload(
        a32_followup_requests_path=a32_followup_requests_path,
        environments_dir=env_dir,
        requests=requests,
        signatures=signatures,
        provenance=provenance,
        experiments=experiments,
        blocked=blocked,
        per_signature=per_signature,
        family_summary=family_summary,
    )


def unique_a32_followup_execution_signatures(
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
                    "source_a32_queue_item_id": str(
                        request.get("source_a32_queue_item_id", "")
                    ),
                    "source_a32_decision": str(
                        request.get("source_a32_decision", "")
                    ),
                    "source_a32_recommended_next_step": str(
                        request.get("source_a32_recommended_next_step", "")
                    ),
                    "source_a32_decision_reasons": [],
                    "candidate_rule_family": str(
                        request.get("candidate_rule_family", "")
                    ),
                    "request_rationales": [],
                    "followup_families": [],
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
            extend_unique_strings(
                signature["source_a32_decision_reasons"],
                request.get("source_a32_decision_reasons", []) or [],
            )
            family = str(request.get("followup_family", ""))
            if family and family not in signature["followup_families"]:
                signature["followup_families"].append(family)
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
        value["followup_families"] = sorted(value.get("followup_families", []) or [])
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
    source_request_ids = [
        str(row.get("request_id", "")) for row in rationales if row.get("request_id")
    ]
    first_request_id = source_request_ids[0] if source_request_ids else ""
    return {
        "request_id": execution_request_id(signature),
        "source_request_ids": source_request_ids,
        "source_first_request_id": first_request_id,
        "source_a32_queue_item_id": str(
            signature.get("source_a32_queue_item_id", "")
        ),
        "source_a32_decision": str(signature.get("source_a32_decision", "")),
        "source_a32_recommended_next_step": str(
            signature.get("source_a32_recommended_next_step", "")
        ),
        "source_a32_decision_reasons": list(
            signature.get("source_a32_decision_reasons", []) or []
        ),
        "a32_decision_counted_as_confirmation": False,
        "game_id": str(signature.get("game_id", "")),
        "hypothesis_tested": (
            "A32 requested patch-similarity scope follow-up tested once "
            "per unique experimental signature"
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
        "candidate_rule_family": str(signature.get("candidate_rule_family", "")),
        "followup_families": list(signature.get("followup_families", []) or []),
    }


def a32_followup_experiment_metadata(
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
    followup_families = list(signature.get("followup_families", []) or [])
    return {
        "execution_signature": str(signature.get("execution_signature", "")),
        "source_a32_queue_item_id": str(
            signature.get("source_a32_queue_item_id", "")
        ),
        "source_a32_decision": str(signature.get("source_a32_decision", "")),
        "source_a32_recommended_next_step": str(
            signature.get("source_a32_recommended_next_step", "")
        ),
        "source_a32_decision_reasons": list(
            signature.get("source_a32_decision_reasons", []) or []
        ),
        "a32_decision_counted_as_confirmation": False,
        "candidate_rule_family": str(signature.get("candidate_rule_family", "")),
        "followup_families": followup_families,
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
        "outside_boundary_failure_counted_as_rule_refutation": False,
        "diagnostic_contradictions_counted_as_refutation": False,
        "a32_decision_counted_as_support": False,
        "observation_counted_as_confirmation": False,
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
        success_support = sum(
            int(row.get("support_events", 0) or 0) for row in success_rows
        )
        success_contradictions = sum(
            int(row.get("contradiction_events", 0) or 0) for row in success_rows
        )
        diagnostic_contradictions = sum(
            int(row.get("contradiction_events", 0) or 0) for row in diagnostic_rows
        )
        followup_families = list(first.get("followup_families", []) or [])
        results.append(
            {
                "execution_signature": key,
                "source_a32_queue_item_id": str(
                    first.get("source_a32_queue_item_id", "")
                ),
                "source_a32_decision": str(first.get("source_a32_decision", "")),
                "source_a32_recommended_next_step": str(
                    first.get("source_a32_recommended_next_step", "")
                ),
                "source_a32_decision_reasons": list(
                    first.get("source_a32_decision_reasons", []) or []
                ),
                "a32_decision_counted_as_confirmation": False,
                "candidate_rule_family": str(first.get("candidate_rule_family", "")),
                "followup_families": followup_families,
                "scope_probe_role": scope_probe_role(followup_families),
                "scope_interpretation": scope_interpretation(
                    followup_families,
                    success_support=success_support,
                    success_contradictions=success_contradictions,
                ),
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
                "success_metric_support_events": success_support,
                "success_metric_contradiction_events": success_contradictions,
                "diagnostic_support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in diagnostic_rows
                ),
                "diagnostic_contradiction_events": diagnostic_contradictions,
                "diagnostic_contradictions_counted_as_refutation": False,
                "grounded_success_metrics": sorted(
                    {
                        str(row.get("metric", ""))
                        for row in success_rows
                        if int(row.get("support_events", 0) or 0) > 0
                        and not bool(row.get("diagnostic_only"))
                    }
                ),
                "signature_has_success_metric_support": success_support > 0,
                "signature_has_success_metric_contradiction": (
                    success_contradictions > 0
                ),
                "outside_boundary_failure_counted_as_rule_refutation": False,
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "controlled_test_required": True,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                "observation_counted_as_confirmation": False,
                "rule_counted_as_confirmation": False,
                "a32_decision_counted_as_support": False,
            }
        )
    for row in blocked:
        results.append(blocked_signature_result(row))
    return tuple(results)


def blocked_signature_result(row: Mapping[str, Any]) -> Dict[str, Any]:
    followup_families = list(row.get("followup_families", []) or [])
    return {
        "execution_signature": str(row.get("execution_signature", "")),
        "source_a32_queue_item_id": str(row.get("source_a32_queue_item_id", "")),
        "source_a32_decision": str(row.get("source_a32_decision", "")),
        "source_a32_recommended_next_step": str(
            row.get("source_a32_recommended_next_step", "")
        ),
        "a32_decision_counted_as_confirmation": False,
        "candidate_rule_family": str(row.get("candidate_rule_family", "")),
        "followup_families": followup_families,
        "scope_probe_role": scope_probe_role(followup_families),
        "scope_interpretation": "not_executed_candidate_only",
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
        "diagnostic_contradictions_counted_as_refutation": False,
        "grounded_success_metrics": [],
        "signature_has_success_metric_support": False,
        "signature_has_success_metric_contradiction": False,
        "outside_boundary_failure_counted_as_rule_refutation": False,
        "status": "BLOCKED_NOT_EXECUTED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "observation_counted_as_confirmation": False,
        "rule_counted_as_confirmation": False,
        "a32_decision_counted_as_support": False,
    }


def build_family_summary(
    per_signature: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_family: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in per_signature:
        for family in row.get("followup_families", []) or []:
            by_family[str(family)].append(dict(row))

    summaries = []
    for family, rows in sorted(by_family.items()):
        executed = [
            row for row in rows if str(row.get("status", "")) != "BLOCKED_NOT_EXECUTED"
        ]
        args_with_support = [
            row
            for row in executed
            if int(row.get("success_metric_support_events", 0) or 0) > 0
        ]
        args_with_success_contradiction = [
            row
            for row in executed
            if int(row.get("success_metric_contradiction_events", 0) or 0) > 0
        ]
        summaries.append(
            {
                "followup_family": family,
                "family_role": family_role(family),
                "family_interpretation": family_interpretation(
                    family,
                    args_with_support=len(args_with_support),
                    args_with_success_contradiction=len(
                        args_with_success_contradiction
                    ),
                    executed_args=len(executed),
                ),
                "signatures_seen": len(rows),
                "signatures_executed": len(executed),
                "blocked_signatures": len(rows) - len(executed),
                "target_args_tested": [
                    dict(row.get("target_action_args", {}) or {}) for row in executed
                ],
                "args_with_success_metric_support": len(args_with_support),
                "args_with_success_metric_contradiction": len(
                    args_with_success_contradiction
                ),
                "controlled_experiments_run": sum(
                    int(row.get("controlled_experiments_run", 0) or 0)
                    for row in executed
                ),
                "success_metric_support_events": sum(
                    int(row.get("success_metric_support_events", 0) or 0)
                    for row in executed
                ),
                "success_metric_contradiction_events": sum(
                    int(row.get("success_metric_contradiction_events", 0) or 0)
                    for row in executed
                ),
                "diagnostic_support_events": sum(
                    int(row.get("diagnostic_support_events", 0) or 0)
                    for row in executed
                ),
                "diagnostic_contradiction_events": sum(
                    int(row.get("diagnostic_contradiction_events", 0) or 0)
                    for row in executed
                ),
                "outside_boundary_failures_counted_as_rule_refutation": False,
                "diagnostic_contradictions_counted_as_refutation": False,
                "a32_decision_counted_as_confirmation": False,
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "controlled_test_required": True,
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
            }
        )
    return tuple(summaries)


def build_a32_followup_results_payload(
    *,
    a32_followup_requests_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    signatures: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    per_signature: Sequence[Mapping[str, Any]],
    family_summary: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    success_rows = [row for row in experiments if row.get("metric_role") == "success_metric"]
    diagnostic_rows = [
        row for row in experiments if row.get("metric_role") == "diagnostic_metric"
    ]
    executed_signatures = [
        row
        for row in per_signature
        if str(row.get("status", "")) != "BLOCKED_NOT_EXECUTED"
    ]
    alternate_scope_expansion_candidates = [
        row
        for row in executed_signatures
        if ALTERNATE_REPOSITIONING_CONTEXT_PROBE
        in (row.get("followup_families", []) or [])
        and int(row.get("success_metric_support_events", 0) or 0) > 0
    ]
    return {
        "config": {
            "a32_followup_requests_path": str(a32_followup_requests_path),
            "environments_dir": str(environments_dir),
            "schema_version": "m3.a32_requested_patch_similarity_followup_results.v1",
            "inputs_read": ["M3.22"],
            "artifacts_not_modified": ["A32", "A33"],
            "execution_policy": (
                "execute_once_per_unique_a32_followup_experimental_signature"
            ),
            "target_action_arg_policy": PATCH_SIMILARITY_EXPANSION_POLICY,
            "family_interpretation_policy": {
                OUTSIDE_KNOWN_Y12_REGION_PROBE: (
                    "boundary_or_negative_control_probe_not_global_rule_refutation"
                ),
                ALTERNATE_REPOSITIONING_CONTEXT_PROBE: (
                    "scope_generalization_probe"
                ),
            },
        },
        "summary": {
            "a32_followup_requests_consumed": len(requests),
            "unique_execution_signatures": len(signatures),
            "unique_target_arg_sets_executed": len(
                {
                    _args_key(row.get("target_action_args", {}) or {})
                    for row in executed_signatures
                }
            ),
            "outside_known_y12_region_signatures": count_signatures_for_family(
                per_signature,
                OUTSIDE_KNOWN_Y12_REGION_PROBE,
            ),
            "alternate_repositioning_context_signatures": count_signatures_for_family(
                per_signature,
                ALTERNATE_REPOSITIONING_CONTEXT_PROBE,
            ),
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
            "alternate_context_args_with_success_metric_support": len(
                alternate_scope_expansion_candidates
            ),
            "outside_boundary_failures_counted_as_rule_refutation": False,
            "diagnostic_contradictions_counted_as_refutation": False,
            "a32_decision_counted_as_confirmation": False,
            "a32_decision_counted_as_support": False,
            "execution_performed": bool(experiments),
            "a32_write_performed": False,
            "a33_write_performed": False,
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
        "family_summary": [dict(row) for row in family_summary],
        "provenance_rationales": [dict(row) for row in provenance],
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
        "a32_decision_counted_as_confirmation": False,
        "a32_decision_counted_as_support": False,
    }


def count_signatures_for_family(
    per_signature: Sequence[Mapping[str, Any]],
    family: str,
) -> int:
    return len(
        [
            row
            for row in per_signature
            if family in (row.get("followup_families", []) or [])
        ]
    )


def scope_probe_role(followup_families: Sequence[str]) -> str:
    families = set(followup_families)
    if families == {OUTSIDE_KNOWN_Y12_REGION_PROBE}:
        return "boundary_or_negative_control_probe"
    if families == {ALTERNATE_REPOSITIONING_CONTEXT_PROBE}:
        return "context_generalization_probe"
    if OUTSIDE_KNOWN_Y12_REGION_PROBE in families:
        return "mixed_probe_including_boundary_check"
    return "scope_followup_probe"


def scope_interpretation(
    followup_families: Sequence[str],
    *,
    success_support: int,
    success_contradictions: int,
) -> str:
    families = set(followup_families)
    has_support = success_support > 0
    has_contradiction = success_contradictions > 0
    if families == {OUTSIDE_KNOWN_Y12_REGION_PROBE}:
        if has_support and not has_contradiction:
            return "outside_region_scope_expansion_candidate_only"
        if has_contradiction and not has_support:
            return "outside_region_boundary_reinforced_candidate_only"
        if has_support and has_contradiction:
            return "outside_region_mixed_boundary_signal_candidate_only"
        return "outside_region_neutral_boundary_probe_candidate_only"
    if families == {ALTERNATE_REPOSITIONING_CONTEXT_PROBE}:
        if has_support and not has_contradiction:
            return "alternate_context_scope_expanded_candidate_only"
        if has_support and has_contradiction:
            return "alternate_context_mixed_scope_signal_candidate_only"
        if has_contradiction:
            return "alternate_context_scope_not_supported_candidate_only"
        return "alternate_context_neutral_candidate_only"
    if has_support:
        return "mixed_scope_followup_support_candidate_only"
    if has_contradiction:
        return "mixed_scope_followup_boundary_candidate_only"
    return "mixed_scope_followup_neutral_candidate_only"


def family_role(family: str) -> str:
    if family == OUTSIDE_KNOWN_Y12_REGION_PROBE:
        return "boundary_negative_control_not_global_refutation"
    if family == ALTERNATE_REPOSITIONING_CONTEXT_PROBE:
        return "context_generalization_probe"
    return "scope_followup_probe"


def family_interpretation(
    family: str,
    *,
    args_with_support: int,
    args_with_success_contradiction: int,
    executed_args: int,
) -> str:
    if executed_args == 0:
        return "not_executed_candidate_only"
    if family == OUTSIDE_KNOWN_Y12_REGION_PROBE:
        if args_with_support > 0:
            return "outside_known_region_showed_candidate_scope_leak"
        if args_with_success_contradiction > 0:
            return "outside_known_region_boundary_reinforced_candidate_only"
        return "outside_known_region_not_expanded_candidate_only"
    if family == ALTERNATE_REPOSITIONING_CONTEXT_PROBE:
        if args_with_support > 0 and args_with_success_contradiction == 0:
            return "alternate_context_scope_expanded_candidate_only"
        if args_with_support > 0:
            return "alternate_context_mixed_scope_signal_candidate_only"
        if args_with_success_contradiction > 0:
            return "alternate_context_not_supported_candidate_only"
        return "alternate_context_neutral_candidate_only"
    if args_with_support > 0:
        return "scope_followup_support_candidate_only"
    if args_with_success_contradiction > 0:
        return "scope_followup_contradiction_candidate_only"
    return "scope_followup_neutral_candidate_only"


def provenance_rationale(signature: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "execution_signature": str(signature.get("execution_signature", "")),
        "target_action": str(signature.get("target_action", "")),
        "target_action_args": dict(signature.get("target_action_args", {}) or {}),
        "source_a32_decision": str(signature.get("source_a32_decision", "")),
        "source_a32_recommended_next_step": str(
            signature.get("source_a32_recommended_next_step", "")
        ),
        "a32_decision_counted_as_confirmation": False,
        "request_rationales": [
            dict(item) for item in signature.get("request_rationales", []) or []
        ],
        "followup_families": list(signature.get("followup_families", []) or []),
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
        "followup_family": str(request.get("followup_family", "")),
        "purpose": str(request.get("purpose", "")),
        "planning_rationale": str(request.get("planning_rationale", "")),
        "resolution_basis": str(request.get("resolution_basis", "")),
        "expected_signal": str(request.get("expected_signal", "")),
        "falsification_criterion": str(request.get("falsification_criterion", "")),
        "source_a32_decision": str(request.get("source_a32_decision", "")),
        "a32_decision_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def blocked_a32_followup_row(
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
            "source_a32_queue_item_id": str(
                signature.get("source_a32_queue_item_id", "")
            ),
            "source_a32_decision": str(signature.get("source_a32_decision", "")),
            "source_a32_recommended_next_step": str(
                signature.get("source_a32_recommended_next_step", "")
            ),
            "a32_decision_counted_as_confirmation": False,
            "candidate_rule_family": str(signature.get("candidate_rule_family", "")),
            "followup_families": list(signature.get("followup_families", []) or []),
            "duplicate_request_rationales_preserved": int(
                signature.get("duplicate_request_rationales_preserved", 0) or 0
            ),
            "duplicate_request_rationales_counted_as_independent": False,
            "outside_boundary_failure_counted_as_rule_refutation": False,
            "diagnostic_contradictions_counted_as_refutation": False,
            "rule_counted_as_confirmation": False,
            "a32_decision_counted_as_support": False,
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
    queue = str(signature.get("source_a32_queue_item_id", ""))
    queue_token = queue.replace("::", "_") or "unknown_queue"
    return f"m3_23::{queue_token}::{signature.get('target_action', '')}_{args_token}"


def extend_unique_strings(target: list[str], values: Sequence[Any]) -> None:
    for value in values:
        text = str(value)
        if text and text not in target:
            target.append(text)


def extend_unique_args(
    target: list[Dict[str, Any]],
    values: Sequence[Mapping[str, Any]],
) -> None:
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


def write_a32_requested_patch_similarity_followup_results(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH
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
        description="Execute M3.23 follow-ups requested by A32 patch review.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_requested_patch_similarity_followup_execution(
        a32_followup_requests_path=args.requests,
        environments_dir=args.environments_dir,
    )
    write_a32_requested_patch_similarity_followup_results(payload, args.out)
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
