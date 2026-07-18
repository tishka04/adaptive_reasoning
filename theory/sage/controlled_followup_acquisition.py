"""SAGE.5h controlled acquisition for SAGE.5g follow-up requests."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _valid_actions

from .a32_review_handoff import (
    DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH,
    a32_intake_recommendation,
    missing_revision_requirements,
)
from .distributed_live_mini_frontier_generation import (
    DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH,
)
from .live_mini_frontier_m3_executor import (
    _make_real_env,
    _prefix_actions_from_request,
    execute_live_prefix_mini_frontier_request,
)
from .live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from .mini_frontier_event_consolidation import (
    DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH,
)
from .policy_loop_guard import action_args, action_name


DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH = (
    Path("diagnostics") / "sage" / "sage5h_controlled_followup_acquisition.json"
)
SAGE5H_SCHEMA_VERSION = "sage.controlled_followup_acquisition.v1"
SAGE5H_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5H"
SAGE5H_COMPLETE = "SAGE_CONTROLLED_FOLLOWUP_ACQUISITION_COMPLETE_CANDIDATE_ONLY"
SAGE5H_PARTIAL_CONTROL_SURFACE_LIMIT = (
    "SAGE_CONTROLLED_FOLLOWUP_ACQUISITION_PARTIAL_CONTROL_SURFACE_LIMIT_"
    "CANDIDATE_ONLY"
)
SAGE5H_NO_ACQUISITION = "SAGE_CONTROLLED_FOLLOWUP_ACQUISITION_NONE_CANDIDATE_ONLY"

FOLLOWUP_ACQUIRED_CONTROL_DIVERSITY = "ACQUIRED_CONTROL_DIVERSITY_CANDIDATE_ONLY"
FOLLOWUP_ACQUIRED_COMPARABLE_SUPPORT = "ACQUIRED_COMPARABLE_SUPPORT_CANDIDATE_ONLY"
FOLLOWUP_ACQUIRED_CROSS_MEASUREMENT = (
    "ACQUIRED_CROSS_MEASUREMENT_ALIGNMENT_CANDIDATE_ONLY"
)
FOLLOWUP_ACQUIRED_CROSS_MEASUREMENT_DIVERGENCE = (
    "ACQUIRED_CROSS_MEASUREMENT_DIVERGENCE_CANDIDATE_ONLY"
)
FOLLOWUP_BLOCKED_CONTROL_SURFACE = "BLOCKED_NO_DISTINCT_LEGAL_CONTROL_ACTION"
FOLLOWUP_BLOCKED_NO_SOURCE_REQUEST = "BLOCKED_NO_SOURCE_REQUEST"
FOLLOWUP_BLOCKED_EXECUTION = "BLOCKED_EXECUTION"
FOLLOWUP_UNSUPPORTED = "BLOCKED_UNSUPPORTED_FOLLOWUP_TYPE"
UPDATED_RECOMMENDATION_CONTROL_SURFACE_EXHAUSTED = (
    "FOLLOWUP_BLOCKED_CONTROL_SURFACE_EXHAUSTED_CANDIDATE_ONLY"
)

CONTROL_DIVERSITY_REQUEST = "ACQUIRE_DISTINCT_CONTROL_ACTION"
SUPPORT_REQUEST = "ACQUIRE_ADDITIONAL_COMPARABLE_SUPPORT"
CROSS_MEASUREMENT_REQUEST = "CROSS_MEASURE_RELATED_NONMERGED_CLUSTER"

CROSS_MEASUREMENT_METRICS = {
    "local_patch": "local_patch_before_after",
    "object_delta": "object_counts_before_after",
}

EnvFactory = Callable[[str], Any]


def run_sage5h_controlled_followup_acquisition(
    *,
    source_sage5g_path: str | Path = DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH,
    source_sage5e_path: str | Path = (
        DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH
    ),
    source_sage5f_path: str | Path = (
        DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH
    ),
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Resolve SAGE.5g followups through exact replay or audited blocking."""
    source_sage5g = _load_json(source_sage5g_path)
    source_sage5e = _load_json(source_sage5e_path)
    source_sage5f = _load_json(source_sage5f_path)
    validate_sage5h_sources(source_sage5g, source_sage5e, source_sage5f)

    candidates = {
        str(row.get("candidate_id", "")): dict(row)
        for row in source_sage5g.get("a32_review_candidate_items", []) or []
        if isinstance(row, Mapping) and str(row.get("candidate_id", ""))
    }
    requests = {
        str(row.get("request_id", "")): dict(row)
        for row in source_sage5e.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping) and str(row.get("request_id", ""))
    }
    clusters = {
        str(row.get("cluster_id", "")): dict(row)
        for row in source_sage5f.get("candidate_mechanism_clusters", []) or []
        if isinstance(row, Mapping) and str(row.get("cluster_id", ""))
    }

    execution_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    experiments: List[Dict[str, Any]] = []
    control_surface_audits: List[Dict[str, Any]] = []
    outcomes: List[Dict[str, Any]] = []

    for followup in source_sage5g.get("requested_followups", []) or []:
        if not isinstance(followup, Mapping):
            continue
        candidate = candidates.get(str(followup.get("candidate_id", "")), {})
        request_type = str(followup.get("request_type", ""))
        if request_type == CONTROL_DIVERSITY_REQUEST:
            outcome = acquire_control_diversity_followup(
                followup=followup,
                candidate=candidate,
                source_requests=requests,
                environments_dir=environments_dir,
                env_factory=env_factory,
                execution_cache=execution_cache,
                experiments=experiments,
                control_surface_audits=control_surface_audits,
            )
        elif request_type == SUPPORT_REQUEST:
            outcome = acquire_comparable_support_followup(
                followup=followup,
                candidate=candidate,
                source_requests=requests,
                source_clusters=clusters,
                environments_dir=environments_dir,
                env_factory=env_factory,
                execution_cache=execution_cache,
                experiments=experiments,
            )
        elif request_type == CROSS_MEASUREMENT_REQUEST:
            outcome = acquire_cross_measurement_followup(
                followup=followup,
                candidate=candidate,
                source_requests=requests,
                source_clusters=clusters,
                environments_dir=environments_dir,
                env_factory=env_factory,
                execution_cache=execution_cache,
                experiments=experiments,
            )
        else:
            outcome = _base_followup_outcome(
                followup,
                completed=False,
                resolution_status=FOLLOWUP_UNSUPPORTED,
                blocked_reason="unsupported_followup_type",
            )
        outcomes.append(outcome)

    updated_assessments = update_candidate_assessments(
        candidates=tuple(candidates.values()),
        outcomes=outcomes,
    )
    executed = [
        row for row in experiments if str(row.get("execution_status", "")) == "EXECUTED"
    ]
    blocked = [
        row for row in experiments if str(row.get("execution_status", "")) != "EXECUTED"
    ]
    summary = summarize_sage5h(
        source_sage5g=source_sage5g,
        outcomes=outcomes,
        experiments=executed,
        blocked_experiments=blocked,
        control_surface_audits=control_surface_audits,
        updated_assessments=updated_assessments,
    )
    payload = {
        "config": {
            "schema_version": SAGE5H_SCHEMA_VERSION,
            "source_sage5g_path": str(source_sage5g_path),
            "source_sage5e_path": str(source_sage5e_path),
            "source_sage5f_path": str(source_sage5f_path),
            "environments_dir": (
                str(environments_dir) if environments_dir is not None else None
            ),
            "inputs_read": ["SAGE.5e", "SAGE.5f", "SAGE.5g"],
            "execution_performed": True,
            "acquisition_policy": {
                "exact_context_replay_required": True,
                "new_control_must_be_legal_and_action_distinct": True,
                "reset_is_not_a_scientific_control": True,
                "cross_measurement_clusters_remain_unmerged": True,
                "followup_events_are_not_scientific_support": True,
                "a32_write_performed": False,
                "a33_write_performed": False,
            },
            "cross_measurement_metrics": dict(CROSS_MEASUREMENT_METRICS),
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5g_summary": dict(source_sage5g.get("summary", {}) or {}),
        "followup_outcomes": outcomes,
        "control_surface_audits": control_surface_audits,
        "controlled_experiments": executed,
        "blocked_controlled_experiments": blocked,
        "updated_candidate_assessments": updated_assessments,
        "summary": summary,
        "comparison": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5H_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "followup_events_counted_as_scientific_support": False,
        "control_surface_block_counted_as_refutation": False,
        "cross_measurement_alignment_counted_as_confirmation": False,
        "candidate_assessment_counted_as_revision": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5h_controlled_followup_acquisition(payload, output_path)
    return payload


def acquire_control_diversity_followup(
    *,
    followup: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_requests: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
    execution_cache: MutableMapping[Tuple[str, str, str], Dict[str, Any]],
    experiments: List[Dict[str, Any]],
    control_surface_audits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    minimum_contexts = max(
        1,
        int(followup.get("minimum_replay_exact_contexts", 1) or 1),
    )
    excluded = {
        str(value)
        for value in followup.get("excluded_control_actions", []) or []
        if str(value)
    }
    target_action = str(followup.get("action", candidate.get("action", "")))
    source_ids = _candidate_source_request_ids(candidate)
    audits: List[Dict[str, Any]] = []
    acquired: List[Dict[str, Any]] = []
    selected_control = ""

    for source_id in source_ids:
        if len(audits) >= minimum_contexts:
            break
        request = source_requests.get(source_id)
        if request is None:
            continue
        audit = audit_context_control_surface(
            request=request,
            followup=followup,
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        audit["audit_id"] = f"sage5h::control_surface_audit::{len(control_surface_audits) + 1:03d}"
        audits.append(audit)
        control_surface_audits.append(audit)
        eligible = [
            str(value)
            for value in audit.get("eligible_distinct_control_actions", []) or []
            if str(value)
        ]
        if not selected_control and eligible:
            selected_control = eligible[0]
        if (
            selected_control
            and selected_control in eligible
            and bool(audit.get("live_prefix_replay_exact", False))
        ):
            result = execute_followup_experiment(
                source_request=request,
                metric=str(candidate.get("predicted_metric", request.get("metric", ""))),
                control_action=selected_control,
                followup_id=str(followup.get("followup_id", "")),
                purpose="control_diversity",
                environments_dir=environments_dir,
                env_factory=env_factory,
                execution_cache=execution_cache,
                experiments=experiments,
            )
            acquired.append(result)

    completed = bool(
        selected_control
        and len(
            [
                row
                for row in acquired
                if str(row.get("execution_status", "")) == "EXECUTED"
                and bool(row.get("live_prefix_replay_exact", False))
            ]
        )
        >= minimum_contexts
    )
    if completed:
        resolution_status = FOLLOWUP_ACQUIRED_CONTROL_DIVERSITY
        blocked_reason = ""
    elif audits and all(
        not (row.get("eligible_distinct_control_actions", []) or []) for row in audits
    ):
        resolution_status = FOLLOWUP_BLOCKED_CONTROL_SURFACE
        blocked_reason = "legal_action_surface_contains_no_new_action_distinct_control"
    else:
        resolution_status = FOLLOWUP_BLOCKED_EXECUTION
        blocked_reason = "control_diversity_execution_incomplete"
    return _base_followup_outcome(
        followup,
        completed=completed,
        resolution_status=resolution_status,
        blocked_reason=blocked_reason,
        control_surface_audit_ids=[str(row.get("audit_id", "")) for row in audits],
        contexts_audited=len(audits),
        acquired_control_actions=[selected_control] if completed else [],
        controlled_experiment_ids=[
            str(row.get("experiment_id", "")) for row in acquired
        ],
        replay_exact_experiments=sum(
            1 for row in acquired if bool(row.get("live_prefix_replay_exact", False))
        ),
    )


def acquire_comparable_support_followup(
    *,
    followup: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_requests: Mapping[str, Mapping[str, Any]],
    source_clusters: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
    execution_cache: MutableMapping[Tuple[str, str, str], Dict[str, Any]],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    current_ids = set(_candidate_source_request_ids(candidate))
    related_ids = [
        str(value)
        for value in candidate.get("related_nonmerged_cluster_ids", []) or []
    ]
    source_id = next(
        (
            request_id
            for cluster_id in related_ids
            for request_id in source_clusters.get(cluster_id, {}).get("request_ids", []) or []
            if str(request_id) not in current_ids and str(request_id) in source_requests
        ),
        "",
    )
    if not source_id:
        return _base_followup_outcome(
            followup,
            completed=False,
            resolution_status=FOLLOWUP_BLOCKED_NO_SOURCE_REQUEST,
            blocked_reason="no_related_nonmerged_context_available",
        )
    request = source_requests[source_id]
    result = execute_followup_experiment(
        source_request=request,
        metric=str(candidate.get("predicted_metric", request.get("metric", ""))),
        control_action="",
        followup_id=str(followup.get("followup_id", "")),
        purpose="comparable_support",
        environments_dir=environments_dir,
        env_factory=env_factory,
        execution_cache=execution_cache,
        experiments=experiments,
    )
    signature_match = _core_effect_signature_matches(candidate, request)
    acquired_support = int(result.get("support_events", 0) or 0) if signature_match else 0
    completed = bool(
        str(result.get("execution_status", "")) == "EXECUTED"
        and bool(result.get("live_prefix_replay_exact", False))
        and acquired_support > 0
    )
    return _base_followup_outcome(
        followup,
        completed=completed,
        resolution_status=(
            FOLLOWUP_ACQUIRED_COMPARABLE_SUPPORT
            if completed
            else FOLLOWUP_BLOCKED_EXECUTION
        ),
        blocked_reason=("" if completed else "comparable_support_not_acquired"),
        source_request_id=source_id,
        controlled_experiment_ids=[str(result.get("experiment_id", ""))],
        comparable_core_effect_signature=signature_match,
        acquired_raw_support_events=acquired_support,
        acquired_context_snapshot_hash=str(
            request.get("context_snapshot_hash", "")
        ),
        acquired_budget=_budget_from_request(request),
    )


def acquire_cross_measurement_followup(
    *,
    followup: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_requests: Mapping[str, Mapping[str, Any]],
    source_clusters: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
    execution_cache: MutableMapping[Tuple[str, str, str], Dict[str, Any]],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cluster_ids = [str(followup.get("source_cluster_id", ""))] + [
        str(value) for value in followup.get("related_cluster_ids", []) or []
    ]
    selected_requests: Dict[str, Mapping[str, Any]] = {}
    for cluster_id in cluster_ids:
        request_id = next(
            (
                str(value)
                for value in source_clusters.get(cluster_id, {}).get("request_ids", []) or []
                if str(value) in source_requests
            ),
            "",
        )
        if request_id:
            selected_requests[cluster_id] = source_requests[request_id]
    if len(selected_requests) != len(cluster_ids):
        return _base_followup_outcome(
            followup,
            completed=False,
            resolution_status=FOLLOWUP_BLOCKED_NO_SOURCE_REQUEST,
            blocked_reason="cross_measurement_cluster_request_missing",
        )

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cluster_id, request in selected_requests.items():
        results[cluster_id] = {}
        for label in followup.get("required_measurements", []) or []:
            metric = CROSS_MEASUREMENT_METRICS.get(str(label), "")
            if not metric:
                continue
            results[cluster_id][str(label)] = execute_followup_experiment(
                source_request=request,
                metric=metric,
                control_action="",
                followup_id=str(followup.get("followup_id", "")),
                purpose=f"cross_measurement:{label}",
                environments_dir=environments_dir,
                env_factory=env_factory,
                execution_cache=execution_cache,
                experiments=experiments,
            )

    flat_results = [row for by_metric in results.values() for row in by_metric.values()]
    all_exact = bool(flat_results) and all(
        str(row.get("execution_status", "")) == "EXECUTED"
        and bool(row.get("live_prefix_replay_exact", False))
        for row in flat_results
    )
    alignment = cross_measurement_alignment(results)
    completed = all_exact
    if completed and alignment.get("all_measurements_aligned", False):
        resolution_status = FOLLOWUP_ACQUIRED_CROSS_MEASUREMENT
    elif completed:
        resolution_status = FOLLOWUP_ACQUIRED_CROSS_MEASUREMENT_DIVERGENCE
    else:
        resolution_status = FOLLOWUP_BLOCKED_EXECUTION
    return _base_followup_outcome(
        followup,
        completed=completed,
        resolution_status=resolution_status,
        blocked_reason=("" if completed else "cross_measurement_incomplete"),
        source_cluster_ids=cluster_ids,
        representative_source_request_ids={
            cluster_id: str(request.get("request_id", ""))
            for cluster_id, request in selected_requests.items()
        },
        controlled_experiment_ids=[
            str(row.get("experiment_id", "")) for row in flat_results
        ],
        measurement_alignment=alignment,
        clusters_remained_unmerged=True,
    )


def audit_context_control_surface(
    *,
    request: Mapping[str, Any],
    followup: Mapping[str, Any],
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    game_id = str(request.get("game_id", ""))
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, environments_dir)
        )
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return _blocked_control_audit(request, f"env_setup_failed:{exc}")
    prefix_env_actions = 0
    for prefix_action in _prefix_actions_from_request(request):
        selected = select_live_action(
            env,
            prefix_action.name,
            action_args=prefix_action.action_args,
        )
        if selected is None:
            return _blocked_control_audit(
                request,
                f"prefix_action_unavailable:{prefix_action.name}",
            )
        frame = _step_env_action(env, selected)
        prefix_env_actions += 1
    replay_signature = state_signature_from_frame(frame)
    expected_signature = str(request.get("context_snapshot_hash", ""))
    replay_exact = replay_signature == expected_signature
    variants = [
        {
            "action": action_name(action),
            "action_args": dict(action_args(action) or {}),
        }
        for action in _valid_actions(env)
        if action_name(action)
    ]
    available_names = sorted(
        {
            str(row.get("action", ""))
            for row in variants
            if str(row.get("action", ""))
        }
    )
    target = str(followup.get("action", request.get("target_action", "")))
    excluded = {
        str(value)
        for value in followup.get("excluded_control_actions", []) or []
        if str(value)
    }
    eligible = sorted(
        name
        for name in available_names
        if name not in excluded and name not in {target, "RESET"}
    )
    return {
        "execution_status": "AUDITED" if replay_exact else "BLOCKED",
        "source_request_id": str(request.get("request_id", "")),
        "source_transition_id": str(request.get("source_transition_id", "")),
        "game_id": game_id,
        "budget": _budget_from_request(request),
        "context_snapshot_hash": expected_signature,
        "replay_state_signature": replay_signature,
        "live_prefix_replay_exact": replay_exact,
        "prefix_env_actions": prefix_env_actions,
        "target_action": target,
        "excluded_control_actions": sorted(excluded),
        "available_action_names": available_names,
        "available_action_variants": variants,
        "eligible_distinct_control_actions": eligible,
        "control_surface_exhausted": not eligible,
        "support": 0,
        "truth_status": SAGE5H_TRUTH_STATUS,
    }


def execute_followup_experiment(
    *,
    source_request: Mapping[str, Any],
    metric: str,
    control_action: str,
    followup_id: str,
    purpose: str,
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
    execution_cache: MutableMapping[Tuple[str, str, str], Dict[str, Any]],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved_control = control_action or _first_control_action(source_request)
    cache_key = (
        str(source_request.get("request_id", "")),
        str(metric),
        resolved_control,
    )
    if cache_key in execution_cache:
        cached = execution_cache[cache_key]
        _append_unique(cached["source_followup_ids"], followup_id)
        _append_unique(cached["acquisition_purposes"], purpose)
        return cached

    request = copy.deepcopy(dict(source_request))
    request["request_id"] = f"sage5h::controlled_request::{len(experiments) + 1:03d}"
    request["metric"] = str(metric)
    request["suggested_control_actions"] = [resolved_control] if resolved_control else []
    result = execute_live_prefix_mini_frontier_request(
        request,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    result.update(
        {
            "experiment_id": f"sage5h::controlled_experiment::{len(experiments) + 1:03d}",
            "source_request_id": str(source_request.get("request_id", "")),
            "source_followup_ids": [followup_id],
            "acquisition_purposes": [purpose],
            "source_truth_status": str(source_request.get("truth_status", "")),
            "truth_status": SAGE5H_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "support_events_counted_as_support": False,
            "observation_counted_as_confirmation": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        }
    )
    execution_cache[cache_key] = result
    experiments.append(result)
    return result


def cross_measurement_alignment(
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    local_signatures = {
        cluster_id: _local_patch_signature(by_metric.get("local_patch", {}))
        for cluster_id, by_metric in results.items()
    }
    object_signatures = {
        cluster_id: _object_delta_signature(by_metric.get("object_delta", {}))
        for cluster_id, by_metric in results.items()
    }
    local_aligned = _all_signatures_equal(local_signatures.values())
    object_aligned = _all_signatures_equal(object_signatures.values())
    return {
        "local_patch_signatures_by_cluster": local_signatures,
        "object_delta_signatures_by_cluster": object_signatures,
        "local_patch_aligned": local_aligned,
        "object_delta_aligned": object_aligned,
        "all_measurements_aligned": local_aligned and object_aligned,
        "alignment_counted_as_confirmation": False,
    }


def update_candidate_assessments(
    *,
    candidates: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        candidate_outcomes = [
            row for row in outcomes if str(row.get("candidate_id", "")) == candidate_id
        ]
        additional_support = sum(
            int(row.get("acquired_raw_support_events", 0) or 0)
            for row in candidate_outcomes
            if bool(row.get("completed", False))
        )
        context_hashes = {
            str(row.get("context_snapshot_hash", ""))
            for row in candidate.get("contexts", []) or []
            if str(row.get("context_snapshot_hash", ""))
        }
        context_hashes.update(
            str(row.get("acquired_context_snapshot_hash", ""))
            for row in candidate_outcomes
            if bool(row.get("completed", False))
            and str(row.get("acquired_context_snapshot_hash", ""))
        )
        base_controls = {
            str(row.get("action", ""))
            for row in candidate.get("control_interventions", []) or []
            if str(row.get("action", ""))
        }
        acquired_controls = {
            str(value)
            for row in candidate_outcomes
            if bool(row.get("completed", False))
            for value in row.get("acquired_control_actions", []) or []
            if str(value)
        }
        raw_support = int(candidate.get("raw_support_events", 0) or 0) + additional_support
        independent_contexts = len(context_hashes)
        distinct_controls = len(base_controls | acquired_controls)
        contradictions = int(candidate.get("contradiction_events", 0) or 0) + sum(
            int(row.get("acquired_contradiction_events", 0) or 0)
            for row in candidate_outcomes
        )
        missing = missing_revision_requirements(
            raw_support_events=raw_support,
            independent_context_events=independent_contexts,
            distinct_control_actions=distinct_controls,
            contradiction_events=contradictions,
        )
        control_surface_exhausted = any(
            str(row.get("resolution_status", "")) == FOLLOWUP_BLOCKED_CONTROL_SURFACE
            for row in candidate_outcomes
        )
        recommendation = a32_intake_recommendation(missing)
        if control_surface_exhausted and "minimum_distinct_control_actions" in missing:
            recommendation = UPDATED_RECOMMENDATION_CONTROL_SURFACE_EXHAUSTED
        cross_outcome = next(
            (
                row
                for row in candidate_outcomes
                if str(row.get("request_type", "")) == CROSS_MEASUREMENT_REQUEST
            ),
            {},
        )
        updated.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": str(candidate.get("candidate_key", "")),
                "game_id": str(candidate.get("game_id", "")),
                "action": str(candidate.get("action", "")),
                "action_args": candidate.get("action_args"),
                "raw_support_events_before": int(
                    candidate.get("raw_support_events", 0) or 0
                ),
                "new_comparable_support_events": additional_support,
                "raw_support_events_after": raw_support,
                "independent_context_events_after": independent_contexts,
                "distinct_control_actions_after": distinct_controls,
                "contradiction_events_after": contradictions,
                "cross_measurement_status": str(
                    cross_outcome.get("resolution_status", "NOT_REQUESTED")
                ),
                "control_surface_exhausted": control_surface_exhausted,
                "missing_revision_requirements": list(missing),
                "a32_intake_recommendation": recommendation,
                "ready_for_A32_intake": not missing,
                "candidate_assessment_counted_as_revision": False,
                "support": 0,
                "truth_status": SAGE5H_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return updated


def summarize_sage5h(
    *,
    source_sage5g: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked_experiments: Sequence[Mapping[str, Any]],
    control_surface_audits: Sequence[Mapping[str, Any]],
    updated_assessments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    completed = [row for row in outcomes if bool(row.get("completed", False))]
    blocked = [row for row in outcomes if not bool(row.get("completed", False))]
    request_count = len(source_sage5g.get("requested_followups", []) or [])
    all_resolved = len(outcomes) == request_count and all(
        bool(row.get("resolution_status", "")) for row in outcomes
    )
    all_completed = bool(outcomes) and len(completed) == len(outcomes)
    partial = bool(completed) and bool(blocked)
    if all_completed:
        outcome_status = SAGE5H_COMPLETE
    elif partial:
        outcome_status = SAGE5H_PARTIAL_CONTROL_SURFACE_LIMIT
    else:
        outcome_status = SAGE5H_NO_ACQUISITION
    resolution_counts = Counter(str(row.get("resolution_status", "")) for row in outcomes)
    return {
        "source_sage5g_outcome_status": str(source_sage5g.get("outcome_status", "")),
        "followup_requests_consumed": request_count,
        "followup_outcomes": len(outcomes),
        "followups_completed": len(completed),
        "followups_blocked": len(blocked),
        "control_diversity_followups_completed": sum(
            1
            for row in completed
            if str(row.get("request_type", "")) == CONTROL_DIVERSITY_REQUEST
        ),
        "control_diversity_followups_blocked": sum(
            1
            for row in blocked
            if str(row.get("request_type", "")) == CONTROL_DIVERSITY_REQUEST
        ),
        "support_followups_completed": sum(
            1 for row in completed if str(row.get("request_type", "")) == SUPPORT_REQUEST
        ),
        "cross_measurement_followups_completed": sum(
            1
            for row in completed
            if str(row.get("request_type", "")) == CROSS_MEASUREMENT_REQUEST
        ),
        "resolution_status_counts": dict(sorted(resolution_counts.items())),
        "control_surface_contexts_audited": len(control_surface_audits),
        "replay_exact_control_surface_audits": sum(
            1
            for row in control_surface_audits
            if bool(row.get("live_prefix_replay_exact", False))
        ),
        "control_surface_exhausted_audits": sum(
            1
            for row in control_surface_audits
            if bool(row.get("control_surface_exhausted", False))
        ),
        "controlled_experiments_executed": len(experiments),
        "controlled_experiments_blocked": len(blocked_experiments),
        "live_prefix_replay_exact_experiments": sum(
            1 for row in experiments if bool(row.get("live_prefix_replay_exact", False))
        ),
        "raw_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in experiments
        ),
        "raw_contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in experiments
        ),
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in experiments
        ),
        "comparable_support_events_acquired": sum(
            int(row.get("acquired_raw_support_events", 0) or 0) for row in completed
        ),
        "cross_measurement_alignments": sum(
            1
            for row in completed
            if str(row.get("request_type", "")) == CROSS_MEASUREMENT_REQUEST
            and bool(
                (row.get("measurement_alignment", {}) or {}).get(
                    "all_measurements_aligned",
                    False,
                )
            )
        ),
        "cross_measurement_divergences": sum(
            1
            for row in completed
            if str(row.get("request_type", "")) == CROSS_MEASUREMENT_REQUEST
            and not bool(
                (row.get("measurement_alignment", {}) or {}).get(
                    "all_measurements_aligned",
                    False,
                )
            )
        ),
        "candidates_ready_for_A32_intake": sum(
            1 for row in updated_assessments if bool(row.get("ready_for_A32_intake", False))
        ),
        "all_followups_resolved": all_resolved,
        "all_requested_followups_completed": all_completed,
        "followup_resolution_gate_passed": all_resolved,
        "gate_passed": all_resolved,
        "outcome_status": outcome_status,
        "followup_events_counted_as_scientific_support": False,
        "control_surface_block_counted_as_refutation": False,
        "cross_measurement_alignment_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE5H_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "execution_performed": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_sage5h_sources(
    source_sage5g: Mapping[str, Any],
    source_sage5e: Mapping[str, Any],
    source_sage5f: Mapping[str, Any],
) -> None:
    for label, source in (
        ("SAGE.5g", source_sage5g),
        ("SAGE.5e", source_sage5e),
        ("SAGE.5f", source_sage5f),
    ):
        summary = dict(source.get("summary", {}) or {})
        if int(source.get("support", summary.get("support", 0)) or 0) != 0:
            raise ValueError(f"{label} support must remain 0")
        if str(source.get("revision_status", "CANDIDATE_ONLY")) != "CANDIDATE_ONLY":
            raise ValueError(f"{label} must remain candidate-only")
        if bool(source.get("revision_performed", False)):
            raise ValueError(f"{label} must not perform revision")
        if int(source.get("wrong_confirmations", 0) or 0) != 0:
            raise ValueError(f"{label} wrong_confirmations must remain 0")
        if bool(source.get("a32_write_performed", False)) or bool(
            source.get("a33_write_performed", False)
        ):
            raise ValueError(f"{label} cannot write A32/A33")
    if bool(source_sage5g.get("candidate_review_item_counted_as_revision", False)):
        raise ValueError("SAGE.5g review items cannot count as revision")
    if bool(source_sage5g.get("candidate_review_item_counted_as_scientific_verdict", False)):
        raise ValueError("SAGE.5g review items cannot count as verdict")


def write_sage5h_controlled_followup_acquisition(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_source_request_ids(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    rows = sorted(
        [dict(row) for row in candidate.get("target_interventions", []) or []],
        key=lambda row: (
            int(row.get("budget", 0) or 0),
            str(row.get("request_id", "")),
        ),
    )
    return tuple(str(row.get("request_id", "")) for row in rows if row.get("request_id"))


def _core_effect_signature_matches(
    candidate: Mapping[str, Any],
    request: Mapping[str, Any],
) -> bool:
    predicted = dict(candidate.get("predicted_effect_signature", {}) or {})
    observed = dict(request.get("diff_signature", {}) or {})
    return bool(
        int(predicted.get("changed_cells", 0) or 0)
        == int(observed.get("changed_cells", 0) or 0)
        and dict(predicted.get("color_transitions", {}) or {})
        == dict(observed.get("color_transitions", {}) or {})
        and bool(predicted.get("terminal_after", False))
        == bool(observed.get("terminal_after", False))
        and int(predicted.get("levels_delta", 0) or 0)
        == int(observed.get("levels_delta", 0) or 0)
    )


def _base_followup_outcome(
    followup: Mapping[str, Any],
    *,
    completed: bool,
    resolution_status: str,
    blocked_reason: str,
    **details: Any,
) -> Dict[str, Any]:
    return {
        "followup_id": str(followup.get("followup_id", "")),
        "candidate_id": str(followup.get("candidate_id", "")),
        "request_type": str(followup.get("request_type", "")),
        "completed": bool(completed),
        "resolution_status": resolution_status,
        "blocked_reason": blocked_reason,
        **details,
        "support": 0,
        "truth_status": SAGE5H_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "followup_result_counted_as_scientific_support": False,
        "followup_result_counted_as_revision": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _blocked_control_audit(
    request: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "execution_status": "BLOCKED",
        "source_request_id": str(request.get("request_id", "")),
        "source_transition_id": str(request.get("source_transition_id", "")),
        "game_id": str(request.get("game_id", "")),
        "budget": _budget_from_request(request),
        "blocked_reason": reason,
        "live_prefix_replay_exact": False,
        "prefix_env_actions": 0,
        "available_action_names": [],
        "available_action_variants": [],
        "eligible_distinct_control_actions": [],
        "control_surface_exhausted": False,
        "support": 0,
        "truth_status": SAGE5H_TRUTH_STATUS,
    }


def _local_patch_signature(experiment: Mapping[str, Any]) -> Dict[str, Any]:
    measurement = dict(experiment.get("target_measurement", {}) or {})
    return {
        "changed_pixels": int(measurement.get("changed_pixels", 0) or 0),
        "local_patch_available": bool(measurement.get("local_patch_available", False)),
        "local_changed_pixels": int(measurement.get("local_changed_pixels", 0) or 0),
    }


def _object_delta_signature(experiment: Mapping[str, Any]) -> Dict[str, Any]:
    measurement = dict(experiment.get("target_measurement", {}) or {})
    return {
        "object_count_delta": int(measurement.get("object_count_delta", 0) or 0),
        "object_count_delta_by_color": dict(
            measurement.get("object_count_delta_by_color", {}) or {}
        ),
    }


def _all_signatures_equal(signatures: Iterable[Mapping[str, Any]]) -> bool:
    canonical = {_canonical_json(dict(value)) for value in signatures}
    return bool(canonical) and len(canonical) == 1


def _first_control_action(request: Mapping[str, Any]) -> str:
    target = str(request.get("target_action", ""))
    return next(
        (
            str(value)
            for value in request.get("suggested_control_actions", []) or []
            if str(value) and str(value) != target
        ),
        "",
    )


def _budget_from_request(request: Mapping[str, Any]) -> int:
    transition = str(request.get("source_transition_id", ""))
    marker = "::budget_"
    if marker in transition:
        return int(transition.split(marker, 1)[1].split("::", 1)[0].split("_", 1)[0])
    return 0


def _append_unique(values: List[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire SAGE.5g controlled follow-up evidence candidate-only.",
    )
    parser.add_argument("--source-sage5g", default=str(DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH))
    parser.add_argument(
        "--source-sage5e",
        default=str(DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH),
    )
    parser.add_argument(
        "--source-sage5f",
        default=str(DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH),
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH),
    )
    args = parser.parse_args(argv)
    run_sage5h_controlled_followup_acquisition(
        source_sage5g_path=args.source_sage5g,
        source_sage5e_path=args.source_sage5e,
        source_sage5f_path=args.source_sage5f,
        environments_dir=args.environments_dir,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
