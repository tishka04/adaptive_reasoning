"""SAGE.6d candidate-only handoff compiler for the second unknown game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .second_unknown_game_event_consolidation import (
    DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH,
    SAGE6C_EVENTS_CONSOLIDATED,
    SAGE6C_SCHEMA_VERSION,
    SAGE6C_TRUTH_STATUS,
    validate_sage6c_source,
)
from .second_unknown_game_m3_execution import DEFAULT_SAGE6B_M3_EXECUTION_PATH


DEFAULT_SAGE6D_HANDOFF_PATH = (
    Path("diagnostics") / "sage" / "sage6d_second_unknown_game_handoff.json"
)

SAGE6D_SCHEMA_VERSION = "sage.second_unknown_game_handoff.v1"
SAGE6D_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_6D"
SAGE6D_HANDOFF_COMPILED = "SAGE_SECOND_UNKNOWN_GAME_HANDOFF_COMPILED_CANDIDATE_ONLY"
SAGE6D_HANDOFF_INCOMPLETE = "SAGE_SECOND_UNKNOWN_GAME_HANDOFF_INCOMPLETE_CANDIDATE_ONLY"

HANDOFF_READY_FOR_FOLLOWUP_EXECUTION = (
    "FOLLOWUPS_PRE_REGISTERED_FOR_EXECUTION_CANDIDATE_ONLY"
)
CONTROL_DIVERSITY_REQUEST = "ACQUIRE_DISTINCT_CONTROL_ACTION"
NEUTRAL_REPLICATION_REQUEST = "REPLICATE_NEUTRAL_CONTEXT"

REQUIRED_SOURCE_FOLLOWUPS = (
    "ADD_DISTINCT_CONTROL_ACTION_PER_BUDGET",
    "REPLICATE_NEUTRAL_CONTEXT",
    "PRESERVE_CONTEXT_CLUSTER_BOUNDARIES",
)


def run_sage6d_second_unknown_game_handoff(
    *,
    source_sage6c_path: str | Path = DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH,
    source_sage6b_path: str | Path = DEFAULT_SAGE6B_M3_EXECUTION_PATH,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Compile the SAGE.6c frontier into pre-registered followups only."""
    source_sage6c = _load_json(source_sage6c_path)
    source_sage6b = _load_json(source_sage6b_path)
    validate_sage6d_sources(source_sage6c, source_sage6b)

    handoff_items, protocols = compile_sage6d_handoff(
        source_sage6c=source_sage6c,
        source_sage6b=source_sage6b,
    )
    gate = build_sage6d_gate(
        source_sage6c=source_sage6c,
        source_sage6b=source_sage6b,
        handoff_items=handoff_items,
        protocols=protocols,
    )
    outcome = (
        SAGE6D_HANDOFF_COMPILED
        if gate and all(gate.values())
        else SAGE6D_HANDOFF_INCOMPLETE
    )
    summary = summarize_sage6d(
        source_sage6c=source_sage6c,
        handoff_items=handoff_items,
        protocols=protocols,
        gate=gate,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE6D_SCHEMA_VERSION,
            "source_sage6c_path": str(source_sage6c_path),
            "source_sage6b_path": str(source_sage6b_path),
            "inputs_read": ["SAGE.6c", "SAGE.6b"],
            "execution_performed": False,
            "compilation_policy": {
                "one_stable_context_selected_per_budget": True,
                "same_new_control_used_across_budgets": True,
                "new_control_must_be_source_suggested": True,
                "neutral_context_replay_is_exact_replication": True,
                "context_replay_and_hash_are_immutable": True,
                "cross_context_substitution_allowed": False,
                "cross_context_merge_allowed": False,
                "pre_registered_outcome_is_not_scientific_verdict": True,
                "raw_events_are_not_scientific_support": True,
                "a32_write_performed": False,
                "a33_write_performed": False,
            },
            "followup_thresholds": {
                "control_diversity_protocols_per_budget": 1,
                "neutral_context_replications": 1,
                "minimum_projected_distinct_control_actions": 2,
            },
            "artifacts_not_modified": [
                "SAGE.6c",
                "SAGE.6b",
                "SAGE.6a",
                "SAGE.6",
                "SAGE.5",
                "A32",
                "A33",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
        },
        "source_sage6c_context": build_source_context(source_sage6c),
        "source_sage6b_context": build_source_context(source_sage6b),
        "handoff_items": handoff_items,
        "pre_registered_followup_protocols": protocols,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE6D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "pre_registered_protocol_counted_as_execution": False,
        "pre_registered_protocol_counted_as_scientific_verdict": False,
        "raw_events_counted_as_support": False,
        "handoff_counted_as_revision": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage6d_second_unknown_game_handoff(payload, output_path)
    return payload


def compile_sage6d_handoff(
    *,
    source_sage6c: Mapping[str, Any],
    source_sage6b: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compile one frontier and four deterministic, pre-registered protocols."""
    frontier = next(
        dict(row)
        for row in source_sage6c.get("candidate_handoff_frontiers", []) or []
        if isinstance(row, Mapping)
        and bool(row.get("ready_for_A32_handoff_compilation", False))
    )
    events_by_request = _records_by_id(source_sage6c, "event_records")
    clusters_by_id = _records_by_id(
        source_sage6c, "context_clusters", id_field="context_cluster_id"
    )
    requests_by_id = _records_by_id(source_sage6b, "selected_execution_requests")
    existing_controls = sorted(
        {
            str(row.get("control_action", ""))
            for row in source_sage6c.get("event_records", []) or []
            if isinstance(row, Mapping) and str(row.get("control_action", ""))
        }
    )
    target_action = str(frontier.get("target_action", ""))
    stable_rows: List[
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for cluster_id in frontier.get("context_cluster_ids", []) or []:
        cluster = clusters_by_id[str(cluster_id)]
        request_id = _single_string(cluster.get("request_ids", []), "request_id")
        stable_rows.append(
            (cluster, events_by_request[request_id], requests_by_id[request_id])
        )

    selected_by_budget: Dict[
        int, tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    for row in sorted(stable_rows, key=_cluster_event_request_sort_key):
        budget = int(row[1].get("budget", 0) or 0)
        selected_by_budget.setdefault(budget, row)

    selected_new_controls = {
        choose_distinct_control(
            suggested_controls=request.get("suggested_control_actions", []) or [],
            target_action=target_action,
            excluded_controls=existing_controls,
        )
        for _, _, request in selected_by_budget.values()
    }
    if len(selected_new_controls) != 1:
        raise ValueError("SAGE.6d requires one common new control across budgets")
    new_control = next(iter(selected_new_controls))

    protocols: List[Dict[str, Any]] = []
    for budget in sorted(selected_by_budget):
        cluster, event, request = selected_by_budget[budget]
        protocols.append(
            build_followup_protocol(
                protocol_id=(
                    f"sage6d::followup_protocol::control_diversity::{budget:03d}"
                ),
                request_type=CONTROL_DIVERSITY_REQUEST,
                cluster=cluster,
                event=event,
                request=request,
                target_action=target_action,
                control_action=new_control,
                prior_control_action=str(frontier.get("control_action", "")),
                expected_prior_effect=float(frontier.get("stable_effect_size", 0.0)),
            )
        )

    exception_id = _single_string(
        frontier.get("exception_context_cluster_ids", []),
        "exception context cluster",
    )
    exception_cluster = clusters_by_id[exception_id]
    exception_request_id = _single_string(
        exception_cluster.get("request_ids", []), "exception request_id"
    )
    exception_event = events_by_request[exception_request_id]
    exception_request = requests_by_id[exception_request_id]
    protocols.append(
        build_followup_protocol(
            protocol_id="sage6d::followup_protocol::neutral_replication::050",
            request_type=NEUTRAL_REPLICATION_REQUEST,
            cluster=exception_cluster,
            event=exception_event,
            request=exception_request,
            target_action=target_action,
            control_action=str(frontier.get("control_action", "")),
            prior_control_action=str(frontier.get("control_action", "")),
            expected_prior_effect=float(exception_event.get("effect_size", 0.0)),
        )
    )

    manifest = build_context_cluster_manifest(
        source_sage6c=source_sage6c,
        selected_protocols=protocols,
        stable_cluster_ids=frontier.get("context_cluster_ids", []) or [],
        exception_cluster_ids=frontier.get("exception_context_cluster_ids", []) or [],
    )
    handoff_id = "sage6d::second_unknown_game_handoff::001"
    handoff = {
        "handoff_id": handoff_id,
        "source_frontier_id": str(frontier.get("frontier_id", "")),
        "source_effect_group_id": str(frontier.get("source_effect_group_id", "")),
        "game_id": str(frontier.get("game_id", "")),
        "candidate_mechanism_family": str(
            frontier.get("candidate_mechanism_family", "")
        ),
        "metric": str(frontier.get("metric", "")),
        "target_action": target_action,
        "executed_control_actions": existing_controls,
        "pre_registered_new_control_actions": [new_control],
        "projected_control_actions_after_execution": sorted(
            {*existing_controls, new_control}
        ),
        "source_budgets": list(frontier.get("budgets", []) or []),
        "stable_context_cluster_ids": list(
            frontier.get("context_cluster_ids", []) or []
        ),
        "exception_context_cluster_ids": list(
            frontier.get("exception_context_cluster_ids", []) or []
        ),
        "context_cluster_manifest": manifest,
        "pre_registered_followup_protocol_ids": [
            str(row.get("protocol_id", "")) for row in protocols
        ],
        "required_followups": list(frontier.get("required_followups", []) or []),
        "raw_support_events": int(frontier.get("raw_support_events", 0) or 0),
        "raw_contradiction_events": int(
            frontier.get("raw_contradiction_events", 0) or 0
        ),
        "raw_neutral_events": int(frontier.get("raw_neutral_events", 0) or 0),
        "handoff_status": HANDOFF_READY_FOR_FOLLOWUP_EXECUTION,
        "ready_for_followup_execution": True,
        "ready_for_A32_review": False,
        "ready_for_A32_review_blocked_reason": (
            "PRE_REGISTERED_FOLLOWUPS_NOT_EXECUTED"
        ),
        "a32_intake_requested": False,
        "execution_performed": False,
        "cross_context_merge_performed": False,
        "context_boundaries_preserved": True,
        "handoff_counted_as_revision": False,
        "raw_events_counted_as_support": False,
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": SAGE6D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    for protocol in protocols:
        protocol["handoff_id"] = handoff_id
    return [handoff], protocols


def choose_distinct_control(
    *,
    suggested_controls: Sequence[Any],
    target_action: str,
    excluded_controls: Sequence[str],
) -> str:
    """Choose the first source-suggested control not already executed."""
    excluded = {str(value) for value in excluded_controls} | {str(target_action)}
    candidates = [
        str(value)
        for value in suggested_controls
        if str(value) and str(value) not in excluded
    ]
    if not candidates:
        raise ValueError("no source-suggested distinct control is available")
    return candidates[0]


def build_followup_protocol(
    *,
    protocol_id: str,
    request_type: str,
    cluster: Mapping[str, Any],
    event: Mapping[str, Any],
    request: Mapping[str, Any],
    target_action: str,
    control_action: str,
    prior_control_action: str,
    expected_prior_effect: float,
) -> Dict[str, Any]:
    neutral_replication = request_type == NEUTRAL_REPLICATION_REQUEST
    suggested = [
        str(value) for value in request.get("suggested_control_actions", []) or []
    ]
    return {
        "protocol_id": protocol_id,
        "handoff_id": "",
        "request_type": request_type,
        "source_context_cluster_id": str(cluster.get("context_cluster_id", "")),
        "source_event_id": str(event.get("event_id", "")),
        "source_request_id": str(event.get("request_id", "")),
        "source_hypothesis_id": str(event.get("source_hypothesis_id", "")),
        "source_transition_id": str(event.get("source_transition_id", "")),
        "game_id": str(event.get("game_id", "")),
        "budget": int(event.get("budget", 0) or 0),
        "source_step": int(event.get("source_step", 0) or 0),
        "hypothesis_family": str(event.get("hypothesis_family", "")),
        "metric": str(event.get("metric", "")),
        "context_state_origin": str(request.get("context_state_origin", "")),
        "context_snapshot_hash": str(event.get("context_snapshot_hash", "")),
        "context_replay": list(request.get("context_replay", []) or []),
        "context_replay_args": [
            dict(row or {}) for row in request.get("context_replay_args", []) or []
        ],
        "target_action": target_action,
        "target_action_args": (
            dict(request.get("target_action_args", {}) or {})
            if request.get("target_action_args") is not None
            else None
        ),
        "control_action": control_action,
        "prior_control_action": prior_control_action,
        "source_suggested_control_actions": suggested,
        "control_action_was_suggested_by_source_request": control_action in suggested,
        "expected_prior_effect_size": float(expected_prior_effect),
        "pre_registered_interpretation": {
            "consistency_condition": (
                "controlled_effect_size_equals_0"
                if neutral_replication
                else "controlled_effect_size_greater_than_0"
            ),
            "deviation_condition": (
                "controlled_effect_size_not_equal_to_0"
                if neutral_replication
                else "controlled_effect_size_less_than_or_equal_to_0"
            ),
            "any_outcome_remains_raw_candidate_only": True,
            "outcome_cannot_trigger_automatic_A32_or_A33_write": True,
        },
        "exact_replay_required": True,
        "context_hash_verification_required": True,
        "target_and_control_must_start_from_same_context": True,
        "cross_context_substitution_allowed": False,
        "cross_context_merge_allowed": False,
        "execution_performed": False,
        "execution_status": "PRE_REGISTERED_NOT_EXECUTED",
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": SAGE6D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "protocol_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def build_context_cluster_manifest(
    *,
    source_sage6c: Mapping[str, Any],
    selected_protocols: Sequence[Mapping[str, Any]],
    stable_cluster_ids: Sequence[Any],
    exception_cluster_ids: Sequence[Any],
) -> List[Dict[str, Any]]:
    protocol_by_cluster = {
        str(row.get("source_context_cluster_id", "")): str(row.get("protocol_id", ""))
        for row in selected_protocols
    }
    stable_ids = {str(value) for value in stable_cluster_ids}
    exception_ids = {str(value) for value in exception_cluster_ids}
    manifest: List[Dict[str, Any]] = []
    for cluster in source_sage6c.get("context_clusters", []) or []:
        if not isinstance(cluster, Mapping):
            continue
        cluster_id = str(cluster.get("context_cluster_id", ""))
        manifest.append(
            {
                "context_cluster_id": cluster_id,
                "context_snapshot_hash": str(cluster.get("context_snapshot_hash", "")),
                "request_ids": list(cluster.get("request_ids", []) or []),
                "budgets": list(cluster.get("budgets", []) or []),
                "source_steps": list(cluster.get("source_steps", []) or []),
                "candidate_status": str(cluster.get("candidate_status", "")),
                "frontier_role": (
                    "STABLE_PATTERN"
                    if cluster_id in stable_ids
                    else "CONTEXTUAL_EXCEPTION"
                    if cluster_id in exception_ids
                    else "OUTSIDE_FRONTIER"
                ),
                "selected_followup_protocol_id": protocol_by_cluster.get(cluster_id),
                "context_preserved": True,
                "cross_context_merge_performed": False,
                "support": 0,
                "truth_status": SAGE6D_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
            }
        )
    return manifest


def build_sage6d_gate(
    *,
    source_sage6c: Mapping[str, Any],
    source_sage6b: Mapping[str, Any],
    handoff_items: Sequence[Mapping[str, Any]],
    protocols: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    source_budgets = {
        int(value)
        for value in source_sage6c.get("summary", {}).get("budgets", []) or []
    }
    diversity = [
        row for row in protocols if row.get("request_type") == CONTROL_DIVERSITY_REQUEST
    ]
    neutral = [
        row
        for row in protocols
        if row.get("request_type") == NEUTRAL_REPLICATION_REQUEST
    ]
    protocol_ids = [str(row.get("protocol_id", "")) for row in protocols]
    context_hashes = [str(row.get("context_snapshot_hash", "")) for row in protocols]
    handoff = handoff_items[0] if len(handoff_items) == 1 else {}
    manifest = list(handoff.get("context_cluster_manifest", []) or [])
    source_clusters = list(source_sage6c.get("context_clusters", []) or [])
    existing_controls = set(handoff.get("executed_control_actions", []) or [])
    projected_controls = set(
        handoff.get("projected_control_actions_after_execution", []) or []
    )
    return {
        "source_sage6c_gate_passed": bool(
            source_sage6c.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source_sage6c.get("gate", {}).values()),
        "source_sage6b_gate_passed": bool(
            source_sage6b.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source_sage6b.get("gate", {}).values()),
        "one_compilable_frontier_produced_one_handoff": len(handoff_items) == 1,
        "one_control_diversity_protocol_per_budget": len(diversity)
        == len(source_budgets)
        and {int(row.get("budget", 0) or 0) for row in diversity} == source_budgets,
        "one_neutral_context_replication_pre_registered": len(neutral) == 1
        and float(neutral[0].get("expected_prior_effect_size", 1.0) or 0.0) == 0.0
        and neutral[0].get("control_action") == neutral[0].get("prior_control_action"),
        "new_control_is_distinct_source_suggested_and_shared": bool(diversity)
        and len({str(row.get("control_action", "")) for row in diversity}) == 1
        and all(
            str(row.get("control_action", "")) not in existing_controls
            and bool(row.get("control_action_was_suggested_by_source_request", False))
            for row in diversity
        ),
        "projected_control_diversity_reaches_threshold": len(projected_controls) >= 2,
        "protocol_ids_and_contexts_unique": "" not in protocol_ids
        and len(protocol_ids) == len(set(protocol_ids))
        and "" not in context_hashes
        and len(context_hashes) == len(set(context_hashes)),
        "all_protocols_require_exact_immutable_contexts": bool(protocols)
        and all(
            bool(row.get("exact_replay_required", False))
            and bool(row.get("context_hash_verification_required", False))
            and bool(row.get("target_and_control_must_start_from_same_context", False))
            and not bool(row.get("cross_context_substitution_allowed", True))
            and not bool(row.get("cross_context_merge_allowed", True))
            and len(row.get("context_replay", []) or [])
            == len(row.get("context_replay_args", []) or [])
            for row in protocols
        ),
        "all_source_context_cluster_boundaries_preserved": len(manifest)
        == len(source_clusters)
        and {str(row.get("context_cluster_id", "")) for row in manifest}
        == {str(row.get("context_cluster_id", "")) for row in source_clusters}
        and all(
            bool(row.get("context_preserved", False))
            and not bool(row.get("cross_context_merge_performed", False))
            for row in manifest
        ),
        "handoff_ready_only_for_followup_execution": bool(
            handoff.get("ready_for_followup_execution", False)
        )
        and not bool(handoff.get("ready_for_A32_review", True))
        and not bool(handoff.get("a32_intake_requested", True)),
        "no_execution_or_scientific_verdict": all(
            not bool(row.get("execution_performed", True))
            and str(row.get("execution_status", "")) == "PRE_REGISTERED_NOT_EXECUTED"
            and not bool(row.get("protocol_counted_as_scientific_verdict", True))
            for row in protocols
        ),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0
            and str(row.get("truth_status", "")) == SAGE6D_TRUTH_STATUS
            and str(row.get("revision_status", "")) == "CANDIDATE_ONLY"
            for row in [*handoff_items, *protocols, *manifest]
        ),
        "source_registry_quarantine_preserved": all(
            int(source.get("source_scoped_mechanics_reused", 0) or 0) == 0
            and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
            and not bool(source.get("scope_generalization_performed", False))
            for source in (source_sage6c, source_sage6b)
        ),
    }


def summarize_sage6d(
    *,
    source_sage6c: Mapping[str, Any],
    handoff_items: Sequence[Mapping[str, Any]],
    protocols: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
    outcome: str,
) -> Dict[str, Any]:
    diversity = [
        row for row in protocols if row.get("request_type") == CONTROL_DIVERSITY_REQUEST
    ]
    neutral = [
        row
        for row in protocols
        if row.get("request_type") == NEUTRAL_REPLICATION_REQUEST
    ]
    handoff = handoff_items[0] if len(handoff_items) == 1 else {}
    return {
        "source_sage6c_outcome_status": str(source_sage6c.get("outcome_status", "")),
        "game_id": str(source_sage6c.get("summary", {}).get("game_id", "")),
        "budgets": list(source_sage6c.get("summary", {}).get("budgets", []) or []),
        "source_candidate_handoff_frontiers": len(
            source_sage6c.get("candidate_handoff_frontiers", []) or []
        ),
        "handoff_items": len(handoff_items),
        "pre_registered_followup_protocols": len(protocols),
        "control_diversity_protocols": len(diversity),
        "control_diversity_budgets": sorted(
            {int(row.get("budget", 0) or 0) for row in diversity}
        ),
        "neutral_context_replication_protocols": len(neutral),
        "pre_registered_new_control_actions": list(
            handoff.get("pre_registered_new_control_actions", []) or []
        ),
        "executed_distinct_control_actions": len(
            set(handoff.get("executed_control_actions", []) or [])
        ),
        "projected_distinct_control_actions_after_execution": len(
            set(handoff.get("projected_control_actions_after_execution", []) or [])
        ),
        "context_clusters_preserved": len(
            handoff.get("context_cluster_manifest", []) or []
        ),
        "protocol_contexts": len(
            {str(row.get("context_snapshot_hash", "")) for row in protocols}
        ),
        "raw_support_events": int(
            source_sage6c.get("summary", {}).get("raw_support_events", 0) or 0
        ),
        "raw_contradiction_events": int(
            source_sage6c.get("summary", {}).get("raw_contradiction_events", 0) or 0
        ),
        "raw_neutral_events": int(
            source_sage6c.get("summary", {}).get("raw_neutral_events", 0) or 0
        ),
        "ready_for_followup_execution": sum(
            1
            for row in handoff_items
            if bool(row.get("ready_for_followup_execution", False))
        ),
        "ready_for_A32_review": sum(
            1 for row in handoff_items if bool(row.get("ready_for_A32_review", False))
        ),
        "execution_performed": False,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "outcome_status": outcome,
        "support": 0,
        "truth_status": SAGE6D_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def validate_sage6d_sources(
    source_sage6c: Mapping[str, Any], source_sage6b: Mapping[str, Any]
) -> None:
    """Reject source drift, verdict leakage, and broken cross-artifact links."""
    validate_sage6c_source(source_sage6b)
    config = dict(source_sage6c.get("config", {}) or {})
    summary = dict(source_sage6c.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6C_SCHEMA_VERSION:
        raise ValueError("SAGE.6c schema version is not supported by SAGE.6d")
    if str(source_sage6c.get("outcome_status", "")) != SAGE6C_EVENTS_CONSOLIDATED:
        raise ValueError("SAGE.6d requires a completed SAGE.6c consolidation")
    if str(source_sage6c.get("truth_status", "")) != SAGE6C_TRUTH_STATUS:
        raise ValueError("SAGE.6c truth must remain unevaluated")
    if (
        str(source_sage6c.get("status", "")) != "UNRESOLVED"
        or str(source_sage6c.get("revision_status", "")) != "CANDIDATE_ONLY"
        or int(source_sage6c.get("support", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6c must remain unresolved candidate-only with support 0")
    if any(
        bool(source_sage6c.get(flag, False))
        for flag in (
            "revision_performed",
            "confirmation_performed",
            "refutation_performed",
            "a32_write_performed",
            "a33_write_performed",
            "scope_generalization_performed",
        )
    ):
        raise ValueError(
            "SAGE.6c cannot contain execution, verdict, or registry writes"
        )
    if (
        not source_sage6c.get("gate")
        or not all(bool(value) for value in source_sage6c.get("gate", {}).values())
        or not bool(summary.get("gate_passed", False))
    ):
        raise ValueError("every SAGE.6c source gate must pass")

    frontiers = [
        row
        for row in source_sage6c.get("candidate_handoff_frontiers", []) or []
        if isinstance(row, Mapping)
    ]
    if len(frontiers) != 1:
        raise ValueError("SAGE.6d requires exactly one SAGE.6c handoff frontier")
    frontier = frontiers[0]
    if (
        not bool(frontier.get("ready_for_A32_handoff_compilation", False))
        or bool(frontier.get("ready_for_A32_review", True))
        or set(frontier.get("required_followups", []) or [])
        != set(REQUIRED_SOURCE_FOLLOWUPS)
    ):
        raise ValueError("SAGE.6c frontier is not eligible for SAGE.6d compilation")

    events = [
        row
        for row in source_sage6c.get("event_records", []) or []
        if isinstance(row, Mapping)
    ]
    clusters = [
        row
        for row in source_sage6c.get("context_clusters", []) or []
        if isinstance(row, Mapping)
    ]
    requests = [
        row
        for row in source_sage6b.get("selected_execution_requests", []) or []
        if isinstance(row, Mapping)
    ]
    event_ids = {str(row.get("request_id", "")) for row in events}
    request_ids = {str(row.get("request_id", "")) for row in requests}
    if "" in event_ids or event_ids != request_ids:
        raise ValueError("SAGE.6b requests and SAGE.6c events must align exactly")
    source_game = str(source_sage6b.get("summary", {}).get("game_id", ""))
    if not source_game or source_game != str(summary.get("game_id", "")):
        raise ValueError("SAGE.6b and SAGE.6c game identities must align")
    if [int(value) for value in summary.get("budgets", []) or []] != [
        int(value)
        for value in source_sage6b.get("summary", {}).get("budgets_available", []) or []
    ]:
        raise ValueError("SAGE.6b and SAGE.6c budgets must align")

    events_by_request = {str(row.get("request_id", "")): row for row in events}
    for request in requests:
        request_id = str(request.get("request_id", ""))
        event = events_by_request[request_id]
        if (
            str(request.get("game_id", "")) != source_game
            or str(event.get("game_id", "")) != source_game
            or str(request.get("context_snapshot_hash", ""))
            != str(event.get("context_snapshot_hash", ""))
            or str(request.get("target_action", ""))
            != str(event.get("target_action", ""))
            or str(request.get("metric", "")) != str(event.get("metric", ""))
            or str(request.get("hypothesis_family", ""))
            != str(event.get("hypothesis_family", ""))
            or int(request.get("source_step", 0) or 0)
            != int(event.get("source_step", 0) or 0)
        ):
            raise ValueError(
                "SAGE.6b request and SAGE.6c event contexts must align exactly"
            )
        if (
            str(request.get("status", "")) != "READY_FOR_M3"
            or str(request.get("replayability", "")) != "LIVE_PREFIX_REPLAY_CONTEXT"
            or len(request.get("context_replay", []) or [])
            != len(request.get("context_replay_args", []) or [])
        ):
            raise ValueError("SAGE.6b requests must remain exactly replayable")

    cluster_ids = {str(row.get("context_cluster_id", "")) for row in clusters}
    frontier_ids = {
        str(value)
        for value in [
            *(frontier.get("context_cluster_ids", []) or []),
            *(frontier.get("exception_context_cluster_ids", []) or []),
        ]
    }
    if "" in cluster_ids or cluster_ids != frontier_ids:
        raise ValueError("SAGE.6c frontier must preserve every context cluster")
    if len(events) != len(clusters) or any(
        int(row.get("events", 0) or 0) != 1
        or not bool(row.get("context_preserved", False))
        or bool(row.get("cross_context_merge_performed", False))
        for row in clusters
    ):
        raise ValueError("SAGE.6c context clusters must remain singleton and unmerged")
    for cluster in clusters:
        request_id = _single_string(
            cluster.get("request_ids", []), "cluster request_id"
        )
        event = events_by_request.get(request_id)
        if event is None or str(cluster.get("context_snapshot_hash", "")) != str(
            event.get("context_snapshot_hash", "")
        ):
            raise ValueError("SAGE.6c clusters and event contexts must align exactly")
    if any(
        int(row.get("support", 0) or 0) != 0
        or str(row.get("truth_status", "")) != SAGE6C_TRUTH_STATUS
        for row in [*events, *clusters, frontier]
    ):
        raise ValueError("SAGE.6c records must remain candidate-only")


def build_source_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": str(source.get("config", {}).get("schema_version", "")),
        "game_id": str(source.get("summary", {}).get("game_id", "")),
        "outcome_status": str(source.get("outcome_status", "")),
        "gate_passed": bool(source.get("summary", {}).get("gate_passed", False)),
        "support": int(source.get("support", 0) or 0),
        "truth_status": str(source.get("truth_status", "")),
        "revision_status": str(source.get("revision_status", "")),
        "source_counted_as_scientific_support": False,
    }


def write_sage6d_second_unknown_game_handoff(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE6D_HANDOFF_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _records_by_id(
    source: Mapping[str, Any], section: str, *, id_field: str = "request_id"
) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get(id_field, "")): dict(row)
        for row in source.get(section, []) or []
        if isinstance(row, Mapping) and str(row.get(id_field, ""))
    }


def _cluster_event_request_sort_key(
    row: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
) -> tuple[int, int, str]:
    _, event, request = row
    return (
        int(event.get("budget", 0) or 0),
        int(request.get("source_step", 0) or 0),
        str(request.get("request_id", "")),
    )


def _single_string(values: Sequence[Any], label: str) -> str:
    materialized = [str(value) for value in values if str(value)]
    if len(materialized) != 1:
        raise ValueError(f"expected exactly one {label}, got {materialized}")
    return materialized[0]


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the SAGE.6c frontier into candidate-only followups."
    )
    parser.add_argument(
        "--source-sage6c", default=str(DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH)
    )
    parser.add_argument(
        "--source-sage6b", default=str(DEFAULT_SAGE6B_M3_EXECUTION_PATH)
    )
    parser.add_argument("--out", default=str(DEFAULT_SAGE6D_HANDOFF_PATH))
    args = parser.parse_args(argv)
    run_sage6d_second_unknown_game_handoff(
        source_sage6c_path=args.source_sage6c,
        source_sage6b_path=args.source_sage6b,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
