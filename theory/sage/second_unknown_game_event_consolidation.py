"""SAGE.6c context-preserving consolidation of second-game M3 events."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .second_unknown_game_m3_execution import (
    DEFAULT_SAGE6B_M3_EXECUTION_PATH,
    SAGE6B_EXECUTION_COMPLETED,
    SAGE6B_SCHEMA_VERSION,
    SAGE6B_TRUTH_STATUS,
    budget_from_sage6a_request,
    experiment_hash_verified,
)


DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH = (
    Path("diagnostics") / "sage" / "sage6c_second_unknown_game_event_consolidation.json"
)

SAGE6C_SCHEMA_VERSION = "sage.second_unknown_game_event_consolidation.v1"
SAGE6C_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_6C"
SAGE6C_EVENTS_CONSOLIDATED = (
    "SAGE_SECOND_UNKNOWN_GAME_EVENTS_CONSOLIDATED_CANDIDATE_ONLY"
)
SAGE6C_CONSOLIDATION_INCOMPLETE = (
    "SAGE_SECOND_UNKNOWN_GAME_EVENT_CONSOLIDATION_INCOMPLETE_CANDIDATE_ONLY"
)

POSITIVE_CONTEXT_CANDIDATE_ONLY = "POSITIVE_CONTEXT_CANDIDATE_ONLY"
NEUTRAL_CONTEXT_CANDIDATE_ONLY = "NEUTRAL_CONTEXT_CANDIDATE_ONLY"
NEGATIVE_CONTEXT_CANDIDATE_ONLY = "NEGATIVE_CONTEXT_CANDIDATE_ONLY"
MIXED_CONTEXT_CANDIDATE_ONLY = "MIXED_CONTEXT_CANDIDATE_ONLY"

STABLE_POSITIVE_MULTI_BUDGET_CANDIDATE_ONLY = (
    "STABLE_POSITIVE_MULTI_BUDGET_CANDIDATE_ONLY"
)
LOCAL_NEUTRAL_CANDIDATE_ONLY = "LOCAL_NEUTRAL_CANDIDATE_ONLY"
LOCAL_NEGATIVE_CANDIDATE_ONLY = "LOCAL_NEGATIVE_CANDIDATE_ONLY"
LOCAL_POSITIVE_CANDIDATE_ONLY = "LOCAL_POSITIVE_CANDIDATE_ONLY"
MIXED_EFFECT_GROUP_CANDIDATE_ONLY = "MIXED_EFFECT_GROUP_CANDIDATE_ONLY"

STABLE_POSITIVE_WITH_NEUTRAL_EXCEPTION = (
    "STABLE_POSITIVE_MULTI_BUDGET_WITH_NEUTRAL_EXCEPTION_CANDIDATE_ONLY"
)

DEFAULT_MIN_STABLE_CONTEXTS = 3
DEFAULT_MIN_STABLE_BUDGETS = 2
DEFAULT_MAX_POSITIVE_EFFECT_SPREAD = 0.0
DEFAULT_MIN_DISTINCT_CONTROLS_FOR_A32_REVIEW = 2


def run_sage6c_second_unknown_game_event_consolidation(
    *,
    source_sage6b_path: str | Path = DEFAULT_SAGE6B_M3_EXECUTION_PATH,
    output_path: str | Path | None = None,
    min_stable_contexts: int = DEFAULT_MIN_STABLE_CONTEXTS,
    min_stable_budgets: int = DEFAULT_MIN_STABLE_BUDGETS,
    max_positive_effect_spread: float = DEFAULT_MAX_POSITIVE_EFFECT_SPREAD,
    min_distinct_controls_for_a32_review: int = (
        DEFAULT_MIN_DISTINCT_CONTROLS_FOR_A32_REVIEW
    ),
) -> Dict[str, Any]:
    """Consolidate SAGE.6b events without merging distinct live contexts."""
    source = _load_json(source_sage6b_path)
    validate_sage6c_source(source)
    events = build_sage6c_event_records(source)
    context_clusters = build_context_preserving_clusters(events)
    effect_groups = build_effect_signature_groups(
        events,
        context_clusters=context_clusters,
        min_stable_contexts=min_stable_contexts,
        min_stable_budgets=min_stable_budgets,
        max_positive_effect_spread=max_positive_effect_spread,
    )
    assessment = assess_cross_budget_stability(
        events=events,
        context_clusters=context_clusters,
        effect_groups=effect_groups,
        source_budgets=source.get("summary", {}).get("budgets_available", []),
        min_distinct_controls_for_a32_review=(min_distinct_controls_for_a32_review),
    )
    frontiers = build_candidate_handoff_frontiers(
        assessment=assessment,
        effect_groups=effect_groups,
    )
    gate = build_sage6c_gate(
        source=source,
        events=events,
        context_clusters=context_clusters,
        effect_groups=effect_groups,
        assessment=assessment,
        frontiers=frontiers,
    )
    outcome = (
        SAGE6C_EVENTS_CONSOLIDATED
        if gate and all(gate.values())
        else SAGE6C_CONSOLIDATION_INCOMPLETE
    )
    summary = summarize_sage6c(
        source=source,
        events=events,
        context_clusters=context_clusters,
        effect_groups=effect_groups,
        assessment=assessment,
        frontiers=frontiers,
        gate=gate,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE6C_SCHEMA_VERSION,
            "source_sage6b_path": str(source_sage6b_path),
            "execution_performed": False,
            "min_stable_contexts": int(min_stable_contexts),
            "min_stable_budgets": int(min_stable_budgets),
            "max_positive_effect_spread": float(max_positive_effect_spread),
            "min_distinct_controls_for_a32_review": int(
                min_distinct_controls_for_a32_review
            ),
            "context_consolidation_policy": {
                "one_cluster_per_context_snapshot_hash": True,
                "cross_context_merge_allowed": False,
                "effect_signature_groups_are_comparison_indexes_only": True,
                "neutral_event_is_not_contradiction": True,
                "support_events_are_not_scientific_support": True,
                "stable_pattern_is_not_scientific_verdict": True,
                "a32_a33_write_performed": False,
            },
            "inputs_read": ["SAGE.6b"],
            "artifacts_not_modified": [
                "SAGE.6b",
                "SAGE.6a",
                "SAGE.6",
                "SAGE.5",
                "A32.5",
                "A33.1",
                "A33.2",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
        },
        "source_sage6b_context": build_source_sage6b_context(source),
        "event_records": events,
        "context_clusters": context_clusters,
        "effect_signature_groups": effect_groups,
        "cross_budget_stability_assessment": assessment,
        "candidate_handoff_frontiers": frontiers,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE6C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "support_events_counted_as_support": False,
        "contradiction_events_counted_as_refutation": False,
        "stable_pattern_counted_as_scientific_verdict": False,
        "effect_groups_counted_as_merged_evidence": False,
        "candidate_handoff_frontier_counted_as_revision": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage6c_second_unknown_game_event_consolidation(payload, output_path)
    return payload


def build_sage6c_event_records(source: Mapping[str, Any]) -> List[Dict[str, Any]]:
    requests = {
        str(row.get("request_id", "")): dict(row)
        for row in source.get("selected_execution_requests", []) or []
        if isinstance(row, Mapping)
    }
    records: List[Dict[str, Any]] = []
    raw_rows = sorted(
        (
            dict(row)
            for row in source.get("controlled_experiments", []) or []
            if isinstance(row, Mapping)
            and str(row.get("execution_status", "")) == "EXECUTED"
        ),
        key=lambda row: (
            budget_from_sage6a_request(row),
            int(
                requests.get(str(row.get("request_id", "")), {}).get("source_step", 0)
                or 0
            ),
            str(row.get("request_id", "")),
        ),
    )
    for index, raw in enumerate(raw_rows, start=1):
        request = requests.get(str(raw.get("request_id", "")), {})
        effect_size = float(
            raw.get("controlled_delta", {}).get(
                "effect_size",
                float(raw.get("target_signal", 0.0) or 0.0)
                - float(raw.get("control_signal", 0.0) or 0.0),
            )
            or 0.0
        )
        records.append(
            {
                "event_id": f"sage6c::controlled_event::{index:03d}",
                "request_id": str(raw.get("request_id", "")),
                "source_hypothesis_id": str(raw.get("source_hypothesis_id", "")),
                "source_transition_id": str(raw.get("source_transition_id", "")),
                "game_id": str(raw.get("game_id", "")),
                "budget": budget_from_sage6a_request(raw),
                "source_step": int(request.get("source_step", 0) or 0),
                "context_snapshot_hash": str(raw.get("context_snapshot_hash", "")),
                "hypothesis_family": str(raw.get("hypothesis_family", "")),
                "metric": str(raw.get("metric", "")),
                "target_action": str(raw.get("target_action", "")),
                "target_action_args": (
                    dict(raw.get("target_action_args", {}) or {})
                    if raw.get("target_action_args") is not None
                    else None
                ),
                "control_action": str(raw.get("control_action", "")),
                "target_signal": float(raw.get("target_signal", 0.0) or 0.0),
                "control_signal": float(raw.get("control_signal", 0.0) or 0.0),
                "effect_size": effect_size,
                "absolute_effect_size": abs(effect_size),
                "effect_direction": effect_direction(effect_size),
                "support_events": int(raw.get("support_events", 0) or 0),
                "contradiction_events": int(raw.get("contradiction_events", 0) or 0),
                "neutral_events": int(raw.get("neutral_events", 0) or 0),
                "live_prefix_replay_exact": bool(
                    raw.get("live_prefix_replay_exact", False)
                ),
                "target_context_signature_verified": bool(
                    raw.get("target_context_signature_verified", False)
                ),
                "control_context_signature_verified": bool(
                    raw.get("control_context_signature_verified", False)
                ),
                "support": 0,
                "truth_status": SAGE6C_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "support_events_counted_as_support": False,
                "contradiction_event_counted_as_refutation": False,
                "observation_counted_as_confirmation": False,
            }
        )
    return records


def build_context_preserving_clusters(
    events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("context_snapshot_hash", ""))].append(event)

    clusters: List[Dict[str, Any]] = []
    ordered_groups = sorted(
        grouped.values(),
        key=lambda rows: (
            min(int(row.get("budget", 0) or 0) for row in rows),
            min(int(row.get("source_step", 0) or 0) for row in rows),
            str(rows[0].get("context_snapshot_hash", "")),
        ),
    )
    for index, rows in enumerate(ordered_groups, start=1):
        directions = sorted({str(row.get("effect_direction", "")) for row in rows})
        status = context_candidate_status(directions)
        clusters.append(
            {
                "context_cluster_id": f"sage6c::context_cluster::{index:03d}",
                "context_snapshot_hash": str(rows[0].get("context_snapshot_hash", "")),
                "event_ids": [str(row.get("event_id", "")) for row in rows],
                "request_ids": [str(row.get("request_id", "")) for row in rows],
                "game_id": str(rows[0].get("game_id", "")),
                "budgets": sorted({int(row.get("budget", 0) or 0) for row in rows}),
                "source_steps": sorted(
                    {int(row.get("source_step", 0) or 0) for row in rows}
                ),
                "target_actions": sorted(
                    {str(row.get("target_action", "")) for row in rows}
                ),
                "control_actions": sorted(
                    {str(row.get("control_action", "")) for row in rows}
                ),
                "effect_directions": directions,
                "effect_sizes": [
                    float(row.get("effect_size", 0.0) or 0.0) for row in rows
                ],
                "raw_support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in rows
                ),
                "raw_contradiction_events": sum(
                    int(row.get("contradiction_events", 0) or 0) for row in rows
                ),
                "raw_neutral_events": sum(
                    int(row.get("neutral_events", 0) or 0) for row in rows
                ),
                "events": len(rows),
                "distinct_contexts": 1,
                "context_preserved": True,
                "cross_context_merge_performed": False,
                "candidate_status": status,
                "status": "UNRESOLVED",
                "support": 0,
                "truth_status": SAGE6C_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "wrong_confirmations": 0,
                "cluster_status_counted_as_scientific_verdict": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return clusters


def build_effect_signature_groups(
    events: Sequence[Mapping[str, Any]],
    *,
    context_clusters: Sequence[Mapping[str, Any]],
    min_stable_contexts: int,
    min_stable_budgets: int,
    max_positive_effect_spread: float,
) -> List[Dict[str, Any]]:
    cluster_by_event = {
        str(event_id): str(cluster.get("context_cluster_id", ""))
        for cluster in context_clusters
        for event_id in cluster.get("event_ids", []) or []
    }
    grouped: dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[effect_signature_group_key(event)].append(event)

    groups: List[Dict[str, Any]] = []
    for index, (group_key, rows) in enumerate(sorted(grouped.items()), start=1):
        budgets = sorted({int(row.get("budget", 0) or 0) for row in rows})
        contexts = sorted({str(row.get("context_snapshot_hash", "")) for row in rows})
        effects = [float(row.get("effect_size", 0.0) or 0.0) for row in rows]
        directions = sorted({str(row.get("effect_direction", "")) for row in rows})
        effect_spread = max(effects) - min(effects) if effects else 0.0
        stable_positive = bool(
            directions == ["positive"]
            and len(contexts) >= int(min_stable_contexts)
            and len(budgets) >= int(min_stable_budgets)
            and effect_spread <= float(max_positive_effect_spread)
            and all(bool(row.get("live_prefix_replay_exact", False)) for row in rows)
            and all(
                bool(row.get("target_context_signature_verified", False))
                and bool(row.get("control_context_signature_verified", False))
                for row in rows
            )
        )
        groups.append(
            {
                "effect_group_id": f"sage6c::effect_signature_group::{index:03d}",
                "effect_group_key": group_key,
                "game_id": str(rows[0].get("game_id", "")),
                "hypothesis_family": str(rows[0].get("hypothesis_family", "")),
                "metric": str(rows[0].get("metric", "")),
                "target_action": str(rows[0].get("target_action", "")),
                "control_action": str(rows[0].get("control_action", "")),
                "effect_directions": directions,
                "effect_sizes": effects,
                "effect_size_min": min(effects) if effects else 0.0,
                "effect_size_max": max(effects) if effects else 0.0,
                "effect_size_spread": effect_spread,
                "budgets": budgets,
                "context_snapshot_hashes": contexts,
                "context_cluster_ids": [
                    cluster_by_event[str(row.get("event_id", ""))] for row in rows
                ],
                "event_ids": [str(row.get("event_id", "")) for row in rows],
                "request_ids": [str(row.get("request_id", "")) for row in rows],
                "events": len(rows),
                "contexts": len(contexts),
                "distinct_budgets": len(budgets),
                "raw_support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in rows
                ),
                "raw_contradiction_events": sum(
                    int(row.get("contradiction_events", 0) or 0) for row in rows
                ),
                "raw_neutral_events": sum(
                    int(row.get("neutral_events", 0) or 0) for row in rows
                ),
                "stable_positive_multi_budget": stable_positive,
                "candidate_status": effect_group_candidate_status(
                    directions=directions,
                    stable_positive=stable_positive,
                    distinct_budgets=len(budgets),
                ),
                "comparison_index_only": True,
                "cross_context_merge_performed": False,
                "group_counted_as_merged_evidence": False,
                "status": "UNRESOLVED",
                "support": 0,
                "truth_status": SAGE6C_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "wrong_confirmations": 0,
                "group_status_counted_as_scientific_verdict": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return groups


def assess_cross_budget_stability(
    *,
    events: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
    effect_groups: Sequence[Mapping[str, Any]],
    source_budgets: Sequence[int],
    min_distinct_controls_for_a32_review: int,
) -> Dict[str, Any]:
    stable_groups = [
        row
        for row in effect_groups
        if bool(row.get("stable_positive_multi_budget", False))
    ]
    stable = max(
        stable_groups,
        key=lambda row: (
            int(row.get("contexts", 0) or 0),
            int(row.get("distinct_budgets", 0) or 0),
            str(row.get("effect_group_id", "")),
        ),
        default={},
    )
    stable_event_ids = set(stable.get("event_ids", []) or [])
    exception_events = [
        row for row in events if str(row.get("event_id", "")) not in stable_event_ids
    ]
    cluster_by_event = {
        str(event_id): str(cluster.get("context_cluster_id", ""))
        for cluster in context_clusters
        for event_id in cluster.get("event_ids", []) or []
    }
    source_budget_set = {int(value) for value in source_budgets}
    stable_budget_set = {int(value) for value in stable.get("budgets", []) or []}
    controls = sorted({str(row.get("control_action", "")) for row in events})
    contradictions = sum(int(row.get("contradiction_events", 0) or 0) for row in events)
    neutral_exceptions = [
        row
        for row in exception_events
        if str(row.get("effect_direction", "")) == "neutral"
    ]
    negative_exceptions = [
        row
        for row in exception_events
        if str(row.get("effect_direction", "")) == "negative"
    ]
    stable_across_all_budgets = bool(stable) and stable_budget_set == source_budget_set
    control_diversity_sufficient = len(controls) >= int(
        min_distinct_controls_for_a32_review
    )
    ready_for_handoff_compilation = bool(
        stable_across_all_budgets and contradictions == 0 and not negative_exceptions
    )
    return {
        "assessment_id": "sage6c::cross_budget_stability::001",
        "game_id": str(events[0].get("game_id", "")) if events else "",
        "protocol_signature": {
            "hypothesis_families": sorted(
                {str(row.get("hypothesis_family", "")) for row in events}
            ),
            "metrics": sorted({str(row.get("metric", "")) for row in events}),
            "target_actions": sorted(
                {str(row.get("target_action", "")) for row in events}
            ),
            "control_actions": controls,
        },
        "source_budgets": sorted(source_budget_set),
        "stable_positive_effect_group_id": str(stable.get("effect_group_id", "")),
        "stable_positive_contexts": int(stable.get("contexts", 0) or 0),
        "stable_positive_events": int(stable.get("events", 0) or 0),
        "stable_positive_budgets": list(stable.get("budgets", []) or []),
        "stable_positive_effect_size": (
            float(stable.get("effect_size_min", 0.0) or 0.0) if stable else None
        ),
        "stable_positive_effect_spread": (
            float(stable.get("effect_size_spread", 0.0) or 0.0) if stable else None
        ),
        "stable_positive_across_all_budgets": stable_across_all_budgets,
        "exception_event_ids": [
            str(row.get("event_id", "")) for row in exception_events
        ],
        "exception_context_cluster_ids": [
            cluster_by_event[str(row.get("event_id", ""))] for row in exception_events
        ],
        "neutral_context_exceptions": len(neutral_exceptions),
        "negative_context_exceptions": len(negative_exceptions),
        "context_sensitive_exception_detected": bool(exception_events),
        "raw_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in events
        ),
        "raw_contradiction_events": contradictions,
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in events
        ),
        "distinct_control_actions": len(controls),
        "minimum_distinct_controls_for_a32_review": int(
            min_distinct_controls_for_a32_review
        ),
        "control_diversity_sufficient_for_a32_review": control_diversity_sufficient,
        "ready_for_A32_handoff_compilation": ready_for_handoff_compilation,
        "ready_for_A32_review": bool(
            ready_for_handoff_compilation and control_diversity_sufficient
        ),
        "required_followups": [
            "ADD_DISTINCT_CONTROL_ACTION_PER_BUDGET",
            "REPLICATE_NEUTRAL_CONTEXT",
            "PRESERVE_CONTEXT_CLUSTER_BOUNDARIES",
        ],
        "candidate_status": (
            STABLE_POSITIVE_WITH_NEUTRAL_EXCEPTION
            if stable_across_all_budgets
            and neutral_exceptions
            and not negative_exceptions
            else MIXED_EFFECT_GROUP_CANDIDATE_ONLY
        ),
        "stable_pattern_counted_as_scientific_verdict": False,
        "exception_counted_as_refutation": False,
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": SAGE6C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def build_candidate_handoff_frontiers(
    *,
    assessment: Mapping[str, Any],
    effect_groups: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not bool(assessment.get("ready_for_A32_handoff_compilation", False)):
        return []
    stable_id = str(assessment.get("stable_positive_effect_group_id", ""))
    stable = next(
        (
            row
            for row in effect_groups
            if str(row.get("effect_group_id", "")) == stable_id
        ),
        {},
    )
    return [
        {
            "frontier_id": "sage6c::candidate_handoff_frontier::001",
            "source_assessment_id": str(assessment.get("assessment_id", "")),
            "source_effect_group_id": stable_id,
            "candidate_status": "READY_FOR_A32_HANDOFF_COMPILATION_CANDIDATE_ONLY",
            "ready_for_A32_handoff_compilation": True,
            "ready_for_A32_review": False,
            "ready_for_A32_review_blocked_reason": (
                "INSUFFICIENT_DISTINCT_CONTROL_ACTIONS"
            ),
            "game_id": str(assessment.get("game_id", "")),
            "candidate_mechanism_family": str(stable.get("hypothesis_family", "")),
            "metric": str(stable.get("metric", "")),
            "target_action": str(stable.get("target_action", "")),
            "control_action": str(stable.get("control_action", "")),
            "budgets": list(stable.get("budgets", []) or []),
            "context_cluster_ids": list(stable.get("context_cluster_ids", []) or []),
            "exception_context_cluster_ids": list(
                assessment.get("exception_context_cluster_ids", []) or []
            ),
            "stable_effect_size": assessment.get("stable_positive_effect_size"),
            "raw_support_events": int(assessment.get("raw_support_events", 0) or 0),
            "raw_contradiction_events": int(
                assessment.get("raw_contradiction_events", 0) or 0
            ),
            "raw_neutral_events": int(assessment.get("raw_neutral_events", 0) or 0),
            "required_followups": list(assessment.get("required_followups", []) or []),
            "handoff_frontier_counted_as_revision": False,
            "raw_events_counted_as_support": False,
            "status": "UNRESOLVED",
            "support": 0,
            "truth_status": SAGE6C_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "a32_write_performed": False,
            "a33_write_performed": False,
        }
    ]


def build_sage6c_gate(
    *,
    source: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
    effect_groups: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
    frontiers: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    event_ids = [str(row.get("event_id", "")) for row in events]
    request_ids = [str(row.get("request_id", "")) for row in events]
    context_hashes = [str(row.get("context_snapshot_hash", "")) for row in events]
    source_budgets = {
        int(value) for value in source.get("summary", {}).get("budgets_available", [])
    }
    event_budgets = {int(row.get("budget", 0) or 0) for row in events}
    raw_total = sum(
        int(row.get("support_events", 0) or 0)
        + int(row.get("contradiction_events", 0) or 0)
        + int(row.get("neutral_events", 0) or 0)
        for row in events
    )
    return {
        "source_sage6b_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source.get("gate", {}).values()),
        "source_execution_count_reproduced": len(events)
        == int(source.get("summary", {}).get("requests_executed", 0) or 0),
        "event_ids_unique": "" not in event_ids
        and len(event_ids) == len(set(event_ids)),
        "request_ids_unique": "" not in request_ids
        and len(request_ids) == len(set(request_ids)),
        "context_snapshot_hashes_unique": "" not in context_hashes
        and len(context_hashes) == len(set(context_hashes)),
        "all_source_budgets_represented": event_budgets == source_budgets,
        "all_replays_exact_and_hash_verified": bool(events)
        and all(
            bool(row.get("live_prefix_replay_exact", False))
            and bool(row.get("target_context_signature_verified", False))
            and bool(row.get("control_context_signature_verified", False))
            for row in events
        ),
        "one_context_cluster_per_event": len(context_clusters) == len(events)
        and all(int(row.get("events", 0) or 0) == 1 for row in context_clusters),
        "no_cross_context_merge": all(
            bool(row.get("context_preserved", False))
            and not bool(row.get("cross_context_merge_performed", False))
            for row in context_clusters
        )
        and all(
            not bool(row.get("cross_context_merge_performed", False))
            and bool(row.get("comparison_index_only", False))
            for row in effect_groups
        ),
        "raw_event_accounting_exact": raw_total == len(events),
        "stable_positive_group_spans_all_budgets": bool(
            assessment.get("stable_positive_across_all_budgets", False)
        ),
        "neutral_exception_not_counted_as_refutation": int(
            assessment.get("neutral_context_exceptions", 0) or 0
        )
        >= 1
        and int(assessment.get("negative_context_exceptions", 0) or 0) == 0
        and not bool(assessment.get("exception_counted_as_refutation", False)),
        "handoff_frontier_compilable_but_not_a32_ready": len(frontiers) == 1
        and bool(frontiers[0].get("ready_for_A32_handoff_compilation", False))
        and not bool(frontiers[0].get("ready_for_A32_review", False))
        and not bool(
            assessment.get("control_diversity_sufficient_for_a32_review", False)
        ),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0
            and str(row.get("truth_status", "")) == SAGE6C_TRUTH_STATUS
            for row in [
                *events,
                *context_clusters,
                *effect_groups,
                assessment,
                *frontiers,
            ]
        ),
        "source_registry_quarantine_preserved": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        )
        == 0
        and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
        and not bool(source.get("scope_generalization_performed", False)),
    }


def summarize_sage6c(
    *,
    source: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
    effect_groups: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
    frontiers: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
    outcome: str,
) -> Dict[str, Any]:
    stable_groups = [
        row
        for row in effect_groups
        if bool(row.get("stable_positive_multi_budget", False))
    ]
    neutral_groups = [
        row
        for row in effect_groups
        if str(row.get("candidate_status", "")) == LOCAL_NEUTRAL_CANDIDATE_ONLY
    ]
    return {
        "source_sage6b_outcome_status": str(source.get("outcome_status", "")),
        "game_id": str(assessment.get("game_id", "")),
        "budgets": list(assessment.get("source_budgets", []) or []),
        "event_records": len(events),
        "context_clusters": len(context_clusters),
        "singleton_context_clusters": sum(
            1 for row in context_clusters if int(row.get("events", 0) or 0) == 1
        ),
        "cross_context_merges_performed": sum(
            1
            for row in context_clusters
            if bool(row.get("cross_context_merge_performed", False))
        ),
        "all_contexts_preserved_without_merge": all(
            bool(row.get("context_preserved", False))
            and not bool(row.get("cross_context_merge_performed", False))
            for row in context_clusters
        ),
        "effect_signature_groups": len(effect_groups),
        "stable_positive_multi_budget_groups": len(stable_groups),
        "neutral_effect_groups": len(neutral_groups),
        "stable_positive_contexts": int(
            assessment.get("stable_positive_contexts", 0) or 0
        ),
        "stable_positive_events": int(assessment.get("stable_positive_events", 0) or 0),
        "stable_positive_budgets": list(
            assessment.get("stable_positive_budgets", []) or []
        ),
        "stable_positive_effect_size": assessment.get("stable_positive_effect_size"),
        "stable_positive_effect_spread": assessment.get(
            "stable_positive_effect_spread"
        ),
        "stable_positive_across_all_budgets": bool(
            assessment.get("stable_positive_across_all_budgets", False)
        ),
        "neutral_context_exceptions": int(
            assessment.get("neutral_context_exceptions", 0) or 0
        ),
        "negative_context_exceptions": int(
            assessment.get("negative_context_exceptions", 0) or 0
        ),
        "context_sensitive_exception_detected": bool(
            assessment.get("context_sensitive_exception_detected", False)
        ),
        "distinct_control_actions": int(
            assessment.get("distinct_control_actions", 0) or 0
        ),
        "control_diversity_sufficient_for_a32_review": bool(
            assessment.get("control_diversity_sufficient_for_a32_review", False)
        ),
        "candidate_handoff_frontiers": len(frontiers),
        "ready_for_A32_handoff_compilation": sum(
            1
            for row in frontiers
            if bool(row.get("ready_for_A32_handoff_compilation", False))
        ),
        "ready_for_A32_review": sum(
            1 for row in frontiers if bool(row.get("ready_for_A32_review", False))
        ),
        "raw_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in events
        ),
        "raw_contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in events
        ),
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in events
        ),
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "outcome_status": outcome,
        "support": 0,
        "truth_status": SAGE6C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def build_source_sage6b_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(source.get("summary", {}) or {})
    return {
        "source_outcome_status": str(source.get("outcome_status", "")),
        "game_id": str(summary.get("game_id", "")),
        "budgets": list(summary.get("budgets_available", []) or []),
        "requests_selected": int(summary.get("requests_selected", 0) or 0),
        "requests_executed": int(summary.get("requests_executed", 0) or 0),
        "requests_blocked": int(summary.get("requests_blocked", 0) or 0),
        "live_prefix_replay_exact_events": int(
            summary.get("live_prefix_replay_exact_events", 0) or 0
        ),
        "context_snapshot_hash_verified_events": int(
            summary.get("context_snapshot_hash_verified_events", 0) or 0
        ),
        "source_scoped_mechanics_reused": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            source.get("cross_game_mechanics_imported", 0) or 0
        ),
        "scope_generalization_performed": bool(
            source.get("scope_generalization_performed", False)
        ),
        "source_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE6C_TRUTH_STATUS,
    }


def validate_sage6c_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6B_SCHEMA_VERSION:
        raise ValueError("SAGE.6b schema version is not supported by SAGE.6c")
    if str(source.get("outcome_status", "")) != SAGE6B_EXECUTION_COMPLETED:
        raise ValueError("SAGE.6c requires a completed SAGE.6b execution")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.6b source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE6B_TRUTH_STATUS:
        raise ValueError("SAGE.6b truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.6b source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.6b support must remain 0")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.6b cannot perform a scientific verdict")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.6b wrong_confirmations must remain 0")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.6b cannot write A32/A33")
    if (
        bool(source.get("policy_result_counted_as_confirmation", False))
        or bool(source.get("support_events_counted_as_support", False))
        or bool(source.get("mini_frontier_execution_counted_as_evidence", False))
    ):
        raise ValueError("SAGE.6b execution cannot count as evidence")
    if bool(source.get("contradiction_events_counted_as_refutation", False)):
        raise ValueError("SAGE.6b raw contradictions cannot count as refutations")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6b cannot import source-game mechanics")
    if bool(source.get("scope_generalization_performed", False)):
        raise ValueError("SAGE.6b cannot generalize the source registry scope")
    gate = dict(source.get("gate", {}) or {})
    if (
        not gate
        or not all(bool(value) for value in gate.values())
        or not bool(summary.get("gate_passed", False))
    ):
        raise ValueError("every SAGE.6b source gate must pass")
    game_id = str(summary.get("game_id", ""))
    if not game_id or game_id != str(config.get("game_id", "")):
        raise ValueError("SAGE.6b game identity must align")
    budgets = [int(value) for value in summary.get("budgets_available", []) or []]
    if not budgets or budgets != [
        int(value) for value in config.get("budgets", []) or []
    ]:
        raise ValueError("SAGE.6b budgets must align")
    selected = [
        row
        for row in source.get("selected_execution_requests", []) or []
        if isinstance(row, Mapping)
    ]
    experiments = [
        row
        for row in source.get("controlled_experiments", []) or []
        if isinstance(row, Mapping)
    ]
    if len(selected) != int(summary.get("requests_selected", 0) or 0):
        raise ValueError("SAGE.6b selected request count must match its summary")
    if len(experiments) != int(summary.get("requests_executed", 0) or 0):
        raise ValueError("SAGE.6b experiment count must match its summary")
    if (
        source.get("blocked_replay_events", [])
        or int(summary.get("requests_blocked", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6c requires SAGE.6b without blocked replays")
    selected_ids = {str(row.get("request_id", "")) for row in selected}
    experiment_ids = {str(row.get("request_id", "")) for row in experiments}
    if "" in selected_ids or selected_ids != experiment_ids:
        raise ValueError("every SAGE.6b selected request must have one experiment")
    if not all(
        str(row.get("execution_status", "")) == "EXECUTED"
        and bool(row.get("live_prefix_replay_exact", False))
        and experiment_hash_verified(row)
        and int(row.get("support", 0) or 0) == 0
        and str(row.get("truth_status", "")) == SAGE6B_TRUTH_STATUS
        for row in experiments
    ):
        raise ValueError("SAGE.6b experiments must remain exact and candidate-only")
    audit = list(source.get("selection_audit", []) or [])
    if not audit or any(
        bool(row.get("outcome_metrics_read_for_selection", False))
        for row in audit
        if isinstance(row, Mapping)
    ):
        raise ValueError("SAGE.6b selection must remain pre-execution")


def context_candidate_status(directions: Sequence[str]) -> str:
    if directions == ["positive"]:
        return POSITIVE_CONTEXT_CANDIDATE_ONLY
    if directions == ["neutral"]:
        return NEUTRAL_CONTEXT_CANDIDATE_ONLY
    if directions == ["negative"]:
        return NEGATIVE_CONTEXT_CANDIDATE_ONLY
    return MIXED_CONTEXT_CANDIDATE_ONLY


def effect_group_candidate_status(
    *,
    directions: Sequence[str],
    stable_positive: bool,
    distinct_budgets: int,
) -> str:
    if stable_positive:
        return STABLE_POSITIVE_MULTI_BUDGET_CANDIDATE_ONLY
    if directions == ["neutral"]:
        return LOCAL_NEUTRAL_CANDIDATE_ONLY
    if directions == ["negative"]:
        return LOCAL_NEGATIVE_CANDIDATE_ONLY
    if directions == ["positive"] and distinct_budgets >= 1:
        return LOCAL_POSITIVE_CANDIDATE_ONLY
    return MIXED_EFFECT_GROUP_CANDIDATE_ONLY


def effect_signature_group_key(event: Mapping[str, Any]) -> str:
    payload = {
        "hypothesis_family": str(event.get("hypothesis_family", "")),
        "metric": str(event.get("metric", "")),
        "target_action": str(event.get("target_action", "")),
        "target_action_args": event.get("target_action_args"),
        "control_action": str(event.get("control_action", "")),
        "effect_direction": str(event.get("effect_direction", "")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def effect_direction(effect_size: float) -> str:
    if float(effect_size) > 0:
        return "positive"
    if float(effect_size) < 0:
        return "negative"
    return "neutral"


def write_sage6c_second_unknown_game_event_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate SAGE.6b events without merging contexts."
    )
    parser.add_argument(
        "--source-sage6b", default=str(DEFAULT_SAGE6B_M3_EXECUTION_PATH)
    )
    parser.add_argument("--out", default=str(DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH))
    parser.add_argument(
        "--min-stable-contexts", type=int, default=DEFAULT_MIN_STABLE_CONTEXTS
    )
    parser.add_argument(
        "--min-stable-budgets", type=int, default=DEFAULT_MIN_STABLE_BUDGETS
    )
    parser.add_argument(
        "--max-positive-effect-spread",
        type=float,
        default=DEFAULT_MAX_POSITIVE_EFFECT_SPREAD,
    )
    parser.add_argument(
        "--min-distinct-controls-for-a32-review",
        type=int,
        default=DEFAULT_MIN_DISTINCT_CONTROLS_FOR_A32_REVIEW,
    )
    args = parser.parse_args(argv)
    run_sage6c_second_unknown_game_event_consolidation(
        source_sage6b_path=args.source_sage6b,
        output_path=args.out,
        min_stable_contexts=args.min_stable_contexts,
        min_stable_budgets=args.min_stable_budgets,
        max_positive_effect_spread=args.max_positive_effect_spread,
        min_distinct_controls_for_a32_review=(
            args.min_distinct_controls_for_a32_review
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
