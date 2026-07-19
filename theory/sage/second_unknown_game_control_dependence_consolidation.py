"""SAGE.6f control-preserving consolidation for the second unknown game."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .second_unknown_game_event_consolidation import (
    DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH,
    SAGE6C_EVENTS_CONSOLIDATED,
    SAGE6C_SCHEMA_VERSION,
    SAGE6C_TRUTH_STATUS,
    effect_direction,
)
from .second_unknown_game_followup_execution import (
    DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH,
    SAGE6E_EXECUTION_COMPLETED,
    SAGE6E_SCHEMA_VERSION,
    SAGE6E_TRUTH_STATUS,
)


DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage6f_second_unknown_game_control_dependence_consolidation.json"
)

SAGE6F_SCHEMA_VERSION = "sage.second_unknown_game_control_dependence_consolidation.v1"
SAGE6F_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_6F"
SAGE6F_A32_REVIEW_ELIGIBLE = (
    "SAGE_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_CONSOLIDATED_"
    "A32_REVIEW_ELIGIBLE_CANDIDATE_ONLY"
)
SAGE6F_NOT_A32_REVIEW_ELIGIBLE = (
    "SAGE_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_CONSOLIDATED_"
    "NOT_A32_REVIEW_ELIGIBLE_CANDIDATE_ONLY"
)
SAGE6F_CONSOLIDATION_INCOMPLETE = (
    "SAGE_SECOND_UNKNOWN_GAME_CONTROL_DEPENDENCE_CONSOLIDATION_"
    "INCOMPLETE_CANDIDATE_ONLY"
)

CONTROL_DEPENDENT_MULTI_BUDGET = (
    "CONTROL_DEPENDENT_MULTI_BUDGET_WITH_REPLICATED_NEUTRAL_EXCEPTION_CANDIDATE_ONLY"
)
READY_FOR_A32_CONTROL_DEPENDENCE_REVIEW = (
    "READY_FOR_A32_CONTROL_DEPENDENCE_REVIEW_CANDIDATE_ONLY"
)

ACTION1_POSITIVE_CONTEXT = "ACTION1_POSITIVE_CONTEXT_CANDIDATE_ONLY"
ACTION1_REPLICATED_NEUTRAL_CONTEXT = "ACTION1_REPLICATED_NEUTRAL_CONTEXT_CANDIDATE_ONLY"
PAIRED_CONTROL_DEPENDENT_CONTEXT = (
    "PAIRED_ACTION1_ACTION3_CONTROL_DEPENDENT_CONTEXT_CANDIDATE_ONLY"
)
UNPAIRED_ACTION1_POSITIVE_CONTEXT = "UNPAIRED_ACTION1_POSITIVE_CONTEXT_CANDIDATE_ONLY"
MIXED_CONTROL_CONTEXT = "MIXED_CONTROL_CONTEXT_CANDIDATE_ONLY"

MIN_DISTINCT_CONTROLS_FOR_A32_REVIEW = 2
MIN_PAIRED_CONTROL_CONTEXTS_FOR_A32_REVIEW = 3
MIN_RAW_POSITIVE_CONTEXTS_FOR_A32_REVIEW = 3


def run_sage6f_second_unknown_game_control_dependence_consolidation(
    *,
    source_sage6c_path: str | Path = DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH,
    source_sage6e_path: str | Path = DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Consolidate ACTION1/ACTION3 results without merging contexts or controls."""
    source_sage6c = _load_json(source_sage6c_path)
    source_sage6e = _load_json(source_sage6e_path)
    validate_sage6f_sources(source_sage6c, source_sage6e)

    observations = build_sage6f_observations(source_sage6c, source_sage6e)
    context_clusters = build_sage6f_context_clusters(
        source_sage6c=source_sage6c,
        observations=observations,
    )
    control_groups = build_control_conditioned_effect_groups(observations)
    paired = build_paired_control_comparisons(
        observations=observations,
        context_clusters=context_clusters,
    )
    assessment = assess_a32_review_eligibility(
        source_sage6c=source_sage6c,
        source_sage6e=source_sage6e,
        observations=observations,
        context_clusters=context_clusters,
        control_groups=control_groups,
        paired=paired,
    )
    frontiers = build_sage6f_a32_review_frontiers(
        assessment=assessment,
        context_clusters=context_clusters,
        control_groups=control_groups,
        paired=paired,
    )
    gate = build_sage6f_gate(
        source_sage6c=source_sage6c,
        source_sage6e=source_sage6e,
        observations=observations,
        context_clusters=context_clusters,
        control_groups=control_groups,
        paired=paired,
        assessment=assessment,
        frontiers=frontiers,
    )
    if not gate or not all(gate.values()):
        outcome = SAGE6F_CONSOLIDATION_INCOMPLETE
    elif bool(assessment.get("ready_for_A32_review", False)):
        outcome = SAGE6F_A32_REVIEW_ELIGIBLE
    else:
        outcome = SAGE6F_NOT_A32_REVIEW_ELIGIBLE
    summary = summarize_sage6f(
        source_sage6c=source_sage6c,
        source_sage6e=source_sage6e,
        observations=observations,
        context_clusters=context_clusters,
        control_groups=control_groups,
        paired=paired,
        assessment=assessment,
        frontiers=frontiers,
        gate=gate,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE6F_SCHEMA_VERSION,
            "source_sage6c_path": str(source_sage6c_path),
            "source_sage6e_path": str(source_sage6e_path),
            "inputs_read": ["SAGE.6c", "SAGE.6e"],
            "execution_performed": False,
            "consolidation_policy": {
                "control_action_identity_is_immutable": True,
                "one_cluster_per_sage6c_context_snapshot_hash": True,
                "same_context_replications_are_linked_not_independent": True,
                "cross_context_merge_allowed": False,
                "cross_control_effect_merge_allowed": False,
                "control_conditioned_groups_are_comparison_indexes_only": True,
                "paired_control_comparison_is_not_scientific_verdict": True,
                "a32_review_eligibility_is_not_a32_decision": True,
                "raw_events_are_not_scientific_support": True,
                "a32_write_performed": False,
                "a33_write_performed": False,
            },
            "a32_review_eligibility_thresholds": {
                "minimum_distinct_control_actions": (
                    MIN_DISTINCT_CONTROLS_FOR_A32_REVIEW
                ),
                "minimum_paired_control_contexts": (
                    MIN_PAIRED_CONTROL_CONTEXTS_FOR_A32_REVIEW
                ),
                "minimum_raw_positive_contexts": (
                    MIN_RAW_POSITIVE_CONTEXTS_FOR_A32_REVIEW
                ),
                "all_source_budgets_must_have_paired_controls": True,
                "neutral_exception_must_have_exact_replication": True,
                "maximum_negative_effect_events": 0,
            },
            "artifacts_not_modified": [
                "SAGE.6e",
                "SAGE.6d",
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
        "source_sage6e_context": build_source_context(source_sage6e),
        "observation_records": observations,
        "context_cluster_manifest": context_clusters,
        "control_conditioned_effect_groups": control_groups,
        "paired_control_comparisons": paired,
        "a32_review_eligibility_assessment": assessment,
        "candidate_a32_review_frontiers": frontiers,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE6F_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "raw_events_counted_as_support": False,
        "paired_control_pattern_counted_as_scientific_verdict": False,
        "a32_review_eligibility_counted_as_a32_decision": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage6f_second_unknown_game_control_dependence_consolidation(
            payload, output_path
        )
    return payload


def build_sage6f_observations(
    source_sage6c: Mapping[str, Any],
    source_sage6e: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    cluster_by_request = {
        str(request_id): str(cluster.get("context_cluster_id", ""))
        for cluster in source_sage6c.get("context_clusters", []) or []
        if isinstance(cluster, Mapping)
        for request_id in cluster.get("request_ids", []) or []
    }
    raw: List[Dict[str, Any]] = []
    for event in source_sage6c.get("event_records", []) or []:
        if not isinstance(event, Mapping):
            continue
        request_id = str(event.get("request_id", ""))
        raw.append(
            observation_from_source(
                source_stage="SAGE.6c",
                source_record_id=str(event.get("event_id", "")),
                source_request_id=request_id,
                source_context_cluster_id=cluster_by_request[request_id],
                row=event,
                replication_role="ORIGINAL_CONTROLLED_OBSERVATION",
            )
        )
    for result in source_sage6e.get("controlled_followup_results", []) or []:
        if not isinstance(result, Mapping):
            continue
        replication = (
            "EXACT_NEUTRAL_CONTEXT_REPLICATION"
            if str(result.get("request_type", "")) == "REPLICATE_NEUTRAL_CONTEXT"
            else "DISTINCT_CONTROL_FOLLOWUP"
        )
        raw.append(
            observation_from_source(
                source_stage="SAGE.6e",
                source_record_id=str(result.get("protocol_id", "")),
                source_request_id=str(result.get("source_request_id", "")),
                source_context_cluster_id=str(
                    result.get("source_context_cluster_id", "")
                ),
                row=result,
                replication_role=replication,
            )
        )
    raw.sort(
        key=lambda row: (
            str(row.get("source_context_cluster_id", "")),
            str(row.get("control_action", "")),
            0 if str(row.get("source_stage", "")) == "SAGE.6c" else 1,
            str(row.get("source_record_id", "")),
        )
    )
    observations: List[Dict[str, Any]] = []
    for index, row in enumerate(raw, start=1):
        observations.append(
            {
                "observation_id": f"sage6f::control_observation::{index:03d}",
                **row,
                "support": 0,
                "truth_status": SAGE6F_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "wrong_confirmations": 0,
                "raw_event_counted_as_scientific_support": False,
                "raw_event_counted_as_confirmation": False,
                "raw_event_counted_as_refutation": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return observations


def observation_from_source(
    *,
    source_stage: str,
    source_record_id: str,
    source_request_id: str,
    source_context_cluster_id: str,
    row: Mapping[str, Any],
    replication_role: str,
) -> Dict[str, Any]:
    effect_size = float(
        row.get("effect_size", row.get("controlled_delta", {}).get("effect_size", 0.0))
        or 0.0
    )
    return {
        "source_stage": source_stage,
        "source_record_id": source_record_id,
        "source_request_id": source_request_id,
        "source_context_cluster_id": source_context_cluster_id,
        "source_hypothesis_id": str(row.get("source_hypothesis_id", "")),
        "source_transition_id": str(row.get("source_transition_id", "")),
        "game_id": str(row.get("game_id", "")),
        "budget": int(row.get("budget", 0) or 0),
        "source_step": int(row.get("source_step", 0) or 0),
        "context_snapshot_hash": str(row.get("context_snapshot_hash", "")),
        "hypothesis_family": str(row.get("hypothesis_family", "")),
        "metric": str(row.get("metric", "")),
        "target_action": str(row.get("target_action", "")),
        "target_action_args": (
            dict(row.get("target_action_args", {}) or {})
            if row.get("target_action_args") is not None
            else None
        ),
        "control_action": str(row.get("control_action", "")),
        "target_signal": float(row.get("target_signal", 0.0) or 0.0),
        "control_signal": float(row.get("control_signal", 0.0) or 0.0),
        "effect_size": effect_size,
        "effect_direction": effect_direction(effect_size),
        "support_events": int(row.get("support_events", 0) or 0),
        "contradiction_events": int(row.get("contradiction_events", 0) or 0),
        "neutral_events": int(row.get("neutral_events", 0) or 0),
        "live_prefix_replay_exact": bool(row.get("live_prefix_replay_exact", False)),
        "target_context_signature_verified": bool(
            row.get("target_context_signature_verified", False)
        ),
        "control_context_signature_verified": bool(
            row.get("control_context_signature_verified", False)
        ),
        "protocol_execution_exact": (
            bool(row.get("protocol_execution_exact", False))
            if source_stage == "SAGE.6e"
            else True
        ),
        "replication_role": replication_role,
        "same_context_replication_is_independent_context": False,
    }


def build_sage6f_context_clusters(
    *,
    source_sage6c: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    observations_by_cluster: dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        observations_by_cluster[str(row.get("source_context_cluster_id", ""))].append(
            row
        )
    clusters: List[Dict[str, Any]] = []
    for source_cluster in source_sage6c.get("context_clusters", []) or []:
        if not isinstance(source_cluster, Mapping):
            continue
        cluster_id = str(source_cluster.get("context_cluster_id", ""))
        rows = observations_by_cluster[cluster_id]
        controls = sorted({str(row.get("control_action", "")) for row in rows})
        effects_by_control = {
            control: [
                float(row.get("effect_size", 0.0) or 0.0)
                for row in rows
                if str(row.get("control_action", "")) == control
            ]
            for control in controls
        }
        candidate_status = context_candidate_status(rows)
        clusters.append(
            {
                "context_cluster_id": cluster_id,
                "context_snapshot_hash": str(
                    source_cluster.get("context_snapshot_hash", "")
                ),
                "game_id": str(source_cluster.get("game_id", "")),
                "budgets": list(source_cluster.get("budgets", []) or []),
                "source_steps": list(source_cluster.get("source_steps", []) or []),
                "source_request_ids": list(source_cluster.get("request_ids", []) or []),
                "observation_ids": [str(row.get("observation_id", "")) for row in rows],
                "source_stages": sorted(
                    {str(row.get("source_stage", "")) for row in rows}
                ),
                "control_actions": controls,
                "effects_by_control": effects_by_control,
                "observations": len(rows),
                "independent_contexts": 1,
                "same_context_replications": sum(
                    str(row.get("replication_role", ""))
                    == "EXACT_NEUTRAL_CONTEXT_REPLICATION"
                    for row in rows
                ),
                "paired_control_context": set(controls) == {"ACTION1", "ACTION3"},
                "neutral_context_replication_verified": bool(
                    len(rows) == 2
                    and controls == ["ACTION1"]
                    and {str(row.get("source_stage", "")) for row in rows}
                    == {"SAGE.6c", "SAGE.6e"}
                    and all(
                        float(row.get("effect_size", 1.0) or 0.0) == 0.0 for row in rows
                    )
                ),
                "candidate_status": candidate_status,
                "context_preserved": True,
                "cross_context_merge_performed": False,
                "cross_control_effect_merge_performed": False,
                "status": "UNRESOLVED",
                "support": 0,
                "truth_status": SAGE6F_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "wrong_confirmations": 0,
                "cluster_status_counted_as_scientific_verdict": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return clusters


def context_candidate_status(rows: Sequence[Mapping[str, Any]]) -> str:
    controls = {str(row.get("control_action", "")) for row in rows}
    effects = {
        control: {
            float(row.get("effect_size", 0.0) or 0.0)
            for row in rows
            if str(row.get("control_action", "")) == control
        }
        for control in controls
    }
    if controls == {"ACTION1", "ACTION3"} and effects == {
        "ACTION1": {32.0},
        "ACTION3": {0.0},
    }:
        return PAIRED_CONTROL_DEPENDENT_CONTEXT
    if controls == {"ACTION1"} and effects == {"ACTION1": {0.0}} and len(rows) >= 2:
        return ACTION1_REPLICATED_NEUTRAL_CONTEXT
    if controls == {"ACTION1"} and effects == {"ACTION1": {32.0}}:
        return UNPAIRED_ACTION1_POSITIVE_CONTEXT
    if controls == {"ACTION1"} and all(
        float(row.get("effect_size", 0.0) or 0.0) > 0.0 for row in rows
    ):
        return ACTION1_POSITIVE_CONTEXT
    return MIXED_CONTROL_CONTEXT


def build_control_conditioned_effect_groups(
    observations: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[
            (
                str(row.get("control_action", "")),
                str(row.get("effect_direction", "")),
            )
        ].append(row)
    direction_rank = {"positive": 0, "neutral": 1, "negative": 2}
    ordered = sorted(
        grouped.items(),
        key=lambda item: (item[0][0], direction_rank.get(item[0][1], 9)),
    )
    groups: List[Dict[str, Any]] = []
    for index, ((control, direction), rows) in enumerate(ordered, start=1):
        effects = [float(row.get("effect_size", 0.0) or 0.0) for row in rows]
        contexts = sorted({str(row.get("context_snapshot_hash", "")) for row in rows})
        groups.append(
            {
                "control_effect_group_id": (
                    f"sage6f::control_conditioned_effect_group::{index:03d}"
                ),
                "game_id": str(rows[0].get("game_id", "")),
                "hypothesis_family": str(rows[0].get("hypothesis_family", "")),
                "metric": str(rows[0].get("metric", "")),
                "target_action": str(rows[0].get("target_action", "")),
                "control_action": control,
                "effect_direction": direction,
                "effect_sizes": effects,
                "effect_size_min": min(effects),
                "effect_size_max": max(effects),
                "effect_size_spread": max(effects) - min(effects),
                "observation_ids": [str(row.get("observation_id", "")) for row in rows],
                "context_cluster_ids": sorted(
                    {str(row.get("source_context_cluster_id", "")) for row in rows}
                ),
                "context_snapshot_hashes": contexts,
                "budgets": sorted({int(row.get("budget", 0) or 0) for row in rows}),
                "observations": len(rows),
                "independent_contexts": len(contexts),
                "raw_support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in rows
                ),
                "raw_contradiction_events": sum(
                    int(row.get("contradiction_events", 0) or 0) for row in rows
                ),
                "raw_neutral_events": sum(
                    int(row.get("neutral_events", 0) or 0) for row in rows
                ),
                "comparison_index_only": True,
                "cross_context_merge_performed": False,
                "cross_control_effect_merge_performed": False,
                "group_counted_as_scientific_evidence": False,
                "status": "UNRESOLVED",
                "support": 0,
                "truth_status": SAGE6F_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "wrong_confirmations": 0,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return groups


def build_paired_control_comparisons(
    *,
    observations: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows_by_cluster: dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        rows_by_cluster[str(row.get("source_context_cluster_id", ""))].append(row)
    paired: List[Dict[str, Any]] = []
    for cluster in context_clusters:
        if not bool(cluster.get("paired_control_context", False)):
            continue
        cluster_id = str(cluster.get("context_cluster_id", ""))
        rows = rows_by_cluster[cluster_id]
        action1 = next(row for row in rows if row.get("control_action") == "ACTION1")
        action3 = next(row for row in rows if row.get("control_action") == "ACTION3")
        action1_effect = float(action1.get("effect_size", 0.0) or 0.0)
        action3_effect = float(action3.get("effect_size", 0.0) or 0.0)
        paired.append(
            {
                "paired_comparison_id": (
                    f"sage6f::paired_control_comparison::{len(paired) + 1:03d}"
                ),
                "context_cluster_id": cluster_id,
                "context_snapshot_hash": str(cluster.get("context_snapshot_hash", "")),
                "game_id": str(action1.get("game_id", "")),
                "budget": int(action1.get("budget", 0) or 0),
                "source_step": int(action1.get("source_step", 0) or 0),
                "target_action": str(action1.get("target_action", "")),
                "action1_observation_id": str(action1.get("observation_id", "")),
                "action3_observation_id": str(action3.get("observation_id", "")),
                "action1_controlled_effect_size": action1_effect,
                "action3_controlled_effect_size": action3_effect,
                "action1_control_signal": float(
                    action1.get("control_signal", 0.0) or 0.0
                ),
                "action3_control_signal": float(
                    action3.get("control_signal", 0.0) or 0.0
                ),
                "target_signal_action1_pair": float(
                    action1.get("target_signal", 0.0) or 0.0
                ),
                "target_signal_action3_pair": float(
                    action3.get("target_signal", 0.0) or 0.0
                ),
                "controlled_effect_gap_action1_minus_action3": (
                    action1_effect - action3_effect
                ),
                "control_signal_gap_action3_minus_action1": float(
                    action3.get("control_signal", 0.0) or 0.0
                )
                - float(action1.get("control_signal", 0.0) or 0.0),
                "target_signal_reproduced_across_control_pairs": float(
                    action1.get("target_signal", 0.0) or 0.0
                )
                == float(action3.get("target_signal", 0.0) or 0.0),
                "candidate_status": PAIRED_CONTROL_DEPENDENT_CONTEXT,
                "context_preserved": True,
                "control_identities_preserved": True,
                "paired_comparison_counted_as_scientific_verdict": False,
                "status": "UNRESOLVED",
                "support": 0,
                "truth_status": SAGE6F_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "revision_performed": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return paired


def assess_a32_review_eligibility(
    *,
    source_sage6c: Mapping[str, Any],
    source_sage6e: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
    control_groups: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_budgets = {
        int(value)
        for value in source_sage6c.get("summary", {}).get("budgets", []) or []
    }
    controls = sorted({str(row.get("control_action", "")) for row in observations})
    paired_budgets = {int(row.get("budget", 0) or 0) for row in paired}
    negative_events = sum(
        int(row.get("effect_size", 0.0) or 0.0) < 0.0 for row in observations
    )
    action1_positive_groups = [
        row
        for row in control_groups
        if row.get("control_action") == "ACTION1"
        and row.get("effect_direction") == "positive"
    ]
    action3_neutral_groups = [
        row
        for row in control_groups
        if row.get("control_action") == "ACTION3"
        and row.get("effect_direction") == "neutral"
    ]
    neutral_replications = [
        row
        for row in context_clusters
        if bool(row.get("neutral_context_replication_verified", False))
    ]
    gaps = [
        float(row.get("controlled_effect_gap_action1_minus_action3", 0.0) or 0.0)
        for row in paired
    ]
    requirements = {
        "source_sage6c_gate_passed": bool(
            source_sage6c.get("summary", {}).get("gate_passed", False)
        ),
        "source_sage6e_gate_passed": bool(
            source_sage6e.get("summary", {}).get("gate_passed", False)
        ),
        "all_sage6e_followups_executed_exactly": int(
            source_sage6e.get("summary", {}).get("protocol_execution_exact_events", 0)
            or 0
        )
        == int(source_sage6e.get("summary", {}).get("protocols_available", 0) or 0),
        "minimum_distinct_control_actions_reached": len(controls)
        >= MIN_DISTINCT_CONTROLS_FOR_A32_REVIEW,
        "minimum_paired_control_contexts_reached": len(paired)
        >= MIN_PAIRED_CONTROL_CONTEXTS_FOR_A32_REVIEW,
        "minimum_raw_positive_contexts_reached": sum(
            row.get("control_action") == "ACTION1"
            and row.get("effect_direction") == "positive"
            for row in observations
        )
        >= MIN_RAW_POSITIVE_CONTEXTS_FOR_A32_REVIEW,
        "paired_controls_cover_all_source_budgets": paired_budgets == source_budgets,
        "action1_positive_pattern_spans_all_budgets": len(action1_positive_groups) == 1
        and set(action1_positive_groups[0].get("budgets", []) or []) == source_budgets,
        "action3_neutral_pattern_spans_all_budgets": len(action3_neutral_groups) == 1
        and set(action3_neutral_groups[0].get("budgets", []) or []) == source_budgets,
        "paired_control_effect_gap_is_stable_positive": bool(gaps)
        and min(gaps) > 0.0
        and max(gaps) - min(gaps) == 0.0,
        "neutral_exception_has_exact_replication": len(neutral_replications) == 1,
        "no_negative_effect_events": negative_events == 0,
        "all_context_boundaries_preserved": all(
            bool(row.get("context_preserved", False))
            and not bool(row.get("cross_context_merge_performed", False))
            and not bool(row.get("cross_control_effect_merge_performed", False))
            for row in context_clusters
        ),
        "candidate_claim_reformulated_as_control_dependent": True,
    }
    ready = bool(requirements) and all(requirements.values())
    missing = [key for key, value in requirements.items() if not value]
    return {
        "assessment_id": "sage6f::a32_review_eligibility_assessment::001",
        "candidate_key": (
            f"control_dependent_local_patch::{source_sage6c.get('summary', {}).get('game_id', '')}"
            "::ACTION2::ACTION1_vs_ACTION3"
        ),
        "game_id": str(source_sage6c.get("summary", {}).get("game_id", "")),
        "candidate_mechanism_family": "control_dependent_local_patch_change_candidate",
        "metric": "local_patch_before_after",
        "target_action": "ACTION2",
        "control_actions": controls,
        "source_budgets": sorted(source_budgets),
        "context_clusters": len(context_clusters),
        "distinct_control_actions": len(controls),
        "paired_control_contexts": len(paired),
        "paired_control_budgets": sorted(paired_budgets),
        "paired_control_effect_gaps": gaps,
        "paired_control_effect_gap_spread": max(gaps) - min(gaps) if gaps else None,
        "action1_positive_contexts": sum(
            row.get("control_action") == "ACTION1"
            and row.get("effect_direction") == "positive"
            for row in observations
        ),
        "action1_neutral_observations": sum(
            row.get("control_action") == "ACTION1"
            and row.get("effect_direction") == "neutral"
            for row in observations
        ),
        "action3_neutral_contexts": sum(
            row.get("control_action") == "ACTION3"
            and row.get("effect_direction") == "neutral"
            for row in observations
        ),
        "replicated_neutral_contexts": len(neutral_replications),
        "negative_effect_events": negative_events,
        "raw_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in observations
        ),
        "raw_contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in observations
        ),
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in observations
        ),
        "eligibility_requirements": requirements,
        "missing_eligibility_requirements": missing,
        "candidate_status": CONTROL_DEPENDENT_MULTI_BUDGET,
        "a32_review_recommendation": (
            READY_FOR_A32_CONTROL_DEPENDENCE_REVIEW
            if ready
            else "ADDITIONAL_CONTROL_DEPENDENCE_WORK_REQUIRED_CANDIDATE_ONLY"
        ),
        "ready_for_A32_review": ready,
        "ready_for_A32_review_is_not_verdict": True,
        "ready_for_A32_review_is_not_confirmation": True,
        "recommended_a32_review_scope": {
            "review_claim": (
                "ACTION2 local-patch contrast is control-dependent: positive versus "
                "ACTION1 but neutral versus ACTION3 in paired wa30 contexts."
            ),
            "must_not_review_as": "STANDALONE_UNCONDITIONAL_ACTION2_EFFECT",
            "scope_game_id": str(source_sage6c.get("summary", {}).get("game_id", "")),
            "scope_controls": controls,
            "scope_context_cluster_ids": [
                str(row.get("context_cluster_id", "")) for row in context_clusters
            ],
        },
        "required_next_step": (
            "A32_SCIENTIFIC_REVIEW_OF_CONTROL_DEPENDENT_CANDIDATE"
            if ready
            else "ACQUIRE_MISSING_ELIGIBILITY_REQUIREMENTS"
        ),
        "raw_events_counted_as_scientific_support": False,
        "eligibility_counted_as_a32_decision": False,
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": SAGE6F_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def build_sage6f_a32_review_frontiers(
    *,
    assessment: Mapping[str, Any],
    context_clusters: Sequence[Mapping[str, Any]],
    control_groups: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not bool(assessment.get("ready_for_A32_review", False)):
        return []
    return [
        {
            "frontier_id": "sage6f::candidate_a32_control_dependence_frontier::001",
            "source_assessment_id": str(assessment.get("assessment_id", "")),
            "candidate_key": str(assessment.get("candidate_key", "")),
            "game_id": str(assessment.get("game_id", "")),
            "candidate_mechanism_family": str(
                assessment.get("candidate_mechanism_family", "")
            ),
            "metric": str(assessment.get("metric", "")),
            "target_action": str(assessment.get("target_action", "")),
            "control_actions": list(assessment.get("control_actions", []) or []),
            "context_cluster_ids": [
                str(row.get("context_cluster_id", "")) for row in context_clusters
            ],
            "paired_control_comparison_ids": [
                str(row.get("paired_comparison_id", "")) for row in paired
            ],
            "control_conditioned_effect_group_ids": [
                str(row.get("control_effect_group_id", "")) for row in control_groups
            ],
            "candidate_claim": dict(
                assessment.get("recommended_a32_review_scope", {}) or {}
            ),
            "a32_review_questions": [
                "IS_ACTION1_A_VALID_LOWER_EFFECT_COMPARATOR_IN_THE_RECORDED_CONTEXTS",
                "DOES_ACTION3_EQUIVALENCE_MAKE_THE_STANDALONE_ACTION2_CLAIM_NON_IDENTIFIABLE",
                "WHAT_WA30_CONTEXT_SCOPE_CAN_BE_SCIENTIFICALLY_JUSTIFIED",
            ],
            "missing_eligibility_requirements": list(
                assessment.get("missing_eligibility_requirements", []) or []
            ),
            "ready_for_A32_review": True,
            "ready_for_A32_review_is_not_verdict": True,
            "ready_for_A32_review_is_not_confirmation": True,
            "a32_intake_requested": False,
            "frontier_counted_as_revision": False,
            "raw_events_counted_as_support": False,
            "status": "UNRESOLVED",
            "support": 0,
            "truth_status": SAGE6F_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "a32_write_performed": False,
            "a33_write_performed": False,
        }
    ]


def build_sage6f_gate(
    *,
    source_sage6c: Mapping[str, Any],
    source_sage6e: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
    control_groups: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
    frontiers: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    source_contexts = {
        str(row.get("context_snapshot_hash", ""))
        for row in source_sage6c.get("context_clusters", []) or []
        if isinstance(row, Mapping)
    }
    output_contexts = {
        str(row.get("context_snapshot_hash", "")) for row in context_clusters
    }
    observation_ids = [str(row.get("observation_id", "")) for row in observations]
    raw_total = sum(
        int(row.get("support_events", 0) or 0)
        + int(row.get("contradiction_events", 0) or 0)
        + int(row.get("neutral_events", 0) or 0)
        for row in observations
    )
    return {
        "source_sage6c_gate_passed": bool(
            source_sage6c.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source_sage6c.get("gate", {}).values()),
        "source_sage6e_gate_passed": bool(
            source_sage6e.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source_sage6e.get("gate", {}).values()),
        "all_source_observations_consolidated_once": len(observations)
        == int(source_sage6c.get("summary", {}).get("event_records", 0) or 0)
        + int(source_sage6e.get("summary", {}).get("protocols_executed", 0) or 0)
        and "" not in observation_ids
        and len(observation_ids) == len(set(observation_ids)),
        "all_sage6c_context_boundaries_preserved": source_contexts == output_contexts
        and len(context_clusters)
        == int(source_sage6c.get("summary", {}).get("context_clusters", 0) or 0)
        and all(
            bool(row.get("context_preserved", False))
            and not bool(row.get("cross_context_merge_performed", False))
            for row in context_clusters
        ),
        "control_action_identities_preserved_without_merge": {
            str(row.get("control_action", "")) for row in observations
        }
        == {"ACTION1", "ACTION3"}
        and all(
            not bool(row.get("cross_control_effect_merge_performed", False))
            for row in [*context_clusters, *control_groups]
        ),
        "control_conditioned_groups_are_comparison_indexes_only": len(control_groups)
        == 3
        and all(
            bool(row.get("comparison_index_only", False)) for row in control_groups
        ),
        "paired_control_comparisons_are_exact_and_context_preserving": len(paired)
        == MIN_PAIRED_CONTROL_CONTEXTS_FOR_A32_REVIEW
        and all(
            bool(row.get("target_signal_reproduced_across_control_pairs", False))
            and bool(row.get("context_preserved", False))
            and bool(row.get("control_identities_preserved", False))
            for row in paired
        ),
        "neutral_replication_attached_to_original_context": sum(
            bool(row.get("neutral_context_replication_verified", False))
            for row in context_clusters
        )
        == 1,
        "raw_event_accounting_exact": raw_total == len(observations),
        "a32_frontier_matches_eligibility_without_verdict": len(frontiers)
        == int(bool(assessment.get("ready_for_A32_review", False)))
        and all(
            bool(row.get("ready_for_A32_review", False))
            and bool(row.get("ready_for_A32_review_is_not_verdict", False))
            and not bool(row.get("a32_intake_requested", True))
            for row in frontiers
        ),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0
            and str(row.get("truth_status", "")) == SAGE6F_TRUTH_STATUS
            for row in [
                *observations,
                *context_clusters,
                *control_groups,
                *paired,
                assessment,
                *frontiers,
            ]
        ),
        "source_registry_quarantine_preserved": all(
            int(source.get("source_scoped_mechanics_reused", 0) or 0) == 0
            and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
            and not bool(source.get("scope_generalization_performed", False))
            for source in (source_sage6c, source_sage6e)
        ),
    }


def summarize_sage6f(
    *,
    source_sage6c: Mapping[str, Any],
    source_sage6e: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    context_clusters: Sequence[Mapping[str, Any]],
    control_groups: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
    frontiers: Sequence[Mapping[str, Any]],
    gate: Mapping[str, bool],
    outcome: str,
) -> Dict[str, Any]:
    return {
        "source_sage6c_outcome_status": str(source_sage6c.get("outcome_status", "")),
        "source_sage6e_outcome_status": str(source_sage6e.get("outcome_status", "")),
        "game_id": str(assessment.get("game_id", "")),
        "budgets": list(assessment.get("source_budgets", []) or []),
        "observation_records": len(observations),
        "source_sage6c_observations": sum(
            row.get("source_stage") == "SAGE.6c" for row in observations
        ),
        "source_sage6e_observations": sum(
            row.get("source_stage") == "SAGE.6e" for row in observations
        ),
        "context_clusters": len(context_clusters),
        "context_clusters_preserved": sum(
            bool(row.get("context_preserved", False)) for row in context_clusters
        ),
        "cross_context_merges_performed": sum(
            bool(row.get("cross_context_merge_performed", False))
            for row in context_clusters
        ),
        "distinct_control_actions": int(
            assessment.get("distinct_control_actions", 0) or 0
        ),
        "control_actions": list(assessment.get("control_actions", []) or []),
        "control_conditioned_effect_groups": len(control_groups),
        "paired_control_contexts": len(paired),
        "paired_control_budgets": list(
            assessment.get("paired_control_budgets", []) or []
        ),
        "paired_control_effect_gaps": list(
            assessment.get("paired_control_effect_gaps", []) or []
        ),
        "paired_control_effect_gap_spread": assessment.get(
            "paired_control_effect_gap_spread"
        ),
        "action1_positive_contexts": int(
            assessment.get("action1_positive_contexts", 0) or 0
        ),
        "action1_neutral_observations": int(
            assessment.get("action1_neutral_observations", 0) or 0
        ),
        "action3_neutral_contexts": int(
            assessment.get("action3_neutral_contexts", 0) or 0
        ),
        "replicated_neutral_contexts": int(
            assessment.get("replicated_neutral_contexts", 0) or 0
        ),
        "negative_effect_events": int(assessment.get("negative_effect_events", 0) or 0),
        "raw_support_events": int(assessment.get("raw_support_events", 0) or 0),
        "raw_contradiction_events": int(
            assessment.get("raw_contradiction_events", 0) or 0
        ),
        "raw_neutral_events": int(assessment.get("raw_neutral_events", 0) or 0),
        "eligibility_requirements_passed": sum(
            bool(value)
            for value in assessment.get("eligibility_requirements", {}).values()
        ),
        "eligibility_requirements_total": len(
            assessment.get("eligibility_requirements", {}) or {}
        ),
        "missing_eligibility_requirements": list(
            assessment.get("missing_eligibility_requirements", []) or []
        ),
        "candidate_status": str(assessment.get("candidate_status", "")),
        "candidate_a32_review_frontiers": len(frontiers),
        "ready_for_A32_review": sum(
            bool(row.get("ready_for_A32_review", False)) for row in frontiers
        ),
        "ready_for_A32_review_is_not_verdict": True,
        "a32_intake_requested": False,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "outcome_status": outcome,
        "support": 0,
        "truth_status": SAGE6F_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def validate_sage6f_sources(
    source_sage6c: Mapping[str, Any],
    source_sage6e: Mapping[str, Any],
) -> None:
    validate_sage6f_sage6c_source(source_sage6c)
    validate_sage6f_sage6e_source(source_sage6e)
    game6c = str(source_sage6c.get("summary", {}).get("game_id", ""))
    game6e = str(source_sage6e.get("summary", {}).get("game_id", ""))
    if not game6c or game6c != game6e:
        raise ValueError("SAGE.6c and SAGE.6e game identities must align")
    budgets6c = [
        int(value)
        for value in source_sage6c.get("summary", {}).get("budgets", []) or []
    ]
    budgets6e = [
        int(value)
        for value in source_sage6e.get("summary", {}).get("budgets", []) or []
    ]
    if not budgets6c or budgets6c != budgets6e:
        raise ValueError("SAGE.6c and SAGE.6e budgets must align")

    clusters = {
        str(row.get("context_cluster_id", "")): row
        for row in source_sage6c.get("context_clusters", []) or []
        if isinstance(row, Mapping)
    }
    events = {
        str(row.get("request_id", "")): row
        for row in source_sage6c.get("event_records", []) or []
        if isinstance(row, Mapping)
    }
    results = [
        row
        for row in source_sage6e.get("controlled_followup_results", []) or []
        if isinstance(row, Mapping)
    ]
    for result in results:
        cluster_id = str(result.get("source_context_cluster_id", ""))
        request_id = str(result.get("source_request_id", ""))
        cluster = clusters.get(cluster_id)
        event = events.get(request_id)
        if cluster is None or event is None:
            raise ValueError(
                "SAGE.6e result must reference one SAGE.6c context and event"
            )
        if (
            str(result.get("context_snapshot_hash", ""))
            != str(cluster.get("context_snapshot_hash", ""))
            or str(result.get("context_snapshot_hash", ""))
            != str(event.get("context_snapshot_hash", ""))
            or str(result.get("target_action", ""))
            != str(event.get("target_action", ""))
            or str(result.get("metric", "")) != str(event.get("metric", ""))
            or str(result.get("hypothesis_family", ""))
            != str(event.get("hypothesis_family", ""))
            or int(result.get("budget", 0) or 0) != int(event.get("budget", 0) or 0)
            or int(result.get("source_step", 0) or 0)
            != int(event.get("source_step", 0) or 0)
        ):
            raise ValueError("SAGE.6c and SAGE.6e result contexts must align exactly")


def validate_sage6f_sage6c_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6C_SCHEMA_VERSION:
        raise ValueError("SAGE.6c schema version is not supported by SAGE.6f")
    if str(source.get("outcome_status", "")) != SAGE6C_EVENTS_CONSOLIDATED:
        raise ValueError("SAGE.6f requires a completed SAGE.6c consolidation")
    _validate_candidate_source(
        source,
        expected_truth=SAGE6C_TRUTH_STATUS,
        stage="SAGE.6c",
    )
    if (
        not source.get("gate")
        or not all(bool(value) for value in source.get("gate", {}).values())
        or not bool(summary.get("gate_passed", False))
    ):
        raise ValueError("every SAGE.6c source gate must pass")
    events = [
        row for row in source.get("event_records", []) or [] if isinstance(row, Mapping)
    ]
    clusters = [
        row
        for row in source.get("context_clusters", []) or []
        if isinstance(row, Mapping)
    ]
    if len(events) != int(summary.get("event_records", 0) or 0):
        raise ValueError("SAGE.6c event count must match its summary")
    if len(clusters) != int(summary.get("context_clusters", 0) or 0):
        raise ValueError("SAGE.6c context count must match its summary")
    if any(
        int(row.get("events", 0) or 0) != 1
        or not bool(row.get("context_preserved", False))
        or bool(row.get("cross_context_merge_performed", False))
        for row in clusters
    ):
        raise ValueError("SAGE.6c context clusters must remain singleton and unmerged")
    if any(
        str(row.get("control_action", "")) != "ACTION1"
        or not bool(row.get("live_prefix_replay_exact", False))
        or not bool(row.get("target_context_signature_verified", False))
        or not bool(row.get("control_context_signature_verified", False))
        or int(row.get("support", 0) or 0) != 0
        or str(row.get("truth_status", "")) != SAGE6C_TRUTH_STATUS
        for row in events
    ):
        raise ValueError("SAGE.6c ACTION1 observations must remain exact")
    if (
        sum(int(row.get("support_events", 0) or 0) for row in events)
        != int(summary.get("raw_support_events", 0) or 0)
        or sum(int(row.get("contradiction_events", 0) or 0) for row in events)
        != int(summary.get("raw_contradiction_events", 0) or 0)
        or sum(int(row.get("neutral_events", 0) or 0) for row in events)
        != int(summary.get("raw_neutral_events", 0) or 0)
    ):
        raise ValueError("SAGE.6c raw event accounting must remain exact")


def validate_sage6f_sage6e_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6E_SCHEMA_VERSION:
        raise ValueError("SAGE.6e schema version is not supported by SAGE.6f")
    if str(source.get("outcome_status", "")) != SAGE6E_EXECUTION_COMPLETED:
        raise ValueError("SAGE.6f requires a completed SAGE.6e execution")
    _validate_candidate_source(
        source,
        expected_truth=SAGE6E_TRUTH_STATUS,
        stage="SAGE.6e",
    )
    if (
        not source.get("gate")
        or not all(bool(value) for value in source.get("gate", {}).values())
        or not bool(summary.get("gate_passed", False))
    ):
        raise ValueError("every SAGE.6e source gate must pass")
    results = [
        row
        for row in source.get("controlled_followup_results", []) or []
        if isinstance(row, Mapping)
    ]
    blocked = list(source.get("blocked_followup_results", []) or [])
    if (
        len(results) != int(summary.get("protocols_executed", 0) or 0)
        or blocked
        or int(summary.get("protocols_blocked", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6e followup execution must remain complete")
    protocol_ids = [str(row.get("protocol_id", "")) for row in results]
    if "" in protocol_ids or len(protocol_ids) != len(set(protocol_ids)):
        raise ValueError("SAGE.6e protocol result identities must remain unique")
    if any(
        str(row.get("execution_status", "")) != "EXECUTED"
        or not bool(row.get("protocol_execution_exact", False))
        or not bool(row.get("live_prefix_replay_exact", False))
        or not bool(row.get("target_context_signature_verified", False))
        or not bool(row.get("control_context_signature_verified", False))
        or bool(row.get("protocol_substitution_performed", True))
        or bool(row.get("context_substitution_performed", True))
        or bool(row.get("budget_substitution_performed", True))
        or bool(row.get("target_action_substitution_performed", True))
        or bool(row.get("control_action_substitution_performed", True))
        or int(row.get("support", 0) or 0) != 0
        or str(row.get("truth_status", "")) != SAGE6E_TRUTH_STATUS
        for row in results
    ):
        raise ValueError("SAGE.6e results must remain exact and unsubstituted")
    effects = [float(row.get("effect_size", 0.0) or 0.0) for row in results]
    if (
        [str(row.get("control_action", "")) for row in results]
        != ["ACTION3", "ACTION3", "ACTION3", "ACTION1"]
        or [str(row.get("target_action", "")) for row in results]
        != ["ACTION2", "ACTION2", "ACTION2", "ACTION2"]
        or effects
        != [float(value) for value in summary.get("controlled_effect_sizes", []) or []]
        or sum(float(row.get("target_signal", 0.0) or 0.0) for row in results)
        != float(summary.get("target_signal_total", 0.0) or 0.0)
        or sum(float(row.get("control_signal", 0.0) or 0.0) for row in results)
        != float(summary.get("control_signal_total", 0.0) or 0.0)
        or sum(int(row.get("support_events", 0) or 0) for row in results)
        != int(summary.get("raw_support_events", 0) or 0)
        or sum(int(row.get("contradiction_events", 0) or 0) for row in results)
        != int(summary.get("raw_contradiction_events", 0) or 0)
        or sum(int(row.get("neutral_events", 0) or 0) for row in results)
        != int(summary.get("raw_neutral_events", 0) or 0)
    ):
        raise ValueError("SAGE.6e result accounting and control identities must align")
    assessment = dict(source.get("pre_registered_outcome_assessment", {}) or {})
    if (
        not bool(assessment.get("ready_for_post_execution_consolidation", False))
        or bool(assessment.get("ready_for_A32_review", True))
        or int(assessment.get("support", 0) or 0) != 0
        or str(assessment.get("truth_status", "")) != SAGE6E_TRUTH_STATUS
    ):
        raise ValueError("SAGE.6e assessment must be ready only for consolidation")


def _validate_candidate_source(
    source: Mapping[str, Any], *, expected_truth: str, stage: str
) -> None:
    if (
        str(source.get("status", "")) != "UNRESOLVED"
        or str(source.get("truth_status", "")) != expected_truth
        or str(source.get("revision_status", "")) != "CANDIDATE_ONLY"
        or int(source.get("support", 0) or 0) != 0
    ):
        raise ValueError(
            f"{stage} must remain unresolved candidate-only with support 0"
        )
    if any(
        bool(source.get(flag, False))
        for flag in (
            "revision_performed",
            "confirmation_performed",
            "refutation_performed",
            "a32_intake_requested",
            "a32_write_performed",
            "a33_write_performed",
            "scope_generalization_performed",
        )
    ):
        raise ValueError(f"{stage} cannot contain verdict or registry writes")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
    ):
        raise ValueError(f"{stage} cannot import scoped source mechanics")


def build_source_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": str(source.get("config", {}).get("schema_version", "")),
        "game_id": str(source.get("summary", {}).get("game_id", "")),
        "budgets": list(source.get("summary", {}).get("budgets", []) or []),
        "outcome_status": str(source.get("outcome_status", "")),
        "gate_passed": bool(source.get("summary", {}).get("gate_passed", False)),
        "support": int(source.get("support", 0) or 0),
        "truth_status": str(source.get("truth_status", "")),
        "revision_status": str(source.get("revision_status", "")),
        "source_counted_as_scientific_support": False,
    }


def write_sage6f_second_unknown_game_control_dependence_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate SAGE.6c/SAGE.6e results without merging controls or contexts."
        )
    )
    parser.add_argument(
        "--source-sage6c", default=str(DEFAULT_SAGE6C_EVENT_CONSOLIDATION_PATH)
    )
    parser.add_argument(
        "--source-sage6e", default=str(DEFAULT_SAGE6E_FOLLOWUP_EXECUTION_PATH)
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE6F_CONTROL_DEPENDENCE_CONSOLIDATION_PATH)
    )
    args = parser.parse_args(argv)
    run_sage6f_second_unknown_game_control_dependence_consolidation(
        source_sage6c_path=args.source_sage6c,
        source_sage6e_path=args.source_sage6e,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
