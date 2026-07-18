"""SAGE.5g candidate-only compiler for unknown-game A32 review handoffs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .mini_frontier_event_consolidation import (
    DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH,
)


DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH = (
    Path("diagnostics") / "sage" / "sage5g_a32_review_handoff.json"
)
SAGE5G_SCHEMA_VERSION = "sage.a32_review_handoff.v1"
SAGE5G_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5G"
SAGE5G_HANDOFF_COMPILED = "SAGE_A32_REVIEW_HANDOFF_COMPILED_CANDIDATE_ONLY"
SAGE5G_NO_HANDOFF_ITEMS = "SAGE_A32_REVIEW_HANDOFF_NO_ITEMS_CANDIDATE_ONLY"

FOLLOWUP_REQUIRED_CONTROL_DIVERSITY = "FOLLOWUP_REQUIRED_CONTROL_DIVERSITY"
FOLLOWUP_REQUIRED_SUPPORT = "FOLLOWUP_REQUIRED_SUPPORT"
FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY = (
    "FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY"
)
FOLLOWUP_REQUIRED_CONTEXT_DIVERSITY = "FOLLOWUP_REQUIRED_CONTEXT_DIVERSITY"
FOLLOWUP_REQUIRED_CONTRADICTION_REVIEW = "FOLLOWUP_REQUIRED_CONTRADICTION_REVIEW"
READY_FOR_A32_INTAKE_CANDIDATE_ONLY = "READY_FOR_A32_INTAKE_CANDIDATE_ONLY"

MIN_SUPPORT_EVENTS = 3
MIN_INDEPENDENT_CONTEXT_EVENTS = 2
MIN_DISTINCT_CONTROL_ACTIONS = 2


@dataclass(frozen=True)
class A32ReviewCandidateItem:
    """One auditable, non-verdict candidate prepared for later A32 intake."""

    candidate_id: str
    candidate_key: str
    game_id: str
    hypothesis_family: str
    action: str
    action_args: Dict[str, Any] | None
    predicted_metric: str
    predicted_effect_signature: Dict[str, Any]
    source_cluster_ids: Tuple[str, ...]
    related_nonmerged_cluster_ids: Tuple[str, ...]
    contexts: Tuple[Dict[str, Any], ...]
    budgets: Tuple[int, ...]
    target_interventions: Tuple[Dict[str, Any], ...]
    control_interventions: Tuple[Dict[str, Any], ...]
    raw_support_events: int
    independent_context_events: int
    distinct_control_actions: int
    contradiction_events: int
    a32_intake_recommendation: str
    missing_revision_requirements: Tuple[str, ...]
    requested_followups: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_key": self.candidate_key,
            "game_id": self.game_id,
            "hypothesis_family": self.hypothesis_family,
            "action": self.action,
            "action_args": (
                dict(self.action_args) if self.action_args is not None else None
            ),
            "predicted_metric": self.predicted_metric,
            "predicted_effect_signature": dict(self.predicted_effect_signature),
            "source_cluster_ids": list(self.source_cluster_ids),
            "related_nonmerged_cluster_ids": list(
                self.related_nonmerged_cluster_ids
            ),
            "contexts": [dict(row) for row in self.contexts],
            "budgets": list(self.budgets),
            "target_interventions": [dict(row) for row in self.target_interventions],
            "control_interventions": [
                dict(row) for row in self.control_interventions
            ],
            "raw_support_events": int(self.raw_support_events),
            "independent_context_events": int(self.independent_context_events),
            "distinct_control_actions": int(self.distinct_control_actions),
            "contradiction_events": int(self.contradiction_events),
            "a32_intake_recommendation": self.a32_intake_recommendation,
            "missing_revision_requirements": list(
                self.missing_revision_requirements
            ),
            "requested_followups": [dict(row) for row in self.requested_followups],
            "ready_for_A32_review": True,
            "ready_for_A32_review_is_not_verdict": True,
            "independent_context_events_counted_as_scientific_support": False,
            "candidate_review_item_counted_as_revision": False,
            "execution_performed": False,
            "revision_performed": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "status": "UNRESOLVED",
            "revision_status": "CANDIDATE_ONLY",
            "support": 0,
            "truth_status": SAGE5G_TRUTH_STATUS,
            "wrong_confirmations": 0,
        }


def run_sage5g_a32_review_handoff(
    *,
    source_sage5f_path: str | Path = (
        DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH
    ),
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Compile SAGE.5f robust clusters into candidate-only A32 review items."""
    source = _load_json(source_sage5f_path)
    validate_sage5f_source(source)
    items = build_a32_review_candidate_items(source)
    followups = tuple(
        dict(row)
        for item in items
        for row in item.requested_followups
    )
    summary = summarize_sage5g(source=source, items=items, followups=followups)
    payload = {
        "config": {
            "schema_version": SAGE5G_SCHEMA_VERSION,
            "source_sage5f_path": str(source_sage5f_path),
            "inputs_read": ["SAGE.5f"],
            "execution_performed": False,
            "compilation_policy": {
                "candidate_review_item_is_not_revision": True,
                "independent_context_events_are_not_scientific_support": True,
                "distinct_budgets_are_not_distinct_controls": True,
                "related_clusters_are_linked_not_merged": True,
                "a32_write_performed": False,
                "a33_write_performed": False,
            },
            "a32_revision_thresholds": {
                "minimum_support_events": MIN_SUPPORT_EVENTS,
                "minimum_independent_context_events": (
                    MIN_INDEPENDENT_CONTEXT_EVENTS
                ),
                "minimum_distinct_control_actions": MIN_DISTINCT_CONTROL_ACTIONS,
                "maximum_contradiction_events": 0,
            },
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5f_summary": dict(source.get("summary", {}) or {}),
        "a32_review_candidate_items": [item.to_dict() for item in items],
        "requested_followups": list(followups),
        "summary": summary,
        "comparison": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5G_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "candidate_review_item_counted_as_revision": False,
        "independent_context_events_counted_as_scientific_support": False,
        "candidate_review_item_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5g_a32_review_handoff(payload, output_path)
    return payload


def build_a32_review_candidate_items(
    source: Mapping[str, Any],
) -> Tuple[A32ReviewCandidateItem, ...]:
    clusters = [
        dict(row)
        for row in source.get("candidate_mechanism_clusters", []) or []
        if isinstance(row, Mapping)
    ]
    clusters_by_id = {
        str(row.get("cluster_id", "")): row
        for row in clusters
        if str(row.get("cluster_id", ""))
    }
    events_by_request = {
        str(row.get("request_id", "")): dict(row)
        for row in source.get("event_records", []) or []
        if isinstance(row, Mapping) and str(row.get("request_id", ""))
    }
    items: List[A32ReviewCandidateItem] = []
    ready_frontiers = [
        row
        for row in source.get("candidate_a32_review_frontiers", []) or []
        if isinstance(row, Mapping) and bool(row.get("ready_for_A32_review", False))
    ]
    for index, frontier in enumerate(ready_frontiers, start=1):
        cluster_id = str(frontier.get("source_cluster_id", ""))
        if cluster_id not in clusters_by_id:
            raise ValueError(f"SAGE.5f frontier references unknown cluster: {cluster_id}")
        cluster = clusters_by_id[cluster_id]
        events = tuple(
            events_by_request[request_id]
            for request_id in cluster.get("request_ids", []) or []
            if str(request_id) in events_by_request
        )
        if len(events) != len(cluster.get("request_ids", []) or []):
            raise ValueError(f"SAGE.5f cluster has missing event records: {cluster_id}")
        related = related_nonmerged_clusters(cluster, clusters)
        items.append(
            candidate_item_from_cluster(
                index=index,
                cluster=cluster,
                events=events,
                related_clusters=related,
            )
        )
    return tuple(items)


def candidate_item_from_cluster(
    *,
    index: int,
    cluster: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    related_clusters: Sequence[Mapping[str, Any]],
) -> A32ReviewCandidateItem:
    candidate_id = f"sage5g::a32_review_candidate::{index:03d}"
    action = _single_value(cluster.get("actions", []) or [])
    action_args = _action_args(cluster, events)
    metrics = sorted(
        {
            str(row.get("metric", ""))
            for row in events
            if str(row.get("metric", ""))
        }
    )
    predicted_metric = metrics[0] if len(metrics) == 1 else "multi_metric"
    contexts = _context_records(events)
    target_interventions = _target_interventions(events)
    control_interventions = _control_interventions(events)
    context_hashes = _independent_context_hashes(events)
    control_actions = {
        str(row.get("control_action", ""))
        for row in events
        if str(row.get("control_action", ""))
    }
    raw_support_events = int(cluster.get("raw_support_events", 0) or 0)
    contradiction_events = int(cluster.get("raw_contradiction_events", 0) or 0)
    missing = missing_revision_requirements(
        raw_support_events=raw_support_events,
        independent_context_events=len(context_hashes),
        distinct_control_actions=len(control_actions),
        contradiction_events=contradiction_events,
    )
    recommendation = a32_intake_recommendation(missing)
    followups = requested_followups_for_candidate(
        candidate_id=candidate_id,
        action=action,
        action_args=action_args,
        raw_support_events=raw_support_events,
        independent_context_events=len(context_hashes),
        control_actions=sorted(control_actions),
        contradiction_events=contradiction_events,
        source_cluster_id=str(cluster.get("cluster_id", "")),
        related_cluster_ids=[
            str(row.get("cluster_id", "")) for row in related_clusters
        ],
        missing=missing,
    )
    candidate_key = "::".join(
        (
            "mechanic_prediction",
            str(cluster.get("game_id", "")),
            action,
            str(cluster.get("hypothesis_family", "")),
            predicted_metric,
            f"args={_canonical_json(action_args)}",
        )
    )
    return A32ReviewCandidateItem(
        candidate_id=candidate_id,
        candidate_key=candidate_key,
        game_id=str(cluster.get("game_id", "")),
        hypothesis_family=str(cluster.get("hypothesis_family", "")),
        action=action,
        action_args=action_args,
        predicted_metric=predicted_metric,
        predicted_effect_signature=predicted_effect_signature(cluster, events),
        source_cluster_ids=(str(cluster.get("cluster_id", "")),),
        related_nonmerged_cluster_ids=tuple(
            str(row.get("cluster_id", "")) for row in related_clusters
        ),
        contexts=contexts,
        budgets=tuple(sorted({int(row.get("budget", 0) or 0) for row in events})),
        target_interventions=target_interventions,
        control_interventions=control_interventions,
        raw_support_events=raw_support_events,
        independent_context_events=len(context_hashes),
        distinct_control_actions=len(control_actions),
        contradiction_events=contradiction_events,
        a32_intake_recommendation=recommendation,
        missing_revision_requirements=missing,
        requested_followups=followups,
    )


def related_nonmerged_clusters(
    cluster: Mapping[str, Any],
    all_clusters: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    identity = _related_effect_identity(cluster)
    cluster_id = str(cluster.get("cluster_id", ""))
    related = [
        dict(row)
        for row in all_clusters
        if str(row.get("cluster_id", "")) != cluster_id
        and str(row.get("game_id", "")) == str(cluster.get("game_id", ""))
        and _related_effect_identity(row) == identity
    ]
    return tuple(sorted(related, key=lambda row: str(row.get("cluster_id", ""))))


def predicted_effect_signature(
    cluster: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    observed = dict(cluster.get("observed_effect_pattern", {}) or {})
    first = dict(events[0]) if events else {}
    return {
        "changed_cells": _singleton_or_values(observed.get("changed_cells_values", [])),
        "color_transitions": dict(first.get("color_transitions", {}) or {}),
        "component_delta_by_color": dict(
            first.get("component_delta_by_color", {}) or {}
        ),
        "bbox_shapes": sorted(
            str(key) for key in (observed.get("bbox_shape_counts", {}) or {})
        ),
        "same_bbox_shape": bool(observed.get("same_bbox_shape", False)),
        "terminal_after": _singleton_or_values(
            observed.get("terminal_after_values", [])
        ),
        "levels_delta": _singleton_or_values(observed.get("levels_delta_values", [])),
        "effect_size": _singleton_or_values(observed.get("effect_size_values", [])),
        "mean_effect_size": float(observed.get("mean_effect_size", 0.0) or 0.0),
        "replay_exact": bool(cluster.get("all_replays_exact", False)),
    }


def missing_revision_requirements(
    *,
    raw_support_events: int,
    independent_context_events: int,
    distinct_control_actions: int,
    contradiction_events: int,
) -> Tuple[str, ...]:
    missing: List[str] = []
    if raw_support_events < MIN_SUPPORT_EVENTS:
        missing.append("minimum_support_events")
    if independent_context_events < MIN_INDEPENDENT_CONTEXT_EVENTS:
        missing.append("minimum_independent_context_events")
    if distinct_control_actions < MIN_DISTINCT_CONTROL_ACTIONS:
        missing.append("minimum_distinct_control_actions")
    if contradiction_events > 0:
        missing.append("zero_contradiction_events")
    return tuple(missing)


def a32_intake_recommendation(missing: Sequence[str]) -> str:
    missing_set = set(missing)
    if "zero_contradiction_events" in missing_set:
        return FOLLOWUP_REQUIRED_CONTRADICTION_REVIEW
    support_missing = "minimum_support_events" in missing_set
    controls_missing = "minimum_distinct_control_actions" in missing_set
    if support_missing and controls_missing:
        return FOLLOWUP_REQUIRED_SUPPORT_AND_CONTROL_DIVERSITY
    if controls_missing:
        return FOLLOWUP_REQUIRED_CONTROL_DIVERSITY
    if support_missing:
        return FOLLOWUP_REQUIRED_SUPPORT
    if "minimum_independent_context_events" in missing_set:
        return FOLLOWUP_REQUIRED_CONTEXT_DIVERSITY
    return READY_FOR_A32_INTAKE_CANDIDATE_ONLY


def requested_followups_for_candidate(
    *,
    candidate_id: str,
    action: str,
    action_args: Mapping[str, Any] | None,
    raw_support_events: int,
    independent_context_events: int,
    control_actions: Sequence[str],
    contradiction_events: int,
    source_cluster_id: str,
    related_cluster_ids: Sequence[str],
    missing: Sequence[str],
) -> Tuple[Dict[str, Any], ...]:
    requests: List[Dict[str, Any]] = []
    missing_set = set(missing)
    if "minimum_support_events" in missing_set:
        requests.append(
            _followup(
                candidate_id,
                "support",
                "ACQUIRE_ADDITIONAL_COMPARABLE_SUPPORT",
                action=action,
                action_args=action_args,
                additional_events_required=max(0, MIN_SUPPORT_EVENTS - raw_support_events),
                minimum_replay_exact_contexts=1,
            )
        )
    if "minimum_independent_context_events" in missing_set:
        requests.append(
            _followup(
                candidate_id,
                "context_diversity",
                "ACQUIRE_INDEPENDENT_REPLAY_EXACT_CONTEXT",
                action=action,
                action_args=action_args,
                additional_contexts_required=max(
                    0,
                    MIN_INDEPENDENT_CONTEXT_EVENTS - independent_context_events,
                ),
            )
        )
    if "minimum_distinct_control_actions" in missing_set:
        requests.append(
            _followup(
                candidate_id,
                "control_diversity",
                "ACQUIRE_DISTINCT_CONTROL_ACTION",
                action=action,
                action_args=action_args,
                excluded_control_actions=list(control_actions),
                additional_control_actions_required=max(
                    0,
                    MIN_DISTINCT_CONTROL_ACTIONS - len(set(control_actions)),
                ),
                minimum_replay_exact_contexts=(
                    2 if raw_support_events >= MIN_SUPPORT_EVENTS else 1
                ),
            )
        )
    if contradiction_events > 0:
        requests.append(
            _followup(
                candidate_id,
                "contradiction_review",
                "REVIEW_CONTRADICTION_EVENTS",
                action=action,
                action_args=action_args,
                contradiction_events=contradiction_events,
            )
        )
    if related_cluster_ids:
        requests.append(
            _followup(
                candidate_id,
                "cross_measurement",
                "CROSS_MEASURE_RELATED_NONMERGED_CLUSTER",
                action=action,
                action_args=action_args,
                source_cluster_id=source_cluster_id,
                related_cluster_ids=list(related_cluster_ids),
                required_measurements=["local_patch", "object_delta"],
                clusters_must_remain_unmerged=True,
            )
        )
    return tuple(requests)


def summarize_sage5g(
    *,
    source: Mapping[str, Any],
    items: Sequence[A32ReviewCandidateItem],
    followups: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_summary = dict(source.get("summary", {}) or {})
    recommendation_counts = Counter(item.a32_intake_recommendation for item in items)
    expected_items = int(source_summary.get("ready_for_A32_review_candidates", 0) or 0)
    gate_passed = bool(items) and len(items) == expected_items
    return {
        "source_sage5f_outcome_status": str(source.get("outcome_status", "")),
        "source_candidate_mechanism_clusters": int(
            source_summary.get("candidate_mechanism_clusters", 0) or 0
        ),
        "source_ready_for_A32_review_candidates": expected_items,
        "handoff_items": len(items),
        "items_ready_for_A32_review": len(items),
        "items_without_followup_requirements": sum(
            1 for item in items if not item.requested_followups
        ),
        "followup_requests": len(followups),
        "raw_support_events_in_handoff": sum(
            item.raw_support_events for item in items
        ),
        "raw_contradiction_events_in_handoff": sum(
            item.contradiction_events for item in items
        ),
        "independent_context_events_in_handoff": sum(
            item.independent_context_events for item in items
        ),
        "related_nonmerged_cluster_links": sum(
            len(item.related_nonmerged_cluster_ids) for item in items
        ),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "candidate_review_item_counted_as_revision": False,
        "independent_context_events_counted_as_scientific_support": False,
        "execution_performed": False,
        "gate_passed": gate_passed,
        "outcome_status": (
            SAGE5G_HANDOFF_COMPILED if gate_passed else SAGE5G_NO_HANDOFF_ITEMS
        ),
        "support": 0,
        "truth_status": SAGE5G_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_sage5f_source(source: Mapping[str, Any]) -> None:
    summary = dict(source.get("summary", {}) or {})
    if int(source.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("SAGE.5f support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.5f summary support must remain 0")
    if str(source.get("revision_status", "CANDIDATE_ONLY")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.5f must remain candidate-only")
    if bool(source.get("revision_performed", False)):
        raise ValueError("SAGE.5f must not perform revision")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.5f wrong_confirmations must remain 0")
    if bool(source.get("candidate_a32_frontier_counted_as_revision", False)):
        raise ValueError("SAGE.5f frontier cannot count as revision")
    if not bool(source.get("ready_for_A32_review_is_not_verdict", False)):
        raise ValueError("SAGE.5f ready-for-review flag must not be a verdict")
    if bool(source.get("cluster_status_counted_as_scientific_verdict", False)):
        raise ValueError("SAGE.5f cluster status cannot be a verdict")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.5f cannot write A32/A33")
    for cluster in source.get("candidate_mechanism_clusters", []) or []:
        if not isinstance(cluster, Mapping):
            continue
        if int(cluster.get("support", 0) or 0) != 0:
            raise ValueError("SAGE.5f cluster support must remain 0")
        if bool(cluster.get("cluster_status_counted_as_scientific_verdict", False)):
            raise ValueError("SAGE.5f cluster cannot count as a verdict")


def write_sage5g_a32_review_handoff(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _context_records(
    events: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        {
            "request_id": str(row.get("request_id", "")),
            "source_transition_id": str(row.get("source_transition_id", "")),
            "budget": int(row.get("budget", 0) or 0),
            "source_step": int(row.get("source_step", 0) or 0),
            "context_snapshot_hash": str(row.get("context_snapshot_hash", "")),
            "live_prefix_replay_exact": bool(
                row.get("live_prefix_replay_exact", False)
            ),
            "target_context_signature_verified": bool(
                row.get("target_context_signature_verified", False)
            ),
            "control_context_signature_verified": bool(
                row.get("control_context_signature_verified", False)
            ),
        }
        for row in sorted(events, key=_event_sort_key)
    )


def _target_interventions(
    events: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        {
            "request_id": str(row.get("request_id", "")),
            "budget": int(row.get("budget", 0) or 0),
            "context_snapshot_hash": str(row.get("context_snapshot_hash", "")),
            "action": str(row.get("target_action", "")),
            "action_args": (
                dict(row.get("target_action_args", {}) or {})
                if row.get("target_action_args") is not None
                else None
            ),
            "metric": str(row.get("metric", "")),
            "signal": float(row.get("target_signal", 0.0) or 0.0),
            "effect_size_vs_control": float(row.get("effect_size", 0.0) or 0.0),
        }
        for row in sorted(events, key=_event_sort_key)
    )


def _control_interventions(
    events: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        {
            "request_id": str(row.get("request_id", "")),
            "budget": int(row.get("budget", 0) or 0),
            "context_snapshot_hash": str(row.get("context_snapshot_hash", "")),
            "action": str(row.get("control_action", "")),
            "metric": str(row.get("metric", "")),
            "signal": float(row.get("control_signal", 0.0) or 0.0),
            "context_signature_verified": bool(
                row.get("control_context_signature_verified", False)
            ),
        }
        for row in sorted(events, key=_event_sort_key)
    )


def _related_effect_identity(cluster: Mapping[str, Any]) -> Tuple[Any, ...]:
    observed = dict(cluster.get("observed_effect_pattern", {}) or {})
    return (
        tuple(sorted(str(value) for value in cluster.get("actions", []) or [])),
        str(cluster.get("target_action_args_signature", "")),
        tuple(sorted(str(key) for key in (observed.get("color_transition_signatures", {}) or {}))),
        tuple(sorted(int(value) for value in observed.get("changed_cells_values", []) or [])),
        tuple(sorted(bool(value) for value in observed.get("terminal_after_values", []) or [])),
        tuple(sorted(int(value) for value in observed.get("levels_delta_values", []) or [])),
    )


def _independent_context_hashes(
    events: Sequence[Mapping[str, Any]],
) -> set[str]:
    return {
        str(row.get("context_snapshot_hash", ""))
        for row in events
        if str(row.get("context_snapshot_hash", ""))
        and bool(row.get("live_prefix_replay_exact", False))
        and bool(row.get("target_context_signature_verified", False))
        and bool(row.get("control_context_signature_verified", False))
    }


def _action_args(
    cluster: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    if events:
        raw = events[0].get("target_action_args")
        return dict(raw or {}) if raw is not None else None
    signature = str(cluster.get("target_action_args_signature", "null"))
    parsed = json.loads(signature)
    return dict(parsed or {}) if parsed is not None else None


def _followup(
    candidate_id: str,
    suffix: str,
    request_type: str,
    **details: Any,
) -> Dict[str, Any]:
    return {
        "followup_id": f"{candidate_id}::followup::{suffix}",
        "candidate_id": candidate_id,
        "request_type": request_type,
        **details,
        "execution_performed": False,
        "status": "REQUESTED_CANDIDATE_ONLY",
        "support": 0,
        "truth_status": SAGE5G_TRUTH_STATUS,
    }


def _singleton_or_values(values: Iterable[Any]) -> Any:
    materialized = list(values or [])
    if len(materialized) == 1:
        return materialized[0]
    return materialized


def _single_value(values: Sequence[Any]) -> str:
    materialized = [str(value) for value in values if str(value)]
    if len(materialized) != 1:
        raise ValueError(f"expected exactly one target action, got {materialized}")
    return materialized[0]


def _event_sort_key(row: Mapping[str, Any]) -> Tuple[int, int, str]:
    return (
        int(row.get("budget", 0) or 0),
        int(row.get("source_step", 0) or 0),
        str(row.get("request_id", "")),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile SAGE.5f clusters into candidate-only A32 review handoffs.",
    )
    parser.add_argument(
        "--source-sage5f",
        default=str(DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH),
    )
    parser.add_argument("--out", default=str(DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH))
    args = parser.parse_args(argv)
    run_sage5g_a32_review_handoff(
        source_sage5f_path=args.source_sage5f,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
