"""SAGE.5f candidate-only consolidation of live mini-frontier events."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .distributed_live_mini_frontier_generation import (
    DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH,
)


DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage5f_mini_frontier_event_consolidation.json"
)
SAGE5F_SCHEMA_VERSION = "sage.mini_frontier_event_consolidation.v1"
SAGE5F_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5F"
SAGE5F_CLUSTERED = "SAGE_MINI_FRONTIER_EVENTS_CLUSTERED_CANDIDATE_ONLY"
SAGE5F_NO_MULTIBUDGET_CLUSTER = (
    "SAGE_MINI_FRONTIER_EVENTS_NO_MULTIBUDGET_CLUSTER_CANDIDATE_ONLY"
)

ROBUST_MULTI_BUDGET_CANDIDATE_ONLY = "ROBUST_MULTI_BUDGET_CANDIDATE_ONLY"
LOCAL_SUPPORT_CANDIDATE_ONLY = "LOCAL_SUPPORT_CANDIDATE_ONLY"
MIXED_CANDIDATE_ONLY = "MIXED_CANDIDATE_ONLY"
CONTRADICTED_CANDIDATE_ONLY = "CONTRADICTED_CANDIDATE_ONLY"
NEUTRAL_CANDIDATE_ONLY = "NEUTRAL_CANDIDATE_ONLY"
INSUFFICIENT_EXECUTION = "INSUFFICIENT_EXECUTION"


def run_sage5f_mini_frontier_event_consolidation(
    *,
    source_sage5e_path: str | Path = (
        DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH
    ),
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    source = _load_json(source_sage5e_path)
    validate_sage5e_source(source)
    event_records = build_event_records(source)
    clusters = consolidate_candidate_mechanism_clusters(event_records)
    a32_frontiers = candidate_a32_review_frontiers(clusters)
    summary = summarize_sage5f(
        source=source,
        event_records=event_records,
        clusters=clusters,
        a32_frontiers=a32_frontiers,
    )
    payload = {
        "config": {
            "schema_version": SAGE5F_SCHEMA_VERSION,
            "source_sage5e_path": str(source_sage5e_path),
            "inputs_read": ["SAGE.5e"],
            "execution_performed": False,
            "consolidation_policy": {
                "support_events_are_not_scientific_support": True,
                "cluster_status_is_not_scientific_verdict": True,
                "ready_for_A32_review_is_not_verdict": True,
                "dedup_keys_do_not_imply_independence": True,
                "a32_write_performed": False,
                "a33_write_performed": False,
            },
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5e_summary": dict(source.get("summary", {}) or {}),
        "event_records": event_records,
        "candidate_mechanism_clusters": clusters,
        "candidate_a32_review_frontiers": a32_frontiers,
        "summary": summary,
        "comparison": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5F_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "support_events_counted_as_support": False,
        "support_events_counted_as_scientific_support": False,
        "cluster_status_counted_as_scientific_verdict": False,
        "candidate_a32_frontier_counted_as_revision": False,
        "ready_for_A32_review_is_not_verdict": True,
        "policy_result_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5f_mini_frontier_event_consolidation(payload, output_path)
    return payload


def build_event_records(source: Mapping[str, Any]) -> List[Dict[str, Any]]:
    requests = {
        str(row.get("request_id", "")): dict(row)
        for row in source.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping)
    }
    records: List[Dict[str, Any]] = []
    for index, raw in enumerate(source.get("controlled_experiments", []) or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("execution_status", "")) != "EXECUTED":
            continue
        request = requests.get(str(raw.get("request_id", "")), {})
        diff = dict(request.get("diff_signature", {}) or {})
        if not diff:
            diff = _diff_signature_from_request_or_measurement(request, raw)
        budget = _budget_from_row(raw) or _budget_from_row(request)
        effect_size = float(
            (raw.get("controlled_delta", {}) or {}).get(
                "effect_size",
                float(raw.get("target_signal", 0.0) or 0.0)
                - float(raw.get("control_signal", 0.0) or 0.0),
            )
            or 0.0
        )
        record = {
            "event_id": f"sage5f::mini_frontier_event::{index:03d}",
            "request_id": str(raw.get("request_id", "")),
            "source_hypothesis_id": str(raw.get("source_hypothesis_id", "")),
            "source_transition_id": str(raw.get("source_transition_id", "")),
            "game_id": str(raw.get("game_id", "")),
            "budget": int(budget),
            "source_step": int(
                request.get("source_step", _step_from_row(raw) or 0) or 0
            ),
            "hypothesis_family": str(
                raw.get("hypothesis_family", request.get("hypothesis_family", ""))
            ),
            "target_action": str(raw.get("target_action", request.get("target_action", ""))),
            "target_action_args": (
                dict(raw.get("target_action_args", {}) or {})
                if raw.get("target_action_args") is not None
                else None
            ),
            "control_action": str(raw.get("control_action", "")),
            "metric": str(raw.get("metric", request.get("metric", ""))),
            "context_snapshot_hash": str(
                raw.get("context_snapshot_hash", request.get("context_snapshot_hash", ""))
            ),
            "dedup_key": str(request.get("dedup_key", "")),
            "diff_signature": diff,
            "changed_cells": int(diff.get("changed_cells", 0) or 0),
            "changed_bbox": diff.get("changed_bbox"),
            "bbox_shape": bbox_shape(diff.get("changed_bbox")),
            "color_transitions": dict(diff.get("color_transitions", {}) or {}),
            "component_delta_by_color": dict(
                diff.get("component_delta_by_color", {}) or {}
            ),
            "terminal_after": bool(diff.get("terminal_after", False)),
            "levels_delta": int(diff.get("levels_delta", 0) or 0),
            "target_signal": float(raw.get("target_signal", 0.0) or 0.0),
            "control_signal": float(raw.get("control_signal", 0.0) or 0.0),
            "effect_size": effect_size,
            "absolute_effect_size": abs(effect_size),
            "support_events": int(raw.get("support_events", 0) or 0),
            "contradiction_events": int(raw.get("contradiction_events", 0) or 0),
            "neutral_events": int(raw.get("neutral_events", 0) or 0),
            "live_prefix_replay_exact": bool(raw.get("live_prefix_replay_exact", False)),
            "target_context_signature_verified": bool(
                raw.get("target_context_signature_verified", False)
            ),
            "control_context_signature_verified": bool(
                raw.get("control_context_signature_verified", False)
            ),
            "support": 0,
            "truth_status": SAGE5F_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "support_events_counted_as_support": False,
            "observation_counted_as_confirmation": False,
        }
        records.append(record)
    return records


def consolidate_candidate_mechanism_clusters(
    event_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in event_records:
        grouped[mechanism_cluster_key(event)].append(event)

    clusters: List[Dict[str, Any]] = []
    for index, (cluster_key, rows) in enumerate(sorted(grouped.items()), start=1):
        clusters.append(cluster_from_events(index, cluster_key, rows))
    return clusters


def cluster_from_events(
    index: int,
    cluster_key: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    budgets = sorted({int(row.get("budget", 0) or 0) for row in rows})
    actions = sorted({str(row.get("target_action", "")) for row in rows if row.get("target_action")})
    request_ids = [str(row.get("request_id", "")) for row in rows]
    context_hashes = sorted({str(row.get("context_snapshot_hash", "")) for row in rows})
    dedup_keys = sorted({str(row.get("dedup_key", "")) for row in rows if row.get("dedup_key")})
    support_events = sum(int(row.get("support_events", 0) or 0) for row in rows)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in rows
    )
    neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in rows)
    replay_exact = sum(1 for row in rows if bool(row.get("live_prefix_replay_exact", False)))
    candidate_status = cluster_candidate_status(
        support_events=support_events,
        contradiction_events=contradiction_events,
        neutral_events=neutral_events,
        distinct_budgets=len(budgets),
        distinct_contexts=len(context_hashes),
        replay_exact_events=replay_exact,
        executed_events=len(rows),
    )
    ready_for_a32 = candidate_status == ROBUST_MULTI_BUDGET_CANDIDATE_ONLY
    first = dict(rows[0]) if rows else {}
    observed_pattern = observed_effect_pattern(rows)
    return {
        "cluster_id": f"sage5f::candidate_mechanism_cluster::{index:03d}",
        "cluster_key": cluster_key,
        "game_id": str(first.get("game_id", "")),
        "hypothesis_family": str(first.get("hypothesis_family", "")),
        "actions": actions,
        "target_action_args_signature": _canonical_json(first.get("target_action_args")),
        "budgets": budgets,
        "contexts": context_hashes,
        "context_count": len(context_hashes),
        "request_ids": request_ids,
        "source_hypothesis_ids": [
            str(row.get("source_hypothesis_id", "")) for row in rows
        ],
        "source_transition_ids": [
            str(row.get("source_transition_id", "")) for row in rows
        ],
        "observed_effect_pattern": observed_pattern,
        "raw_support_events": support_events,
        "raw_contradiction_events": contradiction_events,
        "raw_neutral_events": neutral_events,
        "support_events": support_events,
        "contradiction_events": contradiction_events,
        "neutral_events": neutral_events,
        "executed_events": len(rows),
        "live_prefix_replay_exact_events": replay_exact,
        "all_replays_exact": replay_exact == len(rows),
        "unique_dedup_keys": len(dedup_keys),
        "dedup_keys_counted_as_independent": False,
        "candidate_status": candidate_status,
        "ready_for_A32_review": ready_for_a32,
        "ready_for_A32_review_is_not_verdict": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": SAGE5F_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "support_events_counted_as_support": False,
        "cluster_status_counted_as_scientific_verdict": False,
        "observation_counted_as_confirmation": False,
    }


def candidate_a32_review_frontiers(
    clusters: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    frontiers: List[Dict[str, Any]] = []
    for index, cluster in enumerate(
        [row for row in clusters if bool(row.get("ready_for_A32_review", False))],
        start=1,
    ):
        frontiers.append(
            {
                "frontier_id": f"sage5f::candidate_a32_frontier::{index:03d}",
                "source_cluster_id": str(cluster.get("cluster_id", "")),
                "handoff_status": "CANDIDATE_ONLY_READY_FOR_A32_REVIEW",
                "ready_for_A32_review": True,
                "ready_for_A32_review_is_not_verdict": True,
                "candidate_mechanism_family": str(cluster.get("hypothesis_family", "")),
                "actions": list(cluster.get("actions", []) or []),
                "budgets": list(cluster.get("budgets", []) or []),
                "raw_support_events": int(cluster.get("raw_support_events", 0) or 0),
                "raw_contradiction_events": int(
                    cluster.get("raw_contradiction_events", 0) or 0
                ),
                "support_events_counted_as_support": False,
                "candidate_a32_frontier_counted_as_revision": False,
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "truth_status": SAGE5F_TRUTH_STATUS,
                "revision_performed": False,
                "wrong_confirmations": 0,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return frontiers


def mechanism_cluster_key(event: Mapping[str, Any]) -> str:
    payload = {
        "family": str(event.get("hypothesis_family", "")),
        "target_action": str(event.get("target_action", "")),
        "target_action_args": event.get("target_action_args"),
        "changed_cells": int(event.get("changed_cells", 0) or 0),
        "color_transitions": dict(event.get("color_transitions", {}) or {}),
        "component_delta_by_color": dict(
            event.get("component_delta_by_color", {}) or {}
        ),
        "terminal_after": bool(event.get("terminal_after", False)),
        "levels_delta": int(event.get("levels_delta", 0) or 0),
    }
    return _canonical_json(payload)


def observed_effect_pattern(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bboxes = [row.get("changed_bbox") for row in rows if row.get("changed_bbox")]
    bbox_shapes = [str(row.get("bbox_shape", "")) for row in rows if row.get("bbox_shape")]
    target_signals = [float(row.get("target_signal", 0.0) or 0.0) for row in rows]
    control_signals = [float(row.get("control_signal", 0.0) or 0.0) for row in rows]
    effect_sizes = [float(row.get("effect_size", 0.0) or 0.0) for row in rows]
    return {
        "metrics": sorted({str(row.get("metric", "")) for row in rows}),
        "color_transition_signatures": _counts(
            _canonical_json(row.get("color_transitions", {}) or {}) for row in rows
        ),
        "component_delta_signatures": _counts(
            _canonical_json(row.get("component_delta_by_color", {}) or {})
            for row in rows
        ),
        "changed_cells_values": sorted({int(row.get("changed_cells", 0) or 0) for row in rows}),
        "bbox_shape_counts": _counts(bbox_shapes),
        "bbox_location_signatures": _counts(_canonical_json(bbox) for bbox in bboxes),
        "bbox_x_min_values": sorted(
            {
                int((bbox or {}).get("x_min", 0) or 0)
                for bbox in bboxes
                if isinstance(bbox, Mapping)
            }
        ),
        "bbox_x_max_values": sorted(
            {
                int((bbox or {}).get("x_max", 0) or 0)
                for bbox in bboxes
                if isinstance(bbox, Mapping)
            }
        ),
        "bbox_y_min_values": sorted(
            {
                int((bbox or {}).get("y_min", 0) or 0)
                for bbox in bboxes
                if isinstance(bbox, Mapping)
            }
        ),
        "bbox_y_max_values": sorted(
            {
                int((bbox or {}).get("y_max", 0) or 0)
                for bbox in bboxes
                if isinstance(bbox, Mapping)
            }
        ),
        "target_signal_values": sorted(set(target_signals)),
        "control_signal_values": sorted(set(control_signals)),
        "effect_size_values": sorted(set(effect_sizes)),
        "mean_effect_size": _mean(effect_sizes),
        "terminal_after_values": sorted({bool(row.get("terminal_after", False)) for row in rows}),
        "levels_delta_values": sorted({int(row.get("levels_delta", 0) or 0) for row in rows}),
        "same_color_transition_pattern": len(
            {
                _canonical_json(row.get("color_transitions", {}) or {})
                for row in rows
            }
        )
        <= 1,
        "same_changed_cell_count": len(
            {int(row.get("changed_cells", 0) or 0) for row in rows}
        )
        <= 1,
        "same_bbox_shape": len(set(bbox_shapes)) <= 1 if bbox_shapes else False,
        "support": 0,
        "truth_status": SAGE5F_TRUTH_STATUS,
    }


def cluster_candidate_status(
    *,
    support_events: int,
    contradiction_events: int,
    neutral_events: int,
    distinct_budgets: int,
    distinct_contexts: int,
    replay_exact_events: int,
    executed_events: int,
) -> str:
    if support_events > 0 and contradiction_events > 0:
        return MIXED_CANDIDATE_ONLY
    if contradiction_events > 0:
        return CONTRADICTED_CANDIDATE_ONLY
    if support_events > 0:
        if (
            distinct_budgets >= 2
            and distinct_contexts >= 2
            and replay_exact_events == executed_events
        ):
            return ROBUST_MULTI_BUDGET_CANDIDATE_ONLY
        return LOCAL_SUPPORT_CANDIDATE_ONLY
    if neutral_events > 0:
        return NEUTRAL_CANDIDATE_ONLY
    return INSUFFICIENT_EXECUTION


def summarize_sage5f(
    *,
    source: Mapping[str, Any],
    event_records: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    a32_frontiers: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_summary = dict(source.get("summary", {}) or {})
    status_counts = Counter(str(row.get("candidate_status", "")) for row in clusters)
    multi_budget_clusters = [
        row for row in clusters if len(row.get("budgets", []) or []) >= 2
    ]
    robust_clusters = [
        row
        for row in clusters
        if str(row.get("candidate_status", "")) == ROBUST_MULTI_BUDGET_CANDIDATE_ONLY
    ]
    raw_support_events = sum(int(row.get("support_events", 0) or 0) for row in event_records)
    raw_contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in event_records
    )
    gate_passed = bool(robust_clusters)
    return {
        "source_sage5e_outcome_status": str(source.get("outcome_status", "")),
        "source_requests_generated": int(
            source_summary.get("effective_requests_generated", 0) or 0
        ),
        "source_requests_executed": int(
            source_summary.get("requests_executed", len(event_records)) or 0
        ),
        "source_support_events": int(source_summary.get("support_events", 0) or 0),
        "source_contradiction_events": int(
            source_summary.get("contradiction_events", 0) or 0
        ),
        "event_records": len(event_records),
        "candidate_mechanism_clusters": len(clusters),
        "multi_budget_clusters": len(multi_budget_clusters),
        "robust_multi_budget_clusters": len(robust_clusters),
        "ready_for_A32_review_candidates": len(a32_frontiers),
        "ready_for_A32_review_is_not_verdict": True,
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "budgets_covered": sorted(
            {int(row.get("budget", 0) or 0) for row in event_records}
        ),
        "actions_covered": _counts(row.get("target_action", "") for row in event_records),
        "families_covered": _counts(
            row.get("hypothesis_family", "") for row in event_records
        ),
        "raw_support_events": raw_support_events,
        "raw_contradiction_events": raw_contradiction_events,
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in event_records
        ),
        "unique_contexts": len(
            {str(row.get("context_snapshot_hash", "")) for row in event_records}
        ),
        "unique_dedup_keys": len(
            {str(row.get("dedup_key", "")) for row in event_records if row.get("dedup_key")}
        ),
        "dedup_keys_counted_as_independent": False,
        "clustered_support_events_counted_as_support": False,
        "cluster_status_counted_as_scientific_verdict": False,
        "candidate_a32_frontier_counted_as_revision": False,
        "execution_performed": False,
        "gate_passed": gate_passed,
        "outcome_status": (
            SAGE5F_CLUSTERED if gate_passed else SAGE5F_NO_MULTIBUDGET_CLUSTER
        ),
        "support": 0,
        "truth_status": SAGE5F_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "policy_result_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_sage5e_source(source: Mapping[str, Any]) -> None:
    summary = dict(source.get("summary", {}) or {})
    if int(source.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("SAGE.5e support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.5e summary support must remain 0")
    if str(source.get("revision_status", "CANDIDATE_ONLY")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.5e must remain candidate-only")
    if bool(source.get("revision_performed", False)):
        raise ValueError("SAGE.5e must not perform revision")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.5e wrong_confirmations must remain 0")
    if bool(source.get("support_events_counted_as_support", False)):
        raise ValueError("SAGE.5e support events cannot count as support")
    if bool(source.get("mini_frontier_execution_counted_as_evidence", False)):
        raise ValueError("SAGE.5e execution cannot count as evidence")
    if bool(source.get("policy_result_counted_as_confirmation", False)):
        raise ValueError("policy result cannot count as confirmation")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.5e cannot write A32/A33")
    for row in source.get("controlled_experiments", []) or []:
        if isinstance(row, Mapping) and int(row.get("support", 0) or 0) != 0:
            raise ValueError("SAGE.5e experiment support must remain 0")
        if isinstance(row, Mapping) and bool(row.get("support_events_counted_as_support", False)):
            raise ValueError("experiment support events cannot count as support")


def bbox_shape(raw_bbox: Any) -> str:
    if not isinstance(raw_bbox, Mapping):
        return ""
    width = int(raw_bbox.get("x_max", 0) or 0) - int(raw_bbox.get("x_min", 0) or 0) + 1
    height = int(raw_bbox.get("y_max", 0) or 0) - int(raw_bbox.get("y_min", 0) or 0) + 1
    if width <= 0 or height <= 0:
        return ""
    return f"{width}x{height}"


def write_sage5f_mini_frontier_event_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _diff_signature_from_request_or_measurement(
    request: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> Dict[str, Any]:
    target = dict(experiment.get("target_measurement", {}) or {})
    return {
        "changed_cells": int(target.get("changed_pixels", 0) or 0),
        "changed_bbox": target.get("patch_bbox"),
        "color_transitions": dict(request.get("color_transitions", {}) or {}),
        "component_delta_by_color": dict(
            request.get("component_delta_by_color", {}) or {}
        ),
        "terminal_after": False,
        "levels_delta": 0,
    }


def _budget_from_row(row: Mapping[str, Any]) -> int:
    transition = str(row.get("source_transition_id", ""))
    marker = "::budget_"
    if marker in transition:
        tail = transition.split(marker, 1)[1]
        return int(tail.split("::", 1)[0].split("_", 1)[0])
    return 0


def _step_from_row(row: Mapping[str, Any]) -> int:
    transition = str(row.get("source_transition_id", ""))
    marker = "::step_"
    if marker in transition:
        tail = transition.split(marker, 1)[1]
        return int(tail.split("::", 1)[0].split("_", 1)[0])
    return 0


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / float(len(values)), 6)


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate SAGE.5e mini-frontier events candidate-only.",
    )
    parser.add_argument(
        "--source-sage5e",
        default=str(DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_SAGE5F_MINI_FRONTIER_EVENT_CONSOLIDATION_PATH),
    )
    args = parser.parse_args(argv)
    run_sage5f_mini_frontier_event_consolidation(
        source_sage5e_path=args.source_sage5e,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
