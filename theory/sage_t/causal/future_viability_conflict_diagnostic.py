"""Post-hoc transition-conflict sensitivities for SAGE.T12.6.1a."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive import _action_from_payload, abstract_state_from_payload
from .future_viability import FutureViabilityObservation, _productive_reach
from .future_viability_hierarchy import (
    HierarchicalViabilityObservation,
    local_composition_signature,
)
from .hazard_diversity_model import local_hazard_signature

CONFLICT_DIAGNOSTIC_FORMAT = (
    "sage-t12.6.1a-future-viability-conflict-diagnostic-v1"
)
CONSOLIDATION_POLICIES = (
    "parent_order",
    "archive_last",
    "modal_future_label",
    "minimum_future_label",
    "maximum_future_label",
    "drop_conflicted_groups",
)


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _edge_outcome(edge: Mapping[str, Any]) -> tuple[bool, bool, bool, str]:
    return (
        bool(edge.get("terminal")),
        bool(edge.get("changed")),
        bool(edge.get("novel")),
        str(edge.get("target_cell_id")),
    )


def _future_label(
    edge: Mapping[str, Any],
    *,
    source_cell_id: str,
    outgoing: Mapping[str, Sequence[Mapping[str, Any]]],
    future_horizon: int,
) -> int:
    if bool(edge.get("terminal")):
        return 0
    return _productive_reach(
        str(edge["target_cell_id"]),
        outgoing=outgoing,
        remaining_horizon=int(future_horizon),
        visited=frozenset({str(source_cell_id)}),
    )


def _immediate_label(edge: Mapping[str, Any]) -> int:
    return (
        4 * int(not bool(edge.get("terminal")))
        + 2 * int(bool(edge.get("changed")))
        + int(bool(edge.get("novel")))
    )


def _parent_selected(
    edges: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Reproduce T12.6.1: retain the last exact repeat, reject later conflicts."""

    outcome = _edge_outcome(edges[0])
    selected = edges[0]
    for edge in edges[1:]:
        if _edge_outcome(edge) != outcome:
            continue
        selected = edge
    return selected


def _select_edge(
    edges: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    source_cell_id: str,
    outgoing: Mapping[str, Sequence[Mapping[str, Any]]],
    future_horizon: int,
) -> Mapping[str, Any]:
    if policy in {"parent_order", "drop_conflicted_groups"}:
        return _parent_selected(edges)
    if policy == "archive_last":
        return edges[-1]

    labelled = [
        (
            _future_label(
                edge,
                source_cell_id=source_cell_id,
                outgoing=outgoing,
                future_horizon=future_horizon,
            ),
            _immediate_label(edge),
            edge,
        )
        for edge in edges
    ]
    if policy == "minimum_future_label":
        return min(
            labelled,
            key=lambda item: (item[0], item[1], _canonical(item[2])),
        )[2]
    if policy == "maximum_future_label":
        return max(
            labelled,
            key=lambda item: (item[0], item[1], _canonical(item[2])),
        )[2]
    if policy == "modal_future_label":
        label_counts = Counter(
            (value, bool(edge.get("terminal")), bool(edge.get("changed")))
            for value, _, edge in labelled
        )
        selected_label = sorted(
            label_counts,
            key=lambda item: (-label_counts[item], _canonical(item)),
        )[0]
        candidates = [
            edge
            for value, _, edge in labelled
            if (
                value,
                bool(edge.get("terminal")),
                bool(edge.get("changed")),
            )
            == selected_label
        ]
        return min(candidates, key=_canonical)
    raise ValueError(f"unsupported T12.6.1a consolidation policy: {policy}")


@dataclass(frozen=True)
class ConflictDiagnosticExtraction:
    observations_by_policy: Mapping[
        str, tuple[HierarchicalViabilityObservation, ...]
    ]
    conflict_rows: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]


def extract_conflict_sensitivities(
    *,
    archive_metas: Sequence[Mapping[str, Any]],
    root: Path,
    corpus: str,
    expected_search_seeds: Sequence[int],
    expected_lineages: Sequence[int],
    expected_arms: Sequence[str],
    future_horizon: int,
    local_radius: int,
    policies: Sequence[str] = CONSOLIDATION_POLICIES,
) -> ConflictDiagnosticExtraction:
    """Resolve observed duplicates without refitting or changing the raw graph."""

    selected_policies = tuple(str(value) for value in policies)
    if selected_policies != CONSOLIDATION_POLICIES:
        raise ValueError("T12.6.1a consolidation policy set changed")
    expected_seed_set = {int(value) for value in expected_search_seeds}
    expected_lineage_set = {int(value) for value in expected_lineages}
    expected_arm_set = {str(value) for value in expected_arms}
    archive_keys: set[tuple[int, int, str]] = set()
    observations: dict[str, list[HierarchicalViabilityObservation]] = {
        policy: [] for policy in selected_policies
    }
    policy_group_counts: Counter[str] = Counter()
    conflict_rows: list[dict[str, Any]] = []
    conflict_groups: set[tuple[int, int, str, str]] = set()
    conflicting_conditions: set[tuple[int, int, str]] = set()
    conflicting_hashes: set[str] = set()
    total_edges = 0

    for meta in archive_metas:
        search_seed = int(meta["search_seed"])
        lineage_seed = int(meta["lineage_seed"])
        arm = str(meta["arm"])
        archive_key = (search_seed, lineage_seed, arm)
        if search_seed not in expected_seed_set:
            raise ValueError("T12.6.1a archive has an unregistered search seed")
        if lineage_seed not in expected_lineage_set:
            raise ValueError("T12.6.1a archive has an unregistered lineage")
        if arm not in expected_arm_set:
            raise ValueError("T12.6.1a archive has an unregistered arm")
        if archive_key in archive_keys:
            raise ValueError("T12.6.1a archive condition is duplicated")
        archive_keys.add(archive_key)
        path = Path(str(meta["path"]))
        if not path.is_absolute():
            path = root / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        states = {
            str(row["cell_id"]): abstract_state_from_payload(dict(row["state"]))
            for row in payload.get("cells", ())
        }
        outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for edge in payload.get("edges", ()):
            outgoing[str(edge["source_cell_id"])].append(edge)
            total_edges += 1

        for source_cell_id, raw_edges in outgoing.items():
            by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for edge in raw_edges:
                action = _action_from_payload(dict(edge["action"]))
                by_action[action.key].append(edge)
            if len(by_action) < 2:
                continue
            source_state = states.get(source_cell_id)
            if source_state is None:
                raise ValueError("T12.6.1a archive edge has no source state")

            group_conflicted = False
            for action_key, action_edges in sorted(by_action.items()):
                first = action_edges[0]
                first_outcome = _edge_outcome(first)
                first_future = _future_label(
                    first,
                    source_cell_id=source_cell_id,
                    outgoing=outgoing,
                    future_horizon=future_horizon,
                )
                first_immediate = _immediate_label(first)
                for later in action_edges[1:]:
                    later_outcome = _edge_outcome(later)
                    if later_outcome == first_outcome:
                        continue
                    group_conflicted = True
                    conflict_groups.add((*archive_key, source_cell_id))
                    conflicting_conditions.add(archive_key)
                    conflicting_hashes.add(str(meta["sha256"]))
                    later_future = _future_label(
                        later,
                        source_cell_id=source_cell_id,
                        outgoing=outgoing,
                        future_horizon=future_horizon,
                    )
                    later_immediate = _immediate_label(later)
                    difference_fields = [
                        field
                        for field, first_value, later_value in (
                            (
                                "terminal",
                                bool(first.get("terminal")),
                                bool(later.get("terminal")),
                            ),
                            (
                                "changed",
                                bool(first.get("changed")),
                                bool(later.get("changed")),
                            ),
                            (
                                "novel",
                                bool(first.get("novel")),
                                bool(later.get("novel")),
                            ),
                            (
                                "target_cell_id",
                                str(first.get("target_cell_id")),
                                str(later.get("target_cell_id")),
                            ),
                        )
                        if first_value != later_value
                    ]
                    first_target = states.get(str(first.get("target_cell_id")))
                    later_target = states.get(str(later.get("target_cell_id")))
                    conflict_rows.append(
                        {
                            "action_key": action_key,
                            "archive_sha256": str(meta["sha256"]),
                            "arm": arm,
                            "difference_pattern": "+".join(difference_fields),
                            "first_edge_id": str(first.get("edge_id")),
                            "first_future_label": first_future,
                            "first_immediate_label": first_immediate,
                            "first_target_cell_id": str(
                                first.get("target_cell_id")
                            ),
                            "future_label_changed": first_future != later_future,
                            "immediate_label_changed": (
                                first_immediate != later_immediate
                            ),
                            "later_edge_id": str(later.get("edge_id")),
                            "later_future_label": later_future,
                            "later_immediate_label": later_immediate,
                            "later_target_cell_id": str(
                                later.get("target_cell_id")
                            ),
                            "lineage_seed": lineage_seed,
                            "same_abstract_target_state": (
                                first_target is not None
                                and first_target == later_target
                            ),
                            "same_target_cell": str(
                                first.get("target_cell_id")
                            )
                            == str(later.get("target_cell_id")),
                            "search_seed": search_seed,
                            "source_cell_id": source_cell_id,
                        }
                    )

            group_id = _checksum(
                {
                    "arm": arm,
                    "corpus": corpus,
                    "lineage_seed": lineage_seed,
                    "search_seed": search_seed,
                    "source_cell_id": source_cell_id,
                }
            )
            for policy in selected_policies:
                if policy == "drop_conflicted_groups" and group_conflicted:
                    continue
                policy_group_counts[policy] += 1
                for action_key, action_edges in sorted(by_action.items()):
                    edge = _select_edge(
                        action_edges,
                        policy=policy,
                        source_cell_id=source_cell_id,
                        outgoing=outgoing,
                        future_horizon=future_horizon,
                    )
                    action = _action_from_payload(dict(edge["action"]))
                    base = FutureViabilityObservation(
                        group_id=group_id,
                        corpus=str(corpus),
                        search_seed=search_seed,
                        lineage_seed=lineage_seed,
                        arm=arm,
                        source_exact_hash=str(edge["source_exact_hash"]),
                        action_key=action_key,
                        action_name=action.action_name,
                        coordinate_grounded=bool(action.action_data),
                        local_signature=local_hazard_signature(
                            source_state,
                            action,
                            radius=int(local_radius),
                        ),
                        productive_reach=_future_label(
                            edge,
                            source_cell_id=source_cell_id,
                            outgoing=outgoing,
                            future_horizon=future_horizon,
                        ),
                        immediate_score=_immediate_label(edge),
                        terminal=bool(edge.get("terminal")),
                        changed=bool(edge.get("changed")),
                        novel=bool(edge.get("novel")),
                    )
                    observations[policy].append(
                        HierarchicalViabilityObservation(
                            base=base,
                            composition_signature=local_composition_signature(
                                source_state,
                                action,
                                radius=int(local_radius),
                            ),
                        )
                    )

    expected_keys = {
        (seed, lineage, arm)
        for seed in expected_seed_set
        for lineage in expected_lineage_set
        for arm in expected_arm_set
    }
    difference_counts = Counter(
        str(row["difference_pattern"]) for row in conflict_rows
    )
    metrics = {
        "all_archive_conditions_present": archive_keys == expected_keys,
        "archive_condition_count": len(archive_keys),
        "conflict_difference_pattern_counts": {
            key: difference_counts[key] for key in sorted(difference_counts)
        },
        "conflicted_archive_condition_count": len(conflicting_conditions),
        "conflicted_decision_group_count": len(conflict_groups),
        "consolidation_policies": list(selected_policies),
        "future_label_conflicts": sum(
            bool(row["future_label_changed"]) for row in conflict_rows
        ),
        "immediate_label_conflicts": sum(
            bool(row["immediate_label_changed"]) for row in conflict_rows
        ),
        "parent_duplicate_action_conflicts": len(conflict_rows),
        "policy_group_counts": dict(sorted(policy_group_counts.items())),
        "policy_observation_counts": {
            policy: len(rows) for policy, rows in sorted(observations.items())
        },
        "search_seeds": sorted(expected_seed_set),
        "source_lineages": sorted(expected_lineage_set),
        "total_archive_edges": total_edges,
        "unique_conflicted_archive_payloads": len(conflicting_hashes),
    }
    return ConflictDiagnosticExtraction(
        observations_by_policy={
            policy: tuple(rows) for policy, rows in observations.items()
        },
        conflict_rows=tuple(conflict_rows),
        metrics=metrics,
    )


__all__ = [
    "CONFLICT_DIAGNOSTIC_FORMAT",
    "CONSOLIDATION_POLICIES",
    "ConflictDiagnosticExtraction",
    "extract_conflict_sensitivities",
]
