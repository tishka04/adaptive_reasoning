"""Exact-state prospective evaluation primitives for SAGE.T12.6.1d."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive import _action_from_payload, abstract_state_from_payload
from .future_viability import FutureViabilityObservation, local_hazard_signature
from .future_viability_hierarchy import (
    HierarchicalFutureViabilityModel,
    HierarchicalViabilityObservation,
    local_composition_signature,
)
from .future_viability_reliability_hierarchy import (
    ReliabilityGatedFutureViabilityModel,
)

PROSPECTIVE_PREDICTION_FORMAT = "sage-t12.6.1d-label-blind-prediction-commitment-v2"


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


def _resolve(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


@dataclass(frozen=True)
class ExactStateCandidate:
    group_id: str
    candidate_id: str
    search_seed: int
    lineage_seed: int
    source_arms: tuple[str, ...]
    archive_sha256: str
    archive_sha256s: tuple[str, ...]
    source_cell_id: str
    source_exact_hash: str
    action_key: str
    action_name: str
    model_observation: HierarchicalViabilityObservation


@dataclass(frozen=True)
class ExactStateExtraction:
    candidates: tuple[ExactStateCandidate, ...]
    labels: Mapping[str, int]
    metrics: Mapping[str, Any]


def _exact_productive_reach(
    exact_hash: str,
    *,
    outgoing: Mapping[str, Sequence[Mapping[str, Any]]],
    remaining_horizon: int,
    visited: frozenset[str],
) -> int:
    """Maximum observed changed, non-terminal exact-state path length."""

    if remaining_horizon <= 0 or exact_hash in visited:
        return 0
    next_visited = visited | {exact_hash}
    return max(
        [0]
        + [
            1
            + _exact_productive_reach(
                str(edge["target_exact_hash"]),
                outgoing=outgoing,
                remaining_horizon=remaining_horizon - 1,
                visited=next_visited,
            )
            for edge in outgoing.get(exact_hash, ())
            if not bool(edge.get("terminal")) and bool(edge.get("changed"))
        ]
    )


def _transition_outcome(edge: Mapping[str, Any]) -> tuple[Any, ...]:
    """Exclude archive-order novelty while retaining transition semantics."""

    return (
        bool(edge.get("terminal")),
        bool(edge.get("changed")),
        bool(edge.get("success")),
        int(edge.get("level_delta", 0)),
        str(edge.get("target_exact_hash")),
    )


def extract_exact_state_candidates(
    *,
    archive_metas: Sequence[Mapping[str, Any]],
    root: Path,
    expected_search_seeds: Sequence[int],
    expected_lineages: Sequence[int],
    expected_arms: Sequence[str],
    future_horizon: int,
    local_radius: int,
    include_labels: bool,
) -> ExactStateExtraction:
    """Deduplicate archives and create exact-replay decision contexts."""

    expected_seeds = {int(value) for value in expected_search_seeds}
    expected_lineages_set = {int(value) for value in expected_lineages}
    expected_arms_set = {str(value) for value in expected_arms}
    conditions: set[tuple[int, int, str]] = set()
    by_archive: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for meta in archive_metas:
        seed = int(meta["search_seed"])
        lineage = int(meta["lineage_seed"])
        arm = str(meta["arm"])
        if seed not in expected_seeds:
            raise ValueError("T12.6.1d archive has an unregistered search seed")
        if lineage not in expected_lineages_set:
            raise ValueError("T12.6.1d archive has an unregistered lineage")
        if arm not in expected_arms_set:
            raise ValueError("T12.6.1d archive has an unregistered arm")
        condition = (seed, lineage, arm)
        if condition in conditions:
            raise ValueError("T12.6.1d archive condition is duplicated")
        conditions.add(condition)
        by_archive[(seed, lineage, str(meta["sha256"]))].append(meta)

    context_edges: dict[tuple[int, int], dict[tuple[str, str], Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    context_arms: dict[tuple[int, int], dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    context_shas: dict[tuple[int, int], dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    context_novelty: dict[tuple[int, int], dict[tuple[str, str], bool]] = defaultdict(
        dict
    )
    context_states: dict[tuple[int, int], dict[str, tuple[str, Any]]] = defaultdict(
        dict
    )
    global_transitions: dict[tuple[str, str], tuple[Any, ...]] = {}
    conflicting_exact_actions: set[tuple[str, str]] = set()
    abstraction_conflicts: set[tuple[int, int, str]] = set()
    novelty_only_repetitions = 0
    repeated_exact_actions = 0
    raw_archive_count = len(archive_metas)
    total_edges = 0
    multi_action_exact_groups = 0
    label_variable_exact_groups = 0
    source_arm_multiplicities: dict[str, int] = defaultdict(int)

    for (seed, lineage, archive_sha), metas in sorted(by_archive.items()):
        context = (seed, lineage)
        paths = sorted(
            {_resolve(str(meta["path"]), root=root) for meta in metas},
            key=str,
        )
        path = paths[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = {
            str(row["cell_id"]): abstract_state_from_payload(dict(row["state"]))
            for row in payload.get("cells", ())
        }
        raw_edges = tuple(payload.get("edges", ()))
        total_edges += len(raw_edges)
        source_arms = tuple(sorted(str(meta["arm"]) for meta in metas))
        source_arm_multiplicities[str(len(source_arms))] += 1
        local_edges: dict[tuple[str, str], Mapping[str, Any]] = {}
        local_novelty: dict[tuple[str, str], bool] = {}
        local_states: dict[str, tuple[str, Any]] = {}
        for edge in raw_edges:
            source_exact = str(edge["source_exact_hash"])
            source_cell = str(edge["source_cell_id"])
            if source_cell not in cells:
                raise ValueError("T12.6.1d exact source has no abstract state")
            action = _action_from_payload(dict(edge["action"]))
            key = (source_exact, action.key)
            prior_state = local_states.get(source_exact)
            if (
                prior_state is not None
                and prior_state[1].signature != cells[source_cell].signature
            ):
                abstraction_conflicts.add((seed, lineage, source_exact))
            else:
                local_states.setdefault(source_exact, (source_cell, cells[source_cell]))
            previous = local_edges.get(key)
            if previous is not None:
                repeated_exact_actions += 1
                if _transition_outcome(previous) != _transition_outcome(edge):
                    conflicting_exact_actions.add(key)
                    continue
                if local_novelty[key] != bool(edge.get("novel")):
                    novelty_only_repetitions += 1
                continue
            local_edges[key] = edge
            local_novelty[key] = bool(edge.get("novel"))

        for source_exact, state_row in local_states.items():
            previous = context_states[context].get(source_exact)
            if previous is not None and previous[1].signature != state_row[1].signature:
                abstraction_conflicts.add((seed, lineage, source_exact))
            else:
                context_states[context].setdefault(source_exact, state_row)
        for key, edge in local_edges.items():
            outcome = _transition_outcome(edge)
            global_previous = global_transitions.get(key)
            if global_previous is not None and global_previous != outcome:
                conflicting_exact_actions.add(key)
            else:
                global_transitions.setdefault(key, outcome)
            previous = context_edges[context].get(key)
            if previous is not None:
                repeated_exact_actions += 1
                if _transition_outcome(previous) != outcome:
                    conflicting_exact_actions.add(key)
                elif context_novelty[context][key] != local_novelty[key]:
                    novelty_only_repetitions += 1
            else:
                context_edges[context][key] = edge
                context_novelty[context][key] = local_novelty[key]
            context_arms[context][key].update(source_arms)
            context_shas[context][key].add(archive_sha)

    candidates: list[ExactStateCandidate] = []
    labels: dict[str, int] = {}
    for (seed, lineage), edges_by_key in sorted(context_edges.items()):
        outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        grouped: dict[str, list[tuple[Any, Mapping[str, Any]]]] = defaultdict(list)
        for (source_exact, _), edge in edges_by_key.items():
            outgoing[source_exact].append(edge)
            grouped[source_exact].append(
                (_action_from_payload(dict(edge["action"])), edge)
            )
        for source_exact, action_edges in sorted(grouped.items()):
            if len(action_edges) < 2:
                continue
            multi_action_exact_groups += 1
            state_row = context_states[(seed, lineage)].get(source_exact)
            if state_row is None:
                raise ValueError("T12.6.1d exact source has no abstract state")
            source_cell, state = state_row
            group_id = _checksum(
                {
                    "lineage_seed": lineage,
                    "search_seed": seed,
                    "source_exact_hash": source_exact,
                }
            )
            group_keys = [(source_exact, action.key) for action, _ in action_edges]
            group_arms = tuple(
                sorted(
                    {
                        arm
                        for key in group_keys
                        for arm in context_arms[(seed, lineage)][key]
                    }
                )
            )
            group_shas = tuple(
                sorted(
                    {
                        sha
                        for key in group_keys
                        for sha in context_shas[(seed, lineage)][key]
                    }
                )
            )
            group_labels = []
            for action, edge in sorted(action_edges, key=lambda item: item[0].key):
                candidate_id = _checksum(
                    {"action_key": action.key, "group_id": group_id}
                )
                base = FutureViabilityObservation(
                    group_id=group_id,
                    corpus="prospective",
                    search_seed=seed,
                    lineage_seed=lineage,
                    arm="+".join(group_arms),
                    source_exact_hash=source_exact,
                    action_key=action.key,
                    action_name=action.action_name,
                    coordinate_grounded=bool(action.action_data),
                    local_signature=local_hazard_signature(
                        state, action, radius=int(local_radius)
                    ),
                    productive_reach=0,
                    immediate_score=0,
                    terminal=False,
                    changed=False,
                    novel=False,
                )
                model_observation = HierarchicalViabilityObservation(
                    base=base,
                    composition_signature=local_composition_signature(
                        state, action, radius=int(local_radius)
                    ),
                )
                candidates.append(
                    ExactStateCandidate(
                        group_id=group_id,
                        candidate_id=candidate_id,
                        search_seed=seed,
                        lineage_seed=lineage,
                        source_arms=group_arms,
                        archive_sha256=group_shas[0],
                        archive_sha256s=group_shas,
                        source_cell_id=source_cell,
                        source_exact_hash=source_exact,
                        action_key=action.key,
                        action_name=action.action_name,
                        model_observation=model_observation,
                    )
                )
                if include_labels:
                    label = (
                        0
                        if bool(edge.get("terminal"))
                        else _exact_productive_reach(
                            str(edge["target_exact_hash"]),
                            outgoing=outgoing,
                            remaining_horizon=int(future_horizon),
                            visited=frozenset({source_exact}),
                        )
                    )
                    labels[candidate_id] = label
                    group_labels.append(label)
            if include_labels and len(set(group_labels)) > 1:
                label_variable_exact_groups += 1

    expected_conditions = {
        (seed, lineage, arm)
        for seed in expected_seeds
        for lineage in expected_lineages_set
        for arm in expected_arms_set
    }
    metrics = {
        "all_archive_conditions_present": conditions == expected_conditions,
        "candidate_count": len(candidates),
        "exact_state_abstraction_conflicts": len(abstraction_conflicts),
        "exact_transition_conflicts": len(conflicting_exact_actions),
        "expected_archive_condition_count": len(expected_conditions),
        "label_variable_exact_groups": label_variable_exact_groups,
        "multi_action_exact_groups": multi_action_exact_groups,
        "novelty_only_repetitions": novelty_only_repetitions,
        "raw_archive_count": raw_archive_count,
        "repeated_exact_actions": repeated_exact_actions,
        "search_seeds": sorted({seed for seed, _, _ in conditions}),
        "source_arm_multiplicities": dict(sorted(source_arm_multiplicities.items())),
        "source_lineages": sorted({lineage for _, lineage, _ in conditions}),
        "total_archive_edges": total_edges,
        "scored_archive_count": len(by_archive),
        "unique_archive_count": len({key[2] for key in by_archive}),
    }
    return ExactStateExtraction(tuple(candidates), labels, metrics)


def commit_label_blind_predictions(
    extraction: ExactStateExtraction,
    *,
    future_model: ReliabilityGatedFutureViabilityModel,
    immediate_model: ReliabilityGatedFutureViabilityModel,
    incumbent_model: HierarchicalFutureViabilityModel,
    binding_shift: int,
) -> dict[str, Any]:
    if extraction.labels:
        raise ValueError("T12.6.1d prediction commitment received opened labels")
    if int(extraction.metrics["exact_transition_conflicts"]) != 0:
        raise ValueError("T12.6.1d exact-state transition integrity failed")
    if int(extraction.metrics.get("exact_state_abstraction_conflicts", 0)) != 0:
        raise ValueError("T12.6.1d exact-state abstraction integrity failed")
    grouped: dict[str, list[ExactStateCandidate]] = defaultdict(list)
    for candidate in extraction.candidates:
        grouped[candidate.group_id].append(candidate)
    rows = []
    for group_id, raw in sorted(grouped.items()):
        candidates = sorted(raw, key=lambda item: item.action_key)
        future = [
            future_model.score_with_audit(item.model_observation) for item in candidates
        ]
        immediate = [
            immediate_model.score(item.model_observation) for item in candidates
        ]
        incumbent = [
            incumbent_model.score(item.model_observation) for item in candidates
        ]
        future_scores = [float(item[0]) for item in future]
        shift = int(binding_shift) % len(candidates)
        swapped = future_scores[shift:] + future_scores[:shift]
        for index, candidate in enumerate(candidates):
            rows.append(
                {
                    "action_key": candidate.action_key,
                    "action_name": candidate.action_name,
                    "archive_sha256": candidate.archive_sha256,
                    "archive_sha256s": list(candidate.archive_sha256s),
                    "binding_swap_score": swapped[index],
                    "candidate_id": candidate.candidate_id,
                    "future_exact_audit": dict(future[index][2]),
                    "future_score": future_scores[index],
                    "future_support_tier": future[index][1],
                    "group_id": group_id,
                    "immediate_score": float(immediate[index][0]),
                    "immediate_support_tier": immediate[index][1],
                    "incumbent_score": float(incumbent[index][0]),
                    "incumbent_support_tier": incumbent[index][1],
                    "lineage_seed": candidate.lineage_seed,
                    "search_seed": candidate.search_seed,
                    "source_arms": list(candidate.source_arms),
                }
            )
    payload = {
        "extraction_metrics": dict(extraction.metrics),
        "format_version": PROSPECTIVE_PREDICTION_FORMAT,
        "label_fields_present": False,
        "rows": rows,
    }
    return {**payload, "prediction_checksum": _checksum(payload)}


def verify_prediction_commitment(payload: Mapping[str, Any]) -> None:
    if payload.get("format_version") != PROSPECTIVE_PREDICTION_FORMAT:
        raise ValueError("unsupported T12.6.1d prediction commitment")
    unsigned = dict(payload)
    checksum = str(unsigned.pop("prediction_checksum"))
    if _checksum(unsigned) != checksum:
        raise ValueError("T12.6.1d prediction commitment checksum mismatch")
    if payload.get("label_fields_present") is not False:
        raise ValueError("T12.6.1d prediction commitment opened labels")
    forbidden = {"productive_reach", "future_label", "oracle", "hit"}
    for row in payload.get("rows", ()):
        if forbidden & set(row):
            raise ValueError("T12.6.1d prediction row contains a forbidden label")


def _selected_index(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return max(
        range(len(rows)),
        key=lambda index: (float(rows[index][field]), str(rows[index]["action_key"])),
    )


def summarize_prospective_cells(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(cells)
    future = sum(bool(row["future_binding_hit"]) for row in cells)
    incumbent = sum(bool(row["incumbent_binding_hit"]) for row in cells)
    immediate = sum(bool(row["immediate_binding_hit"]) for row in cells)
    swap = sum(bool(row["binding_swap_hit"]) for row in cells)
    supported = sum(bool(row["hierarchy_supported"]) for row in cells)
    unique = sum(bool(row["unique_top_score"]) for row in cells)
    recommendations = sum(bool(row["recommendation_issued"]) for row in cells)
    recommendation_hits = sum(bool(row["recommendation_hit"]) for row in cells)
    rejected = sum(bool(row["exact_rejection_exercised"]) for row in cells)
    changed = sum(bool(row["selection_changed_from_incumbent"]) for row in cells)
    corrected = sum(bool(row["incumbent_corrected"]) for row in cells)
    worsened = sum(bool(row["incumbent_worsened"]) for row in cells)
    tiers = (
        "reliable_exact_local_signature",
        "local_composition_signature",
        "action_family_backoff",
        "global_backoff",
    )
    return {
        "binding_swap_hits": swap,
        "binding_swap_top1_accuracy": swap / max(1, count),
        "eligible_groups": count,
        "exact_rejection_exercised_groups": rejected,
        "exact_rejection_exercised_rate": rejected / max(1, count),
        "future_binding_hits": future,
        "future_binding_top1_accuracy": future / max(1, count),
        "future_gain_over_binding_swap": (future - swap) / max(1, count),
        "future_gain_over_immediate": (future - immediate) / max(1, count),
        "future_gain_over_incumbent": (future - incumbent) / max(1, count),
        "hierarchy_coverage": supported / max(1, count),
        "immediate_binding_hits": immediate,
        "immediate_binding_top1_accuracy": immediate / max(1, count),
        "incumbent_binding_hits": incumbent,
        "incumbent_binding_top1_accuracy": incumbent / max(1, count),
        "incumbent_corrected": corrected,
        "incumbent_worsened": worsened,
        "recommendation_accuracy": recommendation_hits / max(1, recommendations),
        "recommendation_coverage": recommendations / max(1, count),
        "recommendation_hits": recommendation_hits,
        "recommendations": recommendations,
        "selected_support_tier_counts": {
            tier: sum(row["selected_support_tier"] == tier for row in cells)
            for tier in tiers
        },
        "selection_changed_from_incumbent": changed,
        "unique_top_rate": unique / max(1, count),
    }


def _seed_blocked_bootstrap_lower_bound(
    cells: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
    lower_quantile: float,
) -> float:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[int(cell["search_seed"])].append(cell)
    search_seeds = sorted(grouped)
    if not search_seeds or repetitions <= 0:
        return -1.0
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(repetitions)):
        sampled = [rng.choice(search_seeds) for _ in search_seeds]
        future = 0
        incumbent = 0
        count = 0
        for sampled_seed in sampled:
            rows = grouped[sampled_seed]
            future += sum(bool(row["future_binding_hit"]) for row in rows)
            incumbent += sum(bool(row["incumbent_binding_hit"]) for row in rows)
            count += len(rows)
        estimates.append((future - incumbent) / max(1, count))
    estimates.sort()
    index = max(
        0,
        min(
            len(estimates) - 1,
            int(math.floor(float(lower_quantile) * (len(estimates) - 1))),
        ),
    )
    return float(estimates[index])


def adjudicate_prediction_commitment(
    commitment: Mapping[str, Any],
    extraction: ExactStateExtraction,
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    bootstrap_lower_quantile: float,
) -> dict[str, Any]:
    verify_prediction_commitment(commitment)
    if int(extraction.metrics["exact_transition_conflicts"]) != 0:
        raise ValueError("T12.6.1d exact-state transition integrity failed")
    if int(extraction.metrics.get("exact_state_abstraction_conflicts", 0)) != 0:
        raise ValueError("T12.6.1d exact-state abstraction integrity failed")
    committed_rows = {str(row["candidate_id"]): row for row in commitment["rows"]}
    extracted_ids = {item.candidate_id for item in extraction.candidates}
    if set(committed_rows) != extracted_ids or set(extraction.labels) != extracted_ids:
        raise ValueError("T12.6.1d committed candidate registry changed")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id, row in committed_rows.items():
        grouped[str(row["group_id"])].append(
            {**dict(row), "productive_reach": int(extraction.labels[candidate_id])}
        )
    cells = []
    for group_id, raw in sorted(grouped.items()):
        rows = sorted(raw, key=lambda item: str(item["action_key"]))
        if len(rows) < 2 or len({int(row["productive_reach"]) for row in rows}) < 2:
            continue
        future_index = _selected_index(rows, "future_score")
        incumbent_index = _selected_index(rows, "incumbent_score")
        immediate_index = _selected_index(rows, "immediate_score")
        swap_index = _selected_index(rows, "binding_swap_score")
        best = max(int(row["productive_reach"]) for row in rows)
        future_hit = int(rows[future_index]["productive_reach"]) == best
        incumbent_hit = int(rows[incumbent_index]["productive_reach"]) == best
        future_scores = [float(row["future_score"]) for row in rows]
        unique = sum(score == max(future_scores) for score in future_scores) == 1
        tier = str(rows[future_index]["future_support_tier"])
        supported = tier in {
            "reliable_exact_local_signature",
            "local_composition_signature",
        }
        recommendation = bool(unique and supported)
        changed = future_index != incumbent_index
        cells.append(
            {
                "archive_sha256": rows[0]["archive_sha256"],
                "archive_sha256s": list(rows[0]["archive_sha256s"]),
                "binding_swap_hit": int(rows[swap_index]["productive_reach"]) == best,
                "exact_rejection_exercised": any(
                    bool(row["future_exact_audit"]["exact_candidate_present"])
                    and not bool(row["future_exact_audit"]["exact_candidate_reliable"])
                    for row in rows
                ),
                "future_binding_hit": future_hit,
                "group_id": group_id,
                "hierarchy_supported": supported,
                "immediate_binding_hit": (
                    int(rows[immediate_index]["productive_reach"]) == best
                ),
                "incumbent_binding_hit": incumbent_hit,
                "incumbent_corrected": bool(
                    changed and future_hit and not incumbent_hit
                ),
                "incumbent_worsened": bool(
                    changed and incumbent_hit and not future_hit
                ),
                "lineage_seed": int(rows[0]["lineage_seed"]),
                "recommendation_hit": bool(recommendation and future_hit),
                "recommendation_issued": recommendation,
                "search_seed": int(rows[0]["search_seed"]),
                "selected_action_key": rows[future_index]["action_key"],
                "selected_support_tier": tier,
                "selection_changed_from_incumbent": changed,
                "source_arms": list(rows[0]["source_arms"]),
                "unique_top_score": unique,
            }
        )
    per_seed = {
        str(seed): summarize_prospective_cells(
            [cell for cell in cells if int(cell["search_seed"]) == seed]
        )
        for seed in sorted({int(cell["search_seed"]) for cell in cells})
    }
    per_lineage = {
        str(lineage): summarize_prospective_cells(
            [cell for cell in cells if int(cell["lineage_seed"]) == lineage]
        )
        for lineage in sorted({int(cell["lineage_seed"]) for cell in cells})
    }
    metrics = summarize_prospective_cells(cells)
    metrics["bootstrap_gain_lower_bound_90"] = _seed_blocked_bootstrap_lower_bound(
        cells,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
        lower_quantile=bootstrap_lower_quantile,
    )
    metrics["bootstrap_lower_quantile"] = bootstrap_lower_quantile
    metrics["bootstrap_repetitions"] = bootstrap_repetitions
    metrics["bootstrap_seed"] = bootstrap_seed
    metrics["per_lineage"] = per_lineage
    metrics["per_search_seed"] = per_seed
    return {"cells": cells, "metrics": metrics}


__all__ = [
    "PROSPECTIVE_PREDICTION_FORMAT",
    "ExactStateCandidate",
    "ExactStateExtraction",
    "_exact_productive_reach",
    "adjudicate_prediction_commitment",
    "commit_label_blind_predictions",
    "extract_exact_state_candidates",
    "summarize_prospective_cells",
    "verify_prediction_commitment",
]
