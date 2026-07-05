"""TaskProgram-guided level-7 repair for AR25.

This diagnostic uses the global rule inferred from levels 1..6 instead of
running a generic novelty beam. It calibrates a simple correspondence score
on successful pre-submit states, then searches level 7 for actions that
increase that score before treating ACTION2 as submit/validation.

Scoring modes:
    default: Match-first scoring prioritizes shape correspondence (original)
    global_correspondence: Family/topology-first scoring, independent of local
        matched_pairs except for diagnostic reporting.
    global_semantic_hybrid: Family/topology-first search with local readiness,
        dotted-violation and unmatched-component repair pressure.
    action_ontology_guided: Goal ontology plus empirical ACTION_i affordances
        learned by discover_action_ontology.py.
    dotted_constraint_repair: Constraint-first scoring prioritizes fixing
        boundary violations before matching. Uses aggressive bonuses for:
        - Reducing dotted_constraint_violations (+1000)
        - Cursor proximity to violations (+200)
        - Unmatched reduction (+100)
        - ACTION6 transforming near violations (+50)
        - Penalty for increasing violations (-500)

Examples:
    python task_program_guided_level7.py --game ar25 --horizon 8 --beam-width 48
    python task_program_guided_level7.py --game ar25 --scoring-mode dotted_constraint_repair
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from level7_frontier_recovery import (
    ACTION_NAMES,
    PROJECT_ROOT,
    Arcade,
    OperationMode,
    _available_names_from_raw,
    _component_delta,
    _coordinate_candidates,
    _diff_bbox,
    _hash_grid,
    _primary_grid,
    _replay_prefix,
    _resolve_full_game_id,
    _state_name,
    _step_branch,
    find_level_frontier,
)
from trace_replay_verifier import _load_selected_episode
from trace_rule_inference import EpisodeBundle, build_level_segments


DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "task_program_guided_level7"
DEFAULT_RULE_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "rule_inference"
DEFAULT_ACTION_ONTOLOGY_DIR = PROJECT_ROOT / "diagnostics" / "action_ontology"
DEFAULT_HUMAN_BREAKS_DIR = PROJECT_ROOT / "diagnostics" / "human_biggest_second_breaks"


AUTO_LEVELUP_GAP_FEATURES = [
    "first_count",
    "second_count",
    "first_size_sum",
    "second_size_sum",
    "first_largest_size",
    "second_largest_size",
    "largest_offender_size",
    "unmatched_total",
    "offending_count",
    "dotted_violations",
    "mean_x_offset",
    "mean_y_offset",
    "nearest_distance_mean",
    "nearest_distance_min",
    "nearest_distance_max",
    "boundary_contact_total",
    "size_balance_abs",
    "count_balance_abs",
]

LEVEL7_FRAGMENTATION_FEATURES = [
    "second_count",
    "second_size_sum",
    "second_largest_size",
    "largest_offender_size",
    "first_largest_size",
    "unmatched_total",
    "nearest_distance_mean",
]


@dataclass(frozen=True)
class Component:
    """Connected component summary for one color."""

    color: int
    size: int
    min_y: int
    min_x: int
    max_y: int
    max_x: int
    centroid_y: float
    centroid_x: float

    @property
    def height(self) -> int:
        return int(self.max_y - self.min_y + 1)

    @property
    def width(self) -> int:
        return int(self.max_x - self.min_x + 1)

    def to_report(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "size": self.size,
            "min_y": self.min_y,
            "min_x": self.min_x,
            "max_y": self.max_y,
            "max_x": self.max_x,
            "centroid_y": round(self.centroid_y, 3),
            "centroid_x": round(self.centroid_x, 3),
            "height": self.height,
            "width": self.width,
        }


@dataclass
class MatchScore:
    """Explicit score for the colored-shape correspondence hypothesis."""

    score: float
    pair_colors: Tuple[int, int]
    matched_pairs: int
    unmatched_first: int
    unmatched_second: int
    shape_overlap_or_alignment: float
    cursor_near_target: int
    dotted_constraint_violations: int
    component_counts: Dict[str, int]
    best_pairs: List[Dict[str, Any]] = field(default_factory=list)

    def to_report(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "pair_colors": list(self.pair_colors),
            "matched_pairs": int(self.matched_pairs),
            "unmatched_first": int(self.unmatched_first),
            "unmatched_second": int(self.unmatched_second),
            "shape_overlap_or_alignment": round(float(self.shape_overlap_or_alignment), 4),
            "cursor_near_target": int(self.cursor_near_target),
            "dotted_constraint_violations": int(self.dotted_constraint_violations),
            "component_counts": dict(self.component_counts),
            "best_pairs": list(self.best_pairs[:8]),
        }


@dataclass
class GlobalCorrespondenceScore:
    """Family-level structural correspondence between two color systems."""

    score: float
    pair_colors: Tuple[int, int]
    structure_similarity: float
    count_similarity: float
    size_distribution_similarity: float
    orientation_similarity: float
    spatial_order_similarity: float
    boundary_relation_similarity: float
    centroid_relation_similarity: float
    component_counts: Dict[str, int]
    family_signatures: Dict[str, Any]

    def to_report(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "pair_colors": list(self.pair_colors),
            "structure_similarity": round(float(self.structure_similarity), 4),
            "count_similarity": round(float(self.count_similarity), 4),
            "size_distribution_similarity": round(float(self.size_distribution_similarity), 4),
            "orientation_similarity": round(float(self.orientation_similarity), 4),
            "spatial_order_similarity": round(float(self.spatial_order_similarity), 4),
            "boundary_relation_similarity": round(float(self.boundary_relation_similarity), 4),
            "centroid_relation_similarity": round(float(self.centroid_relation_similarity), 4),
            "component_counts": dict(self.component_counts),
            "family_signatures": dict(self.family_signatures),
        }


@dataclass
class RuleModel:
    """Calibrated correspondence model from levels 1..6."""

    pair_colors: Tuple[int, int]
    ready_threshold: float
    ready_scores: List[float]
    ready_level_up_actions: Dict[str, int]
    expected_matched_pairs: int
    expected_unmatched_total: int
    pair_color_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_report(self) -> Dict[str, Any]:
        return {
            "pair_colors": list(self.pair_colors),
            "ready_threshold": round(float(self.ready_threshold), 4),
            "ready_scores": [round(float(item), 4) for item in self.ready_scores],
            "ready_level_up_actions": dict(self.ready_level_up_actions),
            "expected_matched_pairs": int(self.expected_matched_pairs),
            "expected_unmatched_total": int(self.expected_unmatched_total),
            "pair_color_candidates": list(self.pair_color_candidates),
        }


@dataclass
class GuidedNode:
    """One branch in TaskProgram-guided level-7 search."""

    actions: List[str]
    action_data: List[Optional[Dict[str, int]]]
    state: str
    level: int
    grid_hash: str
    search_score: float
    match: MatchScore
    depth: int
    global_correspondence: Optional[GlobalCorrespondenceScore] = None
    reached_next_level: bool = False
    died: bool = False
    submit_not_ready: bool = False
    action3_contradiction: bool = False
    readiness_gate_passed: bool = False
    danger_prefix_len: int = 0
    consecutive_submit: int = 0
    submit_gate_passed: bool = False
    spatial_operator_change: int = 0
    max_action6_spatial_change: int = 0
    action6_transform_count: int = 0
    rewrite_saturated: bool = False
    post_saturation_probe_phase: bool = False
    repeat_penalty: float = 0.0
    match_delta_from_root: float = 0.0
    match_delta_from_parent: float = 0.0
    global_delta_from_root: float = 0.0
    global_delta_from_parent: float = 0.0
    unmatched_delta_from_root: int = 0
    dotted_delta_from_root: int = 0
    available_actions: List[str] = field(default_factory=list)
    path_hashes: List[str] = field(default_factory=list)
    env: Any = field(default=None, repr=False, compare=False)
    grid: Optional[List[List[int]]] = field(default=None, repr=False, compare=False)
    # Dotted constraint repair tracking (for logging/analysis)
    violation_components_before: List[Dict[str, Any]] = field(default_factory=list)
    violation_components_after: List[Dict[str, Any]] = field(default_factory=list)
    action_responsible: str = ""
    distance_action6_to_violation: float = 0.0
    cursor_near_dotted_violation: bool = False
    # Submit probe tracking
    consecutive_submits: int = 0
    # Target info for cursor_target mode
    last_target_info: Dict[str, Any] = field(default_factory=dict, repr=False)
    # Track if we've had a valid submit (for forbidding ACTION3 before submit)
    had_valid_submit: bool = False
    # Cumulative counters for rewrite_until_saturation mode
    action3_productive_count: int = 0
    cumulative_changed_cells: int = 0
    action3_changed_cells_sequence: List[int] = field(default_factory=list)
    first_noop_index: Optional[int] = None
    action_affordance: Dict[str, Any] = field(default_factory=dict)
    auto_levelup_state: Dict[str, Any] = field(default_factory=dict)
    fragmentation_guidance: Dict[str, Any] = field(default_factory=dict)
    fragmentation_guardrails: Dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> Dict[str, Any]:
        return {
            "actions": list(self.actions),
            "action_data": list(self.action_data),
            "state": self.state,
            "level": int(self.level),
            "grid_hash": self.grid_hash,
            "search_score": round(float(self.search_score), 4),
            "match_score": self.match.to_report(),
            "global_correspondence_score": (
                self.global_correspondence.to_report()
                if self.global_correspondence is not None
                else None
            ),
            "depth": int(self.depth),
            "reached_next_level": bool(self.reached_next_level),
            "died": bool(self.died),
            "submit_not_ready": bool(self.submit_not_ready),
            "action3_contradiction": bool(self.action3_contradiction),
            "readiness_gate_passed": bool(self.readiness_gate_passed),
            "danger_prefix_len": int(self.danger_prefix_len),
            "consecutive_submit": int(self.consecutive_submit),
            "submit_gate_passed": bool(self.submit_gate_passed),
            "spatial_operator_change": int(self.spatial_operator_change),
            "max_action6_spatial_change": int(self.max_action6_spatial_change),
            "action6_transform_count": int(self.action6_transform_count),
            "rewrite_saturated": bool(self.rewrite_saturated),
            "post_saturation_probe_phase": bool(self.post_saturation_probe_phase),
            "repeat_penalty": round(float(self.repeat_penalty), 4),
            "match_delta_from_root": round(float(self.match_delta_from_root), 4),
            "match_delta_from_parent": round(float(self.match_delta_from_parent), 4),
            "global_delta_from_root": round(float(self.global_delta_from_root), 4),
            "global_delta_from_parent": round(float(self.global_delta_from_parent), 4),
            "unmatched_delta_from_root": int(self.unmatched_delta_from_root),
            "dotted_delta_from_root": int(self.dotted_delta_from_root),
            "available_actions": list(self.available_actions),
            "violation_components_before": list(self.violation_components_before),
            "violation_components_after": list(self.violation_components_after),
            "action_responsible": self.action_responsible,
            "distance_action6_to_violation": round(float(self.distance_action6_to_violation), 4),
            "cursor_near_dotted_violation": bool(self.cursor_near_dotted_violation),
            "consecutive_submits": int(self.consecutive_submits),
            "last_target_info": dict(self.last_target_info),
            "action3_count": int(self.actions.count("ACTION3")),
            "action3_productive_count": int(self.action3_productive_count),
            "cumulative_changed_cells": int(self.cumulative_changed_cells),
            "changed_cells_sequence": list(self.action3_changed_cells_sequence),
            "first_noop_index": self.first_noop_index,
            "unique_grid_hashes": int(len(set(self.path_hashes))),
            "action_affordance": dict(self.action_affordance),
            "auto_levelup_state": dict(self.auto_levelup_state),
            "fragmentation_guidance": dict(self.fragmentation_guidance),
            "fragmentation_guardrails": dict(self.fragmentation_guardrails),
        }


def _connected_components_for_colors(
    grid: Sequence[Sequence[int]],
    colors: Iterable[int],
) -> List[Component]:
    target = set(int(color) for color in colors)
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2:
        return []
    height, width = arr.shape
    seen = np.zeros(arr.shape, dtype=bool)
    out: List[Component] = []
    for y in range(height):
        for x in range(width):
            color = int(arr[y, x])
            if color not in target or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if seen[ny, nx] or int(arr[ny, nx]) != color:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            out.append(
                Component(
                    color=color,
                    size=len(cells),
                    min_y=min(ys),
                    min_x=min(xs),
                    max_y=max(ys),
                    max_x=max(xs),
                    centroid_y=float(sum(ys) / len(ys)),
                    centroid_x=float(sum(xs) / len(xs)),
                )
            )
    out.sort(key=lambda item: (item.color, -item.size, item.min_y, item.min_x))
    return out


def _bbox_iou(a: Component, b: Component) -> float:
    ix0 = max(a.min_x, b.min_x)
    iy0 = max(a.min_y, b.min_y)
    ix1 = min(a.max_x, b.max_x)
    iy1 = min(a.max_y, b.max_y)
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = float((ix1 - ix0 + 1) * (iy1 - iy0 + 1))
    area_a = float(a.width * a.height)
    area_b = float(b.width * b.height)
    return inter / max(1.0, area_a + area_b - inter)


def _pair_alignment(a: Component, b: Component, shape: Tuple[int, int]) -> Dict[str, Any]:
    height, width = shape
    diagonal = float(np.hypot(max(1, height - 1), max(1, width - 1)))
    centroid_distance = float(np.hypot(a.centroid_y - b.centroid_y, a.centroid_x - b.centroid_x))
    distance_score = max(0.0, 1.0 - centroid_distance / max(1.0, diagonal))
    size_score = 1.0 - abs(a.size - b.size) / max(1.0, float(max(a.size, b.size)))
    height_score = 1.0 - abs(a.height - b.height) / max(1.0, float(max(a.height, b.height)))
    width_score = 1.0 - abs(a.width - b.width) / max(1.0, float(max(a.width, b.width)))
    row_alignment = 1.0 - abs(a.centroid_y - b.centroid_y) / max(1.0, float(height - 1))
    col_alignment = 1.0 - abs(a.centroid_x - b.centroid_x) / max(1.0, float(width - 1))
    iou = _bbox_iou(a, b)
    score = (
        1.4 * size_score
        + 0.9 * height_score
        + 0.9 * width_score
        + 1.1 * max(row_alignment, col_alignment)
        + 0.7 * distance_score
        + 1.0 * iou
    )
    return {
        "score": round(float(score), 4),
        "size_score": round(float(size_score), 4),
        "shape_score": round(float((height_score + width_score) * 0.5), 4),
        "row_alignment": round(float(row_alignment), 4),
        "col_alignment": round(float(col_alignment), 4),
        "bbox_iou": round(float(iou), 4),
        "centroid_distance": round(float(centroid_distance), 4),
        "first": a.to_report(),
        "second": b.to_report(),
    }


def _safe_similarity(left: float, right: float) -> float:
    return max(0.0, 1.0 - abs(float(left) - float(right)) / max(1.0, abs(float(left)), abs(float(right))))


def _histogram_similarity(left: Sequence[float], right: Sequence[float], *, bins: int = 8) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    min_value = min(min(left), min(right), 0.0)
    max_value = max(max(left), max(right), 1.0)
    if max_value <= min_value:
        return 1.0
    hist_range = (float(min_value), float(max_value))
    left_hist, _ = np.histogram(np.array(left, dtype=np.float32), bins=bins, range=hist_range)
    right_hist, _ = np.histogram(np.array(right, dtype=np.float32), bins=bins, range=hist_range)
    left_sum = float(left_hist.sum())
    right_sum = float(right_hist.sum())
    if left_sum <= 0 or right_sum <= 0:
        return 0.0
    left_norm = left_hist.astype(np.float32) / left_sum
    right_norm = right_hist.astype(np.float32) / right_sum
    return float(1.0 - 0.5 * np.abs(left_norm - right_norm).sum())


def _component_orientation(component: Component) -> float:
    if component.width == component.height:
        return 0.0
    return float((component.width - component.height) / max(1.0, float(max(component.width, component.height))))


def _family_signature(
    components: Sequence[Component],
    *,
    shape: Tuple[int, int],
) -> Dict[str, Any]:
    height, width = shape
    sizes = [float(component.size) for component in components]
    orientations = [_component_orientation(component) for component in components]
    boundary_contacts = [
        int(
            component.min_y <= 0
            or component.min_x <= 0
            or component.max_y >= height - 1
            or component.max_x >= width - 1
        )
        for component in components
    ]
    centroids = sorted(
        [(float(component.centroid_y), float(component.centroid_x)) for component in components]
    )
    nearest_distances: List[float] = []
    for idx, (y, x) in enumerate(centroids):
        candidates = [
            float(np.hypot(y - oy, x - ox))
            for j, (oy, ox) in enumerate(centroids)
            if j != idx
        ]
        if candidates:
            nearest_distances.append(min(candidates))
    return {
        "count": len(components),
        "sizes": sizes,
        "orientations": orientations,
        "boundary_contact_ratio": float(sum(boundary_contacts) / max(1, len(boundary_contacts))),
        "sorted_y": [item[0] for item in centroids],
        "sorted_x": sorted(item[1] for item in centroids),
        "nearest_distances": nearest_distances,
        "large_component_count": sum(1 for component in components if component.size >= 16),
        "small_component_count": sum(1 for component in components if component.size < 16),
    }


def global_correspondence_score(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
) -> GlobalCorrespondenceScore:
    """Score family-level/topological correspondence, not local component pairs."""

    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    first = _family_signature(first_components, shape=shape)
    second = _family_signature(second_components, shape=shape)

    count_similarity = _safe_similarity(float(first["count"]), float(second["count"]))
    size_distribution_similarity = _histogram_similarity(first["sizes"], second["sizes"], bins=8)
    orientation_similarity = _histogram_similarity(first["orientations"], second["orientations"], bins=8)
    spatial_y_similarity = _histogram_similarity(first["sorted_y"], second["sorted_y"], bins=8)
    spatial_x_similarity = _histogram_similarity(first["sorted_x"], second["sorted_x"], bins=8)
    spatial_order_similarity = 0.5 * spatial_y_similarity + 0.5 * spatial_x_similarity
    boundary_relation_similarity = _safe_similarity(
        float(first["boundary_contact_ratio"]),
        float(second["boundary_contact_ratio"]),
    )
    centroid_relation_similarity = _histogram_similarity(
        first["nearest_distances"],
        second["nearest_distances"],
        bins=8,
    )
    structure_similarity = (
        0.18 * count_similarity
        + 0.18 * size_distribution_similarity
        + 0.14 * orientation_similarity
        + 0.20 * spatial_order_similarity
        + 0.14 * boundary_relation_similarity
        + 0.16 * centroid_relation_similarity
    )
    score = 100.0 * structure_similarity
    return GlobalCorrespondenceScore(
        score=score,
        pair_colors=pair_colors,
        structure_similarity=structure_similarity,
        count_similarity=count_similarity,
        size_distribution_similarity=size_distribution_similarity,
        orientation_similarity=orientation_similarity,
        spatial_order_similarity=spatial_order_similarity,
        boundary_relation_similarity=boundary_relation_similarity,
        centroid_relation_similarity=centroid_relation_similarity,
        component_counts={
            str(pair_colors[0]): len(first_components),
            str(pair_colors[1]): len(second_components),
        },
        family_signatures={
            str(pair_colors[0]): {
                "count": first["count"],
                "large_component_count": first["large_component_count"],
                "small_component_count": first["small_component_count"],
                "boundary_contact_ratio": round(float(first["boundary_contact_ratio"]), 4),
            },
            str(pair_colors[1]): {
                "count": second["count"],
                "large_component_count": second["large_component_count"],
                "small_component_count": second["small_component_count"],
                "boundary_contact_ratio": round(float(second["boundary_contact_ratio"]), 4),
            },
        },
    )


_GLOBAL_CORRESPONDENCE_CACHE: Dict[Tuple[Tuple[int, int], str], GlobalCorrespondenceScore] = {}


def _cached_global_correspondence_score(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
) -> GlobalCorrespondenceScore:
    key = (tuple(int(color) for color in pair_colors), _hash_grid(grid))
    cached = _GLOBAL_CORRESPONDENCE_CACHE.get(key)
    if cached is None:
        cached = global_correspondence_score(grid, pair_colors=key[0])
        _GLOBAL_CORRESPONDENCE_CACHE[key] = cached
    return cached


def infer_pair_colors_from_transitions(transitions: Dict[str, int]) -> Tuple[int, int]:
    """Infer yellow/purple-like colors from successful transition activity."""

    activity: Counter[int] = Counter()
    for key, count in transitions.items():
        if "->" not in key:
            continue
        left, right = key.split("->", 1)
        try:
            old = int(left)
            new = int(right)
        except ValueError:
            continue
        for value in (old, new):
            if value in {0, 9}:
                continue
            activity[value] += int(count)
    if len(activity) < 2:
        return (10, 11)
    colors = [color for color, _count in activity.most_common(2)]
    return (int(colors[0]), int(colors[1]))


def match_score(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    cursor_colors: Tuple[int, ...] = (4, 5),
    pair_threshold: float = 4.15,
) -> MatchScore:
    """Score how well the two inferred shape colors correspond."""

    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    cursor_components = _connected_components_for_colors(grid, cursor_colors)
    unused_second: Set[int] = set(range(len(second_components)))
    best_pairs: List[Dict[str, Any]] = []
    total_alignment = 0.0
    matched = 0

    for first in first_components:
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if not candidates:
            continue
        idx, best = max(candidates, key=lambda item: float(item[1]["score"]))
        unused_second.remove(idx)
        best_pairs.append(best)
        total_alignment += float(best["score"])
        if float(best["score"]) >= pair_threshold:
            matched += 1

    unmatched_first = max(0, len(first_components) - matched)
    unmatched_second = max(0, len(second_components) - matched)
    cursor_near_target = _count_cursor_near_unmatched(
        first_components + second_components,
        cursor_components,
        max_distance=8.0,
    )
    dotted_constraint_violations = _boundary_violations(first_components + second_components, shape)
    normalized_alignment = total_alignment / max(1.0, float(max(len(first_components), len(second_components))))
    score = (
        10.0 * matched
        + 2.2 * normalized_alignment
        + 1.4 * cursor_near_target
        - 4.0 * (unmatched_first + unmatched_second)
        - 1.5 * dotted_constraint_violations
        - 0.25 * abs(len(first_components) - len(second_components))
    )
    return MatchScore(
        score=float(score),
        pair_colors=pair_colors,
        matched_pairs=int(matched),
        unmatched_first=int(unmatched_first),
        unmatched_second=int(unmatched_second),
        shape_overlap_or_alignment=float(normalized_alignment),
        cursor_near_target=int(cursor_near_target),
        dotted_constraint_violations=int(dotted_constraint_violations),
        component_counts={
            str(pair_colors[0]): len(first_components),
            str(pair_colors[1]): len(second_components),
            "cursor": len(cursor_components),
        },
        best_pairs=best_pairs,
    )


def _count_cursor_near_unmatched(
    shape_components: Sequence[Component],
    cursor_components: Sequence[Component],
    *,
    max_distance: float,
) -> int:
    count = 0
    for cursor in cursor_components:
        if any(
            np.hypot(cursor.centroid_y - comp.centroid_y, cursor.centroid_x - comp.centroid_x) <= max_distance
            for comp in shape_components
        ):
            count += 1
    return count


def _cursor_near_small_unmatched_target(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    cursor_components: Sequence[Component],
    target_color: int = 11,
    max_size: int = 2,
    max_distance: float = 5.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if cursor is near a small unmatched target component.
    Returns (is_near, target_info_dict).
    """
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    
    # Get components by color
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    
    # Find matched pairs to determine which are unmatched
    unused_second: set = set(range(len(second_components)))
    matched_first_indices: set = set()
    matched_second_indices: set = set()
    
    for first_idx, first in enumerate(first_components):
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if candidates:
            idx, _ = max(candidates, key=lambda item: float(item[1]["score"]))
            unused_second.remove(idx)
            matched_first_indices.add(first_idx)
            matched_second_indices.add(idx)
    
    # Get cursor position
    if not cursor_components:
        return False, {"error": "no cursor"}
    cursor = cursor_components[0]
    cursor_y, cursor_x = cursor.centroid_y, cursor.centroid_x
    
    # Find unmatched components of target color
    unmatched_targets: List[Tuple[Component, float, bool]] = []  # (comp, distance, is_first_color)
    
    if target_color == pair_colors[0]:
        for i, comp in enumerate(first_components):
            if i not in matched_first_indices and comp.size <= max_size:
                dist = np.hypot(cursor_y - comp.centroid_y, cursor_x - comp.centroid_x)
                unmatched_targets.append((comp, dist, True))
    elif target_color == pair_colors[1]:
        for i, comp in enumerate(second_components):
            if i not in matched_second_indices and comp.size <= max_size:
                dist = np.hypot(cursor_y - comp.centroid_y, cursor_x - comp.centroid_x)
                unmatched_targets.append((comp, dist, False))
    else:
        # Check both colors
        for i, comp in enumerate(first_components):
            if i not in matched_first_indices and comp.size <= max_size:
                dist = np.hypot(cursor_y - comp.centroid_y, cursor_x - comp.centroid_x)
                unmatched_targets.append((comp, dist, True))
        for i, comp in enumerate(second_components):
            if i not in matched_second_indices and comp.size <= max_size:
                dist = np.hypot(cursor_y - comp.centroid_y, cursor_x - comp.centroid_x)
                unmatched_targets.append((comp, dist, False))
    
    if not unmatched_targets:
        return False, {"error": "no small unmatched targets"}
    
    # Find nearest
    nearest = min(unmatched_targets, key=lambda x: x[1])
    nearest_comp, nearest_dist, is_first = nearest
    
    is_near = nearest_dist <= max_distance
    
    target_info = {
        "is_near": bool(is_near),
        "color": int(nearest_comp.color),
        "size": int(nearest_comp.size),
        "distance": round(float(nearest_dist), 2),
        "centroid_y": float(nearest_comp.centroid_y),
        "centroid_x": float(nearest_comp.centroid_x),
        "is_unmatched": True,
        "is_first_color": bool(is_first),
    }
    
    return bool(is_near), target_info


def _boundary_violations(components: Sequence[Component], shape: Tuple[int, int]) -> int:
    height, width = shape
    return sum(
        1
        for comp in components
        if comp.min_y <= 0 or comp.min_x <= 0 or comp.max_y >= height - 1 or comp.max_x >= width - 1
    )


def _get_boundary_violation_components(
    components: Sequence[Component], shape: Tuple[int, int]
) -> List[Component]:
    """Return the actual components that violate boundary constraints."""
    height, width = shape
    return [
        comp
        for comp in components
        if comp.min_y <= 0 or comp.min_x <= 0 or comp.max_y >= height - 1 or comp.max_x >= width - 1
    ]


def _distance_to_nearest_violation(
    point_y: float, point_x: float,
    violation_components: Sequence[Component]
) -> float:
    """Compute minimum Euclidean distance from a point to any violation component centroid."""
    if not violation_components:
        return float('inf')
    return min(
        np.hypot(point_y - comp.centroid_y, point_x - comp.centroid_x)
        for comp in violation_components
    )


def _cursor_near_dotted_violation(
    cursor_components: Sequence[Component],
    violation_components: Sequence[Component],
    max_distance: float = 12.0,
) -> bool:
    """Check if any cursor is near a dotted boundary violation."""
    if not cursor_components or violation_components:
        return False
    for cursor in cursor_components:
        for viol in violation_components:
            dist = np.hypot(cursor.centroid_y - viol.centroid_y, cursor.centroid_x - viol.centroid_x)
            if dist <= max_distance:
                return True
    return False


def _get_boundary_contact_cells(
    component: Component,
    grid_shape: Tuple[int, int],
    arr: np.ndarray,
    pair_colors: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Extract cells of a component that touch the grid boundary (dotted line).
    Returns list of (y, x) coordinates that are at the edge of the component
    AND at the edge of the grid.
    """
    height, width = grid_shape
    contact_cells: List[Tuple[int, int]] = []

    # Determine which boundaries this component touches
    touches_top = component.min_y <= 0
    touches_bottom = component.max_y >= height - 1
    touches_left = component.min_x <= 0
    touches_right = component.max_x >= width - 1

    if not (touches_top or touches_bottom or touches_left or touches_right):
        return contact_cells

    # Get all cells of this component
    target_colors = set(pair_colors)
    seen = np.zeros(arr.shape, dtype=bool)
    stack = []

    # Find starting cell
    for y in range(max(0, component.min_y), min(height, component.max_y + 1)):
        for x in range(max(0, component.min_x), min(width, component.max_x + 1)):
            if int(arr[y, x]) in target_colors and not seen[y, x]:
                # This is a cell of the component - flood fill to get all
                stack = [(y, x)]
                seen[y, x] = True
                break
        if stack:
            break

    if not stack:
        return contact_cells

    component_cells: List[Tuple[int, int]] = []
    while stack:
        cy, cx = stack.pop()
        component_cells.append((cy, cx))
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if ny < 0 or nx < 0 or ny >= height or nx >= width:
                continue
            if seen[ny, nx] or int(arr[ny, nx]) not in target_colors:
                continue
            seen[ny, nx] = True
            stack.append((ny, nx))

    # Find cells that are at component boundary AND grid boundary
    for (cy, cx) in component_cells:
        is_at_grid_boundary = (
            cy <= 0 or cy >= height - 1 or cx <= 0 or cx >= width - 1
        )
        if is_at_grid_boundary:
            # Check if it's also at component boundary (has empty neighbor or grid edge)
            neighbors = 0
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if int(arr[ny, nx]) in target_colors:
                    neighbors += 1
            # If it has fewer than 4 neighbors, it's a boundary cell
            if neighbors < 4:
                contact_cells.append((cy, cx))

    return contact_cells


def generate_action6_targets_from_violation_boundaries(
    grid: Sequence[Sequence[int]],
    *,
    violation_components: Sequence[Component],
    pair_colors: Tuple[int, int],
    max_candidates: int = 8,
    cursor_components: Optional[Sequence[Component]] = None,
) -> List[Dict[str, int]]:
    """
    Generate ACTION6 target coordinates based on boundary violation contact points.

    Strategy:
    1. For each component violating a dotted boundary, find cells touching that boundary
    2. Generate candidates at those contact points (not centroids)
    3. Include points near cursor if cursor is close to violation (for reachability)
    4. Add small offsets around contact points for robustness
    """
    arr = np.array(grid, dtype=np.int32)
    height, width = arr.shape if arr.ndim == 2 else (64, 64)

    out: List[Dict[str, int]] = []
    seen: Set[Tuple[int, int]] = set()

    def add_normalized(y: int, x: int) -> None:
        """Add coordinate normalized to 0-63 range."""
        if height <= 1 or width <= 1:
            norm_y, norm_x = 32, 32
        else:
            norm_y = int(round(63.0 * y / (height - 1)))
            norm_x = int(round(63.0 * x / (width - 1)))
        norm_y = max(0, min(63, norm_y))
        norm_x = max(0, min(63, norm_x))
        key = (norm_y, norm_x)
        if key in seen or len(out) >= max_candidates:
            return
        seen.add(key)
        out.append({"x": norm_x, "y": norm_y})

    # Get boundary contact cells for each violation
    boundary_points: List[Tuple[int, int]] = []
    for comp in violation_components:
        contacts = _get_boundary_contact_cells(comp, (height, width), arr, pair_colors)
        boundary_points.extend(contacts)

    # Add the actual contact cells
    for (y, x) in boundary_points:
        add_normalized(y, x)
        if len(out) >= max_candidates:
            return out

    # Add offset points around contact cells (1-2 cells inward)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
    for (y, x) in boundary_points:
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if int(arr[ny, nx]) in set(pair_colors):  # Still inside the shape
                    add_normalized(ny, nx)
                    if len(out) >= max_candidates:
                        return out

    # If cursor is near violation, add cursor-to-boundary direction points
    if cursor_components and violation_components:
        for cursor in cursor_components:
            for viol in violation_components:
                dist = np.hypot(cursor.centroid_y - viol.centroid_y, cursor.centroid_x - viol.centroid_x)
                if dist <= 15.0:  # Cursor is close to violating component
                    # Add point on line from cursor to violation boundary
                    for contact in boundary_points[:2]:  # First 2 contact points
                        cy, cx = contact
                        # Midpoint between cursor and boundary
                        mid_y = int((cursor.centroid_y + cy) / 2)
                        mid_x = int((cursor.centroid_x + cx) / 2)
                        add_normalized(mid_y, mid_x)
                        if len(out) >= max_candidates:
                            return out

    # Fallback: if no boundary points found, use component edges
    if not out and violation_components:
        for comp in violation_components:
            # Use min/max edges as fallback (these are definitely at boundary)
            if comp.min_y <= 0:
                add_normalized(0, int(comp.centroid_x))
            if comp.max_y >= height - 1:
                add_normalized(height - 1, int(comp.centroid_x))
            if comp.min_x <= 0:
                add_normalized(int(comp.centroid_y), 0)
            if comp.max_x >= width - 1:
                add_normalized(int(comp.centroid_y), width - 1)
            if len(out) >= max_candidates:
                return out

    return out


def _load_rule_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _default_action_ontology_path(game_id: str) -> Path:
    return DEFAULT_ACTION_ONTOLOGY_DIR / f"{game_id}.action_ontology.json"


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / max(1, len(values)))


def _rate(part: int, total: int) -> float:
    return float(part / total) if total > 0 else 0.0


def _load_action_ontology_model(
    path: Optional[Path],
    *,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    """Load a compact empirical action model for search-time scoring."""

    if path is None or not path.exists():
        return {
            "available": False,
            "reason": "missing_action_ontology_report",
            "path": str(path) if path else None,
            "action_priors": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = payload.get("action_ontology") or {}
    probes = [
        probe
        for probe in (payload.get("probes") or [])
        if isinstance(probe, dict) and "error" not in probe
    ]
    by_action: Dict[str, List[Dict[str, Any]]] = {}
    for probe in probes:
        by_action.setdefault(str(probe.get("action")), []).append(probe)

    pair_set = set(int(color) for color in pair_colors)
    action_priors: Dict[str, Dict[str, Any]] = {}
    human_auto_changed: List[float] = []
    human_auto_actions: Counter[str] = Counter()
    for action, summary in summaries.items():
        valid = by_action.get(str(action), [])
        count = int(summary.get("count") or len(valid) or 0)
        operator_counts = {
            str(key): int(value)
            for key, value in (summary.get("operator_type_counts") or {}).items()
        }
        if not operator_counts:
            dominant = str(summary.get("dominant_operator_type") or "unknown")
            operator_counts = {dominant: count} if count else {}
        human = [
            probe
            for probe in valid
            if str((probe.get("sample") or {}).get("source")) == "human_pre_auto_levelup"
        ]
        agent = [
            probe
            for probe in valid
            if str((probe.get("sample") or {}).get("source")) == "agent_guided_search"
        ]
        auto = [probe for probe in valid if int(probe.get("level_delta") or 0) > 0]
        human_auto = [probe for probe in human if int(probe.get("level_delta") or 0) > 0]
        agent_auto = [probe for probe in agent if int(probe.get("level_delta") or 0) > 0]
        pair_affect_count = sum(
            1
            for probe in valid
            if pair_set.intersection(int(color) for color in (probe.get("affected_colors") or []))
        )
        human_auto_changed.extend(float(probe.get("changed_cells") or 0) for probe in human_auto)
        human_auto_actions.update(str(action) for _probe in human_auto)

        action_priors[str(action)] = {
            "action": str(action),
            "count": int(count),
            "dominant_operator_type": str(summary.get("dominant_operator_type") or "unknown"),
            "context_dependent": bool(summary.get("context_dependent")),
            "operator_type_counts": operator_counts,
            "operator_type_rates": {
                key: round(_rate(int(value), count), 6)
                for key, value in operator_counts.items()
            },
            "direction_candidates": dict(summary.get("direction_candidates") or {}),
            "no_op_rate": float(summary.get("no_op_rate") or 0.0),
            "auto_levelup_rate": round(_rate(len(auto), len(valid)), 6),
            "human_auto_levelup_rate": round(_rate(len(human_auto), len(human)), 6),
            "agent_auto_levelup_rate": round(_rate(len(agent_auto), len(agent)), 6),
            "pair_affect_rate": round(_rate(pair_affect_count, len(valid)), 6),
            "avg_changed_cells": float(summary.get("avg_changed_cells") or _mean_or_zero([
                float(probe.get("changed_cells") or 0) for probe in valid
            ])),
            "human_auto_changed_cells_avg": _mean_or_zero([
                float(probe.get("changed_cells") or 0) for probe in human_auto
            ]),
        }

    return {
        "available": True,
        "path": str(path),
        "probe_count": int(payload.get("probe_count") or len(probes)),
        "pair_colors": [int(pair_colors[0]), int(pair_colors[1])],
        "human_auto_changed_cells_avg": _mean_or_zero(human_auto_changed),
        "human_auto_actions": dict(human_auto_actions),
        "action_priors": action_priors,
    }


def _boundary_side_counts_for_components(
    components: Sequence[Component],
    *,
    shape: Tuple[int, int],
) -> Dict[str, Dict[str, int]]:
    height, width = shape
    counts: Dict[str, Dict[str, int]] = {}
    for component in components:
        color = str(int(component.color))
        item = counts.setdefault(color, {"top": 0, "bottom": 0, "left": 0, "right": 0})
        if component.min_y <= 0:
            item["top"] += 1
        if component.max_y >= height - 1:
            item["bottom"] += 1
        if component.min_x <= 0:
            item["left"] += 1
        if component.max_x >= width - 1:
            item["right"] += 1
    return counts


def _relative_position_features(
    first_components: Sequence[Component],
    second_components: Sequence[Component],
) -> Dict[str, float]:
    nearest: List[float] = []
    y_offsets: List[float] = []
    x_offsets: List[float] = []
    for first in first_components:
        candidates = [
            (
                float(np.hypot(first.centroid_y - second.centroid_y, first.centroid_x - second.centroid_x)),
                second,
            )
            for second in second_components
        ]
        if not candidates:
            continue
        distance, second = min(candidates, key=lambda item: item[0])
        nearest.append(distance)
        y_offsets.append(float(second.centroid_y - first.centroid_y))
        x_offsets.append(float(second.centroid_x - first.centroid_x))
    return {
        "nearest_distance_mean": _mean_or_zero(nearest),
        "nearest_distance_min": min(nearest, default=0.0),
        "nearest_distance_max": max(nearest, default=0.0),
        "mean_y_offset": _mean_or_zero(y_offsets),
        "mean_x_offset": _mean_or_zero(x_offsets),
    }


def _auto_levelup_state_features(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    match: Optional[MatchScore] = None,
    global_score: Optional[GlobalCorrespondenceScore] = None,
) -> Dict[str, float]:
    match = match or match_score(grid, pair_colors=pair_colors)
    global_score = global_score or global_correspondence_score(grid, pair_colors=pair_colors)
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    all_components = first_components + second_components
    boundary = _boundary_side_counts_for_components(all_components, shape=shape)
    first_key = str(pair_colors[0])
    second_key = str(pair_colors[1])
    first_boundary_total = sum(int(value) for value in (boundary.get(first_key) or {}).values())
    second_boundary_total = sum(int(value) for value in (boundary.get(second_key) or {}).values())
    relative = _relative_position_features(first_components, second_components)
    offenders = _offending_component_diagnostics(grid, pair_colors=pair_colors, limit=8)
    first_size_sum = float(sum(component.size for component in first_components))
    second_size_sum = float(sum(component.size for component in second_components))
    first_largest = float(max((component.size for component in first_components), default=0))
    second_largest = float(max((component.size for component in second_components), default=0))
    largest_offender = offenders.get("largest_offender") or {}
    return {
        "global_score": float(global_score.score),
        "global_structure": float(global_score.structure_similarity),
        "global_count_similarity": float(global_score.count_similarity),
        "global_size_similarity": float(global_score.size_distribution_similarity),
        "global_orientation_similarity": float(global_score.orientation_similarity),
        "global_spatial_order_similarity": float(global_score.spatial_order_similarity),
        "global_boundary_similarity": float(global_score.boundary_relation_similarity),
        "global_centroid_similarity": float(global_score.centroid_relation_similarity),
        "matched_pairs": float(match.matched_pairs),
        "unmatched_total": float(_unmatched_total(match)),
        "dotted_violations": float(match.dotted_constraint_violations),
        "first_count": float(len(first_components)),
        "second_count": float(len(second_components)),
        "count_balance_abs": abs(float(len(first_components) - len(second_components))),
        "first_size_sum": first_size_sum,
        "second_size_sum": second_size_sum,
        "size_balance_abs": abs(first_size_sum - second_size_sum),
        "first_largest_size": first_largest,
        "second_largest_size": second_largest,
        "largest_size_balance_abs": abs(first_largest - second_largest),
        "first_boundary_total": float(first_boundary_total),
        "second_boundary_total": float(second_boundary_total),
        "boundary_contact_total": float(first_boundary_total + second_boundary_total),
        "nearest_distance_mean": float(relative["nearest_distance_mean"]),
        "nearest_distance_min": float(relative["nearest_distance_min"]),
        "nearest_distance_max": float(relative["nearest_distance_max"]),
        "mean_y_offset": float(relative["mean_y_offset"]),
        "mean_x_offset": float(relative["mean_x_offset"]),
        "offending_count": float(offenders.get("offending_count", 0)),
        "violation_count": float(offenders.get("violation_count", 0)),
        "low_alignment_pair_count": float(offenders.get("low_alignment_pair_count", 0)),
        "largest_offender_size": float(largest_offender.get("size") or 0),
    }


def _build_auto_levelup_state_classifier_from_references(
    references: Sequence[Dict[str, Any]],
    *,
    pair_colors: Tuple[int, int],
    reference_scope: str = "all",
    target_level: Optional[int] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for ref in references:
        grid = ref.get("grid")
        if grid is None:
            continue
        features = _auto_levelup_state_features(grid, pair_colors=pair_colors)
        rows.append(
            {
                "label": str(ref.get("label") or f"reference_{len(rows)}"),
                "level": int(ref.get("level") or 0),
                "action": str(ref.get("action") or "UNKNOWN"),
                "hash": _hash_grid(grid),
                "features": features,
            }
        )
    if not rows:
        return {
            "available": False,
            "reason": "no_auto_levelup_references",
            "pair_colors": [int(pair_colors[0]), int(pair_colors[1])],
            "reference_scope": reference_scope,
            "target_level": int(target_level) if target_level is not None else None,
        }

    feature_names = sorted(rows[0]["features"])
    scales: Dict[str, float] = {}
    means: Dict[str, float] = {}
    for name in feature_names:
        values = [float(row["features"].get(name, 0.0)) for row in rows]
        mean_value = _mean_or_zero(values)
        variance = _mean_or_zero([(value - mean_value) ** 2 for value in values])
        spread = max(values) - min(values) if values else 0.0
        scales[name] = max(1.0, float(variance ** 0.5), float(spread) * 0.5)
        means[name] = mean_value

    return {
        "available": True,
        "pair_colors": [int(pair_colors[0]), int(pair_colors[1])],
        "reference_scope": reference_scope,
        "target_level": int(target_level) if target_level is not None else None,
        "reference_count": len(rows),
        "feature_names": feature_names,
        "feature_scale": {name: round(float(scales[name]), 6) for name in feature_names},
        "feature_mean": {name: round(float(means[name]), 6) for name in feature_names},
        "action_counts": dict(Counter(row["action"] for row in rows)),
        "references": rows,
    }


def _filter_auto_levelup_references(
    references: Sequence[Dict[str, Any]],
    *,
    target_level: int,
    reference_scope: str,
) -> List[Dict[str, Any]]:
    if reference_scope == "all":
        return [dict(item) for item in references]
    if reference_scope == "same-level":
        return [
            dict(item)
            for item in references
            if int(item.get("level") or -9999) == int(target_level)
        ]
    if reference_scope == "nearby-level":
        return [
            dict(item)
            for item in references
            if abs(int(item.get("level") or -9999) - int(target_level)) <= 1
        ]
    raise ValueError(f"unknown auto reference scope: {reference_scope}")


def _fit_auto_levelup_state_classifier(
    *,
    selection_steps: Sequence[Any],
    pair_colors: Tuple[int, int],
    max_level: int,
    target_level: int,
    reference_scope: str = "all",
) -> Dict[str, Any]:
    bundle = EpisodeBundle(episode=None, steps=list(selection_steps))
    segments = build_level_segments(bundle, max_level=max(max_level, target_level))
    references = [
        {
            "label": f"human_pre_auto_levelup_{segment.level_number}",
            "level": int(segment.level_number),
            "action": segment.level_up_action,
            "grid": segment.level_up_before_grid,
        }
        for segment in segments
    ]
    selected = _filter_auto_levelup_references(
        references,
        target_level=target_level,
        reference_scope=reference_scope,
    )
    return _build_auto_levelup_state_classifier_from_references(
        selected,
        pair_colors=pair_colors,
        reference_scope=reference_scope,
        target_level=target_level,
    )


def _score_auto_levelup_state(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    match: Optional[MatchScore] = None,
    global_score: Optional[GlobalCorrespondenceScore] = None,
    classifier: Optional[Dict[str, Any]] = None,
    nearest: int = 3,
) -> Dict[str, Any]:
    model = classifier or {}
    if not model.get("available"):
        return {
            "available": False,
            "score": 0.0,
            "reason": model.get("reason", "auto_levelup_state_classifier_unavailable"),
        }
    features = _auto_levelup_state_features(
        grid,
        pair_colors=pair_colors,
        match=match,
        global_score=global_score,
    )
    scales = model.get("feature_scale") or {}
    feature_names = list(model.get("feature_names") or sorted(features))
    references = list(model.get("references") or [])
    scored_refs: List[Dict[str, Any]] = []
    for ref in references:
        ref_features = ref.get("features") or {}
        weighted = 0.0
        total_weight = 0.0
        contributions: List[Tuple[str, float]] = []
        for name in feature_names:
            scale = max(1.0, float(scales.get(name, 1.0)))
            distance = abs(float(features.get(name, 0.0)) - float(ref_features.get(name, 0.0))) / scale
            weight = 1.0
            if name in {
                "unmatched_total",
                "dotted_violations",
                "offending_count",
                "violation_count",
                "boundary_contact_total",
                "count_balance_abs",
                "size_balance_abs",
            }:
                weight = 1.35
            if name.startswith("global_"):
                weight = 0.85
            weighted += weight * distance
            total_weight += weight
            contributions.append((name, distance))
        mean_distance = weighted / max(1.0, total_weight)
        similarity = 100.0 / (1.0 + mean_distance)
        scored_refs.append(
            {
                "label": ref.get("label"),
                "level": int(ref.get("level") or 0),
                "action": ref.get("action"),
                "hash": ref.get("hash"),
                "similarity": round(float(similarity), 4),
                "distance": round(float(mean_distance), 4),
                "top_differences": {
                    name: round(float(value), 4)
                    for name, value in sorted(contributions, key=lambda item: item[1], reverse=True)[:6]
                },
            }
        )
    scored_refs.sort(key=lambda item: float(item["similarity"]), reverse=True)
    best_score = float(scored_refs[0]["similarity"]) if scored_refs else 0.0
    return {
        "available": True,
        "score": round(float(best_score), 4),
        "nearest_references": scored_refs[: max(1, int(nearest))],
        "features": {name: round(float(features.get(name, 0.0)), 4) for name in feature_names},
    }


def _auto_levelup_feature_gap_report(
    node: Optional[GuidedNode],
    *,
    classifier: Optional[Dict[str, Any]],
    features: Sequence[str] = AUTO_LEVELUP_GAP_FEATURES,
    top_n: int = 12,
) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    state = node.auto_levelup_state or {}
    if not state.get("available"):
        return {
            "available": False,
            "reason": state.get("reason", "auto_levelup_state_unavailable"),
        }
    model = classifier or {}
    references = list(model.get("references") or [])
    nearest = (state.get("nearest_references") or [{}])[0]
    nearest_hash = nearest.get("hash")
    nearest_label = nearest.get("label")
    ref = next(
        (
            item
            for item in references
            if item.get("hash") == nearest_hash
            or (nearest_label is not None and item.get("label") == nearest_label)
        ),
        None,
    )
    if not ref:
        return {
            "available": False,
            "reason": "nearest_reference_not_found",
            "nearest_reference": dict(nearest),
        }

    node_features = state.get("features") or {}
    ref_features = ref.get("features") or {}
    scales = model.get("feature_scale") or {}
    comparison: Dict[str, Dict[str, Any]] = {}
    ranked: List[Tuple[str, float]] = []
    for name in sorted(set(features) | set(node_features) | set(ref_features)):
        agent_value = float(node_features.get(name, 0.0))
        reference_value = float(ref_features.get(name, 0.0))
        delta = agent_value - reference_value
        scale = max(1.0, float(scales.get(name, 1.0)))
        normalized_delta = delta / scale
        comparison[name] = {
            "agent": round(float(agent_value), 4),
            "reference": round(float(reference_value), 4),
            "delta_agent_minus_reference": round(float(delta), 4),
            "abs_delta": round(float(abs(delta)), 4),
            "normalized_delta": round(float(normalized_delta), 4),
            "abs_normalized_delta": round(float(abs(normalized_delta)), 4),
        }
        ranked.append((name, abs(normalized_delta)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    tracked = {
        name: comparison[name]
        for name in features
        if name in comparison
    }
    return {
        "available": True,
        "node_actions": list(node.actions),
        "node_search_score": round(float(node.search_score), 4),
        "node_match_score": round(float(node.match.score), 4),
        "node_global_score": round(float(_node_global_score(node)), 4),
        "node_auto_score": round(float(state.get("score", 0.0)), 4),
        "nearest_reference": {
            "label": ref.get("label"),
            "level": int(ref.get("level") or 0),
            "action": ref.get("action"),
            "hash": ref.get("hash"),
            "similarity": nearest.get("similarity"),
            "distance": nearest.get("distance"),
        },
        "tracked_features": tracked,
        "largest_gaps": {
            name: comparison[name]
            for name, _score in ranked[: max(1, int(top_n))]
        },
    }


def _fragmentation_guidance_report(
    *,
    parent_state: Optional[Dict[str, Any]],
    child_state: Optional[Dict[str, Any]],
    classifier: Optional[Dict[str, Any]],
    features: Sequence[str] = LEVEL7_FRAGMENTATION_FEATURES,
    largest_second_pressure: bool = False,
) -> Dict[str, Any]:
    model = classifier or {}
    if not model.get("available"):
        return {
            "available": False,
            "score": 0.0,
            "reason": model.get("reason", "auto_levelup_state_classifier_unavailable"),
        }
    references = list(model.get("references") or [])
    if not references:
        return {
            "available": False,
            "score": 0.0,
            "reason": "missing_fragmentation_reference",
        }

    target_features = references[0].get("features") or {}
    if len(references) > 1:
        # If the scope is broad, use the reference nearest to the child state.
        nearest = ((child_state or {}).get("nearest_references") or [{}])[0]
        nearest_hash = nearest.get("hash")
        nearest_label = nearest.get("label")
        target = next(
            (
                item
                for item in references
                if item.get("hash") == nearest_hash
                or (nearest_label is not None and item.get("label") == nearest_label)
            ),
            None,
        )
        if target:
            target_features = target.get("features") or target_features

    parent_features = (parent_state or {}).get("features") or {}
    child_features = (child_state or {}).get("features") or {}
    scales = model.get("feature_scale") or {}
    per_feature: Dict[str, Dict[str, Any]] = {}
    improvement_score = 0.0
    proximity_score = 0.0
    total_weight = 0.0
    for name in features:
        target_value = float(target_features.get(name, 0.0))
        parent_value = float(parent_features.get(name, 0.0))
        child_value = float(child_features.get(name, 0.0))
        scale = max(1.0, float(scales.get(name, 1.0)))
        parent_distance = abs(parent_value - target_value) / scale
        child_distance = abs(child_value - target_value) / scale
        improvement = parent_distance - child_distance
        bounded_improvement = max(-1.0, min(1.0, improvement))
        proximity = 1.0 / (1.0 + child_distance)
        weight = 1.0
        if name in {"second_count", "second_largest_size", "second_size_sum", "largest_offender_size"}:
            weight = 1.8
        elif name in {"unmatched_total", "first_largest_size"}:
            weight = 1.25
        if largest_second_pressure and name in {"second_largest_size", "largest_offender_size"}:
            weight *= 2.25
        improvement_score += weight * bounded_improvement
        proximity_score += weight * proximity
        total_weight += weight
        per_feature[name] = {
            "agent": round(float(child_value), 4),
            "parent": round(float(parent_value), 4),
            "target": round(float(target_value), 4),
            "delta_agent_minus_target": round(float(child_value - target_value), 4),
            "parent_distance": round(float(parent_distance), 4),
            "child_distance": round(float(child_distance), 4),
            "improvement_toward_target": round(float(improvement), 4),
            "bounded_improvement_toward_target": round(float(bounded_improvement), 4),
        }

    normalized_improvement = improvement_score / max(1.0, total_weight)
    normalized_proximity = proximity_score / max(1.0, total_weight)
    score = 180.0 * normalized_proximity + 120.0 * normalized_improvement
    largest_gaps = {
        name: per_feature[name]
        for name in sorted(
            per_feature,
            key=lambda item: abs(float(per_feature[item]["delta_agent_minus_target"])),
            reverse=True,
        )[:6]
    }
    return {
        "available": True,
        "score": round(float(score), 4),
        "proximity_score": round(float(180.0 * normalized_proximity), 4),
        "improvement_score": round(float(120.0 * normalized_improvement), 4),
        "target_reference": {
            "label": references[0].get("label"),
            "level": int(references[0].get("level") or 0),
            "action": references[0].get("action"),
        },
        "largest_second_pressure": bool(largest_second_pressure),
        "features": per_feature,
        "largest_gaps": largest_gaps,
    }


def _fragmentation_guardrail_report(
    *,
    child_state: Optional[Dict[str, Any]],
    child_match: MatchScore,
    child_global: GlobalCorrespondenceScore,
    global_floor: float = 45.0,
) -> Dict[str, Any]:
    features = (child_state or {}).get("features") or {}
    if not features:
        return {
            "available": False,
            "score": 0.0,
            "reason": "missing_auto_levelup_features",
        }

    dotted = float(child_match.dotted_constraint_violations)
    global_score = float(child_global.score)
    first_size_sum = float(features.get("first_size_sum", 0.0))
    target_first_size_sum = 280.0
    nearest_distance = float(features.get("nearest_distance_mean", 0.0))
    target_nearest_distance = 8.0
    first_count = float(features.get("first_count", 0.0))
    target_first_count = 5.0

    dotted_excess = max(0.0, dotted - 2.0)
    global_deficit = max(0.0, float(global_floor) - global_score)
    first_size_drift = abs(first_size_sum - target_first_size_sum)
    nearest_distance_drift = abs(nearest_distance - target_nearest_distance)
    first_count_drift = abs(first_count - target_first_count)

    penalty = (
        120.0 * dotted_excess
        + 8.0 * global_deficit
        + 0.55 * first_size_drift
        + 10.0 * nearest_distance_drift
        + 18.0 * first_count_drift
    )
    passes = bool(
        dotted_excess <= 0.0
        and global_deficit <= 0.0
        and first_size_drift <= 48.0
        and nearest_distance_drift <= 3.0
    )
    return {
        "available": True,
        "score": round(float(-penalty), 4),
        "penalty": round(float(penalty), 4),
        "passes": passes,
        "global_floor": round(float(global_floor), 4),
        "dotted_excess": round(float(dotted_excess), 4),
        "global_deficit": round(float(global_deficit), 4),
        "first_size_sum": round(float(first_size_sum), 4),
        "first_size_drift": round(float(first_size_drift), 4),
        "nearest_distance_mean": round(float(nearest_distance), 4),
        "nearest_distance_drift": round(float(nearest_distance_drift), 4),
        "first_count": round(float(first_count), 4),
        "first_count_drift": round(float(first_count_drift), 4),
    }


def _parse_pair_colors(value: Optional[Union[str, Sequence[str]]]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = []
        for item in value:
            parts.extend(part.strip() for part in str(item).split(",") if part.strip())
    if len(parts) != 2:
        raise ValueError("--pair-colors must look like '10,11' or '10 11'")
    return (int(parts[0]), int(parts[1]))


def calibrate_rule_model(
    *,
    selection_steps: Sequence[Any],
    episode_id: str,
    rule_report: Dict[str, Any],
    max_level: int,
    pair_colors_override: Optional[Tuple[int, int]] = None,
) -> RuleModel:
    invariants = rule_report.get("cross_level_invariants") or {}
    transitions = invariants.get("top_value_transitions") or {}
    pair_colors = pair_colors_override or infer_pair_colors_from_transitions(transitions)
    bundle = EpisodeBundle(episode=None, steps=list(selection_steps))
    segments = build_level_segments(bundle, max_level=max_level)
    candidate_pairs = _pair_color_candidate_diagnostics(
        segments=segments,
        transitions=transitions,
        selected_pair=pair_colors,
    )
    ready_matches = [
        match_score(segment.level_up_before_grid, pair_colors=pair_colors)
        for segment in segments
    ]
    ready_scores = [item.score for item in ready_matches]
    if ready_scores:
        scores = np.array(ready_scores, dtype=np.float32)
        # Use a very high success-state percentile. A low percentile made complex
        # late levels look "ready" too early and let ACTION2 fire as a loop.
        threshold = float(np.quantile(scores, 0.90))
    else:
        threshold = 0.0
    expected_matched = int(round(float(np.median([item.matched_pairs for item in ready_matches])))) if ready_matches else 0
    expected_unmatched = int(
        round(
            float(
                np.median([
                    item.unmatched_first + item.unmatched_second
                    for item in ready_matches
                ])
            )
        )
    ) if ready_matches else 0
    return RuleModel(
        pair_colors=pair_colors,
        ready_threshold=float(threshold),
        ready_scores=ready_scores,
        ready_level_up_actions=dict(Counter(segment.level_up_action for segment in segments)),
        expected_matched_pairs=expected_matched,
        expected_unmatched_total=expected_unmatched,
        pair_color_candidates=candidate_pairs,
    )


def _pair_color_candidate_diagnostics(
    *,
    segments: Sequence[Any],
    transitions: Dict[str, int],
    selected_pair: Tuple[int, int],
) -> List[Dict[str, Any]]:
    colors: Set[int] = set(selected_pair)
    for key in transitions:
        if "->" not in key:
            continue
        left, right = key.split("->", 1)
        for part in (left, right):
            try:
                value = int(part)
            except ValueError:
                continue
            if value not in {0, 9}:
                colors.add(value)
    if len(colors) < 2:
        colors.update({4, 5, 10, 11})

    out: List[Dict[str, Any]] = []
    sorted_colors = sorted(colors)
    for idx, first in enumerate(sorted_colors):
        for second in sorted_colors[idx + 1 :]:
            ready = [
                match_score(segment.level_up_before_grid, pair_colors=(first, second))
                for segment in segments
            ]
            scores = [item.score for item in ready]
            if scores:
                arr = np.array(scores, dtype=np.float32)
                q90 = float(np.quantile(arr, 0.90))
                avg = float(np.mean(arr))
                best = float(np.max(arr))
            else:
                q90 = avg = best = 0.0
            out.append(
                {
                    "pair_colors": [int(first), int(second)],
                    "selected": (int(first), int(second)) == tuple(selected_pair),
                    "avg_ready_score": round(avg, 4),
                    "best_ready_score": round(best, 4),
                    "q90_ready_score": round(q90, 4),
                    "median_matched_pairs": round(
                        float(np.median([item.matched_pairs for item in ready])) if ready else 0.0,
                        4,
                    ),
                }
            )
    out.sort(
        key=lambda item: (
            bool(item["selected"]),
            float(item["q90_ready_score"]),
            float(item["avg_ready_score"]),
        ),
        reverse=True,
    )
    return out[:12]


def _action_variants(
    action: str,
    *,
    include_action6: bool,
    action6_targets: Sequence[Dict[str, int]],
) -> List[Optional[Dict[str, int]]]:
    if action != "ACTION6":
        return [None]
    if not include_action6:
        return []
    return [dict(target) for target in action6_targets] or [{"x": 31, "y": 31}]


def _allowed_actions(raw: Any, *, include_action6: bool) -> List[str]:
    available = _available_names_from_raw(raw) or ACTION_NAMES
    if include_action6 and "ACTION6" not in available:
        available = list(available) + ["ACTION6"]
    allowed = []
    for action in available:
        if action == "RESET":
            continue
        if action == "ACTION6" and not include_action6:
            continue
        allowed.append(action)
    preferred_order = {name: idx for idx, name in enumerate(["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7", "ACTION6"])}
    return sorted(allowed, key=lambda item: preferred_order.get(item, 99))


def _is_action3_contradiction(
    *,
    action: str,
    parent_grid: Sequence[Sequence[int]],
    child_grid: Sequence[Sequence[int]],
    child_state: str,
) -> bool:
    if action != "ACTION3":
        return False
    bbox = _diff_bbox(parent_grid, child_grid)
    if child_state == "GAME_OVER" and bbox and int(bbox.get("changed_cells", 0)) >= 20:
        return True
    return False


def _danger_prefix_len(actions: Sequence[str], danger_actions: Sequence[str]) -> int:
    count = 0
    for got, danger in zip(actions, danger_actions):
        if got != danger:
            break
        count += 1
    return count


def _consecutive_suffix(actions: Sequence[str], action: str) -> int:
    count = 0
    for item in reversed(actions):
        if item != action:
            break
        count += 1
    return count


def _grid_changed_cells(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> int:
    left = np.array(before)
    right = np.array(after)
    if left.shape != right.shape:
        return int(left.size + right.size)
    return int((left != right).sum())


def _grid_affected_values(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> List[int]:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return []
    mask = left != right
    values = set(int(value) for value in left[mask].ravel())
    values.update(int(value) for value in right[mask].ravel())
    return sorted(values)


def _unmatched_total(match: MatchScore) -> int:
    return int(match.unmatched_first + match.unmatched_second)


def _score_action_affordance(
    *,
    action: str,
    parent: GuidedNode,
    parent_grid: Sequence[Sequence[int]],
    child_grid: Sequence[Sequence[int]],
    child_match: MatchScore,
    child_global: GlobalCorrespondenceScore,
    parent_global: GlobalCorrespondenceScore,
    reached_next: bool,
    died: bool,
    action_ontology_model: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Empirical action prior + observed one-step effect.

    This intentionally avoids assigning hard-coded meanings such as
    ACTION1=left. The prior comes from discovered action probes; the effect
    terms come from the actual transition just sampled by the search.
    """

    model = action_ontology_model or {}
    if not model.get("available"):
        return {
            "available": False,
            "score": 0.0,
            "reason": model.get("reason", "action_ontology_unavailable"),
            "action": action,
        }

    prior = (model.get("action_priors") or {}).get(action) or {}
    changed_cells = _grid_changed_cells(parent_grid, child_grid)
    affected_values = _grid_affected_values(parent_grid, child_grid)
    pair_set = set(int(color) for color in child_match.pair_colors)
    pair_affected = bool(pair_set.intersection(affected_values))
    parent_unmatched = _unmatched_total(parent.match)
    child_unmatched = _unmatched_total(child_match)
    unmatched_parent_delta = int(parent_unmatched - child_unmatched)
    dotted_parent_delta = int(
        parent.match.dotted_constraint_violations - child_match.dotted_constraint_violations
    )
    global_delta_parent = float(child_global.score) - float(parent_global.score)

    operator_rates = prior.get("operator_type_rates") or {}
    auto_rate = float(prior.get("auto_levelup_rate") or 0.0)
    human_auto_rate = float(prior.get("human_auto_levelup_rate") or 0.0)
    pair_affect_rate = float(prior.get("pair_affect_rate") or 0.0)
    no_op_rate = float(prior.get("no_op_rate") or 0.0)
    selector_rate = float(operator_rates.get("selector_or_control_like", 0.0))
    translation_rate = float(operator_rates.get("translation_like", 0.0))
    transform_rate = float(operator_rates.get("transform_like", 0.0))
    contextual_bonus = 24.0 if prior.get("context_dependent") else 0.0

    expected_auto_change = max(1.0, float(model.get("human_auto_changed_cells_avg") or 0.0))
    auto_scale = min(1.35, float(changed_cells) / expected_auto_change)
    no_op = changed_cells <= 0
    actual_effect_score = (
        280.0 * float(reached_next)
        + 95.0 * float(max(0, unmatched_parent_delta))
        + 85.0 * float(max(0, dotted_parent_delta))
        + 42.0 * float(max(0.0, global_delta_parent))
        + 36.0 * float(pair_affected)
        + 18.0 * min(10.0, float(changed_cells) / 64.0)
        - 120.0 * float(max(0, -unmatched_parent_delta))
        - 110.0 * float(max(0, -dotted_parent_delta))
        - 95.0 * float(no_op and no_op_rate >= 0.65)
        - 260.0 * float(died)
    )
    predicted_effect_score = (
        420.0 * human_auto_rate
        + 210.0 * auto_rate
        + 75.0 * selector_rate
        + 70.0 * translation_rate
        + 55.0 * transform_rate
        + 45.0 * pair_affect_rate
        + contextual_bonus
        - 120.0 * no_op_rate
    )
    auto_levelup_affordance = (
        900.0 * float(reached_next)
        + (260.0 * human_auto_rate + 120.0 * auto_rate) * (0.25 + 0.75 * auto_scale)
    )
    score = actual_effect_score + predicted_effect_score + auto_levelup_affordance
    return {
        "available": True,
        "action": action,
        "score": round(float(score), 4),
        "actual_effect_score": round(float(actual_effect_score), 4),
        "predicted_action_effect": round(float(predicted_effect_score), 4),
        "auto_levelup_affordance": round(float(auto_levelup_affordance), 4),
        "changed_cells": int(changed_cells),
        "affected_values": affected_values[:12],
        "pair_colors_affected": bool(pair_affected),
        "unmatched_parent_delta": int(unmatched_parent_delta),
        "dotted_parent_delta": int(dotted_parent_delta),
        "global_delta_parent": round(float(global_delta_parent), 4),
        "empirical_auto_levelup_rate": round(float(auto_rate), 6),
        "human_auto_levelup_rate": round(float(human_auto_rate), 6),
        "no_op_rate": round(float(no_op_rate), 6),
        "dominant_operator_type": prior.get("dominant_operator_type"),
        "direction_candidates": dict(prior.get("direction_candidates") or {}),
    }


def _component_key(component: Component) -> Tuple[int, int, int, int, int, int]:
    return (
        int(component.color),
        int(component.min_y),
        int(component.min_x),
        int(component.max_y),
        int(component.max_x),
        int(component.size),
    )


def _component_bbox(component: Component) -> Dict[str, int]:
    return {
        "min_y": int(component.min_y),
        "min_x": int(component.min_x),
        "max_y": int(component.max_y),
        "max_x": int(component.max_x),
    }


def _bbox_intersects(
    left: Optional[Dict[str, Any]],
    right: Optional[Dict[str, Any]],
    *,
    margin: int = 0,
) -> bool:
    if not left or not right:
        return False
    required = {"min_y", "min_x", "max_y", "max_x"}
    if not required.issubset(left) or not required.issubset(right):
        return False
    return not (
        int(left["max_y"]) < int(right["min_y"]) - margin
        or int(left["min_y"]) > int(right["max_y"]) + margin
        or int(left["max_x"]) < int(right["min_x"]) - margin
        or int(left["min_x"]) > int(right["max_x"]) + margin
    )


def _component_offender_report(
    component: Component,
    *,
    reason: str,
    alignment_score: Optional[float] = None,
    paired_color: Optional[int] = None,
) -> Dict[str, Any]:
    report = component.to_report()
    report.update(
        {
            "reason": reason,
            "bbox": _component_bbox(component),
        }
    )
    if alignment_score is not None:
        report["alignment_score"] = round(float(alignment_score), 4)
    if paired_color is not None:
        report["paired_color"] = int(paired_color)
    return report


def _offending_component_diagnostics(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    pair_threshold: float = 4.15,
    limit: int = 16,
) -> Dict[str, Any]:
    """Identify concrete components that likely block readiness."""

    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    all_components = first_components + second_components
    violations = _get_boundary_violation_components(all_components, shape)

    offender_by_key: Dict[Tuple[int, int, int, int, int, int], Dict[str, Any]] = {}

    def add_offender(
        component: Component,
        *,
        reason: str,
        alignment_score: Optional[float] = None,
        paired_color: Optional[int] = None,
    ) -> None:
        key = _component_key(component)
        report = offender_by_key.get(key)
        if report is None:
            offender_by_key[key] = _component_offender_report(
                component,
                reason=reason,
                alignment_score=alignment_score,
                paired_color=paired_color,
            )
            return
        reasons = set(str(report.get("reason", "")).split("+"))
        reasons.add(reason)
        report["reason"] = "+".join(sorted(item for item in reasons if item))
        if alignment_score is not None:
            report["alignment_score"] = min(
                float(report.get("alignment_score", alignment_score)),
                float(alignment_score),
            )

    unused_second: Set[int] = set(range(len(second_components)))
    paired_second: Set[int] = set()
    low_alignment_pairs: List[Dict[str, Any]] = []
    for first in first_components:
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if not candidates:
            add_offender(first, reason="unpaired_first")
            continue
        idx, alignment = max(candidates, key=lambda item: float(item[1]["score"]))
        unused_second.remove(idx)
        paired_second.add(idx)
        score = float(alignment["score"])
        if score < pair_threshold:
            second = second_components[idx]
            add_offender(
                first,
                reason="low_alignment_pair",
                alignment_score=score,
                paired_color=second.color,
            )
            add_offender(
                second,
                reason="low_alignment_pair",
                alignment_score=score,
                paired_color=first.color,
            )
            low_alignment_pairs.append(alignment)

    for idx, second in enumerate(second_components):
        if idx not in paired_second:
            add_offender(second, reason="unpaired_second")

    for component in violations:
        add_offender(component, reason="dotted_boundary_violation")

    offenders = list(offender_by_key.values())
    offenders.sort(
        key=lambda item: (
            "dotted_boundary_violation" in str(item.get("reason", "")),
            int(item.get("size", 0)),
        ),
        reverse=True,
    )
    largest = max(offenders, key=lambda item: int(item.get("size", 0)), default=None)
    return {
        "pair_colors": list(pair_colors),
        "component_counts": {
            str(pair_colors[0]): len(first_components),
            str(pair_colors[1]): len(second_components),
        },
        "offending_count": len(offenders),
        "violation_count": len(violations),
        "low_alignment_pair_count": len(low_alignment_pairs),
        "largest_offender": largest,
        "target_components": offenders[: max(1, int(limit))],
        "low_alignment_pairs": low_alignment_pairs[:8],
    }


def _target_action6_candidates_from_components(
    components: Sequence[Dict[str, Any]],
    *,
    grid: Sequence[Sequence[int]],
    max_candidates: int = 12,
) -> List[Dict[str, int]]:
    arr = np.array(grid, dtype=np.int32)
    height, width = arr.shape if arr.ndim == 2 else (64, 64)
    out: List[Dict[str, int]] = []
    seen: Set[Tuple[int, int]] = set()

    def add_grid_point(y: float, x: float) -> None:
        if len(out) >= max_candidates:
            return
        if height <= 1 or width <= 1:
            norm_y, norm_x = 32, 32
        else:
            norm_y = int(round(63.0 * float(y) / float(height - 1)))
            norm_x = int(round(63.0 * float(x) / float(width - 1)))
        norm_y = max(0, min(63, norm_y))
        norm_x = max(0, min(63, norm_x))
        key = (norm_y, norm_x)
        if key in seen:
            return
        seen.add(key)
        out.append({"x": norm_x, "y": norm_y})

    for component in components:
        bbox = component.get("bbox") or component
        if not {"min_y", "min_x", "max_y", "max_x"}.issubset(bbox):
            continue
        min_y = int(bbox["min_y"])
        min_x = int(bbox["min_x"])
        max_y = int(bbox["max_y"])
        max_x = int(bbox["max_x"])
        mid_y = 0.5 * (min_y + max_y)
        mid_x = 0.5 * (min_x + max_x)
        for y, x in (
            (mid_y, mid_x),
            (min_y, min_x),
            (min_y, max_x),
            (max_y, min_x),
            (max_y, max_x),
            (mid_y, min_x),
            (mid_y, max_x),
            (min_y, mid_x),
            (max_y, mid_x),
        ):
            add_grid_point(y, x)
            if len(out) >= max_candidates:
                return out
    return out


def _rewrite_is_saturated(node: GuidedNode, *, action3_threshold: int = 48) -> bool:
    return bool(
        node.first_noop_index is not None
        or int(node.actions.count("ACTION3")) >= int(action3_threshold)
    )


def _compute_spatial_coherence(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    target_comp_info: Dict[str, Any],
) -> float:
    """Compute spatial coherence: how well target aligns with existing matched pairs."""
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    
    # Get components
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    
    # Find matched pairs
    unused_second: set = set(range(len(second_components)))
    pairs: List[Tuple[Component, Component, float]] = []
    
    for first in first_components:
        candidates = [
            (idx, _pair_alignment(first, second_components[idx], shape))
            for idx in unused_second
        ]
        if candidates:
            idx, alignment = max(candidates, key=lambda item: float(item[1]["score"]))
            unused_second.remove(idx)
            score = float(alignment["score"])
            pairs.append((first, second_components[idx], score))
    
    if not pairs or not target_comp_info:
        return 0.0
    
    # Get target centroid
    target_cy = target_comp_info.get("centroid_y", 0)
    target_cx = target_comp_info.get("centroid_x", 0)
    
    # Find nearest matched pair
    min_pair_dist = float('inf')
    for first, second, score in pairs:
        pair_center_y = (first.centroid_y + second.centroid_y) / 2
        pair_center_x = (first.centroid_x + second.centroid_x) / 2
        dist = np.hypot(target_cy - pair_center_y, target_cx - pair_center_x)
        if dist < min_pair_dist:
            min_pair_dist = dist
    
    # Spatial coherence: inverse distance to nearest pair (normalized)
    coherence = 1.0 / (1.0 + min_pair_dist / 10.0)
    return coherence


def _compute_target_info_for_gate(
    node: GuidedNode,
    pair_colors: Tuple[int, int],
) -> Tuple[bool, Dict[str, Any]]:
    """Compute target info for cursor_target gate."""
    grid = getattr(node.env, "observation_space", None)
    if grid is None:
        return False, {"error": "no grid"}
    
    grid_data = _primary_grid(grid)
    cursor_components = _connected_components_for_colors(grid_data, [4, 5])
    
    is_near, target_info = _cursor_near_small_unmatched_target(
        grid_data,
        pair_colors=pair_colors,
        cursor_components=cursor_components,
        target_color=11,
        max_size=2,
        max_distance=5.0,
    )
    return is_near, target_info


def _submit_gate_allows(
    node: GuidedNode,
    *,
    rule_model: RuleModel,
    root_match: MatchScore,
    gate_mode: str = "strict",  # "strict" | "semantic" | "probe" | "cursor_target"
    strong_improvement_margin: float = 15.0,
    max_submit_probes: int = 16,
    consecutive_submits: int = 0,
    precomputed_target_info: Optional[Tuple[bool, Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """ACTION2 gate: semantic correspondence readiness (not dotted constraint).
    
    Returns (allowed, target_info_or_none).
    
    Modes:
        strict: Original dotted-constraint based (discredited by audit)
        semantic: match_score >= threshold OR matched_pairs >= expected
        probe: Allow limited ACTION2 probes even without readiness
        cursor_target: Cursor near small unmatched color-11 component
    """
    if gate_mode == "probe":
        # Allow ACTION2 as probe, but track usage
        return (consecutive_submits < max_submit_probes), None
    
    if gate_mode == "semantic":
        # Semantic correspondence readiness (no dotted constraint check)
        return (
            _semantic_readiness_gate_passed(
                node.match,
                rule_model=rule_model,
                root_match=root_match,
                match_delta_from_root=node.match_delta_from_root,
                strong_improvement_margin=strong_improvement_margin,
            ),
            None,
        )
    
    if gate_mode == "cursor_target":
        # Key insight from analysis: ACTION2 succeeds when cursor is near
        # a small UNMATCHED component with HIGH spatial coherence (>= 0.75)
        # Calibrated from successful submits levels 0-6
        if precomputed_target_info:
            return precomputed_target_info[0], precomputed_target_info[1]
        
        is_near, target_info = _compute_target_info_for_gate(node, rule_model.pair_colors)
        if not is_near:
            return False, target_info
        
        # Additional check: spatial coherence with existing pairs
        # Get grid to compute spatial coherence
        grid = getattr(node.env, "observation_space", None)
        if grid is not None:
            grid_data = _primary_grid(grid)
            spatial_score = _compute_spatial_coherence(
                grid_data,
                pair_colors=rule_model.pair_colors,
                target_comp_info=target_info,
            )
            target_info["spatial_coherence_score"] = round(spatial_score, 4)
            # Threshold calibrated from analysis:
            # successes 0-6: min ~0.63, mean 0.81
            # level 7 fails: mean 0.68
            # Using 0.60 to be permissive (not 0.75 which would exclude valid successes)
            is_coherent = spatial_score >= 0.60
            return is_coherent, target_info
        
        return is_near, target_info
    
    # strict mode (original, now known to be wrong)
    allowed = _strict_readiness_gate_passed(
        node.match,
        rule_model=rule_model,
        root_match=root_match,
    )
    return allowed, None


def _semantic_readiness_gate_passed(
    score: MatchScore,
    *,
    rule_model: RuleModel,
    root_match: MatchScore,
    match_delta_from_root: float,
    strong_improvement_margin: float = 15.0,
) -> bool:
    match_ready = score.score >= rule_model.ready_threshold
    pairs_ready = score.matched_pairs >= rule_model.expected_matched_pairs
    strong_improvement = match_delta_from_root >= strong_improvement_margin
    unmatched_improved = _unmatched_total(score) <= _unmatched_total(root_match)
    return bool((match_ready or pairs_ready or strong_improvement) and unmatched_improved)


def _strict_readiness_gate_passed(
    score: MatchScore,
    *,
    rule_model: RuleModel,
    root_match: MatchScore,
) -> bool:
    return bool(
        int(score.matched_pairs) >= int(rule_model.expected_matched_pairs) + 1
        and int(score.dotted_constraint_violations) == 0
        and _unmatched_total(score) < _unmatched_total(root_match)
    )


def _make_guided_child(
    parent: GuidedNode,
    *,
    raw: Any,
    env: Any,
    action: str,
    action_data: Optional[Dict[str, int]],
    rule_model: RuleModel,
    root_match: MatchScore,
    parent_grid: Sequence[Sequence[int]],
    danger_actions: Sequence[str],
    scoring_mode: str = "default",  # "default" | "dotted_constraint_repair" | "cursor_target"
    submit_gate: str = "strict",  # "strict" | "semantic" | "probe" | "cursor_target"
    post_saturation_probe: bool = False,
    parent_rewrite_saturated: bool = False,
    rewrite_saturation_action3_threshold: int = 48,
    action_ontology_model: Optional[Dict[str, Any]] = None,
    action_ontology_weight: float = 1.0,
    auto_levelup_state_classifier: Optional[Dict[str, Any]] = None,
    auto_levelup_classifier_weight: float = 1.0,
    largest_second_pressure: bool = False,
) -> GuidedNode:
    if raw is None:
        state = "ERROR"
        level = parent.level
        grid = parent_grid
        available: List[str] = []
    else:
        state = _state_name(getattr(raw, "state", "UNKNOWN"))
        level = int(getattr(raw, "levels_completed", 0) or 0)
        grid = _primary_grid(raw)
        available = _available_names_from_raw(raw)

    child_match = match_score(grid, pair_colors=rule_model.pair_colors)
    parent_global = parent.global_correspondence or _cached_global_correspondence_score(
        parent_grid,
        pair_colors=rule_model.pair_colors,
    )
    child_global = _cached_global_correspondence_score(grid, pair_colors=rule_model.pair_colors)
    child_auto_levelup_state = _score_auto_levelup_state(
        grid,
        pair_colors=rule_model.pair_colors,
        match=child_match,
        global_score=child_global,
        classifier=auto_levelup_state_classifier,
    )
    died = raw is None or state == "GAME_OVER"
    reached_next = level > parent.level or state == "WIN"
    grid_hash = _hash_grid(grid)
    repeated = grid_hash in set(parent.path_hashes)
    repeat_penalty = parent.repeat_penalty + (1.0 if repeated else 0.0)
    delta_root = child_match.score - root_match.score
    delta_parent = child_match.score - parent.match.score
    global_delta_parent = float(child_global.score) - float(parent_global.score)
    global_delta_root = float(parent.global_delta_from_root) + global_delta_parent
    readiness_gate_passed = _semantic_readiness_gate_passed(
        child_match,
        rule_model=rule_model,
        root_match=root_match,
        match_delta_from_root=delta_root,
    )
    strict_readiness_gate_passed = _strict_readiness_gate_passed(
        child_match,
        rule_model=rule_model,
        root_match=root_match,
    )
    unmatched_delta = _unmatched_total(root_match) - _unmatched_total(child_match)
    dotted_delta = int(root_match.dotted_constraint_violations) - int(child_match.dotted_constraint_violations)

    # Dotted constraint violation tracking for repair analysis
    arr = np.array(grid, dtype=np.int32) if grid else np.zeros((64, 64), dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    first_components = _connected_components_for_colors(grid, [rule_model.pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [rule_model.pair_colors[1]])
    all_components = first_components + second_components
    violation_components = _get_boundary_violation_components(all_components, shape)

    # Track parent violations for comparison
    parent_arr = np.array(parent_grid, dtype=np.int32) if parent_grid else np.zeros((64, 64), dtype=np.int32)
    parent_shape = parent_arr.shape if parent_arr.ndim == 2 else (64, 64)
    parent_first = _connected_components_for_colors(parent_grid, [rule_model.pair_colors[0]])
    parent_second = _connected_components_for_colors(parent_grid, [rule_model.pair_colors[1]])
    parent_all = parent_first + parent_second
    parent_violations = _get_boundary_violation_components(parent_all, parent_shape)

    # Cursor proximity to violations
    cursor_components = _connected_components_for_colors(grid, (4, 5))
    cursor_near_violation = _cursor_near_dotted_violation(cursor_components, violation_components, max_distance=12.0)

    # Distance from ACTION6 target to nearest violation
    distance_to_violation = float('inf')
    if action == "ACTION6" and action_data and violation_components:
        ay = action_data.get("y", 0)
        ax = action_data.get("x", 0)
        distance_to_violation = _distance_to_nearest_violation(ay, ax, violation_components)
    
    # Compute target info for cursor_target mode
    target_info_for_gate: Optional[Tuple[bool, Dict[str, Any]]] = None
    if submit_gate == "cursor_target":
        target_info_for_gate = _compute_target_info_for_gate(parent, rule_model.pair_colors)
    
    submit_ready, _ = _submit_gate_allows(
        parent,
        rule_model=rule_model,
        root_match=root_match,
        gate_mode=submit_gate,
        precomputed_target_info=target_info_for_gate,
    )
    treat_action2_as_submit = scoring_mode not in {
        "action_ontology_guided",
        "auto_levelup_state_classifier",
        "level7_fragmentation_guided",
        "level7_fragmentation_guarded",
    }
    submit_not_ready = (
        treat_action2_as_submit
        and action == "ACTION2"
        and not submit_ready
        and not reached_next
    )
    contradiction = _is_action3_contradiction(
        action=action,
        parent_grid=parent_grid,
        child_grid=grid,
        child_state=state,
    )
    action_bbox = _diff_bbox(parent_grid, grid)
    spatial_operator_change = 0
    spatial_operator_bonus = 0.0
    component_change = _component_delta(parent_grid, grid)  # Compute for all actions
    if action == "ACTION6" and not died and action_bbox:
        spatial_operator_change = int(action_bbox.get("changed_cells", 0) or 0)
        # Keep structured ACTION6 branches alive long enough for follow-up
        # moves, without letting raw spatial change dominate correspondence.
        spatial_operator_bonus = min(
            180.0,
            40.0 + 1.6 * float(min(spatial_operator_change, 60)) + 3.0 * float(component_change),
        )
    max_action6_spatial_change = max(
        int(parent.max_action6_spatial_change),
        int(spatial_operator_change),
    )
    action6_transform_count = int(parent.action6_transform_count) + int(
        action == "ACTION6" and spatial_operator_change > 0
    )
    child_actions = parent.actions + [action]
    danger_prefix = _danger_prefix_len(child_actions, danger_actions)
    consecutive_submit = _consecutive_suffix(child_actions, "ACTION2")
    submit_loop_penalty = max(0, consecutive_submit - 2)
    action_affordance = _score_action_affordance(
        action=action,
        parent=parent,
        parent_grid=parent_grid,
        child_grid=grid,
        child_match=child_match,
        child_global=child_global,
        parent_global=parent_global,
        reached_next=bool(reached_next),
        died=bool(died),
        action_ontology_model=action_ontology_model,
    )
    parent_auto_levelup_score = float((parent.auto_levelup_state or {}).get("score", 0.0))
    child_auto_levelup_score = float(child_auto_levelup_state.get("score", 0.0))
    auto_levelup_state_delta_parent = child_auto_levelup_score - parent_auto_levelup_score
    fragmentation_guidance = _fragmentation_guidance_report(
        parent_state=parent.auto_levelup_state,
        child_state=child_auto_levelup_state,
        classifier=auto_levelup_state_classifier,
        largest_second_pressure=largest_second_pressure,
    )
    fragmentation_guardrails = _fragmentation_guardrail_report(
        child_state=child_auto_levelup_state,
        child_match=child_match,
        child_global=child_global,
    )

    # Scoring mode selection
    if scoring_mode == "dotted_constraint_repair":
        # Constraint-first scoring: prioritize repairing dotted constraints
        # +1000 if dotted violations decrease
        # +200 if cursor near dotted violation
        # +100 if unmatched total decreases
        # +50 if ACTION6 transforms near violation
        # -500 if dotted violations increase
        action6_near_violation = (
            action == "ACTION6"
            and spatial_operator_change > 0
            and distance_to_violation < 20.0
        )
        search_score = (
            5000.0 * float(reached_next)
            + 1000.0 * max(0, dotted_delta)  # Decrease in violations
            + 200.0 * float(cursor_near_violation)
            + 100.0 * max(0, unmatched_delta)
            + 50.0 * float(action6_near_violation)
            + 30.0 * delta_root  # Reduced weight on match score
            + 15.0 * delta_parent
            + 20.0 * child_match.matched_pairs
            - 500.0 * max(0, -dotted_delta)  # Increase in violations (penalty)
            - 90.0 * _unmatched_total(child_match)
            - 150.0 * child_match.dotted_constraint_violations
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 240.0 * float(submit_not_ready)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "cursor_target":
        # Cursor-target scoring: prioritize positioning cursor near UNMATCHED target
        # Key insight from analysis:
        # - high spatial_coherence_score = good (aligns with existing pairs)
        # - has_plausible_color10_context = bad (anti-signal, too much structure)
        # +10000 level up
        # +3000 cursor_near_target (any improvement)
        # +1500 high spatial_coherence
        # +1000 unmatched components exist
        # -300 has_plausible_color10_context (penalty for over-structured targets)
        cursor_delta = child_match.cursor_near_target - parent.match.cursor_near_target
        unmatched_improved = (child_match.unmatched_first + child_match.unmatched_second) < \
                             (parent.match.unmatched_first + parent.match.unmatched_second)
        has_unmatched = (child_match.unmatched_first + child_match.unmatched_second) > 0
        submit_without_unmatched_target = action == "ACTION2" and not has_unmatched
        
        # Get spatial coherence from parent target info if available
        spatial_coherence = parent.last_target_info.get("spatial_coherence_score", 0.0)
        has_plausible_context = parent.last_target_info.get("has_plausible_color10_context", False)
        high_coherence = spatial_coherence >= 0.60
        
        search_score = (
            10000.0 * float(reached_next)
            + 3000.0 * max(0, cursor_delta)  # cursor_near_target becomes 1 or improves
            + 1500.0 * float(high_coherence)  # high spatial coherence bonus
            + 1000.0 * float(has_unmatched)  # have unmatched targets to solve
            + 500.0 * float(unmatched_improved)  # reducing unmatched count
            + 100.0 * delta_root  # match_score improves
            + 60.0 * delta_parent
            + 180.0 * child_match.matched_pairs
            + 80.0 * unmatched_delta
            - 300.0 * float(has_plausible_context)  # Penalty: too much color10 structure
            - 500.0 * float(submit_without_unmatched_target)  # Penalty for submit without targets
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 240.0 * float(submit_not_ready)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    
    elif scoring_mode == "action3_productive":
        # Level 7 insight: ACTION3 is the main transformation engine
        # Human trace: 48 productive ACTION3, then saturation (no-op)
        # Strategy: reward productive ACTION3, penalize no-op, submit only after saturation
        # +10000 level up
        # +1000 ACTION3 if changed_cells > 0 (productive)
        # +500 component_rewrite effect
        # +300 small_11_count changes
        # -800 ACTION3 no-op (saturation signal)
        # -500 repeated no-op hash
        # ACTION2 only after saturation or strong target condition
        
        # Compute ACTION3 effect using grid diff
        if action == "ACTION3":
            arr_parent = np.array(parent_grid)
            arr_child = np.array(grid)
            if arr_parent.shape == arr_child.shape:
                action3_changed_cells = int((arr_parent != arr_child).sum())
            else:
                action3_changed_cells = -1  # Shape change = major rewrite
        else:
            action3_changed_cells = 0
        
        action3_productive = action == "ACTION3" and action3_changed_cells > 0
        action3_noop = action == "ACTION3" and action3_changed_cells == 0
        
        # Detect ACTION3 saturation (consecutive no-ops)
        recent_actions = parent.actions[-3:] if len(parent.actions) >= 3 else parent.actions
        action3_saturation = sum(1 for a in recent_actions if a == "ACTION3") >= 2 and action3_noop
        
        # Allow submit only after ACTION3 saturation or strong cursor target
        submit_ready = action == "ACTION2" and (action3_saturation or child_match.cursor_near_target)
        
        search_score = (
            10000.0 * float(reached_next)
            + 1000.0 * float(action3_productive)  # Productive ACTION3
            + 500.0 * float(action == "ACTION3" and component_change > 0)  # Component rewrite
            + 300.0 * float(unmatched_delta != 0)  # small_11 changes
            + 200.0 * delta_root  # Match score improvement
            + 100.0 * child_match.matched_pairs
            - 800.0 * float(action3_noop)  # No-op penalty
            - 500.0 * float(action3_saturation)  # Saturation signal
            - 500.0 * float(action == "ACTION2" and not submit_ready)  # Submit before ready
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 240.0 * float(submit_not_ready)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    
    elif scoring_mode == "rewrite_until_saturation":
        # Level 7: ACTION3 rewrite engine mode - cumulative scoring
        # Goal: reward long chains of productive ACTION3 until saturation
        # +200 per productive ACTION3 in history (cumulative)
        # +20 per changed cell (cumulative)
        # -500 per no-op ACTION3
        # -100 for non-ACTION3 between rewrites
        
        # Compute current ACTION3 effect
        if action == "ACTION3":
            arr_parent = np.array(parent_grid)
            arr_child = np.array(grid)
            if arr_parent.shape == arr_child.shape:
                action3_changed_cells = int((arr_parent != arr_child).sum())
            else:
                action3_changed_cells = -1  # Major rewrite
        else:
            action3_changed_cells = 0
        
        action3_productive = action == "ACTION3" and action3_changed_cells > 0
        action3_noop = action == "ACTION3" and action3_changed_cells == 0
        
        # Cumulative counters from parent history
        parent_a3_count = sum(1 for a in parent.actions if a == "ACTION3")
        parent_a3_productive_count = getattr(parent, 'action3_productive_count', 0)
        parent_cumulative_changes = getattr(parent, 'cumulative_changed_cells', 0)
        
        # Update counters for this step
        current_productive_count = parent_a3_productive_count + (1 if action3_productive else 0)
        current_cumulative_changes = parent_cumulative_changes + max(0, action3_changed_cells)
        
        # Detect non-ACTION3 interrupting rewrite chain
        non_a3_penalty = 0
        if parent.actions and parent.actions[-1] != "ACTION3" and action != "ACTION3":
            # Chain of non-ACTION3 - small penalty
            non_a3_penalty = 10
        if parent.actions and parent.actions[-1] == "ACTION3" and action != "ACTION3":
            # Breaking the rewrite chain - stronger penalty
            non_a3_penalty = 100
        
        # Detect saturation (consecutive no-ops)
        recent_actions = parent.actions[-3:] if len(parent.actions) >= 3 else parent.actions
        action3_saturation = sum(1 for a in recent_actions if a == "ACTION3") >= 2 and action3_noop
        
        # Saturation is signal to stop, not a penalty
        saturation_bonus = 500 if action3_saturation else 0
        
        search_score = (
            10000.0 * float(reached_next)  # Level up still matters
            + 200.0 * current_productive_count  # Cumulative productive ACTION3 count
            + 20.0 * current_cumulative_changes  # Cumulative changed cells
            + 100.0 * float(action3_productive)  # Immediate productive bonus
            + saturation_bonus  # Signal saturation achieved
            - 500.0 * float(action3_noop)  # No-op penalty
            - 200.0 * float(action == "ACTION2")  # Discourage submit in rewrite mode
            - non_a3_penalty  # Penalty for breaking rewrite chain
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 260.0 * float(danger_prefix)
            - 2.0 * (parent.depth + 1)
        )
    elif scoring_mode == "semantic":
        # Explicit semantic objective for level 7:
        # first increase matched pairs, then reduce unresolved target-side
        # components, then repair dotted violations, then improve cursor
        # proximity. This avoids hiding a one-pair local optimum inside a
        # broad continuous score.
        parent_unmatched_second = int(parent.match.unmatched_second)
        parent_dotted = int(parent.match.dotted_constraint_violations)
        unmatched_second_delta = parent_unmatched_second - int(child_match.unmatched_second)
        dotted_parent_delta = parent_dotted - int(child_match.dotted_constraint_violations)
        matched_delta = int(child_match.matched_pairs) - int(parent.match.matched_pairs)
        cursor_delta = int(child_match.cursor_near_target) - int(parent.match.cursor_near_target)
        search_score = (
            10000.0 * float(reached_next)
            + 1800.0 * float(child_match.matched_pairs)
            + 900.0 * float(max(0, matched_delta))
            + 520.0 * float(max(0, unmatched_second_delta))
            + 360.0 * float(max(0, dotted_parent_delta))
            + 180.0 * float(max(0, cursor_delta))
            + 70.0 * delta_root
            + 35.0 * delta_parent
            - 220.0 * float(child_match.unmatched_second)
            - 120.0 * float(child_match.unmatched_first)
            - 180.0 * float(child_match.dotted_constraint_violations)
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 240.0 * float(submit_not_ready)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "global_correspondence":
        # Family-level/topological objective for AR25 level 7.
        # This intentionally treats local matched_pairs as diagnostic only:
        # the search should prefer states where both color families share a
        # similar global grammar even when no nearby component pair is obvious.
        search_score = (
            10000.0 * float(reached_next)
            + 90.0 * float(child_global.score)
            + 700.0 * float(max(0.0, global_delta_parent))
            + 280.0 * float(max(0.0, global_delta_root))
            + 180.0 * float(max(0, dotted_delta))
            + 25.0 * delta_parent
            - 120.0 * float(max(0, -dotted_delta))
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 240.0 * float(submit_not_ready)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "global_semantic_hybrid":
        # Phase bridge: first use global family structure as the compass,
        # then reward states that become locally actionable for a submit.
        unmatched_total = _unmatched_total(child_match)
        local_phase_active = child_global.score >= 65.0 or global_delta_root >= 28.0
        local_phase_weight = 1.0 if local_phase_active else 0.10
        local_readiness_score = (
            300.0 * float(readiness_gate_passed)
            + 180.0 * float(strict_readiness_gate_passed)
            + 80.0 * float(max(0.0, delta_parent))
            + float(child_match.score)
            - 100.0 * float(child_match.dotted_constraint_violations)
            - 50.0 * float(unmatched_total)
        )
        search_score = (
            10000.0 * float(reached_next)
            + 1000.0 * float(child_global.structure_similarity)
            + 65.0 * float(child_global.score)
            + 220.0 * float(max(0.0, global_delta_root))
            + 90.0 * float(max(0.0, global_delta_parent))
            + local_phase_weight * local_readiness_score
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 240.0 * float(submit_not_ready)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "action_ontology_guided":
        # Goal ontology + empirical action ontology.
        #
        # ACTION_i remains an unknown operator. The action ontology supplies
        # a prior from probes, while action_affordance scores the actual
        # transition sampled in this state.
        unmatched_total = _unmatched_total(child_match)
        local_readiness_score = (
            260.0 * float(readiness_gate_passed)
            + 140.0 * float(strict_readiness_gate_passed)
            + 70.0 * float(max(0.0, delta_parent))
            - 92.0 * float(child_match.dotted_constraint_violations)
            - 46.0 * float(unmatched_total)
        )
        ontology_score = (
            float(action_ontology_weight)
            * float(action_affordance.get("score", 0.0))
        )
        search_score = (
            10000.0 * float(reached_next)
            + 850.0 * float(child_global.structure_similarity)
            + 56.0 * float(child_global.score)
            + 185.0 * float(max(0.0, global_delta_root))
            + 82.0 * float(max(0.0, global_delta_parent))
            + 0.75 * local_readiness_score
            + ontology_score
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 260.0 * float(danger_prefix)
            - 95.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "auto_levelup_state_classifier":
        # State classifier: prefer states that resemble human states just
        # before automatic level transitions, then use action ontology as the
        # local steering prior. ACTION_i remains semantically unknown.
        unmatched_total = _unmatched_total(child_match)
        classifier_score = float(auto_levelup_classifier_weight) * child_auto_levelup_score
        classifier_delta = float(auto_levelup_classifier_weight) * max(0.0, auto_levelup_state_delta_parent)
        ontology_score = 0.70 * float(action_ontology_weight) * float(action_affordance.get("score", 0.0))
        local_repair_score = (
            210.0 * float(readiness_gate_passed)
            + 120.0 * float(strict_readiness_gate_passed)
            + 58.0 * float(max(0.0, delta_parent))
            - 82.0 * float(child_match.dotted_constraint_violations)
            - 38.0 * float(unmatched_total)
        )
        search_score = (
            10000.0 * float(reached_next)
            + 82.0 * classifier_score
            + 420.0 * classifier_delta
            - 180.0 * float(auto_levelup_classifier_weight) * max(0.0, -auto_levelup_state_delta_parent)
            + 32.0 * float(child_global.score)
            + 96.0 * float(max(0.0, global_delta_parent))
            + 0.45 * local_repair_score
            + ontology_score
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 260.0 * float(danger_prefix)
            - 95.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "level7_fragmentation_guided":
        # Level-7 specific pivot: the human pre-auto-levelup signature has
        # many small color-11 fragments. Do not treat unmatched_total as a
        # generic penalty here; moving it toward the human reference can be
        # useful.
        frag_score = float(fragmentation_guidance.get("score", 0.0))
        frag_improvement = float(fragmentation_guidance.get("improvement_score", 0.0))
        ontology_score = 0.42 * float(action_ontology_weight) * float(action_affordance.get("score", 0.0))
        auto_score = float(child_auto_levelup_state.get("score", 0.0))
        search_score = (
            10000.0 * float(reached_next)
            + 38.0 * frag_score
            + 24.0 * max(0.0, frag_improvement)
            + 34.0 * auto_score
            + 22.0 * float(child_global.score)
            + 58.0 * float(max(0.0, global_delta_parent))
            + ontology_score
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 260.0 * float(danger_prefix)
            - 95.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "level7_fragmentation_guarded":
        # V2: keep the fragmentation gradient, but add structural guardrails
        # so we do not destroy the useful level-7 scaffold while fragmenting.
        frag_score = float(fragmentation_guidance.get("score", 0.0))
        frag_improvement = float(fragmentation_guidance.get("improvement_score", 0.0))
        guard_penalty = float(fragmentation_guardrails.get("penalty", 0.0))
        guard_bonus = 220.0 if fragmentation_guardrails.get("passes") else 0.0
        ontology_score = 0.32 * float(action_ontology_weight) * float(action_affordance.get("score", 0.0))
        auto_score = float(child_auto_levelup_state.get("score", 0.0))
        search_score = (
            10000.0 * float(reached_next)
            + 34.0 * frag_score
            + 18.0 * max(0.0, frag_improvement)
            + 42.0 * auto_score
            + 18.0 * float(child_global.score)
            + 44.0 * float(max(0.0, global_delta_parent))
            + ontology_score
            + guard_bonus
            - 4.0 * guard_penalty
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 260.0 * float(danger_prefix)
            - 95.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    elif scoring_mode == "local_repair":
        # Local repair assumes a high-global seed already exists. The score
        # rewards making that state actionable without relying on submit.
        unmatched_total = _unmatched_total(child_match)
        parent_unmatched_total = _unmatched_total(parent.match)
        unmatched_parent_delta = parent_unmatched_total - unmatched_total
        dotted_parent_delta = (
            int(parent.match.dotted_constraint_violations)
            - int(child_match.dotted_constraint_violations)
        )
        search_score = (
            10000.0 * float(reached_next)
            + 520.0 * float(max(0, unmatched_parent_delta))
            + 460.0 * float(max(0, dotted_parent_delta))
            + 320.0 * float(readiness_gate_passed)
            + 180.0 * float(strict_readiness_gate_passed)
            + 90.0 * float(max(0.0, delta_parent))
            + 18.0 * float(child_global.score)
            - 160.0 * float(max(0, -unmatched_parent_delta))
            - 180.0 * float(max(0, -dotted_parent_delta))
            - 100.0 * float(child_match.dotted_constraint_violations)
            - 50.0 * float(unmatched_total)
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    else:
        # Default scoring (original)
        search_score = (
            5000.0 * float(reached_next)
            + 60.0 * delta_root
            + 28.0 * delta_parent
            + 180.0 * child_match.matched_pairs
            + 80.0 * unmatched_delta
            + 70.0 * dotted_delta
            + 18.0 * child_match.cursor_near_target
            + 4.0 * child_match.shape_overlap_or_alignment
            - 36.0 * _unmatched_total(child_match)
            - 90.0 * child_match.dotted_constraint_violations
            - 5000.0 * float(died)
            - 850.0 * float(contradiction)
            - 240.0 * float(submit_not_ready)
            - 260.0 * float(danger_prefix)
            - 180.0 * float(submit_loop_penalty)
            - 35.0 * repeat_penalty
            - 2.0 * (parent.depth + 1)
            + spatial_operator_bonus
        )
    if action == "ACTION5":
        search_score += 15.0
    if action == "ACTION7":
        search_score -= 18.0

    # Convert violation components to serializable format
    viol_before = [
        {"color": c.color, "centroid_y": c.centroid_y, "centroid_x": c.centroid_x, "size": c.size}
        for c in parent_violations
    ]
    viol_after = [
        {"color": c.color, "centroid_y": c.centroid_y, "centroid_x": c.centroid_x, "size": c.size}
        for c in violation_components
    ]

    # Compute cumulative counters for rewrite_until_saturation mode.
    action3_changed_cells = _grid_changed_cells(parent_grid, grid) if action == "ACTION3" else 0
    parent_a3_productive_count = getattr(parent, 'action3_productive_count', 0)
    parent_cumulative_changes = getattr(parent, 'cumulative_changed_cells', 0)
    parent_action3_sequence = list(getattr(parent, "action3_changed_cells_sequence", []))

    is_rewrite_a3 = scoring_mode == "rewrite_until_saturation" and action == "ACTION3"
    child_a3_productive_count = parent_a3_productive_count + (1 if is_rewrite_a3 and action3_changed_cells > 0 else 0)
    child_cumulative_changes = parent_cumulative_changes + (action3_changed_cells if is_rewrite_a3 else 0)
    child_action3_sequence = list(parent_action3_sequence)
    child_first_noop_index = parent.first_noop_index
    if is_rewrite_a3:
        child_action3_sequence.append(int(action3_changed_cells))
        if action3_changed_cells == 0 and child_first_noop_index is None:
            child_first_noop_index = int(parent.depth + 1)
    child_rewrite_saturated = bool(
        scoring_mode == "rewrite_until_saturation"
        and (
            child_first_noop_index is not None
            or child_actions.count("ACTION3") >= int(rewrite_saturation_action3_threshold)
        )
    )

    return GuidedNode(
        actions=child_actions,
        action_data=parent.action_data + [dict(action_data) if action_data else None],
        state=state,
        level=level,
        grid_hash=grid_hash,
        search_score=float(search_score),
        match=child_match,
        depth=parent.depth + 1,
        global_correspondence=child_global,
        reached_next_level=bool(reached_next),
        died=bool(died),
        submit_not_ready=bool(submit_not_ready),
        action3_contradiction=bool(contradiction),
        readiness_gate_passed=bool(readiness_gate_passed),
        danger_prefix_len=int(danger_prefix),
        consecutive_submit=int(consecutive_submit),
        consecutive_submits=int(consecutive_submit),
        submit_gate_passed=bool(submit_ready),
        spatial_operator_change=int(spatial_operator_change),
        max_action6_spatial_change=int(max_action6_spatial_change),
        action6_transform_count=int(action6_transform_count),
        rewrite_saturated=child_rewrite_saturated,
        post_saturation_probe_phase=bool(post_saturation_probe and parent_rewrite_saturated),
        repeat_penalty=repeat_penalty,
        match_delta_from_root=float(delta_root),
        match_delta_from_parent=float(delta_parent),
        global_delta_from_root=float(global_delta_root),
        global_delta_from_parent=float(global_delta_parent),
        unmatched_delta_from_root=int(unmatched_delta),
        dotted_delta_from_root=int(dotted_delta),
        available_actions=available,
        path_hashes=parent.path_hashes + [grid_hash],
        env=env,
        grid=grid,
        violation_components_before=viol_before,
        violation_components_after=viol_after,
        action_responsible=action,
        distance_action6_to_violation=distance_to_violation if distance_to_violation != float('inf') else -1.0,
        cursor_near_dotted_violation=cursor_near_violation,
        last_target_info=target_info_for_gate[1] if target_info_for_gate else {},
        action3_productive_count=child_a3_productive_count,
        cumulative_changed_cells=child_cumulative_changes,
        action3_changed_cells_sequence=child_action3_sequence,
        first_noop_index=child_first_noop_index,
        action_affordance=action_affordance,
        auto_levelup_state=child_auto_levelup_state,
        fragmentation_guidance=fragmentation_guidance,
        fragmentation_guardrails=fragmentation_guardrails,
    )


def _probe_action2_from_best_match_nodes(
    nodes: Sequence[GuidedNode],
    *,
    full_game_id: str,
    rule_model: RuleModel,
    root_match: MatchScore,
    danger_actions: Sequence[str],
    scoring_mode: str,
    submit_gate: str,
    limit: int,
    action_ontology_model: Optional[Dict[str, Any]] = None,
    action_ontology_weight: float = 1.0,
    auto_levelup_state_classifier: Optional[Dict[str, Any]] = None,
    auto_levelup_classifier_weight: float = 1.0,
    largest_second_pressure: bool = False,
) -> List[GuidedNode]:
    """Apply ACTION2 once from top semantic states, independent of beam rank."""

    if limit <= 0:
        return []

    out: List[GuidedNode] = []
    seen: Set[str] = set()
    for node in nodes:
        if len(out) >= limit:
            break
        if node.died or node.reached_next_level or node.env is None:
            continue
        if node.grid_hash in seen:
            continue
        seen.add(node.grid_hash)
        branch_env = copy.deepcopy(node.env)
        parent_grid = _primary_grid(getattr(branch_env, "observation_space", None))
        raw = _step_branch(
            branch_env,
            full_game_id=full_game_id,
            action="ACTION2",
            action_data=None,
        )
        out.append(
            _make_guided_child(
                node,
                raw=raw,
                env=branch_env,
                action="ACTION2",
                action_data=None,
                rule_model=rule_model,
                root_match=root_match,
                parent_grid=parent_grid,
                danger_actions=danger_actions,
                scoring_mode=scoring_mode,
                submit_gate=submit_gate,
                action_ontology_model=action_ontology_model,
                action_ontology_weight=action_ontology_weight,
                auto_levelup_state_classifier=auto_levelup_state_classifier,
                auto_levelup_classifier_weight=auto_levelup_classifier_weight,
                largest_second_pressure=largest_second_pressure,
            )
        )
    return out


def _probe_action6_from_best_match_nodes(
    nodes: Sequence[GuidedNode],
    *,
    full_game_id: str,
    rule_model: RuleModel,
    root_match: MatchScore,
    danger_actions: Sequence[str],
    scoring_mode: str,
    submit_gate: str,
    action6_targets: Sequence[Dict[str, int]],
    limit_nodes: int,
    limit_total: int,
    action_ontology_model: Optional[Dict[str, Any]] = None,
    action_ontology_weight: float = 1.0,
    auto_levelup_state_classifier: Optional[Dict[str, Any]] = None,
    auto_levelup_classifier_weight: float = 1.0,
    largest_second_pressure: bool = False,
) -> List[Dict[str, Any]]:
    """Probe ACTION6 as an operator from top semantic states, not in-beam."""

    if limit_nodes <= 0 or limit_total <= 0:
        return []

    reports: List[Dict[str, Any]] = []
    seen_nodes: Set[str] = set()
    for node in nodes:
        if len(reports) >= limit_total:
            break
        if node.died or node.reached_next_level or node.env is None:
            continue
        if node.grid_hash in seen_nodes:
            continue
        seen_nodes.add(node.grid_hash)
        if len(seen_nodes) > limit_nodes:
            break

        for target in action6_targets:
            if len(reports) >= limit_total:
                break
            branch_env = copy.deepcopy(node.env)
            parent_grid = _primary_grid(getattr(branch_env, "observation_space", None))
            raw = _step_branch(
                branch_env,
                full_game_id=full_game_id,
                action="ACTION6",
                action_data=dict(target),
            )
            child = _make_guided_child(
                node,
                raw=raw,
                env=branch_env,
                action="ACTION6",
                action_data=dict(target),
                rule_model=rule_model,
                root_match=root_match,
                parent_grid=parent_grid,
                danger_actions=danger_actions,
                scoring_mode=scoring_mode,
                submit_gate=submit_gate,
                action_ontology_model=action_ontology_model,
                action_ontology_weight=action_ontology_weight,
                auto_levelup_state_classifier=auto_levelup_state_classifier,
                auto_levelup_classifier_weight=auto_levelup_classifier_weight,
                largest_second_pressure=largest_second_pressure,
            )
            report = child.to_report()
            report.update(
                {
                    "action6_target": dict(target),
                    "matched_pairs_before_action6": int(node.match.matched_pairs),
                    "matched_pairs_after_action6": int(child.match.matched_pairs),
                    "matched_delta_after_action6": int(child.match.matched_pairs)
                    - int(node.match.matched_pairs),
                    "dotted_before_action6": int(node.match.dotted_constraint_violations),
                    "dotted_after_action6": int(child.match.dotted_constraint_violations),
                    "dotted_delta_after_action6": int(node.match.dotted_constraint_violations)
                    - int(child.match.dotted_constraint_violations),
                    "unmatched_second_before_action6": int(node.match.unmatched_second),
                    "unmatched_second_after_action6": int(child.match.unmatched_second),
                    "unmatched_second_delta_after_action6": int(node.match.unmatched_second)
                    - int(child.match.unmatched_second),
                }
            )
            reports.append(report)

    reports.sort(
        key=lambda item: (
            int(item.get("matched_delta_after_action6", 0)),
            int(item.get("matched_pairs_after_action6", 0)),
            int(item.get("dotted_delta_after_action6", 0)),
            int(item.get("spatial_operator_change", 0)),
            float(item.get("search_score", 0.0)),
        ),
        reverse=True,
    )
    return reports


def _node_global_score(node: GuidedNode) -> float:
    return (
        float(node.global_correspondence.score)
        if node.global_correspondence is not None
        else float("-inf")
    )


def _node_breaker_metrics(node: GuidedNode) -> Dict[str, Any]:
    features = (node.auto_levelup_state or {}).get("features") or {}
    second_count = float(features.get("second_count", 0.0))
    second_largest = float(features.get("second_largest_size", 0.0))
    largest_offender = float(features.get("largest_offender_size", 0.0))
    return {
        "actions": list(node.actions),
        "depth": int(node.depth),
        "level": int(node.level),
        "state": node.state,
        "grid_hash": node.grid_hash,
        "search_score": round(float(node.search_score), 4),
        "auto_score": round(float((node.auto_levelup_state or {}).get("score", 0.0)), 4),
        "global_score": round(float(_node_global_score(node)), 4),
        "match_score": round(float(node.match.score), 4),
        "dotted_violations": int(node.match.dotted_constraint_violations),
        "second_count": round(float(second_count), 4),
        "component_count_11": round(float(second_count), 4),
        "second_largest_size": round(float(second_largest), 4),
        "largest_offender_size": round(float(largest_offender), 4),
        "second_size_sum": round(float(features.get("second_size_sum", 0.0)), 4),
        "unmatched_total": round(float(features.get("unmatched_total", 0.0)), 4),
        "guard_pass": bool((node.fragmentation_guardrails or {}).get("passes")),
        "guard_penalty": round(float((node.fragmentation_guardrails or {}).get("penalty", 0.0)), 4),
        "reached_next_level": bool(node.reached_next_level),
        "died": bool(node.died),
    }


def _component_report(component: Optional[Component]) -> Optional[Dict[str, Any]]:
    return component.to_report() if component is not None else None


def _component_center(component: Optional[Component]) -> Tuple[float, float]:
    if component is None:
        return (0.0, 0.0)
    return (float(component.centroid_y), float(component.centroid_x))


def _boundary_contact_report(component: Optional[Component], shape: Tuple[int, int]) -> Dict[str, bool]:
    if component is None:
        return {"top": False, "bottom": False, "left": False, "right": False}
    height, width = shape
    return {
        "top": int(component.min_y) <= 0,
        "bottom": int(component.max_y) >= height - 1,
        "left": int(component.min_x) <= 0,
        "right": int(component.max_x) >= width - 1,
    }


def _cursor_to_largest_distance(
    cursor_components: Sequence[Component],
    largest: Optional[Component],
) -> float:
    if largest is None or not cursor_components:
        return 999.0
    ly, lx = _component_center(largest)
    return min(
        float(np.hypot(float(component.centroid_y) - ly, float(component.centroid_x) - lx))
        for component in cursor_components
    )


def _tail_repeat_len(actions: Sequence[str]) -> int:
    if not actions:
        return 0
    last = actions[-1]
    count = 0
    for action in reversed(actions):
        if action != last:
            break
        count += 1
    return count


def _tail_overlap_score(left: Sequence[str], right: Sequence[str], *, window: int = 4) -> float:
    left_tail = list(left[-window:])
    right_tail = list(right[-window:])
    if not left_tail or not right_tail:
        return 0.0
    size = min(len(left_tail), len(right_tail))
    matches = sum(1 for a, b in zip(left_tail[-size:], right_tail[-size:]) if a == b)
    return float(matches) / float(max(1, size))


def _node_break_context_features(
    node: GuidedNode,
    *,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    grid = node.grid
    if grid is None and node.env is not None:
        grid = _primary_grid(getattr(node.env, "observation_space", None))
    if grid is None:
        grid = [[0]]
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    second_components = sorted(
        _connected_components_for_colors(grid, [pair_colors[1]]),
        key=lambda item: item.size,
        reverse=True,
    )
    cursor_components = sorted(
        _connected_components_for_colors(grid, (4, 5)),
        key=lambda item: item.size,
        reverse=True,
    )
    largest = second_components[0] if second_components else None
    features = (node.auto_levelup_state or {}).get("features") or {}
    cursor_distance = _cursor_to_largest_distance(cursor_components[:8], largest)
    return {
        "available": True,
        "actions_tail": list(node.actions[-8:]),
        "tail_repeat_action": node.actions[-1] if node.actions else None,
        "tail_repeat_len": int(_tail_repeat_len(node.actions)),
        "numeric": {
            "second_largest_size": float(features.get("second_largest_size", 0.0)),
            "second_count": float(features.get("second_count", 0.0)),
            "largest_offender_size": float(features.get("largest_offender_size", 0.0)),
            "unmatched_total": float(features.get("unmatched_total", 0.0)),
            "dotted_violations": float(features.get("dotted_violations", 0.0)),
            "nearest_distance_mean": float(features.get("nearest_distance_mean", 0.0)),
            "global_score": float(_node_global_score(node)),
        },
        "largest_second": _component_report(largest),
        "largest_second_boundary_contact": _boundary_contact_report(largest, shape),
        "cursor_to_largest_distance": round(float(cursor_distance), 4),
        "cursor_components": [_component_report(component) for component in cursor_components[:4]],
    }


def _bbox_center_from_report(component_report: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if not component_report:
        return (0.0, 0.0)
    centroid = component_report.get("centroid") or {}
    if "y" in centroid and "x" in centroid:
        return (float(centroid.get("y", 0.0)), float(centroid.get("x", 0.0)))
    bbox = component_report.get("bbox") or component_report
    return (
        0.5 * (float(bbox.get("min_y", 0.0)) + float(bbox.get("max_y", 0.0))),
        0.5 * (float(bbox.get("min_x", 0.0)) + float(bbox.get("max_x", 0.0))),
    )


def _load_break_context_model(
    path: Optional[Path],
    *,
    top_n: int = 32,
) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {
            "available": False,
            "reason": "missing_human_break_report",
            "path": str(path) if path is not None else None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_items = list(payload.get("break_transitions") or [])
    if not raw_items:
        raw_items = list(payload.get("top_largest_drops") or []) + list(payload.get("top_count_gains") or [])
    seen: Set[Tuple[int, str]] = set()
    refs: List[Dict[str, Any]] = []
    for item in sorted(
        raw_items,
        key=lambda entry: (
            float((entry.get("deltas") or {}).get("second_largest_drop", 0.0)),
            float((entry.get("deltas") or {}).get("second_count_gain", 0.0)),
            int(entry.get("changed_cells", 0)),
        ),
        reverse=True,
    ):
        key = (int(item.get("trace_index", -1)), str(item.get("action")))
        if key in seen:
            continue
        seen.add(key)
        before = item.get("features_before") or {}
        context = item.get("before_context") or {}
        largest = (context.get("second_components") or {}).get("largest")
        cursor_items = list((context.get("cursor_context") or {}).get("top") or [])
        largest_center = _bbox_center_from_report(largest)
        cursor_distance = 999.0
        if cursor_items and largest:
            cursor_distance = min(
                float(np.hypot(cy - largest_center[0], cx - largest_center[1]))
                for cy, cx in (_bbox_center_from_report(cursor) for cursor in cursor_items[:4])
            )
        refs.append(
            {
                "label": f"human_break_{item.get('trace_index')}_{item.get('action')}",
                "trace_index": int(item.get("trace_index", -1)),
                "trace_step": int(item.get("trace_step", -1)),
                "action": item.get("action"),
                "previous_actions": list(item.get("previous_actions") or []),
                "next_actions": list(item.get("next_actions") or []),
                "level_up": bool(item.get("level_up")),
                "deltas": dict(item.get("deltas") or {}),
                "numeric": {
                    "second_largest_size": float(before.get("second_largest_size", 0.0)),
                    "second_count": float(before.get("second_count", 0.0)),
                    "largest_offender_size": float(before.get("largest_offender_size", 0.0)),
                    "unmatched_total": float(before.get("unmatched_total", 0.0)),
                    "dotted_violations": float(before.get("dotted_violations", 0.0)),
                    "nearest_distance_mean": float(before.get("nearest_distance_mean", 0.0)),
                    "global_score": float(before.get("global_score", 0.0)),
                },
                "largest_second": largest,
                "largest_second_boundary_contact": dict(context.get("largest_second_boundary_contact") or {}),
                "cursor_to_largest_distance": round(float(cursor_distance), 4),
            }
        )
        if len(refs) >= int(top_n):
            break
    if not refs:
        return {
            "available": False,
            "reason": "no_human_break_references",
            "path": str(path),
        }
    return {
        "available": True,
        "path": str(path),
        "reference_count": len(refs),
        "references": refs,
    }


def _score_break_context_similarity(
    node: GuidedNode,
    *,
    model: Optional[Dict[str, Any]],
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    if not (model or {}).get("available"):
        return {
            "available": False,
            "score": 0.0,
            "reason": (model or {}).get("reason", "break_context_model_unavailable"),
        }
    node_context = _node_break_context_features(node, pair_colors=pair_colors)
    node_numeric = node_context.get("numeric") or {}
    node_largest = node_context.get("largest_second")
    node_center = _bbox_center_from_report(node_largest)
    node_boundary = node_context.get("largest_second_boundary_contact") or {}
    best: Optional[Dict[str, Any]] = None
    for ref in list((model or {}).get("references") or []):
        ref_numeric = ref.get("numeric") or {}
        numeric_specs = {
            "second_largest_size": 120.0,
            "second_count": 24.0,
            "largest_offender_size": 120.0,
            "unmatched_total": 30.0,
            "dotted_violations": 2.0,
            "nearest_distance_mean": 16.0,
            "global_score": 28.0,
        }
        numeric_distance = 0.0
        for name, scale in numeric_specs.items():
            numeric_distance += abs(
                float(node_numeric.get(name, 0.0)) - float(ref_numeric.get(name, 0.0))
            ) / max(1.0, scale)
        ref_center = _bbox_center_from_report(ref.get("largest_second"))
        bbox_distance = float(np.hypot(node_center[0] - ref_center[0], node_center[1] - ref_center[1])) / 32.0
        ref_boundary = ref.get("largest_second_boundary_contact") or {}
        boundary_matches = sum(
            1
            for side in ("top", "bottom", "left", "right")
            if bool(node_boundary.get(side)) == bool(ref_boundary.get(side))
        )
        boundary_distance = 1.0 - float(boundary_matches) / 4.0
        cursor_distance = abs(
            float(node_context.get("cursor_to_largest_distance", 999.0))
            - float(ref.get("cursor_to_largest_distance", 999.0))
        ) / 32.0
        action_tail_distance = 1.0 - _tail_overlap_score(
            list(node_context.get("actions_tail") or []),
            list(ref.get("previous_actions") or []),
            window=4,
        )
        total_distance = (
            numeric_distance
            + 1.5 * bbox_distance
            + 0.8 * boundary_distance
            + 0.7 * cursor_distance
            + 0.45 * action_tail_distance
        )
        score = 100.0 / (1.0 + total_distance)
        candidate = {
            "score": round(float(score), 4),
            "distance": round(float(total_distance), 4),
            "reference": {
                "label": ref.get("label"),
                "trace_index": ref.get("trace_index"),
                "trace_step": ref.get("trace_step"),
                "action": ref.get("action"),
                "deltas": dict(ref.get("deltas") or {}),
            },
            "distance_terms": {
                "numeric": round(float(numeric_distance), 4),
                "bbox": round(float(bbox_distance), 4),
                "boundary": round(float(boundary_distance), 4),
                "cursor": round(float(cursor_distance), 4),
                "action_tail": round(float(action_tail_distance), 4),
            },
            "node_context": node_context,
        }
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate
    return {
        "available": True,
        **(best or {"score": 0.0, "distance": 999.0}),
    }


def _biggest_second_breaker_candidate_report(
    seed: GuidedNode,
    node: GuidedNode,
    *,
    global_floor: float = 45.0,
    max_dotted: int = 2,
    target_second_largest_max: float = 200.0,
    break_context_model: Optional[Dict[str, Any]] = None,
    pair_colors: Tuple[int, int] = (10, 11),
) -> Dict[str, Any]:
    before = _node_breaker_metrics(seed)
    after = _node_breaker_metrics(node)
    delta_second_largest = float(before["second_largest_size"]) - float(after["second_largest_size"])
    delta_largest_offender = float(before["largest_offender_size"]) - float(after["largest_offender_size"])
    delta_component_count = float(after["component_count_11"]) - float(before["component_count_11"])
    passes_constraints = bool(
        not node.died
        and float(after["global_score"]) >= float(global_floor)
        and int(after["dotted_violations"]) <= int(max_dotted)
    )
    breaks_under_target = bool(
        passes_constraints
        and float(after["second_largest_size"]) < float(target_second_largest_max)
    )
    probe_actions = node.actions[len(seed.actions):]
    probe_action_data = node.action_data[len(seed.action_data):]
    seed_break_context = _score_break_context_similarity(
        seed,
        model=break_context_model,
        pair_colors=pair_colors,
    )
    after_break_context = _score_break_context_similarity(
        node,
        model=break_context_model,
        pair_colors=pair_colors,
    )
    return {
        "seed_actions": list(seed.actions),
        "probe_actions": list(probe_actions),
        "probe_action_data": list(probe_action_data),
        "actions": list(node.actions),
        "before": before,
        "after": after,
        "delta_second_largest_size": round(float(delta_second_largest), 4),
        "delta_largest_offender_size": round(float(delta_largest_offender), 4),
        "delta_component_count_11": round(float(delta_component_count), 4),
        "delta_second_count": round(float(delta_component_count), 4),
        "constraints": {
            "global_floor": round(float(global_floor), 4),
            "max_dotted": int(max_dotted),
            "target_second_largest_max": round(float(target_second_largest_max), 4),
        },
        "passes_constraints": passes_constraints,
        "breaks_under_target": breaks_under_target,
        "seed_break_context": seed_break_context,
        "after_break_context": after_break_context,
        "break_context_score": round(float(after_break_context.get("score", 0.0)), 4),
        "reached_next_level": bool(node.reached_next_level),
        "died": bool(node.died),
    }


def _breaker_report_rank_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    after = item.get("after") or {}
    return (
        bool(item.get("reached_next_level")),
        bool(item.get("breaks_under_target")),
        bool(item.get("passes_constraints")),
        float(item.get("delta_second_largest_size", 0.0)),
        float(item.get("delta_largest_offender_size", 0.0)),
        float(item.get("delta_component_count_11", 0.0)),
        float(item.get("break_context_score", 0.0)),
        -float(after.get("dotted_violations", 99)),
        float(after.get("global_score", float("-inf"))),
        float(after.get("auto_score", 0.0)),
    )


def _breaker_node_rank_key(
    node: GuidedNode,
    *,
    break_context_model: Optional[Dict[str, Any]] = None,
    pair_colors: Tuple[int, int] = (10, 11),
) -> Tuple[Any, ...]:
    metrics = _node_breaker_metrics(node)
    context_score = 0.0
    if (break_context_model or {}).get("available"):
        context_score = float(
            _score_break_context_similarity(
                node,
                model=break_context_model,
                pair_colors=pair_colors,
            ).get("score", 0.0)
        )
    return (
        bool(metrics.get("guard_pass")),
        context_score,
        -float(metrics.get("second_largest_size", 0.0)),
        float(metrics.get("auto_score", 0.0)),
        float((node.fragmentation_guidance or {}).get("score", 0.0)),
        float(node.search_score),
    )


def _probe_biggest_second_component_breaker(
    nodes: Sequence[GuidedNode],
    *,
    full_game_id: str,
    rule_model: RuleModel,
    root_match: MatchScore,
    danger_actions: Sequence[str],
    include_action6: bool,
    action6_targets: Sequence[Dict[str, int]],
    scoring_mode: str,
    submit_gate: str,
    top_k: int,
    depth: int,
    beam_width: int,
    global_floor: float = 45.0,
    max_dotted: int = 2,
    target_second_largest_max: float = 200.0,
    stages: int = 1,
    stage_targets: Sequence[float] = (),
    disable_action2: bool = False,
    action_ontology_model: Optional[Dict[str, Any]] = None,
    action_ontology_weight: float = 1.0,
    auto_levelup_state_classifier: Optional[Dict[str, Any]] = None,
    auto_levelup_classifier_weight: float = 1.0,
    largest_second_pressure: bool = False,
    break_context_model: Optional[Dict[str, Any]] = None,
    return_nodes: bool = False,
) -> Dict[str, Any]:
    """Find the local operator that breaks the largest color-11 component."""

    if top_k <= 0 or depth <= 0 or beam_width <= 0:
        return {
            "enabled": False,
            "reason": "disabled",
            "include_action6": bool(include_action6),
            "break_context_model": {
                "available": bool((break_context_model or {}).get("available")),
                "path": (break_context_model or {}).get("path"),
                "reference_count": int((break_context_model or {}).get("reference_count") or 0),
            },
            "seed_count": 0,
            "evaluated_branches": 0,
            "candidate_count": 0,
            "passing_candidates": 0,
            "reaches_under_target": False,
        }

    seed_pool: List[GuidedNode] = []
    seen: Set[str] = set()
    for node in sorted(
        nodes,
        key=lambda item: _breaker_node_rank_key(
            item,
            break_context_model=break_context_model,
            pair_colors=rule_model.pair_colors,
        ),
        reverse=True,
    ):
        if len(seed_pool) >= int(top_k):
            break
        if node.died or node.reached_next_level or node.env is None:
            continue
        if node.grid_hash in seen:
            continue
        metrics = _node_breaker_metrics(node)
        if float(metrics["global_score"]) < float(global_floor):
            continue
        if int(metrics["dotted_violations"]) > int(max_dotted):
            continue
        seen.add(node.grid_hash)
        seed_pool.append(node)

    if not seed_pool:
        return {
            "enabled": True,
            "reason": "no_guarded_seed",
            "include_action6": bool(include_action6),
            "break_context_model": {
                "available": bool((break_context_model or {}).get("available")),
                "path": (break_context_model or {}).get("path"),
                "reference_count": int((break_context_model or {}).get("reference_count") or 0),
            },
            "seed_count": 0,
            "evaluated_branches": 0,
            "candidate_count": 0,
            "passing_candidates": 0,
            "reaches_under_target": False,
            "constraints": {
                "global_floor": round(float(global_floor), 4),
                "max_dotted": int(max_dotted),
                "target_second_largest_max": round(float(target_second_largest_max), 4),
            },
        }

    stage_count = max(1, int(stages))
    targets = [float(value) for value in stage_targets if value is not None]
    if not targets:
        targets = [float(target_second_largest_max)]
    while len(targets) < stage_count:
        targets.append(float(target_second_largest_max))
    targets = targets[:stage_count]

    all_node_reports: List[Tuple[GuidedNode, Dict[str, Any]]] = []
    stage_summaries: List[Dict[str, Any]] = []
    evaluated = 0
    pruned_death = 0
    pruned_constraints = 0
    current_stage_seeds: List[GuidedNode] = list(seed_pool)

    for stage_index in range(stage_count):
        stage_target = float(targets[stage_index])
        stage_reports: List[Dict[str, Any]] = []
        stage_node_reports: List[Tuple[GuidedNode, Dict[str, Any]]] = []
        stage_evaluated = 0
        stage_pruned_death = 0
        stage_pruned_constraints = 0
        frontier: List[GuidedNode] = list(current_stage_seeds)
        stage_seed_pool = list(current_stage_seeds)

        for _depth in range(1, max(1, int(depth)) + 1):
            next_nodes: List[Tuple[GuidedNode, Dict[str, Any]]] = []
            for parent in frontier:
                if parent.died or parent.reached_next_level or parent.env is None:
                    continue
                parent_grid = _primary_grid(getattr(parent.env, "observation_space", None))
                for action in _allowed_actions(getattr(parent.env, "observation_space", None), include_action6=include_action6):
                    if disable_action2 and action == "ACTION2":
                        continue
                    for action_data in _action_variants(
                        action,
                        include_action6=include_action6,
                        action6_targets=action6_targets,
                    ):
                        branch_env = copy.deepcopy(parent.env)
                        raw = _step_branch(
                            branch_env,
                            full_game_id=full_game_id,
                            action=action,
                            action_data=action_data,
                        )
                        child = _make_guided_child(
                            parent,
                            raw=raw,
                            env=branch_env,
                            action=action,
                            action_data=action_data,
                            rule_model=rule_model,
                            root_match=root_match,
                            parent_grid=parent_grid,
                            danger_actions=danger_actions,
                            scoring_mode=scoring_mode,
                            submit_gate=submit_gate,
                            action_ontology_model=action_ontology_model,
                            action_ontology_weight=action_ontology_weight,
                            auto_levelup_state_classifier=auto_levelup_state_classifier,
                            auto_levelup_classifier_weight=auto_levelup_classifier_weight,
                            largest_second_pressure=largest_second_pressure,
                        )
                        evaluated += 1
                        stage_evaluated += 1
                        seed = stage_seed_pool[0]
                        for candidate_seed in stage_seed_pool:
                            if len(parent.actions) >= len(candidate_seed.actions) and (
                                parent.actions[: len(candidate_seed.actions)] == candidate_seed.actions
                            ):
                                seed = candidate_seed
                                break
                        report = _biggest_second_breaker_candidate_report(
                            seed,
                            child,
                            global_floor=global_floor,
                            max_dotted=max_dotted,
                            target_second_largest_max=stage_target,
                            break_context_model=break_context_model,
                            pair_colors=rule_model.pair_colors,
                        )
                        report["stage"] = int(stage_index + 1)
                        report["stage_target_second_largest_max"] = round(float(stage_target), 4)
                        stage_reports.append(report)
                        stage_node_reports.append((child, report))
                        all_node_reports.append((child, report))
                        if child.died:
                            pruned_death += 1
                            stage_pruned_death += 1
                        if not report["passes_constraints"]:
                            pruned_constraints += 1
                            stage_pruned_constraints += 1
                            continue
                        next_nodes.append((child, report))

            next_nodes.sort(key=lambda item: _breaker_report_rank_key(item[1]), reverse=True)
            frontier = [node for node, _report in next_nodes[: max(1, int(beam_width))]]
            if not frontier:
                break

        stage_reports.sort(key=_breaker_report_rank_key, reverse=True)
        stage_passing = [item for item in stage_reports if item.get("passes_constraints")]
        stage_under_target = [item for item in stage_passing if item.get("breaks_under_target")]
        stage_by_delta = sorted(
            stage_reports,
            key=lambda item: (
                bool(item.get("passes_constraints")),
                float(item.get("delta_second_largest_size", 0.0)),
                float(item.get("delta_largest_offender_size", 0.0)),
                float(item.get("delta_component_count_11", 0.0)),
                float(item.get("break_context_score", 0.0)),
            ),
            reverse=True,
        )
        stage_passing_by_delta = [
            item for item in stage_by_delta if item.get("passes_constraints")
        ]
        stage_summaries.append(
            {
                "stage": int(stage_index + 1),
                "target_second_largest_max": round(float(stage_target), 4),
                "seed_count": len(stage_seed_pool),
                "evaluated_branches": int(stage_evaluated),
                "candidate_count": len(stage_reports),
                "passing_candidates": len(stage_passing),
                "reaches_under_target": bool(stage_under_target),
                "pruned_death": int(stage_pruned_death),
                "pruned_constraints": int(stage_pruned_constraints),
                "seeds": [_node_breaker_metrics(seed) for seed in stage_seed_pool],
                "best": stage_reports[0] if stage_reports else None,
                "best_passing": stage_passing[0] if stage_passing else None,
                "best_delta": stage_by_delta[0] if stage_by_delta else None,
                "best_passing_delta": stage_passing_by_delta[0] if stage_passing_by_delta else None,
                "best_under_target": stage_under_target[0] if stage_under_target else None,
                "top_delta": stage_by_delta[:8],
            }
        )

        if stage_index >= stage_count - 1:
            break

        next_seed_pairs = sorted(
            stage_node_reports,
            key=lambda item: (
                bool(item[1].get("breaks_under_target")),
                bool(item[1].get("passes_constraints")),
                float(item[1].get("delta_second_largest_size", 0.0)),
                -float((item[1].get("after") or {}).get("second_largest_size", 9999.0)),
                float(item[1].get("break_context_score", 0.0)),
            ),
            reverse=True,
        )
        next_seeds: List[GuidedNode] = []
        seen_next: Set[str] = set()
        for node, report in next_seed_pairs:
            if len(next_seeds) >= max(1, int(top_k)):
                break
            if not report.get("passes_constraints"):
                continue
            if node.died or node.reached_next_level or node.env is None:
                continue
            if node.grid_hash in seen_next:
                continue
            seen_next.add(node.grid_hash)
            next_seeds.append(node)
        current_stage_seeds = next_seeds
        if not current_stage_seeds:
            break

    reports = [report for _node, report in all_node_reports]

    reports.sort(key=_breaker_report_rank_key, reverse=True)
    passing = [item for item in reports if item.get("passes_constraints")]
    under_target = [item for item in passing if item.get("breaks_under_target")]
    final_stage_index = int(stage_summaries[-1]["stage"]) if stage_summaries else 1
    final_stage_reports = [
        item for item in reports if int(item.get("stage") or 1) == final_stage_index
    ]
    final_stage_passing = [
        item for item in final_stage_reports if item.get("passes_constraints")
    ]
    final_under_target = [
        item for item in final_stage_passing if item.get("breaks_under_target")
    ]
    by_largest_drop = sorted(
        reports,
        key=lambda item: (
            bool(item.get("passes_constraints")),
            float(item.get("delta_second_largest_size", 0.0)),
            float(item.get("delta_largest_offender_size", 0.0)),
            float(item.get("delta_component_count_11", 0.0)),
            float(item.get("break_context_score", 0.0)),
        ),
        reverse=True,
    )
    passing_by_largest_drop = [item for item in by_largest_drop if item.get("passes_constraints")]

    action_breakdown: Dict[str, Dict[str, Any]] = {}
    for item in reports:
        actions = sorted(set(str(action) for action in item.get("probe_actions", []) if action))
        for action in actions:
            summary = action_breakdown.setdefault(
                action,
                {
                    "count": 0,
                    "passing": 0,
                    "best_delta_second_largest_size": float("-inf"),
                    "best": None,
                },
            )
            summary["count"] += 1
            if item.get("passes_constraints"):
                summary["passing"] += 1
            if float(item.get("delta_second_largest_size", 0.0)) > float(
                summary["best_delta_second_largest_size"]
            ):
                after = item.get("after") or {}
                summary["best_delta_second_largest_size"] = float(item.get("delta_second_largest_size", 0.0))
                summary["best"] = {
                    "probe_actions": list(item.get("probe_actions", [])),
                    "probe_action_data": list(item.get("probe_action_data", [])),
                    "delta_second_largest_size": item.get("delta_second_largest_size"),
                    "delta_component_count_11": item.get("delta_component_count_11"),
                    "second_largest_size": after.get("second_largest_size"),
                    "largest_offender_size": after.get("largest_offender_size"),
                    "component_count_11": after.get("component_count_11"),
                    "global_score": after.get("global_score"),
                    "dotted_violations": after.get("dotted_violations"),
                    "passes_constraints": bool(item.get("passes_constraints")),
                    "breaks_under_target": bool(item.get("breaks_under_target")),
                }
    action_breakdown = {
        action: {
            **summary,
            "best_delta_second_largest_size": (
                None
                if summary["best_delta_second_largest_size"] == float("-inf")
                else round(float(summary["best_delta_second_largest_size"]), 4)
            ),
        }
        for action, summary in sorted(action_breakdown.items())
    }
    result = {
        "enabled": True,
        "include_action6": bool(include_action6),
        "break_context_model": {
            "available": bool((break_context_model or {}).get("available")),
            "path": (break_context_model or {}).get("path"),
            "reference_count": int((break_context_model or {}).get("reference_count") or 0),
        },
        "seed_count": len(seed_pool),
        "evaluated_branches": int(evaluated),
        "candidate_count": len(reports),
        "passing_candidates": len(passing),
        "reaches_level_8": any(item.get("reached_next_level") for item in reports),
        "reaches_any_stage_target": bool(under_target),
        "reaches_under_target": bool(final_under_target),
        "pruned_death": int(pruned_death),
        "pruned_constraints": int(pruned_constraints),
        "depth": int(depth),
        "beam_width": int(beam_width),
        "stages": int(stage_count),
        "stage_targets": [round(float(item), 4) for item in targets],
        "constraints": {
            "global_floor": round(float(global_floor), 4),
            "max_dotted": int(max_dotted),
            "target_second_largest_max": round(float(target_second_largest_max), 4),
        },
        "stage_summaries": stage_summaries,
        "action_breakdown": action_breakdown,
        "seeds": [_node_breaker_metrics(seed) for seed in seed_pool],
        "best": reports[0] if reports else None,
        "best_passing": passing[0] if passing else None,
        "best_delta": by_largest_drop[0] if by_largest_drop else None,
        "best_passing_delta": passing_by_largest_drop[0] if passing_by_largest_drop else None,
        "best_under_target": final_under_target[0] if final_under_target else None,
        "best_any_stage_under_target": under_target[0] if under_target else None,
        "top": reports[:12],
        "top_passing": passing[:12],
        "top_delta": by_largest_drop[:12],
        "top_passing_delta": passing_by_largest_drop[:12],
    }
    if return_nodes:
        # Collect top under-target nodes (final stage first, then any stage)
        under_target_nodes: List[GuidedNode] = []
        seen_ut: Set[str] = set()
        for node, report in all_node_reports:
            if not report.get("breaks_under_target"):
                continue
            if not report.get("passes_constraints"):
                continue
            if node.died or node.reached_next_level or node.env is None:
                continue
            if node.grid_hash in seen_ut:
                continue
            seen_ut.add(node.grid_hash)
            under_target_nodes.append(node)
        # Collect top-passing nodes by delta (even if not under target)
        passing_nodes: List[GuidedNode] = []
        seen_pn: Set[str] = set(seen_ut)
        for node, report in sorted(
            all_node_reports,
            key=lambda item: (
                bool(item[1].get("passes_constraints")),
                float(item[1].get("delta_second_largest_size", 0.0)),
                float(item[1].get("delta_largest_offender_size", 0.0)),
            ),
            reverse=True,
        ):
            if len(passing_nodes) >= 12:
                break
            if not report.get("passes_constraints"):
                continue
            if node.died or node.reached_next_level or node.env is None:
                continue
            if node.grid_hash in seen_pn:
                continue
            seen_pn.add(node.grid_hash)
            passing_nodes.append(node)
        # Collect ALL top-delta nodes regardless of constraints (best morphological progress)
        top_delta_nodes: List[GuidedNode] = []
        seen_td: Set[str] = set(seen_ut) | set(n.grid_hash for n in passing_nodes)
        for node, report in sorted(
            all_node_reports,
            key=lambda item: (
                float(item[1].get("delta_second_largest_size", 0.0)),
                float(item[1].get("delta_largest_offender_size", 0.0)),
                float(item[1].get("delta_component_count_11", 0.0)),
            ),
            reverse=True,
        ):
            if len(top_delta_nodes) >= 12:
                break
            if node.died or node.env is None:
                continue
            if node.grid_hash in seen_td:
                continue
            seen_td.add(node.grid_hash)
            top_delta_nodes.append(node)
        result["_under_target_nodes"] = under_target_nodes
        result["_passing_nodes"] = passing_nodes
        result["_top_delta_nodes"] = top_delta_nodes
    return result


def _repair_delta_report(seed: Optional[GuidedNode], repaired: Optional[GuidedNode]) -> Optional[Dict[str, Any]]:
    if seed is None or repaired is None:
        return None
    return {
        "global_delta": round(float(_node_global_score(repaired) - _node_global_score(seed)), 4),
        "match_delta": round(float(repaired.match.score - seed.match.score), 4),
        "unmatched_delta": int(_unmatched_total(seed.match) - _unmatched_total(repaired.match)),
        "dotted_delta": int(seed.match.dotted_constraint_violations)
        - int(repaired.match.dotted_constraint_violations),
        "readiness_before": bool(seed.readiness_gate_passed),
        "readiness_after": bool(repaired.readiness_gate_passed),
    }


def _run_local_repair_from_best_global(
    nodes: Sequence[GuidedNode],
    *,
    full_game_id: str,
    rule_model: RuleModel,
    root_match: MatchScore,
    danger_actions: Sequence[str],
    include_action6: bool,
    action6_targets: Sequence[Dict[str, int]],
    submit_gate: str,
    top_k: int,
    horizon: int,
    beam_width: int,
    start_global_min: float,
    global_floor: float,
    submit_probe_limit: int,
) -> Dict[str, Any]:
    """Repair locally from high-global states while preserving the abstraction."""

    if top_k <= 0 or horizon <= 0:
        return {
            "enabled": False,
            "reason": "disabled",
            "seed_count": 0,
            "evaluated_branches": 0,
            "repair_candidates": 0,
            "reaches_level_8": False,
        }

    seeds: List[GuidedNode] = []
    seen_seeds: Set[str] = set()
    for node in sorted(nodes, key=lambda item: (_node_global_score(item), item.search_score), reverse=True):
        if len(seeds) >= top_k:
            break
        if node.died or node.reached_next_level or node.env is None:
            continue
        if node.grid_hash in seen_seeds:
            continue
        if _node_global_score(node) < float(start_global_min):
            continue
        seen_seeds.add(node.grid_hash)
        seeds.append(node)

    if not seeds:
        return {
            "enabled": True,
            "reason": "no_high_global_seed",
            "seed_count": 0,
            "evaluated_branches": 0,
            "repair_candidates": 0,
            "reaches_level_8": False,
            "start_global_min": round(float(start_global_min), 4),
            "global_floor": round(float(global_floor), 4),
        }

    beam = list(seeds)
    evaluated: List[GuidedNode] = []
    repair_candidates: List[GuidedNode] = []
    global_floor_skips = 0
    nonrepair_skips = 0
    death_skips = 0
    action2_skips = 0

    for _depth in range(1, max(1, horizon) + 1):
        expanded: List[GuidedNode] = []
        for node in beam:
            if node.died or node.reached_next_level or node.env is None:
                continue
            parent_grid = _primary_grid(getattr(node.env, "observation_space", None))
            for action in _allowed_actions(getattr(node.env, "observation_space", None), include_action6=include_action6):
                if action == "ACTION2":
                    action2_skips += 1
                    continue
                for action_data in _action_variants(
                    action,
                    include_action6=include_action6,
                    action6_targets=action6_targets,
                ):
                    branch_env = copy.deepcopy(node.env)
                    raw = _step_branch(
                        branch_env,
                        full_game_id=full_game_id,
                        action=action,
                        action_data=action_data,
                    )
                    child = _make_guided_child(
                        node,
                        raw=raw,
                        env=branch_env,
                        action=action,
                        action_data=action_data,
                        rule_model=rule_model,
                        root_match=root_match,
                        parent_grid=parent_grid,
                        danger_actions=danger_actions,
                        scoring_mode="local_repair",
                        submit_gate=submit_gate,
                    )
                    evaluated.append(child)
                    if child.died:
                        death_skips += 1
                        continue
                    child_global = _node_global_score(child)
                    if not child.reached_next_level and child_global < float(global_floor):
                        global_floor_skips += 1
                        continue
                    unmatched_improved = _unmatched_total(child.match) < _unmatched_total(node.match)
                    dotted_improved = (
                        int(child.match.dotted_constraint_violations)
                        < int(node.match.dotted_constraint_violations)
                    )
                    semantic_improved = child.match.score > node.match.score + 0.001
                    readiness_improved = child.readiness_gate_passed and not node.readiness_gate_passed
                    if not (
                        child.reached_next_level
                        or unmatched_improved
                        or dotted_improved
                        or semantic_improved
                        or readiness_improved
                    ):
                        nonrepair_skips += 1
                        continue
                    repair_candidates.append(child)
                    expanded.append(child)
        if not expanded:
            break
        expanded.sort(key=lambda item: item.search_score, reverse=True)
        beam = expanded[: max(1, beam_width)]

    safe_candidates = [node for node in repair_candidates if not node.died]
    best_seed = max(seeds, key=lambda item: (_node_global_score(item), item.search_score), default=None)
    best_repair = max(safe_candidates, key=lambda item: item.search_score, default=None)
    best_ready = max(
        (node for node in safe_candidates if node.readiness_gate_passed),
        key=lambda item: item.search_score,
        default=None,
    )
    probe_source = sorted(
        safe_candidates or seeds,
        key=lambda item: (
            int(item.readiness_gate_passed),
            item.search_score,
            _node_global_score(item),
        ),
        reverse=True,
    )
    submit_probes = _probe_action2_from_best_match_nodes(
        probe_source,
        full_game_id=full_game_id,
        rule_model=rule_model,
        root_match=root_match,
        danger_actions=danger_actions,
        scoring_mode="local_repair",
        submit_gate=submit_gate,
        limit=submit_probe_limit,
    )
    reaches_next = any(node.reached_next_level for node in evaluated) or any(
        node.reached_next_level for node in submit_probes
    )
    top_repair = sorted(safe_candidates, key=lambda item: item.search_score, reverse=True)[:10]
    top_ready = sorted(
        (node for node in safe_candidates if node.readiness_gate_passed),
        key=lambda item: item.search_score,
        reverse=True,
    )[:10]
    return {
        "enabled": True,
        "reason": "ok",
        "top_k": int(top_k),
        "horizon": int(horizon),
        "beam_width": int(beam_width),
        "start_global_min": round(float(start_global_min), 4),
        "global_floor": round(float(global_floor), 4),
        "seed_count": len(seeds),
        "seed_hashes": [node.grid_hash for node in seeds],
        "seed_globals": [
            round(float(_node_global_score(node)), 4)
            for node in seeds
        ],
        "best_seed": best_seed.to_report() if best_seed else None,
        "evaluated_branches": len(evaluated),
        "repair_candidates": len(safe_candidates),
        "readiness_gate_states": sum(1 for node in safe_candidates if node.readiness_gate_passed),
        "global_floor_skips": int(global_floor_skips),
        "nonrepair_skips": int(nonrepair_skips),
        "death_skips": int(death_skips),
        "action2_skips": int(action2_skips),
        "reaches_level_8": bool(reaches_next),
        "best_repair": best_repair.to_report() if best_repair else None,
        "best_repair_delta_from_best_seed": _repair_delta_report(best_seed, best_repair),
        "best_ready": best_ready.to_report() if best_ready else None,
        "best_ready_delta_from_best_seed": _repair_delta_report(best_seed, best_ready),
        "submit_probes": [node.to_report() for node in submit_probes],
        "submit_probe_summary": {
            "count": len(submit_probes),
            "reaches_level_8": any(node.reached_next_level for node in submit_probes),
            "wins": any(node.state == "WIN" for node in submit_probes),
            "game_over": sum(1 for node in submit_probes if node.died),
            "not_ready": sum(1 for node in submit_probes if node.submit_not_ready),
            "best": (
                max(
                    submit_probes,
                    key=lambda item: (int(item.reached_next_level), item.search_score),
                ).to_report()
                if submit_probes
                else None
            ),
        },
        "top_repair": [node.to_report() for node in top_repair],
        "top_ready": [node.to_report() for node in top_ready],
    }


def _report_with_targeted_info(node: GuidedNode, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    report = node.to_report()
    if info is not None:
        report["targeted_repair"] = dict(info)
    return report


def _run_targeted_repair_from_best_global(
    nodes: Sequence[GuidedNode],
    *,
    full_game_id: str,
    rule_model: RuleModel,
    root_match: MatchScore,
    danger_actions: Sequence[str],
    include_action6: bool,
    action6_targets: Sequence[Dict[str, int]],
    submit_gate: str,
    top_k: int,
    horizon: int,
    beam_width: int,
    start_global_min: float,
    global_floor: float,
    submit_probe_limit: int,
    max_target_components: int,
) -> Dict[str, Any]:
    """Targeted local repair: only keep actions that touch diagnosed offenders."""

    if top_k <= 0 or horizon <= 0:
        return {
            "enabled": False,
            "reason": "disabled",
            "seed_count": 0,
            "evaluated_branches": 0,
            "targeted_candidates": 0,
            "reaches_level_8": False,
        }

    seeds: List[GuidedNode] = []
    seen_seeds: Set[str] = set()
    for node in sorted(nodes, key=lambda item: (_node_global_score(item), item.search_score), reverse=True):
        if len(seeds) >= top_k:
            break
        if node.died or node.reached_next_level or node.env is None:
            continue
        if node.grid_hash in seen_seeds:
            continue
        if _node_global_score(node) < float(start_global_min):
            continue
        seen_seeds.add(node.grid_hash)
        seeds.append(node)

    if not seeds:
        return {
            "enabled": True,
            "reason": "no_high_global_seed",
            "seed_count": 0,
            "evaluated_branches": 0,
            "targeted_candidates": 0,
            "reaches_level_8": False,
            "start_global_min": round(float(start_global_min), 4),
            "global_floor": round(float(global_floor), 4),
        }

    seed_diagnostics: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        seed_grid = seed.grid or _primary_grid(getattr(seed.env, "observation_space", None))
        seed_diagnostics[seed.grid_hash] = _offending_component_diagnostics(
            seed_grid,
            pair_colors=rule_model.pair_colors,
            limit=max_target_components,
        )

    beam = list(seeds)
    evaluated: List[GuidedNode] = []
    targeted_candidates: List[GuidedNode] = []
    targeted_info: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = {}
    global_floor_skips = 0
    no_target_skips = 0
    untouched_skips = 0
    nonrepair_skips = 0
    death_skips = 0
    action2_skips = 0

    for _depth in range(1, max(1, horizon) + 1):
        expanded: List[GuidedNode] = []
        for node in beam:
            if node.died or node.reached_next_level or node.env is None:
                continue
            parent_grid = _primary_grid(getattr(node.env, "observation_space", None))
            parent_diag = _offending_component_diagnostics(
                parent_grid,
                pair_colors=rule_model.pair_colors,
                limit=max_target_components,
            )
            targets = list(parent_diag.get("target_components") or [])
            target_bboxes = [target.get("bbox") or target for target in targets]
            if not target_bboxes:
                no_target_skips += 1
                continue
            targeted_action6_targets = _target_action6_candidates_from_components(
                targets,
                grid=parent_grid,
                max_candidates=max(1, len(action6_targets) or 12),
            )
            for action in _allowed_actions(getattr(node.env, "observation_space", None), include_action6=include_action6):
                if action == "ACTION2":
                    action2_skips += 1
                    continue
                variants = (
                    [dict(target) for target in (targeted_action6_targets or action6_targets)]
                    if action == "ACTION6"
                    else [None]
                )
                for action_data in variants:
                    branch_env = copy.deepcopy(node.env)
                    raw = _step_branch(
                        branch_env,
                        full_game_id=full_game_id,
                        action=action,
                        action_data=action_data,
                    )
                    child_grid = _primary_grid(raw) if raw is not None else parent_grid
                    action_bbox = _diff_bbox(parent_grid, child_grid)
                    touched_targets = [
                        target
                        for target in targets
                        if _bbox_intersects(action_bbox, target.get("bbox") or target, margin=2)
                    ]
                    touched_largest = bool(
                        parent_diag.get("largest_offender")
                        and _bbox_intersects(
                            action_bbox,
                            (parent_diag.get("largest_offender") or {}).get("bbox"),
                            margin=2,
                        )
                    )
                    child = _make_guided_child(
                        node,
                        raw=raw,
                        env=branch_env,
                        action=action,
                        action_data=action_data,
                        rule_model=rule_model,
                        root_match=root_match,
                        parent_grid=parent_grid,
                        danger_actions=danger_actions,
                        scoring_mode="local_repair",
                        submit_gate=submit_gate,
                    )
                    evaluated.append(child)
                    if child.died:
                        death_skips += 1
                        continue
                    if not child.reached_next_level and not touched_targets:
                        untouched_skips += 1
                        continue
                    child_global = _node_global_score(child)
                    if not child.reached_next_level and child_global < float(global_floor):
                        global_floor_skips += 1
                        continue
                    child_diag = _offending_component_diagnostics(
                        child.grid or child_grid,
                        pair_colors=rule_model.pair_colors,
                        limit=max_target_components,
                    )
                    unmatched_delta = _unmatched_total(node.match) - _unmatched_total(child.match)
                    dotted_delta = (
                        int(node.match.dotted_constraint_violations)
                        - int(child.match.dotted_constraint_violations)
                    )
                    offender_delta = int(parent_diag.get("offending_count", 0)) - int(
                        child_diag.get("offending_count", 0)
                    )
                    readiness_improved = child.readiness_gate_passed and not node.readiness_gate_passed
                    semantic_improved = child.match.score > node.match.score + 0.001
                    if not (
                        child.reached_next_level
                        or unmatched_delta > 0
                        or dotted_delta > 0
                        or offender_delta > 0
                        or readiness_improved
                        or semantic_improved
                    ):
                        nonrepair_skips += 1
                        continue

                    child.search_score += (
                        700.0 * float(max(0, offender_delta))
                        + 420.0 * float(max(0, unmatched_delta))
                        + 420.0 * float(max(0, dotted_delta))
                        + 180.0 * float(touched_largest)
                    )
                    info = {
                        "action_bbox": action_bbox,
                        "touched_target_count": len(touched_targets),
                        "touched_largest_offender": bool(touched_largest),
                        "touched_reasons": sorted(
                            set(str(target.get("reason", "")) for target in touched_targets)
                        ),
                        "parent_offending_count": int(parent_diag.get("offending_count", 0)),
                        "child_offending_count": int(child_diag.get("offending_count", 0)),
                        "offender_delta": int(offender_delta),
                        "unmatched_delta": int(unmatched_delta),
                        "dotted_delta": int(dotted_delta),
                        "parent_largest_offender": parent_diag.get("largest_offender"),
                        "child_largest_offender": child_diag.get("largest_offender"),
                    }
                    key = (tuple(child.actions), child.grid_hash)
                    targeted_info[key] = info
                    targeted_candidates.append(child)
                    expanded.append(child)
        if not expanded:
            break
        expanded.sort(key=lambda item: item.search_score, reverse=True)
        beam = expanded[: max(1, beam_width)]

    safe_candidates = [node for node in targeted_candidates if not node.died]
    best_seed = max(seeds, key=lambda item: (_node_global_score(item), item.search_score), default=None)
    best_targeted = max(safe_candidates, key=lambda item: item.search_score, default=None)
    best_ready = max(
        (node for node in safe_candidates if node.readiness_gate_passed),
        key=lambda item: item.search_score,
        default=None,
    )
    probe_source = sorted(
        safe_candidates or seeds,
        key=lambda item: (
            int(item.readiness_gate_passed),
            item.search_score,
            _node_global_score(item),
        ),
        reverse=True,
    )
    submit_probes = _probe_action2_from_best_match_nodes(
        probe_source,
        full_game_id=full_game_id,
        rule_model=rule_model,
        root_match=root_match,
        danger_actions=danger_actions,
        scoring_mode="local_repair",
        submit_gate=submit_gate,
        limit=submit_probe_limit,
    )
    reaches_next = any(node.reached_next_level for node in evaluated) or any(
        node.reached_next_level for node in submit_probes
    )
    top_targeted = sorted(safe_candidates, key=lambda item: item.search_score, reverse=True)[:10]
    top_ready = sorted(
        (node for node in safe_candidates if node.readiness_gate_passed),
        key=lambda item: item.search_score,
        reverse=True,
    )[:10]

    def extra_for(node: Optional[GuidedNode]) -> Optional[Dict[str, Any]]:
        if node is None:
            return None
        return targeted_info.get((tuple(node.actions), node.grid_hash))

    return {
        "enabled": True,
        "reason": "ok",
        "top_k": int(top_k),
        "horizon": int(horizon),
        "beam_width": int(beam_width),
        "start_global_min": round(float(start_global_min), 4),
        "global_floor": round(float(global_floor), 4),
        "max_target_components": int(max_target_components),
        "seed_count": len(seeds),
        "seed_hashes": [node.grid_hash for node in seeds],
        "seed_globals": [round(float(_node_global_score(node)), 4) for node in seeds],
        "seed_diagnostics": {
            node.grid_hash: seed_diagnostics.get(node.grid_hash)
            for node in seeds
        },
        "best_seed": best_seed.to_report() if best_seed else None,
        "evaluated_branches": len(evaluated),
        "targeted_candidates": len(safe_candidates),
        "readiness_gate_states": sum(1 for node in safe_candidates if node.readiness_gate_passed),
        "global_floor_skips": int(global_floor_skips),
        "no_target_skips": int(no_target_skips),
        "untouched_skips": int(untouched_skips),
        "nonrepair_skips": int(nonrepair_skips),
        "death_skips": int(death_skips),
        "action2_skips": int(action2_skips),
        "reaches_level_8": bool(reaches_next),
        "best_targeted": _report_with_targeted_info(best_targeted, extra_for(best_targeted))
        if best_targeted
        else None,
        "best_targeted_delta_from_best_seed": _repair_delta_report(best_seed, best_targeted),
        "best_ready": _report_with_targeted_info(best_ready, extra_for(best_ready)) if best_ready else None,
        "best_ready_delta_from_best_seed": _repair_delta_report(best_seed, best_ready),
        "submit_probes": [node.to_report() for node in submit_probes],
        "submit_probe_summary": {
            "count": len(submit_probes),
            "reaches_level_8": any(node.reached_next_level for node in submit_probes),
            "wins": any(node.state == "WIN" for node in submit_probes),
            "game_over": sum(1 for node in submit_probes if node.died),
            "not_ready": sum(1 for node in submit_probes if node.submit_not_ready),
            "best": (
                max(
                    submit_probes,
                    key=lambda item: (int(item.reached_next_level), item.search_score),
                ).to_report()
                if submit_probes
                else None
            ),
        },
        "top_targeted": [
            _report_with_targeted_info(node, extra_for(node))
            for node in top_targeted
        ],
        "top_ready": [
            _report_with_targeted_info(node, extra_for(node))
            for node in top_ready
        ],
    }


def run_guided_search(
    *,
    base_env: Any,
    base_raw: Any,
    full_game_id: str,
    rule_model: RuleModel,
    horizon: int,
    beam_width: int,
    include_action6: bool,
    action6_targets: Sequence[Dict[str, int]],
    danger_actions: Sequence[str],
    scoring_mode: str = "default",  # "default" | "dotted_constraint_repair"
    submit_gate: str = "semantic",  # "strict" | "semantic" | "probe"
    max_submit_probes: int = 16,
    strong_improvement_margin: float = 15.0,
    forbid_action3_before_submit: bool = False,
    disable_action2: bool = False,
    disable_action6: bool = False,
    post_saturation_probe: bool = False,
    rewrite_saturation_action3_threshold: int = 48,
    human_w3: Optional[Dict[str, Any]] = None,
    best_match_submit_probes: int = 12,
    action6_operator_probe_nodes: int = 8,
    action6_operator_probe_total: int = 64,
    local_repair_from_best_global: bool = False,
    local_repair_top_k: int = 6,
    local_repair_horizon: int = 12,
    local_repair_beam_width: int = 12,
    local_repair_start_global_min: float = 65.0,
    local_repair_global_floor: float = 60.0,
    local_repair_submit_probes: int = 12,
    targeted_repair_from_best_global: bool = False,
    targeted_repair_top_k: int = 6,
    targeted_repair_horizon: int = 12,
    targeted_repair_beam_width: int = 12,
    targeted_repair_start_global_min: float = 65.0,
    targeted_repair_global_floor: float = 60.0,
    targeted_repair_submit_probes: int = 12,
    targeted_repair_max_components: int = 12,
    targeted_repair_include_action6: bool = False,
    action_ontology_model: Optional[Dict[str, Any]] = None,
    action_ontology_weight: float = 1.0,
    auto_levelup_state_classifier: Optional[Dict[str, Any]] = None,
    auto_levelup_classifier_weight: float = 1.0,
    largest_second_pressure: bool = False,
    probe_biggest_second_breaker: bool = False,
    breaker_top_k: int = 6,
    breaker_depth: int = 2,
    breaker_beam_width: int = 48,
    breaker_global_floor: float = 45.0,
    breaker_max_dotted: int = 2,
    breaker_target_second_largest_max: float = 200.0,
    breaker_include_action6: bool = False,
    breaker_stages: int = 1,
    breaker_stage_targets: Sequence[float] = (),
    breaker_context_model: Optional[Dict[str, Any]] = None,
    enable_breaker_macro: bool = False,
    macro_break_target: float = 200.0,
    breaker_macro_submit_probes: int = 12,
) -> Dict[str, Any]:
    root_grid = _primary_grid(base_raw)
    root_match = match_score(root_grid, pair_colors=rule_model.pair_colors)
    root_global = _cached_global_correspondence_score(root_grid, pair_colors=rule_model.pair_colors)
    root_auto_levelup_state = _score_auto_levelup_state(
        root_grid,
        pair_colors=rule_model.pair_colors,
        match=root_match,
        global_score=root_global,
        classifier=auto_levelup_state_classifier,
    )
    root_fragmentation_guidance = _fragmentation_guidance_report(
        parent_state=root_auto_levelup_state,
        child_state=root_auto_levelup_state,
        classifier=auto_levelup_state_classifier,
        largest_second_pressure=largest_second_pressure,
    )
    root_fragmentation_guardrails = _fragmentation_guardrail_report(
        child_state=root_auto_levelup_state,
        child_match=root_match,
        child_global=root_global,
    )
    root = GuidedNode(
        actions=[],
        action_data=[],
        state=_state_name(getattr(base_raw, "state", "UNKNOWN")),
        level=int(getattr(base_raw, "levels_completed", 0) or 0),
        grid_hash=_hash_grid(root_grid),
        search_score=0.0,
        match=root_match,
        depth=0,
        global_correspondence=root_global,
        auto_levelup_state=root_auto_levelup_state,
        fragmentation_guidance=root_fragmentation_guidance,
        fragmentation_guardrails=root_fragmentation_guardrails,
        available_actions=_available_names_from_raw(base_raw),
        path_hashes=[_hash_grid(root_grid)],
        env=copy.deepcopy(base_env),
        grid=root_grid,
    )

    beam = [root]
    evaluated: List[GuidedNode] = []
    gated_submit_skips = 0
    post_saturation_probe_skips = 0
    post_saturation_probe_actions = {"ACTION1", "ACTION2", "ACTION5", "ACTION7"}
    for _depth in range(1, max(1, horizon) + 1):
        expanded: List[GuidedNode] = []
        for node in beam:
            if node.died or node.reached_next_level:
                expanded.append(node)
                continue
            parent_grid = _primary_grid(getattr(node.env, "observation_space", None))
            rewrite_saturated = _rewrite_is_saturated(
                node,
                action3_threshold=rewrite_saturation_action3_threshold,
            )
            for action in _allowed_actions(getattr(node.env, "observation_space", None), include_action6=include_action6):
                if scoring_mode == "rewrite_until_saturation" and post_saturation_probe:
                    if not rewrite_saturated and action in {"ACTION2", "ACTION6"}:
                        post_saturation_probe_skips += 1
                        continue
                    if rewrite_saturated and action not in post_saturation_probe_actions:
                        post_saturation_probe_skips += 1
                        continue

                # Check ACTION2 restriction (rewrite_until_saturation mode)
                if action == "ACTION2" and disable_action2:
                    continue  # Skip ACTION2 when disabled
                
                # Check ACTION6 restriction (rewrite_until_saturation mode)
                if action == "ACTION6" and disable_action6:
                    continue  # Skip ACTION6 when disabled
                
                # Check ACTION3 restriction
                if action == "ACTION3" and forbid_action3_before_submit and not node.had_valid_submit:
                    continue  # Skip ACTION3 before first valid submit
                
                # Compute target info for cursor_target mode
                gate_target_info = None
                if submit_gate == "cursor_target":
                    gate_target_info = _compute_target_info_for_gate(node, rule_model.pair_colors)
                
                gate_result = _submit_gate_allows(
                    node,
                    rule_model=rule_model,
                    root_match=root.match,
                    gate_mode=submit_gate,
                    strong_improvement_margin=strong_improvement_margin,
                    max_submit_probes=max_submit_probes,
                    consecutive_submits=node.consecutive_submits,
                    precomputed_target_info=gate_target_info,
                )
                gate_allowed = gate_result[0]
                
                if (
                    action == "ACTION2"
                    and not gate_allowed
                    and scoring_mode not in {
                        "action_ontology_guided",
                        "auto_levelup_state_classifier",
                        "level7_fragmentation_guided",
                        "level7_fragmentation_guarded",
                    }
                    and not (scoring_mode == "rewrite_until_saturation" and post_saturation_probe and rewrite_saturated)
                ):
                    gated_submit_skips += 1
                    continue
                for action_data in _action_variants(
                    action,
                    include_action6=include_action6,
                    action6_targets=action6_targets,
                ):
                    branch_env = copy.deepcopy(node.env)
                    raw = _step_branch(
                        branch_env,
                        full_game_id=full_game_id,
                        action=action,
                        action_data=action_data,
                    )
                    child = _make_guided_child(
                        node,
                        raw=raw,
                        env=branch_env,
                        action=action,
                        action_data=action_data,
                        rule_model=rule_model,
                        root_match=root_match,
                        parent_grid=parent_grid,
                        danger_actions=danger_actions,
                        scoring_mode=scoring_mode,
                        submit_gate=submit_gate,
                        post_saturation_probe=post_saturation_probe,
                        parent_rewrite_saturated=rewrite_saturated,
                        rewrite_saturation_action3_threshold=rewrite_saturation_action3_threshold,
                        action_ontology_model=action_ontology_model,
                        action_ontology_weight=action_ontology_weight,
                        auto_levelup_state_classifier=auto_levelup_state_classifier,
                        auto_levelup_classifier_weight=auto_levelup_classifier_weight,
                        largest_second_pressure=largest_second_pressure,
                    )
                    expanded.append(child)
        expanded.sort(key=lambda item: item.search_score, reverse=True)
        evaluated.extend(expanded)
        beam = expanded[: max(1, beam_width)]

    safe = [node for node in evaluated if not node.died]
    best = max(safe or evaluated or [root], key=lambda item: item.search_score)
    best_match = max(safe or evaluated or [root], key=lambda item: item.match.score)
    best_global = max(
        safe or evaluated or [root],
        key=lambda item: (
            item.global_correspondence.score
            if item.global_correspondence is not None
            else float("-inf")
        ),
    )
    best_auto_levelup_state = max(
        safe or evaluated or [root],
        key=lambda item: float((item.auto_levelup_state or {}).get("score", 0.0)),
    )
    best_fragmentation = max(
        safe or evaluated or [root],
        key=lambda item: float((item.fragmentation_guidance or {}).get("score", 0.0)),
    )
    reaches_next = any(node.reached_next_level for node in evaluated)
    improved = [
        node
        for node in safe
        if node.match_delta_from_root > 0.25
    ]
    improved_global = [
        node
        for node in safe
        if node.global_delta_from_root > 0.25
    ]
    readiness_nodes = [
        node
        for node in safe
        if node.readiness_gate_passed
    ]
    submit_attempts = [node for node in evaluated if node.actions and node.actions[-1] == "ACTION2"]
    contradictions = [node for node in evaluated if node.action3_contradiction]
    action6_nodes = [node for node in evaluated if "ACTION6" in node.actions]
    transforming_action6_nodes = [
        node for node in action6_nodes if int(node.max_action6_spatial_change) > 0
    ]
    post_saturation_nodes = [
        node for node in evaluated if node.post_saturation_probe_phase
    ]
    best_post_saturation_probe = max(
        post_saturation_nodes,
        key=lambda item: item.search_score,
        default=None,
    )
    top_match_nodes = sorted(
        safe or evaluated or [root],
        key=lambda item: item.match.score,
        reverse=True,
    )
    top_search_nodes = sorted(
        safe or evaluated or [root],
        key=lambda item: item.search_score,
        reverse=True,
    )
    top_global_nodes = sorted(
        safe or evaluated or [root],
        key=lambda item: (
            item.global_correspondence.score
            if item.global_correspondence is not None
            else float("-inf")
        ),
        reverse=True,
    )
    if scoring_mode == "global_correspondence":
        probe_source_nodes = top_global_nodes
        probe_source = "global_correspondence"
    elif scoring_mode in {
        "global_semantic_hybrid",
        "auto_levelup_state_classifier",
        "level7_fragmentation_guided",
        "level7_fragmentation_guarded",
    }:
        probe_source_nodes = top_search_nodes
        probe_source = (
            "hybrid_search"
            if scoring_mode == "global_semantic_hybrid"
            else scoring_mode
        )
    else:
        probe_source_nodes = top_match_nodes
        probe_source = "match_score"
    best_match_submit_nodes = _probe_action2_from_best_match_nodes(
        probe_source_nodes,
        full_game_id=full_game_id,
        rule_model=rule_model,
        root_match=root_match,
        danger_actions=danger_actions,
        scoring_mode=scoring_mode,
        submit_gate=submit_gate,
        limit=best_match_submit_probes,
        action_ontology_model=action_ontology_model,
        action_ontology_weight=action_ontology_weight,
        auto_levelup_state_classifier=auto_levelup_state_classifier,
        auto_levelup_classifier_weight=auto_levelup_classifier_weight,
        largest_second_pressure=largest_second_pressure,
    )
    action6_operator_probe_reports = _probe_action6_from_best_match_nodes(
        probe_source_nodes,
        full_game_id=full_game_id,
        rule_model=rule_model,
        root_match=root_match,
        danger_actions=danger_actions,
        scoring_mode=scoring_mode,
        submit_gate=submit_gate,
        action6_targets=action6_targets,
        limit_nodes=action6_operator_probe_nodes,
        limit_total=action6_operator_probe_total,
        action_ontology_model=action_ontology_model,
        action_ontology_weight=action_ontology_weight,
        auto_levelup_state_classifier=auto_levelup_state_classifier,
        auto_levelup_classifier_weight=auto_levelup_classifier_weight,
        largest_second_pressure=largest_second_pressure,
    )
    action6_operator_transforming = [
        item for item in action6_operator_probe_reports
        if int(item.get("spatial_operator_change", 0)) > 0
    ]
    local_repair_report = (
        _run_local_repair_from_best_global(
            top_global_nodes,
            full_game_id=full_game_id,
            rule_model=rule_model,
            root_match=root_match,
            danger_actions=danger_actions,
            include_action6=bool((include_action6 and not disable_action6) or targeted_repair_include_action6),
            action6_targets=action6_targets,
            submit_gate=submit_gate,
            top_k=local_repair_top_k,
            horizon=local_repair_horizon,
            beam_width=local_repair_beam_width,
            start_global_min=local_repair_start_global_min,
            global_floor=local_repair_global_floor,
            submit_probe_limit=local_repair_submit_probes,
        )
        if local_repair_from_best_global
        else {
            "enabled": False,
            "reason": "disabled",
            "seed_count": 0,
            "evaluated_branches": 0,
            "repair_candidates": 0,
            "reaches_level_8": False,
        }
    )
    targeted_repair_report = (
        _run_targeted_repair_from_best_global(
            top_global_nodes,
            full_game_id=full_game_id,
            rule_model=rule_model,
            root_match=root_match,
            danger_actions=danger_actions,
            include_action6=bool(include_action6 and not disable_action6),
            action6_targets=action6_targets,
            submit_gate=submit_gate,
            top_k=targeted_repair_top_k,
            horizon=targeted_repair_horizon,
            beam_width=targeted_repair_beam_width,
            start_global_min=targeted_repair_start_global_min,
            global_floor=targeted_repair_global_floor,
            submit_probe_limit=targeted_repair_submit_probes,
            max_target_components=targeted_repair_max_components,
        )
        if targeted_repair_from_best_global
        else {
            "enabled": False,
            "reason": "disabled",
            "seed_count": 0,
            "evaluated_branches": 0,
            "targeted_candidates": 0,
            "reaches_level_8": False,
        }
    )
    breaker_seed_nodes = [
        best,
        best_match,
        best_global,
        best_auto_levelup_state,
        best_fragmentation,
        *top_search_nodes,
        *top_global_nodes,
    ]
    # Macro mode: override breaker params for multi-stage cascade
    effective_breaker_enabled = probe_biggest_second_breaker or enable_breaker_macro
    if enable_breaker_macro:
        macro_final = float(macro_break_target)
        # Auto-compute 3-stage targets: evenly spaced from ~initial down to target
        # Use root second_largest as starting point
        root_features = (root_auto_levelup_state or {}).get("features") or {}
        root_second_largest = float(root_features.get("second_largest_size", 400.0))
        if not breaker_stage_targets:
            gap = root_second_largest - macro_final
            if gap > 0 and gap > 60:
                breaker_stage_targets = [
                    round(root_second_largest - gap / 3.0, 1),
                    round(root_second_largest - 2.0 * gap / 3.0, 1),
                    round(macro_final, 1),
                ]
                breaker_stages = 3
            elif gap > 0:
                breaker_stage_targets = [round(macro_final, 1)]
                breaker_stages = 1
            else:
                breaker_stage_targets = [round(macro_final, 1)]
                breaker_stages = 1
        if breaker_stages < len(breaker_stage_targets):
            breaker_stages = len(breaker_stage_targets)
        breaker_target_second_largest_max = macro_final
        # Use deeper search for macro mode
        if breaker_depth < 4:
            breaker_depth = 4
        # Relax constraints slightly for macro cascade
        if breaker_max_dotted < 3:
            breaker_max_dotted = 3
        if breaker_global_floor > 25.0:
            breaker_global_floor = 25.0

    biggest_second_breaker_report = (
        _probe_biggest_second_component_breaker(
            breaker_seed_nodes,
            full_game_id=full_game_id,
            rule_model=rule_model,
            root_match=root_match,
            danger_actions=danger_actions,
            include_action6=bool((include_action6 and not disable_action6) or breaker_include_action6),
            action6_targets=action6_targets,
            scoring_mode=scoring_mode,
            submit_gate=submit_gate,
            top_k=breaker_top_k,
            depth=breaker_depth,
            beam_width=breaker_beam_width,
            global_floor=breaker_global_floor,
            max_dotted=breaker_max_dotted,
            target_second_largest_max=breaker_target_second_largest_max,
            stages=breaker_stages,
            stage_targets=breaker_stage_targets,
            disable_action2=disable_action2,
            action_ontology_model=action_ontology_model,
            action_ontology_weight=action_ontology_weight,
            auto_levelup_state_classifier=auto_levelup_state_classifier,
            auto_levelup_classifier_weight=auto_levelup_classifier_weight,
            largest_second_pressure=largest_second_pressure,
            break_context_model=breaker_context_model,
            return_nodes=bool(enable_breaker_macro),
        )
        if effective_breaker_enabled
        else {
            "enabled": False,
            "reason": "disabled",
            "seed_count": 0,
            "evaluated_branches": 0,
            "candidate_count": 0,
            "passing_candidates": 0,
            "reaches_under_target": False,
        }
    )
    # Breaker macro: probe ACTION2 from the best under-target nodes
    breaker_macro_submit_nodes: List[GuidedNode] = []
    breaker_macro_report: Dict[str, Any] = {
        "enabled": bool(enable_breaker_macro),
        "macro_break_target": round(float(macro_break_target), 4) if enable_breaker_macro else None,
        "auto_stage_targets": list(breaker_stage_targets) if enable_breaker_macro else [],
        "under_target_node_count": 0,
        "passing_node_count": 0,
        "submit_probe_count": 0,
        "reaches_level_8": False,
        "game_over_count": 0,
    }
    if enable_breaker_macro:
        under_target_nodes = biggest_second_breaker_report.pop("_under_target_nodes", [])
        passing_nodes = biggest_second_breaker_report.pop("_passing_nodes", [])
        top_delta_nodes = biggest_second_breaker_report.pop("_top_delta_nodes", [])
        breaker_macro_report["under_target_node_count"] = len(under_target_nodes)
        breaker_macro_report["passing_node_count"] = len(passing_nodes)
        breaker_macro_report["top_delta_node_count"] = len(top_delta_nodes)
        # Probe ACTION2 from best available: under-target → passing → top-delta
        probe_candidates = list(under_target_nodes) + list(passing_nodes) + list(top_delta_nodes)
        if probe_candidates and breaker_macro_submit_probes > 0:
            breaker_macro_submit_nodes = _probe_action2_from_best_match_nodes(
                probe_candidates,
                full_game_id=full_game_id,
                rule_model=rule_model,
                root_match=root_match,
                danger_actions=danger_actions,
                scoring_mode=scoring_mode,
                submit_gate=submit_gate,
                limit=breaker_macro_submit_probes,
                action_ontology_model=action_ontology_model,
                action_ontology_weight=action_ontology_weight,
                auto_levelup_state_classifier=auto_levelup_state_classifier,
                auto_levelup_classifier_weight=auto_levelup_classifier_weight,
                largest_second_pressure=largest_second_pressure,
            )
        breaker_macro_report["submit_probe_count"] = len(breaker_macro_submit_nodes)
        breaker_macro_report["reaches_level_8"] = any(
            node.reached_next_level for node in breaker_macro_submit_nodes
        )
        breaker_macro_report["game_over_count"] = sum(
            1 for node in breaker_macro_submit_nodes if node.died
        )
        breaker_macro_report["submit_probes"] = [
            node.to_report() for node in breaker_macro_submit_nodes
        ]
        breaker_macro_report["best_submit"] = (
            max(
                breaker_macro_submit_nodes,
                key=lambda item: (int(item.reached_next_level), item.search_score),
            ).to_report()
            if breaker_macro_submit_nodes
            else None
        )
        # Phase 2: continuation search from best macro states
        # Uses fragmentation-first scoring: the compass is fragmentation progress,
        # NOT the main beam's composite score.
        macro_reached_level_8 = any(
            node.reached_next_level for node in breaker_macro_submit_nodes
        )

        def _macro_cont_score(node: GuidedNode) -> float:
            """Score a node purely by fragmentation progress toward human reference."""
            if node.died:
                return -99999.0
            if node.reached_next_level:
                return 99999.0
            features = (node.auto_levelup_state or {}).get("features") or {}
            frag = node.fragmentation_guidance or {}
            frag_score = float(frag.get("score", 0.0))
            frag_improvement = float(frag.get("improvement_score", 0.0))
            auto_score = float((node.auto_levelup_state or {}).get("score", 0.0))
            second_largest = float(features.get("second_largest_size", 9999.0))
            second_count = float(features.get("second_count", 0.0))
            largest_offender = float(features.get("largest_offender_size", 9999.0))
            return (
                80.0 * frag_score
                + 60.0 * max(0.0, frag_improvement)
                + 50.0 * auto_score
                - 2.0 * second_largest
                + 8.0 * second_count
                - 1.5 * largest_offender
                - 35.0 * node.repeat_penalty
                - 2.0 * node.depth
            )

        if not macro_reached_level_8 and probe_candidates:
            # Use top unique macro nodes as new beam roots
            # Narrow beam (12 seeds) but deep horizon — we need depth to fragment
            cont_beam_width = min(12, max(1, beam_width // 4))
            continuation_beam: List[GuidedNode] = []
            seen_cb: Set[str] = set()
            for node in probe_candidates:
                if node.died or node.reached_next_level or node.env is None:
                    continue
                if node.grid_hash in seen_cb:
                    continue
                seen_cb.add(node.grid_hash)
                continuation_beam.append(node)
                if len(continuation_beam) >= cont_beam_width:
                    break
            # Full horizon: we need depth to break further
            continuation_horizon = max(6, horizon)
            # Suppress ACTION2 during continuation — we're fragmenting, not submitting
            cont_suppress_actions = {"ACTION2", "ACTION6"} if disable_action6 else {"ACTION2"}
            continuation_evaluated: List[GuidedNode] = []
            for _cdepth in range(1, continuation_horizon + 1):
                cont_expanded: List[GuidedNode] = []
                for node in continuation_beam:
                    if node.died or node.reached_next_level:
                        cont_expanded.append(node)
                        continue
                    if node.env is None:
                        continue
                    parent_grid = _primary_grid(getattr(node.env, "observation_space", None))
                    for action in _allowed_actions(
                        getattr(node.env, "observation_space", None),
                        include_action6=include_action6,
                    ):
                        if action in cont_suppress_actions:
                            continue
                        if action == "ACTION6" and disable_action6:
                            continue
                        for action_data in _action_variants(
                            action,
                            include_action6=include_action6,
                            action6_targets=action6_targets,
                        ):
                            branch_env = copy.deepcopy(node.env)
                            raw = _step_branch(
                                branch_env,
                                full_game_id=full_game_id,
                                action=action,
                                action_data=action_data,
                            )
                            child = _make_guided_child(
                                node,
                                raw=raw,
                                env=branch_env,
                                action=action,
                                action_data=action_data,
                                rule_model=rule_model,
                                root_match=root_match,
                                parent_grid=parent_grid,
                                danger_actions=danger_actions,
                                scoring_mode=scoring_mode,
                                submit_gate=submit_gate,
                                action_ontology_model=action_ontology_model,
                                action_ontology_weight=action_ontology_weight,
                                auto_levelup_state_classifier=auto_levelup_state_classifier,
                                auto_levelup_classifier_weight=auto_levelup_classifier_weight,
                                largest_second_pressure=largest_second_pressure,
                            )
                            cont_expanded.append(child)
                # Sort by fragmentation-first score, not main beam score
                cont_expanded.sort(key=_macro_cont_score, reverse=True)
                continuation_evaluated.extend(cont_expanded)
                continuation_beam = cont_expanded[: cont_beam_width]
                if not continuation_beam:
                    break
            # Collect continuation results
            cont_safe = [n for n in continuation_evaluated if not n.died]
            cont_level8 = any(n.reached_next_level for n in continuation_evaluated)
            cont_best = max(
                cont_safe or continuation_evaluated or probe_candidates[:1],
                key=_macro_cont_score,
            )
            # Report fragmentation features for best continuation node
            cont_best_features = (cont_best.auto_levelup_state or {}).get("features") or {}
            cont_best_frag = cont_best.fragmentation_guidance or {}
            # Also try ACTION2 probes from continuation best (sorted by frag score)
            cont_submit_nodes: List[GuidedNode] = []
            if not cont_level8:
                cont_top = sorted(
                    cont_safe,
                    key=_macro_cont_score,
                    reverse=True,
                )
                cont_submit_nodes = _probe_action2_from_best_match_nodes(
                    cont_top,
                    full_game_id=full_game_id,
                    rule_model=rule_model,
                    root_match=root_match,
                    danger_actions=danger_actions,
                    scoring_mode=scoring_mode,
                    submit_gate=submit_gate,
                    limit=breaker_macro_submit_probes,
                    action_ontology_model=action_ontology_model,
                    action_ontology_weight=action_ontology_weight,
                    auto_levelup_state_classifier=auto_levelup_state_classifier,
                    auto_levelup_classifier_weight=auto_levelup_classifier_weight,
                    largest_second_pressure=largest_second_pressure,
                )
            cont_submit_level8 = any(
                n.reached_next_level for n in cont_submit_nodes
            )
            breaker_macro_report["continuation"] = {
                "scoring": "fragmentation_first",
                "horizon": int(continuation_horizon),
                "beam_seeds": len(seen_cb),
                "evaluated_branches": len(continuation_evaluated),
                "reaches_level_8": bool(cont_level8 or cont_submit_level8),
                "game_over_count": sum(1 for n in continuation_evaluated if n.died),
                "submit_probe_count": len(cont_submit_nodes),
                "submit_level_8": bool(cont_submit_level8),
                "best": cont_best.to_report() if cont_best else None,
                "best_frag_score": round(float(_macro_cont_score(cont_best)), 4) if cont_best else None,
                "best_second_largest": round(float(cont_best_features.get("second_largest_size", 0.0)), 1),
                "best_second_count": round(float(cont_best_features.get("second_count", 0.0)), 1),
                "best_largest_offender": round(float(cont_best_features.get("largest_offender_size", 0.0)), 1),
                "best_frag_proximity": round(float(cont_best_frag.get("proximity_score", 0.0)), 4),
                "best_frag_improvement": round(float(cont_best_frag.get("improvement_score", 0.0)), 4),
                "best_submit": (
                    max(
                        cont_submit_nodes,
                        key=lambda item: (int(item.reached_next_level), _macro_cont_score(item)),
                    ).to_report()
                    if cont_submit_nodes
                    else None
                ),
                "submit_probes": [
                    n.to_report() for n in cont_submit_nodes
                ],
            }
    agent_w1 = max(
        (
            node
            for node in safe
            if node.actions.count("ACTION3") == int(rewrite_saturation_action3_threshold)
        ),
        key=lambda item: item.search_score,
        default=None,
    )
    agent_w2 = max(
        (node for node in safe if node.first_noop_index is not None),
        key=lambda item: item.search_score,
        default=None,
    )
    window_comparison = (
        _window_comparison_report(
            agent_w1=agent_w1,
            agent_w2=agent_w2,
            human_w3=human_w3,
        )
        if scoring_mode == "rewrite_until_saturation"
        else None
    )
    return {
        "horizon": int(horizon),
        "beam_width": int(beam_width),
        "include_action6": bool(include_action6),
        "scoring_mode": scoring_mode,
        "submit_gate": submit_gate,
        "action_ontology": {
            "available": bool((action_ontology_model or {}).get("available")),
            "path": (action_ontology_model or {}).get("path"),
            "weight": round(float(action_ontology_weight), 4),
            "human_auto_actions": dict((action_ontology_model or {}).get("human_auto_actions") or {}),
            "human_auto_changed_cells_avg": round(
                float((action_ontology_model or {}).get("human_auto_changed_cells_avg") or 0.0),
                4,
            ),
        },
        "auto_levelup_state_classifier": {
            "available": bool((auto_levelup_state_classifier or {}).get("available")),
            "reference_count": int((auto_levelup_state_classifier or {}).get("reference_count") or 0),
            "reference_scope": (auto_levelup_state_classifier or {}).get("reference_scope"),
            "target_level": (auto_levelup_state_classifier or {}).get("target_level"),
            "action_counts": dict((auto_levelup_state_classifier or {}).get("action_counts") or {}),
            "weight": round(float(auto_levelup_classifier_weight), 4),
            "root_score": round(float(root_auto_levelup_state.get("score", 0.0)), 4),
            "root_nearest_references": list(root_auto_levelup_state.get("nearest_references") or []),
        },
        "largest_second_pressure": bool(largest_second_pressure),
        "auto_levelup_feature_gap": {
            "root": _auto_levelup_feature_gap_report(
                root,
                classifier=auto_levelup_state_classifier,
            ),
            "best": _auto_levelup_feature_gap_report(
                best,
                classifier=auto_levelup_state_classifier,
            ),
            "best_auto": _auto_levelup_feature_gap_report(
                best_auto_levelup_state,
                classifier=auto_levelup_state_classifier,
            ),
            "best_global": _auto_levelup_feature_gap_report(
                best_global,
                classifier=auto_levelup_state_classifier,
            ),
            "best_fragmentation": _auto_levelup_feature_gap_report(
                best_fragmentation,
                classifier=auto_levelup_state_classifier,
            ),
        },
        "post_saturation_probe": bool(post_saturation_probe),
        "rewrite_saturation_action3_threshold": int(rewrite_saturation_action3_threshold),
        "root_match_score": root.match.to_report(),
        "root_global_correspondence_score": root_global.to_report(),
        "evaluated_branches": len(evaluated),
        "reaches_level_8": bool(reaches_next),
        "improved_match_states": len(improved),
        "improved_global_states": len(improved_global),
        "readiness_gate_states": len(readiness_nodes),
        "submit_attempts": len(submit_attempts),
        "submit_not_ready": sum(1 for node in submit_attempts if node.submit_not_ready),
        "post_saturation_probe_skips": int(post_saturation_probe_skips),
        "post_saturation_probe_nodes": len(post_saturation_nodes),
        "best_match_submit_probes": [
            node.to_report()
            for node in best_match_submit_nodes
        ],
        "submit_probe_source": probe_source,
        "best_match_submit_probe_summary": {
            "count": len(best_match_submit_nodes),
            "reaches_level_8": any(node.reached_next_level for node in best_match_submit_nodes),
            "wins": any(node.state == "WIN" for node in best_match_submit_nodes),
            "game_over": sum(1 for node in best_match_submit_nodes if node.died),
            "not_ready": sum(1 for node in best_match_submit_nodes if node.submit_not_ready),
            "best": (
                max(
                    best_match_submit_nodes,
                    key=lambda item: (int(item.reached_next_level), item.search_score),
                ).to_report()
                if best_match_submit_nodes
                else None
            ),
        },
        "agent_window_candidates": {
            "W1_at_action3_threshold": _strip_grid(_node_window_report(agent_w1)),
            "W2_first_noop": _strip_grid(_node_window_report(agent_w2)),
        },
        "window_comparison": window_comparison,
        "best_post_saturation_probe": (
            best_post_saturation_probe.to_report()
            if best_post_saturation_probe is not None
            else None
        ),
        "submit_attempts_detail": [
            {
                "path": node.actions,
                "target_color": node.last_target_info.get("color"),
                "target_size": node.last_target_info.get("size"),
                "target_distance": node.last_target_info.get("distance"),
                "target_is_unmatched": node.last_target_info.get("is_unmatched"),
                "target_centroid": {
                    "y": node.last_target_info.get("centroid_y"),
                    "x": node.last_target_info.get("centroid_x"),
                },
                "cursor_position": {
                    "y": node.match.cursor_near_target,  # Approximation
                },
                "match_score": node.match.score,
                "unmatched_total": node.match.unmatched_first + node.match.unmatched_second,
                "reached_next_level": node.reached_next_level,
            }
            for node in submit_attempts[:20]  # Limit to first 20
        ],
        "gated_submit_skips": int(gated_submit_skips),
        "action3_contradictions": len(contradictions),
        "action6_branches": len(action6_nodes),
        "action6_transforming_branches": len(transforming_action6_nodes),
        "max_action6_spatial_change": max(
            (int(node.max_action6_spatial_change) for node in action6_nodes),
            default=0,
        ),
        "action6_operator_probe_summary": {
            "count": len(action6_operator_probe_reports),
            "transforming_branches": len(action6_operator_transforming),
            "max_action6_spatial_change": max(
                (int(item.get("spatial_operator_change", 0)) for item in action6_operator_probe_reports),
                default=0,
            ),
            "best_matched_pairs_after_action6": max(
                (int(item.get("matched_pairs_after_action6", 0)) for item in action6_operator_probe_reports),
                default=0,
            ),
            "best_dotted_delta_after_action6": max(
                (int(item.get("dotted_delta_after_action6", 0)) for item in action6_operator_probe_reports),
                default=0,
            ),
        },
        "known_danger_actions": list(danger_actions),
        "rewrite_logs": _rewrite_logs(best) if scoring_mode == "rewrite_until_saturation" else None,
        "rewrite_logs_best_match": _rewrite_logs(best_match) if scoring_mode == "rewrite_until_saturation" else None,
        "best": best.to_report(),
        "best_match": best_match.to_report(),
        "best_global": best_global.to_report(),
        "best_auto_levelup_state": best_auto_levelup_state.to_report(),
        "best_fragmentation": best_fragmentation.to_report(),
        "top": [
            node.to_report()
            for node in top_search_nodes[:10]
        ],
        "top_match": [
            node.to_report()
            for node in top_match_nodes[:10]
        ],
        "top_global": [
            node.to_report()
            for node in top_global_nodes[:10]
        ],
        "top_contradictions": [
            node.to_report()
            for node in sorted(contradictions, key=lambda item: item.search_score, reverse=True)[:5]
        ],
        "top_action6": (
            action6_operator_probe_reports[:10]
            if not action6_nodes and action6_operator_probe_reports
            else [
            node.to_report()
            for node in sorted(action6_nodes, key=lambda item: item.search_score, reverse=True)[:10]
            ]
        ),
        "top_action6_source": "operator_probe" if not action6_nodes and action6_operator_probe_reports else "beam",
        "top_action6_operator_probes": action6_operator_probe_reports[:10],
        "top_transforming_action6": [
            node.to_report()
            for node in sorted(
                transforming_action6_nodes,
                key=lambda item: item.search_score, reverse=True)[:10]
        ],
        "local_repair_from_best_global": local_repair_report,
        "targeted_repair_from_best_global": targeted_repair_report,
        "biggest_second_breaker_probe": biggest_second_breaker_report,
        "breaker_macro": breaker_macro_report,
    }


def _default_rule_report_path(game_id: str) -> Path:
    return DEFAULT_RULE_REPORT_DIR / f"{game_id}.global_rule_inference.json"


def _default_human_break_report_path(
    *,
    game_id: str,
    episode_id: str,
    pair_colors: Tuple[int, int],
    target_level: int,
) -> Path:
    return DEFAULT_HUMAN_BREAKS_DIR / (
        f"{game_id}.{episode_id}.colors{pair_colors[0]}_{pair_colors[1]}.level{target_level}.biggest_second_breaks.json"
    )


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_logs(node: GuidedNode) -> Dict[str, Any]:
    sequence = list(node.action3_changed_cells_sequence)
    first_noop_ordinal: Optional[int] = None
    for index, changed in enumerate(sequence, start=1):
        if int(changed) == 0:
            first_noop_ordinal = index
            break
    return {
        "depth": int(node.depth),
        "action3_count": int(node.actions.count("ACTION3")),
        "action3_productive_count": int(node.action3_productive_count),
        "action3_noop_count": int(sum(1 for changed in sequence if int(changed) == 0)),
        "cumulative_changed_cells": int(node.cumulative_changed_cells),
        "changed_cells_sequence": sequence,
        "first_noop_index": node.first_noop_index,
        "first_noop_action3_ordinal": first_noop_ordinal,
        "unique_grid_hashes": int(len(set(node.path_hashes))),
    }


def _human_level_action3_noop_window(
    steps: Sequence[Any],
    *,
    level_start_index: int,
    terminal_index: int,
) -> Optional[Dict[str, Any]]:
    action3_count = 0
    productive_count = 0
    changed_sequence: List[int] = []
    for idx in range(level_start_index, terminal_index + 1):
        step = steps[idx]
        if getattr(step, "action", None) != "ACTION3":
            continue
        action3_count += 1
        changed = _grid_changed_cells(step.frame_before, step.frame_after)
        changed_sequence.append(int(changed))
        if changed > 0:
            productive_count += 1
            continue
        return {
            "trace_index": int(idx),
            "trace_step": int(step.step),
            "action3_count": int(action3_count),
            "productive_action3_count": int(productive_count),
            "changed_cells_sequence_tail": changed_sequence[-12:],
            "changed_cells_sequence_len": len(changed_sequence),
            "frame_before_hash": _hash_grid(step.frame_before),
            "frame_after_hash": _hash_grid(step.frame_after),
            "frame_before": step.frame_before,
            "frame_after": step.frame_after,
        }
    return None


def _node_window_report(node: Optional[GuidedNode]) -> Optional[Dict[str, Any]]:
    if node is None or node.grid is None:
        return None
    return {
        "depth": int(node.depth),
        "hash": _hash_grid(node.grid),
        "action_tail": node.actions[-12:],
        "action3_count": int(node.actions.count("ACTION3")),
        "action3_productive_count": int(node.action3_productive_count),
        "first_noop_index": node.first_noop_index,
        "unique_grid_hashes": int(len(set(node.path_hashes))),
        "cumulative_changed_cells": int(node.cumulative_changed_cells),
        "match_score": node.match.to_report(),
        "grid": node.grid,
    }


def _window_comparison_report(
    *,
    agent_w1: Optional[GuidedNode],
    agent_w2: Optional[GuidedNode],
    human_w3: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    w1 = _node_window_report(agent_w1)
    w2 = _node_window_report(agent_w2)
    w3_grid = human_w3.get("frame_before") if human_w3 else None
    w3 = None
    if human_w3 and w3_grid is not None:
        w3 = {
            "trace_index": human_w3["trace_index"],
            "trace_step": human_w3["trace_step"],
            "hash": _hash_grid(w3_grid),
            "action3_count": human_w3["action3_count"],
            "productive_action3_count": human_w3["productive_action3_count"],
            "changed_cells_sequence_tail": human_w3["changed_cells_sequence_tail"],
            "changed_cells_sequence_len": human_w3["changed_cells_sequence_len"],
        }

    def diff(left: Optional[Dict[str, Any]], right_grid: Any) -> Optional[Dict[str, Any]]:
        if not left or left.get("grid") is None or right_grid is None:
            return None
        return {
            "same_hash": _hash_grid(left["grid"]) == _hash_grid(right_grid),
            "changed_cells": _grid_changed_cells(left["grid"], right_grid),
            "bbox": _diff_bbox(left["grid"], right_grid),
        }

    return {
        "W1_agent_at_action3_48": _strip_grid(w1),
        "W2_agent_first_noop": _strip_grid(w2),
        "W3_human_first_noop": w3,
        "W1_vs_W3": diff(w1, w3_grid),
        "W2_vs_W3": diff(w2, w3_grid),
    }


def _strip_grid(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    out = dict(item)
    out.pop("grid", None)
    return out


def _print_report(payload: Dict[str, Any], output: Path) -> None:
    search = payload["guided_search"]
    rule = payload["rule_model"]
    print("=" * 88)
    print("TaskProgram-guided level7")
    print("=" * 88)
    print(f"game:        {payload['game_id']}")
    print(f"episode:     {payload['episode_id']}")
    print(f"pair colors: {rule['pair_colors']}")
    print(f"threshold:   {rule['ready_threshold']}")
    print(f"root score:  {search['root_match_score']['score']}")
    root_global = search.get("root_global_correspondence_score") or {}
    if root_global:
        print(f"root global: {root_global.get('score')}")
    print(f"improved:    {search['improved_match_states']}")
    if "improved_global_states" in search:
        print(f"global +:    {search['improved_global_states']}")
    if "readiness_gate_states" in search:
        print(f"ready states:{search['readiness_gate_states']}")
    print(f"level 8:     {search['reaches_level_8']}")
    if search.get("scoring_mode") in {
        "action_ontology_guided",
        "auto_levelup_state_classifier",
        "level7_fragmentation_guided",
        "level7_fragmentation_guarded",
    }:
        ontology = search.get("action_ontology") or {}
        print(
            "ontology:   "
            f"{ontology.get('available')} "
            f"auto={ontology.get('human_auto_actions', {})}"
        )
        if search.get("scoring_mode") in {
            "auto_levelup_state_classifier",
            "level7_fragmentation_guided",
            "level7_fragmentation_guarded",
        }:
            classifier = search.get("auto_levelup_state_classifier") or {}
            print(
                "auto cls:   "
                f"{classifier.get('available')} "
                f"scope={classifier.get('reference_scope')} "
                f"refs={classifier.get('reference_count')} "
                f"root={classifier.get('root_score')}"
            )
        print(f"ACTION2 ops: {search['submit_attempts']} ({search['submit_not_ready']} old-gate not-ready)")
    else:
        print(f"submit gate: {search.get('submit_gate', 'semantic')}")
        print(f"submits:     {search['submit_attempts']} ({search['submit_not_ready']} not-ready)")
    print(f"gated A2:    {search['gated_submit_skips']}")
    if search.get("post_saturation_probe"):
        print(f"post probe:  {search.get('post_saturation_probe_nodes', 0)} nodes")
    print(f"ACTION6:     {search['action6_branches']} branches")
    action6_probe = search.get("action6_operator_probe_summary") or {}
    if action6_probe.get("count"):
        print(
            "ACTION6 probes: "
            f"{action6_probe.get('count')} "
            f"transforming={action6_probe.get('transforming_branches')} "
            f"max_change={action6_probe.get('max_action6_spatial_change')}"
        )
    print(f"best path:   {search['best']['actions']}")
    print(f"best score:  {search['best']['match_score']['score']}")
    best_global = (search.get("best") or {}).get("global_correspondence_score") or {}
    if best_global:
        print(f"best global: {best_global.get('score')}")
    best_auto = (search.get("best_auto_levelup_state") or {}).get("auto_levelup_state") or {}
    if best_auto.get("available"):
        nearest = (best_auto.get("nearest_references") or [{}])[0]
        print(
            "best auto:  "
            f"{best_auto.get('score')} "
            f"near={nearest.get('label')} "
            f"action={nearest.get('action')}"
        )
        gap = ((search.get("auto_levelup_feature_gap") or {}).get("best_auto") or {})
        largest = gap.get("largest_gaps") or {}
        if largest:
            preview = []
            for name, item in list(largest.items())[:5]:
                preview.append(
                    f"{name}={item.get('delta_agent_minus_reference')}"
                )
            print(f"auto gaps:  {', '.join(preview)}")
    best_frag = (search.get("best_fragmentation") or {}).get("fragmentation_guidance") or {}
    if best_frag.get("available"):
        largest = best_frag.get("largest_gaps") or {}
        preview = []
        for name, item in list(largest.items())[:5]:
            preview.append(f"{name}={item.get('delta_agent_minus_target')}")
        guard = (search.get("best") or {}).get("fragmentation_guardrails") or {}
        print(
            "best frag:  "
            f"{best_frag.get('score')} "
            f"guard_pass={guard.get('passes')} "
            f"guard_penalty={guard.get('penalty')} "
            f"gaps={', '.join(preview)}"
        )
    breaker = search.get("biggest_second_breaker_probe") or {}
    if breaker.get("enabled"):
        best_break = breaker.get("best_passing") or breaker.get("best") or {}
        best_delta_break = breaker.get("best_passing_delta") or breaker.get("best_delta") or {}
        after = best_break.get("after") or {}
        context_model = breaker.get("break_context_model") or {}
        print(
            "big breaker:"
            f" seeds={breaker.get('seed_count')} "
            f"eval={breaker.get('evaluated_branches')} "
            f"passing={breaker.get('passing_candidates')} "
            f"under_target={breaker.get('reaches_under_target')} "
            f"any_stage={breaker.get('reaches_any_stage_target')} "
            f"context={context_model.get('available')}"
        )
        if best_break:
            context = best_break.get("after_break_context") or {}
            ref = (context.get("reference") or {})
            print(
                "best break: "
                f"second_largest={after.get('second_largest_size')} "
                f"delta={best_break.get('delta_second_largest_size')} "
                f"count_delta={best_break.get('delta_component_count_11')} "
                f"global={after.get('global_score')} "
                f"dotted={after.get('dotted_violations')} "
                f"ctx={context.get('score')} "
                f"ref={ref.get('trace_index')}:{ref.get('action')} "
                f"path={best_break.get('probe_actions')}"
            )
        if best_delta_break and best_delta_break is not best_break:
            delta_after = best_delta_break.get("after") or {}
            print(
                "best delta: "
                f"second_largest={delta_after.get('second_largest_size')} "
                f"delta={best_delta_break.get('delta_second_largest_size')} "
                f"count_delta={best_delta_break.get('delta_component_count_11')} "
                f"global={delta_after.get('global_score')} "
                f"dotted={delta_after.get('dotted_violations')} "
                f"path={best_delta_break.get('probe_actions')}"
            )
        for stage in list(breaker.get("stage_summaries") or [])[:4]:
            stage_best = stage.get("best_passing_delta") or stage.get("best_delta") or {}
            stage_after = stage_best.get("after") or {}
            if not stage_best:
                continue
            print(
                f"breaker s{stage.get('stage')}: "
                f"target<{stage.get('target_second_largest_max')} "
                f"delta={stage_best.get('delta_second_largest_size')} "
                f"second_largest={stage_after.get('second_largest_size')} "
                f"count={stage_after.get('component_count_11')} "
                f"under={stage.get('reaches_under_target')} "
                f"path={stage_best.get('probe_actions')}"
            )
    macro = search.get("breaker_macro") or {}
    if macro.get("enabled"):
        print(
            "MACRO BREAK: "
            f"target={macro.get('macro_break_target')} "
            f"stages={macro.get('auto_stage_targets')} "
            f"under_target={macro.get('under_target_node_count')} "
            f"passing={macro.get('passing_node_count')} "
            f"top_delta={macro.get('top_delta_node_count', 0)} "
            f"probes={macro.get('submit_probe_count')} "
            f"level8={macro.get('reaches_level_8')} "
            f"game_over={macro.get('game_over_count')}"
        )
        macro_best = macro.get("best_submit") or {}
        if macro_best:
            print(
                "macro best:  "
                f"score={macro_best.get('search_score')} "
                f"level={macro_best.get('level')} "
                f"reached={macro_best.get('reached_next_level')} "
                f"path={macro_best.get('actions')}"
            )
        cont = macro.get("continuation") or {}
        if cont:
            print(
                "macro cont:  "
                f"scoring={cont.get('scoring')} "
                f"horizon={cont.get('horizon')} "
                f"seeds={cont.get('beam_seeds')} "
                f"eval={cont.get('evaluated_branches')} "
                f"level8={cont.get('reaches_level_8')} "
                f"submit_probes={cont.get('submit_probe_count')} "
                f"submit_level8={cont.get('submit_level_8')} "
                f"game_over={cont.get('game_over_count')}"
            )
            print(
                "cont frag:   "
                f"second_largest={cont.get('best_second_largest')} "
                f"second_count={cont.get('best_second_count')} "
                f"offender={cont.get('best_largest_offender')} "
                f"frag_prox={cont.get('best_frag_proximity')} "
                f"frag_impr={cont.get('best_frag_improvement')} "
                f"frag_score={cont.get('best_frag_score')}"
            )
            cont_best = cont.get("best") or {}
            if cont_best:
                print(
                    "cont best:   "
                    f"score={cont_best.get('search_score')} "
                    f"level={cont_best.get('level')} "
                    f"reached={cont_best.get('reached_next_level')} "
                    f"path={cont_best.get('actions')}"
                )
            cont_submit = cont.get("best_submit") or {}
            if cont_submit:
                print(
                    "cont submit: "
                    f"score={cont_submit.get('search_score')} "
                    f"level={cont_submit.get('level')} "
                    f"reached={cont_submit.get('reached_next_level')} "
                    f"path={cont_submit.get('actions')}"
                )
    best_global_node = (search.get("best_global") or {})
    if best_global_node:
        best_global_score = (best_global_node.get("global_correspondence_score") or {}).get("score")
        print(f"top global:  {best_global_score} path={best_global_node.get('actions')}")
    probe_summary = search.get("best_match_submit_probe_summary") or {}
    if probe_summary.get("count"):
        print(
            "best-match A2 probes: "
            f"{probe_summary.get('count')} "
            f"level8={probe_summary.get('reaches_level_8')} "
            f"game_over={probe_summary.get('game_over')}"
        )
    if search.get("best_post_saturation_probe"):
        print(f"best probe:  {search['best_post_saturation_probe']['actions']}")
    repair = search.get("local_repair_from_best_global") or {}
    if repair.get("enabled"):
        print(
            "local repair: "
            f"seeds={repair.get('seed_count')} "
            f"candidates={repair.get('repair_candidates')} "
            f"ready={repair.get('readiness_gate_states')} "
            f"level8={repair.get('reaches_level_8')}"
        )
        best_repair = repair.get("best_repair") or {}
        if best_repair:
            best_repair_global = (best_repair.get("global_correspondence_score") or {}).get("score")
            best_repair_match = (best_repair.get("match_score") or {}).get("score")
            repair_delta = repair.get("best_repair_delta_from_best_seed") or {}
            print(
                "best repair: "
                f"global={best_repair_global} "
                f"match={best_repair_match} "
                f"delta={repair_delta} "
                f"path={best_repair.get('actions')}"
            )
        repair_probe = repair.get("submit_probe_summary") or {}
        if repair_probe.get("count"):
            print(
                "repair A2:   "
                f"{repair_probe.get('count')} "
                f"level8={repair_probe.get('reaches_level_8')} "
                f"not_ready={repair_probe.get('not_ready')}"
            )
    targeted = search.get("targeted_repair_from_best_global") or {}
    if targeted.get("enabled"):
        print(
            "target repair: "
            f"seeds={targeted.get('seed_count')} "
            f"candidates={targeted.get('targeted_candidates')} "
            f"ready={targeted.get('readiness_gate_states')} "
            f"level8={targeted.get('reaches_level_8')}"
        )
        best_targeted = targeted.get("best_targeted") or {}
        if best_targeted:
            best_targeted_global = (best_targeted.get("global_correspondence_score") or {}).get("score")
            best_targeted_match = (best_targeted.get("match_score") or {}).get("score")
            targeted_delta = targeted.get("best_targeted_delta_from_best_seed") or {}
            targeted_info = best_targeted.get("targeted_repair") or {}
            print(
                "best target: "
                f"global={best_targeted_global} "
                f"match={best_targeted_match} "
                f"delta={targeted_delta} "
                f"touches={targeted_info.get('touched_reasons')} "
                f"path={best_targeted.get('actions')}"
            )
        targeted_probe = targeted.get("submit_probe_summary") or {}
        if targeted_probe.get("count"):
            print(
                "target A2:   "
                f"{targeted_probe.get('count')} "
                f"level8={targeted_probe.get('reaches_level_8')} "
                f"not_ready={targeted_probe.get('not_ready')}"
            )
    # Rewrite mode logs
    if search.get('rewrite_logs'):
        logs = search['rewrite_logs']
        print(f"A3 count:    {logs.get('action3_count', 'N/A')}")
        print(f"A3 productive: {logs.get('action3_productive_count', 'N/A')}")
        print(f"cumulative changes: {logs.get('cumulative_changed_cells', 'N/A')}")
        print(f"first noop:  {logs.get('first_noop_index', 'N/A')}")
        print(f"unique grids: {logs.get('unique_grid_hashes', 'N/A')}")
    windows = payload.get("window_comparison") or {}
    if windows.get("W3_human_first_noop"):
        human = windows["W3_human_first_noop"]
        print(f"human noop:  A3={human.get('action3_count')} trace_idx={human.get('trace_index')}")
    print(f"json:        {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--target-level", type=int, default=7)
    parser.add_argument("--max-rule-level", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--beam-width", "--beam", dest="beam_width", type=int, default=48)
    parser.add_argument("--danger-window", type=int, default=8)
    parser.add_argument("--include-action6", action="store_true")
    parser.add_argument("--action6-candidates", type=int, default=8)
    parser.add_argument("--disable-action2", action="store_true", help="Disable ACTION2 (submit) for rewrite_until_saturation mode")
    parser.add_argument("--disable-action6", action="store_true", help="Disable ACTION6 initially for rewrite_until_saturation mode")
    parser.add_argument(
        "--post-saturation-probe",
        action="store_true",
        help="In rewrite mode, disable submit/click until first no-op or ACTION3 threshold, then probe ACTION2/ACTION1/ACTION5/ACTION7.",
    )
    parser.add_argument(
        "--rewrite-saturation-action3-threshold",
        type=int,
        default=48,
        help="ACTION3 count threshold that opens post-saturation probes if no no-op was observed.",
    )
    parser.add_argument(
        "--pair-colors",
        nargs="+",
        default=None,
        help="Override inferred correspondence colors, e.g. '4,5' or '10,11'.",
    )
    parser.add_argument("--rule-report", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--scoring-mode",
        choices=[
            "default",
            "semantic",
            "global_correspondence",
            "global_semantic_hybrid",
            "action_ontology_guided",
            "auto_levelup_state_classifier",
            "level7_fragmentation_guided",
            "level7_fragmentation_guarded",
            "dotted_constraint_repair",
            "cursor_target",
            "action3_productive",
            "rewrite_until_saturation",
        ],
        default="default",
        help="Scoring mode: semantic/default (match-first), global_correspondence, global_semantic_hybrid, action_ontology_guided, auto_levelup_state_classifier, level7_fragmentation_guided, level7_fragmentation_guarded, dotted_constraint_repair, cursor_target, action3_productive, rewrite_until_saturation",
    )
    parser.add_argument(
        "--action-ontology-report",
        type=Path,
        default=None,
        help="Empirical action ontology JSON from discover_action_ontology.py.",
    )
    parser.add_argument(
        "--action-ontology-weight",
        type=float,
        default=1.0,
        help="Multiplier for empirical action-affordance terms in action_ontology_guided mode.",
    )
    parser.add_argument(
        "--auto-levelup-classifier-max-level",
        type=int,
        default=0,
        help="Human level-up references used by auto_levelup_state_classifier; 0 means --target-level.",
    )
    parser.add_argument(
        "--auto-reference-scope",
        choices=["all", "same-level", "nearby-level"],
        default="all",
        help="Reference scope for auto_levelup_state_classifier: all, same-level, or nearby-level.",
    )
    parser.add_argument(
        "--auto-levelup-classifier-weight",
        type=float,
        default=1.0,
        help="Multiplier for state similarity to human pre-auto-levelup references.",
    )
    parser.add_argument(
        "--largest-second-pressure",
        action="store_true",
        help="In level-7 fragmentation modes, over-weight reducing the largest color-11/offender components.",
    )
    parser.add_argument(
        "--probe-biggest-second-breaker",
        action="store_true",
        help="After the main search, probe action/action-pair operators that reduce the largest color-11 component.",
    )
    parser.add_argument(
        "--breaker-top-k",
        type=int,
        default=6,
        help="Number of unique guarded seeds for biggest-second-component breaker probe.",
    )
    parser.add_argument(
        "--breaker-depth",
        type=int,
        default=2,
        help="Probe depth for biggest-second-component breaker: 1 tests single actions, 2 tests action pairs.",
    )
    parser.add_argument(
        "--breaker-beam",
        type=int,
        default=48,
        help="Beam width inside biggest-second-component breaker probe.",
    )
    parser.add_argument(
        "--breaker-global-floor",
        type=float,
        default=45.0,
        help="Minimum global score preserved by biggest-second-component breaker candidates.",
    )
    parser.add_argument(
        "--breaker-max-dotted",
        type=int,
        default=2,
        help="Maximum dotted violations preserved by biggest-second-component breaker candidates.",
    )
    parser.add_argument(
        "--breaker-target-second-largest-max",
        type=float,
        default=200.0,
        help="Success threshold for largest color-11 component after breaker probe.",
    )
    parser.add_argument(
        "--breaker-stages",
        type=int,
        default=1,
        help="Number of recursive breaker stages. Stage N starts from top candidates produced by stage N-1.",
    )
    parser.add_argument(
        "--breaker-stage-targets",
        nargs="*",
        type=float,
        default=None,
        help="Optional per-stage second_largest targets, e.g. '--breaker-stage-targets 260 200'.",
    )
    parser.add_argument(
        "--breaker-include-action6",
        action="store_true",
        help="Allow ACTION6 only inside the biggest-second-component breaker probe, without enabling it in the main beam.",
    )
    parser.add_argument(
        "--enable-breaker-macro",
        action="store_true",
        help="Enable the breaker macro-operator: run multi-stage cascade to break the largest color-11 component, then probe ACTION2 from best results.",
    )
    parser.add_argument(
        "--macro-break-target",
        type=float,
        default=200.0,
        help="Final second_largest target for the breaker macro cascade. Intermediate stage targets are auto-computed.",
    )
    parser.add_argument(
        "--breaker-macro-submit-probes",
        type=int,
        default=12,
        help="Number of ACTION2 submit probes from best breaker macro under-target states.",
    )
    parser.add_argument(
        "--breaker-context-similarity",
        action="store_true",
        help="Rank breaker seeds/branches by similarity to human pre-break contexts from trace_human_biggest_second_breaks.py.",
    )
    parser.add_argument(
        "--breaker-human-break-report",
        type=Path,
        default=None,
        help="Optional human biggest-second-breaks JSON used by --breaker-context-similarity.",
    )
    parser.add_argument(
        "--submit-gate",
        choices=["strict", "semantic", "probe", "cursor_target"],
        default="semantic",
        help="ACTION2 submit gate: strict (dotted, WRONG), semantic (match-based), probe (limited submits), cursor_target (cursor_near_target >= 1)",
    )
    parser.add_argument("--max-submit-probes", type=int, default=16, help="Max ACTION2 probes in probe mode")
    parser.add_argument(
        "--best-match-submit-probes",
        type=int,
        default=12,
        help="Apply ACTION2 once from the top-N best_match states after the beam.",
    )
    parser.add_argument(
        "--action6-operator-probe-nodes",
        type=int,
        default=8,
        help="Number of top best_match states used for ACTION6 operator probes outside the beam.",
    )
    parser.add_argument(
        "--action6-operator-probe-total",
        type=int,
        default=64,
        help="Maximum total ACTION6 operator probes outside the beam.",
    )
    parser.add_argument(
        "--local-repair-from-best-global",
        action="store_true",
        help="After the main search, repair locally from top global-correspondence states before probing ACTION2.",
    )
    parser.add_argument(
        "--local-repair-top-k",
        type=int,
        default=6,
        help="Number of unique high-global seeds for local repair.",
    )
    parser.add_argument(
        "--local-repair-horizon",
        type=int,
        default=12,
        help="Short horizon for local repair from high-global states.",
    )
    parser.add_argument(
        "--local-repair-beam",
        type=int,
        default=12,
        help="Beam width for local repair from high-global states.",
    )
    parser.add_argument(
        "--local-repair-start-global-min",
        type=float,
        default=65.0,
        help="Minimum global correspondence score required to seed local repair.",
    )
    parser.add_argument(
        "--local-repair-global-floor",
        type=float,
        default=60.0,
        help="Minimum global correspondence score preserved during local repair.",
    )
    parser.add_argument(
        "--local-repair-submit-probes",
        type=int,
        default=12,
        help="ACTION2 probes from repaired local states.",
    )
    parser.add_argument(
        "--targeted-repair-from-best-global",
        action="store_true",
        help="After the main search, diagnose offending components and only keep repairs that touch them.",
    )
    parser.add_argument(
        "--targeted-repair-top-k",
        type=int,
        default=6,
        help="Number of unique high-global seeds for targeted repair.",
    )
    parser.add_argument(
        "--targeted-repair-horizon",
        type=int,
        default=12,
        help="Short horizon for targeted repair from high-global states.",
    )
    parser.add_argument(
        "--targeted-repair-beam",
        type=int,
        default=12,
        help="Beam width for targeted repair from high-global states.",
    )
    parser.add_argument(
        "--targeted-repair-start-global-min",
        type=float,
        default=65.0,
        help="Minimum global correspondence score required to seed targeted repair.",
    )
    parser.add_argument(
        "--targeted-repair-global-floor",
        type=float,
        default=60.0,
        help="Minimum global correspondence score preserved during targeted repair.",
    )
    parser.add_argument(
        "--targeted-repair-submit-probes",
        type=int,
        default=12,
        help="ACTION2 probes from targeted repair states.",
    )
    parser.add_argument(
        "--targeted-repair-max-components",
        type=int,
        default=12,
        help="Maximum diagnosed offender components used as targeted repair regions.",
    )
    parser.add_argument(
        "--targeted-repair-include-action6",
        action="store_true",
        help="Allow ACTION6 only inside targeted repair from offender components, without enabling it in the main beam.",
    )
    parser.add_argument("--strong-improvement-margin", type=float, default=15.0, help="Match score delta threshold for semantic gate")
    parser.add_argument(
        "--forbid-action3-before-submit",
        action="store_true",
        help="Forbid ACTION3 before first valid ACTION2 submit (level 7 hypothesis)"
    )
    args = parser.parse_args()

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(PROJECT_ROOT / "environment_files"),
    )
    full_game_id = _resolve_full_game_id(arc, args.game)
    selection = _load_selected_episode(
        PROJECT_ROOT / "human_traces",
        requested_game=args.game,
        resolved_game_id=full_game_id,
        episode_id=args.episode_id,
        require_win=False,
    )
    frontier = find_level_frontier(
        selection,
        target_level=args.target_level,
        danger_window=args.danger_window,
    )
    base_env, base_raw = _replay_prefix(
        arc,
        full_game_id,
        selection.steps,
        stop_before_index=frontier.level_start_index,
        expected_frame=frontier.level_start_frame,
    )
    rule_report_path = args.rule_report or _default_rule_report_path(full_game_id)
    rule_report = _load_rule_report(rule_report_path)
    rule_model = calibrate_rule_model(
        selection_steps=selection.steps,
        episode_id=selection.episode_id,
        rule_report=rule_report,
        max_level=args.max_rule_level,
        pair_colors_override=_parse_pair_colors(args.pair_colors),
    )
    action_ontology_path = args.action_ontology_report or _default_action_ontology_path(full_game_id)
    action_ontology_model = _load_action_ontology_model(
        action_ontology_path,
        pair_colors=rule_model.pair_colors,
    )
    auto_classifier_max_level = (
        int(args.auto_levelup_classifier_max_level)
        if int(args.auto_levelup_classifier_max_level) > 0
        else int(args.target_level)
    )
    auto_levelup_state_classifier = _fit_auto_levelup_state_classifier(
        selection_steps=selection.steps,
        pair_colors=rule_model.pair_colors,
        max_level=auto_classifier_max_level,
        target_level=int(args.target_level),
        reference_scope=args.auto_reference_scope,
    )
    human_break_report_path = args.breaker_human_break_report or _default_human_break_report_path(
        game_id=full_game_id,
        episode_id=selection.episode_id,
        pair_colors=rule_model.pair_colors,
        target_level=int(args.target_level),
    )
    breaker_context_model = (
        _load_break_context_model(human_break_report_path)
        if args.breaker_context_similarity
        else {
            "available": False,
            "reason": "disabled",
            "path": str(human_break_report_path),
            "reference_count": 0,
        }
    )
    action6_targets: List[Dict[str, int]] = []
    level7_context = (rule_report.get("level7_frontier_context") or {}) if rule_report else {}
    center = level7_context.get("center") or {}
    if "x" in center and "y" in center:
        action6_targets.append({"x": int(center["x"]), "y": int(center["y"])})

    needs_action6_targets = (
        args.include_action6
        or args.breaker_include_action6
        or (args.action6_operator_probe_total > 0 and not args.disable_action6)
    )
    if needs_action6_targets:
        if args.scoring_mode == "dotted_constraint_repair":
            # Use boundary-based target generation focused on dotted violations
            root_grid = _primary_grid(base_raw)
            arr = np.array(root_grid, dtype=np.int32)
            height, width = arr.shape if arr.ndim == 2 else (64, 64)

            first_components = _connected_components_for_colors(root_grid, [rule_model.pair_colors[0]])
            second_components = _connected_components_for_colors(root_grid, [rule_model.pair_colors[1]])
            all_components = first_components + second_components
            violation_components = _get_boundary_violation_components(all_components, (height, width))
            cursor_components = _connected_components_for_colors(root_grid, (4, 5))

            if violation_components:
                action6_targets = generate_action6_targets_from_violation_boundaries(
                    root_grid,
                    violation_components=violation_components,
                    pair_colors=rule_model.pair_colors,
                    max_candidates=max(1, int(args.action6_candidates)),
                    cursor_components=cursor_components,
                )
            else:
                # Fallback to generic candidates if no violations detected
                seeds = list(action6_targets)
                action6_targets = _coordinate_candidates(
                    root_grid,
                    grid_size=16,
                    max_candidates=max(1, int(args.action6_candidates)),
                    seeds=seeds,
                )
        else:
            seeds = list(action6_targets)
            action6_targets = _coordinate_candidates(
                _primary_grid(base_raw),
                grid_size=16,
                max_candidates=max(1, int(args.action6_candidates)),
                seeds=seeds,
            )

    human_w3 = _human_level_action3_noop_window(
        selection.steps,
        level_start_index=frontier.level_start_index,
        terminal_index=frontier.terminal_index,
    )
    guided_search = run_guided_search(
        base_env=base_env,
        base_raw=base_raw,
        full_game_id=full_game_id,
        rule_model=rule_model,
        horizon=args.horizon,
        beam_width=args.beam_width,
        include_action6=bool(args.include_action6),
        action6_targets=action6_targets,
        danger_actions=frontier.danger_actions,
        scoring_mode=args.scoring_mode,
        submit_gate=args.submit_gate,
        max_submit_probes=args.max_submit_probes,
        strong_improvement_margin=args.strong_improvement_margin,
        forbid_action3_before_submit=args.forbid_action3_before_submit,
        disable_action2=args.disable_action2,
        disable_action6=args.disable_action6,
        post_saturation_probe=args.post_saturation_probe,
        rewrite_saturation_action3_threshold=args.rewrite_saturation_action3_threshold,
        human_w3=human_w3,
        best_match_submit_probes=args.best_match_submit_probes,
        action6_operator_probe_nodes=args.action6_operator_probe_nodes,
        action6_operator_probe_total=(
            0 if args.disable_action6 else args.action6_operator_probe_total
        ),
        local_repair_from_best_global=args.local_repair_from_best_global,
        local_repair_top_k=args.local_repair_top_k,
        local_repair_horizon=args.local_repair_horizon,
        local_repair_beam_width=args.local_repair_beam,
        local_repair_start_global_min=args.local_repair_start_global_min,
        local_repair_global_floor=args.local_repair_global_floor,
        local_repair_submit_probes=args.local_repair_submit_probes,
        targeted_repair_from_best_global=args.targeted_repair_from_best_global,
        targeted_repair_top_k=args.targeted_repair_top_k,
        targeted_repair_horizon=args.targeted_repair_horizon,
        targeted_repair_beam_width=args.targeted_repair_beam,
        targeted_repair_start_global_min=args.targeted_repair_start_global_min,
        targeted_repair_global_floor=args.targeted_repair_global_floor,
        targeted_repair_submit_probes=args.targeted_repair_submit_probes,
        targeted_repair_max_components=args.targeted_repair_max_components,
        targeted_repair_include_action6=bool(args.targeted_repair_include_action6 and not args.disable_action6),
        action_ontology_model=action_ontology_model,
        action_ontology_weight=args.action_ontology_weight,
        auto_levelup_state_classifier=auto_levelup_state_classifier,
        auto_levelup_classifier_weight=args.auto_levelup_classifier_weight,
        largest_second_pressure=bool(args.largest_second_pressure),
        probe_biggest_second_breaker=bool(args.probe_biggest_second_breaker),
        breaker_top_k=args.breaker_top_k,
        breaker_depth=args.breaker_depth,
        breaker_beam_width=args.breaker_beam,
        breaker_global_floor=args.breaker_global_floor,
        breaker_max_dotted=args.breaker_max_dotted,
        breaker_target_second_largest_max=args.breaker_target_second_largest_max,
        breaker_include_action6=bool(args.breaker_include_action6),
        breaker_stages=args.breaker_stages,
        breaker_stage_targets=list(args.breaker_stage_targets or []),
        breaker_context_model=breaker_context_model,
        enable_breaker_macro=bool(args.enable_breaker_macro),
        macro_break_target=float(args.macro_break_target),
        breaker_macro_submit_probes=int(args.breaker_macro_submit_probes),
    )
    payload = {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "target_level": int(args.target_level),
        "frontier": {
            "level_start_index": int(frontier.level_start_index),
            "terminal_index": int(frontier.terminal_index),
            "immediate_danger_action": frontier.immediate_danger_action,
            "danger_actions": list(frontier.danger_actions),
        },
        "rule_report": str(rule_report_path) if rule_report_path.exists() else None,
        "rule_model": rule_model.to_report(),
        "action_ontology_report": (
            str(action_ontology_path)
            if action_ontology_path.exists()
            else None
        ),
        "action_ontology_model": {
            key: value
            for key, value in action_ontology_model.items()
            if key != "action_priors"
        },
        "action_ontology_priors": action_ontology_model.get("action_priors", {}),
        "auto_levelup_state_classifier": {
            key: value
            for key, value in auto_levelup_state_classifier.items()
            if key != "references"
        },
        "auto_levelup_state_references": auto_levelup_state_classifier.get("references", []),
        "breaker_context_model": {
            key: value
            for key, value in breaker_context_model.items()
            if key != "references"
        },
        "action6_targets": list(action6_targets),
        "guided_search": guided_search,
        "window_comparison": guided_search.get("window_comparison"),
    }
    colors_suffix = f"colors{rule_model.pair_colors[0]}_{rule_model.pair_colors[1]}"
    action6_suffix = "with_a6" if args.include_action6 else "no_a6"
    scoring_suffix = f".{args.scoring_mode}" if args.scoring_mode != "default" else ""
    auto_scope_suffix = (
        f".scope_{args.auto_reference_scope.replace('-', '_')}"
        if args.scoring_mode in {
            "auto_levelup_state_classifier",
            "level7_fragmentation_guided",
            "level7_fragmentation_guarded",
        }
        else ""
    )
    largest_second_suffix = ".largest_second_pressure" if args.largest_second_pressure else ""
    breaker_suffix = ".biggest_second_breaker" if args.probe_biggest_second_breaker else ""
    breaker_a6_suffix = ".breaker_a6" if args.breaker_include_action6 else ""
    breaker_context_suffix = ".break_context" if args.breaker_context_similarity else ""
    breaker_macro_suffix = ".breaker_macro" if args.enable_breaker_macro else ""
    probe_suffix = ".post_saturation_probe" if args.post_saturation_probe else ""
    output = args.report_dir / (
        f"{full_game_id}.{selection.episode_id}.{colors_suffix}.{action6_suffix}{scoring_suffix}{auto_scope_suffix}{largest_second_suffix}{breaker_suffix}{breaker_a6_suffix}{breaker_context_suffix}{breaker_macro_suffix}{probe_suffix}.guided_level7.json"
    )
    if len(str(output)) >= 240:
        scoring_aliases = {
            "level7_fragmentation_guarded": "l7fg",
            "level7_fragmentation_guided": "l7f",
            "auto_levelup_state_classifier": "auto",
            "action_ontology_guided": "onto",
            "global_semantic_hybrid": "gsem",
            "global_correspondence": "glob",
            "rewrite_until_saturation": "rewrite",
        }
        compact_parts = [
            full_game_id,
            selection.episode_id,
            f"c{rule_model.pair_colors[0]}_{rule_model.pair_colors[1]}",
            "a6" if args.include_action6 else "noa6",
            scoring_aliases.get(args.scoring_mode, args.scoring_mode),
        ]
        if auto_scope_suffix:
            compact_parts.append(args.auto_reference_scope.replace("-", ""))
        if args.largest_second_pressure:
            compact_parts.append("lsp")
        if args.probe_biggest_second_breaker:
            compact_parts.append("bsb")
        if int(args.breaker_stages) > 1:
            compact_parts.append(f"st{int(args.breaker_stages)}")
        if args.breaker_include_action6:
            compact_parts.append("ba6")
        if args.breaker_context_similarity:
            compact_parts.append("bctx")
        if args.enable_breaker_macro:
            compact_parts.append("bmacro")
        if args.post_saturation_probe:
            compact_parts.append("post")
        output = args.report_dir / (".".join(compact_parts) + ".guided_level7.json")
    _write_report(output, payload)
    _print_report(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
