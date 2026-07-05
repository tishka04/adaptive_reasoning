"""Compare human and agent states with similar global correspondence.

The current AR25 diagnosis says the agent can reach states that look globally
coherent, but they still do not trigger the game's automatic level transition.
This script reconstructs agent states from a guided-search report, compares
them with human states in the same global-score band, and probes which
one-step action, if any, causes auto-level-up.

Example:
    python compare_human_agent_global_states.py --game ar25 --pair-colors 10 11
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from level7_frontier_recovery import (
    PROJECT_ROOT,
    Arcade,
    OperationMode,
    _available_names_from_raw,
    _diff_bbox,
    _hash_grid,
    _primary_grid,
    _replay_prefix,
    _resolve_full_game_id,
    _state_name,
    _step_branch,
    find_level_frontier,
)
from task_program_guided_level7 import (
    Component,
    _cached_global_correspondence_score,
    _connected_components_for_colors,
    _grid_changed_cells,
    _offending_component_diagnostics,
    _unmatched_total,
    global_correspondence_score,
    match_score,
)
from trace_replay_verifier import _load_selected_episode
from trace_rule_inference import EpisodeBundle, build_level_segments


DEFAULT_AGENT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "task_program_guided_level7"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "rule_inference"


def _parse_pair_colors(parts: Sequence[str]) -> Tuple[int, int]:
    values: List[str] = []
    for item in parts:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if len(values) != 2:
        raise ValueError("--pair-colors must look like '10 11' or '10,11'")
    return (int(values[0]), int(values[1]))


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _component_brief(component: Component) -> Dict[str, Any]:
    return {
        "color": int(component.color),
        "size": int(component.size),
        "bbox": {
            "min_y": int(component.min_y),
            "min_x": int(component.min_x),
            "max_y": int(component.max_y),
            "max_x": int(component.max_x),
        },
        "centroid": {
            "y": _round(component.centroid_y, 3),
            "x": _round(component.centroid_x, 3),
        },
        "height": int(component.height),
        "width": int(component.width),
    }


def _boundary_side_counts(
    components: Sequence[Component],
    *,
    shape: Tuple[int, int],
) -> Dict[str, Dict[str, int]]:
    height, width = shape
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"top": 0, "bottom": 0, "left": 0, "right": 0})
    for component in components:
        color = str(int(component.color))
        if component.min_y <= 0:
            counts[color]["top"] += 1
        if component.max_y >= height - 1:
            counts[color]["bottom"] += 1
        if component.min_x <= 0:
            counts[color]["left"] += 1
        if component.max_x >= width - 1:
            counts[color]["right"] += 1
    return {key: dict(value) for key, value in counts.items()}


def _relative_position_signature(
    first_components: Sequence[Component],
    second_components: Sequence[Component],
) -> Dict[str, Any]:
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
        "nearest_distance_mean": _round(sum(nearest) / max(1, len(nearest))),
        "nearest_distance_min": _round(min(nearest, default=0.0)),
        "nearest_distance_max": _round(max(nearest, default=0.0)),
        "mean_y_offset_second_minus_first": _round(sum(y_offsets) / max(1, len(y_offsets))),
        "mean_x_offset_second_minus_first": _round(sum(x_offsets) / max(1, len(x_offsets))),
        "samples": len(nearest),
    }


def _component_graph_report(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    arr = np.array(grid, dtype=np.int32)
    shape = arr.shape if arr.ndim == 2 else (64, 64)
    first_components = _connected_components_for_colors(grid, [pair_colors[0]])
    second_components = _connected_components_for_colors(grid, [pair_colors[1]])
    all_components = first_components + second_components
    by_color = {
        str(pair_colors[0]): first_components,
        str(pair_colors[1]): second_components,
    }
    return {
        "shape": [int(shape[0]), int(shape[1])],
        "component_counts": {color: len(components) for color, components in by_color.items()},
        "size_sums": {
            color: int(sum(component.size for component in components))
            for color, components in by_color.items()
        },
        "largest_components": {
            color: [
                _component_brief(component)
                for component in sorted(components, key=lambda item: item.size, reverse=True)[:5]
            ]
            for color, components in by_color.items()
        },
        "boundary_side_counts": _boundary_side_counts(all_components, shape=shape),
        "relative_positions": _relative_position_signature(first_components, second_components),
    }


def _score_state(
    grid: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
) -> Dict[str, Any]:
    local = match_score(grid, pair_colors=pair_colors)
    global_score = global_correspondence_score(grid, pair_colors=pair_colors)
    offenders = _offending_component_diagnostics(grid, pair_colors=pair_colors, limit=16)
    return {
        "hash": _hash_grid(grid),
        "match_score": local.to_report(),
        "global_correspondence_score": global_score.to_report(),
        "component_graph": _component_graph_report(grid, pair_colors=pair_colors),
        "offending_components": offenders,
        "causal_readiness_features": {
            "matched_pairs": int(local.matched_pairs),
            "unmatched_total": int(_unmatched_total(local)),
            "dotted_constraint_violations": int(local.dotted_constraint_violations),
            "offending_count": int(offenders.get("offending_count", 0)),
            "largest_offender_reason": (
                (offenders.get("largest_offender") or {}).get("reason")
            ),
            "largest_offender_color": (
                (offenders.get("largest_offender") or {}).get("color")
            ),
            "largest_offender_size": (
                (offenders.get("largest_offender") or {}).get("size")
            ),
        },
    }


def _state_transition_report(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    *,
    pair_colors: Tuple[int, int],
    state_after: Optional[str] = None,
    level_before: Optional[int] = None,
    level_after: Optional[int] = None,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    before_score = _score_state(before, pair_colors=pair_colors)
    after_score = _score_state(after, pair_colors=pair_colors)
    before_graph = before_score["component_graph"]
    after_graph = after_score["component_graph"]
    colors = [str(pair_colors[0]), str(pair_colors[1])]
    return {
        "available": True,
        "action": action,
        "state_after": state_after,
        "level_before": level_before,
        "level_after": level_after,
        "level_delta": (
            int(level_after) - int(level_before)
            if level_before is not None and level_after is not None
            else None
        ),
        "changed_cells": _grid_changed_cells(before, after),
        "diff_bbox": _diff_bbox(before, after),
        "global_delta": _round(
            after_score["global_correspondence_score"]["score"]
            - before_score["global_correspondence_score"]["score"]
        ),
        "match_delta": _round(after_score["match_score"]["score"] - before_score["match_score"]["score"]),
        "unmatched_delta": int(
            before_score["causal_readiness_features"]["unmatched_total"]
            - after_score["causal_readiness_features"]["unmatched_total"]
        ),
        "dotted_delta": int(
            before_score["causal_readiness_features"]["dotted_constraint_violations"]
            - after_score["causal_readiness_features"]["dotted_constraint_violations"]
        ),
        "offending_delta": int(
            before_score["causal_readiness_features"]["offending_count"]
            - after_score["causal_readiness_features"]["offending_count"]
        ),
        "component_count_delta": {
            color: int(after_graph["component_counts"].get(color, 0))
            - int(before_graph["component_counts"].get(color, 0))
            for color in colors
        },
        "size_sum_delta": {
            color: int(after_graph["size_sums"].get(color, 0))
            - int(before_graph["size_sums"].get(color, 0))
            for color in colors
        },
        "after_readiness_features": after_score["causal_readiness_features"],
    }


def _sample_report(
    *,
    label: str,
    source: str,
    grid: Sequence[Sequence[int]],
    pair_colors: Tuple[int, int],
    extra: Optional[Dict[str, Any]] = None,
    action2_effect: Optional[Dict[str, Any]] = None,
    auto_levelup_effect: Optional[Dict[str, Any]] = None,
    action_probe_effects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    action2_effect = action2_effect or {"available": False}
    auto_levelup_effect = auto_levelup_effect or {"available": False}
    action_probe_effects = action_probe_effects or []
    report = {
        "label": label,
        "source": source,
        **_score_state(grid, pair_colors=pair_colors),
        "action2_effect": action2_effect,
        "auto_levelup_effect": auto_levelup_effect,
        "action_probe_effects": action_probe_effects,
        "causal_correspondence_readiness": {
            "observed_auto_levelup_effect": _observed_auto_levelup_readiness(auto_levelup_effect),
            "legacy_action2_effect": _observed_auto_levelup_readiness(action2_effect),
        },
    }
    if extra:
        report.update(extra)
    return report


def _observed_auto_levelup_readiness(effect: Dict[str, Any]) -> Dict[str, Any]:
    """Score the causal trigger: what happens after the candidate final action?"""

    if not effect or not effect.get("available"):
        return {
            "available": False,
            "score": 0.0,
            "positive": False,
            "reason": "no_auto_levelup_effect",
        }
    level_delta = int(effect.get("level_delta") or 0)
    changed_cells = int(effect.get("changed_cells") or 0)
    unmatched_delta = int(effect.get("unmatched_delta") or 0)
    offending_delta = int(effect.get("offending_delta") or 0)
    dotted_delta = int(effect.get("dotted_delta") or 0)
    component_delta = effect.get("component_count_delta") or {}
    size_delta = effect.get("size_sum_delta") or {}
    component_rewrite = sum(abs(int(value)) for value in component_delta.values())
    mass_rewrite = sum(abs(int(value)) for value in size_delta.values())
    collapse = max(0, unmatched_delta) + max(0, offending_delta) + max(0, dotted_delta)
    damage = max(0, -unmatched_delta) + max(0, -offending_delta) + max(0, -dotted_delta)
    score = (
        1000.0 * float(level_delta > 0)
        + 180.0 * min(1.5, float(changed_cells) / 1000.0)
        + 80.0 * float(max(0, unmatched_delta))
        + 80.0 * float(max(0, offending_delta))
        + 25.0 * float(max(0, dotted_delta))
        + 18.0 * float(component_rewrite)
        + 0.02 * float(mass_rewrite)
        - 90.0 * float(damage)
    )
    return {
        "available": True,
        "score": _round(score),
        "positive": bool(level_delta > 0),
        "action": effect.get("action"),
        "level_delta": int(level_delta),
        "changed_cells": int(changed_cells),
        "unmatched_delta": int(unmatched_delta),
        "offending_delta": int(offending_delta),
        "dotted_delta": int(dotted_delta),
        "component_rewrite": int(component_rewrite),
        "mass_rewrite": int(mass_rewrite),
    }


def _global_score(report: Dict[str, Any]) -> float:
    return float((report.get("global_correspondence_score") or {}).get("score", 0.0))


def _in_global_band(report: Dict[str, Any], *, min_score: float, max_score: float) -> bool:
    score = _global_score(report)
    return float(min_score) <= score <= float(max_score)


def _static_readiness_features(sample: Dict[str, Any]) -> Dict[str, float]:
    graph = sample.get("component_graph") or {}
    readiness = sample.get("causal_readiness_features") or {}
    relative = graph.get("relative_positions") or {}
    counts = graph.get("component_counts") or {}
    size_sums = graph.get("size_sums") or {}
    boundary = graph.get("boundary_side_counts") or {}
    colors = sorted(counts)
    first = colors[0] if colors else "10"
    second = colors[1] if len(colors) > 1 else "11"
    first_boundary = boundary.get(first) or {}
    second_boundary = boundary.get(second) or {}
    boundary_total = sum(int(value) for value in first_boundary.values()) + sum(
        int(value) for value in second_boundary.values()
    )
    return {
        "global_score": float(_global_score(sample)),
        "matched_pairs": float(readiness.get("matched_pairs") or 0),
        "unmatched_total": float(readiness.get("unmatched_total") or 0),
        "dotted_violations": float(readiness.get("dotted_constraint_violations") or 0),
        "offending_count": float(readiness.get("offending_count") or 0),
        "largest_offender_size": float(readiness.get("largest_offender_size") or 0),
        "count_balance_abs": abs(float(counts.get(first) or 0) - float(counts.get(second) or 0)),
        "size_balance_abs": abs(float(size_sums.get(first) or 0) - float(size_sums.get(second) or 0)),
        "boundary_contact_total": float(boundary_total),
        "nearest_distance_mean": float(relative.get("nearest_distance_mean") or 0.0),
        "mean_y_offset": float(relative.get("mean_y_offset_second_minus_first") or 0.0),
        "mean_x_offset": float(relative.get("mean_x_offset_second_minus_first") or 0.0),
    }


def _fit_static_causal_readiness_model(
    positive_samples: Sequence[Dict[str, Any]],
    negative_samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    positives = [
        sample
        for sample in positive_samples
        if ((sample.get("auto_levelup_effect") or {}).get("level_delta") or 0) > 0
    ]
    negatives = [
        sample
        for sample in negative_samples
        if ((sample.get("auto_levelup_effect") or {}).get("level_delta") or 0) <= 0
    ]
    if not positives or not negatives:
        return {
            "available": False,
            "reason": "missing_positive_or_negative_samples",
            "positive_count": len(positives),
            "negative_count": len(negatives),
        }

    feature_names = sorted(_static_readiness_features(positives[0]))
    pos_features = [_static_readiness_features(sample) for sample in positives]
    neg_features = [_static_readiness_features(sample) for sample in negatives]

    weights: Dict[str, float] = {}
    midpoint: Dict[str, float] = {}
    scale: Dict[str, float] = {}
    pos_mean: Dict[str, float] = {}
    neg_mean: Dict[str, float] = {}
    for name in feature_names:
        pos_values = [float(item.get(name, 0.0)) for item in pos_features]
        neg_values = [float(item.get(name, 0.0)) for item in neg_features]
        p_mean = sum(pos_values) / max(1, len(pos_values))
        n_mean = sum(neg_values) / max(1, len(neg_values))
        all_values = pos_values + neg_values
        variance = sum((value - (p_mean + n_mean) / 2.0) ** 2 for value in all_values) / max(1, len(all_values))
        feature_scale = max(1.0, variance ** 0.5)
        pos_mean[name] = p_mean
        neg_mean[name] = n_mean
        midpoint[name] = 0.5 * (p_mean + n_mean)
        scale[name] = feature_scale
        weights[name] = (p_mean - n_mean) / feature_scale

    return {
        "available": True,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_source": "human_auto_levelup_reference",
        "negative_source": "agent_top_global_no_auto_levelup",
        "feature_names": feature_names,
        "weights": {name: _round(weights[name], 6) for name in feature_names},
        "midpoint": {name: _round(midpoint[name], 6) for name in feature_names},
        "scale": {name: _round(scale[name], 6) for name in feature_names},
        "positive_mean": {name: _round(pos_mean[name], 6) for name in feature_names},
        "negative_mean": {name: _round(neg_mean[name], 6) for name in feature_names},
    }


def _estimate_static_causal_readiness(sample: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    if not model.get("available"):
        return {
            "available": False,
            "score": 0.0,
            "probability_like": 0.0,
            "reason": model.get("reason", "model_unavailable"),
        }
    features = _static_readiness_features(sample)
    score = 0.0
    contributions: Dict[str, float] = {}
    for name in model.get("feature_names") or []:
        weight = float((model.get("weights") or {}).get(name, 0.0))
        midpoint = float((model.get("midpoint") or {}).get(name, 0.0))
        scale = max(1.0, float((model.get("scale") or {}).get(name, 1.0)))
        contribution = weight * ((float(features.get(name, 0.0)) - midpoint) / scale)
        contributions[name] = contribution
        score += contribution
    probability_like = float(1.0 / (1.0 + np.exp(-max(-20.0, min(20.0, score)))))
    return {
        "available": True,
        "score": _round(score),
        "probability_like": _round(probability_like),
        "features": {name: _round(value, 6) for name, value in features.items()},
        "top_positive_contributions": {
            name: _round(value, 6)
            for name, value in sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:5]
        },
        "top_negative_contributions": {
            name: _round(value, 6)
            for name, value in sorted(contributions.items(), key=lambda item: item[1])[:5]
        },
    }


def _annotate_static_causal_readiness(
    samples: Sequence[Dict[str, Any]],
    *,
    model: Dict[str, Any],
) -> None:
    for sample in samples:
        causal = sample.setdefault("causal_correspondence_readiness", {})
        causal["static_estimate"] = _estimate_static_causal_readiness(sample, model)


def _human_samples(
    steps: Sequence[Any],
    *,
    pair_colors: Tuple[int, int],
    around_level: int,
    max_level: int,
    min_global: float,
    max_global: float,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    bundle = EpisodeBundle(episode=None, steps=list(steps))
    for segment in build_level_segments(bundle, max_level=max_level):
        effect = _state_transition_report(
            segment.level_up_before_grid,
            segment.level_up_after_grid,
            pair_colors=pair_colors,
            state_after=segment.state_after,
            level_before=segment.level_number - 1,
            level_after=segment.level_number,
            action=segment.level_up_action,
        )
        sample = _sample_report(
            label=f"human_pre_levelup_{segment.level_number}",
            source="human_pre_levelup",
            grid=segment.level_up_before_grid,
            pair_colors=pair_colors,
            extra={
                "level_number": int(segment.level_number),
                "trace_start_step": int(segment.trace_start_step),
                "trace_end_step": int(segment.trace_end_step),
                "action": segment.level_up_action,
                "action_tail": segment.actions[-12:],
            },
            action2_effect=effect if segment.level_up_action == "ACTION2" else {"available": False},
            auto_levelup_effect=effect,
        )
        if _in_global_band(sample, min_score=min_global, max_score=max_global):
            samples.append(sample)

    try:
        frontier = find_level_frontier(
            type("Selection", (), {"steps": list(steps)})(),
            target_level=around_level,
            danger_window=8,
        )
    except Exception:
        return samples

    for idx in range(frontier.level_start_index, frontier.terminal_index + 1):
        step = steps[idx]
        action = getattr(step, "action", "")
        previous_level = (
            int(getattr(steps[idx - 1], "levels_completed_after", 0) or 0)
            if idx > 0
            else max(0, int(step.levels_completed_after) - 1)
        )
        effect = _state_transition_report(
            step.frame_before,
            step.frame_after,
            pair_colors=pair_colors,
            state_after=step.game_state_after,
            level_before=previous_level,
            level_after=int(step.levels_completed_after),
            action=action,
        )
        action2_effect = {"available": False}
        if action == "ACTION2":
            action2_effect = effect
        sample = _sample_report(
            label=f"human_level{around_level}_before_{idx}",
            source="human_level_window",
            grid=step.frame_before,
            pair_colors=pair_colors,
            extra={
                "trace_index": int(idx),
                "trace_step": int(step.step),
                "action": action,
                "state_after": step.game_state_after,
                "levels_completed_after": int(step.levels_completed_after),
            },
            action2_effect=action2_effect,
            auto_levelup_effect=effect,
        )
        if _in_global_band(sample, min_score=min_global, max_score=max_global):
            samples.append(sample)
    return samples


def _human_auto_levelup_reference_samples(
    steps: Sequence[Any],
    *,
    pair_colors: Tuple[int, int],
    max_level: int,
) -> List[Dict[str, Any]]:
    bundle = EpisodeBundle(episode=None, steps=list(steps))
    samples: List[Dict[str, Any]] = []
    for segment in build_level_segments(bundle, max_level=max_level):
        effect = _state_transition_report(
            segment.level_up_before_grid,
            segment.level_up_after_grid,
            pair_colors=pair_colors,
            state_after=segment.state_after,
            level_before=segment.level_number - 1,
            level_after=segment.level_number,
            action=segment.level_up_action,
        )
        samples.append(
            _sample_report(
                label=f"human_auto_levelup_reference_{segment.level_number}",
                source="human_auto_levelup_reference",
                grid=segment.level_up_before_grid,
                pair_colors=pair_colors,
                extra={
                    "level_number": int(segment.level_number),
                    "trace_start_step": int(segment.trace_start_step),
                    "trace_end_step": int(segment.trace_end_step),
                    "action": segment.level_up_action,
                    "action_tail": segment.actions[-12:],
                },
                action2_effect=effect if segment.level_up_action == "ACTION2" else {"available": False},
                auto_levelup_effect=effect,
            )
        )
    return samples


def _default_agent_report_path(game_id: str, pair_colors: Tuple[int, int]) -> Path:
    suffix = f"colors{pair_colors[0]}_{pair_colors[1]}.no_a6.global_semantic_hybrid.guided_level7.json"
    return DEFAULT_AGENT_REPORT_DIR / f"{game_id}.f0fe23029ffa.{suffix}"


def _walk_node_reports(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("actions"), list) and "match_score" in value:
            yield value
        for child in value.values():
            yield from _walk_node_reports(child)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_node_reports(item)


def _agent_candidate_reports(agent_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    search = agent_report.get("guided_search") or {}
    candidates: List[Dict[str, Any]] = []
    for key in (
        "best",
        "best_global",
        "best_match",
        "top",
        "top_global",
        "top_match",
        "best_match_submit_probes",
        "local_repair_from_best_global",
        "targeted_repair_from_best_global",
    ):
        if key in search:
            candidates.extend(_walk_node_reports(search[key]))
    return candidates


def _one_step_action_probe_effects(
    *,
    branch_env: Any,
    raw: Any,
    full_game_id: str,
    pair_colors: Tuple[int, int],
) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    actions = _available_names_from_raw(raw) or []
    before_level = int(getattr(raw, "levels_completed", 0) or 0)
    effects: List[Dict[str, Any]] = []
    for action in actions:
        if action == "RESET":
            continue
        variants = [{"x": 32, "y": 32}] if action == "ACTION6" else [None]
        for action_data in variants:
            probe_env = copy.deepcopy(branch_env)
            before = _primary_grid(getattr(probe_env, "observation_space", None))
            probe_raw = _step_branch(
                probe_env,
                full_game_id=full_game_id,
                action=action,
                action_data=action_data,
            )
            if probe_raw is None:
                continue
            effect = _state_transition_report(
                before,
                _primary_grid(probe_raw),
                pair_colors=pair_colors,
                state_after=_state_name(getattr(probe_raw, "state", "UNKNOWN")),
                level_before=before_level,
                level_after=int(getattr(probe_raw, "levels_completed", before_level) or before_level),
                action=action,
            )
            effect["action_data"] = action_data
            effect["auto_levelup_readiness"] = _observed_auto_levelup_readiness(effect)
            effects.append(effect)
    effects.sort(
        key=lambda item: (
            int(item.get("level_delta") or 0),
            float((item.get("auto_levelup_readiness") or {}).get("score", 0.0)),
            int(item.get("offending_delta") or 0),
            int(item.get("unmatched_delta") or 0),
            int(item.get("changed_cells") or 0),
        ),
        reverse=True,
    )
    return effects


def _replay_agent_node(
    *,
    arc: Arcade,
    full_game_id: str,
    base_env: Any,
    node: Dict[str, Any],
    pair_colors: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    actions = list(node.get("actions") or [])
    action_data = list(node.get("action_data") or [])
    branch_env = copy.deepcopy(base_env)
    raw = getattr(branch_env, "observation_space", None)
    before_level = int(getattr(raw, "levels_completed", 7) or 7)
    for index, action in enumerate(actions):
        data = action_data[index] if index < len(action_data) else None
        raw = _step_branch(
            branch_env,
            full_game_id=full_game_id,
            action=action,
            action_data=data,
        )
        if raw is None:
            return None
    grid = _primary_grid(raw)

    action2_effect: Dict[str, Any] = {"available": False}
    action_probe_effects = _one_step_action_probe_effects(
        branch_env=branch_env,
        raw=raw,
        full_game_id=full_game_id,
        pair_colors=pair_colors,
    )
    auto_levelup_effect = action_probe_effects[0] if action_probe_effects else {"available": False}
    for effect in action_probe_effects:
        if effect.get("action") == "ACTION2":
            action2_effect = effect
            break
    if raw is not None:
        before_level = int(getattr(raw, "levels_completed", before_level) or before_level)

    return _sample_report(
        label=f"agent_{_hash_grid(grid)}",
        source="agent_guided_search",
        grid=grid,
        pair_colors=pair_colors,
        extra={
            "actions": actions,
            "action_data": action_data,
            "reported_grid_hash": node.get("grid_hash"),
            "reported_search_score": node.get("search_score"),
            "reported_ready": node.get("readiness_gate_passed"),
        },
        action2_effect=action2_effect,
        auto_levelup_effect=auto_levelup_effect,
        action_probe_effects=action_probe_effects,
    )


def _agent_samples(
    *,
    arc: Arcade,
    full_game_id: str,
    base_env: Any,
    agent_report: Dict[str, Any],
    pair_colors: Tuple[int, int],
    min_global: float,
    max_global: float,
    limit: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    candidates = sorted(
        _agent_candidate_reports(agent_report),
        key=lambda item: float((item.get("global_correspondence_score") or {}).get("score", 0.0)),
        reverse=True,
    )
    for node in candidates:
        if len(out) >= limit:
            break
        report = _replay_agent_node(
            arc=arc,
            full_game_id=full_game_id,
            base_env=base_env,
            node=node,
            pair_colors=pair_colors,
        )
        if report is None:
            continue
        key = report["hash"]
        if key in seen:
            continue
        if not _in_global_band(report, min_score=min_global, max_score=max_global):
            continue
        seen.add(key)
        out.append(report)
    return out


def _summarize_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"count": 0}
    globals_ = [_global_score(item) for item in samples]
    unmatched = [
        int((item.get("causal_readiness_features") or {}).get("unmatched_total", 0))
        for item in samples
    ]
    dotted = [
        int((item.get("causal_readiness_features") or {}).get("dotted_constraint_violations", 0))
        for item in samples
    ]
    offenders = [
        int((item.get("causal_readiness_features") or {}).get("offending_count", 0))
        for item in samples
    ]
    auto_effects = [
        item.get("auto_levelup_effect") or {}
        for item in samples
        if (item.get("auto_levelup_effect") or {}).get("available")
    ]
    action2_effects = [
        item.get("action2_effect") or {}
        for item in samples
        if (item.get("action2_effect") or {}).get("available")
    ]
    level_ups = [item for item in auto_effects if int(item.get("level_delta") or 0) > 0]
    static_scores = [
        float(
            ((item.get("causal_correspondence_readiness") or {}).get("static_estimate") or {}).get(
                "score",
                0.0,
            )
        )
        for item in samples
        if ((item.get("causal_correspondence_readiness") or {}).get("static_estimate") or {}).get("available")
    ]
    reasons = Counter(
        str((item.get("causal_readiness_features") or {}).get("largest_offender_reason"))
        for item in samples
    )
    return {
        "count": len(samples),
        "global_min": _round(min(globals_)),
        "global_max": _round(max(globals_)),
        "global_avg": _round(sum(globals_) / len(globals_)),
        "unmatched_min": min(unmatched),
        "unmatched_avg": _round(sum(unmatched) / len(unmatched)),
        "dotted_min": min(dotted),
        "dotted_avg": _round(sum(dotted) / len(dotted)),
        "offending_min": min(offenders),
        "offending_avg": _round(sum(offenders) / len(offenders)),
        "largest_offender_reasons": dict(reasons.most_common(8)),
        "static_causal_readiness": {
            "count": len(static_scores),
            "avg": _round(sum(static_scores) / max(1, len(static_scores))),
            "min": _round(min(static_scores, default=0.0)),
            "max": _round(max(static_scores, default=0.0)),
        },
        "auto_levelup_effects": {
            "count": len(auto_effects),
            "level_up_count": len(level_ups),
            "avg_changed_cells": _round(
                sum(int(item.get("changed_cells") or 0) for item in auto_effects)
                / max(1, len(auto_effects))
            ),
            "avg_offending_delta": _round(
                sum(float(item.get("offending_delta") or 0.0) for item in auto_effects)
                / max(1, len(auto_effects))
            ),
            "avg_unmatched_delta": _round(
                sum(float(item.get("unmatched_delta") or 0.0) for item in auto_effects)
                / max(1, len(auto_effects))
            ),
            "avg_dotted_delta": _round(
                sum(float(item.get("dotted_delta") or 0.0) for item in auto_effects)
                / max(1, len(auto_effects))
            ),
            "actions": dict(Counter(str(item.get("action")) for item in auto_effects)),
        },
        "action2_effects_legacy": {
            "count": len(action2_effects),
            "level_up_count": sum(1 for item in action2_effects if int(item.get("level_delta") or 0) > 0),
            "avg_changed_cells": _round(
                sum(int(item.get("changed_cells") or 0) for item in action2_effects)
                / max(1, len(action2_effects))
            ),
        },
    }


def _nearest_pairs(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for agent in left:
        if not right:
            break
        human = min(right, key=lambda item: abs(_global_score(item) - _global_score(agent)))
        agent_ready = agent.get("causal_readiness_features") or {}
        human_ready = human.get("causal_readiness_features") or {}
        pairs.append(
            {
                "agent_label": agent["label"],
                "human_label": human["label"],
                "global_delta_agent_minus_human": _round(_global_score(agent) - _global_score(human)),
                "agent_global": _global_score(agent),
                "human_global": _global_score(human),
                "agent_unmatched": agent_ready.get("unmatched_total"),
                "human_unmatched": human_ready.get("unmatched_total"),
                "agent_dotted": agent_ready.get("dotted_constraint_violations"),
                "human_dotted": human_ready.get("dotted_constraint_violations"),
                "agent_offending": agent_ready.get("offending_count"),
                "human_offending": human_ready.get("offending_count"),
                "agent_largest_offender": {
                    "reason": agent_ready.get("largest_offender_reason"),
                    "color": agent_ready.get("largest_offender_color"),
                    "size": agent_ready.get("largest_offender_size"),
                },
                "human_largest_offender": {
                    "reason": human_ready.get("largest_offender_reason"),
                    "color": human_ready.get("largest_offender_color"),
                    "size": human_ready.get("largest_offender_size"),
                },
                "agent_auto_levelup_effect": agent.get("auto_levelup_effect"),
                "human_auto_levelup_effect": human.get("auto_levelup_effect"),
                "agent_action2_effect_legacy": agent.get("action2_effect"),
                "human_action2_effect_legacy": human.get("action2_effect"),
            }
        )
    return pairs[:limit]


def _print_report(payload: Dict[str, Any], output: Path) -> None:
    print("=" * 88)
    print("Human/agent global-state differential")
    print("=" * 88)
    print(f"game:        {payload['game_id']}")
    print(f"episode:     {payload['episode_id']}")
    print(f"pair colors: {payload['pair_colors']}")
    print(f"global band: {payload['global_band']}")
    print(f"human:       {payload['human_summary']}")
    print(f"human auto:  {payload['human_auto_levelup_reference_summary']}")
    print(f"agent:       {payload['agent_summary']}")
    model = payload.get("static_causal_readiness_model") or {}
    if model.get("available"):
        print(
            "static model:"
            f" positives={model.get('positive_count')} negatives={model.get('negative_count')}"
        )
    nearest = payload.get("nearest_agent_human_pairs") or []
    if nearest:
        first = nearest[0]
        print(
            "nearest:     "
            f"agent_global={first['agent_global']} human_global={first['human_global']} "
            f"agent_unmatched={first['agent_unmatched']} human_unmatched={first['human_unmatched']} "
            f"agent_dotted={first['agent_dotted']} human_dotted={first['human_dotted']}"
        )
    print(f"json:        {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--pair-colors", nargs="+", required=True)
    parser.add_argument("--around-level", type=int, default=7)
    parser.add_argument("--max-level", type=int, default=7)
    parser.add_argument("--min-global", type=float, default=60.0)
    parser.add_argument("--max-global", type=float, default=70.0)
    parser.add_argument("--agent-report", type=Path, default=None)
    parser.add_argument("--agent-limit", type=int, default=24)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    pair_colors = _parse_pair_colors(args.pair_colors)
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
        target_level=int(args.around_level),
        danger_window=8,
    )
    base_env, _base_raw = _replay_prefix(
        arc,
        full_game_id,
        selection.steps,
        stop_before_index=frontier.level_start_index,
        expected_frame=frontier.level_start_frame,
    )
    agent_report_path = args.agent_report or _default_agent_report_path(full_game_id, pair_colors)
    agent_report = json.loads(agent_report_path.read_text(encoding="utf-8"))

    human = _human_samples(
        selection.steps,
        pair_colors=pair_colors,
        around_level=int(args.around_level),
        max_level=int(args.max_level),
        min_global=float(args.min_global),
        max_global=float(args.max_global),
    )
    human_auto_levelup_reference = _human_auto_levelup_reference_samples(
        selection.steps,
        pair_colors=pair_colors,
        max_level=int(args.max_level),
    )
    agent = _agent_samples(
        arc=arc,
        full_game_id=full_game_id,
        base_env=base_env,
        agent_report=agent_report,
        pair_colors=pair_colors,
        min_global=float(args.min_global),
        max_global=float(args.max_global),
        limit=max(1, int(args.agent_limit)),
    )
    static_model = _fit_static_causal_readiness_model(human_auto_levelup_reference, agent)
    for samples in (human, human_auto_levelup_reference, agent):
        _annotate_static_causal_readiness(samples, model=static_model)
    payload = {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "agent_report": str(agent_report_path),
        "pair_colors": list(pair_colors),
        "around_level": int(args.around_level),
        "global_band": {
            "min": float(args.min_global),
            "max": float(args.max_global),
        },
        "frontier": {
            "level_start_index": int(frontier.level_start_index),
            "terminal_index": int(frontier.terminal_index),
            "immediate_danger_action": frontier.immediate_danger_action,
            "danger_actions": list(frontier.danger_actions),
        },
        "static_causal_readiness_model": static_model,
        "human_summary": _summarize_samples(human),
        "human_auto_levelup_reference_summary": _summarize_samples(human_auto_levelup_reference),
        "agent_summary": _summarize_samples(agent),
        "nearest_agent_human_pairs": _nearest_pairs(agent, human, limit=8),
        "human_auto_levelup_reference": sorted(human_auto_levelup_reference, key=_global_score, reverse=True)[:32],
        "human_samples": sorted(human, key=_global_score, reverse=True)[:64],
        "agent_samples": sorted(agent, key=_global_score, reverse=True)[:64],
    }
    output = (
        args.report_dir
        / f"{full_game_id}.human_agent_global_diff.colors{pair_colors[0]}_{pair_colors[1]}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_report(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
