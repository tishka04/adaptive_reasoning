"""Empirically discover what ARC actions do in a game context.

Actions are treated as unknown operators. The script probes ACTION_i on human
and optional agent states, measures grid deltas, and infers an empirical
ontology such as translation-like, selector/control-like, transform-like,
auto-level-up trigger, no-op, or context-dependent.

Example:
    python discover_action_ontology.py --game ar25 --pair-colors 10 11 --target-level 7
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from level7_frontier_recovery import (
    ACTION_NAMES,
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
from task_program_guided_level7 import _grid_changed_cells
from trace_replay_verifier import _load_selected_episode
from trace_rule_inference import EpisodeBundle, build_level_segments


DEFAULT_AGENT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "task_program_guided_level7"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "action_ontology"


@dataclass(frozen=True)
class OntologyComponent:
    color: int
    size: int
    min_y: int
    min_x: int
    max_y: int
    max_x: int
    centroid_y: float
    centroid_x: float

    def to_report(self) -> Dict[str, Any]:
        return {
            "color": int(self.color),
            "size": int(self.size),
            "bbox": {
                "min_y": int(self.min_y),
                "min_x": int(self.min_x),
                "max_y": int(self.max_y),
                "max_x": int(self.max_x),
            },
            "centroid": {
                "y": round(float(self.centroid_y), 3),
                "x": round(float(self.centroid_x), 3),
            },
            "height": int(self.max_y - self.min_y + 1),
            "width": int(self.max_x - self.min_x + 1),
        }


def _parse_pair_colors(parts: Optional[Sequence[str]]) -> Optional[Tuple[int, int]]:
    if not parts:
        return None
    values: List[str] = []
    for item in parts:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if len(values) != 2:
        raise ValueError("--pair-colors must look like '10 11' or '10,11'")
    return (int(values[0]), int(values[1]))


def _background_values(grid: Sequence[Sequence[int]]) -> Set[int]:
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2 or arr.size == 0:
        return {0}
    values, counts = np.unique(arr, return_counts=True)
    dominant = int(values[int(np.argmax(counts))])
    return {0, dominant}


def _connected_components_all(
    grid: Sequence[Sequence[int]],
    *,
    ignore_values: Optional[Set[int]] = None,
) -> List[OntologyComponent]:
    ignore = set(ignore_values or set())
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2:
        return []
    height, width = arr.shape
    seen = np.zeros(arr.shape, dtype=bool)
    components: List[OntologyComponent] = []
    for y in range(height):
        for x in range(width):
            color = int(arr[y, x])
            if color in ignore or seen[y, x]:
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
            components.append(
                OntologyComponent(
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
    return components


def _component_stats(components: Sequence[OntologyComponent]) -> Dict[str, Dict[str, int]]:
    counts: Counter[int] = Counter(component.color for component in components)
    sizes: Counter[int] = Counter()
    for component in components:
        sizes[component.color] += int(component.size)
    colors = sorted(set(counts) | set(sizes))
    return {
        "counts": {str(color): int(counts[color]) for color in colors},
        "size_sums": {str(color): int(sizes[color]) for color in colors},
    }


def _value_transition_counts(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
    *,
    limit: int = 16,
) -> Dict[str, int]:
    left = np.array(before, dtype=np.int32)
    right = np.array(after, dtype=np.int32)
    if left.shape != right.shape:
        return {"shape_mismatch": 1}
    mask = left != right
    transitions: Counter[str] = Counter()
    for old, new in zip(left[mask].ravel(), right[mask].ravel()):
        transitions[f"{int(old)}->{int(new)}"] += 1
    return dict(transitions.most_common(limit))


def _affected_colors(
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


def _component_count_delta(
    before_stats: Dict[str, Dict[str, int]],
    after_stats: Dict[str, Dict[str, int]],
    key: str,
) -> Dict[str, int]:
    before = before_stats.get(key) or {}
    after = after_stats.get(key) or {}
    colors = sorted(set(before) | set(after), key=lambda item: int(item))
    return {
        color: int(after.get(color, 0)) - int(before.get(color, 0))
        for color in colors
    }


def _match_component_shifts(
    before_components: Sequence[OntologyComponent],
    after_components: Sequence[OntologyComponent],
    *,
    affected: Set[int],
) -> List[Dict[str, Any]]:
    after_by_color_size: Dict[Tuple[int, int], List[OntologyComponent]] = defaultdict(list)
    for component in after_components:
        after_by_color_size[(component.color, component.size)].append(component)

    shifts: List[Dict[str, Any]] = []
    used: Set[Tuple[int, int, int, int, int, int]] = set()
    for before in before_components:
        if affected and before.color not in affected:
            continue
        candidates = after_by_color_size.get((before.color, before.size), [])
        best: Optional[OntologyComponent] = None
        best_dist = float("inf")
        for candidate in candidates:
            key = (
                candidate.color,
                candidate.size,
                candidate.min_y,
                candidate.min_x,
                candidate.max_y,
                candidate.max_x,
            )
            if key in used:
                continue
            dist = float(
                np.hypot(
                    candidate.centroid_y - before.centroid_y,
                    candidate.centroid_x - before.centroid_x,
                )
            )
            if dist < best_dist:
                best = candidate
                best_dist = dist
        if best is None:
            continue
        used.add((best.color, best.size, best.min_y, best.min_x, best.max_y, best.max_x))
        dy = float(best.centroid_y - before.centroid_y)
        dx = float(best.centroid_x - before.centroid_x)
        shifts.append(
            {
                "color": int(before.color),
                "size": int(before.size),
                "dy": round(dy, 4),
                "dx": round(dx, 4),
                "distance": round(float(np.hypot(dy, dx)), 4),
                "before": before.to_report(),
                "after": best.to_report(),
            }
        )
    return shifts


def _direction_label(dy: float, dx: float) -> str:
    if abs(dy) < 0.25 and abs(dx) < 0.25:
        return "none"
    vertical = "down" if dy > 0 else "up"
    horizontal = "right" if dx > 0 else "left"
    if abs(dy) >= abs(dx) * 1.5:
        return vertical
    if abs(dx) >= abs(dy) * 1.5:
        return horizontal
    return f"{vertical}_{horizontal}"


def _translation_summary(shifts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    moving = [
        item
        for item in shifts
        if abs(float(item.get("dy", 0.0))) + abs(float(item.get("dx", 0.0))) >= 0.25
    ]
    if not moving:
        return {
            "translation_like": False,
            "moving_components": 0,
            "direction_candidate": "none",
            "mean_dy": 0.0,
            "mean_dx": 0.0,
            "consistency": 0.0,
        }
    mean_dy = float(mean(float(item["dy"]) for item in moving))
    mean_dx = float(mean(float(item["dx"]) for item in moving))
    direction = _direction_label(mean_dy, mean_dx)
    consistent = sum(
        1
        for item in moving
        if abs(float(item["dy"]) - mean_dy) <= 0.75
        and abs(float(item["dx"]) - mean_dx) <= 0.75
    )
    consistency = float(consistent / max(1, len(moving)))
    return {
        "translation_like": bool(consistency >= 0.6),
        "moving_components": len(moving),
        "direction_candidate": direction,
        "mean_dy": round(mean_dy, 4),
        "mean_dx": round(mean_dx, 4),
        "consistency": round(consistency, 4),
        "sample_shifts": list(moving[:6]),
    }


def _classify_probe(
    *,
    changed_cells: int,
    level_delta: int,
    affected: Sequence[int],
    pair_colors: Optional[Tuple[int, int]],
    count_delta: Dict[str, int],
    size_delta: Dict[str, int],
    translation: Dict[str, Any],
) -> Tuple[str, float]:
    if level_delta > 0:
        return "auto_levelup_trigger", 1.0
    if changed_cells <= 0:
        return "no_op", 1.0

    pair_set = set(pair_colors or ())
    affected_set = set(int(color) for color in affected)
    affected_pair = bool(pair_set & affected_set)
    component_count_change = sum(abs(int(value)) for value in count_delta.values())
    size_change = sum(abs(int(value)) for value in size_delta.values())

    if translation.get("translation_like") and component_count_change == 0 and size_change == 0:
        return "translation_like", float(0.6 + 0.4 * float(translation.get("consistency", 0.0)))

    if pair_colors and not affected_pair and changed_cells <= 256:
        return "selector_or_control_like", 0.72

    if component_count_change == 0 and size_change == 0 and changed_cells <= 128:
        return "local_state_or_cursor_update", 0.58

    if component_count_change > 0 or size_change > 0 or changed_cells >= 256:
        return "transform_like", 0.72

    return "contextual_or_mixed", 0.45


def _probe_action(
    *,
    env: Any,
    raw: Any,
    full_game_id: str,
    action: str,
    action_data: Optional[Dict[str, int]],
    pair_colors: Optional[Tuple[int, int]],
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    before_grid = _primary_grid(raw)
    before_level = int(getattr(raw, "levels_completed", 0) or 0)
    before_state = _state_name(getattr(raw, "state", "UNKNOWN"))
    branch_env = copy.deepcopy(env)
    after_raw = _step_branch(
        branch_env,
        full_game_id=full_game_id,
        action=action,
        action_data=action_data,
    )
    after_grid = _primary_grid(after_raw)
    after_level = int(getattr(after_raw, "levels_completed", before_level) or before_level)
    after_state = _state_name(getattr(after_raw, "state", "UNKNOWN"))
    changed_cells = _grid_changed_cells(before_grid, after_grid)
    affected = _affected_colors(before_grid, after_grid)

    ignore_values = _background_values(before_grid) | _background_values(after_grid)
    before_components = _connected_components_all(before_grid, ignore_values=ignore_values)
    after_components = _connected_components_all(after_grid, ignore_values=ignore_values)
    before_stats = _component_stats(before_components)
    after_stats = _component_stats(after_components)
    count_delta = _component_count_delta(before_stats, after_stats, "counts")
    size_delta = _component_count_delta(before_stats, after_stats, "size_sums")
    shifts = _match_component_shifts(before_components, after_components, affected=set(affected))
    translation = _translation_summary(shifts)
    operator_type, confidence = _classify_probe(
        changed_cells=changed_cells,
        level_delta=after_level - before_level,
        affected=affected,
        pair_colors=pair_colors,
        count_delta=count_delta,
        size_delta=size_delta,
        translation=translation,
    )
    return {
        "sample": dict(sample),
        "action": action,
        "action_data": dict(action_data) if action_data else None,
        "before_hash": _hash_grid(before_grid),
        "after_hash": _hash_grid(after_grid),
        "before_state": before_state,
        "after_state": after_state,
        "before_level": int(before_level),
        "after_level": int(after_level),
        "level_delta": int(after_level - before_level),
        "changed_cells": int(changed_cells),
        "diff_bbox": _diff_bbox(before_grid, after_grid),
        "affected_colors": affected,
        "value_transitions": _value_transition_counts(before_grid, after_grid),
        "component_count_delta": count_delta,
        "component_size_delta": size_delta,
        "translation": translation,
        "operator_type": operator_type,
        "confidence": round(float(confidence), 4),
    }


def _default_agent_report_path(game_id: str, pair_colors: Optional[Tuple[int, int]]) -> Optional[Path]:
    if pair_colors is None:
        return None
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


def _agent_node_candidates(agent_report: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    search = agent_report.get("guided_search") or {}
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, ...]] = set()
    for key in ("best", "best_global", "top_global", "top", "targeted_repair_from_best_global"):
        for node in _walk_node_reports(search.get(key)):
            actions = tuple(str(action) for action in (node.get("actions") or []))
            if actions in seen:
                continue
            seen.add(actions)
            out.append(node)
            if len(out) >= limit:
                return out
    return out


def _human_state_specs(selection: Any, *, target_level: int, stride: int, max_states: int) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    step_index_by_step = {int(step.step): idx for idx, step in enumerate(selection.steps)}
    bundle = EpisodeBundle(episode=None, steps=list(selection.steps))
    for segment in build_level_segments(bundle, max_level=target_level):
        idx = step_index_by_step.get(int(segment.trace_end_step))
        if idx is None:
            continue
        specs.append(
            {
                "source": "human_pre_auto_levelup",
                "label": f"human_pre_levelup_{segment.level_number}",
                "step_index": int(idx),
                "trace_step": int(segment.trace_end_step),
                "next_action": segment.level_up_action,
            }
        )

    frontier = find_level_frontier(selection, target_level=target_level, danger_window=8)
    selected: Set[int] = set()
    for idx in range(frontier.level_start_index, frontier.terminal_index + 1):
        if idx < frontier.level_start_index + 8:
            selected.add(idx)
        if stride > 0 and (idx - frontier.level_start_index) % stride == 0:
            selected.add(idx)
    for idx in sorted(selected):
        if idx >= len(selection.steps):
            continue
        step = selection.steps[idx]
        specs.append(
            {
                "source": "human_level_window",
                "label": f"human_level{target_level}_{idx}",
                "step_index": int(idx),
                "trace_step": int(step.step),
                "next_action": step.action,
            }
        )
        if len(specs) >= max_states:
            break
    return specs[:max_states]


def _probe_human_states(
    *,
    arc: Arcade,
    full_game_id: str,
    selection: Any,
    specs: Sequence[Dict[str, Any]],
    pair_colors: Optional[Tuple[int, int]],
    include_action6: bool,
) -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = []
    for spec in specs:
        idx = int(spec["step_index"])
        try:
            env, raw = _replay_prefix(
                arc,
                full_game_id,
                selection.steps,
                stop_before_index=idx,
                expected_frame=selection.steps[idx].frame_before,
            )
        except Exception as exc:
            probes.append({"sample": dict(spec), "error": str(exc)})
            continue
        actions = _available_names_from_raw(raw) or ACTION_NAMES
        for action in actions:
            if action == "RESET":
                continue
            variants = [{"x": 32, "y": 32}] if action == "ACTION6" and include_action6 else [None]
            if action == "ACTION6" and not include_action6:
                continue
            for action_data in variants:
                probes.append(
                    _probe_action(
                        env=env,
                        raw=raw,
                        full_game_id=full_game_id,
                        action=action,
                        action_data=action_data,
                        pair_colors=pair_colors,
                        sample=spec,
                    )
                )
    return probes


def _probe_agent_states(
    *,
    arc: Arcade,
    full_game_id: str,
    base_env: Any,
    nodes: Sequence[Dict[str, Any]],
    pair_colors: Optional[Tuple[int, int]],
    include_action6: bool,
) -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = []
    for node_index, node in enumerate(nodes):
        actions = list(node.get("actions") or [])
        action_data = list(node.get("action_data") or [])
        env = copy.deepcopy(base_env)
        raw = getattr(env, "observation_space", None)
        for index, action in enumerate(actions):
            data = action_data[index] if index < len(action_data) else None
            raw = _step_branch(
                env,
                full_game_id=full_game_id,
                action=action,
                action_data=data,
            )
            if raw is None:
                break
        if raw is None:
            continue
        sample = {
            "source": "agent_guided_search",
            "label": f"agent_node_{node_index}",
            "actions": actions,
            "reported_grid_hash": node.get("grid_hash"),
        }
        available = _available_names_from_raw(raw) or ACTION_NAMES
        for action in available:
            if action == "RESET":
                continue
            variants = [{"x": 32, "y": 32}] if action == "ACTION6" and include_action6 else [None]
            if action == "ACTION6" and not include_action6:
                continue
            for data in variants:
                probes.append(
                    _probe_action(
                        env=env,
                        raw=raw,
                        full_game_id=full_game_id,
                        action=action,
                        action_data=data,
                        pair_colors=pair_colors,
                        sample=sample,
                    )
                )
    return probes


def _summarize_action(probes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [probe for probe in probes if "error" not in probe]
    if not valid:
        return {"count": 0}
    changed = [int(probe.get("changed_cells", 0)) for probe in valid]
    operator_counts = Counter(str(probe.get("operator_type")) for probe in valid)
    source_counts = Counter(str((probe.get("sample") or {}).get("source")) for probe in valid)
    direction_counts = Counter(str((probe.get("translation") or {}).get("direction_candidate")) for probe in valid)
    affected_counter: Counter[str] = Counter()
    for probe in valid:
        for color in probe.get("affected_colors") or []:
            affected_counter[str(color)] += 1
    no_op_count = operator_counts.get("no_op", 0)
    level_up_count = sum(1 for probe in valid if int(probe.get("level_delta", 0)) > 0)
    non_noop_types = {key for key, count in operator_counts.items() if key != "no_op" and count > 0}
    context_dependent = bool(no_op_count > 0 and len(non_noop_types) > 0) or len(non_noop_types) > 1
    return {
        "count": len(valid),
        "source_counts": dict(source_counts),
        "operator_type_counts": dict(operator_counts),
        "dominant_operator_type": operator_counts.most_common(1)[0][0],
        "context_dependent": context_dependent,
        "no_op_rate": round(float(no_op_count / len(valid)), 4),
        "auto_levelup_count": int(level_up_count),
        "avg_changed_cells": round(float(mean(changed)), 4),
        "median_changed_cells": round(float(median(changed)), 4),
        "affected_colors": dict(affected_counter.most_common(12)),
        "direction_candidates": dict(direction_counts.most_common(8)),
        "examples": sorted(
            valid,
            key=lambda item: (
                int(item.get("level_delta", 0)),
                int(item.get("changed_cells", 0)),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )[:8],
    }


def _summarize_ontology(probes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        action = str(probe.get("action", "UNKNOWN"))
        grouped[action].append(probe)
    return {
        action: _summarize_action(grouped[action])
        for action in sorted(grouped, key=lambda item: int(item.replace("ACTION", "") or 99) if item.startswith("ACTION") else 99)
    }


def _print_report(payload: Dict[str, Any], output: Path) -> None:
    print("=" * 88)
    print("Action ontology discovery")
    print("=" * 88)
    print(f"game:        {payload['game_id']}")
    print(f"episode:     {payload['episode_id']}")
    print(f"pair colors: {payload.get('pair_colors')}")
    print(f"probes:      {payload['probe_count']}")
    for action, summary in payload["action_ontology"].items():
        print(
            f"{action}: "
            f"type={summary.get('dominant_operator_type')} "
            f"noop={summary.get('no_op_rate')} "
            f"levelup={summary.get('auto_levelup_count')} "
            f"dir={summary.get('direction_candidates')}"
        )
    print(f"json:        {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--pair-colors", nargs="+", default=None)
    parser.add_argument("--target-level", type=int, default=7)
    parser.add_argument("--human-stride", type=int, default=24)
    parser.add_argument("--max-human-states", type=int, default=32)
    parser.add_argument("--agent-report", type=Path, default=None)
    parser.add_argument("--max-agent-states", type=int, default=16)
    parser.add_argument("--include-action6", action="store_true")
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

    human_specs = _human_state_specs(
        selection,
        target_level=int(args.target_level),
        stride=max(1, int(args.human_stride)),
        max_states=max(1, int(args.max_human_states)),
    )
    probes = _probe_human_states(
        arc=arc,
        full_game_id=full_game_id,
        selection=selection,
        specs=human_specs,
        pair_colors=pair_colors,
        include_action6=bool(args.include_action6),
    )

    agent_report_path = args.agent_report or _default_agent_report_path(full_game_id, pair_colors)
    agent_nodes: List[Dict[str, Any]] = []
    if agent_report_path and agent_report_path.exists() and int(args.max_agent_states) > 0:
        frontier = find_level_frontier(selection, target_level=int(args.target_level), danger_window=8)
        base_env, _base_raw = _replay_prefix(
            arc,
            full_game_id,
            selection.steps,
            stop_before_index=frontier.level_start_index,
            expected_frame=frontier.level_start_frame,
        )
        agent_report = json.loads(agent_report_path.read_text(encoding="utf-8"))
        agent_nodes = _agent_node_candidates(agent_report, limit=max(1, int(args.max_agent_states)))
        probes.extend(
            _probe_agent_states(
                arc=arc,
                full_game_id=full_game_id,
                base_env=base_env,
                nodes=agent_nodes,
                pair_colors=pair_colors,
                include_action6=bool(args.include_action6),
            )
        )

    payload = {
        "game_id": full_game_id,
        "episode_id": selection.episode_id,
        "pair_colors": list(pair_colors) if pair_colors else None,
        "target_level": int(args.target_level),
        "agent_report": str(agent_report_path) if agent_report_path and agent_report_path.exists() else None,
        "human_state_count": len(human_specs),
        "agent_state_count": len(agent_nodes),
        "probe_count": len([probe for probe in probes if "error" not in probe]),
        "action_ontology": _summarize_ontology(probes),
        "probes": probes,
    }
    output = args.report_dir / f"{full_game_id}.action_ontology.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_report(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
