"""M1.3g minimal polymorphic A25 adapter.

This adapter consumes only M1.3f testable mechanic candidates, executes one
concrete environment action per candidate from RESET, and records the requested
before/after measurement. It does not revise hypotheses and keeps every
experiment UNRESOLVED.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .polymorphic_a25_pretest import (
    DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    TESTABLE,
    _extract_live_components,
    _live_contact_pairs,
)

DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "polymorphic_a25_adapter_bp35.json"
)
DEFAULT_TARGET_GAME_ID = "bp35-0a0ad940"
DEFAULT_BP35_TARGET_SPECS = (
    "ACTION6:object_lifecycle_candidate",
    "ACTION6:contact_change_candidate",
    "ACTION3:object_motion_candidate",
)


@dataclass(frozen=True)
class ConcretePolymorphicAction:
    """One concrete action selected for a mechanic experiment."""

    name: str
    action_args: Dict[str, Any] = field(default_factory=dict)
    selection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action_args": dict(self.action_args),
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class PolymorphicMechanicExperiment:
    """One unresolved mechanic experiment produced by the adapter."""

    candidate_id: str
    game_id: str
    candidate_type: str
    action: str
    required_observation: str
    concrete_action: ConcretePolymorphicAction | None = None
    mechanic_experiment_generated: bool = False
    env_actions: int = 0
    measured_delta: Dict[str, Any] = field(default_factory=dict)
    changed_pixels: int = 0
    levels_before: int = 0
    levels_after: int = 0
    game_state_before: str = ""
    game_state_after: str = ""
    test_goal: str = ""
    error: str = ""
    hypothesis_status: str = "UNRESOLVED"
    experiment_status: str = "UNRESOLVED"
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "candidate_type": self.candidate_type,
            "action": self.action,
            "required_observation": self.required_observation,
            "concrete_action": (
                self.concrete_action.to_dict()
                if self.concrete_action is not None
                else None
            ),
            "mechanic_experiment_generated": self.mechanic_experiment_generated,
            "env_actions": int(self.env_actions),
            "measured_delta": dict(self.measured_delta),
            "changed_pixels": int(self.changed_pixels),
            "levels_before": int(self.levels_before),
            "levels_after": int(self.levels_after),
            "game_state_before": self.game_state_before,
            "game_state_after": self.game_state_after,
            "test_goal": self.test_goal,
            "error": self.error,
            "hypothesis_status": self.hypothesis_status,
            "experiment_status": self.experiment_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def load_polymorphic_pretest_candidates(
    path: str | Path = DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
) -> Tuple[Dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(dict(row) for row in payload.get("results", []) or [])


def select_testable_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    game_id: str | None = DEFAULT_TARGET_GAME_ID,
    candidate_specs: Sequence[str] = DEFAULT_BP35_TARGET_SPECS,
    max_candidates: int = 3,
) -> Tuple[Dict[str, Any], ...]:
    """Select testable M1.3f rows, preserving explicit candidate spec order."""
    filtered = [
        dict(row)
        for row in rows
        if str(row.get("testability_status", "")) == TESTABLE
        and str(row.get("status", "")) == "UNRESOLVED"
        and (game_id is None or str(row.get("game_id", "")) == str(game_id))
    ]
    if candidate_specs:
        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for spec in candidate_specs:
            action, candidate_type = _parse_candidate_spec(spec)
            match = next(
                (
                    row
                    for row in filtered
                    if str(row.get("action", "")) == action
                    and str(row.get("candidate_type", "")) == candidate_type
                    and str(row.get("candidate_id", "")) not in seen
                ),
                None,
            )
            if match is None:
                continue
            seen.add(str(match.get("candidate_id", "")))
            selected.append(match)
        return tuple(selected[: max(1, int(max_candidates))])
    ordered = sorted(
        filtered,
        key=lambda row: (
            str(row.get("game_id", "")),
            str(row.get("action", "")),
            str(row.get("candidate_type", "")),
            str(row.get("candidate_id", "")),
        ),
    )
    return tuple(ordered[: max(1, int(max_candidates))])


def run_minimal_polymorphic_a25_adapter(
    *,
    pretest_path: str | Path = DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    game_id: str | None = DEFAULT_TARGET_GAME_ID,
    candidate_specs: Sequence[str] = DEFAULT_BP35_TARGET_SPECS,
    max_candidates: int = 3,
) -> Dict[str, Any]:
    """Run one-action mechanic experiments for selected M1.3f candidates."""
    rows = load_polymorphic_pretest_candidates(pretest_path)
    selected = select_testable_candidates(
        rows,
        game_id=game_id,
        candidate_specs=candidate_specs,
        max_candidates=max_candidates,
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    experiments = [
        execute_polymorphic_candidate(candidate, environments_dir=env_dir)
        for candidate in selected
    ]
    return {
        "config": {
            "pretest_path": str(pretest_path),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "candidate_specs": list(candidate_specs),
            "max_candidates": int(max_candidates),
        },
        "summary": summarize_polymorphic_adapter_results(experiments),
        "experiments": [experiment.to_dict() for experiment in experiments],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def execute_polymorphic_candidate(
    candidate: Mapping[str, Any],
    *,
    environments_dir: str | Path,
) -> PolymorphicMechanicExperiment:
    """Execute one concrete action and record the requested mechanic delta."""
    game_id = str(candidate.get("game_id", ""))
    action_name = str(candidate.get("action", ""))
    required_observation = str(candidate.get("required_observation", ""))
    base = _experiment_shell(candidate)
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction

        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(environments_dir),
        )
        env = arc.make(game_id)
        before_frame = env.step(GameAction.RESET)
        before = snapshot_frame(before_frame)
        selected_action = _select_concrete_action(
            _valid_actions(env),
            action_name=action_name,
            required_observation=required_observation,
        )
        if selected_action is None:
            return _replace_experiment(base, error="no_concrete_action_available")
        after_frame = _step_env_action(env, selected_action)
        if after_frame is None:
            return _replace_experiment(base, error="env_step_returned_none")
        after = snapshot_frame(
            after_frame,
            fallback_available_actions=before.available_actions,
        )
    except Exception as exc:  # pragma: no cover - integration failure path
        return _replace_experiment(base, error=f"env_execution_failed:{exc}")

    measured_delta = measure_required_observation(
        before.grid,
        after.grid,
        required_observation=required_observation,
        action_args=dict(getattr(selected_action, "action_args", {}) or {}),
    )
    changed_pixels = int(measured_delta.get("changed_pixels", 0) or 0)
    return PolymorphicMechanicExperiment(
        candidate_id=base.candidate_id,
        game_id=base.game_id,
        candidate_type=base.candidate_type,
        action=base.action,
        required_observation=base.required_observation,
        concrete_action=ConcretePolymorphicAction(
            name=str(getattr(selected_action, "name", "")),
            action_args=dict(getattr(selected_action, "action_args", {}) or {}),
            selection_reason=_selection_reason(required_observation),
        ),
        mechanic_experiment_generated=True,
        env_actions=1,
        measured_delta=measured_delta,
        changed_pixels=changed_pixels,
        levels_before=before.levels_completed,
        levels_after=after.levels_completed,
        game_state_before=before.game_state,
        game_state_after=after.game_state,
        test_goal=base.test_goal,
    )


def measure_required_observation(
    before_grid: Any,
    after_grid: Any,
    *,
    required_observation: str,
    action_args: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Measure the requested before/after mechanic delta."""
    before = np.asarray(before_grid, dtype=np.int32)
    after = np.asarray(after_grid, dtype=np.int32)
    base = _base_delta(before, after)
    if required_observation == "object_counts_before_after":
        return {
            **base,
            **_object_count_delta(before, after),
        }
    if required_observation == "contact_graph_before_after":
        return {
            **base,
            **_contact_delta(before, after),
        }
    if required_observation == "object_positions_before_after":
        return {
            **base,
            **_object_motion_delta(before, after),
        }
    if required_observation == "object_shape_zone_before_after":
        return {
            **base,
            **_shape_zone_delta(before, after),
        }
    if required_observation == "local_patch_before_after":
        return {
            **base,
            **_local_patch_delta(before, after, action_args or {}),
        }
    return {
        **base,
        "metric": "unknown",
        "measurable": False,
        "changed": False,
    }


def summarize_polymorphic_adapter_results(
    experiments: Sequence[PolymorphicMechanicExperiment],
) -> Dict[str, Any]:
    generated = [
        experiment for experiment in experiments if experiment.mechanic_experiment_generated
    ]
    by_type = Counter(experiment.candidate_type for experiment in generated)
    by_game = Counter(experiment.game_id for experiment in generated)
    observable = [
        experiment
        for experiment in generated
        if bool(experiment.measured_delta.get("changed"))
    ]
    return {
        "mechanic_candidates_consumed": len(experiments),
        "mechanic_experiments_generated": len(generated),
        "env_actions": sum(experiment.env_actions for experiment in experiments),
        "observable_deltas": len(observable),
        "generated_by_type": dict(sorted(by_type.items())),
        "generated_by_game": dict(sorted(by_game.items())),
        "errors": {
            experiment.candidate_id: experiment.error
            for experiment in experiments
            if experiment.error
        },
        "wrong_confirmations": sum(
            experiment.wrong_confirmations for experiment in experiments
        ),
        "revision_performed": any(
            experiment.revision_performed for experiment in experiments
        ),
    }


def write_polymorphic_a25_adapter_result(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _experiment_shell(candidate: Mapping[str, Any]) -> PolymorphicMechanicExperiment:
    return PolymorphicMechanicExperiment(
        candidate_id=str(candidate.get("candidate_id", "")),
        game_id=str(candidate.get("game_id", "")),
        candidate_type=str(candidate.get("candidate_type", "")),
        action=str(candidate.get("action", "")),
        required_observation=str(candidate.get("required_observation", "")),
        test_goal=str(candidate.get("test_goal", "")),
    )


def _replace_experiment(
    experiment: PolymorphicMechanicExperiment,
    *,
    error: str,
) -> PolymorphicMechanicExperiment:
    return PolymorphicMechanicExperiment(
        candidate_id=experiment.candidate_id,
        game_id=experiment.game_id,
        candidate_type=experiment.candidate_type,
        action=experiment.action,
        required_observation=experiment.required_observation,
        test_goal=experiment.test_goal,
        error=error,
    )


def _select_concrete_action(
    valid_actions: Sequence[Any],
    *,
    action_name: str,
    required_observation: str,
) -> Any | None:
    matches = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == str(action_name)
    ]
    if required_observation == "local_patch_before_after":
        matches = [action for action in matches if _has_xy(action)]
    if not matches:
        return None
    seen: set[Tuple[str, Tuple[Tuple[str, str], ...]]] = set()
    for action in matches:
        signature = _action_signature(action)
        if signature in seen:
            continue
        seen.add(signature)
        return action
    return matches[0]


def _selection_reason(required_observation: str) -> str:
    if required_observation == "local_patch_before_after":
        return "polymorphic_adapter:first_matching_position_action"
    return "polymorphic_adapter:first_matching_action"


def _step_env_action(env: Any, action: Any) -> Any:
    action_args = dict(getattr(action, "action_args", {}) or {})
    raw_action = getattr(action, "raw_action", action)
    if action_args:
        return env.step(raw_action, data=action_args)
    return env.step(raw_action)


def _base_delta(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    same_shape = before.shape == after.shape
    changed_pixels = (
        int(np.sum(before != after))
        if same_shape
        else int(max(before.size, after.size))
    )
    return {
        "grid_shape_before": list(before.shape),
        "grid_shape_after": list(after.shape),
        "changed_pixels": changed_pixels,
        "changed_cell_ratio": round(changed_pixels / max(1, before.size), 6),
        "changed": changed_pixels > 0,
        "measurable": same_shape,
    }


def _object_count_delta(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    before_components = _extract_live_components(before)
    after_components = _extract_live_components(after)
    before_counts = _component_counts_by_color(before_components)
    after_counts = _component_counts_by_color(after_components)
    colors = sorted(set(before_counts) | set(after_counts))
    return {
        "metric": "object_counts_before_after",
        "object_count_before": len(before_components),
        "object_count_after": len(after_components),
        "object_count_delta": len(after_components) - len(before_components),
        "object_counts_by_color_before": _string_key_counts(before_counts),
        "object_counts_by_color_after": _string_key_counts(after_counts),
        "object_count_delta_by_color": {
            str(color): int(after_counts.get(color, 0) - before_counts.get(color, 0))
            for color in colors
            if after_counts.get(color, 0) != before_counts.get(color, 0)
        },
    }


def _contact_delta(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    before_pairs = set(_live_contact_pairs(before))
    after_pairs = set(_live_contact_pairs(after))
    added = sorted(after_pairs - before_pairs)
    removed = sorted(before_pairs - after_pairs)
    return {
        "metric": "contact_graph_before_after",
        "contact_pair_count_before": len(before_pairs),
        "contact_pair_count_after": len(after_pairs),
        "contact_pairs_added": [list(pair) for pair in added],
        "contact_pairs_removed": [list(pair) for pair in removed],
        "contact_graph_changed": bool(added or removed),
    }


def _object_motion_delta(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    before_components = list(_extract_live_components(before))
    after_components = list(_extract_live_components(after))
    unused_after = set(range(len(after_components)))
    moved: List[Dict[str, Any]] = []
    matched = 0
    for before_component in before_components:
        match_index = _nearest_component_index(
            before_component,
            after_components,
            unused_after,
        )
        if match_index is None:
            continue
        unused_after.remove(match_index)
        matched += 1
        after_component = after_components[match_index]
        dy = float(after_component["centroid"][0]) - float(before_component["centroid"][0])
        dx = float(after_component["centroid"][1]) - float(before_component["centroid"][1])
        if abs(dy) < 1e-9 and abs(dx) < 1e-9:
            continue
        moved.append(
            {
                "color": int(before_component["color"]),
                "size": int(before_component["size"]),
                "dy": round(dy, 4),
                "dx": round(dx, 4),
            }
        )
    return {
        "metric": "object_positions_before_after",
        "matched_component_count": matched,
        "moved_component_count": len(moved),
        "motion_vectors": moved[:20],
    }


def _shape_zone_delta(before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
    before_zones = _zone_counts(before)
    after_zones = _zone_counts(after)
    zones = sorted(set(before_zones) | set(after_zones))
    return {
        "metric": "object_shape_zone_before_after",
        "zone_counts_before": dict(sorted(before_zones.items())),
        "zone_counts_after": dict(sorted(after_zones.items())),
        "zone_delta": {
            zone: int(after_zones.get(zone, 0) - before_zones.get(zone, 0))
            for zone in zones
            if after_zones.get(zone, 0) != before_zones.get(zone, 0)
        },
        "shape_zone_changed": before_zones != after_zones,
    }


def _local_patch_delta(
    before: np.ndarray,
    after: np.ndarray,
    action_args: Mapping[str, Any],
    *,
    radius: int = 1,
) -> Dict[str, Any]:
    x = _safe_int(action_args.get("x"))
    y = _safe_int(action_args.get("y"))
    if x is None or y is None or before.ndim != 2 or after.ndim != 2:
        return {
            "metric": "local_patch_before_after",
            "local_patch_available": False,
            "local_changed_pixels": 0,
        }
    y0 = max(0, y - radius)
    y1 = min(before.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(before.shape[1], x + radius + 1)
    before_patch = before[y0:y1, x0:x1]
    after_patch = after[y0:y1, x0:x1]
    return {
        "metric": "local_patch_before_after",
        "local_patch_available": True,
        "patch_bbox": [y0, x0, y1 - 1, x1 - 1],
        "local_changed_pixels": int(np.sum(before_patch != after_patch)),
        "local_patch_before": before_patch.tolist(),
        "local_patch_after": after_patch.tolist(),
    }


def _component_counts_by_color(
    components: Sequence[Mapping[str, Any]],
) -> Dict[int, int]:
    counts: Counter[int] = Counter()
    for component in components:
        counts[int(component.get("color", 0))] += 1
    return dict(counts)


def _string_key_counts(counts: Mapping[int, int]) -> Dict[str, int]:
    return {str(color): int(counts[color]) for color in sorted(counts)}


def _nearest_component_index(
    component: Mapping[str, Any],
    after_components: Sequence[Mapping[str, Any]],
    unused_after: set[int],
) -> int | None:
    color = int(component.get("color", 0))
    size = int(component.get("size", 0))
    candidates = [
        index
        for index in unused_after
        if int(after_components[index].get("color", 0)) == color
        and int(after_components[index].get("size", 0)) == size
    ]
    if not candidates:
        candidates = [
            index
            for index in unused_after
            if int(after_components[index].get("color", 0)) == color
        ]
    if not candidates:
        return None
    y, x = (float(value) for value in component.get("centroid", [0.0, 0.0]))
    return min(
        candidates,
        key=lambda index: (
            abs(float(after_components[index]["centroid"][0]) - y)
            + abs(float(after_components[index]["centroid"][1]) - x)
        ),
    )


def _zone_counts(grid: np.ndarray) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for component in _extract_live_components(grid):
        counts[_bbox_zone(grid.shape, component["bbox"])] += 1
    return dict(counts)


def _bbox_zone(shape: Sequence[int], bbox: Sequence[int]) -> str:
    height = max(1, int(shape[0]))
    width = max(1, int(shape[1]))
    y_mid = (int(bbox[0]) + int(bbox[2])) / 2.0
    x_mid = (int(bbox[1]) + int(bbox[3])) / 2.0
    vertical = "top" if y_mid < height / 3 else "bottom" if y_mid >= 2 * height / 3 else "middle"
    horizontal = "left" if x_mid < width / 3 else "right" if x_mid >= 2 * width / 3 else "center"
    return f"{vertical}_{horizontal}"


def _parse_candidate_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"candidate spec must be ACTION:TYPE, got {spec!r}")
    action, candidate_type = spec.split(":", 1)
    return action.strip(), candidate_type.strip()


def _action_signature(action: Any) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (
        str(getattr(action, "name", "")),
        tuple(
            sorted(
                (str(key), str(value))
                for key, value in dict(getattr(action, "action_args", {}) or {}).items()
            )
        ),
    )


def _has_xy(action: Any) -> bool:
    args = dict(getattr(action, "action_args", {}) or {})
    return "x" in args and "y" in args


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_candidate_specs(values: Sequence[str]) -> Tuple[str, ...]:
    if not values:
        return DEFAULT_BP35_TARGET_SPECS
    result: List[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(item)
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3g minimal polymorphic A25 adapter.",
    )
    parser.add_argument(
        "--pretest",
        type=Path,
        default=DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_TARGET_GAME_ID)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec ACTION:TYPE. Can be repeated or comma-separated.",
    )
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_POLYMORPHIC_A25_ADAPTER_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_minimal_polymorphic_a25_adapter(
        pretest_path=args.pretest,
        environments_dir=args.environments_dir,
        game_id=args.game_id or None,
        candidate_specs=_parse_candidate_specs(args.candidate),
        max_candidates=args.max_candidates,
    )
    write_polymorphic_a25_adapter_result(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
