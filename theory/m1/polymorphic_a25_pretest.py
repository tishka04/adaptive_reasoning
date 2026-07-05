"""M1.3f polymorphic A25 interface pretest.

This module asks whether non-color-source mechanic candidates can become
experiments. It does not run A25, does not step the environment, and never
confirms a hypothesis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .live_anchor_ranking import _load_live_grid_and_actions
from .mechanic_grounded_candidates import (
    DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
)

DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "polymorphic_a25_pretest.json"
)

TESTABLE = "testable"
BLOCKED = "blocked"

ACTION_NOT_AVAILABLE = "action_not_available"
MISSING_POSITION_ARGUMENT = "missing_position_argument"
NO_LIVE_OBJECT_ANCHOR = "no_live_object_anchor"
NO_MEASURABLE_BEFORE_AFTER_METRIC = "no_measurable_before_after_metric"
UNSAFE_OR_TERMINAL_ACTION = "unsafe_or_terminal_action"

REQUIRED_OBSERVATIONS: Dict[str, str] = {
    "object_motion_candidate": "object_positions_before_after",
    "contact_change_candidate": "contact_graph_before_after",
    "object_lifecycle_candidate": "object_counts_before_after",
    "shape_zone_candidate": "object_shape_zone_before_after",
    "position_effect_candidate": "local_patch_before_after",
}

UNSAFE_ACTION_NAMES = {"RESET", "SUBMIT", "DONE", "FINISH", "END"}


@dataclass(frozen=True)
class PolymorphicA25PretestRow:
    """A candidate-level pretest row for a future polymorphic A25 adapter."""

    candidate_id: str
    game_id: str
    candidate_type: str
    action: str
    testability_status: str
    required_observation: str
    available_live_affordance: Dict[str, Any] = field(default_factory=dict)
    blocking_reason: str | None = None
    test_goal: str = ""
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "game_id": self.game_id,
            "candidate_type": self.candidate_type,
            "action": self.action,
            "testability_status": self.testability_status,
            "required_observation": self.required_observation,
            "available_live_affordance": dict(self.available_live_affordance),
            "blocking_reason": self.blocking_reason,
            "test_goal": self.test_goal,
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def load_mechanic_candidate_dicts(
    path: str | Path = DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
) -> Tuple[Dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(dict(candidate) for candidate in payload.get("candidates", []) or [])


def evaluate_candidate_testability(
    candidate: Mapping[str, Any],
    *,
    live_grid: Any,
    valid_actions: Sequence[Any],
    candidate_id: str,
) -> PolymorphicA25PretestRow:
    """Classify one mechanic candidate against reset-state live affordances."""
    game_id = str(candidate.get("game_id", ""))
    candidate_type = str(candidate.get("candidate_type", ""))
    action = str(candidate.get("action", ""))
    required_observation = REQUIRED_OBSERVATIONS.get(candidate_type, "unknown")
    grid = np.asarray(live_grid, dtype=np.int32)
    affordance = _available_live_affordance(
        grid,
        valid_actions,
        action=action,
        candidate_type=candidate_type,
        required_observation=required_observation,
    )
    blocking_reason = _blocking_reason(
        action=action,
        candidate_type=candidate_type,
        required_observation=required_observation,
        affordance=affordance,
    )
    return PolymorphicA25PretestRow(
        candidate_id=candidate_id,
        game_id=game_id,
        candidate_type=candidate_type,
        action=action,
        testability_status=TESTABLE if blocking_reason is None else BLOCKED,
        required_observation=required_observation,
        available_live_affordance=affordance,
        blocking_reason=blocking_reason,
        test_goal=str(candidate.get("test_goal", "")),
    )


def run_polymorphic_a25_pretest(
    *,
    candidates_path: str | Path = DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Pretest mechanic candidates against live reset affordances."""
    candidates = load_mechanic_candidate_dicts(candidates_path)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    live_cache: Dict[str, Tuple[np.ndarray, Sequence[Any]]] = {}
    rows: List[PolymorphicA25PretestRow] = []
    for index, candidate in enumerate(candidates):
        game_id = str(candidate.get("game_id", ""))
        if game_id not in live_cache:
            live_cache[game_id] = _load_live_grid_and_actions(game_id, env_dir)
        live_grid, valid_actions = live_cache[game_id]
        rows.append(
            evaluate_candidate_testability(
                candidate,
                live_grid=live_grid,
                valid_actions=valid_actions,
                candidate_id=_candidate_id(candidate, index),
            )
        )
    return {
        "config": {
            "candidates_path": str(candidates_path),
            "environments_dir": str(env_dir),
        },
        "summary": summarize_polymorphic_pretest(rows),
        "results": [row.to_dict() for row in rows],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def summarize_polymorphic_pretest(
    rows: Sequence[PolymorphicA25PretestRow],
) -> Dict[str, Any]:
    testable_rows = [row for row in rows if row.testability_status == TESTABLE]
    blocked_rows = [row for row in rows if row.testability_status == BLOCKED]
    testable_by_type = Counter(row.candidate_type for row in testable_rows)
    testable_by_game = Counter(row.game_id for row in testable_rows)
    total_by_type = Counter(row.candidate_type for row in rows)
    total_by_game = Counter(row.game_id for row in rows)
    blocking_reasons = Counter(
        str(row.blocking_reason)
        for row in blocked_rows
        if row.blocking_reason is not None
    )
    by_type = {
        key: {
            "total": int(total_by_type[key]),
            "testable": int(testable_by_type.get(key, 0)),
            "blocked": int(total_by_type[key] - testable_by_type.get(key, 0)),
        }
        for key in sorted(total_by_type)
    }
    by_game = {
        key: {
            "total": int(total_by_game[key]),
            "testable": int(testable_by_game.get(key, 0)),
            "blocked": int(total_by_game[key] - testable_by_game.get(key, 0)),
        }
        for key in sorted(total_by_game)
    }
    return {
        "mechanic_candidates_total": len(rows),
        "mechanic_candidates_testable": len(testable_rows),
        "testable_by_type": dict(sorted(testable_by_type.items())),
        "testable_by_game": dict(sorted(testable_by_game.items())),
        "blocking_reasons": dict(sorted(blocking_reasons.items())),
        "by_type": by_type,
        "by_game": by_game,
        "wrong_confirmations": 0,
    }


def write_polymorphic_a25_pretest(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _available_live_affordance(
    grid: np.ndarray,
    valid_actions: Sequence[Any],
    *,
    action: str,
    candidate_type: str,
    required_observation: str,
) -> Dict[str, Any]:
    matching_actions = [
        item for item in valid_actions if str(getattr(item, "name", "")) == action
    ]
    position_actions = [
        item for item in matching_actions if _has_live_position_argument(item, grid)
    ]
    components = _extract_live_components(grid)
    contact_pairs = _live_contact_pairs(grid)
    action_names = sorted({str(getattr(item, "name", "")) for item in valid_actions})
    return {
        "valid_action_count": len(valid_actions),
        "available_action_names": action_names,
        "matching_action_count": len(matching_actions),
        "matching_actions_with_position_args": len(position_actions),
        "has_position_argument": bool(position_actions),
        "live_color_count": int(len(set(int(value) for value in np.unique(grid)))),
        "live_component_count": len(components),
        "live_non_background_object_count": len(components),
        "live_contact_pair_count": len(contact_pairs),
        "has_live_object_anchor": bool(components),
        "has_measurable_before_after_metric": _has_measurable_metric(
            grid,
            candidate_type=candidate_type,
            required_observation=required_observation,
            components=components,
        ),
        "unsafe_action": action.upper() in UNSAFE_ACTION_NAMES,
        "required_observation": required_observation,
    }


def _blocking_reason(
    *,
    action: str,
    candidate_type: str,
    required_observation: str,
    affordance: Mapping[str, Any],
) -> str | None:
    if bool(affordance.get("unsafe_action")) or action.upper() in UNSAFE_ACTION_NAMES:
        return UNSAFE_OR_TERMINAL_ACTION
    if int(affordance.get("matching_action_count", 0) or 0) <= 0:
        return ACTION_NOT_AVAILABLE
    if candidate_type == "position_effect_candidate" and not bool(
        affordance.get("has_position_argument")
    ):
        return MISSING_POSITION_ARGUMENT
    if candidate_type in {
        "object_motion_candidate",
        "contact_change_candidate",
        "shape_zone_candidate",
    } and not bool(affordance.get("has_live_object_anchor")):
        return NO_LIVE_OBJECT_ANCHOR
    if not bool(affordance.get("has_measurable_before_after_metric")):
        return NO_MEASURABLE_BEFORE_AFTER_METRIC
    return None


def _has_measurable_metric(
    grid: np.ndarray,
    *,
    candidate_type: str,
    required_observation: str,
    components: Sequence[Mapping[str, Any]],
) -> bool:
    if grid.ndim != 2 or grid.size == 0 or required_observation == "unknown":
        return False
    if candidate_type == "object_lifecycle_candidate":
        return True
    if candidate_type == "contact_change_candidate":
        return len(components) >= 2
    if candidate_type in {"object_motion_candidate", "shape_zone_candidate"}:
        return bool(components)
    if candidate_type == "position_effect_candidate":
        return True
    return False


def _extract_live_components(grid: np.ndarray) -> Tuple[Dict[str, Any], ...]:
    if grid.ndim != 2 or grid.size == 0:
        return tuple()
    background = _background_color(grid)
    visited = np.zeros(grid.shape, dtype=bool)
    components: List[Dict[str, Any]] = []
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if visited[y, x] or color == background:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in _neighbors(cy, cx, height, width):
                    if visited[ny, nx] or int(grid[ny, nx]) != color:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": [min(ys), min(xs), max(ys), max(xs)],
                    "centroid": [
                        round(sum(ys) / len(ys), 4),
                        round(sum(xs) / len(xs), 4),
                    ],
                }
            )
    return tuple(components)


def _live_contact_pairs(grid: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    if grid.ndim != 2 or grid.size == 0:
        return tuple()
    background = _background_color(grid)
    pairs: set[Tuple[int, int]] = set()
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if color == background:
                continue
            for ny, nx in ((y + 1, x), (y, x + 1)):
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                other = int(grid[ny, nx])
                if other == background or other == color:
                    continue
                pairs.add(tuple(sorted((color, other))))
    return tuple(sorted(pairs))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = sorted(
        zip(values.tolist(), counts.tolist()),
        key=lambda item: (-int(item[1]), int(item[0])),
    )
    return int(order[0][0])


def _neighbors(
    y: int,
    x: int,
    height: int,
    width: int,
) -> Tuple[Tuple[int, int], ...]:
    result: List[Tuple[int, int]] = []
    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= ny < height and 0 <= nx < width:
            result.append((ny, nx))
    return tuple(result)


def _has_live_position_argument(action: Any, grid: np.ndarray) -> bool:
    args = dict(getattr(action, "action_args", {}) or {})
    if "x" not in args or "y" not in args:
        return False
    try:
        x = int(args["x"])
        y = int(args["y"])
    except (TypeError, ValueError):
        return False
    if grid.ndim != 2:
        return False
    return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]


def _candidate_id(candidate: Mapping[str, Any], index: int) -> str:
    return ":".join(
        [
            f"m1e{int(index):04d}",
            str(candidate.get("game_id", "")),
            str(candidate.get("action", "")),
            str(candidate.get("candidate_type", "")),
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3f polymorphic A25 interface pretest.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_POLYMORPHIC_A25_PRETEST_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_polymorphic_a25_pretest(
        candidates_path=args.candidates,
        environments_dir=args.environments_dir,
    )
    write_polymorphic_a25_pretest(payload, args.out)
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
