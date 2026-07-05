"""M1.3e mechanic-grounded experiment candidates.

This generator proposes unresolved, non-color-source experiment candidates from
raw M1 observations. It does not run A25 and does not confirm hypotheses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .mechanic_typing import (
    DEFAULT_MECHANIC_TYPING_OUTPUT_PATH,
    load_observation_rows,
)
from .observation_dataset import DEFAULT_OUTPUT_PATH as DEFAULT_OBSERVATION_PATH

DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "mechanic_grounded_candidates.json"
)


@dataclass(frozen=True)
class MechanicGroundedExperimentCandidate:
    """An unresolved experiment candidate grounded in a non-color mechanism."""

    game_id: str
    candidate_type: str
    action: str
    mechanism: str
    support: int
    support_rate: float
    test_goal: str
    evidence: Dict[str, Any]
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "candidate_type": self.candidate_type,
            "action": self.action,
            "mechanism": self.mechanism,
            "support": int(self.support),
            "support_rate": round(float(self.support_rate), 4),
            "test_goal": self.test_goal,
            "evidence": dict(self.evidence),
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def load_mechanic_profiles(
    path: str | Path = DEFAULT_MECHANIC_TYPING_OUTPUT_PATH,
) -> Dict[str, Dict[str, Any]]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        str(profile.get("game_id", "")): dict(profile)
        for profile in payload.get("profiles", []) or []
    }


def generate_mechanic_grounded_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    mechanic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    min_support_count: int = 3,
    min_support_rate: float = 0.35,
    max_candidates_per_game: int = 12,
) -> Tuple[MechanicGroundedExperimentCandidate, ...]:
    """Generate unresolved experiment candidates from raw mechanics."""
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        action = str(row.get("action", ""))
        if action == "RESET":
            continue
        grouped[(str(row.get("game_id", "")), action)].append(row)

    candidates: List[MechanicGroundedExperimentCandidate] = []
    profiles = mechanic_profiles or {}
    for (game_id, action), action_rows in sorted(grouped.items()):
        if len(action_rows) < max(1, int(min_support_count)):
            continue
        profile = profiles.get(game_id, {})
        candidates.extend(
            _candidates_for_action(
                game_id,
                action,
                action_rows,
                profile=profile,
                min_support_count=min_support_count,
                min_support_rate=min_support_rate,
            )
        )

    by_game: Dict[str, List[MechanicGroundedExperimentCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_game[candidate.game_id].append(candidate)

    selected: List[MechanicGroundedExperimentCandidate] = []
    for game_id in sorted(by_game):
        selected.extend(
            _select_diverse_candidates(
                by_game[game_id],
                max_candidates=max_candidates_per_game,
            )
        )
    return tuple(selected)


def run_mechanic_grounded_candidate_generation(
    *,
    observation_path: str | Path = DEFAULT_OBSERVATION_PATH,
    mechanic_typing_path: str | Path = DEFAULT_MECHANIC_TYPING_OUTPUT_PATH,
    min_support_count: int = 3,
    min_support_rate: float = 0.35,
    max_candidates_per_game: int = 12,
) -> Dict[str, Any]:
    rows = load_observation_rows(observation_path)
    profiles = load_mechanic_profiles(mechanic_typing_path)
    candidates = generate_mechanic_grounded_candidates(
        rows,
        mechanic_profiles=profiles,
        min_support_count=min_support_count,
        min_support_rate=min_support_rate,
        max_candidates_per_game=max_candidates_per_game,
    )
    return {
        "config": {
            "observation_path": str(observation_path),
            "mechanic_typing_path": str(mechanic_typing_path),
            "min_support_count": int(min_support_count),
            "min_support_rate": float(min_support_rate),
            "max_candidates_per_game": int(max_candidates_per_game),
        },
        "summary": summarize_candidates(candidates),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def summarize_candidates(
    candidates: Sequence[MechanicGroundedExperimentCandidate],
) -> Dict[str, Any]:
    by_game = Counter(candidate.game_id for candidate in candidates)
    by_type = Counter(candidate.candidate_type for candidate in candidates)
    return {
        "candidate_count": len(candidates),
        "candidate_count_by_game": dict(sorted(by_game.items())),
        "candidate_count_by_type": dict(sorted(by_type.items())),
        "non_color_candidate_types": sorted(by_type),
    }


def _select_diverse_candidates(
    candidates: Sequence[MechanicGroundedExperimentCandidate],
    *,
    max_candidates: int,
) -> List[MechanicGroundedExperimentCandidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.support_rate,
            -candidate.support,
            candidate.candidate_type,
            candidate.action,
        ),
    )
    selected: List[MechanicGroundedExperimentCandidate] = []
    selected_keys: set[Tuple[str, str]] = set()
    for candidate_type in sorted({candidate.candidate_type for candidate in ordered}):
        best = next(
            candidate
            for candidate in ordered
            if candidate.candidate_type == candidate_type
        )
        selected.append(best)
        selected_keys.add((best.candidate_type, best.action))
    for candidate in ordered:
        if len(selected) >= max(1, int(max_candidates)):
            break
        key = (candidate.candidate_type, candidate.action)
        if key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(key)
    return selected[: max(1, int(max_candidates))]


def write_mechanic_grounded_candidates(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidates_for_action(
    game_id: str,
    action: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    min_support_count: int,
    min_support_rate: float,
) -> List[MechanicGroundedExperimentCandidate]:
    result: List[MechanicGroundedExperimentCandidate] = []
    row_count = max(1, len(rows))
    warning = str(profile.get("representation_warning") or "")

    motion_rows = [
        row
        for row in rows
        if bool(row.get("position_changed")) or bool(row.get("object_motion_vectors"))
    ]
    if _supported(motion_rows, row_count, min_support_count, min_support_rate):
        result.append(
            _candidate(
                game_id,
                action,
                "object_motion_candidate",
                "object_motion",
                motion_rows,
                row_count,
                test_goal=f"test whether {action} produces object motion or displacement",
                evidence={
                    "dominant_motion_colors": _dominant_motion_colors(motion_rows),
                    "dominant_motion_directions": _dominant_motion_directions(motion_rows),
                    "mean_motion_vectors": _mean_list_len(motion_rows, "object_motion_vectors"),
                    "mechanic_typing_warning": warning,
                },
            )
        )

    contact_rows = [row for row in rows if bool(row.get("adjacency_changed"))]
    if _supported(contact_rows, row_count, min_support_count, min_support_rate):
        result.append(
            _candidate(
                game_id,
                action,
                "contact_change_candidate",
                "topology_contact",
                contact_rows,
                row_count,
                test_goal=f"test whether {action} changes object contacts/topology",
                evidence={
                    "dominant_changed_contact_pairs": _dominant_contact_pairs(contact_rows),
                    "mechanic_typing_warning": warning,
                },
            )
        )

    lifecycle_rows = [
        row
        for row in rows
        if int(row.get("created_object_count", 0) or 0) > 0
        or int(row.get("removed_object_count", 0) or 0) > 0
        or bool(row.get("object_count_changed"))
    ]
    if _supported(lifecycle_rows, row_count, min_support_count, min_support_rate):
        result.append(
            _candidate(
                game_id,
                action,
                "object_lifecycle_candidate",
                "object_lifecycle",
                lifecycle_rows,
                row_count,
                test_goal=f"test whether {action} creates/removes/replaces objects",
                evidence={
                    "dominant_created_colors": _dominant_lifecycle_colors(
                        lifecycle_rows,
                        direction="created",
                    ),
                    "dominant_removed_colors": _dominant_lifecycle_colors(
                        lifecycle_rows,
                        direction="removed",
                    ),
                    "mean_created_object_count": _mean_number(
                        lifecycle_rows,
                        "created_object_count",
                    ),
                    "mean_removed_object_count": _mean_number(
                        lifecycle_rows,
                        "removed_object_count",
                    ),
                    "mechanic_typing_warning": warning,
                },
            )
        )

    shape_rows = [row for row in rows if bool(row.get("shape_changed"))]
    if _supported(shape_rows, row_count, min_support_count, min_support_rate):
        result.append(
            _candidate(
                game_id,
                action,
                "shape_zone_candidate",
                "shape_zone",
                shape_rows,
                row_count,
                test_goal=f"test whether {action} changes object shape/zone occupancy",
                evidence={
                    "dominant_zones_after": _dominant_zones(shape_rows),
                    "mean_object_count_delta": _mean_number(
                        shape_rows,
                        "object_count_delta",
                    ),
                    "mechanic_typing_warning": warning,
                },
            )
        )

    position_rows = [
        row
        for row in rows
        if bool(row.get("position_changed"))
        or bool(row.get("player_moved"))
        or bool(row.get("action_args"))
    ]
    if _supported(position_rows, row_count, min_support_count, min_support_rate):
        result.append(
            _candidate(
                game_id,
                action,
                "position_effect_candidate",
                "position_effect",
                position_rows,
                row_count,
                test_goal=f"test whether {action} has a position/local-effect dependency",
                evidence={
                    "action_arg_keys": _action_arg_keys(position_rows),
                    "action_arg_rate": round(
                        sum(bool(row.get("action_args")) for row in position_rows)
                        / max(1, len(position_rows)),
                        4,
                    ),
                    "dominant_motion_directions": _dominant_motion_directions(position_rows),
                    "mean_changed_cell_ratio": _mean_number(
                        position_rows,
                        "changed_cell_ratio",
                    ),
                    "mechanic_typing_warning": warning,
                },
            )
        )
    return result


def _candidate(
    game_id: str,
    action: str,
    candidate_type: str,
    mechanism: str,
    support_rows: Sequence[Mapping[str, Any]],
    row_count: int,
    *,
    test_goal: str,
    evidence: Mapping[str, Any],
) -> MechanicGroundedExperimentCandidate:
    support = len(support_rows)
    return MechanicGroundedExperimentCandidate(
        game_id=game_id,
        candidate_type=candidate_type,
        action=action,
        mechanism=mechanism,
        support=support,
        support_rate=round(support / max(1, row_count), 4),
        test_goal=test_goal,
        evidence={
            "action_transition_count": int(row_count),
            "support_transition_count": int(support),
            **dict(evidence),
        },
    )


def _supported(
    support_rows: Sequence[Mapping[str, Any]],
    row_count: int,
    min_support_count: int,
    min_support_rate: float,
) -> bool:
    return len(support_rows) >= max(1, int(min_support_count)) and (
        len(support_rows) / max(1, row_count)
    ) >= float(min_support_rate)


def _dominant_motion_colors(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, int]]:
    counts: Counter[int] = Counter()
    for row in rows:
        for vector in row.get("object_motion_vectors", []) or []:
            counts[int(vector.get("color", -1))] += 1
    return _top_counter(counts)


def _dominant_motion_directions(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        for vector in row.get("object_motion_vectors", []) or []:
            dy = float(vector.get("dy", 0.0) or 0.0)
            dx = float(vector.get("dx", 0.0) or 0.0)
            if abs(dy) >= abs(dx) and dy:
                direction = "down" if dy > 0 else "up"
            elif dx:
                direction = "right" if dx > 0 else "left"
            else:
                direction = "stationary"
            counts[direction] += 1
    return [{"direction": key, "count": int(value)} for key, value in counts.most_common(5)]


def _dominant_contact_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[int, int]] = Counter()
    for row in rows:
        before = _contact_counts(row.get("adjacency_pairs_before", []) or [])
        after = _contact_counts(row.get("adjacency_pairs_after", []) or [])
        for pair in set(before) | set(after):
            if before.get(pair, 0) != after.get(pair, 0):
                counts[pair] += abs(after.get(pair, 0) - before.get(pair, 0))
    return [
        {"colors": [int(first), int(second)], "count": int(count)}
        for (first, second), count in counts.most_common(5)
    ]


def _dominant_lifecycle_colors(
    rows: Sequence[Mapping[str, Any]],
    *,
    direction: str,
) -> List[Dict[str, int]]:
    counts: Counter[int] = Counter()
    for row in rows:
        before = _int_key_counts(row.get("object_counts_by_color_before", {}) or {})
        after = _int_key_counts(row.get("object_counts_by_color_after", {}) or {})
        for color in set(before) | set(after):
            delta = after.get(color, 0) - before.get(color, 0)
            if direction == "created" and delta > 0:
                counts[color] += delta
            elif direction == "removed" and delta < 0:
                counts[color] += abs(delta)
    return _top_counter(counts)


def _dominant_zones(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        shape = tuple(row.get("grid_shape_after") or row.get("grid_shape_before") or (1, 1))
        for measurement in row.get("object_measurements_after", []) or []:
            bbox = measurement.get("bbox")
            if bbox:
                counts[_bbox_zone(shape, bbox)] += 1
    return [{"zone": key, "count": int(value)} for key, value in counts.most_common(5)]


def _action_arg_keys(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for row in rows:
        args = row.get("action_args")
        if isinstance(args, Mapping):
            keys.update(str(key) for key in args)
    return sorted(keys)


def _contact_counts(records: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], int]:
    result: Dict[Tuple[int, int], int] = {}
    for record in records:
        colors = record.get("colors", [])
        if len(colors) != 2:
            continue
        pair = tuple(sorted((int(colors[0]), int(colors[1]))))
        result[pair] = int(record.get("contacts", 0) or 0)
    return result


def _int_key_counts(mapping: Mapping[str, Any]) -> Dict[int, int]:
    return {int(key): int(value) for key, value in mapping.items()}


def _bbox_zone(shape: Sequence[Any], bbox: Sequence[Any]) -> str:
    height = max(1, int(shape[0]))
    width = max(1, int(shape[1]))
    y_mid = (int(bbox[0]) + int(bbox[2])) / 2.0
    x_mid = (int(bbox[1]) + int(bbox[3])) / 2.0
    vertical = "top" if y_mid < height / 3 else "bottom" if y_mid >= 2 * height / 3 else "middle"
    horizontal = "left" if x_mid < width / 3 else "right" if x_mid >= 2 * width / 3 else "center"
    return f"{vertical}_{horizontal}"


def _mean_number(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows), 4)


def _mean_list_len(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(len(row.get(key, []) or []) for row in rows) / len(rows), 4)


def _top_counter(counter: Counter[int]) -> List[Dict[str, int]]:
    return [
        {"color": int(color), "count": int(count)}
        for color, count in counter.most_common(5)
        if color >= 0
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M1.3e mechanic-grounded candidates.",
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATION_PATH)
    parser.add_argument(
        "--mechanic-typing",
        type=Path,
        default=DEFAULT_MECHANIC_TYPING_OUTPUT_PATH,
    )
    parser.add_argument("--min-support-count", type=int, default=3)
    parser.add_argument("--min-support-rate", type=float, default=0.35)
    parser.add_argument("--max-candidates-per-game", type=int, default=12)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MECHANIC_GROUNDED_CANDIDATES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_mechanic_grounded_candidate_generation(
        observation_path=args.observations,
        mechanic_typing_path=args.mechanic_typing,
        min_support_count=args.min_support_count,
        min_support_rate=args.min_support_rate,
        max_candidates_per_game=args.max_candidates_per_game,
    )
    write_mechanic_grounded_candidates(payload, args.out)
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
