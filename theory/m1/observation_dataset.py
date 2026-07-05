"""M1.1 raw transition observation dataset.

This module deliberately stores level-1 facts: what changed, what stayed
numerically measurable, and which human notes were attached to the transition.
It does not emit theory predicates such as the A12 source-target relations.

Run:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m theory.m1.observation_dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

DEFAULT_TRACES_DIR = Path("human_traces")
DEFAULT_OUTPUT_PATH = Path("training") / "m1_observation_dataset.jsonl"
SCHEMA_VERSION = "m1.raw_transition_observation.v1"
ACTION_ID_TO_NAME = {0: "RESET", **{i: f"ACTION{i}" for i in range(1, 8)}}


@dataclass(frozen=True)
class RawTransitionObservation:
    """One raw before/after transition from a human trace.

    The fields are intentionally numeric or direct booleans. They are candidate
    material for M1.2, not claims validated by the scientific loop.
    """

    game_id: str
    episode_id: str
    step: int
    action: str
    action_args: Dict[str, Any] | None
    available_actions: List[str]
    shape_changed: bool
    color_changed: bool
    position_changed: bool
    adjacency_changed: bool
    object_count_changed: bool
    num_cells_changed: int
    player_moved: bool
    level_progressed: bool
    game_over: bool
    grid_shape_before: Tuple[int, int]
    grid_shape_after: Tuple[int, int]
    background_color: int | None
    cell_count_before: int
    cell_count_after: int
    changed_cell_ratio: float
    object_count_before: int
    object_count_after: int
    object_count_delta: int
    counts_by_color_before: Dict[str, int]
    counts_by_color_after: Dict[str, int]
    object_counts_by_color_before: Dict[str, int]
    object_counts_by_color_after: Dict[str, int]
    object_sizes_before: List[int]
    object_sizes_after: List[int]
    object_measurements_before: List[Dict[str, Any]]
    object_measurements_after: List[Dict[str, Any]]
    color_pairs_changed: List[Dict[str, int]]
    adjacency_pairs_before: List[Dict[str, Any]]
    adjacency_pairs_after: List[Dict[str, Any]]
    object_motion_vectors: List[Dict[str, Any]]
    created_object_count: int
    removed_object_count: int
    levels_completed_before: int
    levels_completed_after: int
    intent_text: str
    hypothesis_text: str
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RawTransitionObservation":
        data = dict(row)
        data["grid_shape_before"] = tuple(data["grid_shape_before"])
        data["grid_shape_after"] = tuple(data["grid_shape_after"])
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "RawTransitionObservation":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class _Component:
    component_id: int
    color: int
    points: Tuple[Tuple[int, int], ...]
    bbox: Tuple[int, int, int, int]
    shape_signature: str

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            sum(y for y, _ in self.points) / max(1, self.size),
            sum(x for _, x in self.points) / max(1, self.size),
        )

    @property
    def profile(self) -> Tuple[int, int, int, str]:
        return (self.color, self.size, self.height * 10000 + self.width, self.shape_signature)

    def to_measurement(self) -> Dict[str, Any]:
        cy, cx = self.centroid
        return {
            "component_id": int(self.component_id),
            "color": int(self.color),
            "size": int(self.size),
            "bbox": [int(value) for value in self.bbox],
            "height": int(self.height),
            "width": int(self.width),
            "centroid": [round(float(cy), 4), round(float(cx), 4)],
            "shape_signature": self.shape_signature,
        }


def build_dataset(
    trace_paths: Sequence[str | Path],
    *,
    output_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    include_reset: bool = True,
    max_observations: int | None = None,
) -> List[RawTransitionObservation]:
    """Build and optionally write the M1.1 JSONL dataset."""
    observations: List[RawTransitionObservation] = []
    for trace_path in trace_paths:
        for observation in iter_trace_observations(
            trace_path,
            include_reset=include_reset,
        ):
            observations.append(observation)
            if max_observations is not None and len(observations) >= max_observations:
                if output_path is not None:
                    write_observations_jsonl(observations, output_path)
                return observations

    if output_path is not None:
        write_observations_jsonl(observations, output_path)
    return observations


def iter_trace_observations(
    trace_path: str | Path,
    *,
    include_reset: bool = True,
) -> Iterator[RawTransitionObservation]:
    """Yield raw observations from one ``*.steps.jsonl`` trace."""
    path = Path(trace_path)
    levels_by_episode: Dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            episode_id = str(item.get("episode_id", ""))
            action = _normalize_action_name(item.get("action"))
            levels_before = levels_by_episode[episode_id]
            levels_after = _safe_int(item.get("levels_completed_after"), levels_before)
            levels_by_episode[episode_id] = levels_after
            if action == "RESET" and not include_reset:
                continue
            before = _as_grid(item.get("frame_before"))
            after = _as_grid(item.get("frame_after"))
            if before is None or after is None:
                continue
            try:
                yield _observe_transition(
                    item,
                    before=before,
                    after=after,
                    action=action,
                    levels_before=levels_before,
                    levels_after=levels_after,
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def write_observations_jsonl(
    observations: Iterable[RawTransitionObservation],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(observation.to_json() + "\n")


def load_observations_jsonl(path: str | Path) -> List[RawTransitionObservation]:
    observations: List[RawTransitionObservation] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                observations.append(RawTransitionObservation.from_json(line))
    return observations


def _observe_transition(
    item: Mapping[str, Any],
    *,
    before: np.ndarray,
    after: np.ndarray,
    action: str,
    levels_before: int,
    levels_after: int,
) -> RawTransitionObservation:
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("frame_before and frame_after must be 2D grids")

    background = _infer_background(before, after)
    objects_before = _extract_components(before, background=background)
    objects_after = _extract_components(after, background=background)
    counts_before = _counts_by_color(before)
    counts_after = _counts_by_color(after)
    object_counts_before = _component_counts_by_color(objects_before)
    object_counts_after = _component_counts_by_color(objects_after)
    changed_cells = _changed_cell_count(before, after)
    color_pairs = _changed_color_pairs(before, after)
    contacts_before = _color_contact_pairs(before, background=background)
    contacts_after = _color_contact_pairs(after, background=background)
    motion_vectors = _object_motion_vectors(objects_before, objects_after)
    created_count, removed_count = _created_removed_counts(objects_before, objects_after)

    object_sizes_before = sorted(component.size for component in objects_before)
    object_sizes_after = sorted(component.size for component in objects_after)
    profile_before = Counter(component.profile for component in objects_before)
    profile_after = Counter(component.profile for component in objects_after)
    state = str(item.get("game_state_after", ""))

    return RawTransitionObservation(
        game_id=str(item.get("game_id", "")),
        episode_id=str(item.get("episode_id", "")),
        step=_safe_int(item.get("step"), 0),
        action=action,
        action_args=_clean_mapping_or_none(item.get("action_args")),
        available_actions=_available_action_names(item.get("available_actions") or []),
        shape_changed=profile_before != profile_after,
        color_changed=counts_before != counts_after or bool(color_pairs),
        position_changed=any(vector["distance"] > 0 for vector in motion_vectors),
        adjacency_changed=contacts_before != contacts_after,
        object_count_changed=len(objects_before) != len(objects_after),
        num_cells_changed=changed_cells,
        player_moved=any(
            vector["size"] <= 9 and vector["distance"] > 0
            for vector in motion_vectors
        ),
        level_progressed=levels_after > levels_before,
        game_over=state == "GAME_OVER",
        grid_shape_before=(int(before.shape[0]), int(before.shape[1])),
        grid_shape_after=(int(after.shape[0]), int(after.shape[1])),
        background_color=background,
        cell_count_before=int(before.size),
        cell_count_after=int(after.size),
        changed_cell_ratio=round(changed_cells / max(1, int(before.size)), 6),
        object_count_before=len(objects_before),
        object_count_after=len(objects_after),
        object_count_delta=len(objects_after) - len(objects_before),
        counts_by_color_before=_string_counter(counts_before),
        counts_by_color_after=_string_counter(counts_after),
        object_counts_by_color_before=_string_counter(object_counts_before),
        object_counts_by_color_after=_string_counter(object_counts_after),
        object_sizes_before=object_sizes_before,
        object_sizes_after=object_sizes_after,
        object_measurements_before=[
            component.to_measurement() for component in objects_before
        ],
        object_measurements_after=[
            component.to_measurement() for component in objects_after
        ],
        color_pairs_changed=_color_pair_records(color_pairs),
        adjacency_pairs_before=_contact_pair_records(contacts_before),
        adjacency_pairs_after=_contact_pair_records(contacts_after),
        object_motion_vectors=motion_vectors,
        created_object_count=created_count,
        removed_object_count=removed_count,
        levels_completed_before=levels_before,
        levels_completed_after=levels_after,
        intent_text=str(item.get("intent") or ""),
        hypothesis_text=str(item.get("hypothesis") or ""),
    )


def _as_grid(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.int32)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2:
        return None
    return array


def _extract_components(grid: np.ndarray, *, background: int | None) -> List[_Component]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: List[_Component] = []
    next_id = 0
    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if seen[y, x] or (background is not None and color == background):
                continue
            stack = [(y, x)]
            seen[y, x] = True
            points: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            bbox = (min(ys), min(xs), max(ys), max(xs))
            components.append(
                _Component(
                    component_id=next_id,
                    color=color,
                    points=tuple(sorted(points)),
                    bbox=bbox,
                    shape_signature=_shape_signature(points),
                )
            )
            next_id += 1
    return components


def _shape_signature(points: Sequence[Tuple[int, int]]) -> str:
    min_y = min((point[0] for point in points), default=0)
    min_x = min((point[1] for point in points), default=0)
    offsets = sorted((y - min_y, x - min_x) for y, x in points)
    digest = hashlib.sha1()
    for y, x in offsets:
        digest.update(f"{y}:{x};".encode("ascii"))
    return digest.hexdigest()[:16]


def _counts_by_color(grid: np.ndarray) -> Counter[int]:
    values, counts = np.unique(grid, return_counts=True)
    return Counter({int(value): int(count) for value, count in zip(values, counts)})


def _component_counts_by_color(components: Sequence[_Component]) -> Counter[int]:
    return Counter(component.color for component in components)


def _changed_cell_count(before: np.ndarray, after: np.ndarray) -> int:
    if before.shape != after.shape:
        return int(max(before.size, after.size))
    return int(np.count_nonzero(before != after))


def _changed_color_pairs(before: np.ndarray, after: np.ndarray) -> Counter[Tuple[int, int]]:
    pairs: Counter[Tuple[int, int]] = Counter()
    if before.shape != after.shape:
        return pairs
    mask = before != after
    for source, target in zip(before[mask].ravel(), after[mask].ravel()):
        pairs[(int(source), int(target))] += 1
    return pairs


def _color_contact_pairs(
    grid: np.ndarray,
    *,
    background: int | None,
) -> Counter[Tuple[int, int]]:
    contacts: Counter[Tuple[int, int]] = Counter()
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if background is not None and color == background:
                continue
            for dy, dx in ((1, 0), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                other = int(grid[ny, nx])
                if other == color or (background is not None and other == background):
                    continue
                contacts[tuple(sorted((color, other)))] += 1
    return contacts


def _object_motion_vectors(
    before: Sequence[_Component],
    after: Sequence[_Component],
) -> List[Dict[str, Any]]:
    after_by_profile: Dict[Tuple[int, int, int, str], List[_Component]] = defaultdict(list)
    for component in after:
        after_by_profile[component.profile].append(component)

    used_after: set[int] = set()
    vectors: List[Dict[str, Any]] = []
    for component in before:
        candidates = [
            candidate
            for candidate in after_by_profile.get(component.profile, [])
            if candidate.component_id not in used_after
        ]
        if not candidates:
            continue
        cy, cx = component.centroid
        best = min(candidates, key=lambda candidate: _distance((cy, cx), candidate.centroid))
        used_after.add(best.component_id)
        by, bx = component.centroid
        ay, ax = best.centroid
        dy = round(float(ay - by), 4)
        dx = round(float(ax - bx), 4)
        distance = round(math.hypot(dy, dx), 4)
        if distance == 0:
            continue
        vectors.append(
            {
                "color": int(component.color),
                "size": int(component.size),
                "before_centroid": [round(float(by), 4), round(float(bx), 4)],
                "after_centroid": [round(float(ay), 4), round(float(ax), 4)],
                "dy": dy,
                "dx": dx,
                "distance": distance,
            }
        )
    return sorted(vectors, key=lambda row: (-row["distance"], row["color"], row["size"]))


def _created_removed_counts(
    before: Sequence[_Component],
    after: Sequence[_Component],
) -> Tuple[int, int]:
    before_profiles = Counter(component.profile for component in before)
    after_profiles = Counter(component.profile for component in after)
    created = sum((after_profiles - before_profiles).values())
    removed = sum((before_profiles - after_profiles).values())
    return int(created), int(removed)


def _distance(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return math.hypot(float(left[0] - right[0]), float(left[1] - right[1]))


def _infer_background(before: np.ndarray, after: np.ndarray) -> int | None:
    if before.size == 0 and after.size == 0:
        return None
    values, counts = np.unique(
        np.concatenate([before.ravel(), after.ravel()]),
        return_counts=True,
    )
    return int(values[int(np.argmax(counts))])


def _color_pair_records(counter: Counter[Tuple[int, int]]) -> List[Dict[str, int]]:
    return [
        {
            "before_color": int(before_color),
            "after_color": int(after_color),
            "count": int(count),
        }
        for (before_color, after_color), count in sorted(counter.items())
    ]


def _contact_pair_records(counter: Counter[Tuple[int, int]]) -> List[Dict[str, Any]]:
    return [
        {"colors": [int(first), int(second)], "contacts": int(count)}
        for (first, second), count in sorted(counter.items())
    ]


def _string_counter(counter: Counter[int]) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _clean_mapping_or_none(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return _json_ready(dict(value))


def _available_action_names(values: Iterable[Any]) -> List[str]:
    return [_normalize_action_name(value) for value in values]


def _normalize_action_name(action: Any) -> str:
    if isinstance(action, int):
        return ACTION_ID_TO_NAME.get(action, f"ACTION{action}")
    raw = str(action or "").strip().upper()
    if raw.isdigit():
        return ACTION_ID_TO_NAME.get(int(raw), f"ACTION{raw}")
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(child) for child in value]
    if isinstance(value, list):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _default_trace_paths(trace_dir: Path) -> List[Path]:
    return sorted(trace_dir.glob("*.steps.jsonl"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the M1.1 raw dataset.")
    parser.add_argument("trace_paths", nargs="*", type=Path)
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--exclude-reset", action="store_true")
    parser.add_argument("--max-observations", type=int, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    trace_paths = args.trace_paths or _default_trace_paths(args.traces_dir)
    observations = build_dataset(
        trace_paths,
        output_path=args.out,
        include_reset=not args.exclude_reset,
        max_observations=args.max_observations,
    )
    games = sorted({observation.game_id for observation in observations})
    summary = {
        "output_path": str(args.out),
        "observations": len(observations),
        "games": games,
        "game_count": len(games),
        "trace_count": len(trace_paths),
        "schema_version": SCHEMA_VERSION,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
