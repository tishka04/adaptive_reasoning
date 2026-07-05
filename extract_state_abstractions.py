"""Game-agnostic state abstraction features for the learned-abstraction pivot.

Three explicit layers keep generic and ar25-specific concerns from bleeding
together:

    core_features(grid)          -> fully generic, color-agnostic + per-color
    relational_features(grid)    -> top-k auto-inferred active color pairs
    game_specific_debug(grid, g) -> raw color identity / ar25 match_score

The public ``extract_state_features(grid)`` returns ``core ⊕ relational`` in a
stable column order declared by ``FEATURE_SCHEMA``. The debug layer is NEVER a
model input by default; it is emitted only on explicit request.

This module is intentionally self-contained (numpy only) so it imports fast,
generalizes across games, and never pulls in the ar25 environment stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Schema constants. Changing these changes FEATURE_SCHEMA (and the vector size).
# ---------------------------------------------------------------------------
TOP_N_COLORS = 4
TOP_K_PAIRS = 2
SIZE_BINS = 5
BACKGROUND_VALUES = (0,)

Grid = Sequence[Sequence[int]]

LARGEST_COMPONENT_FEATURE_SCHEMA = [
    "largest_component_bbox_min_y",
    "largest_component_bbox_min_x",
    "largest_component_bbox_max_y",
    "largest_component_bbox_max_x",
    "largest_component_width",
    "largest_component_height",
    "largest_component_bbox_area",
    "largest_component_fill_ratio",
    "largest_component_boundary_contacts",
    "largest_component_centroid_y",
    "largest_component_centroid_x",
    "largest_component_distance_to_nearest",
    "largest_component_nearest_size_ratio",
    "cursor_present",
    "cursor_y",
    "cursor_x",
    "cursor_distance_to_largest_centroid",
    "cursor_distance_to_largest_bbox",
]


@dataclass(frozen=True)
class _Component:
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

    def touches_boundary(self, shape: Tuple[int, int]) -> bool:
        h, w = shape
        return (
            self.min_y <= 0
            or self.min_x <= 0
            or self.max_y >= h - 1
            or self.max_x >= w - 1
        )


# ---------------------------------------------------------------------------
# Connected components (4-connectivity, per nonzero color).
# ---------------------------------------------------------------------------
def _as_array(grid: Grid) -> np.ndarray:
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2:
        arr = np.zeros((1, 1), dtype=np.int32)
    return arr


def _components_by_color(arr: np.ndarray) -> Dict[int, List[_Component]]:
    """Return {color: [components]} for every non-background color."""

    background = set(int(v) for v in BACKGROUND_VALUES)
    if arr.ndim != 2 or arr.size == 0:
        return {}
    height, width = arr.shape
    seen = np.zeros(arr.shape, dtype=bool)
    out: Dict[int, List[_Component]] = {}
    for y in range(height):
        for x in range(width):
            color = int(arr[y, x])
            if color in background or seen[y, x]:
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
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            out.setdefault(color, []).append(
                _Component(
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
    return out


def _colors_by_activity(components: Dict[int, List[_Component]]) -> List[int]:
    """Rank non-background colors by total occupied size (descending)."""

    totals = {
        color: sum(c.size for c in comps)
        for color, comps in components.items()
    }
    return sorted(totals, key=lambda c: (-totals[c], c))


def _nearest_distances(components: List[_Component]) -> Tuple[float, float, float]:
    if len(components) < 2:
        return 0.0, 0.0, 0.0
    centroids = np.array(
        [(c.centroid_y, c.centroid_x) for c in components], dtype=np.float32
    )
    nearest: List[float] = []
    for i in range(len(centroids)):
        deltas = centroids - centroids[i]
        dists = np.hypot(deltas[:, 0], deltas[:, 1])
        dists[i] = np.inf
        nearest.append(float(np.min(dists)))
    return float(min(nearest)), float(np.mean(nearest)), float(max(nearest))


def _largest_component(components: List[_Component]) -> Optional[_Component]:
    if not components:
        return None
    return sorted(
        components,
        key=lambda c: (-c.size, c.color, c.min_y, c.min_x, c.max_y, c.max_x),
    )[0]


def _component_boundary_contacts(component: _Component, shape: Tuple[int, int]) -> int:
    h, w = shape
    return int(component.min_y <= 0) + int(component.min_x <= 0) + int(
        component.max_y >= h - 1
    ) + int(component.max_x >= w - 1)


def _point_to_bbox_distance(y: float, x: float, component: _Component) -> float:
    dy = max(float(component.min_y) - y, 0.0, y - float(component.max_y))
    dx = max(float(component.min_x) - x, 0.0, x - float(component.max_x))
    return float(np.hypot(dy, dx))


def largest_component_local_features(
    grid: Grid,
    *,
    cursor: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Small local feature set for predicting largest-component effects.

    Coordinates and distances are normalized by grid size/diagonal so the
    specialized experiment stays more cross-game than raw pixel features.
    """

    arr = _as_array(grid)
    height, width = (int(arr.shape[0]), int(arr.shape[1]))
    shape = (height, width)
    diag = float(max(1e-6, np.hypot(height, width)))
    components_by_color = _components_by_color(arr)
    components = [c for comps in components_by_color.values() for c in comps]
    largest = _largest_component(components)
    empty = {name: 0.0 for name in LARGEST_COMPONENT_FEATURE_SCHEMA}
    if largest is None:
        return empty

    others = [c for c in components if c is not largest]
    if others:
        dists = [
            float(
                np.hypot(
                    c.centroid_y - largest.centroid_y,
                    c.centroid_x - largest.centroid_x,
                )
            )
            for c in others
        ]
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])
        nearest_size_ratio = float(others[nearest_idx].size / max(1, largest.size))
    else:
        nearest_dist = 0.0
        nearest_size_ratio = 0.0

    bbox_area = float(max(1, largest.width * largest.height))
    feats = {
        "largest_component_bbox_min_y": float(largest.min_y / max(1, height - 1)),
        "largest_component_bbox_min_x": float(largest.min_x / max(1, width - 1)),
        "largest_component_bbox_max_y": float(largest.max_y / max(1, height - 1)),
        "largest_component_bbox_max_x": float(largest.max_x / max(1, width - 1)),
        "largest_component_width": float(largest.width / max(1, width)),
        "largest_component_height": float(largest.height / max(1, height)),
        "largest_component_bbox_area": float(bbox_area / max(1, height * width)),
        "largest_component_fill_ratio": float(largest.size / bbox_area),
        "largest_component_boundary_contacts": float(_component_boundary_contacts(largest, shape) / 4.0),
        "largest_component_centroid_y": float(largest.centroid_y / max(1, height - 1)),
        "largest_component_centroid_x": float(largest.centroid_x / max(1, width - 1)),
        "largest_component_distance_to_nearest": float(nearest_dist / diag),
        "largest_component_nearest_size_ratio": nearest_size_ratio,
        "cursor_present": 0.0,
        "cursor_y": 0.0,
        "cursor_x": 0.0,
        "cursor_distance_to_largest_centroid": 0.0,
        "cursor_distance_to_largest_bbox": 0.0,
    }

    if cursor is not None:
        cy, cx = float(cursor[0]), float(cursor[1])
        feats["cursor_present"] = 1.0
        feats["cursor_y"] = float(cy / max(1, height - 1))
        feats["cursor_x"] = float(cx / max(1, width - 1))
        feats["cursor_distance_to_largest_centroid"] = float(
            np.hypot(cy - largest.centroid_y, cx - largest.centroid_x) / diag
        )
        feats["cursor_distance_to_largest_bbox"] = float(
            _point_to_bbox_distance(cy, cx, largest) / diag
        )
    return {name: float(feats.get(name, 0.0)) for name in LARGEST_COMPONENT_FEATURE_SCHEMA}


def _size_entropy(sizes: Sequence[int]) -> float:
    if not sizes:
        return 0.0
    total = float(sum(sizes))
    if total <= 0:
        return 0.0
    probs = np.array([s / total for s in sizes], dtype=np.float64)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ---------------------------------------------------------------------------
# Layer 1: core (generic) features.
# ---------------------------------------------------------------------------
def core_features(grid: Grid) -> Dict[str, float]:
    arr = _as_array(grid)
    shape = (int(arr.shape[0]), int(arr.shape[1]))
    height, width = shape
    components = _components_by_color(arr)
    all_components = [c for comps in components.values() for c in comps]
    all_sizes = [c.size for c in all_components]

    nonzero = int(np.count_nonzero(~np.isin(arr, list(BACKGROUND_VALUES))))
    total_cells = max(1, arr.size)
    values, counts = (np.unique(arr, return_counts=True) if arr.size else (np.array([]), np.array([])))
    dominant_fraction = float(counts.max() / total_cells) if counts.size else 0.0

    small_components = sum(1 for s in all_sizes if s <= 3)
    fragmentation_ratio = (
        small_components / len(all_components) if all_components else 0.0
    )
    boundary_touch = sum(1 for c in all_components if c.touches_boundary(shape))
    boundary_ratio = boundary_touch / len(all_components) if all_components else 0.0
    near_min, near_mean, near_max = _nearest_distances(all_components)

    if all_components:
        cy = np.array([c.centroid_y for c in all_components], dtype=np.float32)
        cx = np.array([c.centroid_x for c in all_components], dtype=np.float32)
        centroid_spread_y = float(np.std(cy))
        centroid_spread_x = float(np.std(cx))
    else:
        centroid_spread_y = centroid_spread_x = 0.0

    feats: Dict[str, float] = {
        "component_count": float(len(all_components)),
        "distinct_colors": float(len(components)),
        "nonzero_density": float(nonzero / total_cells),
        "dominant_color_fraction": dominant_fraction,
        "largest_component_size": float(max(all_sizes) if all_sizes else 0),
        "mean_component_size": float(np.mean(all_sizes)) if all_sizes else 0.0,
        "fragmentation_ratio": float(fragmentation_ratio),
        "size_entropy": _size_entropy(all_sizes),
        "boundary_contact_ratio": float(boundary_ratio),
        "grid_height": float(height),
        "grid_width": float(width),
        "nearest_distance_min": near_min,
        "nearest_distance_mean": near_mean,
        "nearest_distance_max": near_max,
        "centroid_spread_y": centroid_spread_y,
        "centroid_spread_x": centroid_spread_x,
    }

    # Global size distribution (fractions over fixed bins by component size).
    bins = _size_histogram(all_sizes, SIZE_BINS, height * width)
    for i, frac in enumerate(bins):
        feats[f"size_bin_{i}_frac"] = float(frac)

    # Per-color features for the top-N most active colors (ranked, padded).
    ranked = _colors_by_activity(components)
    for rank in range(TOP_N_COLORS):
        prefix = f"color_{rank}"
        if rank < len(ranked):
            color = ranked[rank]
            comps = components[color]
            sizes = [c.size for c in comps]
            feats[f"{prefix}_present"] = 1.0
            feats[f"{prefix}_component_count"] = float(len(comps))
            feats[f"{prefix}_largest_component"] = float(max(sizes))
            feats[f"{prefix}_total_size"] = float(sum(sizes))
            feats[f"{prefix}_size_mean"] = float(np.mean(sizes))
            feats[f"{prefix}_size_std"] = float(np.std(sizes))
            feats[f"{prefix}_boundary_contacts"] = float(
                sum(1 for c in comps if c.touches_boundary(shape))
            )
            feats[f"{prefix}_centroid_y"] = float(np.mean([c.centroid_y for c in comps]))
            feats[f"{prefix}_centroid_x"] = float(np.mean([c.centroid_x for c in comps]))
        else:
            for suffix in (
                "present",
                "component_count",
                "largest_component",
                "total_size",
                "size_mean",
                "size_std",
                "boundary_contacts",
                "centroid_y",
                "centroid_x",
            ):
                feats[f"{prefix}_{suffix}"] = 0.0
    return feats


def _size_histogram(sizes: Sequence[int], bins: int, max_cells: int) -> List[float]:
    if not sizes:
        return [0.0] * bins
    # Log-scaled edges so small fragments and large blobs separate well.
    edges = np.linspace(0.0, np.log1p(max(1, max_cells)), bins + 1)
    logged = np.log1p(np.array(sizes, dtype=np.float64))
    hist, _ = np.histogram(logged, bins=edges)
    total = float(hist.sum())
    if total <= 0:
        return [0.0] * bins
    return [float(h / total) for h in hist]


# ---------------------------------------------------------------------------
# Layer 2: relational features over top-k auto-inferred active color pairs.
# ---------------------------------------------------------------------------
def _pair_correspondence(
    a_comps: List[_Component],
    b_comps: List[_Component],
) -> Dict[str, float]:
    a_sizes = [c.size for c in a_comps]
    b_sizes = [c.size for c in b_comps]
    count_sim = _safe_similarity(len(a_comps), len(b_comps))
    size_sim = _safe_similarity(sum(a_sizes), sum(b_sizes))
    dist_sim = _size_distribution_similarity(a_sizes, b_sizes)
    nearest = _cross_nearest_distance(a_comps, b_comps)
    correspondence = 100.0 * (
        0.4 * count_sim + 0.3 * size_sim + 0.3 * dist_sim
    )
    return {
        "global_correspondence": float(correspondence),
        "count_sim": float(count_sim),
        "size_sim": float(size_sim),
        "nearest_dist": float(nearest),
        "a_count": float(len(a_comps)),
        "a_largest": float(max(a_sizes) if a_sizes else 0),
        "b_count": float(len(b_comps)),
        "b_largest": float(max(b_sizes) if b_sizes else 0),
        "present": 1.0,
    }


def _safe_similarity(left: float, right: float) -> float:
    left = float(left)
    right = float(right)
    return max(0.0, 1.0 - abs(left - right) / max(1.0, abs(left), abs(right)))


def _size_distribution_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    max_value = max(max(left), max(right), 1)
    rng = (0.0, float(max_value))
    lh, _ = np.histogram(np.array(left, dtype=np.float32), bins=SIZE_BINS, range=rng)
    rh, _ = np.histogram(np.array(right, dtype=np.float32), bins=SIZE_BINS, range=rng)
    ls, rs = float(lh.sum()), float(rh.sum())
    if ls <= 0 or rs <= 0:
        return 0.0
    return float(1.0 - 0.5 * np.abs(lh / ls - rh / rs).sum())


def _cross_nearest_distance(a_comps: List[_Component], b_comps: List[_Component]) -> float:
    if not a_comps or not b_comps:
        return 0.0
    a = np.array([(c.centroid_y, c.centroid_x) for c in a_comps], dtype=np.float32)
    b = np.array([(c.centroid_y, c.centroid_x) for c in b_comps], dtype=np.float32)
    dists = []
    for point in a:
        deltas = b - point
        dists.append(float(np.min(np.hypot(deltas[:, 0], deltas[:, 1]))))
    return float(np.mean(dists))


def _empty_pair_features() -> Dict[str, float]:
    return {
        "global_correspondence": 0.0,
        "count_sim": 0.0,
        "size_sim": 0.0,
        "nearest_dist": 0.0,
        "a_count": 0.0,
        "a_largest": 0.0,
        "b_count": 0.0,
        "b_largest": 0.0,
        "present": 0.0,
    }


def infer_active_color_pairs(grid: Grid, *, k: int = TOP_K_PAIRS) -> List[Tuple[int, int]]:
    """Return the top-k color pairs ranked by combined activity (size)."""

    arr = _as_array(grid)
    components = _components_by_color(arr)
    ranked = _colors_by_activity(components)
    pairs: List[Tuple[int, int]] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            pairs.append((ranked[i], ranked[j]))
    return pairs[:k]


def relational_features(grid: Grid) -> Dict[str, float]:
    arr = _as_array(grid)
    components = _components_by_color(arr)
    ranked = _colors_by_activity(components)
    candidate_pairs: List[Tuple[int, int]] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            candidate_pairs.append((ranked[i], ranked[j]))

    feats: Dict[str, float] = {}
    for rank in range(TOP_K_PAIRS):
        prefix = f"top_pair_{rank}"
        if rank < len(candidate_pairs):
            color_a, color_b = candidate_pairs[rank]
            pair = _pair_correspondence(components[color_a], components[color_b])
        else:
            pair = _empty_pair_features()
        for name, value in pair.items():
            feats[f"{prefix}_{name}"] = float(value)
    return feats


# ---------------------------------------------------------------------------
# Layer 3: game-specific debug (never a model input by default).
# ---------------------------------------------------------------------------
def game_specific_debug(grid: Grid, game_id: str = "") -> Dict[str, object]:
    arr = _as_array(grid)
    components = _components_by_color(arr)
    ranked = _colors_by_activity(components)
    debug: Dict[str, object] = {
        "colors_sorted_by_activity": [int(c) for c in ranked],
        "active_color_pairs": [list(p) for p in infer_active_color_pairs(grid)],
    }
    short = (game_id or "").split("-", 1)[0]
    if short == "ar25":
        # Lazy import keeps the ar25 stack out of the generic path.
        try:
            from task_program_guided_level7 import global_correspondence_score, match_score

            debug["ar25_match_score"] = float(
                match_score(grid, pair_colors=(10, 11)).score
            )
            debug["ar25_global_correspondence"] = float(
                global_correspondence_score(grid, pair_colors=(10, 11)).score
            )
        except Exception as exc:  # pragma: no cover - debug only
            debug["ar25_debug_error"] = str(exc)
    return debug


# ---------------------------------------------------------------------------
# Public API: stable schema, vector, and deltas.
# ---------------------------------------------------------------------------
def extract_state_features(grid: Grid) -> Dict[str, float]:
    """Return core ⊕ relational features as a flat, ordered dict."""

    feats = core_features(grid)
    feats.update(relational_features(grid))
    return feats


def _build_schema() -> List[str]:
    # Use a small empty grid to materialize the deterministic key order.
    blank = [[0, 0], [0, 0]]
    return list(extract_state_features(blank).keys())


FEATURE_SCHEMA: List[str] = _build_schema()
FEATURE_DIM: int = len(FEATURE_SCHEMA)


def features_to_vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([float(feats.get(name, 0.0)) for name in FEATURE_SCHEMA], dtype=np.float32)


def extract_state_vector(grid: Grid) -> np.ndarray:
    return features_to_vector(extract_state_features(grid))


def delta_features(
    prev: Dict[str, float],
    nxt: Dict[str, float],
) -> Dict[str, float]:
    """Return ``next - prev`` for every schema feature, keyed ``delta_<name>``."""

    return {
        f"delta_{name}": float(nxt.get(name, 0.0)) - float(prev.get(name, 0.0))
        for name in FEATURE_SCHEMA
    }


if __name__ == "__main__":
    demo = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 1, 0, 0, 2],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 3, 0],
    ]
    print(f"FEATURE_DIM = {FEATURE_DIM}")
    feats = extract_state_features(demo)
    for name in FEATURE_SCHEMA:
        print(f"  {name:32s} {feats[name]:.4f}")
    print("debug:", game_specific_debug(demo))
