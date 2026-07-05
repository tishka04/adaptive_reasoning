"""A12 cross-game correspondence discovery.

This module intentionally stops before planning. It scans non-ar25 transition
traces, infers source-target color pairs and relation predicates, and records
candidate correspondence hypotheses as unresolved until an external oracle or
active experiment validates them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .correspondence_hypothesis import (
    CorrespondenceHypothesis,
    correspondence_key,
)
from .epistemic_metrics import (
    EpistemicScore,
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    score_beliefs,
)

PredicateGenerator = Callable[..., Iterable[str]]
AnchorExpander = Callable[..., Iterable[Any]]


@dataclass(frozen=True)
class SourceTargetPredicate:
    """A predicate inferred for one source-target color pair."""

    name: str
    support: int = 0


@dataclass(frozen=True)
class DiscoveredCorrespondenceCandidate:
    """A non-ar25 source-target correspondence candidate."""

    game_id: str
    action: str
    source_color: int
    target_color: int
    relation: str = "modifies"
    support: int = 0
    transition_support: int = 0
    predicates: Tuple[SourceTargetPredicate, ...] = ()
    weak_ready_candidates: Tuple[str, ...] = ()
    strong_ready_candidates: Tuple[str, ...] = ()

    @property
    def pair_colors(self) -> Tuple[int, int]:
        return (self.source_color, self.target_color)

    @property
    def key(self) -> str:
        return correspondence_key(self.action, self.relation, self.pair_colors)

    def to_hypothesis(self) -> CorrespondenceHypothesis:
        return CorrespondenceHypothesis(
            action=self.action,
            relation=self.relation,
            pair_colors=self.pair_colors,
            statement=(
                f"{self.action} may {self.relation} a source-target "
                f"relation from color {self.source_color} to {self.target_color}"
            ),
            support=self.transition_support,
            experiments_spent=self.transition_support,
            status=HypothesisStatus.UNRESOLVED,
        )

    def to_record(self) -> HypothesisRecord:
        return HypothesisRecord(
            key=self.key,
            description=(
                f"{self.action} candidate {self.relation} correspondence "
                f"colors {self.source_color}->{self.target_color}"
            ),
            status=HypothesisStatus.UNRESOLVED,
            support=self.transition_support,
            contradictions=0,
            experiments_spent=self.transition_support,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "game_id": self.game_id,
            "action": self.action,
            "source_color": self.source_color,
            "target_color": self.target_color,
            "relation": self.relation,
            "support": self.support,
            "transition_support": self.transition_support,
            "predicates": [
                {"name": predicate.name, "support": predicate.support}
                for predicate in self.predicates
            ],
            "weak_ready_candidates": list(self.weak_ready_candidates),
            "strong_ready_candidates": list(self.strong_ready_candidates),
        }


@dataclass
class CrossGameCorrespondenceDiscoveryRun:
    """Result of A12 trace-only correspondence discovery."""

    game_id: str
    trace_path: Path
    candidates: List[DiscoveredCorrespondenceCandidate] = field(default_factory=list)
    score: EpistemicScore | None = None
    transitions_scanned: int = 0

    @property
    def hypotheses(self) -> List[CorrespondenceHypothesis]:
        return [candidate.to_hypothesis() for candidate in self.candidates]

    @property
    def records(self) -> List[HypothesisRecord]:
        return [candidate.to_record() for candidate in self.candidates]

    @property
    def source_colors(self) -> List[int]:
        return _dedupe(candidate.source_color for candidate in self.candidates)

    @property
    def target_colors(self) -> List[int]:
        return _dedupe(candidate.target_color for candidate in self.candidates)

    @property
    def source_target_predicates(self) -> List[str]:
        names: List[str] = []
        for candidate in self.candidates:
            names.extend(predicate.name for predicate in candidate.predicates)
        return _dedupe(names)

    @property
    def wrong_confirmations(self) -> int:
        return 0 if self.score is None else self.score.wrong_confirmations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
            "transitions_scanned": self.transitions_scanned,
            "source_colors": self.source_colors,
            "target_colors": self.target_colors,
            "source_target_predicates": self.source_target_predicates,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "wrong_confirmations": self.wrong_confirmations,
            "score": self.score.to_dict() if self.score is not None else None,
        }


class _PairStats:
    def __init__(self) -> None:
        self.pixel_support = 0
        self.transition_support = 0
        self.predicates: Counter[str] = Counter()


def discover_cross_game_correspondences(
    trace_path: Path | str,
    *,
    game_id: str = "",
    max_steps: int | None = None,
    min_pixel_support: int = 100,
    top_k: int = 5,
    predicate_generator: PredicateGenerator | None = None,
    anchor_expander: AnchorExpander | None = None,
) -> CrossGameCorrespondenceDiscoveryRun:
    """Infer non-ar25 source-target correspondence candidates from a trace."""
    path = Path(trace_path)
    generate_predicates = predicate_generator or _source_target_predicates
    stats: dict[tuple[str, int, int], _PairStats] = defaultdict(_PairStats)
    inferred_game_id = game_id
    transitions_scanned = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_steps is not None and transitions_scanned >= max_steps:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            inferred_game_id = inferred_game_id or str(item.get("game_id", ""))
            action = _normalize_action_name(item.get("action"))
            if action == "RESET":
                continue
            before = item.get("frame_before")
            after = item.get("frame_after")
            if before is None or after is None:
                continue
            before_grid = np.asarray(before, dtype=np.int32)
            after_grid = np.asarray(after, dtype=np.int32)
            if before_grid.shape != after_grid.shape or before_grid.ndim != 2:
                continue
            background = _infer_background(before_grid, after_grid)
            changed_pairs = _changed_color_pairs(
                before_grid,
                after_grid,
                background=background,
            )
            expanded_anchors = tuple(
                anchor_expander(
                    before_grid,
                    after_grid,
                    background=background,
                )
                if anchor_expander is not None
                else ()
            )
            if not changed_pairs and not expanded_anchors:
                continue
            transitions_scanned += 1
            transition_keys: set[tuple[str, int, int]] = set()
            for (source, target), pixels in changed_pairs.items():
                key = (action, source, target)
                item_stats = stats[key]
                item_stats.pixel_support += int(pixels)
                item_stats.transition_support += 1
                transition_keys.add(key)
                for predicate in generate_predicates(
                    before_grid,
                    after_grid,
                    source_color=source,
                    target_color=target,
                ):
                    item_stats.predicates[predicate] += 1
            for anchor in expanded_anchors:
                source = int(getattr(anchor, "source_color", -1))
                target = int(getattr(anchor, "target_color", -1))
                if source < 0 or target < 0 or source == target:
                    continue
                key = (action, source, target)
                item_stats = stats[key]
                item_stats.pixel_support += max(
                    1,
                    int(getattr(anchor, "support", 1) or 1),
                )
                if key not in transition_keys:
                    item_stats.transition_support += 1
                    transition_keys.add(key)
                for predicate in getattr(anchor, "predicates", ()) or ():
                    item_stats.predicates[str(predicate)] += 1

    candidates = _build_candidates(
        inferred_game_id or path.stem.split(".")[0],
        stats,
        min_pixel_support=min_pixel_support,
        top_k=top_k,
    )
    run = CrossGameCorrespondenceDiscoveryRun(
        game_id=inferred_game_id or path.stem.split(".")[0],
        trace_path=path,
        candidates=candidates,
        transitions_scanned=transitions_scanned,
    )
    run.score = score_beliefs(
        run.records,
        MechanicsOracle(run.game_id),
        experiment_actions=max(1, transitions_scanned),
    )
    return run


def _build_candidates(
    game_id: str,
    stats: dict[tuple[str, int, int], _PairStats],
    *,
    min_pixel_support: int,
    top_k: int,
) -> List[DiscoveredCorrespondenceCandidate]:
    candidates: List[DiscoveredCorrespondenceCandidate] = []
    for (action, source, target), item in stats.items():
        if item.pixel_support < min_pixel_support:
            continue
        predicates = tuple(
            SourceTargetPredicate(name=name, support=support)
            for name, support in item.predicates.most_common()
        )
        predicate_names = {predicate.name for predicate in predicates}
        weak_ready = ["selected_pair_exists"]
        strong_ready = ["source_target_color_transform_repeats"]
        if "aligned_with" in predicate_names:
            strong_ready.append("source_target_relation_satisfied")
        if "same_shape" in predicate_names:
            strong_ready.append("selected_source_matches_target_shape")
        candidates.append(
            DiscoveredCorrespondenceCandidate(
                game_id=game_id,
                action=action,
                source_color=source,
                target_color=target,
                support=item.pixel_support,
                transition_support=item.transition_support,
                predicates=predicates,
                weak_ready_candidates=tuple(weak_ready),
                strong_ready_candidates=tuple(_dedupe(strong_ready)),
            )
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.support,
            candidate.transition_support,
            len(candidate.predicates),
            candidate.key,
        ),
        reverse=True,
    )[: max(1, int(top_k))]


def _changed_color_pairs(
    before: np.ndarray,
    after: np.ndarray,
    *,
    background: int,
) -> Counter[tuple[int, int]]:
    mask = before != after
    pairs: Counter[tuple[int, int]] = Counter()
    if not bool(mask.any()):
        return pairs
    for source, target in zip(before[mask].ravel(), after[mask].ravel()):
        source_color = int(source)
        target_color = int(target)
        if source_color == target_color:
            continue
        if source_color == background or target_color == background:
            continue
        pairs[(source_color, target_color)] += 1
    return pairs


def _source_target_predicates(
    before: np.ndarray,
    after: np.ndarray,
    *,
    source_color: int,
    target_color: int,
) -> List[str]:
    predicates = ["source_target_color_transform"]
    if _both_present(before, source_color, target_color) or _both_present(
        after,
        source_color,
        target_color,
    ):
        predicates.append("paired_with")
    if _same_shape_exists(before, source_color, target_color) or _same_shape_exists(
        after,
        source_color,
        target_color,
    ):
        predicates.append("same_shape")
    if _aligned_exists(before, source_color, target_color) or _aligned_exists(
        after,
        source_color,
        target_color,
    ):
        predicates.append("aligned_with")
    if _adjacent_exists(before, source_color, target_color) or _adjacent_exists(
        after,
        source_color,
        target_color,
    ):
        predicates.append("adjacent_to")
    return predicates


@dataclass(frozen=True)
class _Component:
    color: int
    points: Tuple[Tuple[int, int], ...]
    bbox: Tuple[int, int, int, int]

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


def _components(grid: np.ndarray, color: int) -> List[_Component]:
    mask = grid == int(color)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[_Component] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
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
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            components.append(
                _Component(
                    color=int(color),
                    points=tuple(points),
                    bbox=(min(ys), min(xs), max(ys), max(xs)),
                )
            )
    return components


def _both_present(grid: np.ndarray, source_color: int, target_color: int) -> bool:
    return bool(np.any(grid == int(source_color)) and np.any(grid == int(target_color)))


def _same_shape_exists(
    grid: np.ndarray,
    source_color: int,
    target_color: int,
) -> bool:
    sources = _components(grid, source_color)
    targets = _components(grid, target_color)
    for source in sources:
        for target in targets:
            if (
                source.size == target.size
                and source.height == target.height
                and source.width == target.width
            ):
                return True
    return False


def _aligned_exists(grid: np.ndarray, source_color: int, target_color: int) -> bool:
    sources = _components(grid, source_color)
    targets = _components(grid, target_color)
    for source in sources:
        sy, sx = source.centroid
        for target in targets:
            ty, tx = target.centroid
            if abs(sy - ty) <= 1.0 or abs(sx - tx) <= 1.0:
                return True
    return False


def _adjacent_exists(grid: np.ndarray, source_color: int, target_color: int) -> bool:
    sources = _components(grid, source_color)
    targets = _components(grid, target_color)
    for source in sources:
        for target in targets:
            if _bbox_distance(source.bbox, target.bbox) <= 1:
                return True
    return False


def _bbox_distance(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> int:
    f_y1, f_x1, f_y2, f_x2 = first
    s_y1, s_x1, s_y2, s_x2 = second
    dy = max(0, max(s_y1 - f_y2 - 1, f_y1 - s_y2 - 1))
    dx = max(0, max(s_x1 - f_x2 - 1, f_x1 - s_x2 - 1))
    return max(dy, dx)


def _infer_background(before: np.ndarray, after: np.ndarray) -> int:
    values, counts = np.unique(
        np.concatenate([before.ravel(), after.ravel()]),
        return_counts=True,
    )
    return int(values[int(np.argmax(counts))])


def _normalize_action_name(action: Any) -> str:
    raw = str(action or "").strip().upper()
    if raw.isdigit():
        return "RESET" if int(raw) == 0 else f"ACTION{raw}"
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw


def _dedupe(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover non-ar25 source-target correspondence candidates."
    )
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--game-id", default="")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-pixel-support", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--m1-predicates", action="store_true")
    parser.add_argument("--m1-anchor-expansion", action="store_true")
    parser.add_argument(
        "--m1-accepted-invariants",
        type=Path,
        default=Path("diagnostics/m1/accepted_invariants.json"),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    predicate_generator = None
    anchor_expander = None
    if args.m1_predicates:
        from .m1.predicate_generation import build_m1_predicate_generator

        predicate_generator = build_m1_predicate_generator(
            accepted_invariants_path=args.m1_accepted_invariants,
        )
    if args.m1_anchor_expansion:
        from .m1.anchor_expansion import build_m1_anchor_expander

        anchor_expander = build_m1_anchor_expander()
    run = discover_cross_game_correspondences(
        args.trace_path,
        game_id=args.game_id,
        max_steps=args.max_steps,
        min_pixel_support=args.min_pixel_support,
        top_k=args.top_k,
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
    )
    print(json.dumps(run.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
