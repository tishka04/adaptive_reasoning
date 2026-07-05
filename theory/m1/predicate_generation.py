"""M1.3 conservative predicate generation from accepted M1.2 invariants.

M1.3 is opt-in. The default A12/A25 path remains the historical predicate
generator unless callers explicitly pass the generator returned here.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .invariant_miner import (
    DEFAULT_ACCEPTED_PATH,
    LatentInvariant,
    load_invariants_json,
)

DEFAULT_PRETEST_OUTPUT_PATH = Path("diagnostics") / "m1" / "predicate_coverage_pretest.json"
DEFAULT_BLOCKED_GAMES = ("bp35", "cd82", "dc22")
DEFAULT_PRETEST_TOP_K = 100
DEFAULT_PRETEST_MIN_PIXEL_SUPPORT = 1
HISTORICAL_PREDICATES = (
    "source_target_color_transform",
    "paired_with",
    "same_shape",
    "aligned_with",
    "adjacent_to",
)
ACTIONABLE_INVARIANTS = frozenset(
    {
        ("object_creation", "appears"),
        ("object_removal", "appears"),
        ("object_motion", "appears"),
        ("object_count", "preserved"),
        ("color_pair_change", "appears"),
        ("adjacency", "preserved"),
        ("adjacency", "modified"),
    }
)
EXCLUDED_INITIAL_INVARIANTS = frozenset(
    {
        ("grid_extent", "preserved"),
        ("terminal_state", "absent"),
        ("level_progress", "absent"),
    }
)

PredicateGenerator = Callable[..., Iterable[str]]
AnchorExpander = Callable[..., Iterable[Any]]


@dataclass(frozen=True)
class PredicateInjectionRule:
    """One accepted invariant that may emit an M1 predicate."""

    invariant_name: str
    predicate_name: str
    attribute: str
    outcome: str
    support: int
    games_supporting: Tuple[str, ...]
    cross_game_score: float
    novelty_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_name": self.invariant_name,
            "predicate_name": self.predicate_name,
            "attribute": self.attribute,
            "outcome": self.outcome,
            "support": int(self.support),
            "games_supporting": list(self.games_supporting),
            "cross_game_score": round(float(self.cross_game_score), 6),
            "novelty_score": round(float(self.novelty_score), 6),
        }


@dataclass(frozen=True)
class PredicateCoverageTrace:
    """Offline M1.3 predicate-coverage metrics for one trace."""

    game_id: str
    trace_path: str
    mode: str
    candidate_pairs_per_trace: int
    unique_predicates_per_trace: int
    relation_candidates_generated: int
    m1_predicates_per_trace: int = 0
    candidate_count: int = 0
    source_target_predicates: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": self.trace_path,
            "mode": self.mode,
            "candidate_pairs_per_trace": int(self.candidate_pairs_per_trace),
            "unique_predicates_per_trace": int(self.unique_predicates_per_trace),
            "relation_candidates_generated": int(self.relation_candidates_generated),
            "m1_predicates_per_trace": int(self.m1_predicates_per_trace),
            "candidate_count": int(self.candidate_count),
            "source_target_predicates": list(self.source_target_predicates),
        }


@dataclass(frozen=True)
class PredicateCoverageComparison:
    """M1 off/on predicate-coverage pre-test for a trace."""

    game_id: str
    trace_path: str
    off: PredicateCoverageTrace
    on: PredicateCoverageTrace

    @property
    def unique_predicates_delta(self) -> int:
        return self.on.unique_predicates_per_trace - self.off.unique_predicates_per_trace

    @property
    def candidate_pairs_delta(self) -> int:
        return self.on.candidate_pairs_per_trace - self.off.candidate_pairs_per_trace

    @property
    def relation_candidates_delta(self) -> int:
        return (
            self.on.relation_candidates_generated
            - self.off.relation_candidates_generated
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": self.trace_path,
            "off": self.off.to_dict(),
            "on": self.on.to_dict(),
            "deltas": {
                "unique_predicates_per_trace": self.unique_predicates_delta,
                "candidate_pairs_per_trace": self.candidate_pairs_delta,
                "relation_candidates_generated": self.relation_candidates_delta,
            },
        }


@dataclass(frozen=True)
class PredicateCoveragePretest:
    """Offline pre-test result; no live env and no scientific revision."""

    accepted_invariants_path: str
    min_pixel_support: int = DEFAULT_PRETEST_MIN_PIXEL_SUPPORT
    top_k: int = DEFAULT_PRETEST_TOP_K
    anchor_expansion_enabled: bool = False
    rules: Tuple[PredicateInjectionRule, ...] = ()
    comparisons: Tuple[PredicateCoverageComparison, ...] = ()
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    @property
    def average_unique_predicates_off(self) -> float:
        return _average(item.off.unique_predicates_per_trace for item in self.comparisons)

    @property
    def average_unique_predicates_on(self) -> float:
        return _average(item.on.unique_predicates_per_trace for item in self.comparisons)

    @property
    def average_candidate_pairs_off(self) -> float:
        return _average(item.off.candidate_pairs_per_trace for item in self.comparisons)

    @property
    def average_candidate_pairs_on(self) -> float:
        return _average(item.on.candidate_pairs_per_trace for item in self.comparisons)

    @property
    def average_relation_candidates_off(self) -> float:
        return _average(item.off.relation_candidates_generated for item in self.comparisons)

    @property
    def average_relation_candidates_on(self) -> float:
        return _average(item.on.relation_candidates_generated for item in self.comparisons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_invariants_path": self.accepted_invariants_path,
            "min_pixel_support": int(self.min_pixel_support),
            "top_k": int(self.top_k),
            "anchor_expansion_enabled": bool(self.anchor_expansion_enabled),
            "rule_count": len(self.rules),
            "rules": [rule.to_dict() for rule in self.rules],
            "trace_count": len(self.comparisons),
            "averages": {
                "unique_predicates_per_trace": {
                    "off": round(self.average_unique_predicates_off, 4),
                    "on": round(self.average_unique_predicates_on, 4),
                    "delta": round(
                        self.average_unique_predicates_on
                        - self.average_unique_predicates_off,
                        4,
                    ),
                },
                "candidate_pairs_per_trace": {
                    "off": round(self.average_candidate_pairs_off, 4),
                    "on": round(self.average_candidate_pairs_on, 4),
                    "delta": round(
                        self.average_candidate_pairs_on
                        - self.average_candidate_pairs_off,
                        4,
                    ),
                },
                "relation_candidates_generated": {
                    "off": round(self.average_relation_candidates_off, 4),
                    "on": round(self.average_relation_candidates_on, 4),
                    "delta": round(
                        self.average_relation_candidates_on
                        - self.average_relation_candidates_off,
                        4,
                    ),
                },
            },
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def historical_predicates(
    before: np.ndarray,
    after: np.ndarray,
    *,
    source_color: int,
    target_color: int,
) -> List[str]:
    """The original five-predicate A12 vocabulary, kept for fallback."""
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


def actionable_invariant_rules(
    invariants: Iterable[LatentInvariant | Mapping[str, Any]],
    *,
    min_games: int = 2,
) -> Tuple[PredicateInjectionRule, ...]:
    """Select the initial conservative M1.3 injection subset."""
    rules: List[PredicateInjectionRule] = []
    for raw in invariants:
        invariant = raw if isinstance(raw, LatentInvariant) else LatentInvariant.from_dict(raw)
        signature = (invariant.attribute, invariant.outcome)
        if invariant.rejected_reason is not None:
            continue
        if signature in EXCLUDED_INITIAL_INVARIANTS:
            continue
        if signature not in ACTIONABLE_INVARIANTS:
            continue
        if len(invariant.games_supporting) < int(min_games):
            continue
        rules.append(
            PredicateInjectionRule(
                invariant_name=invariant.name,
                predicate_name=f"m1_{invariant.name}",
                attribute=invariant.attribute,
                outcome=invariant.outcome,
                support=invariant.support,
                games_supporting=tuple(sorted(invariant.games_supporting)),
                cross_game_score=invariant.cross_game_score,
                novelty_score=invariant.novelty_score,
            )
        )
    return tuple(
        sorted(
            rules,
            key=lambda rule: (
                -rule.cross_game_score,
                -rule.support,
                rule.predicate_name,
            ),
        )
    )


def load_actionable_invariant_rules(
    path: str | Path = DEFAULT_ACCEPTED_PATH,
    *,
    min_games: int = 2,
) -> Tuple[PredicateInjectionRule, ...]:
    return actionable_invariant_rules(load_invariants_json(path), min_games=min_games)


def generate_predicates(
    before: np.ndarray,
    after: np.ndarray,
    *,
    source_color: int,
    target_color: int,
    rules: Sequence[PredicateInjectionRule] = (),
    include_historical: bool = True,
) -> List[str]:
    """Generate historical predicates plus opted-in M1 invariant predicates."""
    predicates: List[str] = []
    if include_historical:
        predicates.extend(
            historical_predicates(
                before,
                after,
                source_color=source_color,
                target_color=target_color,
            )
        )
    for rule in rules:
        if _rule_holds(
            rule,
            before,
            after,
            source_color=int(source_color),
            target_color=int(target_color),
        ):
            predicates.append(rule.predicate_name)
    return _dedupe(predicates)


def build_m1_predicate_generator(
    *,
    accepted_invariants_path: str | Path = DEFAULT_ACCEPTED_PATH,
    rules: Sequence[PredicateInjectionRule] | None = None,
    min_games: int = 2,
    include_historical: bool = True,
) -> PredicateGenerator:
    selected_rules = (
        tuple(rules)
        if rules is not None
        else load_actionable_invariant_rules(
            accepted_invariants_path,
            min_games=min_games,
        )
    )

    def _generator(
        before: np.ndarray,
        after: np.ndarray,
        *,
        source_color: int,
        target_color: int,
    ) -> List[str]:
        return generate_predicates(
            before,
            after,
            source_color=source_color,
            target_color=target_color,
            rules=selected_rules,
            include_historical=include_historical,
        )

    return _generator


def run_predicate_coverage_pretest(
    trace_paths: Sequence[str | Path],
    *,
    accepted_invariants_path: str | Path = DEFAULT_ACCEPTED_PATH,
    min_pixel_support: int = DEFAULT_PRETEST_MIN_PIXEL_SUPPORT,
    top_k: int = DEFAULT_PRETEST_TOP_K,
    anchor_expander: AnchorExpander | None = None,
) -> PredicateCoveragePretest:
    """Compare M1 off/on coverage without active environment execution."""
    from theory.cross_game_correspondence_discovery import (
        CrossGameCorrespondenceDiscoveryRun,
        discover_cross_game_correspondences,
    )

    rules = load_actionable_invariant_rules(accepted_invariants_path)
    generator = build_m1_predicate_generator(rules=rules)
    comparisons: List[PredicateCoverageComparison] = []
    for trace_path in trace_paths:
        path = Path(trace_path)
        off = discover_cross_game_correspondences(
            path,
            min_pixel_support=min_pixel_support,
            top_k=top_k,
        )
        on = discover_cross_game_correspondences(
            path,
            min_pixel_support=min_pixel_support,
            top_k=top_k,
            predicate_generator=generator,
            anchor_expander=anchor_expander,
        )
        comparisons.append(
            PredicateCoverageComparison(
                game_id=on.game_id,
                trace_path=str(path),
                off=_coverage_from_run(off, "m1_off"),
                on=_coverage_from_run(on, "m1_on"),
            )
        )
    return PredicateCoveragePretest(
        accepted_invariants_path=str(accepted_invariants_path),
        min_pixel_support=int(min_pixel_support),
        top_k=int(top_k),
        anchor_expansion_enabled=anchor_expander is not None,
        rules=rules,
        comparisons=tuple(comparisons),
    )


def write_pretest_result(
    result: PredicateCoveragePretest,
    output_path: str | Path = DEFAULT_PRETEST_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_blocked_trace_paths(
    *,
    traces_dir: str | Path = "human_traces",
    games: Sequence[str] = DEFAULT_BLOCKED_GAMES,
) -> Tuple[Path, ...]:
    paths = sorted(Path(traces_dir).glob("*.steps.jsonl"))
    selected: Dict[str, Path] = {}
    prefixes = tuple(str(game) for game in games)
    for path in paths:
        prefix = path.name.split("-", 1)[0]
        if prefix not in prefixes:
            continue
        previous = selected.get(prefix)
        if previous is None or path.name > previous.name:
            selected[prefix] = path
    return tuple(selected[prefix] for prefix in prefixes if prefix in selected)


def _coverage_from_run(
    run: Any,
    mode: str,
) -> PredicateCoverageTrace:
    pairs = {
        (candidate.action, candidate.source_color, candidate.target_color)
        for candidate in run.candidates
    }
    predicates = tuple(run.source_target_predicates)
    return PredicateCoverageTrace(
        game_id=run.game_id,
        trace_path=str(run.trace_path),
        mode=mode,
        candidate_pairs_per_trace=len(pairs),
        unique_predicates_per_trace=len(predicates),
        relation_candidates_generated=sum(
            len(candidate.predicates) for candidate in run.candidates
        ),
        m1_predicates_per_trace=len(
            [predicate for predicate in predicates if predicate.startswith("m1_")]
        ),
        candidate_count=len(run.candidates),
        source_target_predicates=predicates,
    )


def _rule_holds(
    rule: PredicateInjectionRule,
    before: np.ndarray,
    after: np.ndarray,
    *,
    source_color: int,
    target_color: int,
) -> bool:
    signature = (rule.attribute, rule.outcome)
    if signature == ("color_pair_change", "appears"):
        return _pair_change_pixels(before, after, source_color, target_color) > 0
    if signature == ("object_creation", "appears"):
        return (
            _component_count(after, target_color) > _component_count(before, target_color)
            or (
                _total_component_count(after) > _total_component_count(before)
                and _cell_count(after, target_color) > _cell_count(before, target_color)
            )
        )
    if signature == ("object_removal", "appears"):
        return (
            _component_count(after, source_color) < _component_count(before, source_color)
            or (
                _total_component_count(after) < _total_component_count(before)
                and _cell_count(after, source_color) < _cell_count(before, source_color)
            )
        )
    if signature == ("object_motion", "appears"):
        return _color_object_moved(before, after, source_color) or _color_object_moved(
            before,
            after,
            target_color,
        )
    if signature == ("object_count", "preserved"):
        return _total_component_count(before) == _total_component_count(after)
    if signature == ("adjacency", "preserved"):
        return _adjacent_exists(before, source_color, target_color) and _adjacent_exists(
            after,
            source_color,
            target_color,
        )
    if signature == ("adjacency", "modified"):
        return _adjacent_exists(before, source_color, target_color) != _adjacent_exists(
            after,
            source_color,
            target_color,
        )
    return False


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

    @property
    def shape_signature(self) -> Tuple[Tuple[int, int], ...]:
        min_y = min((point[0] for point in self.points), default=0)
        min_x = min((point[1] for point in self.points), default=0)
        return tuple(sorted((y - min_y, x - min_x) for y, x in self.points))


def _components(grid: np.ndarray, color: int | None = None) -> List[_Component]:
    if color is None:
        background = _infer_background(grid)
        mask = grid != background
    else:
        mask = grid == int(color)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[_Component] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            component_color = int(grid[y, x])
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
                        and mask[ny, nx]
                        and int(grid[ny, nx]) == component_color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            components.append(
                _Component(
                    color=component_color,
                    points=tuple(sorted(points)),
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


def _component_count(grid: np.ndarray, color: int) -> int:
    return len(_components(grid, color))


def _total_component_count(grid: np.ndarray) -> int:
    return len(_components(grid))


def _cell_count(grid: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(grid == int(color)))


def _pair_change_pixels(
    before: np.ndarray,
    after: np.ndarray,
    source_color: int,
    target_color: int,
) -> int:
    if before.shape != after.shape:
        return 0
    return int(
        np.count_nonzero(
            (before == int(source_color)) & (after == int(target_color))
        )
    )


def _color_object_moved(before: np.ndarray, after: np.ndarray, color: int) -> bool:
    before_components = _components(before, color)
    after_components = _components(after, color)
    used_after: set[int] = set()
    for source in before_components:
        candidates = [
            (index, target)
            for index, target in enumerate(after_components)
            if index not in used_after
            and target.size == source.size
            and target.shape_signature == source.shape_signature
        ]
        if not candidates:
            continue
        sy, sx = source.centroid
        best_index, best = min(
            candidates,
            key=lambda item: math.hypot(
                item[1].centroid[0] - sy,
                item[1].centroid[1] - sx,
            ),
        )
        used_after.add(best_index)
        by, bx = source.centroid
        ay, ax = best.centroid
        if round(math.hypot(ay - by, ax - bx), 4) > 0:
            return True
    return False


def _infer_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _average(values: Iterable[int]) -> float:
    items = [int(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M1.3 predicate pre-test.")
    parser.add_argument("trace_paths", nargs="*", type=Path)
    parser.add_argument("--traces-dir", type=Path, default=Path("human_traces"))
    parser.add_argument("--accepted-invariants", type=Path, default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_PRETEST_OUTPUT_PATH)
    parser.add_argument(
        "--min-pixel-support",
        type=int,
        default=DEFAULT_PRETEST_MIN_PIXEL_SUPPORT,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_PRETEST_TOP_K)
    parser.add_argument("--m1-anchor-expansion", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    trace_paths = args.trace_paths or list(
        select_blocked_trace_paths(traces_dir=args.traces_dir)
    )
    anchor_expander = None
    if args.m1_anchor_expansion:
        from .anchor_expansion import build_m1_anchor_expander

        anchor_expander = build_m1_anchor_expander()
    result = run_predicate_coverage_pretest(
        trace_paths,
        accepted_invariants_path=args.accepted_invariants,
        min_pixel_support=args.min_pixel_support,
        top_k=args.top_k,
        anchor_expander=anchor_expander,
    )
    write_pretest_result(result, args.out)
    summary = {
        "output_path": str(args.out),
        "trace_count": len(result.comparisons),
        "rule_count": len(result.rules),
        "average_unique_predicates_off": round(result.average_unique_predicates_off, 4),
        "average_unique_predicates_on": round(result.average_unique_predicates_on, 4),
        "average_candidate_pairs_off": round(result.average_candidate_pairs_off, 4),
        "average_candidate_pairs_on": round(result.average_candidate_pairs_on, 4),
        "average_relation_candidates_off": round(result.average_relation_candidates_off, 4),
        "average_relation_candidates_on": round(result.average_relation_candidates_on, 4),
        "anchor_expansion_enabled": bool(result.anchor_expansion_enabled),
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
