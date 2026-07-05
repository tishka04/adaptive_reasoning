"""M1.3c live-grid compatible anchor ranking.

M1.3c does not validate hypotheses. It ranks and annotates unresolved
source-target candidates so A25 sees pairs that are actually consumable in the
current live grid.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from theory.cross_game_correspondence_discovery import (
    DiscoveredCorrespondenceCandidate,
    SourceTargetPredicate,
    discover_cross_game_correspondences,
)
from theory.non_ar25_active_micro_run import (
    _configure_offline_env,
    _env_dir,
    _valid_actions,
)
from theory.non_ar25_multi_family_experiment import _relation_holds_for_colors
from theory.real_env_option_adapter import snapshot_frame

from .anchor_expansion import build_m1_anchor_expander
from .invariant_miner import DEFAULT_ACCEPTED_PATH
from .predicate_generation import build_m1_predicate_generator

M1_LIVE_PREFERRED_PREDICATES = (
    "same_shape",
    "aligned_with",
    "adjacent_to",
    "paired_with",
)
DEFAULT_LIVE_COMPATIBILITY_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "live_anchor_compatibility_pretest.json"
)
DEFAULT_BLOCKED_GAMES = ("bp35", "cd82", "dc22")

CandidateRanker = Callable[..., Sequence[DiscoveredCorrespondenceCandidate]]


@dataclass(frozen=True)
class LivePairConsumability:
    """Consumability features for one action/source/target pair."""

    action: str
    source_color: int
    target_color: int
    live_color_compatible: bool
    target_live_present: bool
    preferred_predicates: Tuple[str, ...]
    live_preferred_predicates: Tuple[str, ...]
    entering_agenda: bool

    @property
    def pair_key(self) -> Tuple[str, int, int]:
        return (self.action, self.source_color, self.target_color)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source_color": self.source_color,
            "target_color": self.target_color,
            "live_color_compatible": self.live_color_compatible,
            "target_live_present": self.target_live_present,
            "preferred_predicates": list(self.preferred_predicates),
            "live_preferred_predicates": list(self.live_preferred_predicates),
            "entering_agenda": self.entering_agenda,
        }


@dataclass(frozen=True)
class LiveAnchorCompatibilityMetrics:
    """M1.3c KPI block for a candidate set."""

    pair_count: int
    agenda_eligible_pairs: int
    new_pairs_total: int
    new_pairs_live_color_compatible: int
    new_pairs_blocked_by_unselectable_source: int
    new_pairs_target_present: int
    new_pairs_with_2_preferred_predicates: int
    new_pairs_entering_agenda: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_count": int(self.pair_count),
            "agenda_eligible_pairs": int(self.agenda_eligible_pairs),
            "new_pairs_total": int(self.new_pairs_total),
            "new_pairs_live_color_compatible": int(
                self.new_pairs_live_color_compatible
            ),
            "new_pairs_blocked_by_unselectable_source": int(
                self.new_pairs_blocked_by_unselectable_source
            ),
            "new_pairs_target_present": int(self.new_pairs_target_present),
            "new_pairs_with_2_preferred_predicates": int(
                self.new_pairs_with_2_preferred_predicates
            ),
            "new_pairs_entering_agenda": int(self.new_pairs_entering_agenda),
        }


@dataclass(frozen=True)
class LiveAnchorCompatibilityTrace:
    """Live-grid consumability pre-test result for one trace."""

    game_id: str
    trace_path: str
    baseline_candidate_count: int
    m1_candidate_count: int
    ranked_candidate_count: int
    before_ranking: LiveAnchorCompatibilityMetrics
    after_ranking: LiveAnchorCompatibilityMetrics
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": self.trace_path,
            "baseline_candidate_count": int(self.baseline_candidate_count),
            "m1_candidate_count": int(self.m1_candidate_count),
            "ranked_candidate_count": int(self.ranked_candidate_count),
            "before_ranking": self.before_ranking.to_dict(),
            "after_ranking": self.after_ranking.to_dict(),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


@dataclass(frozen=True)
class LiveAnchorCompatibilityPretest:
    """Aggregate M1.3c live-grid compatibility diagnostic."""

    traces: Tuple[LiveAnchorCompatibilityTrace, ...]
    preferred_predicates: Tuple[str, ...] = M1_LIVE_PREFERRED_PREDICATES
    max_candidates: int = 20
    discovery_top_k: int = 100
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    @property
    def trace_count(self) -> int:
        return len(self.traces)

    @property
    def averages_before_ranking(self) -> Dict[str, float]:
        return _average_metrics(trace.before_ranking for trace in self.traces)

    @property
    def averages_after_ranking(self) -> Dict[str, float]:
        return _average_metrics(trace.after_ranking for trace in self.traces)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_count": self.trace_count,
            "preferred_predicates": list(self.preferred_predicates),
            "max_candidates": int(self.max_candidates),
            "discovery_top_k": int(self.discovery_top_k),
            "averages_before_ranking": {
                key: round(value, 4)
                for key, value in self.averages_before_ranking.items()
            },
            "averages_after_ranking": {
                key: round(value, 4)
                for key, value in self.averages_after_ranking.items()
            },
            "traces": [trace.to_dict() for trace in self.traces],
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def build_m1_live_candidate_ranker(
    *,
    preferred_predicates: Sequence[str] = M1_LIVE_PREFERRED_PREDICATES,
    augment_live_predicates: bool = True,
) -> CandidateRanker:
    """Build the deterministic M1.3c live-grid candidate ranker."""

    selected_preferred = tuple(str(predicate) for predicate in preferred_predicates)

    def _ranker(
        candidates: Sequence[DiscoveredCorrespondenceCandidate],
        *,
        live_grid: np.ndarray,
        valid_actions: Sequence[Any],
        max_candidates: int,
        preferred_predicates: Sequence[str] | None = None,
    ) -> Tuple[DiscoveredCorrespondenceCandidate, ...]:
        predicates = (
            tuple(str(item) for item in preferred_predicates)
            if preferred_predicates is not None
            else selected_preferred
        )
        return rank_live_compatible_candidates(
            candidates,
            live_grid=live_grid,
            valid_actions=valid_actions,
            max_candidates=max_candidates,
            preferred_predicates=predicates,
            augment_live_predicates=augment_live_predicates,
        )

    return _ranker


def rank_live_compatible_candidates(
    candidates: Sequence[DiscoveredCorrespondenceCandidate],
    *,
    live_grid: np.ndarray,
    valid_actions: Sequence[Any],
    max_candidates: int,
    preferred_predicates: Sequence[str] = M1_LIVE_PREFERRED_PREDICATES,
    augment_live_predicates: bool = True,
) -> Tuple[DiscoveredCorrespondenceCandidate, ...]:
    """Rank candidates toward pairs A25 can consume in the current grid."""
    grid = np.asarray(live_grid, dtype=np.int32)
    predicates = tuple(str(predicate) for predicate in preferred_predicates)
    prepared = [
        (
            _live_pair_consumability(
                candidate,
                live_grid=grid,
                valid_actions=valid_actions,
                preferred_predicates=predicates,
            ),
            _augment_candidate_live_predicates(
                candidate,
                live_grid=grid,
                preferred_predicates=predicates,
            )
            if augment_live_predicates
            else candidate,
            index,
        )
        for index, candidate in enumerate(candidates)
    ]
    prepared = [
        (
            _live_pair_consumability(
                candidate,
                live_grid=grid,
                valid_actions=valid_actions,
                preferred_predicates=predicates,
            ),
            candidate,
            index,
        )
        for _, candidate, index in prepared
    ]
    ordered = sorted(
        prepared,
        key=lambda item: _candidate_sort_key(item[0], item[1], item[2]),
    )
    return tuple(candidate for _, candidate, _ in ordered[: max(1, int(max_candidates))])


def consumability_metrics(
    candidates: Sequence[DiscoveredCorrespondenceCandidate],
    *,
    baseline_candidates: Sequence[DiscoveredCorrespondenceCandidate] = (),
    live_grid: np.ndarray,
    valid_actions: Sequence[Any],
    preferred_predicates: Sequence[str] = M1_LIVE_PREFERRED_PREDICATES,
) -> LiveAnchorCompatibilityMetrics:
    """Compute M1.3c KPIs for candidates against a live grid."""
    baseline_keys = {_candidate_pair_key(candidate) for candidate in baseline_candidates}
    consumability = [
        _live_pair_consumability(
            candidate,
            live_grid=live_grid,
            valid_actions=valid_actions,
            preferred_predicates=preferred_predicates,
        )
        for candidate in candidates
    ]
    seen_keys = {item.pair_key for item in consumability}
    new_items = [item for item in consumability if item.pair_key not in baseline_keys]
    return LiveAnchorCompatibilityMetrics(
        pair_count=len(seen_keys),
        agenda_eligible_pairs=len({item.pair_key for item in consumability if item.entering_agenda}),
        new_pairs_total=len({item.pair_key for item in new_items}),
        new_pairs_live_color_compatible=len(
            {item.pair_key for item in new_items if item.live_color_compatible}
        ),
        new_pairs_blocked_by_unselectable_source=len(
            {item.pair_key for item in new_items if not item.live_color_compatible}
        ),
        new_pairs_target_present=len(
            {item.pair_key for item in new_items if item.target_live_present}
        ),
        new_pairs_with_2_preferred_predicates=len(
            {
                item.pair_key
                for item in new_items
                if len(item.preferred_predicates) >= 2
            }
        ),
        new_pairs_entering_agenda=len(
            {item.pair_key for item in new_items if item.entering_agenda}
        ),
    )


def run_live_anchor_compatibility_pretest(
    trace_paths: Sequence[str | Path],
    *,
    accepted_invariants_path: str | Path = DEFAULT_ACCEPTED_PATH,
    environments_dir: str | Path | None = None,
    min_pixel_support: int = 1,
    max_candidates: int = 20,
    discovery_top_k: int = 100,
    preferred_predicates: Sequence[str] = M1_LIVE_PREFERRED_PREDICATES,
) -> LiveAnchorCompatibilityPretest:
    """Measure M1.3c live-grid consumability without taking env actions."""
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    predicate_generator = build_m1_predicate_generator(
        accepted_invariants_path=accepted_invariants_path,
    )
    anchor_expander = build_m1_anchor_expander()
    ranker = build_m1_live_candidate_ranker(
        preferred_predicates=preferred_predicates,
    )
    traces: List[LiveAnchorCompatibilityTrace] = []
    for trace_path in trace_paths:
        path = Path(trace_path)
        game_id = _game_id_from_trace(path)
        baseline = discover_cross_game_correspondences(
            path,
            game_id=game_id,
            min_pixel_support=min_pixel_support,
            top_k=discovery_top_k,
        )
        m1 = discover_cross_game_correspondences(
            path,
            game_id=game_id,
            min_pixel_support=min_pixel_support,
            top_k=discovery_top_k,
            predicate_generator=predicate_generator,
            anchor_expander=anchor_expander,
        )
        live_grid, valid_actions = _load_live_grid_and_actions(game_id, env_dir)
        ranked = ranker(
            m1.candidates,
            live_grid=live_grid,
            valid_actions=valid_actions,
            max_candidates=max_candidates,
            preferred_predicates=preferred_predicates,
        )
        traces.append(
            LiveAnchorCompatibilityTrace(
                game_id=game_id,
                trace_path=str(path),
                baseline_candidate_count=len(baseline.candidates),
                m1_candidate_count=len(m1.candidates),
                ranked_candidate_count=len(ranked),
                before_ranking=consumability_metrics(
                    m1.candidates,
                    baseline_candidates=baseline.candidates,
                    live_grid=live_grid,
                    valid_actions=valid_actions,
                    preferred_predicates=preferred_predicates,
                ),
                after_ranking=consumability_metrics(
                    ranked,
                    baseline_candidates=baseline.candidates,
                    live_grid=live_grid,
                    valid_actions=valid_actions,
                    preferred_predicates=preferred_predicates,
                ),
            )
        )
    return LiveAnchorCompatibilityPretest(
        traces=tuple(traces),
        preferred_predicates=tuple(str(item) for item in preferred_predicates),
        max_candidates=max_candidates,
        discovery_top_k=discovery_top_k,
    )


def write_live_anchor_compatibility_pretest(
    result: LiveAnchorCompatibilityPretest,
    output_path: str | Path = DEFAULT_LIVE_COMPATIBILITY_OUTPUT_PATH,
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


def _augment_candidate_live_predicates(
    candidate: DiscoveredCorrespondenceCandidate,
    *,
    live_grid: np.ndarray,
    preferred_predicates: Sequence[str],
) -> DiscoveredCorrespondenceCandidate:
    existing = {predicate.name: predicate for predicate in candidate.predicates}
    additions: List[SourceTargetPredicate] = []
    for predicate in preferred_predicates:
        if predicate in existing:
            continue
        if _relation_holds_for_colors(
            live_grid,
            str(predicate),
            int(candidate.source_color),
            int(candidate.target_color),
        ):
            additions.append(SourceTargetPredicate(name=str(predicate), support=1))
    if not additions:
        return candidate
    return DiscoveredCorrespondenceCandidate(
        game_id=candidate.game_id,
        action=candidate.action,
        source_color=candidate.source_color,
        target_color=candidate.target_color,
        relation=candidate.relation,
        support=candidate.support,
        transition_support=candidate.transition_support,
        predicates=tuple(candidate.predicates) + tuple(additions),
        weak_ready_candidates=candidate.weak_ready_candidates,
        strong_ready_candidates=candidate.strong_ready_candidates,
    )


def _live_pair_consumability(
    candidate: DiscoveredCorrespondenceCandidate,
    *,
    live_grid: np.ndarray,
    valid_actions: Sequence[Any],
    preferred_predicates: Sequence[str],
) -> LivePairConsumability:
    source_colors_by_action = _source_colors_by_action(live_grid, valid_actions)
    live_color_compatible = int(candidate.source_color) in source_colors_by_action.get(
        str(candidate.action),
        set(),
    )
    target_live_present = bool(np.any(live_grid == int(candidate.target_color)))
    candidate_predicates = {predicate.name for predicate in candidate.predicates}
    preferred = tuple(
        predicate
        for predicate in preferred_predicates
        if str(predicate) in candidate_predicates
    )
    live_preferred = tuple(
        predicate
        for predicate in preferred_predicates
        if _relation_holds_for_colors(
            live_grid,
            str(predicate),
            int(candidate.source_color),
            int(candidate.target_color),
        )
    )
    entering_agenda = bool(
        live_color_compatible
        and target_live_present
        and len(preferred) >= 2
    )
    return LivePairConsumability(
        action=str(candidate.action),
        source_color=int(candidate.source_color),
        target_color=int(candidate.target_color),
        live_color_compatible=live_color_compatible,
        target_live_present=target_live_present,
        preferred_predicates=preferred,
        live_preferred_predicates=live_preferred,
        entering_agenda=entering_agenda,
    )


def _candidate_sort_key(
    consumability: LivePairConsumability,
    candidate: DiscoveredCorrespondenceCandidate,
    index: int,
) -> Tuple[Any, ...]:
    return (
        -int(consumability.entering_agenda),
        -int(consumability.live_color_compatible),
        -int(consumability.target_live_present),
        -len(consumability.preferred_predicates),
        -len(consumability.live_preferred_predicates),
        -int(candidate.transition_support),
        -int(candidate.support),
        -len(candidate.predicates),
        index,
    )


def _source_colors_by_action(
    grid: np.ndarray,
    valid_actions: Sequence[Any],
) -> Dict[str, set[int]]:
    result: Dict[str, set[int]] = {}
    for action in valid_actions:
        name = str(getattr(action, "name", ""))
        color = _action_pixel_color(grid, getattr(action, "action_args", {}))
        if name and color is not None:
            result.setdefault(name, set()).add(int(color))
    return result


def _action_pixel_color(
    grid: np.ndarray,
    action_args: Mapping[str, Any],
) -> int | None:
    if "x" not in action_args or "y" not in action_args:
        return None
    try:
        x = int(action_args["x"])
        y = int(action_args["y"])
    except (TypeError, ValueError):
        return None
    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
        return None
    return int(grid[y, x])


def _candidate_pair_key(
    candidate: DiscoveredCorrespondenceCandidate,
) -> Tuple[str, int, int]:
    return (
        str(candidate.action),
        int(candidate.source_color),
        int(candidate.target_color),
    )


def _average_metrics(
    metrics: Iterable[LiveAnchorCompatibilityMetrics],
) -> Dict[str, float]:
    items = list(metrics)
    keys = (
        "pair_count",
        "agenda_eligible_pairs",
        "new_pairs_total",
        "new_pairs_live_color_compatible",
        "new_pairs_blocked_by_unselectable_source",
        "new_pairs_target_present",
        "new_pairs_with_2_preferred_predicates",
        "new_pairs_entering_agenda",
    )
    result: Dict[str, float] = {}
    for key in keys:
        values = [float(getattr(item, key)) for item in items]
        result[key] = sum(values) / len(values) if values else 0.0
    return result


def _load_live_grid_and_actions(
    game_id: str,
    environments_dir: Path,
) -> Tuple[np.ndarray, Sequence[Any]]:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    env = arc.make(game_id)
    frame = env.step(GameAction.RESET)
    snapshot = snapshot_frame(frame)
    return np.asarray(snapshot.grid, dtype=np.int32), _valid_actions(env)


def _game_id_from_trace(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            game_id = str(item.get("game_id", "")).strip()
            if game_id:
                return game_id
            break
    return path.name.split(".")[0]


def _parse_paths(values: Sequence[str]) -> List[Path]:
    return [Path(value) for value in values if value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M1.3c live anchor pre-test.")
    parser.add_argument("trace_paths", nargs="*", type=Path)
    parser.add_argument("--traces-dir", type=Path, default=Path("human_traces"))
    parser.add_argument("--accepted-invariants", type=Path, default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--discovery-top-k", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_LIVE_COMPATIBILITY_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    trace_paths = args.trace_paths or list(
        select_blocked_trace_paths(traces_dir=args.traces_dir)
    )
    result = run_live_anchor_compatibility_pretest(
        trace_paths,
        accepted_invariants_path=args.accepted_invariants,
        environments_dir=args.environments_dir,
        min_pixel_support=args.min_pixel_support,
        max_candidates=args.max_candidates,
        discovery_top_k=args.discovery_top_k,
    )
    write_live_anchor_compatibility_pretest(result, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "trace_count": result.trace_count,
                "averages_before_ranking": result.to_dict()["averages_before_ranking"],
                "averages_after_ranking": result.to_dict()["averages_after_ranking"],
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
