"""M1 grounding autopsy for the first positive A25 cases.

This diagnostic compares trace-derived anchors with what can actually be tested
from the live reset grid. It is analysis-only: no hypothesis is confirmed here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from theory.cross_game_correspondence_discovery import (
    DiscoveredCorrespondenceCandidate,
    discover_cross_game_correspondences,
)
from theory.non_ar25_multi_relation_agenda import (
    run_non_ar25_multi_relation_agenda,
)

from .anchor_expansion import build_m1_anchor_expander
from .invariant_miner import DEFAULT_ACCEPTED_PATH
from .live_anchor_ranking import (
    DEFAULT_BLOCKED_GAMES,
    M1_LIVE_PREFERRED_PREDICATES,
    _candidate_pair_key,
    _live_pair_consumability,
    _load_live_grid_and_actions,
    _source_colors_by_action,
    build_m1_live_candidate_ranker,
    consumability_metrics,
    rank_live_compatible_candidates,
    select_blocked_trace_paths,
)
from .predicate_generation import build_m1_predicate_generator

DEFAULT_GROUNDING_AUTOPSY_JSON_PATH = (
    Path("diagnostics") / "m1" / "grounding_autopsy.json"
)
DEFAULT_GROUNDING_AUTOPSY_MD_PATH = (
    Path("diagnostics") / "m1" / "grounding_autopsy.md"
)


@dataclass(frozen=True)
class PairGroundingDetail:
    """One candidate pair with live-grounding diagnostics."""

    action: str
    source_color: int
    target_color: int
    support: int
    transition_support: int
    predicates: Tuple[str, ...]
    preferred_predicates: Tuple[str, ...]
    live_preferred_predicates: Tuple[str, ...]
    live_source_colors_for_action: Tuple[int, ...]
    live_color_compatible: bool
    target_live_present: bool
    entering_agenda: bool
    ranked_top20: bool
    block_reason: str

    @property
    def pair_key(self) -> Tuple[str, int, int]:
        return (self.action, self.source_color, self.target_color)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source_color": int(self.source_color),
            "target_color": int(self.target_color),
            "support": int(self.support),
            "transition_support": int(self.transition_support),
            "predicates": list(self.predicates),
            "preferred_predicates": list(self.preferred_predicates),
            "live_preferred_predicates": list(self.live_preferred_predicates),
            "live_source_colors_for_action": list(self.live_source_colors_for_action),
            "live_color_compatible": bool(self.live_color_compatible),
            "target_live_present": bool(self.target_live_present),
            "entering_agenda": bool(self.entering_agenda),
            "ranked_top20": bool(self.ranked_top20),
            "block_reason": self.block_reason,
        }


@dataclass(frozen=True)
class GroundingFunnel:
    """Per-game hypothesis grounding funnel."""

    pairs_discovered: int
    pairs_target_present: int
    pairs_actionable_source: int
    pairs_blocked_by_unselectable_source: int
    pairs_live_compatible: int
    pairs_with_2_preferred_predicates: int
    pairs_entering_agenda: int
    pairs_generating_env_action: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs_discovered": int(self.pairs_discovered),
            "pairs_target_present": int(self.pairs_target_present),
            "pairs_actionable_source": int(self.pairs_actionable_source),
            "pairs_blocked_by_unselectable_source": int(
                self.pairs_blocked_by_unselectable_source
            ),
            "pairs_live_compatible": int(self.pairs_live_compatible),
            "pairs_with_2_preferred_predicates": int(
                self.pairs_with_2_preferred_predicates
            ),
            "pairs_entering_agenda": int(self.pairs_entering_agenda),
            "pairs_generating_env_action": int(self.pairs_generating_env_action),
        }


def run_grounding_autopsy(
    trace_paths: Sequence[str | Path] = (),
    *,
    accepted_invariants_path: str | Path = DEFAULT_ACCEPTED_PATH,
    environments_dir: str | Path | None = None,
    min_pixel_support: int = 1,
    max_candidates: int = 20,
    discovery_top_k: int = 100,
    run_agenda: bool = True,
) -> Dict[str, Any]:
    """Run a comparative grounding autopsy for blocked non-ar25 traces."""
    from theory.non_ar25_active_micro_run import _env_dir

    selected_paths = (
        tuple(Path(path) for path in trace_paths)
        if trace_paths
        else select_blocked_trace_paths(games=DEFAULT_BLOCKED_GAMES)
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    predicate_generator = build_m1_predicate_generator(
        accepted_invariants_path=accepted_invariants_path,
    )
    anchor_expander = build_m1_anchor_expander()
    ranker = build_m1_live_candidate_ranker()

    games: List[Dict[str, Any]] = []
    for trace_path in selected_paths:
        game = _autopsy_game(
            trace_path,
            env_dir=env_dir,
            predicate_generator=predicate_generator,
            anchor_expander=anchor_expander,
            ranker=ranker,
            min_pixel_support=min_pixel_support,
            max_candidates=max_candidates,
            discovery_top_k=discovery_top_k,
            run_agenda=run_agenda,
        )
        games.append(game)

    return {
        "config": {
            "accepted_invariants_path": str(accepted_invariants_path),
            "environments_dir": str(env_dir),
            "min_pixel_support": int(min_pixel_support),
            "max_candidates": int(max_candidates),
            "discovery_top_k": int(discovery_top_k),
            "run_agenda": bool(run_agenda),
            "preferred_predicates": list(M1_LIVE_PREFERRED_PREDICATES),
        },
        "summary_table": [_summary_row(game) for game in games],
        "games": games,
        "interpretation": _interpretation(games),
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def write_grounding_autopsy(
    payload: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_GROUNDING_AUTOPSY_JSON_PATH,
    md_path: str | Path = DEFAULT_GROUNDING_AUTOPSY_MD_PATH,
) -> None:
    json_output = Path(json_path)
    md_output = Path(md_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_output.write_text(render_grounding_autopsy_markdown(payload), encoding="utf-8")


def render_grounding_autopsy_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# M1.3c grounding autopsy",
        "",
        "This diagnostic asks why dc22 succeeds where bp35/cd82 do not.",
        "",
        "## Summary",
        "",
        "| game | new pairs | actionable-source new pairs | entering agenda | relation candidates | env actions | error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("summary_table", []):
        lines.append(
            "| {game_id} | {new_pairs} | {live_compatible} | {entering_agenda} | "
            "{relation_candidate_count} | {env_actions} | {error} |".format(
                game_id=row["game_id"],
                new_pairs=row["new_pairs"],
                live_compatible=row["live_compatible"],
                entering_agenda=row["entering_agenda"],
                relation_candidate_count=row["relation_candidate_count"],
                env_actions=row["env_actions"],
                error=row["error"] or "",
            )
        )

    lines.extend(
        [
            "",
            "## Grounding funnel",
            "",
            "| game | discovered | target present | actionable source | blocked source | live-compatible | entering agenda | env-action pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for game in payload.get("games", []):
        funnel = game.get("grounding_funnel", {})
        lines.append(
            "| {game_id} | {discovered} | {target_present} | {actionable} | "
            "{blocked_source} | {live_compatible} | {entering} | {env_pairs} |".format(
                game_id=game.get("game_id", ""),
                discovered=funnel.get("pairs_discovered", 0),
                target_present=funnel.get("pairs_target_present", 0),
                actionable=funnel.get("pairs_actionable_source", 0),
                blocked_source=funnel.get(
                    "pairs_blocked_by_unselectable_source",
                    0,
                ),
                live_compatible=funnel.get("pairs_live_compatible", 0),
                entering=funnel.get("pairs_entering_agenda", 0),
                env_pairs=funnel.get("pairs_generating_env_action", 0),
            )
        )

    lines.extend(
        [
            "",
            "## New-pair grounding funnel",
            "",
            "| game | discovered | target present | actionable source | blocked source | live-compatible | entering agenda | env-action pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for game in payload.get("games", []):
        funnel = game.get("new_pair_grounding_funnel", {})
        lines.append(
            "| {game_id} | {discovered} | {target_present} | {actionable} | "
            "{blocked_source} | {live_compatible} | {entering} | {env_pairs} |".format(
                game_id=game.get("game_id", ""),
                discovered=funnel.get("pairs_discovered", 0),
                target_present=funnel.get("pairs_target_present", 0),
                actionable=funnel.get("pairs_actionable_source", 0),
                blocked_source=funnel.get(
                    "pairs_blocked_by_unselectable_source",
                    0,
                ),
                live_compatible=funnel.get("pairs_live_compatible", 0),
                entering=funnel.get("pairs_entering_agenda", 0),
                env_pairs=funnel.get("pairs_generating_env_action", 0),
            )
        )

    lines.extend(["", "## dc22 live-compatible new pairs", ""])
    dc22 = next(
        (
            game
            for game in payload.get("games", [])
            if str(game.get("game_id", "")).startswith("dc22")
        ),
        None,
    )
    if dc22 is None:
        lines.append("No dc22 trace found.")
    else:
        details = dc22.get("ranked_live_compatible_new_pairs", [])
        if not details:
            lines.append("No live-compatible new dc22 pair found.")
        else:
            lines.extend(
                [
                    "| action | pair | support | predicates | live predicates | entering agenda |",
                    "|---|---|---:|---|---|---|",
                ]
            )
            for item in details:
                lines.append(
                    "| {action} | {source}->{target} | {support} | {predicates} | "
                    "{live_predicates} | {entering} |".format(
                        action=item["action"],
                        source=item["source_color"],
                        target=item["target_color"],
                        support=item["support"],
                        predicates=", ".join(item["preferred_predicates"]),
                        live_predicates=", ".join(item["live_preferred_predicates"]),
                        entering=str(item["entering_agenda"]).lower(),
                    )
                )

    lines.extend(["", "## Interpretation", ""])
    for sentence in payload.get("interpretation", []):
        lines.append(f"- {sentence}")
    lines.append("")
    return "\n".join(lines)


def grounding_block_reason(
    *,
    live_color_compatible: bool,
    target_live_present: bool,
    preferred_predicate_count: int,
) -> str:
    if not live_color_compatible:
        return "source_not_selectable_for_action"
    if not target_live_present:
        return "target_not_present_in_live_grid"
    if int(preferred_predicate_count) < 2:
        return "not_enough_preferred_predicates"
    return "agenda_eligible"


def _autopsy_game(
    trace_path: Path,
    *,
    env_dir: Path,
    predicate_generator: Any,
    anchor_expander: Any,
    ranker: Any,
    min_pixel_support: int,
    max_candidates: int,
    discovery_top_k: int,
    run_agenda: bool,
) -> Dict[str, Any]:
    game_id = _game_id_from_trace(trace_path)
    baseline = discover_cross_game_correspondences(
        trace_path,
        game_id=game_id,
        min_pixel_support=min_pixel_support,
        top_k=discovery_top_k,
    )
    m1 = discover_cross_game_correspondences(
        trace_path,
        game_id=game_id,
        min_pixel_support=min_pixel_support,
        top_k=discovery_top_k,
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
    )
    live_grid, valid_actions = _load_live_grid_and_actions(game_id, env_dir)
    ranked = tuple(
        ranker(
            m1.candidates,
            live_grid=live_grid,
            valid_actions=valid_actions,
            max_candidates=max_candidates,
            preferred_predicates=M1_LIVE_PREFERRED_PREDICATES,
        )
    )
    baseline_keys = {_candidate_pair_key(candidate) for candidate in baseline.candidates}
    ranked_keys = {_candidate_pair_key(candidate) for candidate in ranked}
    details = [
        _pair_detail(
            candidate,
            live_grid=live_grid,
            valid_actions=valid_actions,
            ranked_top20=_candidate_pair_key(candidate) in ranked_keys,
        )
        for candidate in ranked
        if _candidate_pair_key(candidate) not in baseline_keys
    ]
    ranked_all_details = [
        _pair_detail(
            candidate,
            live_grid=live_grid,
            valid_actions=valid_actions,
            ranked_top20=True,
        )
        for candidate in ranked
    ]
    all_new_details = [
        _pair_detail(
            candidate,
            live_grid=live_grid,
            valid_actions=valid_actions,
            ranked_top20=_candidate_pair_key(candidate) in ranked_keys,
        )
        for candidate in m1.candidates
        if _candidate_pair_key(candidate) not in baseline_keys
    ]
    agenda = (
        _run_m1_agenda(
            game_id,
            trace_path,
            predicate_generator=predicate_generator,
            anchor_expander=anchor_expander,
            ranker=ranker,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
            environments_dir=env_dir,
        )
        if run_agenda
        else {}
    )
    before_metrics = consumability_metrics(
        m1.candidates,
        baseline_candidates=baseline.candidates,
        live_grid=live_grid,
        valid_actions=valid_actions,
        preferred_predicates=M1_LIVE_PREFERRED_PREDICATES,
    )
    after_metrics = consumability_metrics(
        ranked,
        baseline_candidates=baseline.candidates,
        live_grid=live_grid,
        valid_actions=valid_actions,
        preferred_predicates=M1_LIVE_PREFERRED_PREDICATES,
    )
    source_colors = _source_colors_by_action(live_grid, valid_actions)
    return {
        "game_id": game_id,
        "trace_path": str(trace_path),
        "baseline_candidate_count": len(baseline.candidates),
        "m1_candidate_count": len(m1.candidates),
        "ranked_candidate_count": len(ranked),
        "live_source_colors_by_action": {
            action: sorted(colors) for action, colors in sorted(source_colors.items())
        },
        "grounding_funnel": _grounding_funnel(
            ranked_all_details,
            agenda=agenda,
        ).to_dict(),
        "new_pair_grounding_funnel": _grounding_funnel(
            details,
            agenda=agenda,
            only_new_pairs=True,
        ).to_dict(),
        "before_ranking_metrics": before_metrics.to_dict(),
        "after_ranking_metrics": after_metrics.to_dict(),
        "agenda": agenda,
        "ranked_pairs": [detail.to_dict() for detail in ranked_all_details],
        "ranked_new_pairs": [detail.to_dict() for detail in details],
        "ranked_live_compatible_new_pairs": [
            detail.to_dict() for detail in details if detail.live_color_compatible
        ],
        "ranked_entering_agenda_new_pairs": [
            detail.to_dict() for detail in details if detail.entering_agenda
        ],
        "all_new_pair_block_reasons": _block_reason_counts(all_new_details),
        "ranked_new_pair_block_reasons": _block_reason_counts(details),
    }


def _pair_detail(
    candidate: DiscoveredCorrespondenceCandidate,
    *,
    live_grid: Any,
    valid_actions: Sequence[Any],
    ranked_top20: bool,
) -> PairGroundingDetail:
    consumability = _live_pair_consumability(
        candidate,
        live_grid=live_grid,
        valid_actions=valid_actions,
        preferred_predicates=M1_LIVE_PREFERRED_PREDICATES,
    )
    source_colors = _source_colors_by_action(live_grid, valid_actions)
    block_reason = grounding_block_reason(
        live_color_compatible=consumability.live_color_compatible,
        target_live_present=consumability.target_live_present,
        preferred_predicate_count=len(consumability.preferred_predicates),
    )
    return PairGroundingDetail(
        action=str(candidate.action),
        source_color=int(candidate.source_color),
        target_color=int(candidate.target_color),
        support=int(candidate.support),
        transition_support=int(candidate.transition_support),
        predicates=tuple(predicate.name for predicate in candidate.predicates),
        preferred_predicates=tuple(consumability.preferred_predicates),
        live_preferred_predicates=tuple(consumability.live_preferred_predicates),
        live_source_colors_for_action=tuple(
            sorted(source_colors.get(str(candidate.action), set()))
        ),
        live_color_compatible=bool(consumability.live_color_compatible),
        target_live_present=bool(consumability.target_live_present),
        entering_agenda=bool(consumability.entering_agenda),
        ranked_top20=bool(ranked_top20),
        block_reason=block_reason,
    )


def _run_m1_agenda(
    game_id: str,
    trace_path: Path,
    *,
    predicate_generator: Any,
    anchor_expander: Any,
    ranker: Any,
    max_candidates: int,
    min_pixel_support: int,
    environments_dir: Path,
) -> Dict[str, Any]:
    result = run_non_ar25_multi_relation_agenda(
        game_id=game_id,
        trace_path=trace_path,
        environments_dir=environments_dir,
        max_candidates=max_candidates,
        min_pixel_support=min_pixel_support,
        preferred_predicates=M1_LIVE_PREFERRED_PREDICATES,
        predicate_generator=predicate_generator,
        anchor_expander=anchor_expander,
        candidate_ranker=ranker,
    )
    return {
        "error": result.error,
        "raw_discovered_candidates": int(result.raw_discovered_candidates),
        "discovered_candidates": int(result.discovered_candidates),
        "candidate_prediction_count": int(result.candidate_prediction_count),
        "relation_candidate_count": int(result.relation_candidate_count),
        "agenda_pairs": [list(item.pair_colors) for item in result.agenda_items],
        "agenda_predicates": [item.predicate for item in result.agenda_items],
        "env_actions": int(result.env_actions),
        "wrong_confirmations": int(result.wrong_confirmations),
    }


def _summary_row(game: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = game.get("after_ranking_metrics", {})
    agenda = game.get("agenda", {})
    funnel = game.get("grounding_funnel", {})
    return {
        "game_id": game.get("game_id", ""),
        "new_pairs": int(metrics.get("new_pairs_total", 0)),
        "live_compatible": int(metrics.get("new_pairs_live_color_compatible", 0)),
        "entering_agenda": int(metrics.get("new_pairs_entering_agenda", 0)),
        "pairs_discovered": int(funnel.get("pairs_discovered", 0)),
        "pairs_target_present": int(funnel.get("pairs_target_present", 0)),
        "pairs_actionable_source": int(funnel.get("pairs_actionable_source", 0)),
        "pairs_blocked_by_unselectable_source": int(
            funnel.get("pairs_blocked_by_unselectable_source", 0)
        ),
        "pairs_generating_env_action": int(
            funnel.get("pairs_generating_env_action", 0)
        ),
        "relation_candidate_count": int(agenda.get("relation_candidate_count", 0) or 0),
        "env_actions": int(agenda.get("env_actions", 0) or 0),
        "error": str(agenda.get("error", "")),
    }


def _block_reason_counts(details: Iterable[PairGroundingDetail]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for detail in details:
        counts[detail.block_reason] = counts.get(detail.block_reason, 0) + 1
    return dict(sorted(counts.items()))


def _grounding_funnel(
    details: Sequence[PairGroundingDetail],
    *,
    agenda: Mapping[str, Any],
    only_new_pairs: bool = False,
) -> GroundingFunnel:
    agenda_pairs = {
        (int(pair[0]), int(pair[1]))
        for pair in agenda.get("agenda_pairs", []) or []
        if len(pair) >= 2
    }
    env_action_pairs = agenda_pairs if int(agenda.get("env_actions", 0) or 0) > 0 else set()
    unique_pairs = {detail.pair_key for detail in details}
    return GroundingFunnel(
        pairs_discovered=len(unique_pairs),
        pairs_target_present=len(
            {detail.pair_key for detail in details if detail.target_live_present}
        ),
        pairs_actionable_source=len(
            {detail.pair_key for detail in details if detail.live_color_compatible}
        ),
        pairs_blocked_by_unselectable_source=len(
            {
                detail.pair_key
                for detail in details
                if detail.block_reason == "source_not_selectable_for_action"
            }
        ),
        pairs_live_compatible=len(
            {
                detail.pair_key
                for detail in details
                if detail.live_color_compatible and detail.target_live_present
            }
        ),
        pairs_with_2_preferred_predicates=len(
            {
                detail.pair_key
                for detail in details
                if len(detail.preferred_predicates) >= 2
            }
        ),
        pairs_entering_agenda=len(
            {detail.pair_key for detail in details if detail.entering_agenda}
        ),
        pairs_generating_env_action=len(
            {
                detail.pair_key
                for detail in details
                if (detail.source_color, detail.target_color) in env_action_pairs
            }
        )
        if only_new_pairs
        else len(env_action_pairs),
    )


def _interpretation(games: Sequence[Mapping[str, Any]]) -> List[str]:
    positive = [
        game
        for game in games
        if int(game.get("after_ranking_metrics", {}).get("new_pairs_entering_agenda", 0))
        > 0
    ]
    blocked = [
        game
        for game in games
        if int(
            game.get("after_ranking_metrics", {}).get(
                "new_pairs_live_color_compatible",
                0,
            )
        )
        == 0
    ]
    return [
        "dc22 is the positive control: at least one new M1 pair is live-compatible and enters A25.",
        "bp35/cd82 produce new pairs, but their new sources are not selectable from the reset grid.",
        "The bottleneck is now hypothesis grounding: trace anchor -> live object -> actionable source.",
        "This diagnostic remains analysis-only and does not count trace support as proof.",
        f"Positive games: {', '.join(game['game_id'] for game in positive) or 'none'}.",
        f"Blocked-by-source games: {', '.join(game['game_id'] for game in blocked) or 'none'}.",
    ]


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
    parser = argparse.ArgumentParser(description="Run M1 grounding autopsy.")
    parser.add_argument("trace_paths", nargs="*", type=Path)
    parser.add_argument("--accepted-invariants", type=Path, default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--discovery-top-k", type=int, default=100)
    parser.add_argument("--skip-agenda", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_GROUNDING_AUTOPSY_JSON_PATH)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_GROUNDING_AUTOPSY_MD_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_grounding_autopsy(
        _parse_paths([str(path) for path in args.trace_paths]),
        accepted_invariants_path=args.accepted_invariants,
        environments_dir=args.environments_dir,
        min_pixel_support=args.min_pixel_support,
        max_candidates=args.max_candidates,
        discovery_top_k=args.discovery_top_k,
        run_agenda=not args.skip_agenda,
    )
    write_grounding_autopsy(
        payload,
        json_path=args.json_out,
        md_path=args.md_out,
    )
    print(
        json.dumps(
            {
                "json_output_path": str(args.json_out),
                "md_output_path": str(args.md_out),
                "summary_table": payload["summary_table"],
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
