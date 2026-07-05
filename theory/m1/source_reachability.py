"""M1.3d source reachability diagnostics.

This module is deliberately diagnostic-only. It converts grounded candidate
pairs blocked by an unselectable source into typed preparation problems. It
does not search for a preparation plan and does not confirm hypotheses.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .grounding_autopsy import DEFAULT_GROUNDING_AUTOPSY_JSON_PATH

SOURCE_NOT_SELECTABLE_REASON = "source_not_selectable_for_action"
DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "source_reachability_problems.json"
)


@dataclass(frozen=True)
class SourceAlignmentProblem:
    """A pair whose desired source is not selectable in the live grid."""

    game_id: str
    trace_path: str
    action: str
    desired_source_color: int
    target_color: int
    available_live_sources: Tuple[int, ...]
    candidate_pair: Tuple[str, int, int]
    block_reason: str
    source_scope: str
    support: int = 0
    transition_support: int = 0
    target_live_present: bool = False
    predicates: Tuple[str, ...] = ()
    preferred_predicates: Tuple[str, ...] = ()
    live_preferred_predicates: Tuple[str, ...] = ()
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": self.trace_path,
            "action": self.action,
            "desired_source_color": int(self.desired_source_color),
            "target_color": int(self.target_color),
            "available_live_sources": list(self.available_live_sources),
            "candidate_pair": [
                self.candidate_pair[0],
                int(self.candidate_pair[1]),
                int(self.candidate_pair[2]),
            ],
            "block_reason": self.block_reason,
            "source_scope": self.source_scope,
            "support": int(self.support),
            "transition_support": int(self.transition_support),
            "target_live_present": bool(self.target_live_present),
            "predicates": list(self.predicates),
            "preferred_predicates": list(self.preferred_predicates),
            "live_preferred_predicates": list(self.live_preferred_predicates),
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


def source_alignment_problem_from_dict(
    payload: Mapping[str, Any],
) -> SourceAlignmentProblem:
    """Build a SourceAlignmentProblem from a JSON-ready dictionary."""
    pair = list(payload.get("candidate_pair", []))
    action = str(payload.get("action", pair[0] if pair else ""))
    desired_source = int(
        payload.get("desired_source_color", pair[1] if len(pair) > 1 else 0)
    )
    target = int(payload.get("target_color", pair[2] if len(pair) > 2 else 0))
    return SourceAlignmentProblem(
        game_id=str(payload.get("game_id", "")),
        trace_path=str(payload.get("trace_path", "")),
        action=action,
        desired_source_color=desired_source,
        target_color=target,
        available_live_sources=tuple(
            sorted(int(value) for value in payload.get("available_live_sources", []))
        ),
        candidate_pair=(action, desired_source, target),
        block_reason=str(payload.get("block_reason", SOURCE_NOT_SELECTABLE_REASON)),
        source_scope=str(payload.get("source_scope", "")),
        support=int(payload.get("support", 0) or 0),
        transition_support=int(payload.get("transition_support", 0) or 0),
        target_live_present=bool(payload.get("target_live_present", False)),
        predicates=tuple(str(value) for value in payload.get("predicates", []) or []),
        preferred_predicates=tuple(
            str(value) for value in payload.get("preferred_predicates", []) or []
        ),
        live_preferred_predicates=tuple(
            str(value)
            for value in payload.get("live_preferred_predicates", []) or []
        ),
        status=str(payload.get("status", "UNRESOLVED")),
        trace_support_counted_as_proof=bool(
            payload.get("trace_support_counted_as_proof", False)
        ),
        prior_counted_as_proof=bool(payload.get("prior_counted_as_proof", False)),
    )


def load_grounding_autopsy(path: str | Path = DEFAULT_GROUNDING_AUTOPSY_JSON_PATH) -> Dict[str, Any]:
    """Load a previously generated M1.3c+ grounding autopsy artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_source_alignment_problems(
    payload: Mapping[str, Any],
    *,
    source_scope: str = "ranked_new_pairs",
) -> Tuple[SourceAlignmentProblem, ...]:
    """Extract typed source alignment problems from an autopsy payload."""
    if source_scope not in {"ranked_new_pairs", "ranked_pairs"}:
        raise ValueError(
            "source_scope must be 'ranked_new_pairs' or 'ranked_pairs'"
        )
    problems: List[SourceAlignmentProblem] = []
    for game in payload.get("games", []) or []:
        game_id = str(game.get("game_id", ""))
        trace_path = str(game.get("trace_path", ""))
        for item in game.get(source_scope, []) or []:
            if item.get("block_reason") != SOURCE_NOT_SELECTABLE_REASON:
                continue
            problem = _problem_from_pair_detail(
                game_id=game_id,
                trace_path=trace_path,
                source_scope=source_scope,
                item=item,
            )
            problems.append(problem)
    return tuple(problems)


def summarize_source_alignment_problems(
    problems: Iterable[SourceAlignmentProblem],
) -> Dict[str, Any]:
    """Summarize typed source reachability blockers by game."""
    by_game: Dict[str, Dict[str, Any]] = {}
    for problem in problems:
        row = by_game.setdefault(
            problem.game_id,
            {
                "problem_count": 0,
                "actions": {},
                "desired_source_colors": set(),
                "target_colors": set(),
                "available_live_sources_by_action": {},
            },
        )
        row["problem_count"] += 1
        row["actions"][problem.action] = row["actions"].get(problem.action, 0) + 1
        row["desired_source_colors"].add(int(problem.desired_source_color))
        row["target_colors"].add(int(problem.target_color))
        row["available_live_sources_by_action"][problem.action] = sorted(
            set(row["available_live_sources_by_action"].get(problem.action, []))
            | set(problem.available_live_sources)
        )
    return {
        game_id: {
            "problem_count": int(row["problem_count"]),
            "actions": dict(sorted(row["actions"].items())),
            "desired_source_colors": sorted(row["desired_source_colors"]),
            "target_colors": sorted(row["target_colors"]),
            "available_live_sources_by_action": {
                action: list(colors)
                for action, colors in sorted(
                    row["available_live_sources_by_action"].items()
                )
            },
        }
        for game_id, row in sorted(by_game.items())
    }


def run_source_reachability_analysis(
    *,
    grounding_autopsy_path: str | Path = DEFAULT_GROUNDING_AUTOPSY_JSON_PATH,
) -> Dict[str, Any]:
    """Build the M1.3d-a source reachability diagnostic payload."""
    payload = load_grounding_autopsy(grounding_autopsy_path)
    ranked_pairs = extract_source_alignment_problems(
        payload,
        source_scope="ranked_pairs",
    )
    ranked_new_pairs = extract_source_alignment_problems(
        payload,
        source_scope="ranked_new_pairs",
    )
    return {
        "config": {
            "grounding_autopsy_path": str(grounding_autopsy_path),
            "block_reason": SOURCE_NOT_SELECTABLE_REASON,
        },
        "ranked_pairs_summary": summarize_source_alignment_problems(ranked_pairs),
        "ranked_new_pairs_summary": summarize_source_alignment_problems(
            ranked_new_pairs
        ),
        "ranked_pairs_problem_count": len(ranked_pairs),
        "ranked_new_pairs_problem_count": len(ranked_new_pairs),
        "ranked_pairs_problems": [problem.to_dict() for problem in ranked_pairs],
        "ranked_new_pairs_problems": [
            problem.to_dict() for problem in ranked_new_pairs
        ],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def write_source_reachability_analysis(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _problem_from_pair_detail(
    *,
    game_id: str,
    trace_path: str,
    source_scope: str,
    item: Mapping[str, Any],
) -> SourceAlignmentProblem:
    action = str(item.get("action", ""))
    desired_source = int(item.get("source_color", 0))
    target = int(item.get("target_color", 0))
    available_sources = tuple(
        sorted(int(color) for color in item.get("live_source_colors_for_action", []))
    )
    return SourceAlignmentProblem(
        game_id=game_id,
        trace_path=trace_path,
        action=action,
        desired_source_color=desired_source,
        target_color=target,
        available_live_sources=available_sources,
        candidate_pair=(action, desired_source, target),
        block_reason=str(item.get("block_reason", "")),
        source_scope=source_scope,
        support=int(item.get("support", 0) or 0),
        transition_support=int(item.get("transition_support", 0) or 0),
        target_live_present=bool(item.get("target_live_present", False)),
        predicates=tuple(str(value) for value in item.get("predicates", []) or []),
        preferred_predicates=tuple(
            str(value) for value in item.get("preferred_predicates", []) or []
        ),
        live_preferred_predicates=tuple(
            str(value)
            for value in item.get("live_preferred_predicates", []) or []
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract M1.3d source alignment problems from grounding autopsy.",
    )
    parser.add_argument(
        "--grounding-autopsy",
        type=Path,
        default=DEFAULT_GROUNDING_AUTOPSY_JSON_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_source_reachability_analysis(
        grounding_autopsy_path=args.grounding_autopsy,
    )
    write_source_reachability_analysis(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "ranked_pairs_summary": payload["ranked_pairs_summary"],
                "ranked_new_pairs_summary": payload["ranked_new_pairs_summary"],
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
