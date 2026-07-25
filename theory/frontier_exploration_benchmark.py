"""Paired SAGE.9v frontier-oriented exploration evaluation.

Both arms receive the same games, seeds, resets, action budgets, and legacy
proposals.  The only configured difference is SAGE.9v.  Classification is
performed after the episodes and is never fed back to either controller.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import game_splits

from .unified_cognition_ab_benchmark import (
    DEFAULT_HELD_OUT_GAMES,
    run_unified_cognition_ab_benchmark,
)


SCHEMA_VERSION = "sage.frontier_exploration.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage9v_frontier_exploration_benchmark.json"
)


def run_frontier_exploration_benchmark(
    *,
    game_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (0,),
    action_budget_per_reset: int = 80,
    resets: int = 4,
    environments_dir: str | Path | None = None,
    env_factory: Callable[[str], Any] | None = None,
    include_traces: bool = False,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    games = tuple(str(game) for game in (game_ids or DEFAULT_HELD_OUT_GAMES))
    common = {
        "game_ids": games,
        "seeds": tuple(int(seed) for seed in seeds),
        "action_budget_per_reset": int(action_budget_per_reset),
        "resets": int(resets),
        "environments_dir": environments_dir,
        "env_factory": env_factory,
        "include_traces": include_traces,
    }
    active = run_unified_cognition_ab_benchmark(
        **common,
        enable_frontier_oriented_exploration=True,
    )
    ablated = run_unified_cognition_ab_benchmark(
        **common,
        enable_frontier_oriented_exploration=False,
    )
    payload = summarize_frontier_exploration_protocol(active, ablated)
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def summarize_frontier_exploration_protocol(
    active: Mapping[str, Any],
    ablated: Mapping[str, Any],
) -> Dict[str, Any]:
    active_games = tuple(str(game) for game in active["held_out_games"])
    ablated_games = tuple(str(game) for game in ablated["held_out_games"])
    active_protocol = dict(active["paired_protocol"])
    ablated_protocol = dict(ablated["paired_protocol"])
    protocol_gate = bool(
        active_games == ablated_games
        and tuple(active["seeds"]) == tuple(ablated["seeds"])
        and active["action_budget_per_reset"]
        == ablated["action_budget_per_reset"]
        and active["resets_per_game_seed_arm"]
        == ablated["resets_per_game_seed_arm"]
        and active_protocol.get("protocol_gate_passed")
        and ablated_protocol.get("protocol_gate_passed")
        and active_protocol.get(
            "frontier_oriented_exploration_enabled_in_unified"
        )
        and not ablated_protocol.get(
            "frontier_oriented_exploration_enabled_in_unified"
        )
    )

    games = []
    for game_id in active_games:
        active_metrics = _aggregate_game(active, game_id)
        ablated_metrics = _aggregate_game(ablated, game_id)
        intervention_gate = bool(
            active_metrics["frontier_stagnation_detections"] > 0
            and active_metrics["frontier_experiments"] > 0
            and active_metrics["frontier_untested_actuator_actions"] > 0
        )
        information_gate = bool(
            intervention_gate
            and (
                active_metrics["frontier_novel_states"] > 0
                or active_metrics["frontier_novel_effects"] > 0
                or active_metrics["frontier_terminal_credits"] > 0
            )
        )
        frontier_access_gate = bool(
            intervention_gate
            and active_metrics["frontier_novel_states"] > 0
        )
        causal_progress = bool(
            active_metrics["max_level_reached"]
            > ablated_metrics["max_level_reached"]
            or active_metrics["levels_completed"]
            > ablated_metrics["levels_completed"]
            or active_metrics["wins"] > ablated_metrics["wins"]
        )
        games.append({
            "game_id": game_id,
            "active": active_metrics,
            "frontier_exploration_ablated": ablated_metrics,
            "intervention_gate_passed": intervention_gate,
            "information_gate_passed": information_gate,
            "frontier_access_gate_passed": frontier_access_gate,
            "causal_progress_advantage": causal_progress,
            "causal_frontier_progress_gate_passed": bool(
                frontier_access_gate and causal_progress
            ),
        })

    access_games = [
        report for report in games
        if report["frontier_access_gate_passed"]
    ]
    causal_games = [
        report for report in games
        if report["causal_frontier_progress_gate_passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "paired_sage9v_frontier_exploration",
        "held_out_games": list(active_games),
        "seeds": list(active["seeds"]),
        "action_budget_per_reset": active["action_budget_per_reset"],
        "resets_per_game_seed_arm": (
            active["resets_per_game_seed_arm"]
        ),
        "protocol": {
            "protocol_gate_passed": protocol_gate,
            "only_sage9v_differs_between_unified_arms": True,
            "game_specific_rules_used": False,
            "cross_game_online_memory_used": False,
            "classification_is_post_evaluation_only": True,
            "active_and_ablation_have_identical_exam_budget": True,
        },
        "frontier_access_games": len(access_games),
        "causal_frontier_progress_games": len(causal_games),
        "any_frontier_access_gate_passed": bool(access_games),
        "any_causal_frontier_progress_gate_passed": bool(causal_games),
        "games": games,
        "active_benchmark": dict(active),
        "frontier_exploration_ablation_benchmark": dict(ablated),
    }


def _aggregate_game(
    payload: Mapping[str, Any],
    game_id: str,
) -> Dict[str, int | float]:
    rows = [
        dict(pair["unified"])
        for pair in payload["pairs"]
        if str(pair["game_id"]) == str(game_id)
    ]
    summed = (
        "levels_completed_delta",
        "wins",
        "frontier_stagnation_detections",
        "frontier_entries",
        "frontier_experiments",
        "frontier_sequence_actions",
        "frontier_multi_step_sequences",
        "frontier_untested_state_actions",
        "frontier_untested_actuator_actions",
        "frontier_untested_object_actions",
        "frontier_productive_experiments",
        "frontier_novel_effects",
        "frontier_novel_states",
        "frontier_terminal_credits",
    )
    totals: Counter[str] = Counter()
    information_gain = 0.0
    controller_errors = 0
    for row in rows:
        for key in summed:
            totals[key] += int(row.get(key, 0) or 0)
        errors = row.get("controller_errors", []) or []
        controller_errors += (
            len(errors)
            if isinstance(errors, (list, tuple))
            else int(errors)
        )
        information_gain += float(
            row.get("frontier_information_gain", 0.0) or 0.0
        )
    return {
        "game_seed_runs": len(rows),
        "levels_completed": totals["levels_completed_delta"],
        "max_level_reached": max(
            (int(row.get("max_level_reached", 0) or 0) for row in rows),
            default=0,
        ),
        "wins": totals["wins"],
        "controller_errors": controller_errors,
        **{
            key: totals[key]
            for key in summed
            if key not in {"levels_completed_delta", "wins"}
        },
        "frontier_information_gain": round(information_gain, 4),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired SAGE.9v frontier exploration screen.",
    )
    parser.add_argument(
        "--games",
        default=",".join(
            game_splits.resolve("public_unseen_split", full_ids=False)
        ),
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--resets", type=int, default=4)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--include-traces", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    games = [
        game_splits.resolve_full_game_id(item.strip())
        for item in str(args.games).split(",")
        if item.strip()
    ]
    seeds = [
        int(item.strip())
        for item in str(args.seeds).split(",")
        if item.strip()
    ]
    payload = run_frontier_exploration_benchmark(
        game_ids=games,
        seeds=seeds,
        action_budget_per_reset=args.budget,
        resets=args.resets,
        environments_dir=args.environments_dir,
        include_traces=args.include_traces,
        write_path=args.out,
    )
    print(json.dumps({
        "frontier_access_games": payload["frontier_access_games"],
        "causal_frontier_progress_games": (
            payload["causal_frontier_progress_games"]
        ),
        "protocol_gate_passed": payload["protocol"][
            "protocol_gate_passed"
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["protocol"]["protocol_gate_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "run_frontier_exploration_benchmark",
    "summarize_frontier_exploration_protocol",
]
