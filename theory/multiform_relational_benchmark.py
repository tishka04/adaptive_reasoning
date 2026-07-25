"""Paired SAGE.9w terminal multi-form relation evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import game_splits

from .unified_cognition_ab_benchmark import (
    DEFAULT_HELD_OUT_GAMES,
    MULTIFORM_RELATION_FAMILIES,
    run_unified_cognition_ab_benchmark,
)


SCHEMA_VERSION = "sage.multiform_relational_induction.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage9w_multiform_relational_benchmark.json"
)


def run_multiform_relational_benchmark(
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
        enable_terminal_multiform_relational_induction=True,
    )
    ablated = run_unified_cognition_ab_benchmark(
        **common,
        enable_terminal_multiform_relational_induction=False,
    )
    payload = summarize_multiform_relational_protocol(active, ablated)
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def summarize_multiform_relational_protocol(
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
            "terminal_multiform_relational_induction_enabled_in_unified"
        )
        and not ablated_protocol.get(
            "terminal_multiform_relational_induction_enabled_in_unified"
        )
    )

    games = []
    for game_id in active_games:
        active_metrics = _aggregate_game(active, game_id)
        ablated_metrics = _aggregate_game(ablated, game_id)
        acquisition = bool(
            active_metrics["terminal_multiform_terminal_examples"] >= 2
            and active_metrics[
                "terminal_multiform_confirmed_patterns"
            ]
            > 0
            and active_metrics[
                "terminal_multiform_confirmed_families"
            ]
            >= 2
        )
        policy = bool(
            acquisition
            and active_metrics["terminal_multiform_selections"] > 0
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
            "multiform_relational_induction_ablated": ablated_metrics,
            "multiform_acquisition_gate_passed": acquisition,
            "multiform_policy_gate_passed": policy,
            "causal_progress_advantage": causal_progress,
            "causal_multiform_progress_gate_passed": bool(
                policy and causal_progress
            ),
        })

    acquisition_games = [
        report for report in games
        if report["multiform_acquisition_gate_passed"]
    ]
    policy_games = [
        report for report in games
        if report["multiform_policy_gate_passed"]
    ]
    causal_games = [
        report for report in games
        if report["causal_multiform_progress_gate_passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "paired_sage9w_multiform_relational_induction",
        "held_out_games": list(active_games),
        "seeds": list(active["seeds"]),
        "action_budget_per_reset": active["action_budget_per_reset"],
        "resets_per_game_seed_arm": (
            active["resets_per_game_seed_arm"]
        ),
        "protocol": {
            "protocol_gate_passed": protocol_gate,
            "only_sage9w_differs_between_unified_arms": True,
            "terminal_learning_online_only": True,
            "game_specific_rules_used": False,
            "cross_game_online_memory_used": False,
            "classification_is_post_evaluation_only": True,
            "active_and_ablation_have_identical_exam_budget": True,
        },
        "multiform_acquisition_games": len(acquisition_games),
        "multiform_policy_games": len(policy_games),
        "causal_multiform_progress_games": len(causal_games),
        "any_multiform_acquisition_gate_passed": bool(acquisition_games),
        "any_multiform_policy_gate_passed": bool(policy_games),
        "any_causal_multiform_progress_gate_passed": bool(causal_games),
        "games": games,
        "active_benchmark": dict(active),
        "multiform_relational_ablation_benchmark": dict(ablated),
    }


def _aggregate_game(
    payload: Mapping[str, Any],
    game_id: str,
) -> Dict[str, int]:
    rows = [
        dict(pair["unified"])
        for pair in payload["pairs"]
        if str(pair["game_id"]) == str(game_id)
    ]
    metrics = (
        "levels_completed_delta",
        "wins",
        "terminal_multiform_observations",
        "terminal_multiform_terminal_examples",
        "terminal_multiform_patterns_observed",
        "terminal_multiform_pattern_hypotheses",
        "terminal_multiform_confirmed_patterns",
        "terminal_multiform_confirmed_families",
        "terminal_multiform_actuator_models",
        "terminal_multiform_terminal_pattern_credits",
        "terminal_multiform_selections",
        "terminal_multiform_transferred_selections",
        "terminal_multiform_unsafe_model_blocks",
        *tuple(
            f"terminal_multiform_{family}_observations"
            for family in MULTIFORM_RELATION_FAMILIES
        ),
        *tuple(
            f"terminal_multiform_{family}_selections"
            for family in MULTIFORM_RELATION_FAMILIES
        ),
    )
    totals: Counter[str] = Counter()
    controller_errors = 0
    for row in rows:
        for key in metrics:
            totals[key] += int(row.get(key, 0) or 0)
        errors = row.get("controller_errors", []) or []
        controller_errors += (
            len(errors)
            if isinstance(errors, (list, tuple))
            else int(errors)
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
            for key in metrics
            if key not in {"levels_completed_delta", "wins"}
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired SAGE.9w multi-form relation screen.",
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
    payload = run_multiform_relational_benchmark(
        game_ids=games,
        seeds=seeds,
        action_budget_per_reset=args.budget,
        resets=args.resets,
        environments_dir=args.environments_dir,
        include_traces=args.include_traces,
        write_path=args.out,
    )
    print(json.dumps({
        "multiform_acquisition_games": (
            payload["multiform_acquisition_games"]
        ),
        "multiform_policy_games": payload["multiform_policy_games"],
        "causal_multiform_progress_games": (
            payload["causal_multiform_progress_games"]
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
    "run_multiform_relational_benchmark",
    "summarize_multiform_relational_protocol",
]
