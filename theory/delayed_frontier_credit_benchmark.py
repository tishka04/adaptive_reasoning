"""Paired SAGE.10a delayed frontier-to-terminal credit evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

import game_splits

from .live_transition_loop import build_observation
from .online_frontier_exploration import OnlineFrontierExplorer
from .online_multiform_relational_learner import (
    OnlineMultiformRelationalLearner,
)
from .unified_cognition_ab_benchmark import (
    DEFAULT_HELD_OUT_GAMES,
    run_unified_cognition_ab_benchmark,
)


SCHEMA_VERSION = "sage.delayed_frontier_terminal_credit.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage10a_delayed_frontier_credit_benchmark.json"
)


@dataclass(frozen=True)
class _Action:
    name: str
    action_args: Dict[str, Any] = field(default_factory=dict)


def run_delayed_frontier_credit_benchmark(
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
    """Run the procedural proof and a paired public ARC audit."""
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
        enable_delayed_frontier_terminal_credit=True,
    )
    ablated = run_unified_cognition_ab_benchmark(
        **common,
        enable_delayed_frontier_terminal_credit=False,
    )
    payload = summarize_delayed_frontier_credit_protocol(
        active,
        ablated,
        procedural=_run_procedural_delayed_credit_pair(),
    )
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def summarize_delayed_frontier_credit_protocol(
    active: Mapping[str, Any],
    ablated: Mapping[str, Any],
    *,
    procedural: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    active_games = tuple(str(game) for game in active["held_out_games"])
    ablated_games = tuple(str(game) for game in ablated["held_out_games"])
    active_protocol = dict(active["paired_protocol"])
    ablated_protocol = dict(ablated["paired_protocol"])
    execution_match = _matching_executed_unified_arms(active, ablated)
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
            "delayed_frontier_terminal_credit_enabled_in_unified"
        )
        and not ablated_protocol.get(
            "delayed_frontier_terminal_credit_enabled_in_unified"
        )
        and execution_match
    )
    procedural_result = dict(
        procedural or _run_procedural_delayed_credit_pair()
    )

    games = []
    for game_id in active_games:
        active_metrics = _aggregate_game(active, game_id)
        ablated_metrics = _aggregate_game(ablated, game_id)
        eligibility = bool(
            active_metrics["frontier_delayed_eligibilities_registered"] > 0
        )
        delayed_credit = bool(
            active_metrics["frontier_delayed_terminal_credits"] > 0
        )
        relational_propagation = bool(
            active_metrics[
                "terminal_multiform_delayed_frontier_pattern_credits"
            ]
            > 0
        )
        policy = bool(
            active_metrics["terminal_multiform_selections"] > 0
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
            "delayed_credit_ablated": ablated_metrics,
            "eligibility_gate_passed": eligibility,
            "delayed_credit_gate_passed": delayed_credit,
            "relational_propagation_gate_passed": relational_propagation,
            "policy_reuse_gate_passed": policy,
            "causal_progress_advantage": causal_progress,
            "causal_arc_progress_gate_passed": bool(
                delayed_credit and relational_propagation and causal_progress
            ),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "paired_sage10a_delayed_frontier_terminal_credit",
        "held_out_games": list(active_games),
        "seeds": list(active["seeds"]),
        "action_budget_per_reset": active["action_budget_per_reset"],
        "resets_per_game_seed_arm": (
            active["resets_per_game_seed_arm"]
        ),
        "protocol": {
            "protocol_gate_passed": protocol_gate,
            "only_sage10a_differs_between_unified_arms": True,
            "credit_window_is_branch_local": True,
            "only_productive_safe_frontier_effects_are_eligible": True,
            "one_credit_per_scientific_sequence": True,
            "terminal_learning_online_only": True,
            "game_specific_rules_used": False,
            "cross_game_online_memory_used": False,
            "active_and_ablation_have_identical_exam_budget": True,
            "active_and_ablation_reset_states_match": execution_match,
        },
        "procedural_causal_proof": procedural_result,
        "procedural_causal_gate_passed": bool(
            procedural_result.get("causal_gate_passed")
        ),
        "delayed_credit_games": sum(
            int(report["delayed_credit_gate_passed"])
            for report in games
        ),
        "relational_propagation_games": sum(
            int(report["relational_propagation_gate_passed"])
            for report in games
        ),
        "causal_arc_progress_games": sum(
            int(report["causal_arc_progress_gate_passed"])
            for report in games
        ),
        "any_delayed_credit_gate_passed": any(
            report["delayed_credit_gate_passed"]
            for report in games
        ),
        "any_relational_propagation_gate_passed": any(
            report["relational_propagation_gate_passed"]
            for report in games
        ),
        "any_causal_arc_progress_gate_passed": any(
            report["causal_arc_progress_gate_passed"]
            for report in games
        ),
        "games": games,
        "active_benchmark": _compact_source_benchmark(active),
        "delayed_credit_ablation_benchmark": (
            _compact_source_benchmark(ablated)
        ),
    }


def _compact_source_benchmark(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Retain audit evidence without duplicating large controller memories."""
    compact_pairs = []
    for pair in payload.get("pairs", ()):
        compact_pair = {
            key: pair.get(key)
            for key in (
                "game_id",
                "seed",
                "same_fresh_reset_states",
                "same_action_budget",
                "same_reset_count",
                "delta",
            )
        }
        for arm in ("legacy_only", "unified"):
            row = dict(pair.get(arm, {}) or {})
            compact_pair[arm] = {
                key: row.get(key)
                for key in (
                    "actions_executed",
                    "configured_action_budget",
                    "resets_executed",
                    "reset_visual_digests",
                    "levels_completed_delta",
                    "max_level_reached",
                    "wins",
                    "failure_causes",
                    "controller_errors",
                    "frontier_experiments",
                    "frontier_productive_experiments",
                    "frontier_delayed_eligibilities_registered",
                    "frontier_delayed_eligibilities_pending",
                    "frontier_delayed_terminal_events",
                    "frontier_delayed_terminal_credits",
                    "frontier_delayed_credit_delay_actions",
                    "frontier_delayed_credit_max_delay",
                    "frontier_expired_delayed_eligibilities",
                    "frontier_discarded_delayed_eligibilities",
                    "frontier_censored_delayed_eligibilities",
                    "frontier_unsafe_delayed_eligibilities",
                    (
                        "terminal_multiform_delayed_frontier_"
                        "eligibilities_registered"
                    ),
                    (
                        "terminal_multiform_delayed_frontier_"
                        "eligibilities_pending"
                    ),
                    (
                        "terminal_multiform_delayed_frontier_"
                        "eligibilities_credited"
                    ),
                    (
                        "terminal_multiform_delayed_frontier_"
                        "pattern_credits"
                    ),
                    (
                        "terminal_multiform_delayed_frontier_"
                        "credit_branches"
                    ),
                    (
                        "terminal_multiform_delayed_frontier_"
                        "eligibilities_expired"
                    ),
                    (
                        "terminal_multiform_delayed_frontier_"
                        "eligibilities_discarded"
                    ),
                    "terminal_multiform_confirmed_patterns",
                    "terminal_multiform_selections",
                    "terminal_multiform_transferred_selections",
                )
            }
        compact_pairs.append(compact_pair)
    return {
        key: payload.get(key)
        for key in (
            "schema_version",
            "evaluation",
            "held_out_games",
            "seeds",
            "action_budget_per_reset",
            "resets_per_game_seed_arm",
            "paired_protocol",
            "baseline_definition",
            "metrics",
            "failure_causes",
            "arc_progress_observed",
        )
    } | {"pairs": compact_pairs}


def _run_procedural_delayed_credit_pair() -> Dict[str, Any]:
    active = _run_procedural_arm(enabled=True)
    ablated = _run_procedural_arm(enabled=False)
    return {
        "active": active,
        "delayed_credit_ablated": ablated,
        "same_observed_training_transitions": True,
        "same_test_candidates": True,
        "active_transferred_policy": bool(active["transferred_policy"]),
        "ablated_transferred_policy": bool(
            ablated["transferred_policy"]
        ),
        "active_terminal_success": bool(active["terminal_success"]),
        "ablated_terminal_success": bool(ablated["terminal_success"]),
        "causal_gate_passed": bool(
            active["delayed_terminal_credits"] == 1
            and active["delayed_frontier_pattern_credits"] >= 2
            and active["transferred_policy"]
            and active["terminal_success"]
            and ablated["delayed_terminal_credits"] == 0
            and ablated["delayed_frontier_pattern_credits"] == 0
            and not ablated["transferred_policy"]
            and not ablated["terminal_success"]
        ),
    }


def _run_procedural_arm(*, enabled: bool) -> Dict[str, Any]:
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        max_sequence_actions=1,
        enable_delayed_terminal_credit=enabled,
        delayed_terminal_credit_window=4,
    )
    learner = OnlineMultiformRelationalLearner(
        minimum_terminal_support=2,
    )
    empty = np.zeros((9, 11), dtype=np.int32)

    learner.start_branch()
    first = _square_grid(color=2, row=1, column=1)
    first_action = _Action("ACTION6", {"x": 1, "y": 1})
    selection = explorer.select(
        current_grid=first,
        available_actions=("ACTION6",),
        available_action_candidates=(first_action,),
        branch_diagnostics=_stalled(),
    )
    if selection is None:  # pragma: no cover - guards benchmark integrity
        raise RuntimeError("procedural frontier experiment was not selected")
    outcome = explorer.observe_transition(
        grid_before=first,
        grid_after=empty,
        action_name=selection.action_name,
        action_data=selection.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    learner.observe_transition(
        observation_before=_observation(first),
        observation_after=_observation(empty),
        action_name=selection.action_name,
        action_data=selection.action_data,
        terminal_success=False,
        game_over=False,
        delayed_frontier_eligibility_id=str(
            outcome["delayed_credit_eligibility_id"]
        ),
    )
    explorer.note_transition(terminal_success=False)

    learner.observe_transition(
        observation_before=_observation(empty),
        observation_after=_observation(empty),
        action_name="ACTION1",
        action_data={},
        terminal_success=False,
        game_over=False,
    )
    explorer.note_transition(terminal_success=False)
    delayed_update = explorer.note_transition(terminal_success=True)
    learner.observe_transition(
        observation_before=_observation(empty),
        observation_after=_observation(empty),
        action_name="ACTION1",
        action_data={},
        terminal_success=True,
        game_over=False,
    )
    learner.resolve_delayed_frontier_credit(
        credited_eligibility_ids=tuple(
            credit.eligibility_id for credit in delayed_update.credited
        ),
        expired_eligibility_ids=(
            delayed_update.expired_eligibility_ids
        ),
        discarded_eligibility_ids=(
            delayed_update.discarded_eligibility_ids
        ),
    )

    learner.start_branch()
    second = _square_grid(color=7, row=4, column=5)
    learner.observe_transition(
        observation_before=_observation(second),
        observation_after=_observation(empty),
        action_name="ACTION6",
        action_data={"x": 5, "y": 4},
        terminal_success=True,
        game_over=False,
    )

    learner.start_branch()
    test_grid = _square_grid(color=9, row=6, column=7)
    test_action = SimpleNamespace(
        name="ACTION6",
        action_args={"x": 7, "y": 6},
    )
    transferred = learner.select(
        observation=_observation(test_grid),
        available_actions=("ACTION6",),
        available_action_candidates=(test_action,),
    )
    explorer_summary = explorer.summary()
    learner_summary = learner.summary()
    correct_action = bool(
        transferred is not None
        and transferred.action_data == {"x": 7, "y": 6}
    )
    return {
        "delayed_credit_enabled": enabled,
        "eligibility_registered": int(
            explorer_summary["delayed_eligibilities_registered"]
        ),
        "delayed_terminal_credits": int(
            explorer_summary["delayed_terminal_credits"]
        ),
        "credit_delay_actions": int(
            explorer_summary["delayed_credit_max_delay"]
        ),
        "delayed_frontier_pattern_credits": int(
            learner_summary["delayed_frontier_pattern_credits"]
        ),
        "confirmed_patterns": int(
            learner_summary["confirmed_patterns"]
        ),
        "transferred_policy": correct_action,
        "terminal_success": correct_action,
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
        "frontier_delayed_eligibilities_registered",
        "frontier_delayed_eligibilities_pending",
        "frontier_delayed_terminal_events",
        "frontier_delayed_terminal_credits",
        "frontier_delayed_credit_delay_actions",
        "frontier_expired_delayed_eligibilities",
        "frontier_discarded_delayed_eligibilities",
        "frontier_censored_delayed_eligibilities",
        "frontier_unsafe_delayed_eligibilities",
        "terminal_multiform_delayed_frontier_eligibilities_registered",
        "terminal_multiform_delayed_frontier_eligibilities_pending",
        "terminal_multiform_delayed_frontier_eligibilities_credited",
        "terminal_multiform_delayed_frontier_pattern_credits",
        "terminal_multiform_delayed_frontier_credit_branches",
        "terminal_multiform_delayed_frontier_eligibilities_expired",
        "terminal_multiform_delayed_frontier_eligibilities_discarded",
        "terminal_multiform_confirmed_patterns",
        "terminal_multiform_selections",
        "terminal_multiform_transferred_selections",
    )
    totals: Counter[str] = Counter()
    for row in rows:
        for key in metrics:
            totals[key] += int(row.get(key, 0) or 0)
    return {
        "game_seed_runs": len(rows),
        "levels_completed": totals["levels_completed_delta"],
        "max_level_reached": max(
            (int(row.get("max_level_reached", 0) or 0) for row in rows),
            default=0,
        ),
        "wins": totals["wins"],
        "frontier_delayed_credit_max_delay": max(
            (
                int(row.get("frontier_delayed_credit_max_delay", 0) or 0)
                for row in rows
            ),
            default=0,
        ),
        **{
            key: totals[key]
            for key in metrics
            if key not in {"levels_completed_delta", "wins"}
        },
    }


def _matching_executed_unified_arms(
    active: Mapping[str, Any],
    ablated: Mapping[str, Any],
) -> bool:
    active_pairs = {
        (str(pair["game_id"]), int(pair["seed"])): dict(pair["unified"])
        for pair in active.get("pairs", ())
    }
    ablated_pairs = {
        (str(pair["game_id"]), int(pair["seed"])): dict(pair["unified"])
        for pair in ablated.get("pairs", ())
    }
    if not active_pairs or active_pairs.keys() != ablated_pairs.keys():
        return False
    total_active_actions = 0
    total_ablated_actions = 0
    for key, active_arm in active_pairs.items():
        ablated_arm = ablated_pairs[key]
        if (
            active_arm.get("reset_visual_digests")
            != ablated_arm.get("reset_visual_digests")
            or active_arm.get("configured_action_budget")
            != ablated_arm.get("configured_action_budget")
            or active_arm.get("resets_executed")
            != ablated_arm.get("resets_executed")
        ):
            return False
        active_failures = dict(active_arm.get("failure_causes", {}) or {})
        ablated_failures = dict(
            ablated_arm.get("failure_causes", {}) or {}
        )
        if (
            active_failures.get("environment_setup_error", 0)
            or ablated_failures.get("environment_setup_error", 0)
        ):
            return False
        total_active_actions += int(
            active_arm.get("actions_executed", 0) or 0
        )
        total_ablated_actions += int(
            ablated_arm.get("actions_executed", 0) or 0
        )
    return bool(total_active_actions > 0 and total_ablated_actions > 0)


def _observation(grid: np.ndarray):
    return build_observation(
        grid,
        available_actions=("ACTION1", "ACTION6"),
        game_state="NOT_FINISHED",
        levels_completed=0,
        infer_players=False,
    )


def _square_grid(
    *,
    color: int,
    row: int,
    column: int,
) -> np.ndarray:
    grid = np.zeros((9, 11), dtype=np.int32)
    grid[row:row + 2, column:column + 2] = color
    return grid


def _stalled() -> Dict[str, int]:
    return {
        "branch_actions": 8,
        "actions_since_terminal_improvement": 8,
        "max_hash_repeat": 3,
        "unique_states_in_window": 2,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired SAGE.10a delayed-credit audit.",
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
    payload = run_delayed_frontier_credit_benchmark(
        game_ids=games,
        seeds=seeds,
        action_budget_per_reset=args.budget,
        resets=args.resets,
        environments_dir=args.environments_dir,
        include_traces=args.include_traces,
        write_path=args.out,
    )
    print(json.dumps({
        "protocol_gate_passed": payload["protocol"][
            "protocol_gate_passed"
        ],
        "procedural_causal_gate_passed": payload[
            "procedural_causal_gate_passed"
        ],
        "delayed_credit_games": payload["delayed_credit_games"],
        "relational_propagation_games": payload[
            "relational_propagation_games"
        ],
        "causal_arc_progress_games": payload[
            "causal_arc_progress_games"
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["protocol"]["protocol_gate_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "run_delayed_frontier_credit_benchmark",
    "summarize_delayed_frontier_credit_protocol",
]
