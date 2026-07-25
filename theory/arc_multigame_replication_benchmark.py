"""SAGE.9z multi-game, multi-seed, multi-budget ARC replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import game_splits

from .natural_structural_break_benchmark import (
    run_natural_structural_break_benchmark,
)
from .unified_cognition_ab_benchmark import DEFAULT_HELD_OUT_GAMES


SCHEMA_VERSION = "sage.arc_multigame_replication.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage9z_arc_multigame_replication_benchmark.json"
)


def run_arc_multigame_replication_benchmark(
    *,
    game_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (0, 1),
    action_budgets: Sequence[int] = (80, 160),
    resets: int = 4,
    environments_dir: str | Path | None = None,
    env_factory: Callable[[str], Any] | None = None,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    games = tuple(str(game) for game in (game_ids or DEFAULT_HELD_OUT_GAMES))
    seed_values = tuple(int(seed) for seed in seeds)
    budgets = tuple(sorted({
        max(1, int(budget)) for budget in action_budgets
    }))
    rows = []
    condition_protocols = []
    for budget in budgets:
        condition = run_natural_structural_break_benchmark(
            game_ids=games,
            seeds=seed_values,
            action_budget_per_reset=budget,
            resets=int(resets),
            environments_dir=environments_dir,
            env_factory=env_factory,
            include_traces=False,
        )
        rows.extend(_condition_rows(condition, budget=budget))
        condition_protocols.append({
            "action_budget": budget,
            "protocol_gate_passed": bool(
                condition["protocol"]["protocol_gate_passed"]
            ),
            "held_out_games": list(condition["held_out_games"]),
            "seeds": list(condition["seeds"]),
            "resets_per_game_seed_arm": int(
                condition["resets_per_game_seed_arm"]
            ),
            "relation_permutation_used": bool(
                condition["protocol"]["relation_permutation_used"]
            ),
        })
    payload = summarize_arc_multigame_replication_rows(
        rows,
        expected_games=games,
        expected_seeds=seed_values,
        expected_budgets=budgets,
        resets=int(resets),
        condition_protocols=condition_protocols,
    )
    if write_path is not None:
        target = Path(write_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def summarize_arc_multigame_replication_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_games: Sequence[str],
    expected_seeds: Sequence[int],
    expected_budgets: Sequence[int],
    resets: int,
    condition_protocols: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    reports = [dict(row) for row in rows]
    games = tuple(str(game) for game in expected_games)
    seeds = tuple(int(seed) for seed in expected_seeds)
    budgets = tuple(sorted(int(budget) for budget in expected_budgets))
    expected_conditions = {
        (game, seed, budget)
        for game in games
        for seed in seeds
        for budget in budgets
    }
    observed_conditions = {
        (
            str(row["game_id"]),
            int(row["seed"]),
            int(row["action_budget"]),
        )
        for row in reports
    }
    protocol_gate = bool(
        len(games) >= 2
        and len(seeds) >= 2
        and len(budgets) >= 2
        and expected_conditions == observed_conditions
        and all(
            bool(protocol.get("protocol_gate_passed"))
            and not bool(protocol.get("relation_permutation_used"))
            for protocol in condition_protocols
        )
        and all(
            int(row.get("active_controller_errors", 0)) == 0
            and int(row.get("ablated_controller_errors", 0)) == 0
            for row in reports
        )
    )

    game_reports = []
    for game_id in games:
        game_rows = [
            row for row in reports
            if str(row["game_id"]) == game_id
        ]
        causal_rows = [
            row for row in game_rows
            if row.get("causal_natural_revision_gate_passed")
        ]
        replicated_seeds = sorted({
            int(row["seed"]) for row in causal_rows
        })
        replicated_budgets = sorted({
            int(row["action_budget"]) for row in causal_rows
        })
        replication_gate = bool(
            len(replicated_seeds) >= 2
            and len(replicated_budgets) >= 2
        )
        game_reports.append({
            "game_id": game_id,
            "conditions": len(game_rows),
            "natural_break_conditions": sum(
                bool(row.get("natural_break_candidate"))
                for row in game_rows
            ),
            "terminal_revision_conditions": sum(
                bool(row.get("terminal_revision_observed"))
                for row in game_rows
            ),
            "causal_revision_conditions": len(causal_rows),
            "replicated_seeds": replicated_seeds,
            "replicated_budgets": replicated_budgets,
            "replicated_natural_revision_gate_passed": (
                replication_gate
            ),
            "max_active_level": max(
                (
                    int(row.get("active_max_level_reached", 0))
                    for row in game_rows
                ),
                default=0,
            ),
            "max_ablated_level": max(
                (
                    int(row.get("ablated_max_level_reached", 0))
                    for row in game_rows
                ),
                default=0,
            ),
        })
    replicated_games = [
        report for report in game_reports
        if report["replicated_natural_revision_gate_passed"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": (
            "post_hoc_arc_multigame_multiseed_multibudget_replication"
        ),
        "held_out_games": list(games),
        "seeds": list(seeds),
        "action_budgets": list(budgets),
        "resets_per_game_seed_arm": int(resets),
        "protocol": {
            "protocol_gate_passed": protocol_gate,
            "multiple_games_required": True,
            "multiple_seeds_required": True,
            "multiple_budgets_required": True,
            "natural_relation_only": True,
            "game_specific_rules_used": False,
            "cross_game_online_memory_used": False,
            "classification_is_post_evaluation_only": True,
            "active_and_ablation_have_identical_exam_budget": True,
        },
        "condition_protocols": [
            dict(protocol) for protocol in condition_protocols
        ],
        "conditions_evaluated": len(reports),
        "natural_break_conditions": sum(
            bool(row.get("natural_break_candidate"))
            for row in reports
        ),
        "terminal_revision_conditions": sum(
            bool(row.get("terminal_revision_observed"))
            for row in reports
        ),
        "causal_revision_conditions": sum(
            bool(row.get("causal_natural_revision_gate_passed"))
            for row in reports
        ),
        "replicated_games": len(replicated_games),
        "any_replicated_natural_revision_gate_passed": bool(
            protocol_gate and replicated_games
        ),
        "games": game_reports,
        "rows": reports,
    }


def _condition_rows(
    payload: Mapping[str, Any],
    *,
    budget: int,
) -> list[Dict[str, Any]]:
    active_pairs = {
        (str(pair["game_id"]), int(pair["seed"])): pair
        for pair in payload["active_benchmark"]["pairs"]
    }
    ablated_pairs = {
        (str(pair["game_id"]), int(pair["seed"])): pair
        for pair in payload[
            "structural_revision_ablation_benchmark"
        ]["pairs"]
    }
    rows = []
    for key in sorted(active_pairs):
        game_id, seed = key
        active = dict(active_pairs[key]["unified"])
        ablated = dict(ablated_pairs[key]["unified"])
        natural_break = bool(
            int(active.get("structural_breaks_detected", 0)) > 0
            and int(
                active.get(
                    "structural_revision_hypotheses_generated",
                    0,
                )
            )
            > 0
        )
        terminal_revision = bool(
            int(active.get("structural_revision_confirmations", 0))
            > 0
        )
        causal_advantage = bool(
            int(active.get("max_level_reached", 0))
            > int(ablated.get("max_level_reached", 0))
            or int(active.get("levels_completed_delta", 0))
            > int(ablated.get("levels_completed_delta", 0))
            or int(active.get("wins", 0))
            > int(ablated.get("wins", 0))
        )
        rows.append({
            "game_id": game_id,
            "seed": seed,
            "action_budget": int(budget),
            "natural_break_candidate": natural_break,
            "terminal_revision_observed": terminal_revision,
            "causal_progress_advantage": causal_advantage,
            "causal_natural_revision_gate_passed": bool(
                natural_break
                and terminal_revision
                and causal_advantage
            ),
            "active_max_level_reached": int(
                active.get("max_level_reached", 0)
            ),
            "ablated_max_level_reached": int(
                ablated.get("max_level_reached", 0)
            ),
            "active_levels_completed": int(
                active.get("levels_completed_delta", 0)
            ),
            "ablated_levels_completed": int(
                ablated.get("levels_completed_delta", 0)
            ),
            "active_wins": int(active.get("wins", 0)),
            "ablated_wins": int(ablated.get("wins", 0)),
            "active_breaks_detected": int(
                active.get("structural_breaks_detected", 0)
            ),
            "active_revision_hypotheses": int(
                active.get(
                    "structural_revision_hypotheses_generated",
                    0,
                )
            ),
            "active_revision_confirmations": int(
                active.get("structural_revision_confirmations", 0)
            ),
            "active_controller_errors": _error_count(
                active.get("controller_errors", 0)
            ),
            "ablated_controller_errors": _error_count(
                ablated.get("controller_errors", 0)
            ),
        })
    return rows


def _error_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    return int(value or 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.9z ARC replication matrix.",
    )
    parser.add_argument(
        "--games",
        default=",".join(
            game_splits.resolve("public_unseen_split", full_ids=False)
        ),
    )
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--budgets", default="80,160")
    parser.add_argument("--resets", type=int, default=4)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
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
    budgets = [
        int(item.strip())
        for item in str(args.budgets).split(",")
        if item.strip()
    ]
    payload = run_arc_multigame_replication_benchmark(
        game_ids=games,
        seeds=seeds,
        action_budgets=budgets,
        resets=args.resets,
        environments_dir=args.environments_dir,
        write_path=args.out,
    )
    print(json.dumps({
        "conditions_evaluated": payload["conditions_evaluated"],
        "natural_break_conditions": payload[
            "natural_break_conditions"
        ],
        "terminal_revision_conditions": payload[
            "terminal_revision_conditions"
        ],
        "causal_revision_conditions": payload[
            "causal_revision_conditions"
        ],
        "replicated_games": payload["replicated_games"],
        "replication_gate_passed": payload[
            "any_replicated_natural_revision_gate_passed"
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["protocol"]["protocol_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_arc_multigame_replication_benchmark",
    "summarize_arc_multigame_replication_rows",
]
