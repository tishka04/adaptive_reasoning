"""Compact long-budget SAGE performance and efficiency benchmark.

Unlike the scientific A/B runner, this track executes only the unified
controller.  It measures level completion, wins, action cost per completed
level, wall-clock cost, and a normalized completion-by-efficiency proxy after
each SAGE increment.  Step traces and controller records are intentionally
discarded so overnight runs remain compact.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import game_splits

from .unified_cognition_ab_benchmark import (
    DEFAULT_HELD_OUT_GAMES,
    EnvFactory,
    _env_dir,
    _run_arm,
)
from .unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)


SCHEMA_VERSION = "sage.benchmark_score.v1"
DEFAULT_OUTPUT_PATH = (
    Path("diagnostics") / "perf" / "budget_saturation.json"
)
DEFAULT_HISTORY_PATH = (
    Path("diagnostics") / "perf" / "score_history.json"
)


def run_benchmark_score(
    *,
    game_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (0, 1),
    action_budgets: Sequence[int] = (500, 1500, 4000),
    resets: int = 8,
    environments_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
    controller_factory: Callable[
        [str], UnifiedCognitiveController
    ] | None = None,
    label: str = "current",
    enable_subeffect_eligibility_relay: bool = True,
    enable_generalized_frontier_stall_detection: bool = True,
    enable_per_level_frontier_rearming: bool = True,
    enable_level_route_memory: bool = True,
    enable_level_route_shortening: bool = True,
    write_path: str | Path | None = DEFAULT_OUTPUT_PATH,
    history_path: str | Path | None = DEFAULT_HISTORY_PATH,
) -> Dict[str, Any]:
    """Run the no-ablation-arm performance track and write compact JSON."""
    games = tuple(str(game) for game in (game_ids or DEFAULT_HELD_OUT_GAMES))
    seed_values = tuple(int(seed) for seed in seeds)
    budgets = tuple(sorted({
        max(1, int(budget)) for budget in action_budgets
    }))
    reset_count = max(1, int(resets))
    env_dir = (
        Path(environments_dir)
        if environments_dir is not None
        else _env_dir()
    )
    effective_factory = controller_factory or _configured_controller_factory(
        enable_subeffect_eligibility_relay=(
            enable_subeffect_eligibility_relay
        ),
        enable_generalized_frontier_stall_detection=(
            enable_generalized_frontier_stall_detection
        ),
        enable_per_level_frontier_rearming=(
            enable_per_level_frontier_rearming
        ),
        enable_level_route_memory=enable_level_route_memory,
        enable_level_route_shortening=enable_level_route_shortening,
    )
    rows = []
    started = time.perf_counter()
    for budget in budgets:
        for game_id in games:
            for seed in seed_values:
                condition_started = time.perf_counter()
                arm = _run_arm(
                    arm="unified",
                    game_id=game_id,
                    seed=seed,
                    action_budget_per_reset=budget,
                    resets=reset_count,
                    env_dir=env_dir,
                    env_factory=env_factory,
                    controller_factory=effective_factory,
                )
                rows.append(_compact_condition(
                    arm,
                    budget=budget,
                    wall_clock_seconds=(
                        time.perf_counter() - condition_started
                    ),
                ))
    payload = summarize_benchmark_score(
        rows,
        games=games,
        seeds=seed_values,
        budgets=budgets,
        resets=reset_count,
        label=str(label),
        wall_clock_seconds=time.perf_counter() - started,
        feature_flags={
            "subeffect_eligibility_relay": bool(
                enable_subeffect_eligibility_relay
            ),
            "generalized_frontier_stall_detection": bool(
                enable_generalized_frontier_stall_detection
            ),
            "per_level_frontier_rearming": bool(
                enable_per_level_frontier_rearming
            ),
            "level_route_memory": bool(enable_level_route_memory),
            "level_route_shortening": bool(
                enable_level_route_shortening
            ),
        },
    )
    if write_path is not None:
        _write_json(Path(write_path), payload)
    if history_path is not None:
        append_score_history(Path(history_path), payload)
    return payload


def summarize_benchmark_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    games: Sequence[str],
    seeds: Sequence[int],
    budgets: Sequence[int],
    resets: int,
    label: str,
    wall_clock_seconds: float,
    feature_flags: Mapping[str, bool],
) -> Dict[str, Any]:
    reports = [dict(row) for row in rows]
    saturation = []
    for game_id in games:
        for budget in budgets:
            selected = [
                row for row in reports
                if row["game_id"] == str(game_id)
                and int(row["action_budget_per_reset"]) == int(budget)
            ]
            saturation.append({
                "game_id": str(game_id),
                "action_budget_per_reset": int(budget),
                "conditions": len(selected),
                "max_level_reached": max(
                    (
                        int(row["max_level_reached"])
                        for row in selected
                    ),
                    default=0,
                ),
                "levels_completed": sum(
                    int(row["levels_completed"]) for row in selected
                ),
                "wins": sum(int(row["wins"]) for row in selected),
                "actions_executed": sum(
                    int(row["actions_executed"]) for row in selected
                ),
                "normalized_score_proxy": round(
                    sum(
                        float(row["normalized_score_proxy"])
                        for row in selected
                    ) / max(1, len(selected)),
                    8,
                ),
                "wall_clock_seconds": round(
                    sum(
                        float(row["wall_clock_seconds"])
                        for row in selected
                    ),
                    4,
                ),
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "unified_only_long_budget_completion_efficiency",
        "label": str(label),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_games": [str(game) for game in games],
        "seeds": [int(seed) for seed in seeds],
        "action_budgets": [int(budget) for budget in budgets],
        "resets_per_game_seed_condition": int(resets),
        "feature_flags": dict(feature_flags),
        "score_definition": {
            "per_level_efficiency": (
                "levels_gained_on_transition / actions_since_previous_level"
            ),
            "normalized_score_proxy": (
                "sum(per_level_efficiency) / resets"
            ),
            "higher_is_better": True,
        },
        "wall_clock_seconds": round(float(wall_clock_seconds), 4),
        "total_levels_completed": sum(
            int(row["levels_completed"]) for row in reports
        ),
        "total_wins": sum(int(row["wins"]) for row in reports),
        "maximum_level_reached": max(
            (int(row["max_level_reached"]) for row in reports),
            default=0,
        ),
        "normalized_score_proxy": round(
            sum(float(row["normalized_score_proxy"]) for row in reports),
            8,
        ),
        "saturation_table": saturation,
        "rows": reports,
    }


def append_score_history(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    history: Dict[str, Any] = {
        "schema_version": "sage.score_history.v1",
        "runs": [],
    }
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            history = loaded
            history.setdefault("schema_version", "sage.score_history.v1")
            history.setdefault("runs", [])
    history["runs"].append({
        "label": str(payload.get("label", "")),
        "created_at_utc": str(payload.get("created_at_utc", "")),
        "feature_flags": dict(payload.get("feature_flags", {}) or {}),
        "action_budgets": list(payload.get("action_budgets", []) or []),
        "resets_per_game_seed_condition": int(
            payload.get("resets_per_game_seed_condition", 0) or 0
        ),
        "total_levels_completed": int(
            payload.get("total_levels_completed", 0) or 0
        ),
        "total_wins": int(payload.get("total_wins", 0) or 0),
        "maximum_level_reached": int(
            payload.get("maximum_level_reached", 0) or 0
        ),
        "normalized_score_proxy": float(
            payload.get("normalized_score_proxy", 0.0) or 0.0
        ),
        "wall_clock_seconds": float(
            payload.get("wall_clock_seconds", 0.0) or 0.0
        ),
        "saturation_table": list(
            payload.get("saturation_table", []) or []
        ),
    })
    _write_json(path, history)


def _compact_condition(
    arm: Mapping[str, Any],
    *,
    budget: int,
    wall_clock_seconds: float,
) -> Dict[str, Any]:
    level_events = []
    score = 0.0
    for attempt in arm.get("attempts", []) or []:
        previous_completion_action = 0
        for action_index, step in enumerate(
            attempt.get("trace", []) or [],
            start=1,
        ):
            levels_gained = max(
                0,
                int(step.get("levels_after", 0) or 0)
                - int(step.get("levels_before", 0) or 0),
            )
            if levels_gained <= 0:
                continue
            actions_used = max(
                1,
                action_index - previous_completion_action,
            )
            score += float(levels_gained) / float(actions_used)
            level_events.append({
                "reset_index": int(attempt.get("reset_index", 0) or 0),
                "level_after": int(step.get("levels_after", 0) or 0),
                "levels_gained": levels_gained,
                "cumulative_actions": action_index,
                "actions_used_for_level": actions_used,
            })
            previous_completion_action = action_index
    resets = max(1, int(arm.get("resets_executed", 0) or 0))
    metric_names = (
        "frontier_subeffect_relays_created",
        "frontier_effect_novelty_stalls",
        "frontier_actuator_coverage_stalls",
        "frontier_zero_terminal_branch_stalls",
        "frontier_per_level_rearms",
        "frontier_delayed_terminal_credits",
        "level_routes_observed",
        "level_routes_confirmed",
        "level_route_replay_actions",
        "level_route_shortening_confirmations",
        "level_route_shortening_actions_saved",
    )
    return {
        "game_id": str(arm.get("game_id", "")),
        "seed": int(arm.get("seed", 0) or 0),
        "action_budget_per_reset": int(budget),
        "resets": resets,
        "actions_executed": int(arm.get("actions_executed", 0) or 0),
        "max_level_reached": int(
            arm.get("max_level_reached", 0) or 0
        ),
        "levels_completed": int(
            arm.get("levels_completed_delta", 0) or 0
        ),
        "wins": int(arm.get("wins", 0) or 0),
        "actions_to_each_level": level_events,
        "normalized_score_proxy": round(score / resets, 8),
        "wall_clock_seconds": round(float(wall_clock_seconds), 4),
        "controller_errors": len(
            arm.get("controller_errors", []) or []
        ),
        "failure_causes": dict(arm.get("failure_causes", {}) or {}),
        "mechanism_counters": {
            name: int(arm.get(name, 0) or 0)
            for name in metric_names
        },
    }


def _configured_controller_factory(
    **flags: bool,
) -> Callable[[str], UnifiedCognitiveController]:
    def factory(game_id: str) -> UnifiedCognitiveController:
        return UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(**flags),
        )

    return factory


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run compact long-budget SAGE performance scoring.",
    )
    parser.add_argument(
        "--games",
        default=",".join(DEFAULT_HELD_OUT_GAMES),
    )
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--budgets", default="500,1500,4000")
    parser.add_argument("--resets", type=int, default=8)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--label", default="current")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument(
        "--disable-subeffect-eligibility-relay",
        action="store_true",
    )
    parser.add_argument(
        "--disable-generalized-frontier-stall-detection",
        action="store_true",
    )
    parser.add_argument(
        "--disable-per-level-frontier-rearming",
        action="store_true",
    )
    parser.add_argument("--disable-level-route-memory", action="store_true")
    parser.add_argument(
        "--disable-level-route-shortening",
        action="store_true",
    )
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
    payload = run_benchmark_score(
        game_ids=games,
        seeds=seeds,
        action_budgets=budgets,
        resets=args.resets,
        environments_dir=args.environments_dir,
        label=args.label,
        enable_subeffect_eligibility_relay=(
            not args.disable_subeffect_eligibility_relay
        ),
        enable_generalized_frontier_stall_detection=(
            not args.disable_generalized_frontier_stall_detection
        ),
        enable_per_level_frontier_rearming=(
            not args.disable_per_level_frontier_rearming
        ),
        enable_level_route_memory=(
            not args.disable_level_route_memory
        ),
        enable_level_route_shortening=(
            not args.disable_level_route_shortening
        ),
        write_path=args.out,
        history_path=args.history,
    )
    print(json.dumps({
        "total_levels_completed": payload["total_levels_completed"],
        "total_wins": payload["total_wins"],
        "maximum_level_reached": payload["maximum_level_reached"],
        "normalized_score_proxy": payload["normalized_score_proxy"],
        "wall_clock_seconds": payload["wall_clock_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HISTORY_PATH",
    "DEFAULT_OUTPUT_PATH",
    "SCHEMA_VERSION",
    "append_score_history",
    "run_benchmark_score",
    "summarize_benchmark_score",
]
