"""P2.3 long-budget saturation probe for the P1 movement-refresh policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p1.bp35_sage_candidate_policy_probe import (
    CONDITIONAL_MOVEMENT_REFRESH_POLICY,
    DEFAULT_GAME_ID,
    DEFAULT_TIE_BREAK_SEEDS,
    MOVEMENT_REFRESH_CANDIDATES,
    candidate_policy_memory_from_scope,
    execute_probe_condition,
    summarize_probe_steps,
)
from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir
from theory.p2.policy_frontier_records import TRUTH_STATUS


DEFAULT_P2_LONG_BUDGET_SATURATION_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p2" / "bp35_long_budget_saturation_probe.json"
)
DEFAULT_LONG_BUDGETS = (48, 64, 96, 128, 192, 256)

NO_SATURATION = "NO_SATURATION_ACTION6_REMAINS_PRODUCTIVE"
REFRESH_UNLOCKED = "SATURATION_REFRESH_UNLOCKED_ACTION6"
REFRESH_POST_UNOBSERVED = "SATURATION_REFRESH_ATTEMPTED_POST_REFRESH_UNOBSERVED"
TRUE_FRONTIER = "TRUE_FRONTIER_AFTER_FAILED_MOVEMENT_REFRESH"


def run_long_budget_saturation_probe(
    *,
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    budgets: Sequence[int] = DEFAULT_LONG_BUDGETS,
    tie_break_seeds: Sequence[int] = DEFAULT_TIE_BREAK_SEEDS,
    game_id: str = DEFAULT_GAME_ID,
) -> Dict[str, Any]:
    scope_payload = _load_json(scope_consolidation_path)
    memory = candidate_policy_memory_from_scope(scope_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    run_results = []
    for budget in [int(value) for value in budgets]:
        for tie_break_seed in [int(value) for value in tie_break_seeds]:
            steps = execute_probe_condition(
                condition=CONDITIONAL_MOVEMENT_REFRESH_POLICY,
                memory=memory,
                environments_dir=env_dir,
                budget=budget,
                game_id=game_id,
                tie_break_seed=tie_break_seed,
            )
            summary = summarize_probe_steps(
                CONDITIONAL_MOVEMENT_REFRESH_POLICY,
                steps,
            )
            classification = classify_saturation_outcome(summary, budget=budget)
            run_results.append(
                {
                    "budget": budget,
                    "tie_break_seed": tie_break_seed,
                    "condition": CONDITIONAL_MOVEMENT_REFRESH_POLICY,
                    "summary": summary,
                    "classification": classification,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": TRUTH_STATUS,
                    "revision_performed": False,
                    "wrong_confirmations": 0,
                    "budget_exhausted_counted_as_saturation": False,
                    "policy_result_counted_as_scientific_verdict": False,
                }
            )

    aggregate = aggregate_saturation_probe_results(run_results)
    return {
        "config": {
            "schema_version": "p2.long_budget_saturation_probe.v1",
            "scope_consolidation_path": str(scope_consolidation_path),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "condition": CONDITIONAL_MOVEMENT_REFRESH_POLICY,
            "budgets": [int(value) for value in budgets],
            "tie_break_seeds": [int(value) for value in tie_break_seeds],
            "refresh_candidates": list(MOVEMENT_REFRESH_CANDIDATES),
            "inputs_read": ["M3.24"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["A40", "M2", "M3", "A32", "A33"],
            "saturation_definition": (
                "no_effective_ACTION6_available + movement_refresh_attempted "
                "+ observed_no_new_useful_ACTION6_after_refresh"
            ),
            "budget_exhaustion_is_not_saturation": True,
        },
        "candidate_policy_memory": memory.to_dict(),
        "run_results": run_results,
        "frontier_observations": [
            row
            for row in run_results
            if row["classification"]["true_frontier_triggered"]
        ],
        "aggregate": aggregate,
        "summary": {
            **aggregate,
            "a40_write_performed": False,
            "m2_write_performed": False,
            "m3_write_performed": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": "UNRESOLVED",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "budget_exhausted_counted_as_saturation": False,
        "long_budget_probe_counted_as_confirmation": False,
    }


def classify_saturation_outcome(
    summary: Mapping[str, Any],
    *,
    budget: int,
) -> Dict[str, Any]:
    steps = int(summary.get("policy_steps", 0) or 0)
    triggers = int(summary.get("conditional_movement_refresh_triggers", 0) or 0)
    action6_after_refresh = int(
        summary.get("action6_after_conditional_movement_refresh_steps", 0) or 0
    )
    useful_after_refresh = int(
        summary.get(
            "useful_action6_after_conditional_movement_refresh_steps",
            0,
        )
        or 0
    )
    new_after_refresh = int(
        summary.get("new_action6_affordances_after_movement_refresh", 0) or 0
    )
    budget_exhausted = steps >= int(budget)
    movement_refresh_attempted = triggers > 0
    post_refresh_observed = action6_after_refresh > 0
    movement_refresh_unlocked = movement_refresh_attempted and (
        useful_after_refresh > 0 or new_after_refresh > 0
    )
    true_frontier = bool(
        movement_refresh_attempted
        and post_refresh_observed
        and not movement_refresh_unlocked
    )
    if not movement_refresh_attempted:
        outcome = NO_SATURATION
    elif movement_refresh_unlocked:
        outcome = REFRESH_UNLOCKED
    elif not post_refresh_observed:
        outcome = REFRESH_POST_UNOBSERVED
    else:
        outcome = TRUE_FRONTIER

    return {
        "outcome": outcome,
        "budget": int(budget),
        "policy_steps": steps,
        "budget_exhausted": budget_exhausted,
        "budget_exhausted_counted_as_saturation": False,
        "movement_refresh_attempted": movement_refresh_attempted,
        "no_effective_action6_inferred_from_refresh_trigger": (
            movement_refresh_attempted
        ),
        "movement_refresh_triggers": triggers,
        "post_refresh_action6_observed": post_refresh_observed,
        "action6_after_movement_refresh_steps": action6_after_refresh,
        "useful_action6_after_movement_refresh_steps": useful_after_refresh,
        "new_action6_affordances_after_movement_refresh": new_after_refresh,
        "movement_refresh_unlocked_action6": movement_refresh_unlocked,
        "true_frontier_triggered": true_frontier,
        "ready_for_p2_4_handoff": true_frontier,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def aggregate_saturation_probe_results(
    run_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    classifications = [
        dict(row.get("classification", {}) or {}) for row in run_results
    ]
    outcome_counts = Counter(str(row.get("outcome", "")) for row in classifications)
    budget_exhausted_runs = len(
        [row for row in classifications if bool(row.get("budget_exhausted", False))]
    )
    true_frontier_runs = int(outcome_counts.get(TRUE_FRONTIER, 0))
    return {
        "budget_runs": len(run_results),
        "budgets_tested": sorted({int(row.get("budget", 0) or 0) for row in run_results}),
        "tie_break_seeds_tested": sorted(
            {int(row.get("tie_break_seed", 0) or 0) for row in run_results}
        ),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "no_saturation_runs": int(outcome_counts.get(NO_SATURATION, 0)),
        "movement_refresh_unlock_runs": int(outcome_counts.get(REFRESH_UNLOCKED, 0)),
        "post_refresh_unobserved_runs": int(
            outcome_counts.get(REFRESH_POST_UNOBSERVED, 0)
        ),
        "true_frontier_runs": true_frontier_runs,
        "budget_exhausted_runs": budget_exhausted_runs,
        "budget_exhausted_counted_as_saturation": False,
        "real_frontier_ready_for_p2_4": true_frontier_runs > 0,
        "a40_write_performed": False,
        "m2_write_performed": False,
        "m3_write_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_long_budget_saturation_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P2_LONG_BUDGET_SATURATION_PROBE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P2.3 long-budget saturation probe.",
    )
    parser.add_argument(
        "--scope-consolidation",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--tie-break-seeds",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P2_LONG_BUDGET_SATURATION_PROBE_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_long_budget_saturation_probe(
        scope_consolidation_path=args.scope_consolidation,
        environments_dir=args.environments_dir,
        budgets=tuple(args.budgets or DEFAULT_LONG_BUDGETS),
        tie_break_seeds=tuple(args.tie_break_seeds or DEFAULT_TIE_BREAK_SEEDS),
        game_id=args.game_id,
    )
    write_long_budget_saturation_probe(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
