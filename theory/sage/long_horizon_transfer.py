"""SAGE.4 known-game long-horizon transfer on ar25.

SAGE.1b (anti-loop) and SAGE.3 (subgoal switch after exhaustion) were validated
on the short bp35 instance. SAGE.4 checks whether that loop discipline *transfers*
to a longer-horizon known game (ar25) across several budgets, without collapsing
into repetition, without illegal actions, and with at least one useful subgoal
capability beyond bp35.

The gate is NOT "solve ar25". Each artefact keeps ``support=0``,
``policy_result_counted_as_confirmation=false`` and
``truth_status=NOT_EVALUATED_BY_SAGE_4``. No A32/A33 write is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .known_game_closed_loop_scaffold import (
    DEFAULT_M2_FUSED_REQUESTS_PATH,
    DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH,
    DEFAULT_M3_FUSED_RESULTS_PATH,
    DEFAULT_P1_POLICY_PROBE_PATH,
    DEFAULT_P1_UTILITY_HANDOFF_PATH,
)
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    run_sage3_subgoal_switch_probe,
)


DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage4_long_horizon_transfer_results.json"
)
SAGE4_SCHEMA_VERSION = "sage.long_horizon_transfer_results.v1"
SAGE4_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_4"
DEFAULT_GAME_ID = "ar25-e3c63847"
DEFAULT_BUDGETS: tuple[int, ...] = (50, 150, 300)

# Discipline gate thresholds (candidate-only, not scientific verdicts).
REPEAT_COLLAPSE_THRESHOLD = 0.5
POST_SWITCH_REPEAT_THRESHOLD = 0.5

EnvFactory = Callable[[str], Any]


def run_sage4_long_horizon_transfer(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run the SAGE.3 loop on a longer-horizon known game across budgets."""
    per_budget: List[Dict[str, Any]] = []
    for budget in budgets:
        run = run_sage3_subgoal_switch_probe(
            m2_fused_requests_path=m2_fused_requests_path,
            m3_fused_results_path=m3_fused_results_path,
            m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
            p1_policy_probe_path=p1_policy_probe_path,
            p1_utility_handoff_path=p1_utility_handoff_path,
            environments_dir=environments_dir,
            output_path=None,
            game_id=game_id,
            budget=int(budget),
            max_counterfactual_collections=max_counterfactual_collections,
            env_factory=env_factory,
        )
        per_budget.append(_budget_record(int(budget), run))

    comparison = _build_transfer_comparison(per_budget)
    payload = {
        "config": {
            "schema_version": SAGE4_SCHEMA_VERSION,
            "game_id": game_id,
            "budgets": [int(b) for b in budgets],
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "long_horizon_transfer_probe": True,
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
            "gate_thresholds": {
                "repeat_collapse_threshold": REPEAT_COLLAPSE_THRESHOLD,
                "post_switch_repeat_threshold": POST_SWITCH_REPEAT_THRESHOLD,
            },
        },
        "per_budget_results": per_budget,
        "comparison": comparison,
        "status": "UNRESOLVED",
        "outcome_status": comparison["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE4_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage4_long_horizon_results(payload, output_path)
    return payload


def _budget_record(budget: int, run: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(run.get("summary", {}) or {})
    metrics = {
        "levels_completed": int(summary.get("levels_completed", 0) or 0),
        "terminal_rate": float(summary.get("terminal_rate", 0.0) or 0.0),
        "subgoal_switches": int(summary.get("subgoal_switches", 0) or 0),
        "subgoal_switch_success_rate": float(
            summary.get("subgoal_switch_success_rate", 0.0) or 0.0
        ),
        "new_candidate_targets_discovered": int(
            summary.get("new_candidate_targets_discovered", 0) or 0
        ),
        "active_counterfactuals_after_exhaustion": int(
            summary.get("active_counterfactuals_after_exhaustion", 0) or 0
        ),
        "rerun_m2_m3_requested": int(summary.get("rerun_m2_m3_requested", 0) or 0),
        "rerun_m2_m3_effective_requests_generated": int(
            summary.get("rerun_m2_m3_effective_requests_generated", 0) or 0
        ),
        "post_switch_repeat_rate": float(
            summary.get("post_switch_repeat_rate", 0.0) or 0.0
        ),
        "repeated_action_arg_rate": float(
            summary.get("repeated_action_arg_rate", 0.0) or 0.0
        ),
        "unique_state_signatures": int(summary.get("unique_state_signatures", 0) or 0),
        "env_steps": int(summary.get("env_steps", 0) or 0),
        "selected_action_always_legal": bool(
            summary.get("selected_action_always_legal", False)
        ),
        "subgoals_used": list(summary.get("subgoals_used", []) or []),
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE4_TRUTH_STATUS,
    }
    gate = _evaluate_gate(metrics)
    return {
        "budget": int(budget),
        "metrics": metrics,
        "gate": gate,
        "sage3_outcome_status": str(summary.get("outcome_status", "")),
    }


def _evaluate_gate(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    legal = bool(metrics.get("selected_action_always_legal", False))
    no_repeat_collapse = (
        float(metrics.get("repeated_action_arg_rate", 1.0)) < REPEAT_COLLAPSE_THRESHOLD
    )
    switch_discipline = (
        float(metrics.get("post_switch_repeat_rate", 1.0)) < POST_SWITCH_REPEAT_THRESHOLD
    )
    switches_happened = int(metrics.get("subgoal_switches", 0)) > 0
    useful_subgoal_capability = (
        float(metrics.get("subgoal_switch_success_rate", 0.0)) > 0.0
        or int(metrics.get("active_counterfactuals_after_exhaustion", 0)) >= 1
        or int(metrics.get("new_candidate_targets_discovered", 0)) >= 1
    )
    passed = bool(
        legal
        and no_repeat_collapse
        and switch_discipline
        and switches_happened
        and useful_subgoal_capability
    )
    return {
        "selected_action_always_legal": legal,
        "no_repeat_collapse": no_repeat_collapse,
        "post_switch_repeat_discipline": switch_discipline,
        "subgoal_switches_happened": switches_happened,
        "useful_subgoal_capability": useful_subgoal_capability,
        "gate_passed": passed,
    }


def _build_transfer_comparison(
    per_budget: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    gates = [bool(row.get("gate", {}).get("gate_passed", False)) for row in per_budget]
    passed = sum(1 for g in gates if g)
    total = len(gates)
    all_passed = total > 0 and passed == total
    any_passed = passed > 0

    if all_passed:
        outcome_status = "SAGE_LOOP_DISCIPLINE_TRANSFERS_TO_LONG_HORIZON_CANDIDATE_ONLY"
    elif any_passed:
        outcome_status = "SAGE_LOOP_DISCIPLINE_PARTIAL_TRANSFER_CANDIDATE_ONLY"
    else:
        outcome_status = "SAGE_LOOP_DISCIPLINE_TRANSFER_FAILED_CANDIDATE_ONLY"

    useful_beyond_bp35 = any(
        int(row.get("metrics", {}).get("new_candidate_targets_discovered", 0)) >= 1
        for row in per_budget
    )
    return {
        "budgets_evaluated": [int(row.get("budget", 0)) for row in per_budget],
        "budgets_gate_passed": passed,
        "budgets_total": total,
        "all_budgets_gate_passed": all_passed,
        "any_budget_gate_passed": any_passed,
        "discovered_new_candidate_targets_beyond_bp35": useful_beyond_bp35,
        "outcome_status": outcome_status,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE4_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def write_sage4_long_horizon_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.4 long-horizon transfer probe on a known game.",
    )
    parser.add_argument("--m2-fused-requests", default=str(DEFAULT_M2_FUSED_REQUESTS_PATH))
    parser.add_argument("--m3-fused-results", default=str(DEFAULT_M3_FUSED_RESULTS_PATH))
    parser.add_argument(
        "--m3-counterfactual-feasibility",
        default=str(DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH),
    )
    parser.add_argument("--p1-policy-probe", default=str(DEFAULT_P1_POLICY_PROBE_PATH))
    parser.add_argument(
        "--p1-utility-handoff",
        default=str(DEFAULT_P1_UTILITY_HANDOFF_PATH),
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH))
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
    )
    parser.add_argument(
        "--max-counterfactual-collections",
        type=int,
        default=DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    )
    args = parser.parse_args(argv)
    run_sage4_long_horizon_transfer(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budgets=args.budgets,
        max_counterfactual_collections=args.max_counterfactual_collections,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
