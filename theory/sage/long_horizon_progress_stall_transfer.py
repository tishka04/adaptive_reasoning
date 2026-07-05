"""SAGE.4c long-horizon transfer rerun with progress-stall trigger."""

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
from .long_horizon_transfer import (
    DEFAULT_BUDGETS,
    DEFAULT_GAME_ID,
    DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH,
    POST_SWITCH_REPEAT_THRESHOLD,
    REPEAT_COLLAPSE_THRESHOLD,
)
from .progress_stall_trigger import (
    DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    DEFAULT_PROGRESS_STALL_WINDOW,
    DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    DEFAULT_SAME_ACTION_ARG_REPEATS,
)
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    run_sage3_subgoal_switch_probe,
)


DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage4c_long_horizon_progress_stall_transfer_results.json"
)
SAGE4C_SCHEMA_VERSION = "sage.long_horizon_progress_stall_transfer_results.v1"
SAGE4C_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_4C"
SAGE4C_PARTIAL_TRANSFER = (
    "SAGE_PROGRESS_STALL_LONG_HORIZON_PARTIAL_TRANSFER_CANDIDATE_ONLY"
)
SAGE4C_ALL_BUDGETS_TRANSFER = (
    "SAGE_PROGRESS_STALL_LONG_HORIZON_ALL_BUDGETS_TRANSFER_CANDIDATE_ONLY"
)
SAGE4C_TRANSFER_FAILED = (
    "SAGE_PROGRESS_STALL_LONG_HORIZON_TRANSFER_FAILED_CANDIDATE_ONLY"
)

EnvFactory = Callable[[str], Any]


def run_sage4c_long_horizon_progress_stall_transfer(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    baseline_sage4_path: str | Path = DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    progress_stall_window: int = DEFAULT_PROGRESS_STALL_WINDOW,
    same_action_arg_repeats: int = DEFAULT_SAME_ACTION_ARG_REPEATS,
    low_state_novelty_threshold: int = DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    repeated_action_arg_rate_threshold: float = (
        DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD
    ),
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    baseline = _load_optional_json(baseline_sage4_path)
    baseline_rates = _baseline_repeat_rates(baseline)
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
            enable_progress_stall_trigger=True,
            progress_stall_window=progress_stall_window,
            same_action_arg_repeats=same_action_arg_repeats,
            low_state_novelty_threshold=low_state_novelty_threshold,
            repeated_action_arg_rate_threshold=repeated_action_arg_rate_threshold,
        )
        per_budget.append(
            _budget_record(
                int(budget),
                run,
                baseline_repeat_rate=baseline_rates.get(int(budget)),
            )
        )

    comparison = _build_comparison(per_budget, baseline)
    payload = {
        "config": {
            "schema_version": SAGE4C_SCHEMA_VERSION,
            "game_id": game_id,
            "budgets": [int(budget) for budget in budgets],
            "baseline_sage4_path": str(baseline_sage4_path),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "long_horizon_transfer_rerun": True,
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1", "SAGE.4"],
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
        "truth_status": SAGE4C_TRUTH_STATUS,
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
        write_sage4c_long_horizon_progress_stall_results(payload, output_path)
    return payload


def _budget_record(
    budget: int,
    run: Mapping[str, Any],
    *,
    baseline_repeat_rate: float | None,
) -> Dict[str, Any]:
    summary = dict(run.get("summary", {}) or {})
    repeated_rate = float(summary.get("repeated_action_arg_rate", 0.0) or 0.0)
    improved_vs_baseline = (
        baseline_repeat_rate is not None and repeated_rate < float(baseline_repeat_rate)
    )
    active_or_rerun = (
        int(summary.get("active_counterfactuals_after_exhaustion", 0) or 0) > 0
        or int(summary.get("rerun_m2_m3_requested", 0) or 0) > 0
    )
    metrics = {
        "levels_completed": int(summary.get("levels_completed", 0) or 0),
        "terminal_rate": float(summary.get("terminal_rate", 0.0) or 0.0),
        "progress_stall_detected": bool(
            summary.get("progress_stall_detected", False)
        ),
        "progress_stall_switches": int(
            summary.get("progress_stall_switches", 0) or 0
        ),
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
        "repeated_action_arg_rate": repeated_rate,
        "baseline_sage4_repeated_action_arg_rate": baseline_repeat_rate,
        "repeated_action_arg_rate_lower_than_sage4": bool(improved_vs_baseline),
        "unique_state_signatures": int(summary.get("unique_state_signatures", 0) or 0),
        "env_steps": int(summary.get("env_steps", 0) or 0),
        "selected_action_always_legal": bool(
            summary.get("selected_action_always_legal", False)
        ),
        "subgoals_used": list(summary.get("subgoals_used", []) or []),
        "active_counterfactual_or_rerun_requested": bool(active_or_rerun),
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE4C_TRUTH_STATUS,
    }
    gate = _evaluate_sage4c_gate(metrics)
    return {
        "budget": int(budget),
        "metrics": metrics,
        "gate": gate,
        "sage3_outcome_status": str(summary.get("outcome_status", "")),
    }


def _evaluate_sage4c_gate(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    legal = bool(metrics.get("selected_action_always_legal", False))
    progress_stall = bool(metrics.get("progress_stall_detected", False))
    switches_happened = int(metrics.get("subgoal_switches", 0) or 0) > 0
    no_repeat_collapse = (
        float(metrics.get("repeated_action_arg_rate", 1.0)) < REPEAT_COLLAPSE_THRESHOLD
    )
    switch_discipline = (
        float(metrics.get("post_switch_repeat_rate", 1.0)) < POST_SWITCH_REPEAT_THRESHOLD
    )
    improved = bool(metrics.get("repeated_action_arg_rate_lower_than_sage4", False))
    useful = bool(metrics.get("active_counterfactual_or_rerun_requested", False))
    gate_passed = bool(
        legal
        and progress_stall
        and switches_happened
        and no_repeat_collapse
        and switch_discipline
        and improved
        and useful
    )
    return {
        "selected_action_always_legal": legal,
        "progress_stall_detected": progress_stall,
        "subgoal_switches_happened": switches_happened,
        "no_repeat_collapse": no_repeat_collapse,
        "post_switch_repeat_discipline": switch_discipline,
        "repeated_action_arg_rate_lower_than_sage4": improved,
        "active_counterfactual_or_rerun_requested": useful,
        "gate_passed": gate_passed,
    }


def _build_comparison(
    per_budget: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    gates = [bool(row.get("gate", {}).get("gate_passed", False)) for row in per_budget]
    passed = sum(1 for gate in gates if gate)
    total = len(gates)
    any_passed = passed > 0
    all_passed = total > 0 and passed == total
    budgets_with_progress = [
        int(row.get("budget", 0))
        for row in per_budget
        if bool(row.get("metrics", {}).get("progress_stall_detected", False))
    ]
    budgets_with_switches = [
        int(row.get("budget", 0))
        for row in per_budget
        if int(row.get("metrics", {}).get("subgoal_switches", 0) or 0) > 0
    ]
    improved = [
        int(row.get("budget", 0))
        for row in per_budget
        if bool(
            row.get("metrics", {}).get("repeated_action_arg_rate_lower_than_sage4", False)
        )
    ]
    if all_passed:
        outcome = SAGE4C_ALL_BUDGETS_TRANSFER
    elif any_passed:
        outcome = SAGE4C_PARTIAL_TRANSFER
    else:
        outcome = SAGE4C_TRANSFER_FAILED
    return {
        "budgets_evaluated": [int(row.get("budget", 0)) for row in per_budget],
        "budgets_gate_passed": passed,
        "budgets_total": total,
        "all_budgets_gate_passed": all_passed,
        "any_budget_gate_passed": any_passed,
        "budgets_with_progress_stall_detected": budgets_with_progress,
        "budgets_with_subgoal_switches": budgets_with_switches,
        "budgets_with_repetition_improved_vs_sage4": improved,
        "baseline_sage4_outcome_status": str(baseline.get("outcome_status", "")),
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE4C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def write_sage4c_long_horizon_progress_stall_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _baseline_repeat_rates(baseline: Mapping[str, Any]) -> Dict[int, float]:
    rows = baseline.get("per_budget_results", []) or []
    rates: Dict[int, float] = {}
    for row in rows:
        budget = int(row.get("budget", 0) or 0)
        metrics = dict(row.get("metrics", {}) or {})
        rates[budget] = float(metrics.get("repeated_action_arg_rate", 0.0) or 0.0)
    return rates


def _load_optional_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAGE.4c: long-horizon transfer rerun with progress-stall trigger."
        ),
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
    parser.add_argument("--baseline-sage4", default=str(DEFAULT_SAGE4_LONG_HORIZON_RESULTS_PATH))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--out",
        default=str(DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH),
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--max-counterfactual-collections",
        type=int,
        default=DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    )
    parser.add_argument(
        "--progress-stall-window",
        type=int,
        default=DEFAULT_PROGRESS_STALL_WINDOW,
    )
    parser.add_argument(
        "--same-action-arg-repeats",
        type=int,
        default=DEFAULT_SAME_ACTION_ARG_REPEATS,
    )
    parser.add_argument(
        "--low-state-novelty-threshold",
        type=int,
        default=DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    )
    parser.add_argument(
        "--repeated-action-arg-rate-threshold",
        type=float,
        default=DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    )
    args = parser.parse_args(argv)
    run_sage4c_long_horizon_progress_stall_transfer(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        baseline_sage4_path=args.baseline_sage4,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budgets=args.budgets,
        max_counterfactual_collections=args.max_counterfactual_collections,
        progress_stall_window=args.progress_stall_window,
        same_action_arg_repeats=args.same_action_arg_repeats,
        low_state_novelty_threshold=args.low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=args.repeated_action_arg_rate_threshold,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
