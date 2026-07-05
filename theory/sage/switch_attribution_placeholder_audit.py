"""SAGE.5b switch attribution and rerun placeholder audit."""

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
from .long_horizon_transfer import DEFAULT_BUDGETS
from .progress_stall_trigger import (
    DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    DEFAULT_PROGRESS_STALL_WINDOW,
    DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    DEFAULT_SAME_ACTION_ARG_REPEATS,
    PROGRESS_STALL_TRIGGER_REASON,
)
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    SUBGOAL_COUNTERFACTUAL,
    SUBGOAL_EXPLORE_TARGET,
    SUBGOAL_REPOSITION,
    SUBGOAL_RERUN,
    SUBGOAL_SAFE_HOLD,
    TRIGGER_REASON,
    run_sage3_subgoal_switch_probe,
)
from .unknown_game_bounded_probe import (
    DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    DEFAULT_UNKNOWN_GAME_ID,
)


DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage5b_switch_attribution_placeholder_audit.json"
)
SAGE5B_SCHEMA_VERSION = "sage.switch_attribution_placeholder_audit.v1"
SAGE5B_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5B"
PLACEHOLDER_DEPENDENCY_THRESHOLD = 0.5

SAGE5B_PLACEHOLDER_REDUCED = (
    "SAGE_SWITCH_ATTRIBUTION_PLACEHOLDER_DEPENDENCY_REDUCED_CANDIDATE_ONLY"
)
SAGE5B_PLACEHOLDER_HIGH = (
    "SAGE_SWITCH_ATTRIBUTION_PLACEHOLDER_DEPENDENCY_HIGH_CANDIDATE_ONLY"
)

EnvFactory = Callable[[str], Any]


def run_sage5b_switch_attribution_placeholder_audit(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    source_sage5_path: str | Path = DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    game_id: str | None = None,
    budgets: Sequence[int] | None = None,
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    progress_stall_window: int = DEFAULT_PROGRESS_STALL_WINDOW,
    same_action_arg_repeats: int = DEFAULT_SAME_ACTION_ARG_REPEATS,
    low_state_novelty_threshold: int = DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    repeated_action_arg_rate_threshold: float = (
        DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD
    ),
    placeholder_dependency_threshold: float = PLACEHOLDER_DEPENDENCY_THRESHOLD,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    source_sage5 = _load_optional_json(source_sage5_path)
    source_config = dict(source_sage5.get("config", {}) or {})
    resolved_game_id = str(
        game_id or source_config.get("game_id", "") or DEFAULT_UNKNOWN_GAME_ID
    )
    resolved_budgets = [
        int(value)
        for value in (
            budgets
            if budgets is not None
            else source_config.get("budgets", list(DEFAULT_BUDGETS))
        )
    ]
    per_budget: List[Dict[str, Any]] = []
    for budget in resolved_budgets:
        run = run_sage3_subgoal_switch_probe(
            m2_fused_requests_path=m2_fused_requests_path,
            m3_fused_results_path=m3_fused_results_path,
            m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
            p1_policy_probe_path=p1_policy_probe_path,
            p1_utility_handoff_path=p1_utility_handoff_path,
            environments_dir=environments_dir,
            output_path=None,
            game_id=resolved_game_id,
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
            _budget_audit(
                int(budget),
                run,
                placeholder_dependency_threshold=placeholder_dependency_threshold,
            )
        )
    comparison = _build_comparison(
        per_budget,
        source_sage5=source_sage5,
        placeholder_dependency_threshold=placeholder_dependency_threshold,
    )
    payload = {
        "config": {
            "schema_version": SAGE5B_SCHEMA_VERSION,
            "game_id": resolved_game_id,
            "budgets": resolved_budgets,
            "source_sage5_path": str(source_sage5_path),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "placeholder_dependency_threshold": float(
                placeholder_dependency_threshold
            ),
            "switch_attribution_audit": True,
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1", "SAGE.5"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5_context": _source_sage5_context(source_sage5),
        "per_budget_results": per_budget,
        "comparison": comparison,
        "summary": comparison,
        "status": "UNRESOLVED",
        "outcome_status": comparison["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5B_TRUTH_STATUS,
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
        write_sage5b_switch_attribution_placeholder_audit(payload, output_path)
    return payload


def _budget_audit(
    budget: int,
    run: Mapping[str, Any],
    *,
    placeholder_dependency_threshold: float,
) -> Dict[str, Any]:
    summary = dict(run.get("summary", {}) or {})
    switch_steps = [
        dict(row)
        for row in run.get("steps", []) or []
        if bool(row.get("is_subgoal_switch", False))
    ]
    switch_events = [dict(row) for row in run.get("subgoal_switch_events", []) or []]
    total_switches = len(switch_steps)
    by_reason = _count_by_key(switch_steps, "trigger_reason")
    by_subgoal = _count_by_key(switch_steps, "selected_subgoal")
    placeholder_switches = sum(
        1
        for row in switch_steps
        if bool(row.get("placeholder_action_used", False))
        or str(row.get("selected_subgoal", "")) == SUBGOAL_RERUN
    )
    exploratory_switches = sum(
        1
        for row in switch_steps
        if str(row.get("selected_subgoal", ""))
        in {SUBGOAL_REPOSITION, SUBGOAL_EXPLORE_TARGET, SUBGOAL_COUNTERFACTUAL}
        and not bool(row.get("placeholder_action_used", False))
    )
    terminal_guard_switches = sum(
        1
        for row in switch_steps
        if bool(row.get("safe_hold", False))
        or str(row.get("selected_subgoal", "")) == SUBGOAL_SAFE_HOLD
        or str(row.get("trigger_reason", "")) == "terminal_state_safe_hold"
    )
    rerun_requested = int(summary.get("rerun_m2_m3_requested", 0) or 0)
    rerun_effective = int(
        summary.get("rerun_m2_m3_effective_requests_generated", 0) or 0
    )
    placeholder_ratio = _ratio(placeholder_switches, total_switches)
    effective_ratio = _ratio(rerun_effective, rerun_requested)
    dependency_under_threshold = (
        placeholder_ratio <= float(placeholder_dependency_threshold)
    )
    return {
        "budget": int(budget),
        "switch_attribution": {
            "total_switches": total_switches,
            "switches_due_to_success_like_targets_exhausted_or_loop_guard": int(
                by_reason.get(TRIGGER_REASON, 0)
            ),
            "switches_due_to_progress_stall_or_repeat_collapse": int(
                by_reason.get(PROGRESS_STALL_TRIGGER_REASON, 0)
            ),
            "switches_due_to_terminal_guard": terminal_guard_switches,
            "switches_by_trigger_reason": by_reason,
            "switches_by_subgoal": by_subgoal,
            "true_exploratory_switches": exploratory_switches,
            "placeholder_rerun_m2_m3_switches": placeholder_switches,
            "active_counterfactual_switches": int(
                by_subgoal.get(SUBGOAL_COUNTERFACTUAL, 0)
            ),
            "reposition_switches": int(by_subgoal.get(SUBGOAL_REPOSITION, 0)),
            "new_candidate_target_switches": int(
                by_subgoal.get(SUBGOAL_EXPLORE_TARGET, 0)
            ),
            "placeholder_switch_ratio": placeholder_ratio,
            "placeholder_dependency_threshold": float(
                placeholder_dependency_threshold
            ),
            "placeholder_dependency_under_threshold": dependency_under_threshold,
        },
        "placeholder_audit": {
            "rerun_m2_m3_requested": rerun_requested,
            "rerun_m2_m3_effective_requests_generated": rerun_effective,
            "effective_request_ratio": effective_ratio,
            "effective_requests_generated_per_rerun_requested": effective_ratio,
            "placeholder_action_counted_as_subgoal_success": bool(
                summary.get("placeholder_action_counted_as_subgoal_success", False)
            ),
            "placeholder_dependency_high": not dependency_under_threshold,
        },
        "source_summary": {
            "env_steps": int(summary.get("env_steps", 0) or 0),
            "selected_action_always_legal": bool(
                summary.get("selected_action_always_legal", False)
            ),
            "progress_stall_detected": bool(
                summary.get("progress_stall_detected", False)
            ),
            "subgoal_switches": int(summary.get("subgoal_switches", 0) or 0),
            "active_counterfactuals_after_exhaustion": int(
                summary.get("active_counterfactuals_after_exhaustion", 0) or 0
            ),
            "new_candidate_targets_discovered": int(
                summary.get("new_candidate_targets_discovered", 0) or 0
            ),
            "levels_completed": int(summary.get("levels_completed", 0) or 0),
            "support": 0,
            "truth_status": SAGE5B_TRUTH_STATUS,
        },
        "switch_events_sample": switch_events[:5],
        "support": 0,
        "truth_status": SAGE5B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "policy_result_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _build_comparison(
    per_budget: Sequence[Mapping[str, Any]],
    *,
    source_sage5: Mapping[str, Any],
    placeholder_dependency_threshold: float,
) -> Dict[str, Any]:
    total_switches = sum(
        int(row.get("switch_attribution", {}).get("total_switches", 0) or 0)
        for row in per_budget
    )
    success_like = sum(
        int(
            row.get("switch_attribution", {}).get(
                "switches_due_to_success_like_targets_exhausted_or_loop_guard", 0
            )
            or 0
        )
        for row in per_budget
    )
    progress_stall = sum(
        int(
            row.get("switch_attribution", {}).get(
                "switches_due_to_progress_stall_or_repeat_collapse", 0
            )
            or 0
        )
        for row in per_budget
    )
    terminal_guard = sum(
        int(
            row.get("switch_attribution", {}).get(
                "switches_due_to_terminal_guard", 0
            )
            or 0
        )
        for row in per_budget
    )
    exploratory = sum(
        int(row.get("switch_attribution", {}).get("true_exploratory_switches", 0) or 0)
        for row in per_budget
    )
    placeholders = sum(
        int(
            row.get("switch_attribution", {}).get(
                "placeholder_rerun_m2_m3_switches", 0
            )
            or 0
        )
        for row in per_budget
    )
    rerun_requested = sum(
        int(row.get("placeholder_audit", {}).get("rerun_m2_m3_requested", 0) or 0)
        for row in per_budget
    )
    rerun_effective = sum(
        int(
            row.get("placeholder_audit", {}).get(
                "rerun_m2_m3_effective_requests_generated", 0
            )
            or 0
        )
        for row in per_budget
    )
    placeholder_ratio = _ratio(placeholders, total_switches)
    effective_ratio = _ratio(rerun_effective, rerun_requested)
    placeholder_under_threshold = (
        placeholder_ratio <= float(placeholder_dependency_threshold)
    )
    outcome = (
        SAGE5B_PLACEHOLDER_REDUCED
        if placeholder_under_threshold and rerun_effective > 0
        else SAGE5B_PLACEHOLDER_HIGH
    )
    return {
        "source_sage5_outcome_status": str(source_sage5.get("outcome_status", "")),
        "budgets_evaluated": [int(row.get("budget", 0)) for row in per_budget],
        "switches_total": total_switches,
        "switches_due_to_success_like_targets_exhausted_or_loop_guard": success_like,
        "switches_due_to_progress_stall_or_repeat_collapse": progress_stall,
        "switches_due_to_terminal_guard": terminal_guard,
        "true_exploratory_switches": exploratory,
        "placeholder_rerun_m2_m3_switches": placeholders,
        "placeholder_switch_ratio": placeholder_ratio,
        "placeholder_dependency_threshold": float(placeholder_dependency_threshold),
        "placeholder_dependency_under_threshold": placeholder_under_threshold,
        "rerun_m2_m3_requested": rerun_requested,
        "rerun_m2_m3_effective_requests_generated": rerun_effective,
        "effective_request_ratio": effective_ratio,
        "active_counterfactual_switches_total": sum(
            int(row.get("switch_attribution", {}).get("active_counterfactual_switches", 0) or 0)
            for row in per_budget
        ),
        "active_counterfactuals_after_exhaustion_total": sum(
            int(row.get("source_summary", {}).get("active_counterfactuals_after_exhaustion", 0) or 0)
            for row in per_budget
        ),
        "levels_completed_max": max(
            [int(row.get("source_summary", {}).get("levels_completed", 0) or 0) for row in per_budget]
            or [0]
        ),
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE5B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _source_sage5_context(source_sage5: Mapping[str, Any]) -> Dict[str, Any]:
    comparison = dict(source_sage5.get("comparison", {}) or {})
    return {
        "source_artifact": "SAGE.5",
        "unknown_game": bool(comparison.get("unknown_game", False)),
        "budgets_gate_passed": int(comparison.get("budgets_gate_passed", 0) or 0),
        "budgets_total": int(comparison.get("budgets_total", 0) or 0),
        "outcome_status": str(source_sage5.get("outcome_status", "")),
        "source_counted_as_scientific_evidence": False,
        "support": 0,
        "truth_status": SAGE5B_TRUTH_STATUS,
    }


def write_sage5b_switch_attribution_placeholder_audit(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _count_by_key(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "") or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _load_optional_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.5b switch attribution and placeholder audit.",
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
    parser.add_argument("--source-sage5", default=str(DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH))
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
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
    parser.add_argument(
        "--placeholder-dependency-threshold",
        type=float,
        default=PLACEHOLDER_DEPENDENCY_THRESHOLD,
    )
    args = parser.parse_args(argv)
    run_sage5b_switch_attribution_placeholder_audit(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        source_sage5_path=args.source_sage5,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budgets=args.budgets,
        max_counterfactual_collections=args.max_counterfactual_collections,
        progress_stall_window=args.progress_stall_window,
        same_action_arg_repeats=args.same_action_arg_repeats,
        low_state_novelty_threshold=args.low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=args.repeated_action_arg_rate_threshold,
        placeholder_dependency_threshold=args.placeholder_dependency_threshold,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
