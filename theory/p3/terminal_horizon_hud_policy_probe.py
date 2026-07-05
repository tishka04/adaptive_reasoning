"""P3.2b terminal-horizon policy probe with observed HUD-bar source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
)
from theory.m3.objective_refined_window_executor import (
    DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
)
from theory.non_ar25_active_micro_run import _env_dir
from theory.p1.bp35_sage_candidate_policy_probe import (
    DEFAULT_GAME_ID,
    DEFAULT_TIE_BREAK_SEEDS,
    CandidatePolicyMemory,
)
from theory.p3.terminal_horizon_policy_probe import (
    BASELINE_POLICY,
    DEFAULT_K_VALUES,
    DEFAULT_P3_HORIZON_BUDGETS,
    OBJECTIVE_MODE_POLICY,
    STOP_AT_HORIZON_POLICY,
    TRUTH_STATUS,
    compare_horizon_candidate,
    default_condition_executor,
    execute_terminal_horizon_policy_condition,
    horizon_policy_summary,
    infer_empirical_terminal_budget_estimate,
    run_terminal_horizon_policy_probe,
    validate_refined_window_terminal_signal,
)


DEFAULT_P3_HUD_POLICY_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "bp35_terminal_horizon_hud_policy_probe.json"
)
P3_HUD_POLICY_SCHEMA_VERSION = "p3.terminal_horizon_hud_policy_probe.v1"


def hud_condition_executor(
    condition: str,
    budget: int,
    seed: int,
    terminal_budget_estimate: int,
    k_objective: int,
    k_stop: int,
    memory: CandidatePolicyMemory,
    env_dir: Path,
    game_id: str,
) -> Mapping[str, Any]:
    if condition == BASELINE_POLICY:
        return default_condition_executor(
            condition,
            budget,
            seed,
            terminal_budget_estimate,
            k_objective,
            k_stop,
            memory,
            env_dir,
            game_id,
        )
    steps, event = execute_terminal_horizon_policy_condition(
        condition=condition,
        memory=memory,
        environments_dir=env_dir,
        budget=budget,
        game_id=game_id,
        terminal_budget_estimate=terminal_budget_estimate,
        k_objective=k_objective,
        k_stop=k_stop,
        tie_break_seed=seed,
        horizon_source_mode="hud_bar",
    )
    return horizon_policy_summary(
        condition=condition,
        steps=steps,
        budget=budget,
        tie_break_seed=seed,
        terminal_budget_estimate=terminal_budget_estimate,
        k_objective=k_objective,
        k_stop=k_stop,
        event=event,
    )


def summarize_hud_policy_sources(payload: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_results = [
        dict(row.get("candidate", {}) or {})
        for row in payload.get("run_results", []) or []
    ]
    comparisons = [dict(row) for row in payload.get("comparisons", []) or []]
    source_counts = Counter(
        str(row.get("terminal_horizon_source", ""))
        for row in candidate_results
        if row.get("terminal_horizon_source")
    )
    hud_source_runs = source_counts.get("hud_bar", 0)
    fallback_source_runs = source_counts.get("empirical_fallback", 0)
    triggered = [
        row for row in candidate_results if bool(row.get("terminal_horizon_triggered"))
    ]
    trigger_sources = Counter(
        str((row.get("horizon_trigger_log", {}) or {}).get("source", ""))
        for row in candidate_results
        if row.get("horizon_trigger_log")
    )
    return {
        "hud_policy_probe": "P3.2b",
        "horizon_source_mode": "hud_bar",
        "candidate_runs": len(candidate_results),
        "candidate_terminal_horizon_source_counts": dict(source_counts),
        "candidate_hud_bar_source_runs": hud_source_runs,
        "candidate_empirical_fallback_source_runs": fallback_source_runs,
        "horizon_triggered_runs": len(triggered),
        "horizon_trigger_source_counts": dict(trigger_sources),
        "hud_bar_trigger_source_runs": trigger_sources.get("hud_bar", 0),
        "terminal_avoidance_signal_runs": len(
            [row for row in comparisons if bool(row.get("terminal_avoidance_signal"))]
        ),
        "objective_completion_signal_runs": len(
            [row for row in comparisons if bool(row.get("objective_completion_signal"))]
        ),
        "terminal_avoidance_only_runs": len(
            [row for row in comparisons if bool(row.get("terminal_avoidance_only"))]
        ),
        "action6_prefix_count_used_as_decision_variable": False,
        "terminal_avoidance_counted_as_completion": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "candidate_policy_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }


def run_terminal_horizon_hud_policy_probe(
    *,
    refined_window_results_path: str | Path = DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    budgets: Sequence[int] = DEFAULT_P3_HORIZON_BUDGETS,
    tie_break_seeds: Sequence[int] = DEFAULT_TIE_BREAK_SEEDS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    game_id: str = DEFAULT_GAME_ID,
) -> Dict[str, Any]:
    payload = run_terminal_horizon_policy_probe(
        refined_window_results_path=refined_window_results_path,
        scope_consolidation_path=scope_consolidation_path,
        environments_dir=environments_dir,
        budgets=budgets,
        tie_break_seeds=tie_break_seeds,
        k_values=k_values,
        game_id=game_id,
        condition_executor=hud_condition_executor,
    )
    source_summary = summarize_hud_policy_sources(payload)
    payload["config"] = {
        **dict(payload.get("config", {}) or {}),
        "schema_version": P3_HUD_POLICY_SCHEMA_VERSION,
        "horizon_observation_mode": "hud_bar",
        "inputs_read": ["M3.O4", "M3.24", "HUD.2"],
    }
    payload["terminal_horizon_estimator"] = {
        **dict(payload.get("terminal_horizon_estimator", {}) or {}),
        "source": "hud_bar_with_empirical_warmup_fallback",
        "horizon_observation_mode": "hud_bar",
        "hud_bar_validated_by": "HUD.2",
        "hud_bar_source_expected_after_warmup": True,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }
    payload["hud_policy_source_summary"] = source_summary
    payload["summary"] = {
        **dict(payload.get("summary", {}) or {}),
        "terminal_horizon_source": "hud_bar_with_empirical_warmup_fallback",
        "hud_policy_probe": "P3.2b",
        "horizon_observation_mode": "hud_bar",
        "candidate_hud_bar_source_runs": source_summary["candidate_hud_bar_source_runs"],
        "hud_bar_trigger_source_runs": source_summary["hud_bar_trigger_source_runs"],
        "candidate_empirical_fallback_source_runs": (
            source_summary["candidate_empirical_fallback_source_runs"]
        ),
        "action6_prefix_count_used_as_decision_variable": False,
        "terminal_avoidance_counted_as_completion": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }
    payload["status"] = "UNRESOLVED"
    payload["support"] = 0
    payload["revision_status"] = "CANDIDATE_ONLY"
    payload["truth_status"] = TRUTH_STATUS
    payload["candidate_policy_counted_as_confirmation"] = False
    payload["policy_result_counted_as_scientific_verdict"] = False
    return payload


def write_terminal_horizon_hud_policy_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P3_HUD_POLICY_PROBE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P3.2b terminal horizon policy probe using observed HUD-bar source.",
    )
    parser.add_argument(
        "--refined-window-results",
        type=Path,
        default=DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--scope-consolidation",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budgets", type=int, nargs="*", default=list(DEFAULT_P3_HORIZON_BUDGETS))
    parser.add_argument("--tie-break-seeds", type=int, nargs="*", default=list(DEFAULT_TIE_BREAK_SEEDS))
    parser.add_argument("--k-values", type=int, nargs="*", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--out", type=Path, default=DEFAULT_P3_HUD_POLICY_PROBE_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_terminal_horizon_hud_policy_probe(
        refined_window_results_path=args.refined_window_results,
        scope_consolidation_path=args.scope_consolidation,
        environments_dir=args.environments_dir,
        budgets=tuple(args.budgets or DEFAULT_P3_HORIZON_BUDGETS),
        tie_break_seeds=tuple(args.tie_break_seeds or DEFAULT_TIE_BREAK_SEEDS),
        k_values=tuple(args.k_values or DEFAULT_K_VALUES),
        game_id=args.game_id,
    )
    write_terminal_horizon_hud_policy_probe(payload, args.out)


if __name__ == "__main__":
    main()

