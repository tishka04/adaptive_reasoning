"""P3.2 terminal-horizon objective mode policy probe for bp35."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
)
from theory.m3.objective_refined_window_executor import (
    DEFAULT_OBJECTIVE_REFINED_WINDOW_RESULTS_OUTPUT_PATH,
)
from theory.m3.objective_stop_switch_experiment_executor import (
    execute_decision_step,
    is_terminal_game_state,
    update_rollout_memory,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.p1.bp35_sage_candidate_policy_probe import (
    DEFAULT_GAME_ID,
    DEFAULT_TIE_BREAK_SEEDS,
    PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
    CandidatePolicyMemory,
    ProbeDecision,
    ProbeStep,
    candidate_policy_memory_from_scope,
    execute_probe_condition,
    select_probe_decision,
    state_signature,
    summarize_probe_steps,
)
from theory.p3.terminal_aware_stop_policy_probe import (
    validate_refined_window_terminal_signal,
)
from theory.p3.terminal_horizon_estimator import (
    TerminalHorizonEstimate,
    estimate_terminal_horizon,
)
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_P3_TERMINAL_HORIZON_POLICY_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "bp35_terminal_horizon_policy_probe.json"
)
DEFAULT_P3_HORIZON_BUDGETS = (64, 96)
DEFAULT_K_VALUES = (1, 2, 4, 6, 8, 12)
P3_TERMINAL_HORIZON_SCHEMA_VERSION = "p3.terminal_horizon_policy_probe.v1"
TRUTH_STATUS = "NOT_EVALUATED_BY_P3_AGENT_PROBE"
BASELINE_POLICY = "patch_similarity_soft_stale_action6_prefix"
STOP_AT_HORIZON_POLICY = "terminal_horizon_stop_guard"
OBJECTIVE_MODE_POLICY = "terminal_horizon_objective_mode"
OBJECTIVE_MODE_ACTIONS = ("ACTION3", "ACTION4")
STOP_OR_HOLD_SEMANTICS = "stop_rollout_without_extra_env_action_when_horizon_guard_fires"


def infer_empirical_terminal_budget_estimate(refined_payload: Mapping[str, Any]) -> int:
    signal = validate_refined_window_terminal_signal(refined_payload)
    # P3.1 stopped at the prefix immediately before the observed terminal step.
    return int(signal["stop_threshold_action6_prefix_count"]) + 1


def env_actions_used(steps: Sequence[ProbeStep]) -> int:
    return sum(int(getattr(step, "env_actions", 1) or 0) for step in steps)


def choose_objective_mode_decision(
    *,
    valid_actions: Sequence[Any],
    action_counts: Mapping[str, int],
    tie_break_seed: int,
) -> ProbeDecision | None:
    available = {str(getattr(action, "name", "")) for action in valid_actions}
    candidates = [name for name in OBJECTIVE_MODE_ACTIONS if name in available]
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda name: (int(action_counts.get(name, 0) or 0), (OBJECTIVE_MODE_ACTIONS.index(name) + tie_break_seed) % len(OBJECTIVE_MODE_ACTIONS)),
    )
    return ProbeDecision(
        condition=OBJECTIVE_MODE_POLICY,
        action_name=ordered[0],
        decision_reason="terminal_horizon_objective_mode_action",
        candidate_policy_used=True,
        candidate_score_details={
            "objective_mode_actions": list(OBJECTIVE_MODE_ACTIONS),
            "selection_policy": "least_used_objective_action",
        },
    )


def execute_terminal_horizon_policy_condition(
    *,
    condition: str,
    memory: CandidatePolicyMemory,
    environments_dir: str | Path,
    budget: int,
    game_id: str,
    terminal_budget_estimate: int,
    k_objective: int,
    k_stop: int,
    tie_break_seed: int = 0,
    horizon_source_mode: str = "empirical_fallback",
) -> tuple[tuple[ProbeStep, ...], dict[str, Any]]:
    env = _make_env(game_id, environments_dir)
    current_frame = _reset_env(env)
    action_history: list[str] = []
    used_action6_args: list[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    seen_states: set[str] = set()
    steps: list[ProbeStep] = []
    horizon_logs: list[dict[str, Any]] = []
    initial = snapshot_frame(current_frame)
    grid_history = [initial.grid]
    seen_states.add(state_signature(initial.grid, initial.levels_completed, initial.game_state))

    stop_event: dict[str, Any] = {
        "terminal_horizon_triggered": False,
        "horizon_source_mode": horizon_source_mode,
        "terminal_horizon_source": (
            "hud_bar_or_fallback"
            if horizon_source_mode == "hud_bar"
            else "empirical_fallback"
        ),
        "estimated_moves_remaining": None,
        "k_objective": int(k_objective),
        "k_stop": int(k_stop),
        "trigger_reason": None,
        "objective_mode_entered": False,
        "objective_mode_actions_selected": [],
        "stop_triggered": False,
        "stop_trigger_step": None,
        "stop_or_hold_semantics": STOP_OR_HOLD_SEMANTICS,
    }

    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(current_frame)
        if is_terminal_game_state(before.game_state):
            stop_event["trigger_reason"] = "terminal_state_reached_before_horizon_policy"
            break
        use_hud = horizon_source_mode == "hud_bar"
        estimate = estimate_terminal_horizon(
            observation=before.grid if use_hud else None,
            history=grid_history[:-1] if use_hud else None,
            policy_state={"env_actions_executed": env_actions_used(steps)},
            terminal_budget_estimate=terminal_budget_estimate,
        )
        log = {
            "step": step_index,
            **estimate.to_dict(),
            "k_objective": int(k_objective),
            "k_stop": int(k_stop),
            "terminal_horizon_triggered": False,
            "objective_mode_entered": False,
            "stop_guard_entered": False,
            "trigger_reason": None,
        }
        remaining = estimate.estimated_moves_remaining
        if remaining is not None and remaining <= int(k_stop):
            log.update(
                {
                    "terminal_horizon_triggered": True,
                    "stop_guard_entered": True,
                    "trigger_reason": "moves_remaining_below_stop_threshold",
                }
            )
            horizon_logs.append(log)
            stop_event.update(
                {
                    "terminal_horizon_triggered": True,
                    "terminal_horizon_source": estimate.source,
                    "estimated_moves_remaining": remaining,
                    "trigger_reason": "moves_remaining_below_stop_threshold",
                    "stop_triggered": True,
                    "stop_trigger_step": step_index,
                }
            )
            break

        valid_actions = list(_valid_actions(env))
        if (
            condition == OBJECTIVE_MODE_POLICY
            and remaining is not None
            and remaining <= int(k_objective)
        ):
            decision = choose_objective_mode_decision(
                valid_actions=valid_actions,
                action_counts=action_counts,
                tie_break_seed=tie_break_seed,
            )
            log.update(
                {
                    "terminal_horizon_triggered": True,
                    "objective_mode_entered": True,
                    "trigger_reason": "moves_remaining_below_objective_threshold",
                }
            )
            stop_event["objective_mode_entered"] = True
            if decision is None:
                log["trigger_reason"] = "objective_mode_no_available_action"
                horizon_logs.append(log)
                stop_event.update(
                    {
                        "terminal_horizon_triggered": True,
                        "terminal_horizon_source": estimate.source,
                        "estimated_moves_remaining": remaining,
                        "trigger_reason": "objective_mode_no_available_action",
                        "stop_triggered": True,
                        "stop_trigger_step": step_index,
                    }
                )
                break
        elif condition == STOP_AT_HORIZON_POLICY and remaining is not None and remaining <= int(k_objective):
            log.update(
                {
                    "terminal_horizon_triggered": True,
                    "stop_guard_entered": True,
                    "trigger_reason": "moves_remaining_below_candidate_stop_threshold",
                }
            )
            horizon_logs.append(log)
            stop_event.update(
                {
                    "terminal_horizon_triggered": True,
                    "terminal_horizon_source": estimate.source,
                    "estimated_moves_remaining": remaining,
                    "trigger_reason": "moves_remaining_below_candidate_stop_threshold",
                    "stop_triggered": True,
                    "stop_trigger_step": step_index,
                }
            )
            break
        else:
            decision = select_probe_decision(
                condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
                memory=memory,
                before_grid=before.grid,
                valid_actions=valid_actions,
                action_history=tuple(action_history),
                used_action6_args=tuple(used_action6_args),
                action_counts=action_counts,
                previous_steps=tuple(steps),
                tie_break_seed=tie_break_seed,
            )
        horizon_logs.append(log)
        current_frame, step = execute_decision_step(
            env,
            current_frame,
            decision=decision,
            memory=memory,
            condition_label=condition,
            step_index=step_index,
            seen_states=seen_states,
        )
        steps.append(step)
        try:
            after_snapshot = snapshot_frame(current_frame)
            grid_history.append(after_snapshot.grid)
        except ValueError:
            pass
        if log["objective_mode_entered"]:
            stop_event["objective_mode_actions_selected"].append(step.policy_selected_action)
        seen_states.add(step.state_signature_after)
        update_rollout_memory(
            step,
            action_history=action_history,
            used_action6_args=used_action6_args,
            action_counts=action_counts,
        )

    stop_event["horizon_logs"] = horizon_logs
    stop_event["moves_used"] = env_actions_used(steps)
    stop_event["horizon_source_counts"] = dict(
        Counter(str(row.get("source", "")) for row in horizon_logs if row.get("source"))
    )
    stop_event["hud_bar_source_steps"] = len(
        [row for row in horizon_logs if row.get("source") == "hud_bar"]
    )
    stop_event["empirical_fallback_source_steps"] = len(
        [row for row in horizon_logs if row.get("source") == "empirical_fallback"]
    )
    use_hud_final = horizon_source_mode == "hud_bar" and bool(grid_history)
    final_estimate = estimate_terminal_horizon(
        observation=grid_history[-1] if use_hud_final else None,
        history=grid_history[:-1] if use_hud_final else None,
        policy_state={"env_actions_executed": env_actions_used(steps)},
        terminal_budget_estimate=terminal_budget_estimate,
    )
    stop_event["final_horizon_estimate"] = final_estimate.to_dict()
    return tuple(steps), stop_event


def horizon_policy_summary(
    *,
    condition: str,
    steps: Sequence[ProbeStep],
    budget: int,
    tie_break_seed: int,
    terminal_budget_estimate: int,
    k_objective: int | None,
    k_stop: int | None,
    event: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    summary = summarize_probe_steps(condition, steps)
    final_game_state = str(summary.get("final_game_state", ""))
    terminal = is_terminal_game_state(final_game_state)
    final_estimate = dict((event or {}).get("final_horizon_estimate", {}) or {})
    horizon_logs = list((event or {}).get("horizon_logs", []) or [])
    trigger_logs = [
        dict(row)
        for row in horizon_logs
        if bool(row.get("terminal_horizon_triggered"))
    ]
    return {
        **summary,
        "budget": int(budget),
        "tie_break_seed": int(tie_break_seed),
        "terminal_budget_estimate": int(terminal_budget_estimate),
        "terminal_horizon_source": str(
            final_estimate.get("source")
            or (event or {}).get("terminal_horizon_source")
            or "empirical_fallback"
        ),
        "horizon_source_mode": (event or {}).get("horizon_source_mode"),
        "horizon_source_counts": dict((event or {}).get("horizon_source_counts", {}) or {}),
        "hud_bar_source_steps": int((event or {}).get("hud_bar_source_steps", 0) or 0),
        "empirical_fallback_source_steps": int(
            (event or {}).get("empirical_fallback_source_steps", 0) or 0
        ),
        "estimated_moves_remaining": final_estimate.get("estimated_moves_remaining"),
        "terminal_fraction_remaining": final_estimate.get("terminal_fraction_remaining"),
        "terminal_horizon_evidence": dict(final_estimate.get("evidence", {}) or {}),
        "moves_used": int((event or {}).get("moves_used", env_actions_used(steps)) or 0),
        "k_objective": k_objective,
        "k_stop": k_stop,
        "terminal_horizon_triggered": bool((event or {}).get("terminal_horizon_triggered")),
        "horizon_trigger_log": trigger_logs[-1] if trigger_logs else None,
        "trigger_reason": (event or {}).get("trigger_reason"),
        "objective_mode_entered": bool((event or {}).get("objective_mode_entered")),
        "objective_mode_actions_selected": list(
            (event or {}).get("objective_mode_actions_selected", []) or []
        ),
        "stop_triggered": bool((event or {}).get("stop_triggered")),
        "stop_trigger_step": (event or {}).get("stop_trigger_step"),
        "terminal_state_after_rollout": terminal,
        "objective_completion_signal": int(summary.get("final_levels_completed", 0) or 0) > 0,
        "terminal_avoidance_counted_as_completion": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


ConditionExecutor = Callable[
    [str, int, int, int, int, int, CandidatePolicyMemory, Path, str],
    Mapping[str, Any],
]


def default_condition_executor(
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
        steps = execute_probe_condition(
            condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            memory=memory,
            environments_dir=env_dir,
            budget=budget,
            game_id=game_id,
            tie_break_seed=seed,
        )
        return horizon_policy_summary(
            condition=BASELINE_POLICY,
            steps=steps,
            budget=budget,
            tie_break_seed=seed,
            terminal_budget_estimate=terminal_budget_estimate,
            k_objective=None,
            k_stop=None,
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


def compare_horizon_candidate(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_terminal = bool(baseline.get("terminal_state_after_rollout"))
    candidate_terminal = bool(candidate.get("terminal_state_after_rollout"))
    baseline_levels = int(baseline.get("final_levels_completed", 0) or 0)
    candidate_levels = int(candidate.get("final_levels_completed", 0) or 0)
    progress_delta = round(
        float(candidate.get("progress_proxy", 0.0) or 0.0)
        - float(baseline.get("progress_proxy", 0.0) or 0.0),
        4,
    )
    return {
        "condition": str(candidate.get("condition", "")),
        "budget": int(candidate.get("budget", baseline.get("budget", 0)) or 0),
        "tie_break_seed": int(candidate.get("tie_break_seed", baseline.get("tie_break_seed", 0)) or 0),
        "k_objective": candidate.get("k_objective"),
        "k_stop": candidate.get("k_stop"),
        "terminal_horizon_source": str(candidate.get("terminal_horizon_source", "")),
        "estimated_moves_remaining": candidate.get("estimated_moves_remaining"),
        "terminal_horizon_triggered": bool(candidate.get("terminal_horizon_triggered")),
        "trigger_reason": candidate.get("trigger_reason"),
        "objective_mode_entered": bool(candidate.get("objective_mode_entered")),
        "baseline_final_game_state": str(baseline.get("final_game_state", "")),
        "candidate_final_game_state": str(candidate.get("final_game_state", "")),
        "baseline_terminal": baseline_terminal,
        "candidate_terminal": candidate_terminal,
        "terminal_avoidance_signal": bool(baseline_terminal and not candidate_terminal),
        "objective_completion_signal": bool(candidate_levels > baseline_levels),
        "terminal_avoidance_only": bool(baseline_terminal and not candidate_terminal and candidate_levels <= baseline_levels),
        "objective_progress_proxy_delta": progress_delta,
        "terminal_avoidance_counted_as_completion": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def summarize_by_condition_k(comparisons: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups: dict[tuple[str, int | None, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in comparisons:
        groups[(str(row.get("condition", "")), row.get("k_objective"), row.get("k_stop"))].append(row)
    summaries = []
    for (condition, k_objective, k_stop), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summaries.append(
            {
                "condition": condition,
                "k_objective": k_objective,
                "k_stop": k_stop,
                "runs": len(rows),
                "terminal_horizon_triggered_runs": len(
                    [row for row in rows if bool(row.get("terminal_horizon_triggered"))]
                ),
                "objective_mode_entered_runs": len(
                    [row for row in rows if bool(row.get("objective_mode_entered"))]
                ),
                "terminal_avoidance_signal_runs": len(
                    [row for row in rows if bool(row.get("terminal_avoidance_signal"))]
                ),
                "objective_completion_signal_runs": len(
                    [row for row in rows if bool(row.get("objective_completion_signal"))]
                ),
                "terminal_avoidance_only_runs": len(
                    [row for row in rows if bool(row.get("terminal_avoidance_only"))]
                ),
                "mean_objective_progress_proxy_delta": round(
                    sum(float(row.get("objective_progress_proxy_delta", 0.0) or 0.0) for row in rows)
                    / max(1, len(rows)),
                    4,
                ),
                "terminal_avoidance_counted_as_completion": False,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
            }
        )
    return summaries


def run_terminal_horizon_policy_probe(
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
    condition_executor: ConditionExecutor | None = None,
) -> Dict[str, Any]:
    refined_payload = _load_json(refined_window_results_path)
    terminal_signal = validate_refined_window_terminal_signal(refined_payload)
    terminal_budget_estimate = infer_empirical_terminal_budget_estimate(refined_payload)
    scope_payload = _load_json(scope_consolidation_path)
    memory = candidate_policy_memory_from_scope(scope_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    if condition_executor is None:
        _configure_offline_env(env_dir)
        condition_executor = default_condition_executor

    run_results: list[Dict[str, Any]] = []
    comparisons: list[Dict[str, Any]] = []
    for budget in budgets:
        for seed in tie_break_seeds:
            baseline = dict(
                condition_executor(
                    BASELINE_POLICY,
                    int(budget),
                    int(seed),
                    terminal_budget_estimate,
                    0,
                    0,
                    memory,
                    env_dir,
                    game_id,
                )
            )
            for k in [int(value) for value in k_values]:
                stop_summary = dict(
                    condition_executor(
                        STOP_AT_HORIZON_POLICY,
                        int(budget),
                        int(seed),
                        terminal_budget_estimate,
                        k,
                        k,
                        memory,
                        env_dir,
                        game_id,
                    )
                )
                objective_summary = dict(
                    condition_executor(
                        OBJECTIVE_MODE_POLICY,
                        int(budget),
                        int(seed),
                        terminal_budget_estimate,
                        k,
                        1,
                        memory,
                        env_dir,
                        game_id,
                    )
                )
                stop_comparison = compare_horizon_candidate(
                    baseline=baseline,
                    candidate=stop_summary,
                )
                objective_comparison = compare_horizon_candidate(
                    baseline=baseline,
                    candidate=objective_summary,
                )
                run_results.extend(
                    [
                        {
                            "budget": int(budget),
                            "tie_break_seed": int(seed),
                            "k_value": k,
                            "baseline": baseline,
                            "candidate": stop_summary,
                            "comparison": stop_comparison,
                            "support": 0,
                            "revision_status": "CANDIDATE_ONLY",
                            "truth_status": TRUTH_STATUS,
                        },
                        {
                            "budget": int(budget),
                            "tie_break_seed": int(seed),
                            "k_value": k,
                            "baseline": baseline,
                            "candidate": objective_summary,
                            "comparison": objective_comparison,
                            "support": 0,
                            "revision_status": "CANDIDATE_ONLY",
                            "truth_status": TRUTH_STATUS,
                        },
                    ]
                )
                comparisons.extend([stop_comparison, objective_comparison])

    condition_k_summaries = summarize_by_condition_k(comparisons)
    summary = {
        "terminal_horizon_estimator_integrated": True,
        "terminal_horizon_source": "empirical_fallback",
        "terminal_budget_estimate": terminal_budget_estimate,
        "k_values_tested": [int(value) for value in k_values],
        "budgets_tested": [int(value) for value in budgets],
        "tie_break_seeds_tested": [int(value) for value in tie_break_seeds],
        "baseline_runs": len(budgets) * len(tie_break_seeds),
        "candidate_policy_runs": len(comparisons),
        "stop_guard_runs": len([row for row in comparisons if row.get("condition") == STOP_AT_HORIZON_POLICY]),
        "objective_mode_runs": len([row for row in comparisons if row.get("condition") == OBJECTIVE_MODE_POLICY]),
        "terminal_avoidance_signal_runs": len(
            [row for row in comparisons if bool(row.get("terminal_avoidance_signal"))]
        ),
        "objective_completion_signal_runs": len(
            [row for row in comparisons if bool(row.get("objective_completion_signal"))]
        ),
        "objective_mode_entered_runs": len(
            [row for row in comparisons if bool(row.get("objective_mode_entered"))]
        ),
        "terminal_avoidance_only_runs": len(
            [row for row in comparisons if bool(row.get("terminal_avoidance_only"))]
        ),
        "terminal_avoidance_counted_as_completion": False,
        "candidate_policy_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a33_ready": False,
    }
    return {
        "config": {
            "schema_version": P3_TERMINAL_HORIZON_SCHEMA_VERSION,
            "refined_window_results_path": str(refined_window_results_path),
            "scope_consolidation_path": str(scope_consolidation_path),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "baseline_policy": BASELINE_POLICY,
            "candidate_policies": [STOP_AT_HORIZON_POLICY, OBJECTIVE_MODE_POLICY],
            "inputs_read": ["M3.O4", "M3.24"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33"],
        },
        "source_terminal_signal": terminal_signal,
        "terminal_horizon_estimator": {
            "observer": "TerminalHorizonObserver",
            "source": "empirical_fallback",
            "fusion_priority": [
                "environment_metadata",
                "hud_bar",
                "empirical_fallback",
                "unknown",
            ],
            "terminal_budget_estimate": terminal_budget_estimate,
            "moves_used_variable": "env_actions_executed",
            "action6_prefix_count_used_as_decision_variable": False,
            "terminal_fraction_remaining_available": True,
            "evidence_tracking_enabled": True,
            "hud_bar_detector_available": True,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        },
        "candidate_policy_memory": memory.to_dict(),
        "run_results": run_results,
        "comparisons": comparisons,
        "condition_k_summaries": condition_k_summaries,
        "summary": summary,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": TRUTH_STATUS,
        "execution_performed": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "terminal_avoidance_counted_as_completion": False,
        "candidate_policy_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a33_ready": False,
    }


def write_terminal_horizon_policy_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P3_TERMINAL_HORIZON_POLICY_PROBE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P3.2 terminal-horizon objective mode policy probe.",
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
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P3_TERMINAL_HORIZON_POLICY_PROBE_OUTPUT_PATH,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_terminal_horizon_policy_probe(
        refined_window_results_path=args.refined_window_results,
        scope_consolidation_path=args.scope_consolidation,
        environments_dir=args.environments_dir,
        budgets=tuple(args.budgets or DEFAULT_P3_HORIZON_BUDGETS),
        tie_break_seeds=tuple(args.tie_break_seeds or DEFAULT_TIE_BREAK_SEEDS),
        k_values=tuple(args.k_values or DEFAULT_K_VALUES),
        game_id=args.game_id,
    )
    write_terminal_horizon_policy_probe(payload, args.out)


if __name__ == "__main__":
    main()
