"""P3.G1 objective-aware abstract mechanic policy probe."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.p1.bp35_sage_candidate_policy_probe import (
    DEFAULT_GAME_ID,
    ProbeDecision,
    concrete_action_for_decision,
    measure_probe_metrics,
    state_signature,
)
from theory.p3.abstract_mechanic_policy_probe import (
    ABSTRACT_MECHANIC_POLICY,
    DEFAULT_ABSTRACT_POLICY_ADAPTER_OUTPUT_PATH,
    DEFAULT_SYMBOLIC_MODEL_PATH,
    GREEDY_CHANGED_PIXELS_POLICY,
    RANDOM_AVAILABLE_POLICY,
    STOP_ACTION,
    TERMINAL_HORIZON_GUARD_POLICY,
    TRUTH_STATUS,
    action_has_actor_effect,
    action_has_relation_effect,
    concrete_named_action,
    deterministic_action_args,
    deterministic_name_index,
    is_game_over,
    mean,
    select_abstract_model_decision,
    select_random_available_decision,
    validate_policy_adapter_source,
    validate_symbolic_model_source,
    write_json,
)
from theory.p3.terminal_horizon_estimator import estimate_terminal_horizon
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_HUD_POLICY_PROBE_PATH = (
    Path("diagnostics") / "p3" / "bp35_terminal_horizon_hud_policy_probe.json"
)
DEFAULT_P3G0_POLICY_PROBE_PATH = (
    Path("diagnostics") / "p3" / "abstract_mechanic_policy_probe.json"
)
DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "objective_aware_abstract_policy_adapter.json"
)
DEFAULT_OBJECTIVE_AWARE_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "objective_aware_abstract_policy_probe.json"
)
DEFAULT_OBJECTIVE_AWARE_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "objective_aware_abstract_policy_utility_consolidation.json"
)

P3G1_SCHEMA_VERSION = "p3.objective_aware_abstract_policy_probe.v1"
DEFAULT_BUDGETS = (8, 16, 32, 64)
DEFAULT_TIE_BREAK_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TERMINAL_RISK_LAMBDAS = (0.0, 1.0, 3.0, 10.0, 30.0)
DEFAULT_TERMINAL_RISK_WINDOW = 8
DEFAULT_K_STOP = 1

OBJECTIVE_AWARE_POLICY_PREFIX = "objective_aware_abstract_policy_lambda_"
POLICY_OBJECTIVE_USEFUL_CANDIDATE_ONLY = "POLICY_OBJECTIVE_USEFUL_CANDIDATE_ONLY"
POLICY_PROGRESS_ONLY_CANDIDATE_ONLY = "POLICY_PROGRESS_ONLY_CANDIDATE_ONLY"
POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY = (
    "POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY"
)
POLICY_HARMFUL_CANDIDATE_ONLY = "POLICY_HARMFUL_CANDIDATE_ONLY"

BASELINE_CONDITIONS = (
    RANDOM_AVAILABLE_POLICY,
    GREEDY_CHANGED_PIXELS_POLICY,
    TERMINAL_HORIZON_GUARD_POLICY,
    ABSTRACT_MECHANIC_POLICY,
)


def build_objective_aware_abstract_policy_adapter(
    *,
    abstract_adapter_path: str | Path = DEFAULT_ABSTRACT_POLICY_ADAPTER_OUTPUT_PATH,
    hud_policy_probe_path: str | Path = DEFAULT_HUD_POLICY_PROBE_PATH,
    symbolic_model_path: str | Path = DEFAULT_SYMBOLIC_MODEL_PATH,
    terminal_risk_lambdas: Sequence[float] = DEFAULT_TERMINAL_RISK_LAMBDAS,
    terminal_risk_window: int = DEFAULT_TERMINAL_RISK_WINDOW,
    k_stop: int = DEFAULT_K_STOP,
) -> Dict[str, Any]:
    abstract_payload = _load_json(abstract_adapter_path)
    validate_policy_adapter_source(abstract_payload)
    hud_payload = _load_json(hud_policy_probe_path)
    validate_hud_policy_source(hud_payload)
    symbolic_payload = _load_json(symbolic_model_path)
    validate_symbolic_model_source(symbolic_payload)

    abstract_adapter = dict(abstract_payload.get("policy_adapter", {}) or {})
    lambda_values = [float(value) for value in terminal_risk_lambdas]
    policy_variants = [
        {
            "condition": objective_condition_name(value),
            "lambda_terminal_risk": float(value),
            "terminal_risk_window": int(terminal_risk_window),
            "k_stop": int(k_stop),
            "score_formula": (
                "3*relation_progress + 2*actor_effect + novelty + "
                "2*objective_progress_proxy - lambda_terminal_risk*terminal_risk "
                "- 0.75*repetition"
            ),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        }
        for value in lambda_values
    ]
    adapter = {
        "policy_adapter_id": "p3g1_1::bp35::objective_aware_abstract_policy_adapter",
        "source_abstract_policy_adapter_path": str(abstract_adapter_path),
        "source_hud_policy_probe_path": str(hud_policy_probe_path),
        "source_symbolic_model_path": str(symbolic_model_path),
        "adapter_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        "actor_candidates": list(abstract_adapter.get("actor_candidates", []) or []),
        "action_candidates": list(abstract_adapter.get("action_candidates", []) or []),
        "relation_targets": list(abstract_adapter.get("relation_targets", []) or []),
        "ignored_or_caveated_entities": list(
            abstract_adapter.get("ignored_or_caveated_entities", []) or []
        ),
        "dynamic_invariants_observed_not_semantic": list(
            abstract_adapter.get("dynamic_invariants_observed_not_semantic", []) or []
        ),
        "terminal_horizon_source": "hud_bar_with_empirical_warmup_fallback",
        "terminal_horizon_source_counted_as_confirmation": False,
        "lambda_terminal_risk_values": lambda_values,
        "terminal_risk_window": int(terminal_risk_window),
        "k_stop": int(k_stop),
        "policy_variants": policy_variants,
        "objective_aware_scoring": {
            "relation_progress_weight": 3.0,
            "actor_effect_weight": 2.0,
            "novelty_weight": 1.0,
            "objective_progress_proxy_weight": 2.0,
            "terminal_risk_weight": "lambda_terminal_risk",
            "repetition_penalty": -0.75,
            "terminal_adjusted_progress_metric": "progress_proxy if not terminal else 0",
        },
        "candidate_model_counted_as_confirmed_mechanic": False,
        "policy_adapter_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    return {
        "config": {
            "schema_version": P3G1_SCHEMA_VERSION,
            "stage": "P3.G1.1",
            "abstract_adapter_path": str(abstract_adapter_path),
            "hud_policy_probe_path": str(hud_policy_probe_path),
            "symbolic_model_path": str(symbolic_model_path),
            "execution_performed": False,
        },
        "summary": {
            "actor_candidates": len(adapter["actor_candidates"]),
            "action_candidates": len(adapter["action_candidates"]),
            "relation_targets": len(adapter["relation_targets"]),
            "lambda_terminal_risk_values": lambda_values,
            "policy_variants": len(policy_variants),
            "ready_for_objective_aware_policy_probe": bool(
                adapter["actor_candidates"] and adapter["action_candidates"]
            ),
            "candidate_model_counted_as_confirmed_mechanic": False,
            "policy_adapter_counted_as_confirmation": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "objective_aware_policy_adapter": adapter,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_adapter_counted_as_confirmation": False,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def run_objective_aware_abstract_policy_rollout(
    *,
    objective_adapter_path: str | Path = DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    tie_break_seeds: Sequence[int] = DEFAULT_TIE_BREAK_SEEDS,
    game_id: str = DEFAULT_GAME_ID,
    condition_executor: Callable[[str, int, int, Mapping[str, Any], Path, str], Mapping[str, Any]]
    | None = None,
) -> Dict[str, Any]:
    payload = _load_json(objective_adapter_path)
    validate_objective_adapter_source(payload)
    adapter = dict(payload.get("objective_aware_policy_adapter", {}) or {})
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    conditions = list(BASELINE_CONDITIONS) + [
        str(row["condition"]) for row in adapter.get("policy_variants", []) or []
    ]
    if condition_executor is None:

        def default_condition_executor(
            condition: str,
            budget: int,
            seed: int,
            policy_adapter: Mapping[str, Any],
            env_path: Path,
            active_game_id: str,
        ) -> Mapping[str, Any]:
            steps, stop_event = execute_objective_aware_condition(
                condition=condition,
                adapter=policy_adapter,
                budget=budget,
                tie_break_seed=seed,
                environments_dir=env_path,
                game_id=active_game_id,
            )
            return summarize_objective_aware_steps(
                condition=condition,
                steps=steps,
                budget=budget,
                tie_break_seed=seed,
                stop_event=stop_event,
            )

        condition_executor = default_condition_executor

    summaries: list[Dict[str, Any]] = []
    traces: list[Dict[str, Any]] = []
    for budget in budgets:
        for seed in tie_break_seeds:
            for condition in conditions:
                summary = dict(
                    condition_executor(
                        condition,
                        int(budget),
                        int(seed),
                        adapter,
                        env_dir,
                        game_id,
                    )
                )
                summaries.append(summary)
                traces.append(
                    {
                        "condition": condition,
                        "budget": int(budget),
                        "tie_break_seed": int(seed),
                        "summary": summary,
                        "support": 0,
                        "revision_status": "CANDIDATE_ONLY",
                        "truth_status": TRUTH_STATUS,
                    }
                )
    aggregate = aggregate_objective_policy_summaries(
        summaries,
        objective_conditions=[str(row["condition"]) for row in adapter.get("policy_variants", []) or []],
    )
    return {
        "config": {
            "schema_version": P3G1_SCHEMA_VERSION,
            "stage": "P3.G1.2",
            "objective_adapter_path": str(objective_adapter_path),
            "conditions": conditions,
            "budgets": [int(value) for value in budgets],
            "tie_break_seeds": [int(value) for value in tie_break_seeds],
            "game_id": game_id,
            "environments_dir": str(env_dir),
        },
        "summary": {
            **aggregate,
            "policy_result_counted_as_scientific_verdict": False,
            "candidate_model_counted_as_confirmed_mechanic": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "condition_summaries": summaries,
        "rollout_traces": traces,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def consolidate_objective_aware_abstract_policy_utility(
    *,
    rollout_path: str | Path = DEFAULT_OBJECTIVE_AWARE_PROBE_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(rollout_path)
    validate_objective_rollout_source(payload)
    summary = dict(payload.get("summary", {}) or {})
    condition_aggregates = dict(summary.get("condition_aggregates", {}) or {})
    g0 = dict(condition_aggregates.get(ABSTRACT_MECHANIC_POLICY, {}) or {})
    objective_conditions = [
        condition
        for condition in condition_aggregates
        if str(condition).startswith(OBJECTIVE_AWARE_POLICY_PREFIX)
    ]
    objective_rows = [dict(condition_aggregates[condition]) for condition in objective_conditions]
    best_objective = max(
        objective_rows,
        key=lambda row: (
            float(row.get("mean_terminal_adjusted_progress", 0.0) or 0.0),
            -float(row.get("terminal_rate", 1.0) or 1.0),
            float(row.get("mean_progress_proxy", 0.0) or 0.0),
        ),
        default={},
    )
    g0_progress = float(g0.get("mean_progress_proxy", 0.0) or 0.0)
    g0_terminal = float(g0.get("terminal_rate", 0.0) or 0.0)
    candidate_progress = float(best_objective.get("mean_progress_proxy", 0.0) or 0.0)
    candidate_terminal = float(best_objective.get("terminal_rate", 0.0) or 0.0)
    candidate_adjusted = float(best_objective.get("mean_terminal_adjusted_progress", 0.0) or 0.0)
    g0_adjusted = float(g0.get("mean_terminal_adjusted_progress", 0.0) or 0.0)
    candidate_levels = float(best_objective.get("mean_levels_completed", 0.0) or 0.0)
    progress_kept = candidate_progress >= g0_progress
    terminal_reduced = candidate_terminal < g0_terminal
    objective_completion = candidate_levels > float(g0.get("mean_levels_completed", 0.0) or 0.0)
    if objective_completion or (progress_kept and terminal_reduced):
        status = POLICY_OBJECTIVE_USEFUL_CANDIDATE_ONLY
    elif candidate_adjusted > g0_adjusted and candidate_terminal <= g0_terminal:
        status = POLICY_TERMINAL_SAFE_BUT_PASSIVE_CANDIDATE_ONLY
    elif candidate_progress >= g0_progress and candidate_terminal >= g0_terminal:
        status = POLICY_PROGRESS_ONLY_CANDIDATE_ONLY
    else:
        status = POLICY_HARMFUL_CANDIDATE_ONLY
    return {
        "config": {
            "schema_version": P3G1_SCHEMA_VERSION,
            "stage": "P3.G1.3",
            "rollout_path": str(rollout_path),
            "execution_performed": False,
        },
        "summary": {
            "policy_utility_status": status,
            "best_objective_aware_condition": str(best_objective.get("condition", "")),
            "best_objective_aware_lambda_terminal_risk": best_objective.get(
                "lambda_terminal_risk"
            ),
            "p3g0_mean_progress_proxy": g0_progress,
            "best_objective_mean_progress_proxy": candidate_progress,
            "p3g0_terminal_rate": g0_terminal,
            "best_objective_terminal_rate": candidate_terminal,
            "p3g0_terminal_adjusted_progress": g0_adjusted,
            "best_objective_terminal_adjusted_progress": candidate_adjusted,
            "objective_completion_signal": objective_completion,
            "progress_proxy_preserved_vs_p3g0": progress_kept,
            "terminal_rate_reduced_vs_p3g0": terminal_reduced,
            "policy_result_counted_as_scientific_verdict": False,
            "candidate_model_counted_as_confirmed_mechanic": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "condition_aggregates": condition_aggregates,
        "objective_aware_policy_utility_record": {
            "policy_utility_status": status,
            "best_objective_aware_condition": str(best_objective.get("condition", "")),
            "policy_result_counted_as_scientific_verdict": False,
            "candidate_model_counted_as_confirmed_mechanic": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def execute_objective_aware_condition(
    *,
    condition: str,
    adapter: Mapping[str, Any],
    budget: int,
    tie_break_seed: int,
    environments_dir: str | Path,
    game_id: str,
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    env = _make_env(game_id, environments_dir)
    current_frame = _reset_env(env)
    steps: list[Dict[str, Any]] = []
    action_history: list[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    seen_states: set[str] = set()
    grid_history: list[Any] = []
    initial = snapshot_frame(current_frame)
    grid_history.append(initial.grid)
    seen_states.add(state_signature(initial.grid, initial.levels_completed, initial.game_state))
    stop_event = {
        "stop_triggered": False,
        "trigger_reason": "",
        "terminal_horizon_source": "",
        "estimated_moves_remaining": None,
        "lambda_terminal_risk": lambda_from_condition(condition),
        "terminal_risk_window": int(adapter.get("terminal_risk_window", DEFAULT_TERMINAL_RISK_WINDOW) or DEFAULT_TERMINAL_RISK_WINDOW),
    }
    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(current_frame)
        if is_game_over(before.game_state):
            stop_event["trigger_reason"] = "terminal_state_reached_before_next_action"
            break
        valid_actions = list(_valid_actions(env))
        horizon = estimate_terminal_horizon(
            observation=before.grid,
            history=grid_history[:-1],
            policy_state={"env_actions_executed": len(action_history)},
            terminal_budget_estimate=64,
        )
        decision = select_objective_aware_decision(
            condition=condition,
            adapter=adapter,
            env=env,
            current_frame=before,
            valid_actions=valid_actions,
            action_counts=action_counts,
            action_history=tuple(action_history),
            tie_break_seed=tie_break_seed,
            horizon_estimate=horizon.to_dict(),
            environments_dir=Path(environments_dir),
            game_id=game_id,
        )
        if decision.action_name == STOP_ACTION:
            stop_event.update(
                {
                    "stop_triggered": True,
                    "trigger_reason": decision.decision_reason,
                    "terminal_horizon_source": decision.candidate_score_details.get(
                        "terminal_horizon_source", ""
                    ),
                    "estimated_moves_remaining": decision.candidate_score_details.get(
                        "estimated_moves_remaining"
                    ),
                }
            )
            break
        selected = concrete_action_for_decision(valid_actions, decision)
        if selected is None:
            steps.append(
                error_policy_step(
                    step_index=step_index,
                    condition=condition,
                    decision=decision,
                    before=before,
                    error="selected_action_not_available",
                )
            )
            break
        after_frame = _step_env_action(env, selected)
        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        action_name = str(getattr(selected, "name", decision.action_name))
        action_args = dict(getattr(selected, "action_args", {}) or {})
        measurements = measure_probe_metrics(before.grid, after.grid, action_args)
        after_signature = state_signature(after.grid, after.levels_completed, after.game_state)
        cycle = after_signature in seen_states
        changed_pixels = float(measurements["changed_pixels"].get("changed_pixels", 0) or 0)
        relation_expected = action_has_relation_effect(adapter, action_name)
        actor_effect_expected = action_has_actor_effect(adapter, action_name)
        useful_new_state = bool(
            changed_pixels > 0
            and not cycle
            and after.levels_completed >= before.levels_completed
        )
        step = {
            "step": step_index,
            "condition": condition,
            "policy_selected_action": action_name,
            "action_args": action_args,
            "decision_reason": decision.decision_reason,
            "candidate_policy_used": decision.candidate_policy_used,
            "candidate_score": decision.candidate_score,
            "candidate_score_details": dict(decision.candidate_score_details),
            "terminal_horizon_source": horizon.source,
            "estimated_moves_remaining": horizon.estimated_moves_remaining,
            "changed_pixels": changed_pixels,
            "actor_effect_expected": actor_effect_expected,
            "relation_delta_expected": relation_expected,
            "actor_relation_delta_count": int(relation_expected and changed_pixels > 0),
            "action_effect_usefulness": int(actor_effect_expected and changed_pixels > 0),
            "new_relation_state": int(relation_expected and useful_new_state),
            "useful_new_state": useful_new_state,
            "dead_end_or_cycle": cycle,
            "state_signature_before": state_signature(
                before.grid, before.levels_completed, before.game_state
            ),
            "state_signature_after": after_signature,
            "levels_before": int(before.levels_completed),
            "levels_after": int(after.levels_completed),
            "game_state_before": str(before.game_state),
            "game_state_after": str(after.game_state),
            "terminal_state_after": is_game_over(after.game_state),
            "measurements": measurements,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "wrong_confirmations": 0,
        }
        steps.append(step)
        seen_states.add(after_signature)
        action_counts[action_name] += 1
        action_history.append({"action": action_name, "action_args": action_args})
        current_frame = after_frame
        grid_history.append(after.grid)
    return tuple(steps), stop_event


def select_objective_aware_decision(
    *,
    condition: str,
    adapter: Mapping[str, Any],
    env: Any,
    current_frame: Any,
    valid_actions: Sequence[Any],
    action_counts: Mapping[str, int],
    action_history: Sequence[Mapping[str, Any]],
    tie_break_seed: int,
    horizon_estimate: Mapping[str, Any],
    environments_dir: Path,
    game_id: str,
) -> ProbeDecision:
    if condition == GREEDY_CHANGED_PIXELS_POLICY:
        return select_fast_greedy_changed_pixels_decision(
            env=env,
            current_frame=current_frame,
            valid_actions=valid_actions,
            action_history=action_history,
            environments_dir=environments_dir,
            game_id=game_id,
            tie_break_seed=tie_break_seed,
        )
    if condition == TERMINAL_HORIZON_GUARD_POLICY:
        remaining = horizon_estimate.get("estimated_moves_remaining")
        if remaining is not None and int(remaining) <= int(adapter.get("k_stop", DEFAULT_K_STOP) or DEFAULT_K_STOP):
            return stop_decision(
                condition=condition,
                reason="terminal_horizon_guard_stop",
                horizon_estimate=horizon_estimate,
            )
        return select_random_available_decision(
            valid_actions=valid_actions,
            action_counts=action_counts,
            condition=condition,
            tie_break_seed=tie_break_seed,
        )
    if condition == ABSTRACT_MECHANIC_POLICY:
        return select_abstract_model_decision(
            adapter=adapter,
            valid_actions=valid_actions,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
            horizon_estimate=horizon_estimate,
        )
    if condition.startswith(OBJECTIVE_AWARE_POLICY_PREFIX):
        return select_objective_aware_model_decision(
            condition=condition,
            adapter=adapter,
            valid_actions=valid_actions,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
            horizon_estimate=horizon_estimate,
            env=env,
            current_frame=current_frame,
        )
    return select_random_available_decision(
        valid_actions=valid_actions,
        action_counts=action_counts,
        condition=condition,
        tie_break_seed=tie_break_seed,
    )


def select_objective_aware_model_decision(
    *,
    condition: str,
    adapter: Mapping[str, Any],
    valid_actions: Sequence[Any],
    action_counts: Mapping[str, int],
    tie_break_seed: int,
    horizon_estimate: Mapping[str, Any],
    env: Any,
    current_frame: Any,
) -> ProbeDecision:
    remaining = horizon_estimate.get("estimated_moves_remaining")
    k_stop = int(adapter.get("k_stop", DEFAULT_K_STOP) or DEFAULT_K_STOP)
    if remaining is not None and int(remaining) <= k_stop:
        return stop_decision(
            condition=condition,
            reason="objective_aware_terminal_stop_guard",
            horizon_estimate=horizon_estimate,
        )
    lambda_terminal_risk = lambda_from_condition(condition)
    risk_window = int(adapter.get("terminal_risk_window", DEFAULT_TERMINAL_RISK_WINDOW) or DEFAULT_TERMINAL_RISK_WINDOW)
    valid_names = {str(getattr(action, "name", "")) for action in valid_actions}
    scored: list[Dict[str, Any]] = []
    for candidate in adapter.get("action_candidates", []) or []:
        action = str(candidate.get("action", ""))
        if action not in valid_names:
            continue
        projected = projected_action_delta(
            env=env,
            current_frame=current_frame,
            valid_actions=valid_actions,
            action_name=action,
            action_args=deterministic_action_args(valid_actions, action, tie_break_seed),
        )
        relation_feature = 1.0 if "distance_decreases" in set(candidate.get("relation_effects", []) or []) else 0.0
        actor_feature = 1.0 if candidate.get("candidate_effects", []) else 0.0
        novelty_feature = 1.0 / (1.0 + float(action_counts.get(action, 0) or 0))
        objective_proxy = 1.0 if projected.get("changed_pixels", 0.0) > 0 and not projected.get("terminal", False) else 0.0
        terminal_risk = terminal_risk_score(
            estimated_moves_remaining=remaining,
            risk_window=risk_window,
            projected_terminal=bool(projected.get("terminal", False)),
        )
        repetition = float(action_counts.get(action, 0) or 0)
        score = (
            3.0 * relation_feature
            + 2.0 * actor_feature
            + novelty_feature
            + 2.0 * objective_proxy
            - lambda_terminal_risk * terminal_risk
            - 0.75 * repetition
        )
        scored.append(
            {
                "action": action,
                "action_args": deterministic_action_args(valid_actions, action, tie_break_seed),
                "score": round(score, 4),
                "lambda_terminal_risk": lambda_terminal_risk,
                "terminal_risk_score": round(terminal_risk, 4),
                "relation_progress_score": relation_feature,
                "actor_effect_score": actor_feature,
                "novelty_score": round(novelty_feature, 4),
                "objective_progress_proxy": objective_proxy,
                "repetition_count": repetition,
                "projected_changed_pixels": projected.get("changed_pixels"),
                "projected_terminal": bool(projected.get("terminal", False)),
            }
        )
    if not scored:
        return select_random_available_decision(
            valid_actions=valid_actions,
            action_counts=action_counts,
            condition=condition,
            tie_break_seed=tie_break_seed,
        )
    best = max(
        scored,
        key=lambda row: (
            float(row.get("score", 0.0) or 0.0),
            -deterministic_name_index(str(row.get("action", "")), tie_break_seed),
        ),
    )
    if float(best.get("score", 0.0) or 0.0) < 0.0:
        return stop_decision(
            condition=condition,
            reason="objective_aware_terminal_risk_score_stop",
            horizon_estimate=horizon_estimate,
            details={"best_action_score": best},
        )
    return ProbeDecision(
        condition=condition,
        action_name=str(best["action"]),
        action_args=dict(best.get("action_args", {}) or {}),
        decision_reason="objective_aware_relation_progress_with_terminal_risk",
        candidate_policy_used=True,
        candidate_score=float(best["score"]),
        candidate_score_details=best,
    )


def select_fast_greedy_changed_pixels_decision(
    *,
    env: Any,
    current_frame: Any,
    valid_actions: Sequence[Any],
    action_history: Sequence[Mapping[str, Any]],
    environments_dir: Path,
    game_id: str,
    tie_break_seed: int,
) -> ProbeDecision:
    before = as_snapshot(current_frame)
    scored: list[Dict[str, Any]] = []
    for action in valid_actions:
        name = str(getattr(action, "name", ""))
        if not name:
            continue
        args = dict(getattr(action, "action_args", {}) or {})
        changed = one_step_changed_pixels_from_env_copy(
            env=env,
            before_grid=before.grid,
            action=action,
        )
        if changed is None:
            changed = one_step_changed_pixels_after_replay(
                action_name=name,
                action_args=args,
                action_history=action_history,
                environments_dir=environments_dir,
                game_id=game_id,
            )
        scored.append({"action": name, "action_args": args, "changed_pixels": changed})
    if not scored:
        return ProbeDecision(
            condition=GREEDY_CHANGED_PIXELS_POLICY,
            action_name="",
            decision_reason="greedy_changed_pixels_no_available_action",
        )
    best = max(
        scored,
        key=lambda row: (
            float(row.get("changed_pixels", 0.0) or 0.0),
            -deterministic_name_index(str(row.get("action", "")), tie_break_seed),
            -deterministic_args_index(dict(row.get("action_args", {}) or {}), tie_break_seed),
        ),
    )
    return ProbeDecision(
        condition=GREEDY_CHANGED_PIXELS_POLICY,
        action_name=str(best["action"]),
        action_args=dict(best["action_args"]),
        decision_reason="greedy_changed_pixels_one_step_lookahead_fast_copy",
        candidate_score=float(best["changed_pixels"]),
        candidate_score_details=best,
    )


def summarize_objective_aware_steps(
    *,
    condition: str,
    steps: Sequence[Mapping[str, Any]],
    budget: int,
    tie_break_seed: int,
    stop_event: Mapping[str, Any],
) -> Dict[str, Any]:
    final = dict(steps[-1]) if steps else {}
    terminal = bool(final.get("terminal_state_after", False))
    levels = [int(row.get("levels_after", 0) or 0) for row in steps]
    best_level = max(levels or [0])
    useful_states = sum(1 for row in steps if bool(row.get("useful_new_state")))
    cycles = sum(1 for row in steps if bool(row.get("dead_end_or_cycle")))
    actor_relation = sum(int(row.get("actor_relation_delta_count", 0) or 0) for row in steps)
    action_effect = sum(int(row.get("action_effect_usefulness", 0) or 0) for row in steps)
    progress_proxy = (
        best_level * 100.0
        + useful_states * 5.0
        + actor_relation * 3.0
        + action_effect * 2.0
        - cycles
    )
    terminal_adjusted = 0.0 if terminal else progress_proxy
    return {
        "condition": condition,
        "budget": int(budget),
        "tie_break_seed": int(tie_break_seed),
        "lambda_terminal_risk": lambda_from_condition(condition),
        "policy_steps": len(steps),
        "final_levels_completed": int(final.get("levels_after", 0) or 0),
        "best_level_reached": int(best_level),
        "final_game_state": str(final.get("game_state_after", "NOT_STARTED")),
        "terminal_state_after_rollout": terminal,
        "steps_survived": len(steps),
        "terminal_avoidance": not terminal,
        "progress_proxy": round(progress_proxy, 4),
        "terminal_adjusted_progress": round(terminal_adjusted, 4),
        "actor_relation_delta_count": int(actor_relation),
        "distance_decreases_count": int(actor_relation),
        "new_relation_states": sum(int(row.get("new_relation_state", 0) or 0) for row in steps),
        "action_effect_usefulness": int(action_effect),
        "stale_action_rate": round(float(cycles) / max(1, len(steps)), 4),
        "stop_event": dict(stop_event),
        "policy_result_counted_as_scientific_verdict": False,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def aggregate_objective_policy_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    objective_conditions: Sequence[str],
) -> Dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition", ""))].append(row)
    aggregates = {
        condition: aggregate_condition_rows(condition, condition_rows)
        for condition, condition_rows in sorted(by_condition.items())
    }
    g0 = aggregates.get(ABSTRACT_MECHANIC_POLICY, {})
    objective_rows = [aggregates.get(condition, {}) for condition in objective_conditions]
    best_objective = max(
        objective_rows,
        key=lambda row: (
            float(row.get("mean_terminal_adjusted_progress", 0.0) or 0.0),
            -float(row.get("terminal_rate", 1.0) or 1.0),
            float(row.get("mean_progress_proxy", 0.0) or 0.0),
        ),
        default={},
    )
    return {
        "rollout_runs": len(rows),
        "conditions": list(sorted(by_condition)),
        "objective_aware_conditions": list(objective_conditions),
        "condition_aggregates": aggregates,
        "p3g0_mean_progress_proxy": float(g0.get("mean_progress_proxy", 0.0) or 0.0),
        "p3g0_terminal_rate": float(g0.get("terminal_rate", 0.0) or 0.0),
        "p3g0_terminal_adjusted_progress": float(
            g0.get("mean_terminal_adjusted_progress", 0.0) or 0.0
        ),
        "best_objective_aware_condition": str(best_objective.get("condition", "")),
        "best_objective_aware_lambda_terminal_risk": best_objective.get(
            "lambda_terminal_risk"
        ),
        "best_objective_mean_progress_proxy": float(
            best_objective.get("mean_progress_proxy", 0.0) or 0.0
        ),
        "best_objective_terminal_rate": float(best_objective.get("terminal_rate", 0.0) or 0.0),
        "best_objective_terminal_adjusted_progress": float(
            best_objective.get("mean_terminal_adjusted_progress", 0.0) or 0.0
        ),
        "objective_aware_progress_preserved_vs_p3g0": float(
            best_objective.get("mean_progress_proxy", 0.0) or 0.0
        )
        >= float(g0.get("mean_progress_proxy", 0.0) or 0.0),
        "objective_aware_terminal_rate_reduced_vs_p3g0": float(
            best_objective.get("terminal_rate", 0.0) or 0.0
        )
        < float(g0.get("terminal_rate", 0.0) or 0.0),
    }


def aggregate_condition_rows(condition: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "condition": condition,
        "runs": len(rows),
        "lambda_terminal_risk": lambda_from_condition(condition),
        "mean_progress_proxy": round(mean(row.get("progress_proxy", 0.0) for row in rows), 4),
        "mean_terminal_adjusted_progress": round(
            mean(row.get("terminal_adjusted_progress", 0.0) for row in rows),
            4,
        ),
        "mean_levels_completed": round(mean(row.get("final_levels_completed", 0) for row in rows), 4),
        "terminal_rate": round(mean(1.0 if row.get("terminal_state_after_rollout") else 0.0 for row in rows), 4),
        "mean_steps_survived": round(mean(row.get("steps_survived", 0) for row in rows), 4),
        "mean_actor_relation_delta_count": round(mean(row.get("actor_relation_delta_count", 0) for row in rows), 4),
        "mean_action_effect_usefulness": round(mean(row.get("action_effect_usefulness", 0) for row in rows), 4),
        "mean_stale_action_rate": round(mean(row.get("stale_action_rate", 0.0) for row in rows), 4),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def projected_action_delta(
    *,
    env: Any,
    current_frame: Any,
    valid_actions: Sequence[Any],
    action_name: str,
    action_args: Mapping[str, Any],
) -> Dict[str, Any]:
    before = as_snapshot(current_frame)
    selected = concrete_named_action(valid_actions, action_name, action_args)
    if selected is None:
        return {"changed_pixels": -1.0, "terminal": False}
    projected = one_step_projection_from_env_copy(
        env=env,
        before_grid=before.grid,
        action=selected,
    )
    if projected is None:
        return {"changed_pixels": 0.0, "terminal": False}
    return projected


def one_step_changed_pixels_from_env_copy(
    *,
    env: Any,
    before_grid: Any,
    action: Any,
) -> float | None:
    projected = one_step_projection_from_env_copy(
        env=env,
        before_grid=before_grid,
        action=action,
    )
    if projected is None:
        return None
    return float(projected.get("changed_pixels", 0.0) or 0.0)


def one_step_projection_from_env_copy(
    *,
    env: Any,
    before_grid: Any,
    action: Any,
) -> Dict[str, Any] | None:
    try:
        cloned = copy.deepcopy(env)
        after_frame = _step_env_action(cloned, action)
        after = snapshot_frame(after_frame)
    except Exception:
        return None
    before = np.asarray(before_grid)
    after_grid = np.asarray(after.grid)
    if before.shape != after_grid.shape:
        changed_pixels = float(max(before.size, after_grid.size))
    else:
        changed_pixels = float(np.count_nonzero(before != after_grid))
    return {
        "changed_pixels": changed_pixels,
        "terminal": is_game_over(after.game_state),
        "game_state_after": str(after.game_state),
        "levels_after": int(after.levels_completed),
    }


def as_snapshot(frame_or_snapshot: Any) -> Any:
    if hasattr(frame_or_snapshot, "grid") and hasattr(frame_or_snapshot, "game_state"):
        return frame_or_snapshot
    return snapshot_frame(frame_or_snapshot)


def one_step_changed_pixels_after_replay(
    *,
    action_name: str,
    action_args: Mapping[str, Any],
    action_history: Sequence[Mapping[str, Any]],
    environments_dir: Path,
    game_id: str,
) -> float:
    env = _make_env(game_id, environments_dir)
    current = _reset_env(env)
    for row in action_history:
        selected = concrete_named_action(
            _valid_actions(env),
            str(row.get("action", "")),
            dict(row.get("action_args", {}) or {}),
        )
        if selected is None:
            return -1.0
        current = _step_env_action(env, selected)
    before = snapshot_frame(current)
    selected_action = concrete_named_action(_valid_actions(env), action_name, action_args)
    if selected_action is None:
        return -1.0
    after_frame = _step_env_action(env, selected_action)
    after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
    before_grid = np.asarray(before.grid)
    after_grid = np.asarray(after.grid)
    if before_grid.shape != after_grid.shape:
        return float(max(before_grid.size, after_grid.size))
    return float(np.count_nonzero(before_grid != after_grid))


def terminal_risk_score(
    *,
    estimated_moves_remaining: Any,
    risk_window: int,
    projected_terminal: bool,
) -> float:
    if projected_terminal:
        return 1.0
    if estimated_moves_remaining is None:
        return 0.0
    remaining = max(0, int(estimated_moves_remaining))
    window = max(1, int(risk_window))
    if remaining > window:
        return 0.0
    return float(window - remaining + 1) / float(window)


def stop_decision(
    *,
    condition: str,
    reason: str,
    horizon_estimate: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
) -> ProbeDecision:
    return ProbeDecision(
        condition=condition,
        action_name=STOP_ACTION,
        decision_reason=reason,
        candidate_policy_used=True,
        candidate_score_details={
            "estimated_moves_remaining": horizon_estimate.get("estimated_moves_remaining"),
            "terminal_horizon_source": str(horizon_estimate.get("source", "")),
            **dict(details or {}),
        },
    )


def error_policy_step(
    *,
    step_index: int,
    condition: str,
    decision: ProbeDecision,
    before: Any,
    error: str,
) -> Dict[str, Any]:
    return {
        "step": int(step_index),
        "condition": condition,
        "policy_selected_action": str(decision.action_name),
        "action_args": dict(decision.action_args),
        "decision_reason": str(decision.decision_reason),
        "error": error,
        "levels_before": int(before.levels_completed),
        "levels_after": int(before.levels_completed),
        "game_state_before": str(before.game_state),
        "game_state_after": str(before.game_state),
        "terminal_state_after": is_game_over(before.game_state),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def lambda_from_condition(condition: str) -> float | None:
    if not str(condition).startswith(OBJECTIVE_AWARE_POLICY_PREFIX):
        return None
    raw = str(condition)[len(OBJECTIVE_AWARE_POLICY_PREFIX) :].replace("_", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def objective_condition_name(value: float) -> str:
    if float(value).is_integer():
        encoded = str(int(value))
    else:
        encoded = str(float(value)).replace(".", "_")
    return f"{OBJECTIVE_AWARE_POLICY_PREFIX}{encoded}"


def deterministic_args_index(args: Mapping[str, Any], seed: int) -> int:
    encoded = json.dumps(dict(args), sort_keys=True)
    return sum(ord(ch) for ch in encoded) + int(seed)


def validate_hud_policy_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.2b support must remain 0")
    if bool(payload.get("candidate_policy_counted_as_confirmation", False)):
        raise ValueError("P3.2b policy cannot be confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("P3.2b policy result cannot be scientific verdict")
    if summary.get("action6_prefix_count_used_as_decision_variable") is not False:
        raise ValueError("P3.2b must use terminal horizon, not ACTION6 prefix count")


def validate_objective_adapter_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G1 adapter support must remain 0")
    if bool(payload.get("policy_adapter_counted_as_confirmation", False)):
        raise ValueError("P3.G1 adapter cannot be confirmation")
    if bool(payload.get("candidate_model_counted_as_confirmed_mechanic", False)):
        raise ValueError("P3.G1 candidate model cannot be confirmed")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("P3.G1 cannot write A32/A33")
    if not bool(summary.get("ready_for_objective_aware_policy_probe", False)):
        raise ValueError("P3.G1 adapter is not ready")


def validate_objective_rollout_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G1 rollout support must remain 0")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("P3.G1 rollout cannot be verdict")
    if bool(payload.get("candidate_model_counted_as_confirmed_mechanic", False)):
        raise ValueError("P3.G1 model cannot be confirmed")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("P3.G1 cannot write A32/A33")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P3.G1 objective-aware abstract policy probe.")
    parser.add_argument("--stage", choices=("adapter", "rollout", "consolidation", "all"), default="all")
    parser.add_argument("--abstract-adapter", default=str(DEFAULT_ABSTRACT_POLICY_ADAPTER_OUTPUT_PATH))
    parser.add_argument("--hud-policy-probe", default=str(DEFAULT_HUD_POLICY_PROBE_PATH))
    parser.add_argument("--symbolic-model", default=str(DEFAULT_SYMBOLIC_MODEL_PATH))
    parser.add_argument("--adapter-out", default=str(DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH))
    parser.add_argument("--adapter", default=str(DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH))
    parser.add_argument("--rollout-out", default=str(DEFAULT_OBJECTIVE_AWARE_PROBE_OUTPUT_PATH))
    parser.add_argument("--rollout", default=str(DEFAULT_OBJECTIVE_AWARE_PROBE_OUTPUT_PATH))
    parser.add_argument("--consolidation-out", default=str(DEFAULT_OBJECTIVE_AWARE_CONSOLIDATION_OUTPUT_PATH))
    parser.add_argument("--budgets", type=int, nargs="*", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_TIE_BREAK_SEEDS))
    parser.add_argument("--lambdas", type=float, nargs="*", default=list(DEFAULT_TERMINAL_RISK_LAMBDAS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage in {"adapter", "all"}:
        adapter = build_objective_aware_abstract_policy_adapter(
            abstract_adapter_path=args.abstract_adapter,
            hud_policy_probe_path=args.hud_policy_probe,
            symbolic_model_path=args.symbolic_model,
            terminal_risk_lambdas=tuple(args.lambdas or DEFAULT_TERMINAL_RISK_LAMBDAS),
        )
        write_json(adapter, args.adapter_out)
    if args.stage in {"rollout", "all"}:
        rollout = run_objective_aware_abstract_policy_rollout(
            objective_adapter_path=args.adapter if args.stage == "rollout" else args.adapter_out,
            budgets=tuple(args.budgets or DEFAULT_BUDGETS),
            tie_break_seeds=tuple(args.seeds or DEFAULT_TIE_BREAK_SEEDS),
        )
        write_json(rollout, args.rollout_out)
    if args.stage in {"consolidation", "all"}:
        consolidation = consolidate_objective_aware_abstract_policy_utility(
            rollout_path=args.rollout if args.stage == "consolidation" else args.rollout_out
        )
        write_json(consolidation, args.consolidation_out)
        print(
            json.dumps(
                {
                    "adapter_out": args.adapter_out,
                    "rollout_out": args.rollout_out,
                    "consolidation_out": args.consolidation_out,
                    "policy_utility_status": consolidation["summary"]["policy_utility_status"],
                    "best_objective_aware_condition": consolidation["summary"][
                        "best_objective_aware_condition"
                    ],
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
