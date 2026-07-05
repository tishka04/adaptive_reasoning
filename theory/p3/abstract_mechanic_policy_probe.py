"""P3.G0 abstract-mechanic candidate policy probe."""

from __future__ import annotations

import argparse
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
from theory.p3.terminal_horizon_estimator import estimate_terminal_horizon
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_SYMBOLIC_MODEL_PATH = (
    Path("diagnostics") / "m3" / "generic_candidate_symbolic_mechanism_model.json"
)
DEFAULT_ABSTRACT_POLICY_ADAPTER_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "abstract_mechanic_policy_adapter.json"
)
DEFAULT_ABSTRACT_POLICY_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "abstract_mechanic_policy_probe.json"
)
DEFAULT_ABSTRACT_POLICY_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "p3" / "abstract_mechanic_policy_utility_consolidation.json"
)

TRUTH_STATUS = "NOT_EVALUATED_BY_P3_AGENT_PROBE"
ABSTRACT_POLICY_SCHEMA_VERSION = "p3.abstract_mechanic_policy_probe.v1"
DEFAULT_BUDGETS = (8, 16)
DEFAULT_TIE_BREAK_SEEDS = (0, 1, 2)

RANDOM_AVAILABLE_POLICY = "baseline_random_available_action"
GREEDY_CHANGED_PIXELS_POLICY = "baseline_greedy_changed_pixels"
TERMINAL_HORIZON_GUARD_POLICY = "baseline_terminal_horizon_guard"
ABSTRACT_MECHANIC_POLICY = "abstract_mechanic_policy_from_m3g0"
P3G0_CONDITIONS = (
    RANDOM_AVAILABLE_POLICY,
    GREEDY_CHANGED_PIXELS_POLICY,
    TERMINAL_HORIZON_GUARD_POLICY,
    ABSTRACT_MECHANIC_POLICY,
)
STOP_ACTION = "__STOP__"

POLICY_USEFUL_CANDIDATE_ONLY = "POLICY_USEFUL_CANDIDATE_ONLY"
POLICY_NEUTRAL_CANDIDATE_ONLY = "POLICY_NEUTRAL_CANDIDATE_ONLY"
POLICY_HARMFUL_CANDIDATE_ONLY = "POLICY_HARMFUL_CANDIDATE_ONLY"


def build_abstract_mechanic_policy_adapter(
    *,
    symbolic_model_path: str | Path = DEFAULT_SYMBOLIC_MODEL_PATH,
) -> Dict[str, Any]:
    payload = _load_json(symbolic_model_path)
    validate_symbolic_model_source(payload)
    model = dict(payload.get("candidate_symbolic_model", {}) or {})
    action_models = dict(model.get("action_models", {}) or {})
    relation_effects = list(
        (model.get("relation_model", {}) or {}).get("actor_relation_effects", [])
        or []
    )
    actor_entities = [
        str(row.get("entity_id", ""))
        for row in model.get("actor_candidates", []) or []
        if row.get("entity_id")
    ]
    action_candidates = []
    for action_name, row in sorted(action_models.items()):
        effects = [str(value) for value in row.get("candidate_effects", []) or []]
        relations = [str(value) for value in row.get("relation_effects", []) or []]
        action_candidates.append(
            {
                "action": str(action_name),
                "candidate_effects": effects,
                "relation_effects": relations,
                "model_status": str(row.get("status", "CANDIDATE_ONLY")),
                "score_weights": {
                    "relation_progress_score": 3.0 if "distance_decreases" in relations else 0.0,
                    "actor_effect_score": 1.0 * len(effects),
                    "novelty_score": 1.0,
                    "repetition_penalty": -0.75,
                    "terminal_risk_penalty": -999.0,
                },
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
            }
        )
    adapter = {
        "policy_adapter_id": "p3g0_1::bp35::abstract_mechanic_policy_adapter",
        "source_symbolic_model_path": str(symbolic_model_path),
        "source_symbolic_model_id": str(model.get("candidate_symbolic_model_id", "")),
        "adapter_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        "actor_candidates": actor_entities,
        "action_candidates": action_candidates,
        "relation_targets": sorted(
            {
                str(row.get("target_entity", ""))
                for row in relation_effects
                if row.get("target_entity")
            }
        ),
        "ignored_or_caveated_entities": [
            {
                "entity_id": str(row.get("entity_id", "")),
                "family": str(row.get("family", "")),
                "reason": "symbolic_model_caveat_not_policy_signal",
                "semantic_interpretation": "unknown",
            }
            for row in model.get("caveats", []) or []
        ],
        "dynamic_invariants_observed_not_semantic": list(
            sorted((model.get("dynamic_invariants", {}) or {}).keys())
        ),
        "policy_scoring": {
            "relation_progress_score": "candidate action has distance_decreases relation effect",
            "actor_effect_score": "candidate action affects actor candidate",
            "novelty_score": "lower action count preferred",
            "repetition_penalty": "repeated action penalized but not banned",
            "terminal_risk_penalty": "estimated terminal horizon may stop risky action",
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
            "symbolic_model_path": str(symbolic_model_path),
            "schema_version": ABSTRACT_POLICY_SCHEMA_VERSION,
            "stage": "P3.G0.1",
            "execution_performed": False,
        },
        "summary": {
            "actor_candidates": len(actor_entities),
            "action_candidates": len(action_candidates),
            "relation_targets": len(adapter["relation_targets"]),
            "ignored_or_caveated_entities": len(adapter["ignored_or_caveated_entities"]),
            "ready_for_abstract_policy_probe": bool(actor_entities and action_candidates),
            "candidate_model_counted_as_confirmed_mechanic": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "policy_adapter": adapter,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_adapter_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def run_abstract_mechanic_policy_rollout(
    *,
    policy_adapter_path: str | Path = DEFAULT_ABSTRACT_POLICY_ADAPTER_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    tie_break_seeds: Sequence[int] = DEFAULT_TIE_BREAK_SEEDS,
    game_id: str = DEFAULT_GAME_ID,
    condition_executor: Callable[[str, int, int, Mapping[str, Any], Path, str], Mapping[str, Any]]
    | None = None,
) -> Dict[str, Any]:
    adapter_payload = _load_json(policy_adapter_path)
    validate_policy_adapter_source(adapter_payload)
    adapter = dict(adapter_payload.get("policy_adapter", {}) or {})
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    condition_summaries: list[Dict[str, Any]] = []
    rollout_traces: list[Dict[str, Any]] = []
    if condition_executor is None:

        def default_condition_executor(
            condition: str,
            budget: int,
            seed: int,
            policy_adapter: Mapping[str, Any],
            env_path: Path,
            active_game_id: str,
        ) -> Mapping[str, Any]:
            steps, stop_event = execute_abstract_policy_condition(
                condition=condition,
                adapter=policy_adapter,
                budget=budget,
                tie_break_seed=seed,
                environments_dir=env_path,
                game_id=active_game_id,
            )
            return summarize_abstract_policy_steps(
                condition=condition,
                steps=steps,
                budget=budget,
                tie_break_seed=seed,
                stop_event=stop_event,
            )

        condition_executor = default_condition_executor

    for budget in budgets:
        for seed in tie_break_seeds:
            for condition in P3G0_CONDITIONS:
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
                condition_summaries.append(summary)
                rollout_traces.append(
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
    aggregate = aggregate_policy_summaries(condition_summaries)
    return {
        "config": {
            "policy_adapter_path": str(policy_adapter_path),
            "schema_version": ABSTRACT_POLICY_SCHEMA_VERSION,
            "stage": "P3.G0.2",
            "conditions": list(P3G0_CONDITIONS),
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
        "condition_summaries": condition_summaries,
        "rollout_traces": rollout_traces,
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


def consolidate_abstract_mechanic_policy_utility(
    *,
    rollout_path: str | Path = DEFAULT_ABSTRACT_POLICY_PROBE_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(rollout_path)
    validate_policy_rollout_source(payload)
    aggregate = dict(payload.get("summary", {}) or {})
    condition_aggregates = dict(aggregate.get("condition_aggregates", {}) or {})
    candidate = dict(condition_aggregates.get(ABSTRACT_MECHANIC_POLICY, {}) or {})
    baselines = [
        dict(condition_aggregates.get(condition, {}) or {})
        for condition in P3G0_CONDITIONS
        if condition != ABSTRACT_MECHANIC_POLICY
    ]
    candidate_progress = float(candidate.get("mean_progress_proxy", 0.0) or 0.0)
    baseline_best_progress = max(
        [float(row.get("mean_progress_proxy", 0.0) or 0.0) for row in baselines] or [0.0]
    )
    candidate_relation = float(candidate.get("mean_actor_relation_delta_count", 0.0) or 0.0)
    baseline_best_relation = max(
        [float(row.get("mean_actor_relation_delta_count", 0.0) or 0.0) for row in baselines] or [0.0]
    )
    useful = bool(
        candidate_progress > baseline_best_progress
        or candidate_relation > baseline_best_relation
    )
    harmful = bool(
        candidate_progress < baseline_best_progress
        and candidate_relation < baseline_best_relation
    )
    status = (
        POLICY_USEFUL_CANDIDATE_ONLY
        if useful
        else (POLICY_HARMFUL_CANDIDATE_ONLY if harmful else POLICY_NEUTRAL_CANDIDATE_ONLY)
    )
    return {
        "config": {
            "rollout_path": str(rollout_path),
            "schema_version": ABSTRACT_POLICY_SCHEMA_VERSION,
            "stage": "P3.G0.3",
            "execution_performed": False,
        },
        "summary": {
            "policy_utility_status": status,
            "candidate_mean_progress_proxy": candidate_progress,
            "best_baseline_mean_progress_proxy": baseline_best_progress,
            "candidate_mean_actor_relation_delta_count": candidate_relation,
            "best_baseline_mean_actor_relation_delta_count": baseline_best_relation,
            "candidate_beats_best_baseline_progress": candidate_progress > baseline_best_progress,
            "candidate_beats_best_baseline_relation_metric": candidate_relation > baseline_best_relation,
            "policy_result_counted_as_scientific_verdict": False,
            "candidate_model_counted_as_confirmed_mechanic": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "condition_aggregates": condition_aggregates,
        "policy_utility_record": {
            "policy": ABSTRACT_MECHANIC_POLICY,
            "policy_utility_status": status,
            "model_source": str((payload.get("config", {}) or {}).get("policy_adapter_path", "")),
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


def execute_abstract_policy_condition(
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
        decision = select_abstract_probe_decision(
            condition=condition,
            adapter=adapter,
            valid_actions=valid_actions,
            action_counts=action_counts,
            action_history=tuple(action_history),
            tie_break_seed=tie_break_seed,
            horizon_estimate=horizon.to_dict(),
            current_frame=before,
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
        useful_new_state = bool(changed_pixels > 0 and not cycle and after.levels_completed >= before.levels_completed)
        step = {
            "step": step_index,
            "condition": condition,
            "policy_selected_action": action_name,
            "action_args": action_args,
            "decision_reason": decision.decision_reason,
            "candidate_policy_used": decision.candidate_policy_used,
            "candidate_score": decision.candidate_score,
            "candidate_score_details": dict(decision.candidate_score_details),
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


def select_abstract_probe_decision(
    *,
    condition: str,
    adapter: Mapping[str, Any],
    valid_actions: Sequence[Any],
    action_counts: Mapping[str, int],
    action_history: Sequence[Mapping[str, Any]],
    tie_break_seed: int,
    horizon_estimate: Mapping[str, Any],
    current_frame: Any,
    environments_dir: Path,
    game_id: str,
) -> ProbeDecision:
    if condition == GREEDY_CHANGED_PIXELS_POLICY:
        return select_greedy_changed_pixels_decision(
            valid_actions=valid_actions,
            action_history=action_history,
            environments_dir=environments_dir,
            game_id=game_id,
            tie_break_seed=tie_break_seed,
        )
    if condition == TERMINAL_HORIZON_GUARD_POLICY:
        remaining = horizon_estimate.get("estimated_moves_remaining")
        if remaining is not None and int(remaining) <= 1:
            return ProbeDecision(
                condition=condition,
                action_name=STOP_ACTION,
                decision_reason="terminal_horizon_guard_stop",
                candidate_policy_used=True,
                candidate_score_details={
                    "estimated_moves_remaining": int(remaining),
                    "terminal_horizon_source": str(horizon_estimate.get("source", "")),
                },
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
    return select_random_available_decision(
        valid_actions=valid_actions,
        action_counts=action_counts,
        condition=condition,
        tie_break_seed=tie_break_seed,
    )


def select_random_available_decision(
    *,
    valid_actions: Sequence[Any],
    action_counts: Mapping[str, int],
    condition: str,
    tie_break_seed: int,
) -> ProbeDecision:
    candidates = sorted(
        {str(getattr(action, "name", "")) for action in valid_actions if getattr(action, "name", "")}
    )
    if not candidates:
        return ProbeDecision(condition=condition, action_name="", decision_reason="no_available_action")
    selected = min(
        candidates,
        key=lambda name: (int(action_counts.get(name, 0) or 0), (candidates.index(name) + tie_break_seed) % len(candidates)),
    )
    args = deterministic_action_args(valid_actions, selected, tie_break_seed)
    return ProbeDecision(
        condition=condition,
        action_name=selected,
        action_args=args,
        decision_reason="deterministic_random_available_action",
    )


def select_abstract_model_decision(
    *,
    adapter: Mapping[str, Any],
    valid_actions: Sequence[Any],
    action_counts: Mapping[str, int],
    tie_break_seed: int,
    horizon_estimate: Mapping[str, Any],
) -> ProbeDecision:
    remaining = horizon_estimate.get("estimated_moves_remaining")
    if remaining is not None and int(remaining) <= 1:
        return ProbeDecision(
            condition=ABSTRACT_MECHANIC_POLICY,
            action_name=STOP_ACTION,
            decision_reason="abstract_policy_terminal_risk_guard",
            candidate_policy_used=True,
            candidate_score_details={
                "estimated_moves_remaining": int(remaining),
                "terminal_horizon_source": str(horizon_estimate.get("source", "")),
            },
        )
    valid_names = {str(getattr(action, "name", "")) for action in valid_actions}
    scored: list[Dict[str, Any]] = []
    for candidate in adapter.get("action_candidates", []) or []:
        action = str(candidate.get("action", ""))
        if action not in valid_names:
            continue
        weights = dict(candidate.get("score_weights", {}) or {})
        score = (
            float(weights.get("relation_progress_score", 0.0) or 0.0)
            + float(weights.get("actor_effect_score", 0.0) or 0.0)
            + float(weights.get("novelty_score", 0.0) or 0.0)
            + float(weights.get("repetition_penalty", 0.0) or 0.0)
            * int(action_counts.get(action, 0) or 0)
        )
        scored.append(
            {
                "action": action,
                "score": round(score, 4),
                "weights": weights,
                "candidate_effects": list(candidate.get("candidate_effects", []) or []),
                "relation_effects": list(candidate.get("relation_effects", []) or []),
            }
        )
    if scored:
        best = max(
            scored,
            key=lambda row: (
                float(row.get("score", 0.0) or 0.0),
                -deterministic_name_index(str(row.get("action", "")), tie_break_seed),
            ),
        )
        return ProbeDecision(
            condition=ABSTRACT_MECHANIC_POLICY,
            action_name=str(best["action"]),
            decision_reason="abstract_model_relation_progress_action",
            candidate_policy_used=True,
            candidate_score=float(best["score"]),
            candidate_score_details=best,
        )
    return select_random_available_decision(
        valid_actions=valid_actions,
        action_counts=action_counts,
        condition=ABSTRACT_MECHANIC_POLICY,
        tie_break_seed=tie_break_seed,
    )


def select_greedy_changed_pixels_decision(
    *,
    valid_actions: Sequence[Any],
    action_history: Sequence[Mapping[str, Any]],
    environments_dir: Path,
    game_id: str,
    tie_break_seed: int,
) -> ProbeDecision:
    scored: list[Dict[str, Any]] = []
    for action in valid_actions:
        name = str(getattr(action, "name", ""))
        if not name:
            continue
        args = dict(getattr(action, "action_args", {}) or {})
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
        decision_reason="greedy_changed_pixels_one_step_lookahead",
        candidate_score=float(best["changed_pixels"]),
        candidate_score_details=best,
    )


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
    return float(np.count_nonzero(before_grid != after_grid)) if before_grid.shape == after_grid.shape else float(max(before_grid.size, after_grid.size))


def summarize_abstract_policy_steps(
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
    model_action_steps = sum(1 for row in steps if action_has_model_entry(row.get("policy_selected_action", ""), condition))
    progress_proxy = best_level * 100.0 + useful_states * 5.0 + actor_relation * 3.0 + action_effect * 2.0 - cycles
    return {
        "condition": condition,
        "budget": int(budget),
        "tie_break_seed": int(tie_break_seed),
        "policy_steps": len(steps),
        "final_levels_completed": int(final.get("levels_after", 0) or 0),
        "best_level_reached": int(best_level),
        "final_game_state": str(final.get("game_state_after", "NOT_STARTED")),
        "terminal_state_after_rollout": terminal,
        "steps_survived": len(steps),
        "terminal_avoidance": not terminal,
        "progress_proxy": round(progress_proxy, 4),
        "best_progress_proxy": round(progress_proxy, 4),
        "actor_relation_delta_count": int(actor_relation),
        "distance_decreases_count": int(actor_relation),
        "new_relation_states": sum(int(row.get("new_relation_state", 0) or 0) for row in steps),
        "action_effect_usefulness": int(action_effect),
        "stale_action_rate": round(float(cycles) / max(1, len(steps)), 4),
        "candidate_model_action_steps": int(model_action_steps),
        "stop_event": dict(stop_event),
        "policy_result_counted_as_scientific_verdict": False,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def aggregate_policy_summaries(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition", ""))].append(row)
    aggregates: Dict[str, Dict[str, Any]] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        aggregates[condition] = aggregate_condition_rows(condition, condition_rows)
    candidate = aggregates.get(ABSTRACT_MECHANIC_POLICY, {})
    baselines = [
        aggregates.get(condition, {})
        for condition in P3G0_CONDITIONS
        if condition != ABSTRACT_MECHANIC_POLICY
    ]
    best_baseline_progress = max(
        [float(row.get("mean_progress_proxy", 0.0) or 0.0) for row in baselines] or [0.0]
    )
    best_baseline_relation = max(
        [float(row.get("mean_actor_relation_delta_count", 0.0) or 0.0) for row in baselines] or [0.0]
    )
    return {
        "rollout_runs": len(rows),
        "conditions": list(sorted(by_condition)),
        "condition_aggregates": aggregates,
        "candidate_mean_progress_proxy": float(candidate.get("mean_progress_proxy", 0.0) or 0.0),
        "best_baseline_mean_progress_proxy": best_baseline_progress,
        "candidate_mean_actor_relation_delta_count": float(candidate.get("mean_actor_relation_delta_count", 0.0) or 0.0),
        "best_baseline_mean_actor_relation_delta_count": best_baseline_relation,
        "candidate_beats_best_baseline_progress": float(candidate.get("mean_progress_proxy", 0.0) or 0.0) > best_baseline_progress,
        "candidate_beats_best_baseline_relation_metric": float(candidate.get("mean_actor_relation_delta_count", 0.0) or 0.0) > best_baseline_relation,
    }


def aggregate_condition_rows(condition: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "condition": condition,
        "runs": len(rows),
        "mean_progress_proxy": round(mean(row.get("progress_proxy", 0.0) for row in rows), 4),
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


def validate_symbolic_model_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    model = dict(payload.get("candidate_symbolic_model", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G0.7 support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.7 summary support must remain 0")
    if str(model.get("model_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("M3.G0.7 model must remain candidate-only")
    if bool(payload.get("model_counted_as_confirmation", False)) or bool(model.get("model_counted_as_confirmation", False)):
        raise ValueError("candidate model cannot be confirmation")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("M3.G0.7 cannot write A32/A33")
    if not bool(summary.get("ready_for_policy_probe_candidate_only", False)):
        raise ValueError("M3.G0.7 model is not ready for policy probe")


def validate_policy_adapter_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G0.1 adapter support must remain 0")
    if bool(payload.get("policy_adapter_counted_as_confirmation", False)):
        raise ValueError("adapter cannot be confirmation")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("P3.G0.1 cannot write A32/A33")
    if not bool(summary.get("ready_for_abstract_policy_probe", False)):
        raise ValueError("adapter is not ready for abstract policy probe")


def validate_policy_rollout_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("P3.G0.2 rollout support must remain 0")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")
    if bool(payload.get("candidate_model_counted_as_confirmed_mechanic", False)):
        raise ValueError("candidate model cannot be confirmed mechanic")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("P3.G0.2 cannot write A32/A33")


def action_has_relation_effect(adapter: Mapping[str, Any], action_name: str) -> bool:
    for candidate in adapter.get("action_candidates", []) or []:
        if str(candidate.get("action", "")) == str(action_name):
            return "distance_decreases" in {
                str(value) for value in candidate.get("relation_effects", []) or []
            }
    return False


def action_has_actor_effect(adapter: Mapping[str, Any], action_name: str) -> bool:
    for candidate in adapter.get("action_candidates", []) or []:
        if str(candidate.get("action", "")) == str(action_name):
            return bool(candidate.get("candidate_effects", []) or [])
    return False


def action_has_model_entry(action_name: Any, condition: str) -> bool:
    return condition == ABSTRACT_MECHANIC_POLICY and str(action_name) in {"ACTION3", "ACTION4"}


def deterministic_action_args(valid_actions: Sequence[Any], action_name: str, seed: int) -> Dict[str, Any]:
    matches = [
        dict(getattr(action, "action_args", {}) or {})
        for action in valid_actions
        if str(getattr(action, "name", "")) == str(action_name)
    ]
    if not matches:
        return {}
    ordered = sorted(matches, key=lambda row: json.dumps(row, sort_keys=True))
    return dict(ordered[int(seed) % len(ordered)])


def concrete_named_action(
    valid_actions: Sequence[Any],
    action_name: str,
    action_args: Mapping[str, Any],
) -> Any | None:
    decision = ProbeDecision(
        condition="replay",
        action_name=action_name,
        action_args=dict(action_args),
    )
    return concrete_action_for_decision(valid_actions, decision)


def deterministic_name_index(name: str, seed: int) -> int:
    names = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
    return (names.index(name) if name in names else len(names)) + int(seed)


def deterministic_args_index(args: Mapping[str, Any], seed: int) -> int:
    encoded = json.dumps(dict(args), sort_keys=True)
    return sum(ord(ch) for ch in encoded) + int(seed)


def is_game_over(game_state: Any) -> bool:
    return str(game_state or "").upper() == "GAME_OVER"


def mean(values: Sequence[Any]) -> float:
    vals = [float(value or 0.0) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def write_json(payload: Mapping[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P3.G0 abstract mechanic policy probe.")
    parser.add_argument("--stage", choices=("adapter", "rollout", "consolidation", "all"), default="all")
    parser.add_argument("--symbolic-model", default=str(DEFAULT_SYMBOLIC_MODEL_PATH))
    parser.add_argument("--adapter-out", default=str(DEFAULT_ABSTRACT_POLICY_ADAPTER_OUTPUT_PATH))
    parser.add_argument("--adapter", default=str(DEFAULT_ABSTRACT_POLICY_ADAPTER_OUTPUT_PATH))
    parser.add_argument("--rollout-out", default=str(DEFAULT_ABSTRACT_POLICY_PROBE_OUTPUT_PATH))
    parser.add_argument("--rollout", default=str(DEFAULT_ABSTRACT_POLICY_PROBE_OUTPUT_PATH))
    parser.add_argument("--consolidation-out", default=str(DEFAULT_ABSTRACT_POLICY_CONSOLIDATION_OUTPUT_PATH))
    parser.add_argument("--budgets", type=int, nargs="*", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_TIE_BREAK_SEEDS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage in {"adapter", "all"}:
        adapter_payload = build_abstract_mechanic_policy_adapter(
            symbolic_model_path=args.symbolic_model
        )
        write_json(adapter_payload, args.adapter_out)
    if args.stage in {"rollout", "all"}:
        rollout_payload = run_abstract_mechanic_policy_rollout(
            policy_adapter_path=args.adapter if args.stage == "rollout" else args.adapter_out,
            budgets=tuple(args.budgets or DEFAULT_BUDGETS),
            tie_break_seeds=tuple(args.seeds or DEFAULT_TIE_BREAK_SEEDS),
        )
        write_json(rollout_payload, args.rollout_out)
    if args.stage in {"consolidation", "all"}:
        consolidation_payload = consolidate_abstract_mechanic_policy_utility(
            rollout_path=args.rollout if args.stage == "consolidation" else args.rollout_out
        )
        write_json(consolidation_payload, args.consolidation_out)
        print(
            json.dumps(
                {
                    "adapter_out": args.adapter_out,
                    "rollout_out": args.rollout_out,
                    "consolidation_out": args.consolidation_out,
                    "policy_utility_status": consolidation_payload["summary"][
                        "policy_utility_status"
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
