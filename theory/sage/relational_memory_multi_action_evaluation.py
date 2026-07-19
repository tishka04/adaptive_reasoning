"""SAGE.8c paired multi-action conversion test for relational memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.a34.control_dependent_relational_usage_probe import (
    DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH,
)
from theory.a34.parameterized_relational_usage_probe import (
    DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
)
from theory.m1.polymorphic_a25_adapter import (
    _step_env_action,
    measure_required_observation,
)
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir
from theory.real_env_option_adapter import snapshot_frame

from .live_mini_frontier_m3_executor import (
    EnvFactory,
    _make_real_env,
    _measurement_for_delta,
)
from .live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from .relational_memory_ab_evaluation import (
    DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
    SAGE8B_LOCAL_ONLY_GAIN,
    SAGE8B_SCHEMA_VERSION,
    SAGE8B_TRUTH_STATUS,
    build_sage8b_evaluation_specifications,
    summarize_primary_metrics,
    summarize_secondary_metrics,
    validate_sage8b_sources,
)
from .relational_memory_policy import (
    DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
    LOWER_EFFECT_COMPARATOR_MATCH,
    PolicyActionOption,
    apply_relational_memory_policy,
)


DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage8c_relational_memory_multi_action_evaluation.json"
)

SAGE8C_SCHEMA_VERSION = "sage.relational_memory_multi_action_evaluation.v1"
SAGE8C_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8C"
SAGE8C_ARC_GAIN = "SAGE_RELATIONAL_MEMORY_MULTI_ACTION_ARC_SCORE_GAIN_OBSERVED"
SAGE8C_LOCAL_ONLY = (
    "SAGE_RELATIONAL_MEMORY_MULTI_ACTION_LOCAL_GAIN_WITHOUT_ARC_SCORE_CONVERSION"
)
SAGE8C_NO_GAIN = "SAGE_RELATIONAL_MEMORY_MULTI_ACTION_NO_GAIN_OBSERVED"
SAGE8C_ARC_REGRESSION = (
    "SAGE_RELATIONAL_MEMORY_MULTI_ACTION_ARC_SCORE_REGRESSION_OBSERVED"
)

DEFAULT_CONTINUATION_HORIZON = 16
DEFAULT_CONTINUATION_TAIL_WINDOW = 16
CONTINUATION_POLICY = "CYCLIC_RECENT_PREFIX_TAIL_OUTCOME_BLIND"


def run_sage8c_relational_memory_multi_action_evaluation(
    *,
    sage8b_path: str | Path = DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
    policy_path: str | Path = DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
    a34_2_path: str | Path = (
        DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH
    ),
    a34_3_path: str | Path = DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
    environments_dir: str | Path | None = None,
    continuation_horizon: int = DEFAULT_CONTINUATION_HORIZON,
    continuation_tail_window: int = DEFAULT_CONTINUATION_TAIL_WINDOW,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Measure whether one memory decision converts over a fixed continuation."""
    source_sage8b = _load_json(sage8b_path)
    policy_source = _load_json(policy_path)
    a34_2 = _load_json(a34_2_path)
    a34_3 = _load_json(a34_3_path)
    validate_sage8c_sources(source_sage8b, policy_source, a34_2, a34_3)
    _validate_rollout_config(continuation_horizon, continuation_tail_window)
    base_specifications = build_sage8b_evaluation_specifications(
        policy_source, a34_2, a34_3
    )
    specifications = tuple(
        build_sage8c_rollout_specification(
            row,
            continuation_horizon=continuation_horizon,
            continuation_tail_window=continuation_tail_window,
        )
        for row in base_specifications
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    episodes = tuple(
        execute_sage8c_paired_rollout(
            specification,
            policy=policy_source["policy"],
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        for specification in specifications
    )
    primary_metrics = summarize_primary_metrics(episodes)
    secondary_metrics = summarize_secondary_metrics(episodes)
    rollout_metrics = summarize_rollout_metrics(episodes)
    gate = build_sage8c_gate(
        specifications,
        episodes,
        primary_metrics,
        continuation_horizon=continuation_horizon,
    )
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8c multi-action evaluation gate did not pass")
    summary = summarize_sage8c(
        episodes,
        primary_metrics,
        secondary_metrics,
        rollout_metrics,
        gate,
    )
    payload = {
        "config": {
            "schema_version": SAGE8C_SCHEMA_VERSION,
            "sage8b_path": str(sage8b_path),
            "policy_path": str(policy_path),
            "a34_2_path": str(a34_2_path),
            "a34_3_path": str(a34_3_path),
            "environments_dir": str(env_dir),
            "continuation_horizon": int(continuation_horizon),
            "continuation_tail_window": int(continuation_tail_window),
            "continuation_policy": CONTINUATION_POLICY,
            "inputs_read": [
                "SAGE.8b_EVALUATION",
                "SAGE.8a_POLICY",
                "A34.2_REPLAYS",
                "A34.3_REPLAYS",
            ],
            "evaluation_design": {
                "paired_exact_context_replays": True,
                "same_prefix_between_arms": True,
                "same_continuation_schedule_between_arms": True,
                "only_initial_decision_differs": True,
                "with_memory_initial_action_uses_sage8a_policy": True,
                "continuation_derived_only_from_pre_decision_action_history": True,
                "continuation_selected_before_rollout_execution": True,
                "outcome_fields_read_for_continuation_selection": [],
                "all_eleven_registered_contexts_required": True,
                "primary_metrics": ["levels_completed", "win_rate"],
                "secondary_metrics": ["initial_local_patch_before_after"],
                "local_signal_gain_is_not_arc_score_gain": True,
                "early_stop_only_on_terminal_state": True,
            },
            "artifacts_not_modified": [
                "A33.3",
                "A33.4",
                "A34.2",
                "A34.3",
                "SAGE.8a",
                "SAGE.8b",
            ],
        },
        "rollout_specifications": [
            _public_specification(row) for row in specifications
        ],
        "paired_rollouts": [dict(row) for row in episodes],
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "rollout_metrics": rollout_metrics,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "EVALUATED",
        "truth_status": SAGE8C_TRUTH_STATUS,
        "comparative_evaluation_performed": True,
        "multi_action_live_rollout_performed": True,
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "support": 0,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "scope_generalization_performed": False,
        "levels_completed": primary_metrics["levels_completed"][
            "with_memory_max_after"
        ],
        "win_rate": primary_metrics["win_rate"]["with_memory"],
        "primary_arc_progress_improved": summary["primary_arc_progress_improved"],
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_sage8c_relational_memory_multi_action_evaluation(payload, output_path)
    return payload


def build_sage8c_rollout_specification(
    base: Mapping[str, Any],
    *,
    continuation_horizon: int,
    continuation_tail_window: int,
) -> Dict[str, Any]:
    """Freeze an outcome-blind shared continuation from past action history."""
    request = dict(base.get("request", {}) or {})
    names = [str(value) for value in request.get("context_replay", []) or []]
    raw_args = list(request.get("context_replay_args", []) or [])
    history = [
        {
            "action": name,
            "action_args": (
                dict(raw_args[index])
                if index < len(raw_args) and isinstance(raw_args[index], Mapping)
                else {}
            ),
        }
        for index, name in enumerate(names)
    ]
    if not history:
        raise ValueError("SAGE.8c continuation requires non-empty replay history")
    tail = history[-min(len(history), continuation_tail_window) :]
    schedule = [dict(tail[index % len(tail)]) for index in range(continuation_horizon)]
    return {
        **dict(base),
        "evaluation_id": str(base.get("evaluation_id", "")).replace(
            "sage8b::paired", "sage8c::multi_action"
        ),
        "continuation_policy": CONTINUATION_POLICY,
        "continuation_horizon": int(continuation_horizon),
        "continuation_tail_window": int(continuation_tail_window),
        "continuation_history_length": len(history),
        "continuation_tail_length": len(tail),
        "continuation_schedule": schedule,
        "same_schedule_for_both_arms": True,
        "schedule_selected_before_execution": True,
        "outcome_fields_read_for_schedule": [],
    }


def execute_sage8c_paired_rollout(
    specification: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run the fixed shared continuation after the baseline or memory decision."""
    proposed = PolicyActionOption(
        str(specification.get("no_memory_action", "")),
        dict(specification.get("no_memory_action_args", {}) or {}),
    )
    memory_option = PolicyActionOption(
        str(specification.get("memory_action", "")),
        dict(specification.get("memory_action_args", {}) or {}),
    )
    equivalent = PolicyActionOption(
        str(specification.get("equivalent_action", "")),
        dict(specification.get("equivalent_action_args", {}) or {}),
    )
    decision = apply_relational_memory_policy(
        policy,
        game_id=str(specification.get("game_id", "")),
        context_snapshot_hash=str(specification.get("context_snapshot_hash", "")),
        proposed_action_raw=proposed,
        valid_actions=(proposed, memory_option, equivalent),
        metric=str(specification.get("metric", "")),
    )
    if not bool(decision.get("relational_memory_applied", False)):
        raise ValueError("SAGE.8c requires an applied SAGE.8a policy decision")
    request = dict(specification.get("request", {}) or {})
    schedule = tuple(
        dict(row) for row in specification.get("continuation_schedule", []) or []
    )
    no_memory = _execute_multi_action_arm(
        request,
        initial_action=proposed.name,
        initial_action_args=proposed.action_args,
        continuation_schedule=schedule,
        arm="sage8c_no_memory",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    with_memory = _execute_multi_action_arm(
        request,
        initial_action=str(decision.get("selected_action", "")),
        initial_action_args=dict(decision.get("selected_action_args", {}) or {}),
        continuation_schedule=schedule,
        arm="sage8c_with_relational_memory",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if any(
        str(arm.get("status", "")) != "EXECUTED" for arm in (no_memory, with_memory)
    ):
        reasons = [str(arm.get("reason", "")) for arm in (no_memory, with_memory)]
        raise ValueError(f"SAGE.8c paired rollout arm blocked: {reasons}")
    no_before = _signature_payload(no_memory.get("before_signature", ""))
    memory_before = _signature_payload(with_memory.get("before_signature", ""))
    no_final = _signature_payload(no_memory.get("final_signature", ""))
    memory_final = _signature_payload(with_memory.get("final_signature", ""))
    before_levels = int(memory_before.get("levels_completed", 0) or 0)
    no_levels = int(no_final.get("levels_completed", 0) or 0)
    memory_levels = int(memory_final.get("levels_completed", 0) or 0)
    no_signal = float(no_memory.get("initial_local_signal", 0.0) or 0.0)
    memory_signal = float(with_memory.get("initial_local_signal", 0.0) or 0.0)
    return {
        "evaluation_id": str(specification.get("evaluation_id", "")),
        "replay_source": str(specification.get("replay_source", "")),
        "game_id": str(specification.get("game_id", "")),
        "context_snapshot_hash": str(specification.get("context_snapshot_hash", "")),
        "source_request_id": str(specification.get("source_request_id", "")),
        "source_step": int(specification.get("source_step", 0) or 0),
        "budget": int(specification.get("budget", 0) or 0),
        "metric": str(specification.get("metric", "")),
        "policy_entry_id": str(specification.get("policy_entry_id", "")),
        "policy_decision": _serializable_decision(decision),
        "relational_memory_consulted": True,
        "relational_memory_applied": True,
        "same_prefix_between_arms": (
            no_memory.get("before_signature") == with_memory.get("before_signature")
        ),
        "replay_exact_both_arms": all(
            bool(arm.get("context_snapshot_hash_verified", False))
            for arm in (no_memory, with_memory)
        ),
        "same_continuation_schedule_between_arms": (
            no_memory.get("continuation_schedule")
            == with_memory.get("continuation_schedule")
            == list(schedule)
        ),
        "continuation_policy": CONTINUATION_POLICY,
        "continuation_horizon": int(specification.get("continuation_horizon", 0) or 0),
        "continuation_schedule": list(schedule),
        "schedule_selected_before_execution": True,
        "outcome_fields_read_for_schedule": [],
        "no_memory_action": proposed.name,
        "no_memory_action_args": dict(proposed.action_args),
        "with_memory_action": str(decision.get("selected_action", "")),
        "with_memory_action_args": dict(decision.get("selected_action_args", {}) or {}),
        "no_memory_arm": dict(no_memory),
        "with_memory_arm": dict(with_memory),
        "levels_completed_before": before_levels,
        "no_memory_levels_completed_after": no_levels,
        "with_memory_levels_completed_after": memory_levels,
        "no_memory_levels_completed_delta": no_levels
        - int(no_before.get("levels_completed", 0) or 0),
        "with_memory_levels_completed_delta": memory_levels - before_levels,
        "no_memory_win": _signature_won(no_final),
        "with_memory_win": _signature_won(memory_final),
        "no_memory_terminal": _signature_terminal(no_final),
        "with_memory_terminal": _signature_terminal(memory_final),
        "no_memory_continuation_steps_executed": int(
            no_memory.get("continuation_steps_executed", 0) or 0
        ),
        "with_memory_continuation_steps_executed": int(
            with_memory.get("continuation_steps_executed", 0) or 0
        ),
        "no_memory_local_signal": no_signal,
        "with_memory_local_signal": memory_signal,
        "local_signal_gain": memory_signal - no_signal,
        "primary_metrics": ["levels_completed", "win_rate"],
        "local_signal_is_secondary_only": True,
        "truth_reevaluated": False,
        "support_counted": 0,
        "wrong_confirmations": 0,
    }


def _execute_multi_action_arm(
    request: Mapping[str, Any],
    *,
    initial_action: str,
    initial_action_args: Mapping[str, Any],
    continuation_schedule: Sequence[Mapping[str, Any]],
    arm: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    game_id = str(request.get("game_id", ""))
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, environments_dir)
        )
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return {"status": "BLOCKED", "reason": f"env_setup_failed:{exc}", "arm": arm}
    replay_names = list(request.get("context_replay", []) or [])
    replay_args = list(request.get("context_replay_args", []) or [])
    for index, name in enumerate(replay_names):
        args = (
            dict(replay_args[index])
            if index < len(replay_args) and isinstance(replay_args[index], Mapping)
            else {}
        )
        selected = select_live_action(env, str(name), action_args=args)
        if selected is None:
            return {
                "status": "BLOCKED",
                "reason": f"prefix_action_unavailable:{name}",
                "arm": arm,
            }
        frame = _step_env_action(env, selected)
    replay_signature = state_signature_from_frame(frame)
    expected_signature = str(request.get("context_snapshot_hash", ""))
    if replay_signature != expected_signature:
        return {
            "status": "BLOCKED",
            "reason": "context_snapshot_hash_mismatch",
            "arm": arm,
            "replay_state_signature": replay_signature,
            "target_state_signature": expected_signature,
        }
    selected_initial = select_live_action(
        env, initial_action, action_args=initial_action_args
    )
    if selected_initial is None:
        return {
            "status": "BLOCKED",
            "reason": f"initial_action_unavailable:{initial_action}",
            "arm": arm,
        }
    before_frame = frame
    before = snapshot_frame(before_frame)
    frame = _step_env_action(env, selected_initial)
    after_initial = snapshot_frame(
        frame, fallback_available_actions=before.available_actions
    )
    initial_measurement = measure_required_observation(
        before.grid,
        after_initial.grid,
        required_observation=str(request.get("metric", "")),
        action_args=dict(initial_action_args),
    )
    measurement_for_delta = _measurement_for_delta(
        initial_measurement, metric=str(request.get("metric", ""))
    )
    trace = [
        _trace_row(
            0,
            phase="initial_decision",
            action=initial_action,
            action_args=initial_action_args,
            frame=frame,
        )
    ]
    continuation_steps_executed = 0
    stopped_on_terminal = _signature_terminal(
        _signature_payload(trace[-1]["signature"])
    )
    if not stopped_on_terminal:
        for index, row in enumerate(continuation_schedule, start=1):
            action = str(row.get("action", ""))
            action_args = dict(row.get("action_args", {}) or {})
            selected = select_live_action(env, action, action_args=action_args)
            if selected is None:
                return {
                    "status": "BLOCKED",
                    "reason": f"continuation_action_unavailable:{index}:{action}",
                    "arm": arm,
                    "continuation_steps_executed": continuation_steps_executed,
                }
            frame = _step_env_action(env, selected)
            continuation_steps_executed += 1
            trace.append(
                _trace_row(
                    index,
                    phase="shared_continuation",
                    action=action,
                    action_args=action_args,
                    frame=frame,
                )
            )
            if _signature_terminal(_signature_payload(trace[-1]["signature"])):
                stopped_on_terminal = True
                break
    return {
        "status": "EXECUTED",
        "arm": arm,
        "context_snapshot_hash_verified": True,
        "replay_state_signature": replay_signature,
        "before_signature": state_signature_from_frame(before_frame),
        "after_initial_signature": trace[0]["signature"],
        "final_signature": state_signature_from_frame(frame),
        "initial_action": initial_action,
        "initial_action_args": dict(initial_action_args),
        "initial_local_signal": float(
            measurement_for_delta.get("local_changed_pixels", 0) or 0
        ),
        "initial_signal_source": str(
            measurement_for_delta.get("observed_signal_source", "")
        ),
        "continuation_schedule": [dict(row) for row in continuation_schedule],
        "continuation_steps_requested": len(continuation_schedule),
        "continuation_steps_executed": continuation_steps_executed,
        "stopped_on_terminal": stopped_on_terminal,
        "all_selected_actions_legal": True,
        "trace": trace,
    }


def summarize_rollout_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    count = len(episodes)
    requested = sum(int(row.get("continuation_horizon", 0) or 0) for row in episodes)
    no_steps = sum(
        int(row.get("no_memory_continuation_steps_executed", 0) or 0)
        for row in episodes
    )
    memory_steps = sum(
        int(row.get("with_memory_continuation_steps_executed", 0) or 0)
        for row in episodes
    )
    no_terminals = sum(bool(row.get("no_memory_terminal", False)) for row in episodes)
    memory_terminals = sum(
        bool(row.get("with_memory_terminal", False)) for row in episodes
    )
    return {
        "episodes_per_arm": count,
        "continuation_steps_requested_per_arm": requested,
        "no_memory_continuation_steps_executed": no_steps,
        "with_memory_continuation_steps_executed": memory_steps,
        "no_memory_terminal_episodes": no_terminals,
        "with_memory_terminal_episodes": memory_terminals,
        "no_memory_terminal_rate": no_terminals / count if count else 0.0,
        "with_memory_terminal_rate": memory_terminals / count if count else 0.0,
        "all_actions_legal": all(
            bool(row.get("no_memory_arm", {}).get("all_selected_actions_legal", False))
            and bool(
                row.get("with_memory_arm", {}).get("all_selected_actions_legal", False)
            )
            for row in episodes
        ),
    }


def build_sage8c_gate(
    specifications: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    *,
    continuation_horizon: int,
) -> Dict[str, bool]:
    expected_hashes = {
        str(row.get("context_snapshot_hash", "")) for row in specifications
    }
    observed_hashes = {str(row.get("context_snapshot_hash", "")) for row in episodes}
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    return {
        "all_eleven_exact_contexts_evaluated_once": len(specifications)
        == len(episodes)
        == len(expected_hashes)
        == 11
        and expected_hashes == observed_hashes,
        "fixed_positive_multi_action_horizon": continuation_horizon > 1
        and all(
            int(row.get("continuation_horizon", 0) or 0) == continuation_horizon
            for row in episodes
        ),
        "memory_policy_applied_in_every_treatment_arm": all(
            bool(row.get("relational_memory_applied", False))
            and str(row.get("policy_decision", {}).get("decision_reason", ""))
            == LOWER_EFFECT_COMPARATOR_MATCH
            for row in episodes
        ),
        "all_prefix_replays_exact": all(
            bool(row.get("replay_exact_both_arms", False)) for row in episodes
        ),
        "same_prefix_used_between_arms": all(
            bool(row.get("same_prefix_between_arms", False)) for row in episodes
        ),
        "same_outcome_blind_continuation_used_between_arms": all(
            bool(row.get("same_continuation_schedule_between_arms", False))
            and bool(row.get("schedule_selected_before_execution", False))
            and not list(row.get("outcome_fields_read_for_schedule", []) or [])
            for row in episodes
        ),
        "all_rollout_actions_legal": all(
            bool(row.get("no_memory_arm", {}).get("all_selected_actions_legal", False))
            and bool(
                row.get("with_memory_arm", {}).get("all_selected_actions_legal", False)
            )
            for row in episodes
        ),
        "levels_completed_recorded_as_primary": all(
            key in levels
            for key in (
                "no_memory_total_delta",
                "with_memory_total_delta",
                "absolute_delta_gain",
                "improved",
            )
        ),
        "win_rate_recorded_as_primary": int(wins.get("episodes_per_arm", 0) or 0) == 11
        and 0.0 <= float(wins.get("no_memory", -1.0)) <= 1.0
        and 0.0 <= float(wins.get("with_memory", -1.0)) <= 1.0,
        "no_truth_reevaluation_or_support_counting": all(
            not bool(row.get("truth_reevaluated", True))
            and int(row.get("support_counted", -1) or 0) == 0
            for row in episodes
        ),
    }


def summarize_sage8c(
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    secondary_metrics: Mapping[str, Any],
    rollout_metrics: Mapping[str, Any],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    level_gain = int(levels.get("absolute_delta_gain", 0) or 0)
    win_gain = float(wins.get("absolute_gain", 0.0) or 0.0)
    primary_improved = level_gain > 0 or win_gain > 0.0
    primary_regressed = level_gain < 0 or win_gain < 0.0
    secondary_improved = bool(secondary_metrics.get("improved", False))
    if primary_improved and not primary_regressed:
        outcome = SAGE8C_ARC_GAIN
    elif primary_regressed:
        outcome = SAGE8C_ARC_REGRESSION
    elif secondary_improved:
        outcome = SAGE8C_LOCAL_ONLY
    else:
        outcome = SAGE8C_NO_GAIN
    gate_passed = bool(gate) and all(bool(value) for value in gate.values())
    return {
        "paired_rollouts_evaluated": len(episodes),
        "games_evaluated": sorted({str(row.get("game_id", "")) for row in episodes}),
        "continuation_horizon": max(
            (int(row.get("continuation_horizon", 0) or 0) for row in episodes),
            default=0,
        ),
        "memory_policy_applications": sum(
            bool(row.get("relational_memory_applied", False)) for row in episodes
        ),
        "exact_paired_replays": sum(
            bool(row.get("replay_exact_both_arms", False)) for row in episodes
        ),
        "no_memory_continuation_steps_executed": int(
            rollout_metrics.get("no_memory_continuation_steps_executed", 0) or 0
        ),
        "with_memory_continuation_steps_executed": int(
            rollout_metrics.get("with_memory_continuation_steps_executed", 0) or 0
        ),
        "no_memory_levels_completed_delta_total": int(
            levels.get("no_memory_total_delta", 0) or 0
        ),
        "with_memory_levels_completed_delta_total": int(
            levels.get("with_memory_total_delta", 0) or 0
        ),
        "levels_completed_absolute_gain": level_gain,
        "levels_completed_improved": bool(levels.get("improved", False)),
        "no_memory_wins": int(wins.get("no_memory_wins", 0) or 0),
        "with_memory_wins": int(wins.get("with_memory_wins", 0) or 0),
        "no_memory_win_rate": float(wins.get("no_memory", 0.0) or 0.0),
        "with_memory_win_rate": float(wins.get("with_memory", 0.0) or 0.0),
        "win_rate_absolute_gain": win_gain,
        "win_rate_improved": bool(wins.get("improved", False)),
        "primary_arc_progress_improved": primary_improved,
        "primary_arc_progress_regressed": primary_regressed,
        "secondary_initial_local_signal_gain": float(
            secondary_metrics.get("absolute_gain", 0.0) or 0.0
        ),
        "secondary_initial_local_signal_improved": secondary_improved,
        "local_signal_counted_as_arc_progress": False,
        "continuation_selected_from_outcomes": False,
        "truth_reevaluations": 0,
        "support_counted": 0,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": gate_passed,
        "outcome_status": outcome,
    }


def validate_sage8c_sources(
    source_sage8b: Mapping[str, Any],
    policy_source: Mapping[str, Any],
    a34_2: Mapping[str, Any],
    a34_3: Mapping[str, Any],
) -> None:
    validate_sage8b_sources(policy_source, a34_2, a34_3)
    summary = dict(source_sage8b.get("summary", {}) or {})
    if (
        str(source_sage8b.get("config", {}).get("schema_version", ""))
        != SAGE8B_SCHEMA_VERSION
        or str(source_sage8b.get("outcome_status", "")) != SAGE8B_LOCAL_ONLY_GAIN
        or str(source_sage8b.get("truth_status", "")) != SAGE8B_TRUTH_STATUS
        or str(source_sage8b.get("status", "")) != "EVALUATED"
        or not bool(source_sage8b.get("comparative_evaluation_performed", False))
        or not bool(summary.get("gate_passed", False))
        or bool(summary.get("primary_arc_progress_improved", True))
        or float(summary.get("secondary_local_signal_gain", 0.0) or 0.0) <= 0.0
        or not all(bool(value) for value in source_sage8b.get("gate", {}).values())
    ):
        raise ValueError("SAGE.8c requires the completed local-only SAGE.8b evaluation")
    source_hashes = {
        str(row.get("context_snapshot_hash", ""))
        for row in source_sage8b.get("paired_episodes", []) or []
    }
    policy_hashes = {
        str(value)
        for entry in policy_source.get("policy_entries", []) or []
        for value in entry.get("context_snapshot_hashes", []) or []
    }
    if len(source_hashes) != 11 or source_hashes != policy_hashes:
        raise ValueError("SAGE.8b and SAGE.8a context scopes must match exactly")


def write_sage8c_relational_memory_multi_action_evaluation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_rollout_config(horizon: int, tail_window: int) -> None:
    if int(horizon) <= 1:
        raise ValueError("SAGE.8c continuation_horizon must be greater than one")
    if int(tail_window) <= 0:
        raise ValueError("SAGE.8c continuation_tail_window must be positive")


def _trace_row(
    index: int,
    *,
    phase: str,
    action: str,
    action_args: Mapping[str, Any],
    frame: Any,
) -> Dict[str, Any]:
    signature = state_signature_from_frame(frame)
    payload = _signature_payload(signature)
    return {
        "rollout_step": int(index),
        "phase": phase,
        "action": action,
        "action_args": dict(action_args),
        "signature": signature,
        "levels_completed": int(payload.get("levels_completed", 0) or 0),
        "game_state": str(payload.get("game_state", "")),
    }


def _public_specification(specification: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in specification.items() if key != "request"}


def _serializable_decision(decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in decision.items() if key != "selected_action_raw"
    }


def _signature_payload(value: Any) -> Dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _signature_won(signature: Mapping[str, Any]) -> bool:
    return str(signature.get("game_state", "")).upper() in {"WIN", "WON", "VICTORY"}


def _signature_terminal(signature: Mapping[str, Any]) -> bool:
    state = str(signature.get("game_state", "")).upper()
    return state not in {"", "NOT_FINISHED", "PLAYING", "IN_PROGRESS"}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sage8b", default=str(DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH)
    )
    parser.add_argument(
        "--policy", default=str(DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH)
    )
    parser.add_argument(
        "--a34-2",
        default=str(DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH),
    )
    parser.add_argument(
        "--a34-3", default=str(DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH)
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--continuation-horizon", type=int, default=DEFAULT_CONTINUATION_HORIZON
    )
    parser.add_argument(
        "--continuation-tail-window",
        type=int,
        default=DEFAULT_CONTINUATION_TAIL_WINDOW,
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH),
    )
    args = parser.parse_args(argv)
    payload = run_sage8c_relational_memory_multi_action_evaluation(
        sage8b_path=args.sage8b,
        policy_path=args.policy,
        a34_2_path=args.a34_2,
        a34_3_path=args.a34_3,
        environments_dir=args.environments_dir,
        continuation_horizon=args.continuation_horizon,
        continuation_tail_window=args.continuation_tail_window,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
