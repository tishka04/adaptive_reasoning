"""SAGE.1 known-game technical live runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .known_game_closed_loop_scaffold import (
    DEFAULT_M2_FUSED_REQUESTS_PATH,
    DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH,
    DEFAULT_M3_FUSED_RESULTS_PATH,
    DEFAULT_P1_POLICY_PROBE_PATH,
    DEFAULT_P1_UTILITY_HANDOFF_PATH,
    hypothesis_context_summary,
    m3_tests_summary,
    policy_context_summary,
)
from .live_prefix_counterfactual_collector import (
    LivePrefixAction,
    collect_live_prefix_counterfactual,
    select_live_action,
    state_signature_from_frame,
)
from .policy_loop_guard import (
    DEFAULT_SAGE1B_POLICY_LOOP_GUARD_RESULTS_PATH,
    LOOP_GUARD_SWITCH_REASON,
    SAGE1B_SCHEMA_VERSION,
    SAGE1B_TRUTH_STATUS,
    apply_policy_loop_guard,
    max_consecutive_same_action_arg_repeats,
)


DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage1_known_game_closed_loop_results.json"
)
SAGE1_SCHEMA_VERSION = "sage.known_game_closed_loop_results.v1"
SAGE1_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_1"
DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_BUDGET = 20
DEFAULT_LOOP_GUARD_MAX_REPEATS = 2


def run_sage1_known_game_closed_loop(
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
    budget: int = DEFAULT_BUDGET,
    env_factory: Any | None = None,
    enable_loop_guard: bool = False,
    loop_guard_max_repeats: int = DEFAULT_LOOP_GUARD_MAX_REPEATS,
) -> Dict[str, Any]:
    inputs = {
        "m2_fused_requests": _load_json(m2_fused_requests_path),
        "m3_fused_results": _load_json(m3_fused_results_path),
        "m3_counterfactual_feasibility": _load_json(m3_counterfactual_feasibility_path),
        "p1_policy_probe": _load_json(p1_policy_probe_path),
        "p1_utility_handoff": _load_json(p1_utility_handoff_path),
    }
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    try:
        env = env_factory(game_id) if env_factory is not None else _make_real_env(game_id, env_dir)
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        truth_status = SAGE1B_TRUTH_STATUS if enable_loop_guard else SAGE1_TRUTH_STATUS
        payload = blocked_runner_payload(
            reason=f"env_setup_failed:{exc}",
            game_id=game_id,
            budget=budget,
            env_dir=env_dir,
            inputs=inputs,
            truth_status=truth_status,
            enable_loop_guard=enable_loop_guard,
            loop_guard_max_repeats=loop_guard_max_repeats,
            config_paths=_config_paths(
                m2_fused_requests_path,
                m3_fused_results_path,
                m3_counterfactual_feasibility_path,
                p1_policy_probe_path,
                p1_utility_handoff_path,
            ),
        )
        if output_path is not None:
            write_sage1_known_game_results(payload, output_path)
        return payload

    payload = execute_sage1_loop(
        inputs=inputs,
        env=env,
        frame=frame,
        game_id=game_id,
        budget=budget,
        env_dir=env_dir,
        env_factory=env_factory,
        truth_status=SAGE1B_TRUTH_STATUS if enable_loop_guard else SAGE1_TRUTH_STATUS,
        enable_loop_guard=enable_loop_guard,
        loop_guard_max_repeats=loop_guard_max_repeats,
        config_paths=_config_paths(
            m2_fused_requests_path,
            m3_fused_results_path,
            m3_counterfactual_feasibility_path,
            p1_policy_probe_path,
            p1_utility_handoff_path,
        ),
    )
    if output_path is not None:
        write_sage1_known_game_results(payload, output_path)
    return payload


def execute_sage1_loop(
    *,
    inputs: Mapping[str, Any],
    env: Any,
    frame: Any,
    game_id: str,
    budget: int,
    env_dir: Path,
    env_factory: Any | None,
    truth_status: str,
    enable_loop_guard: bool,
    loop_guard_max_repeats: int,
    config_paths: Mapping[str, str],
) -> Dict[str, Any]:
    policy_memory = dict(
        inputs.get("p1_policy_probe", {}).get("candidate_policy_memory", {}) or {}
    )
    prefix: list[LivePrefixAction] = []
    observations: list[Dict[str, Any]] = [
        live_observation_record(frame, step=0, truth_status=truth_status)
    ]
    steps: list[Dict[str, Any]] = []
    invalid_actions = 0
    counterfactual_results: list[Dict[str, Any]] = []
    counterfactual_done = False
    frontier = counterfactual_frontier(inputs)

    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(frame)
        valid_actions = _valid_actions(env)
        decision = select_runner_action(
            step_index=step_index,
            valid_actions=valid_actions,
            policy_memory=policy_memory,
            prefix=prefix,
            frontier=frontier,
            enable_loop_guard=enable_loop_guard,
            loop_guard_max_repeats=loop_guard_max_repeats,
        )
        if decision["selected_action_raw"] is None:
            invalid_actions += 1
            steps.append(
                error_step_record(
                    step_index,
                    game_id,
                    decision,
                    "no_legal_action",
                    truth_status=truth_status,
                )
            )
            break

        before_signature = state_signature_from_frame(frame)
        try:
            after_frame = _step_env_action(env, decision["selected_action_raw"])
        except Exception as exc:  # pragma: no cover - integration failure path
            invalid_actions += 1
            steps.append(
                error_step_record(
                    step_index,
                    game_id,
                    decision,
                    f"env_step_failed:{exc}",
                    before_signature=before_signature,
                    truth_status=truth_status,
                )
            )
            break

        after_signature = state_signature_from_frame(after_frame)
        action = LivePrefixAction(
            name=str(decision["action"]),
            action_args=dict(decision["action_args"]),
        )
        prefix.append(action)
        step_record = {
            "step": step_index,
            "game_id": game_id,
            "available_actions_source": "real_env_live_api",
            "synthetic_available_actions_used": False,
            "real_env_available_actions_used": True,
            "available_actions": list(decision["available_actions"]),
            "selected_action": action.name,
            "selected_action_args": dict(action.action_args),
            "selected_action_legal": True,
            "invalid_action_selected": False,
            "decision_reason": str(decision["decision_reason"]),
            "candidate_policy_used": bool(decision["candidate_policy_used"]),
            "state_signature_before": before_signature,
            "state_signature_after": after_signature,
            "state_changed": before_signature != after_signature,
            "levels_before": before.levels_completed,
            "levels_after": snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            ).levels_completed,
            "game_state_before": before.game_state,
            "game_state_after": snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            ).game_state,
            "env_actions": 1,
            "offline_counterfactual_allowed": False,
            "active_counterfactual_collection_allowed": True,
            "support": 0,
            "truth_status": truth_status,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "loop_guard_enabled": bool(decision.get("loop_guard_enabled", False)),
            "loop_guard_triggered": bool(decision.get("loop_guard_triggered", False)),
            "repeated_same_action_args_detected": bool(
                decision.get("repeated_same_action_args_detected", False)
            ),
            "fallback_loop_interrupted": bool(
                decision.get("fallback_loop_interrupted", False)
            ),
            "switch_action_selected_after_exhaustion": bool(
                decision.get("switch_action_selected_after_exhaustion", False)
            ),
            "selected_switch_action": str(decision.get("selected_switch_action", "")),
            "selected_switch_action_args": dict(
                decision.get("selected_switch_action_args", {}) or {}
            ),
            "blocked_action": str(decision.get("blocked_action", "")),
            "blocked_action_args": dict(decision.get("blocked_action_args", {}) or {}),
            "consecutive_repeats_before": int(
                decision.get("consecutive_repeats_before", 0) or 0
            ),
            "max_same_action_arg_repeats": int(
                decision.get("max_same_action_arg_repeats", loop_guard_max_repeats) or 0
            ),
            "loop_guard_reason": str(decision.get("loop_guard_reason", "")),
        }
        steps.append(step_record)
        frame = after_frame
        observations.append(
            live_observation_record(
                frame,
                step=step_index + 1,
                truth_status=truth_status,
            )
        )

        if frontier and not counterfactual_done and len(prefix) >= 1:
            alternative_action = str(frontier.get("target_action", "") or "ACTION3")
            target_signature = state_signature_from_frame(frame)
            result = collect_live_prefix_counterfactual(
                game_id=game_id,
                prefix_actions=tuple(prefix),
                target_state_signature=target_signature,
                alternative_action=alternative_action,
                alternative_action_args=frontier.get("target_action_args"),
                environments_dir=env_dir,
                env_factory=env_factory,
            )
            counterfactual_results.append(result.to_dict())
            counterfactual_done = True

    logger_contract = {
        "available_actions_source": "real_env_live_api",
        "real_env_available_actions_used": True,
        "synthetic_available_actions_used": False,
        "selected_action_always_legal": invalid_actions == 0 and bool(steps),
        "invalid_action_selected": invalid_actions > 0,
        "offline_counterfactual_allowed": False,
        "active_counterfactual_collection_allowed": True,
        "active_counterfactual_collection_attempted": bool(counterfactual_results),
        "policy_result_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "support": 0,
        "truth_status": truth_status,
        "revision_status": "CANDIDATE_ONLY",
        "loop_guard_enabled": bool(enable_loop_guard),
        "loop_guard_max_repeats": int(loop_guard_max_repeats),
        "repeated_same_action_args_detected": any(
            bool(row.get("repeated_same_action_args_detected", False)) for row in steps
        ),
        "fallback_loop_interrupted": any(
            bool(row.get("fallback_loop_interrupted", False)) for row in steps
        ),
        "switch_action_selected_after_exhaustion": any(
            bool(row.get("switch_action_selected_after_exhaustion", False))
            for row in steps
        ),
    }
    summary = summarize_sage1(
        steps=steps,
        counterfactual_results=counterfactual_results,
        logger_contract=logger_contract,
        inputs=inputs,
        truth_status=truth_status,
    )
    return {
        "config": {
            **dict(config_paths),
            "schema_version": (
                SAGE1B_SCHEMA_VERSION if enable_loop_guard else SAGE1_SCHEMA_VERSION
            ),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "budget": int(budget),
            "technical_live_run": True,
            "benchmark_run": False,
            "loop_guard_enabled": bool(enable_loop_guard),
            "loop_guard_max_repeats": int(loop_guard_max_repeats),
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "logger_contract": logger_contract,
        "input_summaries": {
            "hypothesis_context": hypothesis_context_summary(inputs),
            "m3_tests": m3_tests_summary(inputs),
            "policy_context": policy_context_summary(inputs),
        },
        "live_observations": observations,
        "live_steps": steps,
        "active_counterfactual_collections": counterfactual_results,
        "summary": summary,
        "status": "UNRESOLVED",
        "truth_status": truth_status,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def select_runner_action(
    *,
    step_index: int,
    valid_actions: Sequence[Any],
    policy_memory: Mapping[str, Any],
    prefix: Sequence[LivePrefixAction],
    frontier: Mapping[str, Any] | None = None,
    enable_loop_guard: bool = False,
    loop_guard_max_repeats: int = DEFAULT_LOOP_GUARD_MAX_REPEATS,
) -> Dict[str, Any]:
    names = tuple(
        sorted(
            {
                str(getattr(action, "name", ""))
                for action in valid_actions
                if str(getattr(action, "name", ""))
            }
        )
    )
    reposition = str(policy_memory.get("repositioning_action", "ACTION4") or "ACTION4")
    target = str(policy_memory.get("target_action", "ACTION6") or "ACTION6")
    if step_index == 0:
        selected = select_from_actions(valid_actions, reposition, {})
        if selected is not None:
            return finalize_decision_with_loop_guard(
                decision_payload(
                    selected,
                    names,
                    "candidate_policy_live_repositioning",
                    candidate_policy_used=True,
                ),
                valid_actions=valid_actions,
                prefix=prefix,
                target_action=target,
                frontier=frontier,
                enable_loop_guard=enable_loop_guard,
                loop_guard_max_repeats=loop_guard_max_repeats,
            )
    success_args = [
        dict(row)
        for row in policy_memory.get("known_success_args", []) or []
        if isinstance(row, Mapping)
    ]
    used_args = {
        tuple(sorted(action.action_args.items()))
        for action in prefix
        if action.name == target and action.action_args
    }
    for args in success_args:
        if tuple(sorted(args.items())) in used_args:
            continue
        selected = select_from_actions(valid_actions, target, args)
        if selected is not None:
            return finalize_decision_with_loop_guard(
                decision_payload(
                    selected,
                    names,
                    "candidate_policy_live_success_like_target",
                    candidate_policy_used=True,
                ),
                valid_actions=valid_actions,
                prefix=prefix,
                target_action=target,
                frontier=frontier,
                enable_loop_guard=enable_loop_guard,
                loop_guard_max_repeats=loop_guard_max_repeats,
            )
    selected = select_from_actions(valid_actions, target, {})
    if selected is not None:
        return finalize_decision_with_loop_guard(
            decision_payload(
                selected,
                names,
                "candidate_policy_live_target_fallback",
                candidate_policy_used=True,
            ),
            valid_actions=valid_actions,
            prefix=prefix,
            target_action=target,
            frontier=frontier,
            enable_loop_guard=enable_loop_guard,
            loop_guard_max_repeats=loop_guard_max_repeats,
        )
    for action in valid_actions:
        name = str(getattr(action, "name", ""))
        if name and name != "RESET":
            return finalize_decision_with_loop_guard(
                decision_payload(
                    action,
                    names,
                    "neutral_legal_fallback",
                    candidate_policy_used=False,
                ),
                valid_actions=valid_actions,
                prefix=prefix,
                target_action=target,
                frontier=frontier,
                enable_loop_guard=enable_loop_guard,
                loop_guard_max_repeats=loop_guard_max_repeats,
            )
    return {
        "selected_action_raw": None,
        "action": "",
        "action_args": {},
        "available_actions": list(names),
        "decision_reason": "no_legal_action_available",
        "candidate_policy_used": False,
        "loop_guard_enabled": bool(enable_loop_guard),
        "loop_guard_triggered": False,
        "repeated_same_action_args_detected": False,
        "fallback_loop_interrupted": False,
        "switch_action_selected_after_exhaustion": False,
        "selected_switch_action": "",
        "selected_switch_action_args": {},
        "blocked_action": "",
        "blocked_action_args": {},
        "consecutive_repeats_before": 0,
        "max_same_action_arg_repeats": int(loop_guard_max_repeats),
        "loop_guard_reason": "no_legal_action_available",
    }


def finalize_decision_with_loop_guard(
    decision: Dict[str, Any],
    *,
    valid_actions: Sequence[Any],
    prefix: Sequence[LivePrefixAction],
    target_action: str,
    frontier: Mapping[str, Any] | None,
    enable_loop_guard: bool,
    loop_guard_max_repeats: int,
) -> Dict[str, Any]:
    if not enable_loop_guard:
        decision.update(
            {
                "loop_guard_enabled": False,
                "loop_guard_triggered": False,
                "repeated_same_action_args_detected": False,
                "fallback_loop_interrupted": False,
                "switch_action_selected_after_exhaustion": False,
                "selected_switch_action": "",
                "selected_switch_action_args": {},
                "blocked_action": "",
                "blocked_action_args": {},
                "consecutive_repeats_before": 0,
                "max_same_action_arg_repeats": int(loop_guard_max_repeats),
                "loop_guard_reason": "loop_guard_disabled",
            }
        )
        return decision

    switch_preference = preferred_switch_actions(frontier, target_action)
    guard = apply_policy_loop_guard(
        proposed_action=decision.get("selected_action_raw"),
        valid_actions=valid_actions,
        prefix=prefix,
        decision_reason=str(decision.get("decision_reason", "")),
        target_action=target_action,
        max_same_action_arg_repeats=loop_guard_max_repeats,
        switch_preference=switch_preference,
    )
    if guard.selected_action_raw is not decision.get("selected_action_raw"):
        decision.update(
            decision_payload(
                guard.selected_action_raw,
                decision.get("available_actions", []),
                LOOP_GUARD_SWITCH_REASON,
                candidate_policy_used=False,
            )
        )
    decision.update(guard.to_dict())
    return decision


def preferred_switch_actions(
    frontier: Mapping[str, Any] | None,
    target_action: str,
) -> tuple[str, ...]:
    names: list[str] = []
    frontier_action = str((frontier or {}).get("target_action", "") or "")
    if frontier_action and frontier_action != target_action:
        names.append(frontier_action)
    for name in ("ACTION3", "ACTION4", "ACTION7"):
        if name != target_action and name not in names:
            names.append(name)
    return tuple(names)


def select_from_actions(
    valid_actions: Sequence[Any],
    action_name: str,
    action_args: Mapping[str, Any],
) -> Any | None:
    wanted = str(action_name)
    wanted_args = {str(key): str(value) for key, value in dict(action_args or {}).items()}
    fallback = None
    for action in valid_actions:
        if str(getattr(action, "name", "")) != wanted:
            continue
        actual_args = {
            str(key): str(value)
            for key, value in dict(getattr(action, "action_args", {}) or {}).items()
        }
        if wanted_args and actual_args == wanted_args:
            return action
        if not wanted_args and not actual_args:
            return action
        if fallback is None:
            fallback = action
    return fallback if not wanted_args else None


def decision_payload(
    selected: Any,
    available_action_names: Sequence[str],
    reason: str,
    *,
    candidate_policy_used: bool,
) -> Dict[str, Any]:
    return {
        "selected_action_raw": selected,
        "action": str(getattr(selected, "name", "")),
        "action_args": dict(getattr(selected, "action_args", {}) or {}),
        "available_actions": list(available_action_names),
        "decision_reason": reason,
        "candidate_policy_used": candidate_policy_used,
    }


def live_observation_record(
    frame: Any,
    *,
    step: int,
    truth_status: str = SAGE1_TRUTH_STATUS,
) -> Dict[str, Any]:
    snapshot = snapshot_frame(frame)
    return {
        "step": int(step),
        "available_actions": list(snapshot.available_actions),
        "available_actions_source": "real_env_live_api",
        "synthetic_available_actions_used": False,
        "real_env_available_actions_used": True,
        "state_signature": state_signature_from_frame(frame),
        "game_state": snapshot.game_state,
        "levels_completed": snapshot.levels_completed,
        "truth_status": truth_status,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
    }


def counterfactual_frontier(inputs: Mapping[str, Any]) -> Dict[str, Any] | None:
    rows = (
        inputs.get("m3_counterfactual_feasibility", {})
        .get("counterfactual_probe_results", [])
        or []
    )
    for row in rows:
        frontier = dict(row.get("recommended_frontier", {}) or {})
        if frontier:
            return frontier
    return None


def summarize_sage1(
    *,
    steps: Sequence[Mapping[str, Any]],
    counterfactual_results: Sequence[Mapping[str, Any]],
    logger_contract: Mapping[str, Any],
    inputs: Mapping[str, Any],
    truth_status: str = SAGE1_TRUTH_STATUS,
) -> Dict[str, Any]:
    replay_exact = [
        row for row in counterfactual_results if bool(row.get("live_prefix_replay_exact"))
    ]
    return {
        "known_game_live_run_completed": bool(steps),
        "technical_live_run": True,
        "benchmark_run": False,
        "live_steps_executed": len(steps),
        "env_actions": sum(int(row.get("env_actions", 0) or 0) for row in steps),
        "real_env_available_actions_used": bool(
            logger_contract.get("real_env_available_actions_used", False)
        ),
        "available_actions_source": str(logger_contract.get("available_actions_source", "")),
        "synthetic_available_actions_used": bool(
            logger_contract.get("synthetic_available_actions_used", True)
        ),
        "selected_action_always_legal": bool(
            logger_contract.get("selected_action_always_legal", False)
        ),
        "invalid_action_selected": bool(logger_contract.get("invalid_action_selected", True)),
        "offline_counterfactual_allowed": False,
        "active_counterfactual_collection_attempted": bool(counterfactual_results),
        "live_prefix_replay_exact": bool(replay_exact),
        "live_prefix_replay_exact_count": len(replay_exact),
        "counterfactual_collections": len(counterfactual_results),
        "loop_guard_enabled": bool(logger_contract.get("loop_guard_enabled", False)),
        "loop_guard_max_repeats": int(
            logger_contract.get("loop_guard_max_repeats", 0) or 0
        ),
        "repeated_same_action_args_detected": bool(
            logger_contract.get("repeated_same_action_args_detected", False)
        ),
        "fallback_loop_interrupted": bool(
            logger_contract.get("fallback_loop_interrupted", False)
        ),
        "switch_action_selected_after_exhaustion": bool(
            logger_contract.get("switch_action_selected_after_exhaustion", False)
        ),
        "loop_guard_switches": sum(
            1 for row in steps if bool(row.get("fallback_loop_interrupted", False))
        ),
        "max_same_action_arg_repeats": max_consecutive_same_action_arg_repeats(
            steps
        ),
        "policy_result_counted_as_confirmation": False,
        "m2_ready_for_m3": hypothesis_context_summary(inputs)["m2_ready_for_m3"],
        "m3_fused_requests_executed": m3_tests_summary(inputs)[
            "m3_fused_requests_executed"
        ],
        "support": 0,
        "truth_status": truth_status,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def error_step_record(
    step_index: int,
    game_id: str,
    decision: Mapping[str, Any],
    error: str,
    *,
    before_signature: str = "",
    truth_status: str = SAGE1_TRUTH_STATUS,
) -> Dict[str, Any]:
    return {
        "step": int(step_index),
        "game_id": game_id,
        "available_actions_source": "real_env_live_api",
        "synthetic_available_actions_used": False,
        "real_env_available_actions_used": True,
        "available_actions": list(decision.get("available_actions", []) or []),
        "selected_action": str(decision.get("action", "")),
        "selected_action_args": dict(decision.get("action_args", {}) or {}),
        "selected_action_legal": False,
        "invalid_action_selected": True,
        "decision_reason": str(decision.get("decision_reason", "")),
        "error": error,
        "state_signature_before": before_signature,
        "env_actions": 0,
        "support": 0,
        "truth_status": truth_status,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def blocked_runner_payload(
    *,
    reason: str,
    game_id: str,
    budget: int,
    env_dir: Path,
    inputs: Mapping[str, Any],
    truth_status: str,
    enable_loop_guard: bool,
    loop_guard_max_repeats: int,
    config_paths: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "config": {
            **dict(config_paths),
            "schema_version": (
                SAGE1B_SCHEMA_VERSION if enable_loop_guard else SAGE1_SCHEMA_VERSION
            ),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "budget": int(budget),
            "technical_live_run": True,
            "benchmark_run": False,
            "loop_guard_enabled": bool(enable_loop_guard),
            "loop_guard_max_repeats": int(loop_guard_max_repeats),
        },
        "summary": {
            "known_game_live_run_completed": False,
            "blocked_reason": reason,
            "support": 0,
            "truth_status": truth_status,
            "revision_status": "CANDIDATE_ONLY",
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "input_summaries": {
            "hypothesis_context": hypothesis_context_summary(inputs),
            "m3_tests": m3_tests_summary(inputs),
            "policy_context": policy_context_summary(inputs),
        },
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": truth_status,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_sage1_known_game_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_real_env(game_id: str, env_dir: Path) -> Any:
    _configure_offline_env(env_dir)
    return _make_env(game_id, env_dir)


def _config_paths(
    m2_fused_requests_path: str | Path,
    m3_fused_results_path: str | Path,
    m3_counterfactual_feasibility_path: str | Path,
    p1_policy_probe_path: str | Path,
    p1_utility_handoff_path: str | Path,
) -> Dict[str, str]:
    return {
        "m2_fused_requests_path": str(m2_fused_requests_path),
        "m3_fused_results_path": str(m3_fused_results_path),
        "m3_counterfactual_feasibility_path": str(m3_counterfactual_feasibility_path),
        "p1_policy_probe_path": str(p1_policy_probe_path),
        "p1_utility_handoff_path": str(p1_utility_handoff_path),
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.1 known-game technical live runner.",
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
    parser.add_argument("--out", default=None)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--enable-loop-guard", action="store_true")
    parser.add_argument(
        "--loop-guard-max-repeats",
        type=int,
        default=DEFAULT_LOOP_GUARD_MAX_REPEATS,
    )
    args = parser.parse_args(argv)
    output_path = args.out
    if output_path is None:
        output_path = (
            DEFAULT_SAGE1B_POLICY_LOOP_GUARD_RESULTS_PATH
            if args.enable_loop_guard
            else DEFAULT_SAGE1_KNOWN_GAME_RESULTS_PATH
        )
    run_sage1_known_game_closed_loop(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        environments_dir=args.environments_dir,
        output_path=output_path,
        game_id=args.game_id,
        budget=args.budget,
        enable_loop_guard=args.enable_loop_guard,
        loop_guard_max_repeats=args.loop_guard_max_repeats,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
