"""SAGE.2 known-game closed-loop policy probe.

This probe is the first SAGE step that measures a *comparative* trajectory on a
known game instead of only proving technical live integration (SAGE.1/1b). It
runs several short policies from the same RESET on the same real env wrapper and
reports per-policy metrics plus a candidate-only comparison.

It never confirms or refutes a mechanic. Every artefact keeps ``support=0``,
``revision_status=CANDIDATE_ONLY`` and
``policy_result_counted_as_confirmation=false``. No A32/A33 write is performed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

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
from .known_game_closed_loop_runner import (
    counterfactual_frontier,
    decision_payload,
    select_runner_action,
    select_from_actions,
)
from .live_prefix_counterfactual_collector import (
    LivePrefixAction,
    collect_live_prefix_counterfactual,
    state_signature_from_frame,
)
from .policy_loop_guard import action_key


DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage2_known_game_policy_probe_results.json"
)
SAGE2_SCHEMA_VERSION = "sage.known_game_policy_probe_results.v1"
SAGE2_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_2"
DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_BUDGET = 20
DEFAULT_SEED = 0
DEFAULT_LOOP_GUARD_MAX_REPEATS = 2

POLICY_RANDOM_LEGAL = "random_legal"
POLICY_REPEAT_FALLBACK = "repeat_action6_fallback"
POLICY_SAGE1_NO_GUARD = "sage1_without_loop_guard"
POLICY_SAGE1B_GUARD = "sage1b_with_loop_guard"
PROBE_POLICIES: tuple[str, ...] = (
    POLICY_RANDOM_LEGAL,
    POLICY_REPEAT_FALLBACK,
    POLICY_SAGE1_NO_GUARD,
    POLICY_SAGE1B_GUARD,
)

_TERMINAL_STATES = {"GAME_OVER", "WIN", "TERMINATED", "FINISHED"}

EnvFactory = Callable[[str], Any]


def run_sage2_known_game_policy_probe(
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
    seed: int = DEFAULT_SEED,
    loop_guard_max_repeats: int = DEFAULT_LOOP_GUARD_MAX_REPEATS,
    policies: Sequence[str] = PROBE_POLICIES,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run all probe policies on the same known game and compare them."""
    inputs = {
        "m2_fused_requests": _load_json(m2_fused_requests_path),
        "m3_fused_results": _load_json(m3_fused_results_path),
        "m3_counterfactual_feasibility": _load_json(m3_counterfactual_feasibility_path),
        "p1_policy_probe": _load_json(p1_policy_probe_path),
        "p1_utility_handoff": _load_json(p1_utility_handoff_path),
    }
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    policy_memory = dict(
        inputs.get("p1_policy_probe", {}).get("candidate_policy_memory", {}) or {}
    )
    frontier = counterfactual_frontier(inputs)

    policy_runs: List[Dict[str, Any]] = []
    for policy in policies:
        run = _run_single_policy(
            policy=policy,
            game_id=game_id,
            budget=budget,
            env_dir=env_dir,
            env_factory=env_factory,
            policy_memory=policy_memory,
            frontier=frontier,
            seed=seed,
            loop_guard_max_repeats=loop_guard_max_repeats,
        )
        policy_runs.append(run)

    comparison = build_comparison(policy_runs)
    payload = {
        "config": {
            **_config_paths(
                m2_fused_requests_path,
                m3_fused_results_path,
                m3_counterfactual_feasibility_path,
                p1_policy_probe_path,
                p1_utility_handoff_path,
            ),
            "schema_version": SAGE2_SCHEMA_VERSION,
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "budget": int(budget),
            "seed": int(seed),
            "loop_guard_max_repeats": int(loop_guard_max_repeats),
            "policies": list(policies),
            "known_game_policy_probe": True,
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "input_summaries": {
            "hypothesis_context": hypothesis_context_summary(inputs),
            "m3_tests": m3_tests_summary(inputs),
            "policy_context": policy_context_summary(inputs),
        },
        "policy_runs": policy_runs,
        "comparison": comparison,
        "status": "UNRESOLVED",
        "truth_status": SAGE2_TRUTH_STATUS,
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
        write_sage2_policy_probe_results(payload, output_path)
    return payload


def _run_single_policy(
    *,
    policy: str,
    game_id: str,
    budget: int,
    env_dir: Path,
    env_factory: EnvFactory | None,
    policy_memory: Mapping[str, Any],
    frontier: Mapping[str, Any] | None,
    seed: int,
    loop_guard_max_repeats: int,
) -> Dict[str, Any]:
    collect_counterfactual = policy.startswith("sage")
    rng = random.Random(f"{policy}:{seed}")
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, env_dir)
        )
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return _blocked_policy_run(policy, f"env_setup_failed:{exc}", budget)

    prefix: List[LivePrefixAction] = []
    steps: List[Dict[str, Any]] = []
    counterfactuals: List[Dict[str, Any]] = []
    counterfactual_done = False
    initial_signature = state_signature_from_frame(frame)

    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(frame)
        valid_actions = _valid_actions(env)
        decision = _select_for_policy(
            policy=policy,
            step_index=step_index,
            valid_actions=valid_actions,
            policy_memory=policy_memory,
            prefix=prefix,
            frontier=frontier,
            rng=rng,
            loop_guard_max_repeats=loop_guard_max_repeats,
        )
        if decision.get("selected_action_raw") is None:
            steps.append(_error_step_record(policy, step_index, decision, game_id))
            break

        before_signature = state_signature_from_frame(frame)
        try:
            after_frame = _step_env_action(env, decision["selected_action_raw"])
        except Exception as exc:  # pragma: no cover - integration failure path
            steps.append(
                _error_step_record(
                    policy,
                    step_index,
                    decision,
                    game_id,
                    error=f"env_step_failed:{exc}",
                    before_signature=before_signature,
                )
            )
            break

        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        after_signature = state_signature_from_frame(after_frame)
        action = LivePrefixAction(
            name=str(decision["action"]),
            action_args=dict(decision["action_args"]),
        )
        prefix.append(action)
        steps.append(
            _policy_step_record(
                policy=policy,
                step_index=step_index,
                game_id=game_id,
                decision=decision,
                action=action,
                before=before,
                after=after,
                before_signature=before_signature,
                after_signature=after_signature,
            )
        )
        frame = after_frame

        if (
            collect_counterfactual
            and frontier
            and not counterfactual_done
            and len(prefix) >= 1
        ):
            alternative_action = str(frontier.get("target_action", "") or "ACTION3")
            result = collect_live_prefix_counterfactual(
                game_id=game_id,
                prefix_actions=tuple(prefix),
                target_state_signature=state_signature_from_frame(frame),
                alternative_action=alternative_action,
                alternative_action_args=frontier.get("target_action_args"),
                environments_dir=env_dir,
                env_factory=env_factory,
            )
            counterfactuals.append(result.to_dict())
            counterfactual_done = True

    summary = summarize_policy_run(
        policy=policy,
        steps=steps,
        counterfactuals=counterfactuals,
        initial_signature=initial_signature,
    )
    return {
        "policy": policy,
        "is_sage_policy": collect_counterfactual,
        "loop_guard_enabled": policy == POLICY_SAGE1B_GUARD,
        "steps": steps,
        "active_counterfactual_collections": counterfactuals,
        "summary": summary,
    }


def _select_for_policy(
    *,
    policy: str,
    step_index: int,
    valid_actions: Sequence[Any],
    policy_memory: Mapping[str, Any],
    prefix: Sequence[LivePrefixAction],
    frontier: Mapping[str, Any] | None,
    rng: random.Random,
    loop_guard_max_repeats: int,
) -> Dict[str, Any]:
    if policy == POLICY_SAGE1_NO_GUARD:
        return select_runner_action(
            step_index=step_index,
            valid_actions=valid_actions,
            policy_memory=policy_memory,
            prefix=prefix,
            frontier=frontier,
            enable_loop_guard=False,
            loop_guard_max_repeats=loop_guard_max_repeats,
        )
    if policy == POLICY_SAGE1B_GUARD:
        return select_runner_action(
            step_index=step_index,
            valid_actions=valid_actions,
            policy_memory=policy_memory,
            prefix=prefix,
            frontier=frontier,
            enable_loop_guard=True,
            loop_guard_max_repeats=loop_guard_max_repeats,
        )
    if policy == POLICY_RANDOM_LEGAL:
        return _baseline_random_decision(valid_actions, rng)
    if policy == POLICY_REPEAT_FALLBACK:
        return _baseline_repeat_fallback_decision(valid_actions, policy_memory)
    raise ValueError(f"unknown_probe_policy:{policy}")


def _baseline_random_decision(
    valid_actions: Sequence[Any],
    rng: random.Random,
) -> Dict[str, Any]:
    legal = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) and str(getattr(action, "name", "")) != "RESET"
    ]
    if not legal:
        return _no_legal_action_decision()
    selected = rng.choice(legal)
    return _baseline_decision_payload(selected, valid_actions, "random_legal_baseline")


def _baseline_repeat_fallback_decision(
    valid_actions: Sequence[Any],
    policy_memory: Mapping[str, Any],
) -> Dict[str, Any]:
    target = str(policy_memory.get("target_action", "ACTION6") or "ACTION6")
    selected = select_from_actions(valid_actions, target, {})
    if selected is None:
        for action in valid_actions:
            name = str(getattr(action, "name", ""))
            if name and name != "RESET":
                selected = action
                break
    if selected is None:
        return _no_legal_action_decision()
    return _baseline_decision_payload(
        selected,
        valid_actions,
        "repeat_target_fallback_baseline",
    )


def _baseline_decision_payload(
    selected: Any,
    valid_actions: Sequence[Any],
    reason: str,
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
    decision = decision_payload(selected, names, reason, candidate_policy_used=False)
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
            "max_same_action_arg_repeats": 0,
            "loop_guard_reason": "loop_guard_not_applicable_to_baseline",
        }
    )
    return decision


def _no_legal_action_decision() -> Dict[str, Any]:
    return {
        "selected_action_raw": None,
        "action": "",
        "action_args": {},
        "available_actions": [],
        "decision_reason": "no_legal_action_available",
        "candidate_policy_used": False,
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
        "max_same_action_arg_repeats": 0,
        "loop_guard_reason": "no_legal_action_available",
    }


def _policy_step_record(
    *,
    policy: str,
    step_index: int,
    game_id: str,
    decision: Mapping[str, Any],
    action: LivePrefixAction,
    before: Any,
    after: Any,
    before_signature: str,
    after_signature: str,
) -> Dict[str, Any]:
    return {
        "policy": policy,
        "step": int(step_index),
        "game_id": game_id,
        "available_actions_source": "real_env_live_api",
        "synthetic_available_actions_used": False,
        "real_env_available_actions_used": True,
        "available_actions": list(decision.get("available_actions", []) or []),
        "selected_action": action.name,
        "selected_action_args": dict(action.action_args),
        "selected_action_legal": True,
        "invalid_action_selected": False,
        "decision_reason": str(decision.get("decision_reason", "")),
        "candidate_policy_used": bool(decision.get("candidate_policy_used", False)),
        "state_signature_before": before_signature,
        "state_signature_after": after_signature,
        "state_changed": before_signature != after_signature,
        "levels_before": before.levels_completed,
        "levels_after": after.levels_completed,
        "game_state_before": before.game_state,
        "game_state_after": after.game_state,
        "terminal_after": _is_terminal(after.game_state),
        "env_actions": 1,
        "loop_guard_enabled": bool(decision.get("loop_guard_enabled", False)),
        "loop_guard_triggered": bool(decision.get("loop_guard_triggered", False)),
        "repeated_same_action_args_detected": bool(
            decision.get("repeated_same_action_args_detected", False)
        ),
        "fallback_loop_interrupted": bool(
            decision.get("fallback_loop_interrupted", False)
        ),
        "blocked_action": str(decision.get("blocked_action", "")),
        "blocked_action_args": dict(decision.get("blocked_action_args", {}) or {}),
        "support": 0,
        "truth_status": SAGE2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _error_step_record(
    policy: str,
    step_index: int,
    decision: Mapping[str, Any],
    game_id: str,
    *,
    error: str = "no_legal_action",
    before_signature: str = "",
) -> Dict[str, Any]:
    return {
        "policy": policy,
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
        "terminal_after": False,
        "support": 0,
        "truth_status": SAGE2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def summarize_policy_run(
    *,
    policy: str,
    steps: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
    initial_signature: str,
) -> Dict[str, Any]:
    executed = [row for row in steps if not row.get("invalid_action_selected", False)]
    n = len(executed)
    levels = [int(row.get("levels_after", 0) or 0) for row in executed]
    terminal_steps = sum(1 for row in executed if bool(row.get("terminal_after", False)))
    changed_steps = sum(1 for row in executed if bool(row.get("state_changed", False)))
    loop_guard_switches = sum(
        1 for row in executed if bool(row.get("fallback_loop_interrupted", False))
    )
    signatures = {initial_signature}
    for row in executed:
        signatures.add(str(row.get("state_signature_after", "")))
    repeated_pairs = 0
    previous_key = ""
    max_repeats = 0
    current_repeats = 0
    for row in executed:
        key = action_key(
            str(row.get("selected_action", "")),
            dict(row.get("selected_action_args", {}) or {}),
        )
        if key and key == previous_key:
            repeated_pairs += 1
            current_repeats += 1
        else:
            current_repeats = 1 if key else 0
        previous_key = key
        max_repeats = max(max_repeats, current_repeats)
    return {
        "policy": policy,
        "steps_executed": n,
        "invalid_action_selected": any(
            row.get("invalid_action_selected", False) for row in steps
        ),
        "selected_action_always_legal": n == len(steps) and n > 0,
        "levels_completed": max(levels) if levels else 0,
        "terminal_rate": _ratio(terminal_steps, n),
        "state_changed_rate": _ratio(changed_steps, n),
        "unique_state_signatures": len(signatures),
        "repeated_action_arg_rate": _ratio(repeated_pairs, n),
        "max_same_action_arg_repeats": int(max_repeats),
        "loop_guard_switches": int(loop_guard_switches),
        "active_counterfactual_collections": len(counterfactuals),
        "live_prefix_replay_exact_count": sum(
            1 for row in counterfactuals if bool(row.get("live_prefix_replay_exact"))
        ),
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def build_comparison(policy_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries = {
        str(run.get("policy", "")): dict(run.get("summary", {}) or {})
        for run in policy_runs
    }
    sage1b = summaries.get(POLICY_SAGE1B_GUARD, {})
    sage1 = summaries.get(POLICY_SAGE1_NO_GUARD, {})
    repeat = summaries.get(POLICY_REPEAT_FALLBACK, {})
    random_ = summaries.get(POLICY_RANDOM_LEGAL, {})

    loop_discipline_improved = bool(
        sage1b
        and sage1
        and int(sage1b.get("loop_guard_switches", 0) or 0) > 0
        and float(sage1b.get("repeated_action_arg_rate", 1.0))
        < float(sage1.get("repeated_action_arg_rate", 1.0))
    )
    explores_more_than_repeat = bool(
        sage1b
        and repeat
        and int(sage1b.get("unique_state_signatures", 0) or 0)
        >= int(repeat.get("unique_state_signatures", 0) or 0)
        and float(sage1b.get("repeated_action_arg_rate", 1.0))
        <= float(repeat.get("repeated_action_arg_rate", 0.0))
    )

    if loop_discipline_improved and explores_more_than_repeat:
        outcome_status = "SAGE_1B_IMPROVES_LOOP_DISCIPLINE_CANDIDATE_ONLY"
    elif loop_discipline_improved:
        outcome_status = "SAGE_1B_LOOP_DISCIPLINE_DELTA_ONLY_CANDIDATE_ONLY"
    else:
        outcome_status = "NO_AGENTIC_ADVANTAGE_OBSERVED_CANDIDATE_ONLY"

    metric_best = {
        metric: _best_policy(summaries, metric, lower_is_better)
        for metric, lower_is_better in (
            ("levels_completed", False),
            ("state_changed_rate", False),
            ("unique_state_signatures", False),
            ("repeated_action_arg_rate", True),
            ("terminal_rate", True),
        )
    }
    return {
        "baselines_compared": [
            POLICY_RANDOM_LEGAL,
            POLICY_REPEAT_FALLBACK,
            POLICY_SAGE1_NO_GUARD,
            POLICY_SAGE1B_GUARD,
        ],
        "per_policy_summary": summaries,
        "metric_best_policy": metric_best,
        "sage1b_loop_discipline_improved_vs_sage1": loop_discipline_improved,
        "sage1b_explores_at_least_as_much_as_repeat_fallback": explores_more_than_repeat,
        "outcome_status": outcome_status,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE2_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _best_policy(
    summaries: Mapping[str, Mapping[str, Any]],
    metric: str,
    lower_is_better: bool,
) -> Dict[str, Any]:
    best_policy = ""
    best_value: float | None = None
    for policy, summary in summaries.items():
        if metric not in summary:
            continue
        value = float(summary.get(metric, 0.0) or 0.0)
        if best_value is None:
            best_policy, best_value = policy, value
            continue
        if (lower_is_better and value < best_value) or (
            not lower_is_better and value > best_value
        ):
            best_policy, best_value = policy, value
    return {"policy": best_policy, "value": best_value}


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _is_terminal(game_state: Any) -> bool:
    return str(game_state).upper() in _TERMINAL_STATES


def _blocked_policy_run(policy: str, reason: str, budget: int) -> Dict[str, Any]:
    return {
        "policy": policy,
        "is_sage_policy": policy.startswith("sage"),
        "loop_guard_enabled": policy == POLICY_SAGE1B_GUARD,
        "steps": [],
        "active_counterfactual_collections": [],
        "summary": {
            "policy": policy,
            "steps_executed": 0,
            "blocked_reason": reason,
            "levels_completed": 0,
            "terminal_rate": 0.0,
            "state_changed_rate": 0.0,
            "unique_state_signatures": 0,
            "repeated_action_arg_rate": 0.0,
            "max_same_action_arg_repeats": 0,
            "loop_guard_switches": 0,
            "active_counterfactual_collections": 0,
            "policy_result_counted_as_confirmation": False,
            "support": 0,
            "truth_status": SAGE2_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
    }


def write_sage2_policy_probe_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH,
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
        description="Run the SAGE.2 known-game closed-loop policy probe.",
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
    parser.add_argument("--out", default=str(DEFAULT_SAGE2_POLICY_PROBE_RESULTS_PATH))
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--loop-guard-max-repeats",
        type=int,
        default=DEFAULT_LOOP_GUARD_MAX_REPEATS,
    )
    args = parser.parse_args(argv)
    run_sage2_known_game_policy_probe(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budget=args.budget,
        seed=args.seed,
        loop_guard_max_repeats=args.loop_guard_max_repeats,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
