"""SAGE.3 subgoal switch after success-like exhaustion.

SAGE.1b removed a degenerate fallback loop, but after the success-like ACTION6
targets are exhausted (or the loop guard would trigger), the runner still does
not know *what to look for next*. SAGE.3 is a mode-transition module, not a
better guard: when exhaustion/loop is detected it chooses a new, legal,
non-repetitive subgoal among a small set of candidate modes:

1. ``active_counterfactual_collection`` - collect a live counterfactual
2. ``repositioning`` - move with the repositioning action
3. ``explore_new_candidate_action6_target`` - try an unused ACTION6 target
4. ``rerun_m2_m3`` - flag a re-derivation request and take a neutral legal action
5. ``stop_safe_hold`` - hold when a terminal risk is present

The minimal criterion is NOT winning: after exhaustion, SAGE must produce a new
behaviour that is non-trivial, legal, non-repetitive and measurable. Every
artefact keeps ``support=0``,
``policy_result_counted_as_confirmation=false`` and
``truth_status=NOT_EVALUATED_BY_SAGE_3``. No A32/A33 write is performed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
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
    select_from_actions,
    select_runner_action,
)
from .live_prefix_counterfactual_collector import (
    LivePrefixAction,
    collect_live_prefix_counterfactual,
    state_signature_from_frame,
)
from .policy_loop_guard import action_args, action_key, action_name
from .progress_stall_trigger import (
    PROGRESS_STALL_TRIGGER_REASON,
    ProgressStallTriggerConfig,
    evaluate_progress_stall_trigger,
)


DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage3_subgoal_switch_results.json"
)
SAGE3_SCHEMA_VERSION = "sage.subgoal_switch_results.v1"
SAGE3_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_3"
DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_BUDGET = 20
DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS = 8

TRIGGER_REASON = "success_like_targets_exhausted_or_loop_guard"
EXHAUSTION_DECISION_REASON = "candidate_policy_live_target_fallback"

SUBGOAL_SAFE_HOLD = "stop_safe_hold"
SUBGOAL_REPOSITION = "repositioning"
SUBGOAL_EXPLORE_TARGET = "explore_new_candidate_action6_target"
SUBGOAL_COUNTERFACTUAL = "active_counterfactual_collection"
SUBGOAL_RERUN = "rerun_m2_m3"
DEFAULT_SUBGOAL_ORDER: tuple[str, ...] = (
    SUBGOAL_REPOSITION,
    SUBGOAL_EXPLORE_TARGET,
    SUBGOAL_COUNTERFACTUAL,
    SUBGOAL_RERUN,
)

_TERMINAL_STATES = {"GAME_OVER", "WIN", "TERMINATED", "FINISHED"}

EnvFactory = Callable[[str], Any]


@dataclass(frozen=True)
class SubgoalSwitchDecision:
    """Decision metadata for one subgoal switch event."""

    subgoal_switch_triggered: bool
    trigger_reason: str
    trigger_detail: str
    selected_subgoal: str
    selected_action_raw: Any | None
    action: str = ""
    action_args: Dict[str, Any] = field(default_factory=dict)
    selected_action_legal: bool = False
    collect_counterfactual: bool = False
    rerun_m2_m3_requested: bool = False
    rerun_m2_m3_executed: bool = False
    placeholder_action_used: bool = False
    placeholder_action_counted_as_subgoal_success: bool = False
    safe_hold: bool = False
    new_candidate_target: bool = False
    feasible_subgoals: tuple[str, ...] = ()
    blocked_action: str = ""
    blocked_action_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_switch_triggered": self.subgoal_switch_triggered,
            "trigger_reason": self.trigger_reason,
            "trigger_detail": self.trigger_detail,
            "selected_subgoal": self.selected_subgoal,
            "selected_action": self.action,
            "selected_action_args": dict(self.action_args),
            "selected_action_legal": self.selected_action_legal,
            "collect_counterfactual": self.collect_counterfactual,
            "rerun_m2_m3_requested": self.rerun_m2_m3_requested,
            "rerun_m2_m3_executed": self.rerun_m2_m3_executed,
            "placeholder_action_used": self.placeholder_action_used,
            "placeholder_action_counted_as_subgoal_success": (
                self.placeholder_action_counted_as_subgoal_success
            ),
            "safe_hold": self.safe_hold,
            "new_candidate_target": self.new_candidate_target,
            "feasible_subgoals": list(self.feasible_subgoals),
            "blocked_action": self.blocked_action,
            "blocked_action_args": dict(self.blocked_action_args),
            "policy_result_counted_as_confirmation": False,
            "support": 0,
            "truth_status": SAGE3_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
            "a32_write_performed": False,
            "a33_write_performed": False,
        }


def decide_subgoal_switch(
    *,
    valid_actions: Sequence[Any],
    prefix: Sequence[LivePrefixAction],
    policy_memory: Mapping[str, Any],
    frontier: Mapping[str, Any] | None,
    blocked_action: str = "",
    blocked_action_args: Mapping[str, Any] | None = None,
    last_game_state: str = "",
    switch_index: int = 0,
    trigger_reason: str = TRIGGER_REASON,
    trigger_detail: str = "success_like_targets_exhausted",
    subgoal_order: Sequence[str] = DEFAULT_SUBGOAL_ORDER,
) -> SubgoalSwitchDecision:
    """Pick a new legal, non-repetitive subgoal after exhaustion/loop."""
    blocked_key = action_key(blocked_action, dict(blocked_action_args or {}))
    last_key = ""
    if prefix:
        last = prefix[-1]
        last_key = action_key(last.name, last.action_args)

    target = str(policy_memory.get("target_action", "ACTION6") or "ACTION6")
    reposition = str(policy_memory.get("repositioning_action", "ACTION4") or "ACTION4")
    frontier_action = str((frontier or {}).get("target_action", "") or "ACTION3")
    frontier_args = dict((frontier or {}).get("target_action_args") or {})
    used_target_args = {
        tuple(sorted(action.action_args.items()))
        for action in prefix
        if action.name == target and action.action_args
    }

    if _is_terminal(last_game_state):
        return SubgoalSwitchDecision(
            subgoal_switch_triggered=True,
            trigger_reason=trigger_reason,
            trigger_detail="terminal_risk_safe_hold",
            selected_subgoal=SUBGOAL_SAFE_HOLD,
            selected_action_raw=None,
            selected_action_legal=True,
            safe_hold=True,
            feasible_subgoals=(SUBGOAL_SAFE_HOLD,),
            blocked_action=blocked_action,
            blocked_action_args=dict(blocked_action_args or {}),
        )

    candidates: Dict[str, Any] = {}
    rep = _pick_distinct(valid_actions, reposition, {}, blocked_key, last_key)
    if rep is not None:
        candidates[SUBGOAL_REPOSITION] = rep
    explore = _pick_new_target(valid_actions, target, used_target_args, blocked_key, last_key)
    if explore is not None:
        candidates[SUBGOAL_EXPLORE_TARGET] = explore
    cf = _pick_distinct(
        valid_actions, frontier_action, frontier_args, blocked_key, last_key
    )
    if cf is not None:
        candidates[SUBGOAL_COUNTERFACTUAL] = cf
    exclude_keys = {_obj_key(action) for action in candidates.values()}
    rerun = _pick_any_legal(valid_actions, blocked_key, last_key, exclude_keys)
    if rerun is not None:
        candidates[SUBGOAL_RERUN] = rerun

    feasible = tuple(mode for mode in subgoal_order if mode in candidates)
    if not feasible:
        fallback = _pick_any_legal(valid_actions, blocked_key, "", set())
        if fallback is None:
            return SubgoalSwitchDecision(
                subgoal_switch_triggered=True,
                trigger_reason=trigger_reason,
                trigger_detail="no_legal_subgoal_action_available",
                selected_subgoal=SUBGOAL_SAFE_HOLD,
                selected_action_raw=None,
                selected_action_legal=False,
                safe_hold=True,
                blocked_action=blocked_action,
                blocked_action_args=dict(blocked_action_args or {}),
            )
        chosen_mode = SUBGOAL_RERUN
        chosen = fallback
        feasible = (SUBGOAL_RERUN,)
    else:
        chosen_mode = feasible[int(switch_index) % len(feasible)]
        chosen = candidates[chosen_mode]

    is_rerun = chosen_mode == SUBGOAL_RERUN
    return SubgoalSwitchDecision(
        subgoal_switch_triggered=True,
        trigger_reason=trigger_reason,
        trigger_detail=trigger_detail,
        selected_subgoal=chosen_mode,
        selected_action_raw=chosen,
        action=action_name(chosen),
        action_args=action_args(chosen),
        selected_action_legal=True,
        collect_counterfactual=chosen_mode == SUBGOAL_COUNTERFACTUAL,
        rerun_m2_m3_requested=is_rerun,
        rerun_m2_m3_executed=False,
        placeholder_action_used=is_rerun,
        placeholder_action_counted_as_subgoal_success=False,
        new_candidate_target=chosen_mode == SUBGOAL_EXPLORE_TARGET,
        feasible_subgoals=feasible,
        blocked_action=blocked_action,
        blocked_action_args=dict(blocked_action_args or {}),
    )


def _pick_distinct(
    valid_actions: Sequence[Any],
    name: str,
    args: Mapping[str, Any],
    blocked_key: str,
    last_key: str,
) -> Any | None:
    selected = select_from_actions(valid_actions, name, dict(args or {}))
    if selected is None:
        return None
    key = _obj_key(selected)
    if key == blocked_key or (last_key and key == last_key):
        return None
    return selected


def _pick_new_target(
    valid_actions: Sequence[Any],
    target: str,
    used_args: set,
    blocked_key: str,
    last_key: str,
) -> Any | None:
    for action in valid_actions:
        if action_name(action) != str(target):
            continue
        args = action_args(action)
        if not args:
            continue
        if tuple(sorted(args.items())) in used_args:
            continue
        key = action_key(str(target), args)
        if key == blocked_key or (last_key and key == last_key):
            continue
        return action
    return None


def _pick_any_legal(
    valid_actions: Sequence[Any],
    blocked_key: str,
    last_key: str,
    exclude_keys: set,
) -> Any | None:
    for action in valid_actions:
        name = action_name(action)
        if not name or name == "RESET":
            continue
        key = _obj_key(action)
        if key == blocked_key:
            continue
        if last_key and key == last_key:
            continue
        if key in exclude_keys:
            continue
        return action
    return None


def _obj_key(action: Any) -> str:
    return action_key(action_name(action), action_args(action))


def run_sage3_subgoal_switch_probe(
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
    subgoal_order: Sequence[str] = DEFAULT_SUBGOAL_ORDER,
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    env_factory: EnvFactory | None = None,
    enable_progress_stall_trigger: bool = False,
    progress_stall_window: int = 8,
    same_action_arg_repeats: int = 4,
    low_state_novelty_threshold: int = 3,
    repeated_action_arg_rate_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Run a SAGE.1b loop that switches subgoals after success-like exhaustion."""
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

    run = _run_sage3_loop(
        game_id=game_id,
        budget=budget,
        env_dir=env_dir,
        env_factory=env_factory,
        policy_memory=policy_memory,
        frontier=frontier,
        subgoal_order=subgoal_order,
        max_counterfactual_collections=max_counterfactual_collections,
        enable_progress_stall_trigger=enable_progress_stall_trigger,
        progress_stall_window=progress_stall_window,
        same_action_arg_repeats=same_action_arg_repeats,
        low_state_novelty_threshold=low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=repeated_action_arg_rate_threshold,
    )

    summary = run["summary"]
    payload = {
        "config": {
            **_config_paths(
                m2_fused_requests_path,
                m3_fused_results_path,
                m3_counterfactual_feasibility_path,
                p1_policy_probe_path,
                p1_utility_handoff_path,
            ),
            "schema_version": SAGE3_SCHEMA_VERSION,
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "budget": int(budget),
            "subgoal_order": list(subgoal_order),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "enable_progress_stall_trigger": bool(enable_progress_stall_trigger),
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "subgoal_switch_probe": True,
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "input_summaries": {
            "hypothesis_context": hypothesis_context_summary(inputs),
            "m3_tests": m3_tests_summary(inputs),
            "policy_context": policy_context_summary(inputs),
        },
        "steps": run["steps"],
        "subgoal_switch_events": run["subgoal_switch_events"],
        "active_counterfactual_collections": run["counterfactuals"],
        "summary": summary,
        "subgoal_switch_triggered": bool(summary.get("subgoal_switches", 0) > 0),
        "trigger_reason": TRIGGER_REASON,
        "selected_subgoal": summary.get("subgoals_used", []),
        "selected_action_legal": bool(summary.get("selected_action_always_legal", False)),
        "status": "UNRESOLVED",
        "outcome_status": summary.get("outcome_status", ""),
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE3_TRUTH_STATUS,
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
        write_sage3_subgoal_switch_results(payload, output_path)
    return payload


def _run_sage3_loop(
    *,
    game_id: str,
    budget: int,
    env_dir: Path,
    env_factory: EnvFactory | None,
    policy_memory: Mapping[str, Any],
    frontier: Mapping[str, Any] | None,
    subgoal_order: Sequence[str],
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    enable_progress_stall_trigger: bool = False,
    progress_stall_window: int = 8,
    same_action_arg_repeats: int = 4,
    low_state_novelty_threshold: int = 3,
    repeated_action_arg_rate_threshold: float = 0.75,
) -> Dict[str, Any]:
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, env_dir)
        )
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return {
            "steps": [],
            "subgoal_switch_events": [],
            "counterfactuals": [],
            "summary": _blocked_summary(f"env_setup_failed:{exc}"),
        }

    prefix: List[LivePrefixAction] = []
    steps: List[Dict[str, Any]] = []
    switch_events: List[Dict[str, Any]] = []
    counterfactuals: List[Dict[str, Any]] = []
    initial_signature = state_signature_from_frame(frame)
    switch_count = 0
    last_game_state = snapshot_frame(frame).game_state
    discovered_targets: set = set()
    previous_key = ""

    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(frame)
        if _is_terminal(last_game_state):
            terminal_hold = SubgoalSwitchDecision(
                subgoal_switch_triggered=True,
                trigger_reason="terminal_state_safe_hold",
                trigger_detail="terminal_state_reached",
                selected_subgoal=SUBGOAL_SAFE_HOLD,
                selected_action_raw=None,
                selected_action_legal=True,
                safe_hold=True,
                feasible_subgoals=(SUBGOAL_SAFE_HOLD,),
            )
            switch_events.append(terminal_hold.to_dict())
            steps.append(_safe_hold_step(step_index, game_id, before, terminal_hold))
            break
        valid_actions = _valid_actions(env)
        base = select_runner_action(
            step_index=step_index,
            valid_actions=valid_actions,
            policy_memory=policy_memory,
            prefix=prefix,
            frontier=frontier,
            enable_loop_guard=False,
        )
        if base.get("selected_action_raw") is None:
            steps.append(_error_step(step_index, game_id, base))
            break

        exhausted = str(base.get("decision_reason", "")) == EXHAUSTION_DECISION_REASON
        stall = None
        if enable_progress_stall_trigger:
            stall = evaluate_progress_stall_trigger(
                steps=steps,
                proposed_action=str(base.get("action", "")),
                proposed_action_args=dict(base.get("action_args", {}) or {}),
                config=ProgressStallTriggerConfig(
                    window_size=progress_stall_window,
                    same_action_arg_repeats=same_action_arg_repeats,
                    low_state_novelty_threshold=low_state_novelty_threshold,
                    repeated_action_arg_rate_threshold=(
                        repeated_action_arg_rate_threshold
                    ),
                ),
            )
        progress_stall = bool(stall and stall.switch_required)
        switch_decision: SubgoalSwitchDecision | None = None
        if exhausted or progress_stall:
            trigger_reason = (
                PROGRESS_STALL_TRIGGER_REASON if progress_stall else TRIGGER_REASON
            )
            trigger_detail = (
                stall.trigger_reason if progress_stall and stall else "success_like_targets_exhausted"
            )
            switch_decision = decide_subgoal_switch(
                valid_actions=valid_actions,
                prefix=prefix,
                policy_memory=policy_memory,
                frontier=frontier,
                blocked_action=str(base.get("action", "")),
                blocked_action_args=dict(base.get("action_args", {}) or {}),
                last_game_state=last_game_state,
                switch_index=switch_count,
                trigger_reason=trigger_reason,
                trigger_detail=trigger_detail,
                subgoal_order=subgoal_order,
            )
            switch_count += 1
            event = switch_decision.to_dict()
            if stall is not None:
                event["progress_stall_trigger"] = stall.to_dict()
            switch_events.append(event)
            if switch_decision.safe_hold and switch_decision.selected_action_raw is None:
                steps.append(
                    _safe_hold_step(
                        step_index,
                        game_id,
                        before,
                        switch_decision,
                    )
                )
                break
            selected_raw = switch_decision.selected_action_raw
            decision_reason = f"subgoal_switch:{switch_decision.selected_subgoal}"
        else:
            selected_raw = base["selected_action_raw"]
            decision_reason = str(base.get("decision_reason", ""))

        before_signature = state_signature_from_frame(frame)
        error_decision = {
            **dict(base),
            "action": action_name(selected_raw),
            "action_args": action_args(selected_raw),
            "decision_reason": decision_reason,
        }
        try:
            after_frame = _step_env_action(env, selected_raw)
            after = snapshot_frame(
                after_frame, fallback_available_actions=before.available_actions
            )
            after_signature = state_signature_from_frame(after_frame)
        except Exception as exc:  # pragma: no cover - integration failure path
            steps.append(
                _error_step(
                    step_index,
                    game_id,
                    error_decision,
                    error=f"env_step_failed:{exc}",
                    before_signature=before_signature,
                )
            )
            break
        action = LivePrefixAction(
            name=action_name(selected_raw),
            action_args=action_args(selected_raw),
        )
        prefix.append(action)
        action_kv = action_key(action.name, action.action_args)
        is_switch = switch_decision is not None
        if is_switch and switch_decision.new_candidate_target and action.action_args:
            discovered_targets.add(tuple(sorted(action.action_args.items())))
        steps.append(
            _step_record(
                step_index=step_index,
                game_id=game_id,
                action=action,
                before=before,
                after=after,
                before_signature=before_signature,
                after_signature=after_signature,
                decision_reason=decision_reason,
                is_switch=is_switch,
                switch_decision=switch_decision,
                repeated=bool(previous_key and action_kv == previous_key),
                progress_stall_trigger=stall.to_dict() if stall is not None else None,
            )
        )
        previous_key = action_kv
        last_game_state = after.game_state
        frame = after_frame

        if (
            is_switch
            and switch_decision.collect_counterfactual
            and frontier
            and len(counterfactuals) < int(max_counterfactual_collections)
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
            row = result.to_dict()
            row["collected_after_exhaustion"] = True
            counterfactuals.append(row)

    summary = _summarize_sage3(
        steps=steps,
        switch_events=switch_events,
        counterfactuals=counterfactuals,
        initial_signature=initial_signature,
        discovered_targets=discovered_targets,
    )
    return {
        "steps": steps,
        "subgoal_switch_events": switch_events,
        "counterfactuals": counterfactuals,
        "summary": summary,
    }


def _step_record(
    *,
    step_index: int,
    game_id: str,
    action: LivePrefixAction,
    before: Any,
    after: Any,
    before_signature: str,
    after_signature: str,
    decision_reason: str,
    is_switch: bool,
    switch_decision: SubgoalSwitchDecision | None,
    repeated: bool,
    progress_stall_trigger: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "step": int(step_index),
        "game_id": game_id,
        "available_actions_source": "real_env_live_api",
        "synthetic_available_actions_used": False,
        "real_env_available_actions_used": True,
        "selected_action": action.name,
        "selected_action_args": dict(action.action_args),
        "selected_action_legal": True,
        "invalid_action_selected": False,
        "decision_reason": decision_reason,
        "is_subgoal_switch": bool(is_switch),
        "selected_subgoal": (
            switch_decision.selected_subgoal if switch_decision is not None else ""
        ),
        "subgoal_switch_triggered": bool(is_switch),
        "trigger_reason": (
            switch_decision.trigger_reason if switch_decision is not None else ""
        ),
        "progress_stall_trigger": dict(progress_stall_trigger or {}),
        "progress_stall_detected": (
            str(switch_decision.trigger_reason) == PROGRESS_STALL_TRIGGER_REASON
            if switch_decision is not None
            else False
        ),
        "new_candidate_target": (
            bool(switch_decision.new_candidate_target) if switch_decision else False
        ),
        "rerun_m2_m3_requested": (
            bool(switch_decision.rerun_m2_m3_requested) if switch_decision else False
        ),
        "rerun_m2_m3_executed": (
            bool(switch_decision.rerun_m2_m3_executed) if switch_decision else False
        ),
        "placeholder_action_used": (
            bool(switch_decision.placeholder_action_used) if switch_decision else False
        ),
        "placeholder_action_counted_as_subgoal_success": (
            bool(switch_decision.placeholder_action_counted_as_subgoal_success)
            if switch_decision
            else False
        ),
        "state_signature_before": before_signature,
        "state_signature_after": after_signature,
        "state_changed": before_signature != after_signature,
        "repeated_previous_action": bool(repeated),
        "levels_before": before.levels_completed,
        "levels_after": after.levels_completed,
        "game_state_before": before.game_state,
        "game_state_after": after.game_state,
        "terminal_after": _is_terminal(after.game_state),
        "env_actions": 1,
        "support": 0,
        "truth_status": SAGE3_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _safe_hold_step(
    step_index: int,
    game_id: str,
    before: Any,
    switch_decision: SubgoalSwitchDecision,
) -> Dict[str, Any]:
    return {
        "step": int(step_index),
        "game_id": game_id,
        "selected_action": "",
        "selected_action_args": {},
        "selected_action_legal": True,
        "invalid_action_selected": False,
        "decision_reason": f"subgoal_switch:{switch_decision.selected_subgoal}",
        "is_subgoal_switch": True,
        "selected_subgoal": switch_decision.selected_subgoal,
        "subgoal_switch_triggered": True,
        "trigger_reason": switch_decision.trigger_reason,
        "safe_hold": True,
        "env_actions": 0,
        "state_changed": False,
        "repeated_previous_action": False,
        "terminal_after": _is_terminal(before.game_state),
        "support": 0,
        "truth_status": SAGE3_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _error_step(
    step_index: int,
    game_id: str,
    decision: Mapping[str, Any],
    *,
    error: str = "no_legal_action",
    before_signature: str = "",
) -> Dict[str, Any]:
    return {
        "step": int(step_index),
        "game_id": game_id,
        "selected_action": str(decision.get("action", "")),
        "selected_action_args": dict(decision.get("action_args", {}) or {}),
        "selected_action_legal": False,
        "invalid_action_selected": True,
        "decision_reason": str(decision.get("decision_reason", "")),
        "error": error,
        "is_subgoal_switch": False,
        "state_signature_before": before_signature,
        "env_actions": 0,
        "terminal_after": False,
        "support": 0,
        "truth_status": SAGE3_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _summarize_sage3(
    *,
    steps: Sequence[Mapping[str, Any]],
    switch_events: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
    initial_signature: str,
    discovered_targets: set,
) -> Dict[str, Any]:
    executed = [row for row in steps if not row.get("invalid_action_selected", False)]
    env_steps = [row for row in executed if int(row.get("env_actions", 0) or 0) > 0]
    switch_steps = [row for row in env_steps if bool(row.get("is_subgoal_switch", False))]
    n = len(env_steps)
    levels = [int(row.get("levels_after", 0) or 0) for row in env_steps]

    signatures = {initial_signature}
    for row in env_steps:
        signatures.add(str(row.get("state_signature_after", "")))
    repeated_pairs = sum(
        1 for row in env_steps if bool(row.get("repeated_previous_action", False))
    )

    switch_successes = sum(
        1
        for row in switch_steps
        if bool(row.get("selected_action_legal", False))
        and bool(row.get("state_changed", False))
        and not bool(row.get("repeated_previous_action", False))
    )
    subgoals_used = sorted(
        {
            str(row.get("selected_subgoal", ""))
            for row in switch_steps
            if str(row.get("selected_subgoal", ""))
        }
    )
    switch_state_changed = sum(
        1 for row in switch_steps if bool(row.get("state_changed", False))
    )
    switch_repeats = sum(
        1 for row in switch_steps if bool(row.get("repeated_previous_action", False))
    )
    switch_terminal = sum(
        1 for row in switch_steps if bool(row.get("terminal_after", False))
    )
    progress_stall_switches = sum(
        1
        for row in switch_steps
        if str(row.get("trigger_reason", "")) == PROGRESS_STALL_TRIGGER_REASON
    )
    rerun_requested = sum(
        1 for row in switch_steps if bool(row.get("rerun_m2_m3_requested", False))
    )
    rerun_executed = sum(
        1 for row in switch_steps if bool(row.get("rerun_m2_m3_executed", False))
    )
    placeholder_steps = sum(
        1 for row in switch_steps if bool(row.get("placeholder_action_used", False))
    )

    non_trivial = bool(switch_steps) and switch_successes > 0
    diverse = len(subgoals_used) >= 2
    if non_trivial and diverse:
        outcome_status = "SAGE_SWITCHES_TO_NEW_LEGAL_SUBGOAL_CANDIDATE_ONLY"
    elif non_trivial:
        outcome_status = "SAGE_SWITCHES_TO_SINGLE_NEW_SUBGOAL_CANDIDATE_ONLY"
    elif switch_steps:
        outcome_status = "SAGE_SWITCH_TRIVIAL_OR_REPETITIVE_CANDIDATE_ONLY"
    else:
        outcome_status = "NO_SUBGOAL_SWITCH_TRIGGERED_CANDIDATE_ONLY"

    return {
        "steps_executed": len(executed),
        "env_steps": n,
        "selected_action_always_legal": len(executed) == len(steps) and bool(steps),
        "invalid_action_selected": any(
            row.get("invalid_action_selected", False) for row in steps
        ),
        "levels_completed": max(levels) if levels else 0,
        "terminal_rate": _ratio(
            sum(1 for row in env_steps if bool(row.get("terminal_after", False))), n
        ),
        "state_changed_rate": _ratio(
            sum(1 for row in env_steps if bool(row.get("state_changed", False))), n
        ),
        "unique_state_signatures": len(signatures),
        "repeated_action_arg_rate": _ratio(repeated_pairs, n),
        "subgoal_switches": len(switch_steps),
        "progress_stall_detected": progress_stall_switches > 0,
        "progress_stall_switches": int(progress_stall_switches),
        "subgoal_switch_success_rate": _ratio(switch_successes, len(switch_steps)),
        "subgoals_used": subgoals_used,
        "new_candidate_targets_discovered": len(discovered_targets),
        "active_counterfactuals_after_exhaustion": sum(
            1 for row in counterfactuals if bool(row.get("collected_after_exhaustion"))
        ),
        "rerun_m2_m3_requested": int(rerun_requested),
        "rerun_m2_m3_effective_requests_generated": int(rerun_executed),
        "placeholder_actions_used": int(placeholder_steps),
        "placeholder_action_counted_as_subgoal_success": False,
        "post_switch_state_changed_rate": _ratio(switch_state_changed, len(switch_steps)),
        "post_switch_repeat_rate": _ratio(switch_repeats, len(switch_steps)),
        "post_switch_terminal_rate": _ratio(switch_terminal, len(switch_steps)),
        "outcome_status": outcome_status,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE3_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _blocked_summary(reason: str) -> Dict[str, Any]:
    return {
        "steps_executed": 0,
        "env_steps": 0,
        "blocked_reason": reason,
        "selected_action_always_legal": False,
        "levels_completed": 0,
        "subgoal_switches": 0,
        "subgoal_switch_success_rate": 0.0,
        "subgoals_used": [],
        "new_candidate_targets_discovered": 0,
        "active_counterfactuals_after_exhaustion": 0,
        "outcome_status": "BLOCKED_CANDIDATE_ONLY",
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE3_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _is_terminal(game_state: Any) -> bool:
    return str(game_state).upper() in _TERMINAL_STATES


def write_sage3_subgoal_switch_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH,
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
        description="Run the SAGE.3 subgoal switch probe.",
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
    parser.add_argument("--out", default=str(DEFAULT_SAGE3_SUBGOAL_SWITCH_RESULTS_PATH))
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--max-counterfactual-collections",
        type=int,
        default=DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    )
    args = parser.parse_args(argv)
    run_sage3_subgoal_switch_probe(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budget=args.budget,
        max_counterfactual_collections=args.max_counterfactual_collections,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
