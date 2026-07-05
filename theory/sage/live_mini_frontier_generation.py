"""SAGE.5c live mini-frontier generation from unknown live states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m2.schema import (
    FalsificationCriterion,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_READY_FOR_M3_STATUS,
    M2_TRUTH_STATUS,
    M3CandidateExperimentRequest,
)
from theory.m2.validators import validate_m3_request
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .known_game_closed_loop_scaffold import (
    DEFAULT_M2_FUSED_REQUESTS_PATH,
    DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH,
    DEFAULT_M3_FUSED_RESULTS_PATH,
    DEFAULT_P1_POLICY_PROBE_PATH,
    DEFAULT_P1_UTILITY_HANDOFF_PATH,
)
from .live_prefix_counterfactual_collector import (
    LivePrefixAction,
    state_signature_from_frame,
)
from .policy_loop_guard import action_args, action_key, action_name
from .progress_stall_trigger import (
    DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    DEFAULT_PROGRESS_STALL_WINDOW,
    DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    DEFAULT_SAME_ACTION_ARG_REPEATS,
)
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    SUBGOAL_RERUN,
    run_sage3_subgoal_switch_probe,
)
from .switch_attribution_placeholder_audit import (
    DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH,
)
from .unknown_game_bounded_probe import (
    DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    DEFAULT_UNKNOWN_GAME_ID,
)


DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage5c_live_mini_frontier_results.json"
)
SAGE5C_SCHEMA_VERSION = "sage.live_mini_frontier_generation.v1"
SAGE5C_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5C"
DEFAULT_MAX_GENERATED_REQUESTS = 20

SAGE5C_GENERATED = "SAGE_LIVE_MINI_FRONTIER_GENERATED_CANDIDATE_ONLY"
SAGE5C_NO_EFFECTIVE_GENERATION = (
    "SAGE_LIVE_MINI_FRONTIER_NO_EFFECTIVE_REQUESTS_CANDIDATE_ONLY"
)

_TERMINAL_STATES = {"GAME_OVER", "WIN", "TERMINATED", "FINISHED"}

EnvFactory = Callable[[str], Any]


def run_sage5c_live_mini_frontier_generation(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    source_sage5_path: str | Path = DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    source_sage5b_path: str | Path = DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH,
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
    max_generated_requests: int = DEFAULT_MAX_GENERATED_REQUESTS,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    source_sage5 = _load_optional_json(source_sage5_path)
    source_sage5b = _load_optional_json(source_sage5b_path)
    source_config = dict(source_sage5.get("config", {}) or {})
    resolved_game_id = str(
        game_id or source_config.get("game_id", "") or DEFAULT_UNKNOWN_GAME_ID
    )
    resolved_budgets = [
        int(value)
        for value in (
            budgets
            if budgets is not None
            else source_config.get("budgets", [50, 150, 300])
        )
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    remaining_generation_budget = max(0, int(max_generated_requests))
    per_budget: List[Dict[str, Any]] = []
    all_hypotheses: List[Dict[str, Any]] = []
    all_requests: List[Dict[str, Any]] = []
    for budget in resolved_budgets:
        run = run_sage3_subgoal_switch_probe(
            m2_fused_requests_path=m2_fused_requests_path,
            m3_fused_results_path=m3_fused_results_path,
            m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
            p1_policy_probe_path=p1_policy_probe_path,
            p1_utility_handoff_path=p1_utility_handoff_path,
            environments_dir=env_dir,
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
        generation = generate_live_mini_frontiers_for_run(
            game_id=resolved_game_id,
            budget=int(budget),
            run=run,
            environments_dir=env_dir,
            max_generated_requests=remaining_generation_budget,
            env_factory=env_factory,
        )
        remaining_generation_budget -= int(generation["effective_requests_generated"])
        all_hypotheses.extend(generation["mini_frontier_hypotheses"])
        all_requests.extend(generation["mini_frontier_m3_requests"])
        per_budget.append(_budget_record(int(budget), run, generation))

    comparison = _build_comparison(
        per_budget,
        source_sage5=source_sage5,
        source_sage5b=source_sage5b,
        max_generated_requests=max_generated_requests,
    )
    payload = {
        "config": {
            "schema_version": SAGE5C_SCHEMA_VERSION,
            "game_id": resolved_game_id,
            "budgets": resolved_budgets,
            "source_sage5_path": str(source_sage5_path),
            "source_sage5b_path": str(source_sage5b_path),
            "environments_dir": str(env_dir),
            "max_generated_requests": int(max_generated_requests),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "live_mini_frontier_generation": True,
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1", "SAGE.5", "SAGE.5b"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5_context": _source_context(source_sage5, "SAGE.5"),
        "source_sage5b_context": _source_context(source_sage5b, "SAGE.5b"),
        "per_budget_results": per_budget,
        "mini_frontier_hypotheses": all_hypotheses,
        "mini_frontier_m3_requests": all_requests,
        "comparison": comparison,
        "summary": comparison,
        "status": "UNRESOLVED",
        "outcome_status": comparison["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "generated_requests_counted_as_support": False,
        "mini_frontier_counted_as_evidence": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5c_live_mini_frontier_generation(payload, output_path)
    return payload


def generate_live_mini_frontiers_for_run(
    *,
    game_id: str,
    budget: int,
    run: Mapping[str, Any],
    environments_dir: str | Path,
    max_generated_requests: int,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Replay a SAGE run and generate mini-frontiers for rerun placeholders."""
    env_dir = Path(environments_dir)
    try:
        env = env_factory(game_id) if env_factory is not None else _make_real_env(game_id, env_dir)
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        blocked = _empty_generation()
        blocked["blocked_reason"] = f"env_setup_failed:{exc}"
        return blocked

    prefix: List[LivePrefixAction] = []
    hypotheses: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    placeholders_seen = 0
    generation_budget_exhausted = False

    for row in run.get("steps", []) or []:
        if int(row.get("env_actions", 0) or 0) <= 0:
            continue
        selected_action = str(row.get("selected_action", ""))
        selected_args = dict(row.get("selected_action_args", {}) or {})
        before_frame = frame
        before = snapshot_frame(before_frame)
        valid_actions = _valid_actions(env)
        selected = _select_action(valid_actions, selected_action, selected_args)
        if selected is None:
            diagnostics.append(
                {
                    "step": int(row.get("step", 0) or 0),
                    "blocked_reason": "replay_action_not_available",
                    "selected_action": selected_action,
                    "selected_action_args": selected_args,
                }
            )
            break
        try:
            after_frame = _step_env_action(env, selected)
        except Exception as exc:  # pragma: no cover - integration failure path
            diagnostics.append(
                {
                    "step": int(row.get("step", 0) or 0),
                    "blocked_reason": f"replay_step_failed:{exc}",
                    "selected_action": selected_action,
                    "selected_action_args": selected_args,
                }
            )
            break
        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        is_placeholder = bool(row.get("placeholder_action_used", False)) or (
            str(row.get("selected_subgoal", "")) == SUBGOAL_RERUN
        )
        if is_placeholder:
            placeholders_seen += 1
            if len(requests) < int(max_generated_requests):
                frontier = generate_live_mini_frontier(
                    game_id=game_id,
                    budget=budget,
                    step=int(row.get("step", 0) or 0),
                    before_frame=before_frame,
                    after_frame=after_frame,
                    before_snapshot=before,
                    after_snapshot=after,
                    action_name=selected_action,
                    action_args=selected_args,
                    valid_actions_before=valid_actions,
                    prefix_actions=prefix,
                )
                hypotheses.append(frontier["hypothesis"])
                request = frontier["m3_request"]
                if request["status"] == M2_READY_FOR_M3_STATUS:
                    requests.append(request)
                diagnostics.append(frontier["diagnostics"])
            else:
                generation_budget_exhausted = True
        prefix.append(
            LivePrefixAction(name=selected_action, action_args=dict(selected_args))
        )
        frame = after_frame
        if _is_terminal(after.game_state):
            # SAGE.3 emits a safe-hold row after terminal; replay stops at the
            # terminal transition.
            continue

    return {
        "placeholder_switches_seen": placeholders_seen,
        "mini_frontier_hypotheses": hypotheses,
        "mini_frontier_m3_requests": requests,
        "generation_diagnostics": diagnostics,
        "effective_requests_generated": len(requests),
        "generation_budget_exhausted": generation_budget_exhausted,
        "support": 0,
        "truth_status": SAGE5C_TRUTH_STATUS,
    }


def generate_live_mini_frontier(
    *,
    game_id: str,
    budget: int,
    step: int,
    before_frame: Any,
    after_frame: Any,
    before_snapshot: Any,
    after_snapshot: Any,
    action_name: str,
    action_args: Mapping[str, Any],
    valid_actions_before: Sequence[Any],
    prefix_actions: Sequence[LivePrefixAction],
) -> Dict[str, Any]:
    diff = live_transition_diff(
        before_snapshot.grid,
        after_snapshot.grid,
        terminal_after=_is_terminal(after_snapshot.game_state),
        levels_delta=int(after_snapshot.levels_completed)
        - int(before_snapshot.levels_completed),
    )
    hypothesis_family = _hypothesis_family(diff)
    metric = _metric_for_family(hypothesis_family)
    expected_signal = _expected_signal_for_family(hypothesis_family)
    hypothesis_id = f"sage5c::live_mini_frontier::{budget:03d}::{step:04d}"
    source_transition_id = (
        f"sage5c::{game_id}::budget_{budget:03d}::step_{step:04d}"
    )
    context_replay = tuple(action.name for action in prefix_actions)
    context_replay_args = tuple(dict(action.action_args) for action in prefix_actions)
    controls = _suggested_controls(valid_actions_before, action_name)
    hypothesis = {
        "hypothesis_id": hypothesis_id,
        "source_request_id": "sage5_placeholder::rerun_m2_m3",
        "game_id": game_id,
        "frontier_context_id": source_transition_id,
        "frontier_reason": "live_rerun_m2_m3_placeholder_replaced_by_mini_frontier",
        "frontier_step": int(step),
        "hypothesis_family": hypothesis_family,
        "candidate_action": action_name,
        "candidate_action_args": dict(action_args),
        "predicted_metric": metric,
        "predicted_effect": _predicted_effect(diff, action_name),
        "rationale": "deterministic live diff from unknown-game transition",
        "diff_summary": diff,
        "context_snapshot": {
            "replay_actions": list(context_replay),
            "replay_action_args": [dict(item) for item in context_replay_args],
            "frame_before_hash": state_signature_from_frame(before_frame),
            "live_state_signature": state_signature_from_frame(before_frame),
            "available_actions": sorted(
                {action_name_ for action_name_ in _action_names(valid_actions_before)}
            ),
            "terminal_state": bool(diff["terminal_after"]),
        },
        "source_generation": {
            "sources": ["live_mini_frontier"],
            "priority_score": 0.0,
            "priority_score_counted_as_support": False,
            "llm_used": False,
            "world_model_used": False,
        },
        "status": "UNRESOLVED",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }
    request = _m3_request(
        hypothesis_id=hypothesis_id,
        game_id=game_id,
        source_transition_id=source_transition_id,
        step=step,
        context_replay=context_replay,
        context_replay_args=context_replay_args,
        context_snapshot_hash=state_signature_from_frame(before_frame),
        target_action=action_name,
        target_action_args=dict(action_args) if action_args else None,
        suggested_control_actions=controls,
        metric=metric,
        expected_signal=expected_signal,
        hypothesis_family=hypothesis_family,
    )
    return {
        "hypothesis": hypothesis,
        "m3_request": request,
        "diagnostics": {
            "step": int(step),
            "hypothesis_id": hypothesis_id,
            "request_id": request["request_id"],
            "request_status": request["status"],
            "changed_cells": diff["changed_cells"],
            "hypothesis_family": hypothesis_family,
            "controls_available": len(controls),
            "support": 0,
            "truth_status": SAGE5C_TRUTH_STATUS,
        },
    }


def live_transition_diff(
    before_grid: Any,
    after_grid: Any,
    *,
    terminal_after: bool,
    levels_delta: int,
) -> Dict[str, Any]:
    before = np.asarray(before_grid)
    after = np.asarray(after_grid)
    changed = before != after
    ys, xs = np.where(changed)
    changed_cells = int(changed.sum())
    if changed_cells:
        bbox = {
            "x_min": int(xs.min()),
            "y_min": int(ys.min()),
            "x_max": int(xs.max()),
            "y_max": int(ys.max()),
        }
    else:
        bbox = None
    transitions: Dict[str, int] = {}
    for before_value, after_value in zip(before[changed].tolist(), after[changed].tolist()):
        key = f"{int(before_value)}->{int(after_value)}"
        transitions[key] = transitions.get(key, 0) + 1
    before_colors = {int(value) for value in np.unique(before).tolist()}
    after_colors = {int(value) for value in np.unique(after).tolist()}
    before_components = _components_by_color(before)
    after_components = _components_by_color(after)
    component_delta = {
        str(color): int(after_components.get(color, 0) - before_components.get(color, 0))
        for color in sorted(before_colors | after_colors)
        if int(after_components.get(color, 0) - before_components.get(color, 0)) != 0
    }
    return {
        "changed_cells": changed_cells,
        "changed_bbox": bbox,
        "color_transitions": dict(sorted(transitions.items())),
        "colors_added": sorted(after_colors - before_colors),
        "colors_removed": sorted(before_colors - after_colors),
        "components_before_by_color": {str(k): v for k, v in sorted(before_components.items())},
        "components_after_by_color": {str(k): v for k, v in sorted(after_components.items())},
        "component_delta_by_color": component_delta,
        "components_created_total": sum(1 for delta in component_delta.values() if delta > 0),
        "components_removed_total": sum(1 for delta in component_delta.values() if delta < 0),
        "terminal_after": bool(terminal_after),
        "levels_delta": int(levels_delta),
    }


def _m3_request(
    *,
    hypothesis_id: str,
    game_id: str,
    source_transition_id: str,
    step: int,
    context_replay: Sequence[str],
    context_replay_args: Sequence[Mapping[str, Any]],
    context_snapshot_hash: str,
    target_action: str,
    target_action_args: Mapping[str, Any] | None,
    suggested_control_actions: Sequence[str],
    metric: str,
    expected_signal: str,
    hypothesis_family: str,
) -> Dict[str, Any]:
    falsification = FalsificationCriterion(
        metric=metric,
        support_condition="target_live_effect_differs_from_dynamic_control",
        failure_condition="target_live_effect_matches_dynamic_control_or_is_unavailable",
        minimum_effect_size=1,
    )
    status = M2_READY_FOR_M3_STATUS if suggested_control_actions else "BLOCKED_NOT_TESTABLE"
    request = M3CandidateExperimentRequest(
        request_id=f"m2m3::{hypothesis_id}",
        source_hypothesis_id=hypothesis_id,
        game_id=game_id,
        context_replay=tuple(context_replay),
        context_replay_args=tuple(dict(item) for item in context_replay_args),
        context_snapshot_hash=context_snapshot_hash,
        target_action=target_action,
        target_action_args=dict(target_action_args) if target_action_args else None,
        suggested_control_actions=tuple(suggested_control_actions),
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=metric,
        expected_signal=expected_signal,
        falsification_criterion=falsification,
        status=status,
        source_episode_id=None,
        source_step=int(step),
        source_transition_id=source_transition_id,
        context_state_origin="sage5_live_prefix_frame_before",
        replayability="LIVE_PREFIX_REPLAY_CONTEXT",
        blocking_reason=None if status == M2_READY_FOR_M3_STATUS else "NO_DYNAMIC_CONTROL_AVAILABLE",
        truth_status=M2_TRUTH_STATUS,
        support=0,
        controlled_test_required=True,
        revision_performed=False,
        wrong_confirmations=0,
    )
    result = validate_m3_request(request)
    if not result.valid:
        raise ValueError(f"invalid sage5c request {request.request_id}: {result.errors}")
    payload = request.to_dict()
    payload["hypothesis_family"] = hypothesis_family
    payload["generated_by"] = "SAGE.5c_live_mini_frontier"
    payload["generated_request_counted_as_support"] = False
    return payload


def _budget_record(
    budget: int,
    run: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> Dict[str, Any]:
    summary = dict(run.get("summary", {}) or {})
    placeholders = int(generation.get("placeholder_switches_seen", 0) or 0)
    effective = int(generation.get("effective_requests_generated", 0) or 0)
    unresolved = max(0, placeholders - effective)
    total_switches = int(summary.get("subgoal_switches", 0) or 0)
    true_exploratory = int(summary.get("new_candidate_targets_discovered", 0) or 0) + int(
        summary.get("active_counterfactuals_after_exhaustion", 0) or 0
    )
    return {
        "budget": int(budget),
        "source_metrics": {
            "subgoal_switches": total_switches,
            "placeholder_switches_seen": placeholders,
            "new_candidate_targets_discovered": int(
                summary.get("new_candidate_targets_discovered", 0) or 0
            ),
            "active_counterfactuals_after_exhaustion": int(
                summary.get("active_counterfactuals_after_exhaustion", 0) or 0
            ),
            "rerun_m2_m3_requested": int(summary.get("rerun_m2_m3_requested", 0) or 0),
            "rerun_m2_m3_effective_requests_generated_before_sage5c": int(
                summary.get("rerun_m2_m3_effective_requests_generated", 0) or 0
            ),
            "levels_completed": int(summary.get("levels_completed", 0) or 0),
            "support": 0,
            "truth_status": SAGE5C_TRUTH_STATUS,
        },
        "generation_metrics": {
            "effective_requests_generated": effective,
            "mini_frontier_hypotheses_generated": len(
                generation.get("mini_frontier_hypotheses", []) or []
            ),
            "unresolved_placeholder_switches_after_generation": unresolved,
            "effective_request_ratio": _ratio(effective, placeholders),
            "source_placeholder_switch_ratio": _ratio(placeholders, total_switches),
            "residual_placeholder_switch_ratio": _ratio(unresolved, total_switches),
            "true_exploratory_or_scientific_switches": true_exploratory + effective,
            "generation_budget_exhausted": bool(
                generation.get("generation_budget_exhausted", False)
            ),
            "support": 0,
            "truth_status": SAGE5C_TRUTH_STATUS,
        },
        "generation_diagnostics": list(generation.get("generation_diagnostics", []) or [])[:5],
        "support": 0,
        "truth_status": SAGE5C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "policy_result_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _build_comparison(
    per_budget: Sequence[Mapping[str, Any]],
    *,
    source_sage5: Mapping[str, Any],
    source_sage5b: Mapping[str, Any],
    max_generated_requests: int,
) -> Dict[str, Any]:
    requested = sum(
        int(row.get("source_metrics", {}).get("placeholder_switches_seen", 0) or 0)
        for row in per_budget
    )
    effective = sum(
        int(row.get("generation_metrics", {}).get("effective_requests_generated", 0) or 0)
        for row in per_budget
    )
    total_switches = sum(
        int(row.get("source_metrics", {}).get("subgoal_switches", 0) or 0)
        for row in per_budget
    )
    unresolved = max(0, requested - effective)
    outcome = SAGE5C_GENERATED if effective > 0 else SAGE5C_NO_EFFECTIVE_GENERATION
    return {
        "source_sage5_outcome_status": str(source_sage5.get("outcome_status", "")),
        "source_sage5b_outcome_status": str(source_sage5b.get("outcome_status", "")),
        "budgets_evaluated": [int(row.get("budget", 0)) for row in per_budget],
        "rerun_m2_m3_requested": requested,
        "effective_requests_generated": effective,
        "effective_request_ratio": _ratio(effective, requested),
        "max_generated_requests": int(max_generated_requests),
        "mini_frontier_hypotheses_generated": sum(
            int(row.get("generation_metrics", {}).get("mini_frontier_hypotheses_generated", 0) or 0)
            for row in per_budget
        ),
        "unresolved_placeholder_switches_after_generation": unresolved,
        "source_placeholder_switch_ratio": _ratio(requested, total_switches),
        "residual_placeholder_switch_ratio": _ratio(unresolved, total_switches),
        "true_exploratory_or_scientific_switches": sum(
            int(row.get("generation_metrics", {}).get("true_exploratory_or_scientific_switches", 0) or 0)
            for row in per_budget
        ),
        "levels_completed_max": max(
            [int(row.get("source_metrics", {}).get("levels_completed", 0) or 0) for row in per_budget]
            or [0]
        ),
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "generated_requests_counted_as_support": False,
        "mini_frontier_counted_as_evidence": False,
        "support": 0,
        "truth_status": SAGE5C_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _hypothesis_family(diff: Mapping[str, Any]) -> str:
    if bool(diff.get("terminal_after", False)):
        return "terminal_risk_candidate"
    if int(diff.get("levels_delta", 0) or 0) > 0:
        return "objective_progress_candidate"
    if int(diff.get("changed_cells", 0) or 0) == 0:
        return "no_effect_or_stale_candidate"
    if diff.get("component_delta_by_color"):
        return "object_delta_candidate"
    return "local_patch_change_candidate"


def _metric_for_family(family: str) -> str:
    if family == "terminal_risk_candidate":
        return "terminal_state_after_rollout"
    if family == "no_effect_or_stale_candidate":
        return "state_change_after_action"
    return "local_patch_before_after"


def _expected_signal_for_family(family: str) -> str:
    return {
        "terminal_risk_candidate": "target_action_terminal_rate_exceeds_dynamic_control",
        "objective_progress_candidate": "target_action_level_progress_exceeds_dynamic_control",
        "no_effect_or_stale_candidate": "target_action_no_effect_profile_vs_dynamic_control",
        "object_delta_candidate": "target_action_object_delta_differs_from_dynamic_control",
        "local_patch_change_candidate": "target_action_local_patch_change_differs_from_dynamic_control",
    }.get(family, "target_action_effect_differs_from_dynamic_control")


def _predicted_effect(diff: Mapping[str, Any], selected_action: str) -> str:
    if bool(diff.get("terminal_after", False)):
        return f"{selected_action} may enter terminal state in this live context"
    changed = int(diff.get("changed_cells", 0) or 0)
    if changed == 0:
        return f"{selected_action} may be stale/no-effect in this live context"
    return f"{selected_action} changes {changed} cells in this live context"


def _suggested_controls(valid_actions: Sequence[Any], target_action: str) -> tuple[str, ...]:
    controls: List[str] = []
    for action in valid_actions:
        name = action_name(action)
        if not name or name == "RESET" or name == target_action:
            continue
        if name not in controls:
            controls.append(name)
    return tuple(controls)


def _action_names(valid_actions: Sequence[Any]) -> tuple[str, ...]:
    return tuple(action_name(action) for action in valid_actions if action_name(action))


def _select_action(
    valid_actions: Sequence[Any],
    name: str,
    args: Mapping[str, Any],
) -> Any | None:
    expected_args = {str(key): str(value) for key, value in dict(args or {}).items()}
    fallback = None
    for action in valid_actions:
        if action_name(action) != str(name):
            continue
        if fallback is None:
            fallback = action
        actual_args = {
            str(key): str(value)
            for key, value in dict(action_args(action) or {}).items()
        }
        if actual_args == expected_args:
            return action
    return fallback if not expected_args else None


def _components_by_color(grid: np.ndarray) -> Dict[int, int]:
    values = [int(value) for value in np.unique(grid).tolist() if int(value) != 0]
    return {value: _component_count(grid == value) for value in values}


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not bool(mask[y, x]) or bool(seen[y, x]):
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and bool(mask[ny, nx])
                        and not bool(seen[ny, nx])
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


def _empty_generation() -> Dict[str, Any]:
    return {
        "placeholder_switches_seen": 0,
        "mini_frontier_hypotheses": [],
        "mini_frontier_m3_requests": [],
        "generation_diagnostics": [],
        "effective_requests_generated": 0,
        "generation_budget_exhausted": False,
        "support": 0,
        "truth_status": SAGE5C_TRUTH_STATUS,
    }


def _source_context(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    comparison = dict(payload.get("comparison", {}) or {})
    return {
        "source_artifact": label,
        "outcome_status": str(payload.get("outcome_status", "")),
        "support": 0,
        "source_counted_as_scientific_evidence": False,
        "summary": {
            key: comparison.get(key)
            for key in (
                "budgets_gate_passed",
                "budgets_total",
                "placeholder_switch_ratio",
                "effective_request_ratio",
                "rerun_m2_m3_requested",
                "rerun_m2_m3_effective_requests_generated",
            )
            if key in comparison
        },
        "truth_status": SAGE5C_TRUTH_STATUS,
    }


def write_sage5c_live_mini_frontier_generation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH,
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


def _load_optional_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _is_terminal(game_state: Any) -> bool:
    return str(game_state).upper() in _TERMINAL_STATES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.5c live mini-frontier generation.",
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
    parser.add_argument("--source-sage5b", default=str(DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH))
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
        "--max-generated-requests",
        type=int,
        default=DEFAULT_MAX_GENERATED_REQUESTS,
    )
    args = parser.parse_args(argv)
    run_sage5c_live_mini_frontier_generation(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        source_sage5_path=args.source_sage5,
        source_sage5b_path=args.source_sage5b,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budgets=args.budgets,
        max_counterfactual_collections=args.max_counterfactual_collections,
        progress_stall_window=args.progress_stall_window,
        same_action_arg_repeats=args.same_action_arg_repeats,
        low_state_novelty_threshold=args.low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=args.repeated_action_arg_rate_threshold,
        max_generated_requests=args.max_generated_requests,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
