"""M3.O2 executor for objective stop/switch experiment cells."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.p1.bp35_sage_candidate_policy_probe import (
    PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
    ProbeDecision,
    ProbeStep,
    candidate_policy_memory_from_scope,
    classify_action6_args,
    concrete_action_for_decision,
    error_step,
    measure_probe_metrics,
    select_probe_decision,
    state_signature,
    summarize_probe_steps,
)
from theory.real_env_option_adapter import snapshot_frame

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_stop_switch_experiment_planner import (
    DEFAULT_OBJECTIVE_STOP_SWITCH_REQUESTS_OUTPUT_PATH,
    READY_FOR_M3_OBJECTIVE_EXPERIMENT,
)


DEFAULT_OBJECTIVE_STOP_SWITCH_RESULTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "objective_stop_switch_experiment_results.json"
)
OBJECTIVE_EXECUTED = "EXECUTED"
OBJECTIVE_STOP_OBSERVED = "STOP_PREFIX_ENDPOINT_OBSERVED"
EARLY_TERMINAL_DURING_PREFIX = "EARLY_TERMINAL_DURING_PREFIX"
BLOCKED_CONTROL_UNAVAILABLE = "BLOCKED_CONTROL_UNAVAILABLE"
NO_EFFECTIVE_PREFIX_ACTION_AVAILABLE = "NO_EFFECTIVE_PREFIX_ACTION_AVAILABLE"
STOP_OR_HOLD_SEMANTICS = "observe_prefix_endpoint_without_extra_action"
PREFIX_POLICY = "patch_similarity_soft_stale_action6_prefix"


@dataclass(frozen=True)
class ObjectiveExecutionCell:
    """One unique objective execution cell shared by one or more hypotheses."""

    cell_signature: str
    game_id: str
    prefix_policy: str
    prefix_action: str
    prefix_length: int
    condition_id: str
    condition_family: str
    post_prefix_policy: str
    post_prefix_action: str | None
    tie_break_seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_signature": self.cell_signature,
            "game_id": self.game_id,
            "prefix_policy": self.prefix_policy,
            "prefix_action": self.prefix_action,
            "prefix_length": int(self.prefix_length),
            "condition_id": self.condition_id,
            "condition_family": self.condition_family,
            "post_prefix_policy": self.post_prefix_policy,
            "post_prefix_action": self.post_prefix_action,
            "tie_break_seed": int(self.tie_break_seed),
        }


def run_objective_stop_switch_experiment_execution(
    *,
    objective_requests_path: str | Path = DEFAULT_OBJECTIVE_STOP_SWITCH_REQUESTS_OUTPUT_PATH,
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    tie_break_seed: int = 0,
    max_cells: int | None = None,
    cell_executor: Callable[[ObjectiveExecutionCell], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Execute deduplicated objective cells without producing scientific verdicts."""
    request_payload = _load_json(objective_requests_path)
    _validate_source_request_payload(request_payload)
    requests = ready_objective_requests(request_payload)
    planned_cells, links = build_execution_cells(
        requests,
        tie_break_seed=tie_break_seed,
    )
    unique_cells = unique_execution_cells(planned_cells)
    if max_cells is not None:
        unique_cells = unique_cells[: max(0, int(max_cells))]
    executed_signatures = {cell.cell_signature for cell in unique_cells}
    executable_links = [
        link for link in links if str(link.get("cell_signature", "")) in executed_signatures
    ]

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    if cell_executor is None:
        _configure_offline_env(env_dir)
        memory = candidate_policy_memory_from_scope(_load_json(scope_consolidation_path))

        def default_executor(cell: ObjectiveExecutionCell) -> Mapping[str, Any]:
            return execute_objective_cell(
                cell,
                memory=memory,
                environments_dir=env_dir,
            )

        cell_executor = default_executor

    cell_results = [dict(cell_executor(cell)) for cell in unique_cells]
    result_by_signature = {
        str(row.get("cell_signature", "")): dict(row) for row in cell_results
    }
    linked_observations = [
        build_hypothesis_observation_link(
            link,
            result_by_signature.get(str(link.get("cell_signature", "")), {}),
        )
        for link in executable_links
    ]
    return build_results_payload(
        objective_requests_path=objective_requests_path,
        scope_consolidation_path=scope_consolidation_path,
        environments_dir=env_dir,
        requests=requests,
        planned_cells=planned_cells,
        unique_cells=unique_cells,
        cell_results=cell_results,
        hypothesis_observation_links=linked_observations,
    )


def build_execution_cells(
    requests: Sequence[Mapping[str, Any]],
    *,
    tie_break_seed: int = 0,
) -> Tuple[Tuple[ObjectiveExecutionCell, ...], Tuple[Dict[str, Any], ...]]:
    cells: list[ObjectiveExecutionCell] = []
    links: list[Dict[str, Any]] = []
    for request in requests:
        for prefix_length in [int(value) for value in request.get("prefix_lengths", [])]:
            for condition in request.get("experimental_conditions", []) or []:
                if not isinstance(condition, Mapping):
                    continue
                cell = make_execution_cell(
                    request,
                    condition,
                    prefix_length=prefix_length,
                    tie_break_seed=tie_break_seed,
                )
                cells.append(cell)
                links.append(
                    {
                        "request_id": str(request.get("request_id", "")),
                        "source_hypothesis_id": str(
                            request.get("source_hypothesis_id", "")
                        ),
                        "hypothesis_family": str(request.get("hypothesis_family", "")),
                        "cell_signature": cell.cell_signature,
                        "game_id": cell.game_id,
                        "prefix_length": cell.prefix_length,
                        "condition_id": cell.condition_id,
                        "duplicate_execution_cell_counted_as_independent": False,
                        "support": 0,
                        "revision_status": "CANDIDATE_ONLY",
                        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                        "wrong_confirmations": 0,
                    }
                )
    return tuple(cells), tuple(links)


def make_execution_cell(
    request: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    prefix_length: int,
    tie_break_seed: int,
) -> ObjectiveExecutionCell:
    game_id = str(request.get("game_id", ""))
    prefix_action = str(request.get("prefix_action", "ACTION6") or "ACTION6")
    condition_id = str(condition.get("condition_id", ""))
    condition_family = str(condition.get("condition_family", ""))
    post_prefix_policy = str(condition.get("post_prefix_policy", ""))
    post_prefix_action = condition.get("post_prefix_action")
    post_action_text = None if post_prefix_action is None else str(post_prefix_action)
    signature = execution_cell_signature(
        game_id=game_id,
        prefix_policy=PREFIX_POLICY,
        prefix_action=prefix_action,
        prefix_length=int(prefix_length),
        condition_id=condition_id,
        post_prefix_policy=post_prefix_policy,
        post_prefix_action=post_action_text,
        tie_break_seed=tie_break_seed,
    )
    return ObjectiveExecutionCell(
        cell_signature=signature,
        game_id=game_id,
        prefix_policy=PREFIX_POLICY,
        prefix_action=prefix_action,
        prefix_length=int(prefix_length),
        condition_id=condition_id,
        condition_family=condition_family,
        post_prefix_policy=post_prefix_policy,
        post_prefix_action=post_action_text,
        tie_break_seed=int(tie_break_seed),
    )


def unique_execution_cells(
    planned_cells: Sequence[ObjectiveExecutionCell],
) -> Tuple[ObjectiveExecutionCell, ...]:
    by_signature: dict[str, ObjectiveExecutionCell] = {}
    for cell in planned_cells:
        by_signature.setdefault(cell.cell_signature, cell)
    return tuple(by_signature[key] for key in by_signature)


def execute_objective_cell(
    cell: ObjectiveExecutionCell,
    *,
    memory: Any,
    environments_dir: str | Path,
) -> Dict[str, Any]:
    env = _make_env(cell.game_id, environments_dir)
    current_frame = _reset_env(env)
    steps: list[ProbeStep] = []
    seen_states: set[str] = set()
    action_history: list[str] = []
    used_action6_args: list[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    initial = snapshot_frame(current_frame)
    seen_states.add(
        state_signature(initial.grid, initial.levels_completed, initial.game_state)
    )

    for index in range(max(0, int(cell.prefix_length))):
        before = snapshot_frame(current_frame)
        if is_terminal_game_state(before.game_state):
            return blocked_cell_result(
                cell,
                status=EARLY_TERMINAL_DURING_PREFIX,
                reason="prefix_reached_terminal_state_before_requested_length",
                steps=steps,
                final_frame=before,
                prefix_steps_executed=len(steps),
            )
        decision = select_probe_decision(
            condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            memory=memory,
            before_grid=before.grid,
            valid_actions=list(_valid_actions(env)),
            action_history=tuple(action_history),
            used_action6_args=tuple(used_action6_args),
            action_counts=action_counts,
            previous_steps=tuple(steps),
            tie_break_seed=cell.tie_break_seed,
        )
        if str(decision.action_name) != cell.prefix_action:
            return blocked_cell_result(
                cell,
                status=NO_EFFECTIVE_PREFIX_ACTION_AVAILABLE,
                reason="prefix_policy_did_not_select_requested_action",
                steps=steps,
                final_frame=before,
                prefix_steps_executed=len(steps),
                decision=decision,
            )
        current_frame, step = execute_decision_step(
            env,
            current_frame,
            decision=decision,
            memory=memory,
            condition_label="objective_prefix",
            step_index=index,
            seen_states=seen_states,
        )
        steps.append(step)
        update_rollout_memory(
            step,
            action_history=action_history,
            used_action6_args=used_action6_args,
            action_counts=action_counts,
        )
        seen_states.add(step.state_signature_after)

    prefix_frame = snapshot_frame(current_frame)
    if is_terminal_game_state(prefix_frame.game_state):
        return blocked_cell_result(
            cell,
            status=EARLY_TERMINAL_DURING_PREFIX,
            reason="prefix_endpoint_terminal_before_post_condition",
            steps=steps,
            final_frame=prefix_frame,
            prefix_steps_executed=len(steps),
        )

    if cell.condition_id == "stop_policy":
        return executed_cell_result(
            cell,
            status=OBJECTIVE_STOP_OBSERVED,
            steps=steps,
            final_frame=prefix_frame,
            prefix_steps_executed=len(steps),
            post_prefix_action_executed=False,
            stop_or_hold_semantics=STOP_OR_HOLD_SEMANTICS,
        )

    post_decision = post_prefix_decision(
        cell,
        memory=memory,
        before_grid=prefix_frame.grid,
        valid_actions=list(_valid_actions(env)),
        action_history=tuple(action_history),
        used_action6_args=tuple(used_action6_args),
        previous_steps=tuple(steps),
        action_counts=action_counts,
    )
    if cell.condition_id == "continue_action6" and (
        str(post_decision.action_name) != cell.prefix_action
    ):
        return blocked_cell_result(
            cell,
            status=BLOCKED_CONTROL_UNAVAILABLE,
            reason="CONTINUE_ACTION6_UNAVAILABLE_AFTER_PREFIX",
            steps=steps,
            final_frame=prefix_frame,
            prefix_steps_executed=len(steps),
            decision=post_decision,
        )
    selected = concrete_action_for_decision(list(_valid_actions(env)), post_decision)
    if selected is None:
        return blocked_cell_result(
            cell,
            status=BLOCKED_CONTROL_UNAVAILABLE,
            reason=blocked_control_reason(cell),
            steps=steps,
            final_frame=prefix_frame,
            prefix_steps_executed=len(steps),
            decision=post_decision,
        )
    current_frame, step = execute_decision_step(
        env,
        current_frame,
        decision=post_decision,
        memory=memory,
        condition_label=cell.condition_id,
        step_index=len(steps),
        seen_states=seen_states,
    )
    steps.append(step)
    final_frame = snapshot_frame(current_frame)
    return executed_cell_result(
        cell,
        status=OBJECTIVE_EXECUTED,
        steps=steps,
        final_frame=final_frame,
        prefix_steps_executed=max(0, len(steps) - 1),
        post_prefix_action_executed=True,
    )


def post_prefix_decision(
    cell: ObjectiveExecutionCell,
    *,
    memory: Any,
    before_grid: Any,
    valid_actions: Sequence[Any],
    action_history: Sequence[str],
    used_action6_args: Sequence[Mapping[str, Any]],
    previous_steps: Sequence[ProbeStep],
    action_counts: Mapping[str, int],
) -> ProbeDecision:
    if cell.condition_id == "continue_action6":
        return select_probe_decision(
            condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            memory=memory,
            before_grid=before_grid,
            valid_actions=valid_actions,
            action_history=action_history,
            used_action6_args=used_action6_args,
            action_counts=action_counts,
            previous_steps=previous_steps,
            tie_break_seed=cell.tie_break_seed,
        )
    return ProbeDecision(
        condition=cell.condition_id,
        action_name=str(cell.post_prefix_action or ""),
        decision_reason="objective_stop_switch_post_prefix_condition",
        candidate_policy_used=False,
    )


def execute_decision_step(
    env: Any,
    current_frame: Any,
    *,
    decision: ProbeDecision,
    memory: Any,
    condition_label: str,
    step_index: int,
    seen_states: set[str],
) -> Tuple[Any, ProbeStep]:
    before = snapshot_frame(current_frame)
    before_signature = state_signature(
        before.grid,
        before.levels_completed,
        before.game_state,
    )
    selected = concrete_action_for_decision(list(_valid_actions(env)), decision)
    if selected is None:
        return current_frame, error_step(
            step_index,
            condition=condition_label,
            decision=decision,
            before=before,
            before_signature=before_signature,
            error="selected_action_not_available",
        )

    from theory.m1.polymorphic_a25_adapter import _step_env_action

    after_frame = _step_env_action(env, selected)
    if after_frame is None:
        return current_frame, error_step(
            step_index,
            condition=condition_label,
            decision=decision,
            before=before,
            before_signature=before_signature,
            error="env_step_returned_no_frame",
        )
    after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
    action_name = str(getattr(selected, "name", decision.action_name))
    action_args = dict(getattr(selected, "action_args", {}) or {})
    measurements = measure_probe_metrics(before.grid, after.grid, action_args)
    after_signature = state_signature(after.grid, after.levels_completed, after.game_state)
    cycle = after_signature in seen_states
    action_class = classify_action6_args(action_args, memory) if action_name == "ACTION6" else ""
    from theory.m1.controlled_followup_experiment import metric_signal

    local_signal = metric_signal(
        measurements["local_patch_before_after"],
        "local_patch_before_after",
    )
    object_signal = metric_signal(
        measurements["object_positions_before_after"],
        "object_positions_before_after",
    )
    changed_pixels = float(measurements["changed_pixels"].get("changed_pixels", 0) or 0)
    contact_signal = metric_signal(
        measurements["contact_graph_before_after"],
        "contact_graph_before_after",
    )
    useful_action6 = bool(
        action_name == "ACTION6" and (local_signal > 0 or object_signal > 0)
    )
    useful_repositioning = bool(
        action_name == getattr(memory, "repositioning_action", "ACTION4")
        and (changed_pixels > 0 or object_signal > 0)
    )
    useful_new_state = bool(
        (useful_action6 or useful_repositioning)
        and not cycle
        and after.levels_completed >= before.levels_completed
    )
    step = ProbeStep(
        step=step_index,
        condition=condition_label,
        policy_selected_action=action_name,
        action_args=action_args,
        decision_reason=decision.decision_reason,
        candidate_policy_used=decision.candidate_policy_used,
        candidate_score=decision.candidate_score,
        candidate_score_details=decision.candidate_score_details,
        action6_arg_class=action_class,
        failure_like_action6_arg=bool(
            action_name == "ACTION6" and action_class in {"known_failure", "outside_boundary"}
        ),
        success_like_action6_arg=bool(
            action_name == "ACTION6"
            and action_class in {"known_success", "alternate_context_success"}
        ),
        repositioning_action=action_name == getattr(memory, "repositioning_action", "ACTION4"),
        local_patch_signal=local_signal,
        object_positions_signal=object_signal,
        changed_pixels=changed_pixels,
        contact_graph_signal=contact_signal,
        useful_action6=useful_action6,
        useful_repositioning=useful_repositioning,
        useful_new_state=useful_new_state,
        dead_end_or_cycle=cycle,
        state_signature_before=before_signature,
        state_signature_after=after_signature,
        levels_before=before.levels_completed,
        levels_after=after.levels_completed,
        game_state_before=before.game_state,
        game_state_after=after.game_state,
        measurements=measurements,
    )
    return after_frame, step


def update_rollout_memory(
    step: ProbeStep,
    *,
    action_history: list[str],
    used_action6_args: list[Dict[str, Any]],
    action_counts: Counter[str],
) -> None:
    action_counts[step.policy_selected_action] += 1
    action_history.append(step.policy_selected_action)
    if step.policy_selected_action == "ACTION6" and step.action_args:
        used_action6_args.append(dict(step.action_args))


def executed_cell_result(
    cell: ObjectiveExecutionCell,
    *,
    status: str,
    steps: Sequence[ProbeStep],
    final_frame: Any,
    prefix_steps_executed: int,
    post_prefix_action_executed: bool,
    stop_or_hold_semantics: str | None = None,
) -> Dict[str, Any]:
    summary = summarize_probe_steps(cell.condition_id, steps)
    objective = objective_metrics_from_summary(summary, final_frame=final_frame)
    return {
        **cell.to_dict(),
        "status": status,
        "execution_performed": True,
        "prefix_steps_executed": int(prefix_steps_executed),
        "post_prefix_action_executed": bool(post_prefix_action_executed),
        "stop_or_hold_semantics": stop_or_hold_semantics,
        "early_terminal_during_prefix": False,
        "blocked_reason": None,
        "objective_metrics": objective,
        "local_diagnostic_metrics": local_diagnostic_metrics_from_summary(summary),
        "selected_action6_args": summary.get("selected_action6_args", []),
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "cell_result_counted_as_confirmation": False,
    }


def blocked_cell_result(
    cell: ObjectiveExecutionCell,
    *,
    status: str,
    reason: str,
    steps: Sequence[ProbeStep],
    final_frame: Any,
    prefix_steps_executed: int,
    decision: ProbeDecision | None = None,
) -> Dict[str, Any]:
    summary = summarize_probe_steps(cell.condition_id, steps)
    objective = objective_metrics_from_summary(summary, final_frame=final_frame)
    return {
        **cell.to_dict(),
        "status": status,
        "execution_performed": False,
        "prefix_steps_executed": int(prefix_steps_executed),
        "post_prefix_action_executed": False,
        "early_terminal_during_prefix": status == EARLY_TERMINAL_DURING_PREFIX,
        "blocked_reason": reason,
        "blocked_decision": decision.to_dict() if hasattr(decision, "to_dict") else (
            {
                "action_name": getattr(decision, "action_name", ""),
                "action_args": dict(getattr(decision, "action_args", {}) or {}),
                "decision_reason": getattr(decision, "decision_reason", ""),
            }
            if decision is not None
            else {}
        ),
        "objective_metrics": objective,
        "local_diagnostic_metrics": local_diagnostic_metrics_from_summary(summary),
        "selected_action6_args": summary.get("selected_action6_args", []),
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "blocked_cell_counted_as_contradiction": False,
        "cell_result_counted_as_confirmation": False,
    }


def objective_metrics_from_summary(
    summary: Mapping[str, Any],
    *,
    final_frame: Any,
) -> Dict[str, Any]:
    final_game_state = str(
        summary.get("final_game_state") or getattr(final_frame, "game_state", "")
    )
    levels = int(
        summary.get("final_levels_completed")
        if summary.get("final_levels_completed") is not None
        else getattr(final_frame, "levels_completed", 0)
    )
    return {
        "final_game_state": final_game_state,
        "terminal_state_after_rollout": is_terminal_game_state(final_game_state),
        "levels_completed_after_rollout": levels,
        "objective_progress_proxy": float(summary.get("progress_proxy", 0.0) or 0.0),
    }


def local_diagnostic_metrics_from_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "local_effect_metric": int(summary.get("useful_new_states", 0) or 0),
        "repeated_action6_count": int(
            summary.get("repeated_action6_args_selected", 0) or 0
        ),
        "useful_action6_steps": int(summary.get("useful_action6_steps", 0) or 0),
    }


def build_hypothesis_observation_link(
    link: Mapping[str, Any],
    cell_result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(link),
        "cell_status": str(cell_result.get("status", "")),
        "execution_performed": bool(cell_result.get("execution_performed", False)),
        "objective_metrics": dict(cell_result.get("objective_metrics", {}) or {}),
        "local_diagnostic_metrics": dict(
            cell_result.get("local_diagnostic_metrics", {}) or {}
        ),
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": int(cell_result.get("neutral_events", 0) or 0),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "cell_link_counted_as_independent_execution": False,
        "cell_link_counted_as_confirmation": False,
    }


def build_results_payload(
    *,
    objective_requests_path: str | Path,
    scope_consolidation_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    planned_cells: Sequence[ObjectiveExecutionCell],
    unique_cells: Sequence[ObjectiveExecutionCell],
    cell_results: Sequence[Mapping[str, Any]],
    hypothesis_observation_links: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = summarize_objective_execution(
        requests=requests,
        planned_cells=planned_cells,
        unique_cells=unique_cells,
        cell_results=cell_results,
        links=hypothesis_observation_links,
    )
    return {
        "config": {
            "schema_version": "m3.objective_stop_switch_results.v1",
            "objective_requests_path": str(objective_requests_path),
            "scope_consolidation_path": str(scope_consolidation_path),
            "environments_dir": str(environments_dir),
            "inputs_read": ["M3.O1", "M3.24"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "prefix_policy": PREFIX_POLICY,
            "duplicate_execution_cells_counted_as_independent": False,
            "stop_or_hold_semantics": STOP_OR_HOLD_SEMANTICS,
        },
        "summary": summary,
        "execution_cells": [dict(row) for row in cell_results],
        "hypothesis_observation_links": [dict(row) for row in hypothesis_observation_links],
        "objective_outcome_table": objective_outcome_table(cell_results),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "duplicate_execution_cells_counted_as_independent": False,
        "objective_result_counted_as_confirmation": False,
        "a32_remains_only_verdict_location": True,
    }


def summarize_objective_execution(
    *,
    requests: Sequence[Mapping[str, Any]],
    planned_cells: Sequence[ObjectiveExecutionCell],
    unique_cells: Sequence[ObjectiveExecutionCell],
    cell_results: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    executed = [row for row in cell_results if bool(row.get("execution_performed"))]
    blocked = [row for row in cell_results if not bool(row.get("execution_performed"))]
    early = [
        row for row in cell_results if bool(row.get("early_terminal_during_prefix"))
    ]
    return {
        "objective_requests_consumed": len(requests),
        "planned_condition_cells": len(planned_cells),
        "unique_execution_cells": len(unique_cells),
        "duplicate_execution_cells": max(0, len(planned_cells) - len(unique_cells)),
        "duplicate_execution_cells_counted_as_independent": False,
        "hypothesis_observation_links": len(links),
        "execution_performed": True,
        "objective_cells_executed": len(executed),
        "blocked_cells": len(blocked),
        "early_terminal_prefix_cells": len(early),
        "blocked_control_unavailable_cells": len(
            [row for row in blocked if row.get("status") == BLOCKED_CONTROL_UNAVAILABLE]
        ),
        "stop_or_hold_unavailable_cells": 0,
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": sum(int(row.get("neutral_events", 0) or 0) for row in cell_results),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "blocked_cells_counted_as_contradictions": False,
        "policy_result_counted_as_scientific_verdict": False,
        "a32_remains_only_verdict_location": True,
    }


def objective_outcome_table(
    cell_results: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows: dict[int, Dict[str, Any]] = {}
    for result in cell_results:
        prefix_length = int(result.get("prefix_length", 0) or 0)
        condition_id = str(result.get("condition_id", ""))
        objective = dict(result.get("objective_metrics", {}) or {})
        row = rows.setdefault(
            prefix_length,
            {
                "prefix_length": prefix_length,
                "conditions": {},
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            },
        )
        row["conditions"][condition_id] = {
            "status": str(result.get("status", "")),
            "final_game_state": objective.get("final_game_state", ""),
            "terminal_state_after_rollout": bool(
                objective.get("terminal_state_after_rollout", False)
            ),
            "levels_completed_after_rollout": int(
                objective.get("levels_completed_after_rollout", 0) or 0
            ),
            "objective_progress_proxy": float(
                objective.get("objective_progress_proxy", 0.0) or 0.0
            ),
        }
    return [rows[key] for key in sorted(rows)]


def ready_objective_requests(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for request in payload.get("objective_stop_switch_experiment_requests", []) or []:
        if not isinstance(request, Mapping):
            continue
        validate_ready_objective_request(request)
        if str(request.get("status", "")) == READY_FOR_M3_OBJECTIVE_EXPERIMENT:
            rows.append(dict(request))
    return tuple(rows)


def validate_ready_objective_request(request: Mapping[str, Any]) -> None:
    if int(request.get("support", 0) or 0) != 0:
        raise ValueError("objective request support must remain 0")
    if bool(request.get("execution_performed", False)):
        raise ValueError("M3.O1 request must not already be executed")
    if str(request.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("objective request must remain candidate-only")
    if str(request.get("truth_status", "")) != M3_REFINEMENT_TRUTH_STATUS:
        raise ValueError("objective request truth_status must remain M3-local")
    if bool(request.get("objective_request_counted_as_support", False)):
        raise ValueError("objective request cannot count as support")
    if bool(request.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")


def _validate_source_request_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source request summary support must remain 0")
    if bool(summary.get("execution_performed", False)):
        raise ValueError("M3.O1 source must be planning-only")
    if bool(summary.get("objective_request_counted_as_support", False)):
        raise ValueError("M3.O1 request cannot count as support")
    if bool(summary.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")


def execution_cell_signature(
    *,
    game_id: str,
    prefix_policy: str,
    prefix_action: str,
    prefix_length: int,
    condition_id: str,
    post_prefix_policy: str,
    post_prefix_action: str | None,
    tie_break_seed: int,
) -> str:
    raw = {
        "condition_id": condition_id,
        "game_id": game_id,
        "post_prefix_action": post_prefix_action,
        "post_prefix_policy": post_prefix_policy,
        "prefix_action": prefix_action,
        "prefix_length": int(prefix_length),
        "prefix_policy": prefix_policy,
        "tie_break_seed": int(tie_break_seed),
    }
    return "m3_o2::" + json.dumps(raw, sort_keys=True, separators=(",", ":"))


def blocked_control_reason(cell: ObjectiveExecutionCell) -> str:
    if cell.condition_id == "continue_action6":
        return "CONTINUE_ACTION6_UNAVAILABLE_AFTER_PREFIX"
    if str(cell.post_prefix_policy) == "switch_to_action":
        return "BLOCKED_CONTROL_UNAVAILABLE"
    return "POST_PREFIX_ACTION_UNAVAILABLE"


def is_terminal_game_state(game_state: Any) -> bool:
    text = str(game_state or "")
    return bool(text and text not in {"NOT_FINISHED", "RUNNING"})


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_objective_stop_switch_experiment_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OBJECTIVE_STOP_SWITCH_RESULTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3.O2 objective stop/switch experiment cells.",
    )
    parser.add_argument(
        "--objective-requests",
        type=Path,
        default=DEFAULT_OBJECTIVE_STOP_SWITCH_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--scope-consolidation",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--tie-break-seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_STOP_SWITCH_RESULTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_stop_switch_experiment_execution(
        objective_requests_path=args.objective_requests,
        scope_consolidation_path=args.scope_consolidation,
        environments_dir=args.environments_dir,
        tie_break_seed=args.tie_break_seed,
        max_cells=args.max_cells,
    )
    write_objective_stop_switch_experiment_results(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
