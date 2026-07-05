"""A39 precondition-aware policy rollout.

A39 re-injects A38 applicability preconditions into the A37-style rollout. It
uses confirmed mechanics only when both scope and dynamic preconditions pass.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from theory.a33.confirmed_mechanics_registry import (
    DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
)
from theory.a35.confirmed_mechanic_scope_map import (
    DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    context_id,
)
from theory.a36.scope_conditioned_policy_probe import DEFAULT_BASELINE_ORDER, metric_signal
from theory.a37.scope_conditioned_policy_rollout import (
    ScopeConditionedRolloutDecision,
    context_signature_for_scope,
    fallback_action_for_rollout,
    fallback_context_signature,
    score_or_level_unchanged_or_improved,
    select_available_action,
    select_available_fallback,
    state_signature,
)
from theory.a38.rollout_aware_scope_refinement import (
    DEFAULT_A38_SCOPE_REFINEMENT_OUTPUT_PATH,
    REFINED_WITH_PRECONDITIONS,
)
from theory.m1.controlled_followup_experiment import (
    _step_env_action,
    measure_required_observation,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_A39_POLICY_ROLLOUT_OUTPUT_PATH = (
    Path("diagnostics") / "a39" / "precondition_aware_policy_rollout.json"
)
DEFAULT_BUDGET = 3
TRUTH_STATUS = "NOT_REEVALUATED_BY_A39"


@dataclass(frozen=True)
class UsagePreconditionCheck:
    """Dynamic applicability check before executing a confirmed mechanic."""

    satisfied: bool
    checked_preconditions: Tuple[str, ...] = field(default_factory=tuple)
    failed_precondition: str = ""
    reason: str = ""
    local_patch_available: bool = False
    target_patch_already_saturated: bool = False
    target_patch_not_already_saturated: bool = False
    patch_bbox: Tuple[int, ...] = field(default_factory=tuple)
    local_patch_current: Any = None
    saturated_reference_patch: Any = None
    blocked_context_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "checked_preconditions": list(self.checked_preconditions),
            "failed_precondition": self.failed_precondition,
            "reason": self.reason,
            "local_patch_available": self.local_patch_available,
            "target_patch_already_saturated": self.target_patch_already_saturated,
            "target_patch_not_already_saturated": self.target_patch_not_already_saturated,
            "patch_bbox": list(self.patch_bbox),
            "local_patch_current": self.local_patch_current,
            "saturated_reference_patch": self.saturated_reference_patch,
            "blocked_context_id": self.blocked_context_id,
        }


@dataclass(frozen=True)
class PreconditionAwarePolicyRolloutStep:
    """One executed A39 rollout step."""

    step: int
    key: str
    game_id: str
    context_signature: Tuple[str, ...]
    context_id: str
    policy_selected_action: str
    fallback_action: str
    predicted_metric: str
    selected_from_confirmed_mechanic: bool
    blocked_confirmed_mechanic: bool
    blocked_action: str
    scope_used: str
    refined_scope_used: str
    context_match: bool
    context_match_reason: str
    decision_reason: str
    precondition_status: str
    failed_precondition: str = ""
    blocked_context_id: str = ""
    avoided_saturated_reuse: bool = False
    fallback_due_to_failed_precondition: bool = False
    action_args: Dict[str, Any] = field(default_factory=dict)
    precondition_check: Dict[str, Any] = field(default_factory=dict)
    measurement: Dict[str, Any] = field(default_factory=dict)
    selected_signal: float = 0.0
    functional_progress: bool = False
    useful_new_state: bool = False
    usage_contradiction: bool = False
    dead_end_or_cycle: bool = False
    state_signature_before: str = ""
    state_signature_after: str = ""
    levels_before: int = 0
    levels_after: int = 0
    game_state_before: str = ""
    game_state_after: str = ""
    env_actions: int = 0
    error: str = ""
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "key": self.key,
            "game_id": self.game_id,
            "context_signature": list(self.context_signature),
            "context_id": self.context_id,
            "policy_selected_action": self.policy_selected_action,
            "fallback_action": self.fallback_action,
            "predicted_metric": self.predicted_metric,
            "selected_from_confirmed_mechanic": self.selected_from_confirmed_mechanic,
            "blocked_confirmed_mechanic": self.blocked_confirmed_mechanic,
            "blocked_action": self.blocked_action,
            "scope_used": self.scope_used,
            "refined_scope_used": self.refined_scope_used,
            "context_match": self.context_match,
            "context_match_reason": self.context_match_reason,
            "decision_reason": self.decision_reason,
            "precondition_status": self.precondition_status,
            "failed_precondition": self.failed_precondition,
            "blocked_context_id": self.blocked_context_id,
            "avoided_saturated_reuse": self.avoided_saturated_reuse,
            "fallback_due_to_failed_precondition": (
                self.fallback_due_to_failed_precondition
            ),
            "action_args": dict(self.action_args),
            "precondition_check": dict(self.precondition_check),
            "measurement": dict(self.measurement),
            "selected_signal": self.selected_signal,
            "functional_progress": self.functional_progress,
            "useful_new_state": self.useful_new_state,
            "usage_contradiction": self.usage_contradiction,
            "dead_end_or_cycle": self.dead_end_or_cycle,
            "state_signature_before": self.state_signature_before,
            "state_signature_after": self.state_signature_after,
            "levels_before": int(self.levels_before),
            "levels_after": int(self.levels_after),
            "game_state_before": self.game_state_before,
            "game_state_after": self.game_state_after,
            "env_actions": int(self.env_actions),
            "error": self.error,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_precondition_aware_policy_rollout(
    *,
    registry_path: str | Path = DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    scope_map_path: str | Path = DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    refinement_path: str | Path = DEFAULT_A38_SCOPE_REFINEMENT_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    budget: int = DEFAULT_BUDGET,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Dict[str, Any]:
    """Run a rollout that blocks confirmed mechanics when A38 preconditions fail."""
    registry_payload = _load_json(registry_path)
    scope_payload = _load_json(scope_map_path)
    refinement_payload = _load_json(refinement_path)
    registry_entries = [
        dict(entry)
        for entry in registry_payload.get("confirmed_mechanics", []) or []
        if isinstance(entry, Mapping)
    ]
    scopes_by_key = _scope_maps_by_key(scope_payload)
    refinements_by_key = _refinements_by_key(refinement_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    steps = execute_precondition_aware_rollout(
        registry_entries,
        scopes_by_key=scopes_by_key,
        refinements_by_key=refinements_by_key,
        environments_dir=env_dir,
        budget=budget,
        baseline_order=baseline_order,
    )
    return {
        "config": {
            "registry_path": str(registry_path),
            "scope_map_path": str(scope_map_path),
            "refinement_path": str(refinement_path),
            "environments_dir": str(env_dir),
            "budget": int(budget),
            "baseline_order": list(baseline_order),
            "inputs_read": ["A33", "A35", "A38"],
            "artifacts_not_read": ["M3", "A32", "A34", "A36", "A37"],
            "artifacts_not_modified": ["A33", "A35", "A38"],
        },
        "summary": summarize_precondition_aware_steps(steps),
        "rollout_steps": [step.to_dict() for step in steps],
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def execute_precondition_aware_rollout(
    registry_entries: Sequence[Mapping[str, Any]],
    *,
    scopes_by_key: Mapping[str, Mapping[str, Any]],
    refinements_by_key: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path,
    budget: int = DEFAULT_BUDGET,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Tuple[PreconditionAwarePolicyRolloutStep, ...]:
    """Execute the A39 policy in one live environment."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    if not registry_entries:
        return tuple()
    first_entry = dict(registry_entries[0])
    game_id = str(first_entry.get("game_id", ""))
    env_dir = Path(environments_dir)
    _configure_offline_env(env_dir)
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(env_dir),
    )
    env = arc.make(game_id)
    current_frame = env.step(GameAction.RESET)

    action_history: list[str] = []
    fallback_counts: Counter[str] = Counter()
    seen_states: set[str] = set()
    steps: list[PreconditionAwarePolicyRolloutStep] = []
    initial = snapshot_frame(current_frame)
    seen_states.add(state_signature(initial.grid, initial.levels_completed, initial.game_state))

    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(current_frame)
        before_signature = state_signature(
            before.grid,
            before.levels_completed,
            before.game_state,
        )
        decision = select_precondition_aware_decision(
            registry_entries,
            scopes_by_key=scopes_by_key,
            action_history=tuple(action_history),
            fallback_counts=fallback_counts,
            baseline_order=baseline_order,
        )
        entry = entry_by_key(registry_entries, decision.key) or first_entry
        refinement = refinements_by_key.get(decision.key, {})
        selected_action = None
        selected_from_mechanic = decision.selected_from_confirmed_mechanic
        blocked_action = ""
        blocked_confirmed = False
        precondition_status = (
            "NOT_APPLICABLE" if not decision.selected_from_confirmed_mechanic else "NOT_CHECKED"
        )
        failed_precondition = ""
        blocked_context = ""
        avoided_saturated_reuse = False
        fallback_due_to_failed_precondition = False
        precondition_payload: Dict[str, Any] = {}
        decision_reason = decision.decision_reason

        if decision.selected_from_confirmed_mechanic:
            candidate_action = select_available_action(
                env,
                desired_action=decision.action,
                predicted_metric=decision.predicted_metric,
                require_metric_args=True,
            )
            if candidate_action is not None:
                check = check_usage_preconditions(
                    before.grid,
                    candidate_action,
                    decision=decision,
                    refinement=refinement,
                    action_history=tuple(action_history),
                )
                precondition_payload = check.to_dict()
                precondition_status = "SATISFIED" if check.satisfied else "FAILED"
                failed_precondition = check.failed_precondition
                blocked_context = check.blocked_context_id
                if check.satisfied:
                    selected_action = candidate_action
                else:
                    blocked_action = decision.action
                    blocked_confirmed = True
                    selected_from_mechanic = False
                    fallback_due_to_failed_precondition = True
                    avoided_saturated_reuse = check.target_patch_already_saturated
                    decision_reason = "failed_precondition_fallback_executed"
            else:
                selected_from_mechanic = False
                decision_reason = "confirmed_action_unavailable_fallback_executed"
        else:
            selected_action = select_available_action(
                env,
                desired_action=decision.action,
                predicted_metric=decision.predicted_metric,
                require_metric_args=False,
            )

        if selected_action is None:
            fallback = select_available_fallback(
                env,
                treatment_action=str(entry.get("action", decision.action)),
                fallback_counts=fallback_counts,
                baseline_order=baseline_order,
            )
            if fallback is None:
                steps.append(
                    error_step(
                        step_index,
                        first_entry,
                        decision,
                        game_id=game_id,
                        precondition_status=precondition_status,
                        failed_precondition=failed_precondition,
                        blocked_context_id=blocked_context,
                        error="no_available_policy_or_fallback_action",
                    )
                )
                break
            selected_action = fallback

        after_frame = _step_env_action(env, selected_action)
        if after_frame is None:
            steps.append(
                error_step(
                    step_index,
                    first_entry,
                    decision,
                    game_id=game_id,
                    precondition_status=precondition_status,
                    failed_precondition=failed_precondition,
                    blocked_context_id=blocked_context,
                    error="env_step_returned_no_frame",
                )
            )
            break

        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        action_name = str(getattr(selected_action, "name", decision.action))
        action_args = dict(getattr(selected_action, "action_args", {}) or {})
        measurement = measure_required_observation(
            before.grid,
            after.grid,
            required_observation=decision.predicted_metric,
            action_args=action_args,
        )
        selected_signal = metric_signal(measurement, decision.predicted_metric)
        no_regression = score_or_level_unchanged_or_improved(
            before_levels=before.levels_completed,
            after_levels=after.levels_completed,
            game_state_after=after.game_state,
        )
        after_signature = state_signature(
            after.grid,
            after.levels_completed,
            after.game_state,
        )
        cycle = after_signature in seen_states
        functional = bool(
            selected_from_mechanic
            and measurement.get("changed", False)
            and selected_signal > 0
            and no_regression
        )
        useful_new_state = bool(functional and not cycle)
        usage_contradiction = bool(selected_from_mechanic and selected_signal <= 0)
        step = PreconditionAwarePolicyRolloutStep(
            step=step_index,
            key=decision.key,
            game_id=game_id,
            context_signature=decision.context_signature,
            context_id=context_id(decision.context_signature),
            policy_selected_action=action_name,
            fallback_action=decision.fallback_action,
            predicted_metric=decision.predicted_metric,
            selected_from_confirmed_mechanic=selected_from_mechanic,
            blocked_confirmed_mechanic=blocked_confirmed,
            blocked_action=blocked_action,
            scope_used=decision.scope_used,
            refined_scope_used=str(refinement.get("refined_scope_assessment", "")),
            context_match=decision.context_match,
            context_match_reason=decision.context_match_reason,
            decision_reason=decision_reason,
            precondition_status=precondition_status,
            failed_precondition=failed_precondition,
            blocked_context_id=blocked_context,
            avoided_saturated_reuse=avoided_saturated_reuse,
            fallback_due_to_failed_precondition=fallback_due_to_failed_precondition,
            action_args=action_args,
            precondition_check=precondition_payload,
            measurement=measurement,
            selected_signal=selected_signal,
            functional_progress=functional,
            useful_new_state=useful_new_state,
            usage_contradiction=usage_contradiction,
            dead_end_or_cycle=cycle,
            state_signature_before=before_signature,
            state_signature_after=after_signature,
            levels_before=before.levels_completed,
            levels_after=after.levels_completed,
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            env_actions=1,
        )
        steps.append(step)
        seen_states.add(after_signature)
        if not selected_from_mechanic:
            fallback_counts[action_name] += 1
        action_history.append(action_name)
        current_frame = after_frame
    return tuple(steps)


def select_precondition_aware_decision(
    registry_entries: Sequence[Mapping[str, Any]],
    *,
    scopes_by_key: Mapping[str, Mapping[str, Any]],
    action_history: Sequence[str],
    fallback_counts: Mapping[str, int],
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> ScopeConditionedRolloutDecision:
    """Select by A35 scope; dynamic A38 checks happen before execution."""
    for entry in registry_entries:
        key = str(entry.get("key", ""))
        scope_map = scopes_by_key.get(key, {})
        if str(scope_map.get("scope_assessment", "")) != "CONTEXTUALLY_STABLE":
            continue
        signature, matched, reason = context_signature_for_scope(
            action_history,
            scope_map,
        )
        if matched:
            return ScopeConditionedRolloutDecision(
                key=key,
                action=str(entry.get("action", "")),
                fallback_action=fallback_action_for_rollout(
                    treatment_action=str(entry.get("action", "")),
                    fallback_counts=fallback_counts,
                    baseline_order=baseline_order,
                ),
                predicted_metric=str(entry.get("predicted_metric", "")),
                selected_from_confirmed_mechanic=True,
                scope_used=str(scope_map.get("scope_assessment", "")),
                context_signature=signature,
                context_match=True,
                context_match_reason=reason,
                decision_reason="covered_scope_precondition_check_required",
            )

    entry = dict(registry_entries[0]) if registry_entries else {}
    action = fallback_action_for_rollout(
        treatment_action=str(entry.get("action", "")),
        fallback_counts=fallback_counts,
        baseline_order=baseline_order,
    )
    signature = fallback_context_signature(action_history, scopes_by_key.values())
    return ScopeConditionedRolloutDecision(
        key=str(entry.get("key", "")),
        action=action,
        fallback_action=action,
        predicted_metric=str(entry.get("predicted_metric", "")),
        selected_from_confirmed_mechanic=False,
        scope_used=str(scopes_by_key.get(str(entry.get("key", "")), {}).get("scope_assessment", "")),
        context_signature=signature,
        context_match=False,
        context_match_reason="no_covered_rollout_context",
        decision_reason="fallback_neutral_exploration",
    )


def check_usage_preconditions(
    before_grid: Any,
    selected_action: Any,
    *,
    decision: ScopeConditionedRolloutDecision,
    refinement: Mapping[str, Any],
    action_history: Sequence[str],
) -> UsagePreconditionCheck:
    """Check A38 preconditions against the current live frame."""
    preconditions = tuple(str(value) for value in refinement.get("usage_preconditions", []) or [])
    if not preconditions:
        return UsagePreconditionCheck(
            satisfied=True,
            checked_preconditions=tuple(),
            reason="no_dynamic_preconditions_declared",
        )
    patch = current_local_patch(before_grid, selected_action)
    local_available = bool(patch.get("local_patch_available", False))
    saturated_patch = saturated_reference_patch(
        patch.get("local_patch_current"),
        refinement,
    )
    already_saturated = saturated_patch is not None
    blocked_context = (
        blocked_context_id_for_decision(
            refinement,
            context_id=context_id(decision.context_signature),
            action_history=action_history,
            action=decision.action,
        )
        if already_saturated
        else ""
    )
    if "local_patch_available=true" in preconditions and not local_available:
        return UsagePreconditionCheck(
            satisfied=False,
            checked_preconditions=preconditions,
            failed_precondition="local_patch_available=true",
            reason="local_patch_not_available",
            local_patch_available=False,
            patch_bbox=tuple(patch.get("patch_bbox", ()) or ()),
            local_patch_current=patch.get("local_patch_current"),
            blocked_context_id=blocked_context,
        )
    if "target_patch_not_already_saturated=true" in preconditions and already_saturated:
        return UsagePreconditionCheck(
            satisfied=False,
            checked_preconditions=preconditions,
            failed_precondition="target_patch_not_already_saturated=true",
            reason="target_patch_already_saturated",
            local_patch_available=local_available,
            target_patch_already_saturated=True,
            target_patch_not_already_saturated=False,
            patch_bbox=tuple(patch.get("patch_bbox", ()) or ()),
            local_patch_current=patch.get("local_patch_current"),
            saturated_reference_patch=saturated_patch,
            blocked_context_id=blocked_context,
        )
    return UsagePreconditionCheck(
        satisfied=True,
        checked_preconditions=preconditions,
        reason="dynamic_preconditions_satisfied",
        local_patch_available=local_available,
        target_patch_already_saturated=False,
        target_patch_not_already_saturated=True,
        patch_bbox=tuple(patch.get("patch_bbox", ()) or ()),
        local_patch_current=patch.get("local_patch_current"),
    )


def current_local_patch(
    grid: Any,
    selected_action: Any,
    *,
    radius: int = 1,
) -> Dict[str, Any]:
    action_args = dict(getattr(selected_action, "action_args", {}) or {})
    x = _safe_int(action_args.get("x"))
    y = _safe_int(action_args.get("y"))
    array = np.asarray(grid, dtype=np.int32)
    if x is None or y is None or array.ndim != 2:
        return {
            "local_patch_available": False,
            "local_patch_current": None,
            "patch_bbox": (),
        }
    y0 = max(0, y - radius)
    y1 = min(array.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(array.shape[1], x + radius + 1)
    return {
        "local_patch_available": True,
        "local_patch_current": array[y0:y1, x0:x1].tolist(),
        "patch_bbox": (y0, x0, y1 - 1, x1 - 1),
    }


def saturated_reference_patch(
    current_patch: Any,
    refinement: Mapping[str, Any],
) -> Any | None:
    if current_patch is None:
        return None
    candidates = []
    for row in refinement.get("positive_usage_contexts", []) or []:
        if isinstance(row, Mapping) and row.get("local_patch_after") is not None:
            candidates.append(row.get("local_patch_after"))
    for row in refinement.get("negative_usage_contexts", []) or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("target_patch_already_saturated"):
            if row.get("local_patch_before") is not None:
                candidates.append(row.get("local_patch_before"))
            if row.get("local_patch_after") is not None:
                candidates.append(row.get("local_patch_after"))
    for candidate in candidates:
        if current_patch == candidate:
            return candidate
    return None


def blocked_context_id_for_decision(
    refinement: Mapping[str, Any],
    *,
    context_id: str,
    action_history: Sequence[str],
    action: str,
) -> str:
    details = [
        dict(row)
        for row in refinement.get("blocked_context_details", []) or []
        if isinstance(row, Mapping)
    ]
    for row in details:
        if str(row.get("context_id", "")) == str(context_id):
            return str(row.get("blocked_context_id", ""))
    previous = ""
    for past_action in reversed(tuple(str(value) for value in action_history)):
        if past_action == str(action):
            previous = past_action
            break
    if previous:
        return f"{context_id}_live_after_{previous}"
    return f"{context_id}_live_after_prior_policy"


def summarize_precondition_aware_steps(
    steps: Sequence[PreconditionAwarePolicyRolloutStep],
) -> Dict[str, Any]:
    return {
        "policy_steps": len(steps),
        "policy_steps_from_confirmed_mechanic": len(
            [step for step in steps if step.selected_from_confirmed_mechanic]
        ),
        "functional_progress_steps": len(
            [step for step in steps if step.functional_progress]
        ),
        "useful_new_states": len([step for step in steps if step.useful_new_state]),
        "usage_contradictions": len(
            [step for step in steps if step.usage_contradiction]
        ),
        "fallback_steps": len(
            [step for step in steps if not step.selected_from_confirmed_mechanic]
        ),
        "fallback_due_to_failed_precondition": len(
            [step for step in steps if step.fallback_due_to_failed_precondition]
        ),
        "avoided_saturated_reuse": len(
            [step for step in steps if step.avoided_saturated_reuse]
        ),
        "blocked_confirmed_mechanic_steps": len(
            [step for step in steps if step.blocked_confirmed_mechanic]
        ),
        "cycle_or_dead_end_detected": any(step.dead_end_or_cycle for step in steps),
        "dead_end_or_cycle_steps": len(
            [step for step in steps if step.dead_end_or_cycle]
        ),
        "errors": len([step for step in steps if step.error]),
        "live_env_actions": sum(step.env_actions for step in steps),
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def error_step(
    step_index: int,
    entry: Mapping[str, Any],
    decision: ScopeConditionedRolloutDecision,
    *,
    game_id: str,
    precondition_status: str,
    failed_precondition: str,
    blocked_context_id: str,
    error: str,
) -> PreconditionAwarePolicyRolloutStep:
    return PreconditionAwarePolicyRolloutStep(
        step=step_index,
        key=decision.key or str(entry.get("key", "")),
        game_id=game_id,
        context_signature=decision.context_signature,
        context_id=context_id(decision.context_signature),
        policy_selected_action=decision.action,
        fallback_action=decision.fallback_action,
        predicted_metric=decision.predicted_metric,
        selected_from_confirmed_mechanic=False,
        blocked_confirmed_mechanic=False,
        blocked_action="",
        scope_used=decision.scope_used,
        refined_scope_used="",
        context_match=decision.context_match,
        context_match_reason=decision.context_match_reason,
        decision_reason=decision.decision_reason,
        precondition_status=precondition_status,
        failed_precondition=failed_precondition,
        blocked_context_id=blocked_context_id,
        error=error,
    )


def entry_by_key(
    registry_entries: Sequence[Mapping[str, Any]],
    key: str,
) -> Mapping[str, Any] | None:
    for entry in registry_entries:
        if str(entry.get("key", "")) == str(key):
            return dict(entry)
    return None


def write_precondition_aware_policy_rollout(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A39_POLICY_ROLLOUT_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scope_maps_by_key(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("key", "")): dict(row)
        for row in payload.get("scope_maps", []) or []
        if isinstance(row, Mapping) and str(row.get("key", ""))
    }


def _refinements_by_key(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("key", "")): dict(row)
        for row in payload.get("scope_refinements", []) or []
        if isinstance(row, Mapping) and str(row.get("key", ""))
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A39 precondition-aware policy rollout.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--scope-map",
        type=Path,
        default=DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    )
    parser.add_argument(
        "--refinement",
        type=Path,
        default=DEFAULT_A38_SCOPE_REFINEMENT_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--out", type=Path, default=DEFAULT_A39_POLICY_ROLLOUT_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_precondition_aware_policy_rollout(
        registry_path=args.registry,
        scope_map_path=args.scope_map,
        refinement_path=args.refinement,
        environments_dir=args.environments_dir,
        budget=args.budget,
    )
    write_precondition_aware_policy_rollout(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "truth_status": TRUTH_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
