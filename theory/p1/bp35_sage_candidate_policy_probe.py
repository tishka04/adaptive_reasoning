"""P1 closed-loop bp35 probe for the M3.24 candidate-only policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from theory.m1.controlled_followup_experiment import metric_signal
from theory.m1.polymorphic_a25_adapter import measure_required_observation
from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m3.a32_requested_patch_similarity_scope_consolidation import (
    DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    SCOPE_EXPANDED_CANDIDATE_ONLY,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_probe.json"
)
DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MATRIX_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_probe_matrix.json"
)
DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_ABLATION_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_ablation_matrix.json"
)
DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_STALE_GUARD_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_stale_guard_matrix.json"
)
DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_SOFT_STALE_GUARD_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_soft_stale_guard_matrix.json"
)
DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_CONDITIONAL_REFRESH_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_conditional_refresh_matrix.json"
)
DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MOVEMENT_REFRESH_OUTPUT_PATH = (
    Path("diagnostics") / "p1" / "bp35_sage_candidate_policy_movement_refresh_matrix.json"
)
DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_BUDGET = 8
DEFAULT_BUDGETS = (4, 8, 12, 16, 24, 32)
DEFAULT_TIE_BREAK_SEEDS = (0, 1, 2)
TRUTH_STATUS = "NOT_EVALUATED_BY_P1_AGENT_PROBE"
BASELINE_POLICY = "baseline_no_m3_candidate_rule"
ACTION4_ONLY_POLICY = "action4_reposition_only"
PATCH_SIMILARITY_ONLY_POLICY = "patch_similarity_only_no_repositioning"
PATCH_SIMILARITY_STALE_GUARD_POLICY = "patch_similarity_stale_guard"
PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY = "patch_similarity_soft_stale_guard"
CONDITIONAL_ACTION4_REFRESH_POLICY = "conditional_action4_refresh"
CONDITIONAL_MOVEMENT_REFRESH_POLICY = "conditional_movement_refresh"
CANDIDATE_POLICY = "candidate_policy_from_m3_24"
MOVEMENT_REFRESH_CANDIDATES = ("ACTION3", "ACTION4", "ACTION1", "ACTION2")
P1_POLICY_CONDITIONS = (BASELINE_POLICY, ACTION4_ONLY_POLICY, CANDIDATE_POLICY)
P1_POLICY_CONDITIONS_WITH_PATCH_ONLY = (
    BASELINE_POLICY,
    ACTION4_ONLY_POLICY,
    PATCH_SIMILARITY_ONLY_POLICY,
    CANDIDATE_POLICY,
)
P1_POLICY_CONDITIONS_WITH_STALE_GUARD = (
    BASELINE_POLICY,
    ACTION4_ONLY_POLICY,
    PATCH_SIMILARITY_ONLY_POLICY,
    PATCH_SIMILARITY_STALE_GUARD_POLICY,
    CANDIDATE_POLICY,
)
P1_POLICY_CONDITIONS_WITH_SOFT_STALE_GUARD = (
    BASELINE_POLICY,
    ACTION4_ONLY_POLICY,
    PATCH_SIMILARITY_ONLY_POLICY,
    PATCH_SIMILARITY_STALE_GUARD_POLICY,
    PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
    CANDIDATE_POLICY,
)
P1_POLICY_CONDITIONS_WITH_CONDITIONAL_REFRESH = (
    BASELINE_POLICY,
    ACTION4_ONLY_POLICY,
    PATCH_SIMILARITY_ONLY_POLICY,
    PATCH_SIMILARITY_STALE_GUARD_POLICY,
    PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
    CONDITIONAL_ACTION4_REFRESH_POLICY,
    CANDIDATE_POLICY,
)
P1_POLICY_CONDITIONS_WITH_MOVEMENT_REFRESH = (
    BASELINE_POLICY,
    ACTION4_ONLY_POLICY,
    PATCH_SIMILARITY_ONLY_POLICY,
    PATCH_SIMILARITY_STALE_GUARD_POLICY,
    PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
    CONDITIONAL_ACTION4_REFRESH_POLICY,
    CONDITIONAL_MOVEMENT_REFRESH_POLICY,
    CANDIDATE_POLICY,
)
SUCCESS_METRICS = ("local_patch_before_after", "object_positions_before_after")
DIAGNOSTIC_METRICS = ("changed_pixels", "contact_graph_before_after")


@dataclass(frozen=True)
class CandidatePolicyMemory:
    """Read-only candidate policy distilled from M3.24."""

    enabled: bool
    source_scope_consolidation_id: str = ""
    source_status: str = ""
    source_revision_status: str = "CANDIDATE_ONLY"
    source_truth_status: str = "NOT_EVALUATED_BY_M3"
    game_id: str = DEFAULT_GAME_ID
    target_action: str = "ACTION6"
    repositioning_action: str = "ACTION4"
    candidate_rule_family: str = "local_patch_transformability"
    known_success_args: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    known_failed_args: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    outside_boundary_args: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    alternate_context_success_args: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    success_metrics: Tuple[str, ...] = SUCCESS_METRICS
    diagnostic_metrics: Tuple[str, ...] = DIAGNOSTIC_METRICS
    ready_for_agent_policy_probe: bool = False
    a33_ready: bool = False
    support: int = 0
    rule_counted_as_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_scope_consolidation_id": self.source_scope_consolidation_id,
            "source_status": self.source_status,
            "source_revision_status": self.source_revision_status,
            "source_truth_status": self.source_truth_status,
            "game_id": self.game_id,
            "target_action": self.target_action,
            "repositioning_action": self.repositioning_action,
            "candidate_rule_family": self.candidate_rule_family,
            "known_success_args": [dict(row) for row in self.known_success_args],
            "known_failed_args": [dict(row) for row in self.known_failed_args],
            "outside_boundary_args": [dict(row) for row in self.outside_boundary_args],
            "alternate_context_success_args": [
                dict(row) for row in self.alternate_context_success_args
            ],
            "success_metrics": list(self.success_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "ready_for_agent_policy_probe": self.ready_for_agent_policy_probe,
            "a33_ready": self.a33_ready,
            "support": int(self.support),
            "rule_counted_as_confirmation": self.rule_counted_as_confirmation,
        }


@dataclass(frozen=True)
class ProbeDecision:
    """One policy decision before execution."""

    condition: str
    action_name: str
    action_args: Dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    candidate_policy_used: bool = False
    candidate_score: float | None = None
    candidate_score_details: Dict[str, Any] = field(default_factory=dict)
    fallback_reason: str = ""


@dataclass(frozen=True)
class ProbeStep:
    """One executed closed-loop probe step."""

    step: int
    condition: str
    policy_selected_action: str
    action_args: Dict[str, Any]
    decision_reason: str
    candidate_policy_used: bool
    candidate_score: float | None
    candidate_score_details: Dict[str, Any]
    action6_arg_class: str
    failure_like_action6_arg: bool
    success_like_action6_arg: bool
    repositioning_action: bool
    local_patch_signal: float
    object_positions_signal: float
    changed_pixels: float
    contact_graph_signal: float
    useful_action6: bool
    useful_repositioning: bool
    useful_new_state: bool
    dead_end_or_cycle: bool
    state_signature_before: str
    state_signature_after: str
    levels_before: int
    levels_after: int
    game_state_before: str
    game_state_after: str
    measurements: Dict[str, Any]
    env_actions: int = 1
    error: str = ""
    truth_status: str = TRUTH_STATUS
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "condition": self.condition,
            "policy_selected_action": self.policy_selected_action,
            "action_args": dict(self.action_args),
            "decision_reason": self.decision_reason,
            "candidate_policy_used": self.candidate_policy_used,
            "candidate_score": self.candidate_score,
            "candidate_score_details": dict(self.candidate_score_details),
            "action6_arg_class": self.action6_arg_class,
            "failure_like_action6_arg": self.failure_like_action6_arg,
            "success_like_action6_arg": self.success_like_action6_arg,
            "repositioning_action": self.repositioning_action,
            "local_patch_signal": float(self.local_patch_signal),
            "object_positions_signal": float(self.object_positions_signal),
            "changed_pixels": float(self.changed_pixels),
            "contact_graph_signal": float(self.contact_graph_signal),
            "useful_action6": self.useful_action6,
            "useful_repositioning": self.useful_repositioning,
            "useful_new_state": self.useful_new_state,
            "dead_end_or_cycle": self.dead_end_or_cycle,
            "state_signature_before": self.state_signature_before,
            "state_signature_after": self.state_signature_after,
            "levels_before": int(self.levels_before),
            "levels_after": int(self.levels_after),
            "game_state_before": self.game_state_before,
            "game_state_after": self.game_state_after,
            "measurements": dict(self.measurements),
            "env_actions": int(self.env_actions),
            "error": self.error,
            "truth_status": self.truth_status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_bp35_sage_candidate_policy_probe(
    *,
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    budget: int = DEFAULT_BUDGET,
    game_id: str = DEFAULT_GAME_ID,
) -> Dict[str, Any]:
    scope_payload = _load_json(scope_consolidation_path)
    memory = candidate_policy_memory_from_scope(scope_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    baseline_steps = execute_probe_condition(
        condition=BASELINE_POLICY,
        memory=memory,
        environments_dir=env_dir,
        budget=budget,
        game_id=game_id,
    )
    candidate_steps = execute_probe_condition(
        condition=CANDIDATE_POLICY,
        memory=memory,
        environments_dir=env_dir,
        budget=budget,
        game_id=game_id,
    )
    condition_summaries = [
        summarize_probe_steps(BASELINE_POLICY, baseline_steps),
        summarize_probe_steps(CANDIDATE_POLICY, candidate_steps),
    ]
    comparison = compare_probe_conditions(condition_summaries)
    return {
        "config": {
            "scope_consolidation_path": str(scope_consolidation_path),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "budget": int(budget),
            "schema_version": "p1.bp35_sage_candidate_policy_probe.v1",
            "conditions": [BASELINE_POLICY, CANDIDATE_POLICY],
            "inputs_read": ["M3.24"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["M3", "A32", "A33"],
            "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        },
        "candidate_policy_memory": memory.to_dict(),
        "summary": {
            "conditions_run": 2,
            "budget_per_condition": int(budget),
            "candidate_policy_probe_ready": bool(
                memory.ready_for_agent_policy_probe and memory.enabled
            ),
            "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
            "a33_ready": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "candidate_policy_counted_as_confirmation": False,
        },
        "condition_summaries": condition_summaries,
        "comparison": comparison,
        "rollout_steps": {
            BASELINE_POLICY: [step.to_dict() for step in baseline_steps],
            CANDIDATE_POLICY: [step.to_dict() for step in candidate_steps],
        },
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "candidate_policy_counted_as_confirmation": False,
    }


def run_bp35_sage_candidate_policy_probe_matrix(
    *,
    scope_consolidation_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    tie_break_seeds: Sequence[int] = DEFAULT_TIE_BREAK_SEEDS,
    game_id: str = DEFAULT_GAME_ID,
    include_patch_similarity_only: bool = False,
    include_stale_guard: bool = False,
    include_soft_stale_guard: bool = False,
    include_conditional_refresh: bool = False,
    include_movement_refresh: bool = False,
) -> Dict[str, Any]:
    scope_payload = _load_json(scope_consolidation_path)
    memory = candidate_policy_memory_from_scope(scope_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    if include_movement_refresh:
        conditions = P1_POLICY_CONDITIONS_WITH_MOVEMENT_REFRESH
    elif include_conditional_refresh:
        conditions = P1_POLICY_CONDITIONS_WITH_CONDITIONAL_REFRESH
    elif include_soft_stale_guard:
        conditions = P1_POLICY_CONDITIONS_WITH_SOFT_STALE_GUARD
    elif include_stale_guard:
        conditions = P1_POLICY_CONDITIONS_WITH_STALE_GUARD
    elif include_patch_similarity_only:
        conditions = P1_POLICY_CONDITIONS_WITH_PATCH_ONLY
    else:
        conditions = P1_POLICY_CONDITIONS

    budget_runs = []
    for budget in [int(value) for value in budgets]:
        for tie_break_seed in [int(value) for value in tie_break_seeds]:
            condition_summaries = []
            for condition in conditions:
                steps = execute_probe_condition(
                    condition=condition,
                    memory=memory,
                    environments_dir=env_dir,
                    budget=budget,
                    game_id=game_id,
                    tie_break_seed=tie_break_seed,
                )
                condition_summaries.append(summarize_probe_steps(condition, steps))
            budget_runs.append(
                {
                    "budget": budget,
                    "tie_break_seed": tie_break_seed,
                    "conditions_run": list(conditions),
                    "condition_summaries": condition_summaries,
                    "comparison": compare_probe_conditions_with_ablation(
                        condition_summaries
                    ),
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": TRUTH_STATUS,
                    "revision_performed": False,
                    "wrong_confirmations": 0,
                    "candidate_policy_counted_as_confirmation": False,
                    "ablation_counted_as_confirmation": False,
                    "patch_similarity_only_counted_as_confirmation": False,
                    "soft_stale_guard_counted_as_confirmation": False,
                    "conditional_refresh_counted_as_confirmation": False,
                    "movement_refresh_counted_as_confirmation": False,
                }
            )

    aggregate = aggregate_probe_matrix(budget_runs)
    return {
        "config": {
            "scope_consolidation_path": str(scope_consolidation_path),
            "environments_dir": str(env_dir),
            "game_id": game_id,
            "budgets": [int(value) for value in budgets],
            "tie_break_seeds": [int(value) for value in tie_break_seeds],
            "schema_version": "p1.bp35_sage_candidate_policy_probe_matrix.v1",
            "conditions": list(conditions),
            "include_patch_similarity_only": bool(include_patch_similarity_only),
            "include_stale_guard": bool(include_stale_guard),
            "include_soft_stale_guard": bool(include_soft_stale_guard),
            "include_conditional_refresh": bool(include_conditional_refresh),
            "include_movement_refresh": bool(include_movement_refresh),
            "movement_refresh_candidates": list(MOVEMENT_REFRESH_CANDIDATES),
            "inputs_read": ["M3.24"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["M3", "A32", "A33"],
            "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
            "ablation_policy_status": "EXPERIMENTAL_POLICY_ABLATION_ONLY",
            "patch_similarity_only_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_patch_similarity_only
                or include_stale_guard
                or include_soft_stale_guard
                or include_conditional_refresh
                or include_movement_refresh
                else "NOT_RUN"
            ),
            "stale_guard_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_stale_guard
                or include_soft_stale_guard
                or include_conditional_refresh
                or include_movement_refresh
                else "NOT_RUN"
            ),
            "soft_stale_guard_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_soft_stale_guard
                or include_conditional_refresh
                or include_movement_refresh
                else "NOT_RUN"
            ),
            "conditional_refresh_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_conditional_refresh or include_movement_refresh
                else "NOT_RUN"
            ),
            "movement_refresh_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_movement_refresh
                else "NOT_RUN"
            ),
        },
        "candidate_policy_memory": memory.to_dict(),
        "summary": {
            "budget_runs": len(budget_runs),
            "budgets_tested": [int(value) for value in budgets],
            "tie_break_seeds_tested": [int(value) for value in tie_break_seeds],
            "conditions_per_run": len(conditions),
            "candidate_policy_probe_ready": bool(
                memory.ready_for_agent_policy_probe and memory.enabled
            ),
            "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
            "ablation_policy_status": "EXPERIMENTAL_POLICY_ABLATION_ONLY",
            "patch_similarity_only_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_patch_similarity_only
                or include_stale_guard
                or include_soft_stale_guard
                or include_conditional_refresh
                or include_movement_refresh
                else "NOT_RUN"
            ),
            "stale_guard_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_stale_guard
                or include_soft_stale_guard
                or include_conditional_refresh
                or include_movement_refresh
                else "NOT_RUN"
            ),
            "soft_stale_guard_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_soft_stale_guard
                or include_conditional_refresh
                or include_movement_refresh
                else "NOT_RUN"
            ),
            "conditional_refresh_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_conditional_refresh or include_movement_refresh
                else "NOT_RUN"
            ),
            "movement_refresh_policy_status": (
                "EXPERIMENTAL_POLICY_ABLATION_ONLY"
                if include_movement_refresh
                else "NOT_RUN"
            ),
            "a33_ready": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "candidate_policy_counted_as_confirmation": False,
            "ablation_counted_as_confirmation": False,
            "patch_similarity_only_counted_as_confirmation": False,
            "soft_stale_guard_counted_as_confirmation": False,
            "conditional_refresh_counted_as_confirmation": False,
            "movement_refresh_counted_as_confirmation": False,
        },
        "aggregate": aggregate,
        "budget_runs": budget_runs,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "candidate_policy_counted_as_confirmation": False,
        "ablation_counted_as_confirmation": False,
        "patch_similarity_only_counted_as_confirmation": False,
        "soft_stale_guard_counted_as_confirmation": False,
        "conditional_refresh_counted_as_confirmation": False,
        "movement_refresh_counted_as_confirmation": False,
    }


def candidate_policy_memory_from_scope(payload: Mapping[str, Any]) -> CandidatePolicyMemory:
    rows = [
        dict(row)
        for row in payload.get("scope_consolidations", []) or []
        if isinstance(row, Mapping)
    ]
    if not rows:
        return CandidatePolicyMemory(enabled=False)
    row = rows[0]
    ready = (
        str(row.get("scope_assessment", "")) == SCOPE_EXPANDED_CANDIDATE_ONLY
        and bool(row.get("ready_for_agent_policy_probe"))
        and not bool(row.get("a33_ready"))
        and int(row.get("support", 0) or 0) == 0
        and str(row.get("revision_status", "")) == "CANDIDATE_ONLY"
    )
    recommended = dict(row.get("recommended_agent_policy_probe", {}) or {})
    return CandidatePolicyMemory(
        enabled=ready,
        source_scope_consolidation_id=str(row.get("scope_consolidation_id", "")),
        source_status=str(row.get("status", "")),
        source_revision_status=str(row.get("revision_status", "")),
        source_truth_status=str(row.get("truth_status", "")),
        game_id=str(row.get("game_id", DEFAULT_GAME_ID)),
        target_action=str(row.get("target_action", "ACTION6")),
        repositioning_action=str(recommended.get("use_repositioning_action", "ACTION4")),
        candidate_rule_family=str(row.get("candidate_rule_family", "")),
        known_success_args=dedupe_args(row.get("known_success_args", []) or []),
        known_failed_args=dedupe_args(row.get("known_failed_args", []) or []),
        outside_boundary_args=dedupe_args(row.get("outside_boundary_args", []) or []),
        alternate_context_success_args=dedupe_args(
            row.get("alternate_context_success_args", []) or []
        ),
        success_metrics=tuple(str(item) for item in row.get("success_metrics", []) or []),
        diagnostic_metrics=tuple(
            str(item) for item in row.get("diagnostic_metrics", []) or []
        ),
        ready_for_agent_policy_probe=bool(row.get("ready_for_agent_policy_probe")),
        a33_ready=bool(row.get("a33_ready")),
        support=int(row.get("support", 0) or 0),
        rule_counted_as_confirmation=bool(row.get("rule_counted_as_confirmation")),
    )


def execute_probe_condition(
    *,
    condition: str,
    memory: CandidatePolicyMemory,
    environments_dir: str | Path,
    budget: int,
    game_id: str,
    tie_break_seed: int = 0,
) -> Tuple[ProbeStep, ...]:
    env = _make_env(game_id, environments_dir)
    current_frame = _reset_env(env)
    action_history: list[str] = []
    used_action6_args: list[Dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    seen_states: set[str] = set()
    steps: list[ProbeStep] = []
    initial = snapshot_frame(current_frame)
    seen_states.add(state_signature(initial.grid, initial.levels_completed, initial.game_state))

    for step_index in range(max(0, int(budget))):
        try:
            before = snapshot_frame(current_frame)
        except ValueError:
            break
        before_signature = state_signature(
            before.grid,
            before.levels_completed,
            before.game_state,
        )
        valid_actions = list(_valid_actions(env))
        decision = select_probe_decision(
            condition=condition,
            memory=memory,
            before_grid=before.grid,
            valid_actions=valid_actions,
            action_history=tuple(action_history),
            used_action6_args=tuple(used_action6_args),
            action_counts=action_counts,
            previous_steps=tuple(steps),
            tie_break_seed=tie_break_seed,
        )
        selected_action = concrete_action_for_decision(valid_actions, decision)
        if selected_action is None:
            steps.append(
                error_step(
                    step_index,
                    condition=condition,
                    decision=decision,
                    before=before,
                    before_signature=before_signature,
                    error="selected_action_not_available",
                )
            )
            break
        after_frame = _step_env_action(env, selected_action)
        if after_frame is None:
            steps.append(
                error_step(
                    step_index,
                    condition=condition,
                    decision=decision,
                    before=before,
                    before_signature=before_signature,
                    error="env_step_returned_no_frame",
                )
            )
            break
        try:
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
        except ValueError as exc:
            steps.append(
                error_step(
                    step_index,
                    condition=condition,
                    decision=decision,
                    before=before,
                    before_signature=before_signature,
                    error=f"snapshot_frame_after_step_failed:{exc}",
                )
            )
            break
        action_name = str(getattr(selected_action, "name", decision.action_name))
        action_args = dict(getattr(selected_action, "action_args", {}) or {})
        measurements = measure_probe_metrics(before.grid, after.grid, action_args)
        after_signature = state_signature(
            after.grid,
            after.levels_completed,
            after.game_state,
        )
        cycle = after_signature in seen_states
        action_class = classify_action6_args(action_args, memory) if action_name == "ACTION6" else ""
        failure_like = bool(action_name == "ACTION6" and action_class in {"known_failure", "outside_boundary"})
        success_like = bool(action_name == "ACTION6" and action_class in {"known_success", "alternate_context_success"})
        local_signal = metric_signal(
            measurements["local_patch_before_after"],
            "local_patch_before_after",
        )
        object_signal = metric_signal(
            measurements["object_positions_before_after"],
            "object_positions_before_after",
        )
        changed_pixels = float(
            measurements["changed_pixels"].get("changed_pixels", 0) or 0
        )
        contact_signal = metric_signal(
            measurements["contact_graph_before_after"],
            "contact_graph_before_after",
        )
        useful_action6 = bool(
            action_name == "ACTION6" and (local_signal > 0 or object_signal > 0)
        )
        useful_repositioning = bool(
            action_name == memory.repositioning_action
            and (changed_pixels > 0 or object_signal > 0)
        )
        useful_new_state = bool(
            (useful_action6 or useful_repositioning)
            and not cycle
            and after.levels_completed >= before.levels_completed
        )
        step = ProbeStep(
            step=step_index,
            condition=condition,
            policy_selected_action=action_name,
            action_args=action_args,
            decision_reason=decision.decision_reason,
            candidate_policy_used=decision.candidate_policy_used,
            candidate_score=decision.candidate_score,
            candidate_score_details=decision.candidate_score_details,
            action6_arg_class=action_class,
            failure_like_action6_arg=failure_like,
            success_like_action6_arg=success_like,
            repositioning_action=action_name == memory.repositioning_action,
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
        steps.append(step)
        seen_states.add(after_signature)
        action_counts[action_name] += 1
        action_history.append(action_name)
        if action_name == "ACTION6" and action_args:
            used_action6_args.append(dict(action_args))
        current_frame = after_frame
    return tuple(steps)


def select_probe_decision(
    *,
    condition: str,
    memory: CandidatePolicyMemory,
    before_grid: Any,
    valid_actions: Sequence[Any],
    action_history: Sequence[str],
    used_action6_args: Sequence[Mapping[str, Any]],
    action_counts: Mapping[str, int],
    previous_steps: Sequence[ProbeStep] = (),
    tie_break_seed: int = 0,
) -> ProbeDecision:
    if condition == ACTION4_ONLY_POLICY and memory.enabled:
        return select_action4_only_decision(
            memory=memory,
            valid_actions=valid_actions,
            action_history=action_history,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
        )
    if condition == PATCH_SIMILARITY_ONLY_POLICY and memory.enabled:
        return select_patch_similarity_only_decision(
            memory=memory,
            before_grid=before_grid,
            valid_actions=valid_actions,
            used_action6_args=used_action6_args,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
        )
    if condition == PATCH_SIMILARITY_STALE_GUARD_POLICY and memory.enabled:
        return select_patch_similarity_stale_guard_decision(
            memory=memory,
            before_grid=before_grid,
            valid_actions=valid_actions,
            used_action6_args=used_action6_args,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
        )
    if condition == PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY and memory.enabled:
        return select_patch_similarity_soft_stale_guard_decision(
            memory=memory,
            before_grid=before_grid,
            valid_actions=valid_actions,
            used_action6_args=used_action6_args,
            previous_steps=previous_steps,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
        )
    if condition == CONDITIONAL_ACTION4_REFRESH_POLICY and memory.enabled:
        return select_conditional_action4_refresh_decision(
            memory=memory,
            before_grid=before_grid,
            valid_actions=valid_actions,
            action_history=action_history,
            used_action6_args=used_action6_args,
            previous_steps=previous_steps,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
        )
    if condition == CONDITIONAL_MOVEMENT_REFRESH_POLICY and memory.enabled:
        return select_conditional_movement_refresh_decision(
            memory=memory,
            before_grid=before_grid,
            valid_actions=valid_actions,
            action_history=action_history,
            used_action6_args=used_action6_args,
            previous_steps=previous_steps,
            action_counts=action_counts,
            tie_break_seed=tie_break_seed,
        )

    if condition != CANDIDATE_POLICY or not memory.enabled:
        return select_baseline_decision(
            valid_actions,
            action_counts=action_counts,
            condition=condition,
            tie_break_seed=tie_break_seed,
        )

    valid_names = {str(getattr(action, "name", "")) for action in valid_actions}
    last_action = str(action_history[-1]) if action_history else ""
    if (
        memory.repositioning_action in valid_names
        and last_action != memory.repositioning_action
    ):
        return ProbeDecision(
            condition=condition,
            action_name=memory.repositioning_action,
            decision_reason="candidate_policy_reposition_before_retarget",
            candidate_policy_used=True,
        )

    candidates = [
        action for action in valid_actions if str(getattr(action, "name", "")) == memory.target_action
    ]
    scored = [
        score_action6_candidate(
            dict(getattr(action, "action_args", {}) or {}),
            before_grid=before_grid,
            memory=memory,
            used_action6_args=used_action6_args,
        )
        for action in candidates
        if getattr(action, "action_args", None)
    ]
    if scored:
        best = min(
            scored,
            key=lambda row: (
                float(row["score"]),
                deterministic_args_tie_break(row["action_args"], tie_break_seed),
            ),
        )
        return ProbeDecision(
            condition=condition,
            action_name=memory.target_action,
            action_args=dict(best["action_args"]),
            decision_reason="candidate_policy_patch_similarity_retarget",
            candidate_policy_used=True,
            candidate_score=float(best["score"]),
            candidate_score_details=best,
        )
    return select_baseline_decision(
        valid_actions,
        action_counts=action_counts,
        condition=condition,
        tie_break_seed=tie_break_seed,
        fallback_reason="candidate_policy_no_action6_candidate_available",
    )


def select_action4_only_decision(
    *,
    memory: CandidatePolicyMemory,
    valid_actions: Sequence[Any],
    action_history: Sequence[str],
    action_counts: Mapping[str, int],
    tie_break_seed: int = 0,
) -> ProbeDecision:
    valid_names = {str(getattr(action, "name", "")) for action in valid_actions}
    last_action = str(action_history[-1]) if action_history else ""
    if (
        memory.repositioning_action in valid_names
        and last_action != memory.repositioning_action
    ):
        return ProbeDecision(
            condition=ACTION4_ONLY_POLICY,
            action_name=memory.repositioning_action,
            decision_reason="action4_only_reposition_without_patch_similarity",
            candidate_policy_used=True,
            candidate_score_details={
                "ablation": "repositioning_without_patch_similarity",
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
            },
        )

    action6_candidates = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == memory.target_action
    ]
    if action6_candidates:
        best = select_deterministic_action6_candidate(
            action6_candidates,
            tie_break_seed=tie_break_seed,
        )
        return ProbeDecision(
            condition=ACTION4_ONLY_POLICY,
            action_name=memory.target_action,
            action_args=dict(getattr(best, "action_args", {}) or {}),
            decision_reason="action4_only_naive_action6_after_reposition",
            candidate_policy_used=True,
            candidate_score_details={
                "ablation": "no_patch_similarity_scoring",
                "tie_break_seed": int(tie_break_seed),
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
            },
        )

    return select_baseline_decision(
        valid_actions,
        action_counts=action_counts,
        condition=ACTION4_ONLY_POLICY,
        tie_break_seed=tie_break_seed,
        fallback_reason="action4_only_no_action6_candidate_available",
    )


def select_patch_similarity_only_decision(
    *,
    memory: CandidatePolicyMemory,
    before_grid: Any,
    valid_actions: Sequence[Any],
    used_action6_args: Sequence[Mapping[str, Any]],
    action_counts: Mapping[str, int],
    tie_break_seed: int = 0,
) -> ProbeDecision:
    candidates = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == memory.target_action
    ]
    scored = [
        score_action6_candidate(
            dict(getattr(action, "action_args", {}) or {}),
            before_grid=before_grid,
            memory=memory,
            used_action6_args=used_action6_args,
        )
        for action in candidates
        if getattr(action, "action_args", None)
    ]
    if scored:
        best = min(
            scored,
            key=lambda row: (
                float(row["score"]),
                deterministic_args_tie_break(row["action_args"], tie_break_seed),
            ),
        )
        details = dict(best)
        details["ablation"] = "patch_similarity_without_repositioning"
        details["repositioning_action_allowed"] = False
        return ProbeDecision(
            condition=PATCH_SIMILARITY_ONLY_POLICY,
            action_name=memory.target_action,
            action_args=dict(best["action_args"]),
            decision_reason="patch_similarity_only_action6_without_reposition",
            candidate_policy_used=True,
            candidate_score=float(best["score"]),
            candidate_score_details=details,
        )

    no_reposition_actions = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) != memory.repositioning_action
    ]
    return select_baseline_decision(
        no_reposition_actions,
        action_counts=action_counts,
        condition=PATCH_SIMILARITY_ONLY_POLICY,
        tie_break_seed=tie_break_seed,
        fallback_reason="patch_similarity_only_no_action6_candidate_available",
    )


def select_patch_similarity_stale_guard_decision(
    *,
    memory: CandidatePolicyMemory,
    before_grid: Any,
    valid_actions: Sequence[Any],
    used_action6_args: Sequence[Mapping[str, Any]],
    action_counts: Mapping[str, int],
    tie_break_seed: int = 0,
) -> ProbeDecision:
    candidates = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == memory.target_action
    ]
    scored = [
        score_action6_candidate(
            dict(getattr(action, "action_args", {}) or {}),
            before_grid=before_grid,
            memory=memory,
            used_action6_args=used_action6_args,
        )
        for action in candidates
        if getattr(action, "action_args", None)
    ]
    fresh_safe = [
        row
        for row in scored
        if not bool(row.get("previously_used"))
        and not bool(row.get("exact_known_failure"))
        and not bool(row.get("exact_outside_boundary"))
    ]
    if fresh_safe:
        best = min(
            fresh_safe,
            key=lambda row: (
                float(row["score"]),
                deterministic_args_tie_break(row["action_args"], tie_break_seed),
            ),
        )
        details = dict(best)
        details["stale_guard"] = "fresh_safe_action6_required"
        details["repositioning_action_allowed"] = False
        details["repeated_args_filtered"] = len(
            [row for row in scored if bool(row.get("previously_used"))]
        )
        details["failure_like_args_filtered"] = len(
            [
                row
                for row in scored
                if bool(row.get("exact_known_failure"))
                or bool(row.get("exact_outside_boundary"))
            ]
        )
        return ProbeDecision(
            condition=PATCH_SIMILARITY_STALE_GUARD_POLICY,
            action_name=memory.target_action,
            action_args=dict(best["action_args"]),
            decision_reason="patch_similarity_stale_guard_fresh_action6",
            candidate_policy_used=True,
            candidate_score=float(best["score"]),
            candidate_score_details=details,
        )

    fallback_actions = [
        action
        for action in valid_actions
        if str(getattr(action, "name", ""))
        not in {memory.repositioning_action, memory.target_action}
    ]
    decision = select_baseline_decision(
        fallback_actions,
        action_counts=action_counts,
        condition=PATCH_SIMILARITY_STALE_GUARD_POLICY,
        tie_break_seed=tie_break_seed,
        fallback_reason="stale_guard_no_fresh_safe_action6_available",
    )
    return ProbeDecision(
        condition=PATCH_SIMILARITY_STALE_GUARD_POLICY,
        action_name=decision.action_name,
        action_args=dict(decision.action_args),
        decision_reason="patch_similarity_stale_guard_no_fresh_action6_fallback",
        candidate_policy_used=True,
        candidate_score=None,
        candidate_score_details={
            "stale_guard_triggered": True,
            "fresh_safe_action6_available": False,
            "scored_action6_candidates": len(scored),
            "repeated_args_filtered": len(
                [row for row in scored if bool(row.get("previously_used"))]
            ),
            "failure_like_args_filtered": len(
                [
                    row
                    for row in scored
                    if bool(row.get("exact_known_failure"))
                    or bool(row.get("exact_outside_boundary"))
                ]
            ),
            "fallback_action": decision.action_name,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        },
        fallback_reason=decision.fallback_reason,
    )


def select_patch_similarity_soft_stale_guard_decision(
    *,
    memory: CandidatePolicyMemory,
    before_grid: Any,
    valid_actions: Sequence[Any],
    used_action6_args: Sequence[Mapping[str, Any]],
    previous_steps: Sequence[ProbeStep],
    action_counts: Mapping[str, int],
    tie_break_seed: int = 0,
) -> ProbeDecision:
    effect_stats = action6_effect_stats(previous_steps)
    candidates = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == memory.target_action
    ]
    scored = []
    for action in candidates:
        action_args = dict(getattr(action, "action_args", {}) or {})
        if not action_args:
            continue
        row = score_action6_candidate(
            action_args,
            before_grid=before_grid,
            memory=memory,
            used_action6_args=used_action6_args,
        )
        stats = effect_stats.get(_args_key(action_args), default_action6_effect_stats())
        row["effect_memory"] = stats
        row["soft_stale_blocked"] = bool(
            stats["times_selected"] > 0
            and stats["consecutive_no_new_effects"] >= 1
        )
        scored.append(row)

    allowed = [
        row
        for row in scored
        if not bool(row.get("exact_known_failure"))
        and not bool(row.get("exact_outside_boundary"))
        and not bool(row.get("soft_stale_blocked"))
    ]
    if allowed:
        best = min(
            allowed,
            key=lambda row: (
                float(row["score"]),
                deterministic_args_tie_break(row["action_args"], tie_break_seed),
            ),
        )
        details = dict(best)
        details["soft_stale_guard"] = "block_only_after_no_new_effect"
        details["repositioning_action_allowed"] = False
        details["blocked_stale_args"] = len(
            [row for row in scored if bool(row.get("soft_stale_blocked"))]
        )
        details["repeated_but_still_effective_args_available"] = len(
            [
                row
                for row in scored
                if bool(row.get("previously_used"))
                and not bool(row.get("soft_stale_blocked"))
            ]
        )
        details["failure_like_args_filtered"] = len(
            [
                row
                for row in scored
                if bool(row.get("exact_known_failure"))
                or bool(row.get("exact_outside_boundary"))
            ]
        )
        return ProbeDecision(
            condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
            action_name=memory.target_action,
            action_args=dict(best["action_args"]),
            decision_reason="patch_similarity_soft_stale_guard_effective_action6",
            candidate_policy_used=True,
            candidate_score=float(best["score"]),
            candidate_score_details=details,
        )

    fallback_actions = [
        action
        for action in valid_actions
        if str(getattr(action, "name", ""))
        not in {memory.repositioning_action, memory.target_action}
    ]
    decision = select_baseline_decision(
        fallback_actions,
        action_counts=action_counts,
        condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        tie_break_seed=tie_break_seed,
        fallback_reason="soft_stale_guard_no_effective_action6_available",
    )
    return ProbeDecision(
        condition=PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY,
        action_name=decision.action_name,
        action_args=dict(decision.action_args),
        decision_reason="patch_similarity_soft_stale_guard_no_effective_action6_fallback",
        candidate_policy_used=True,
        candidate_score=None,
        candidate_score_details={
            "soft_stale_guard_triggered": True,
            "effective_action6_available": False,
            "scored_action6_candidates": len(scored),
            "blocked_stale_args": len(
                [row for row in scored if bool(row.get("soft_stale_blocked"))]
            ),
            "failure_like_args_filtered": len(
                [
                    row
                    for row in scored
                    if bool(row.get("exact_known_failure"))
                    or bool(row.get("exact_outside_boundary"))
                ]
            ),
            "fallback_action": decision.action_name,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        },
        fallback_reason=decision.fallback_reason,
    )


def select_conditional_action4_refresh_decision(
    *,
    memory: CandidatePolicyMemory,
    before_grid: Any,
    valid_actions: Sequence[Any],
    action_history: Sequence[str],
    used_action6_args: Sequence[Mapping[str, Any]],
    previous_steps: Sequence[ProbeStep],
    action_counts: Mapping[str, int],
    tie_break_seed: int = 0,
) -> ProbeDecision:
    effect_stats = action6_effect_stats(previous_steps)
    candidates = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == memory.target_action
    ]
    scored = []
    for action in candidates:
        action_args = dict(getattr(action, "action_args", {}) or {})
        if not action_args:
            continue
        row = score_action6_candidate(
            action_args,
            before_grid=before_grid,
            memory=memory,
            used_action6_args=used_action6_args,
        )
        stats = effect_stats.get(_args_key(action_args), default_action6_effect_stats())
        row["effect_memory"] = stats
        row["soft_stale_blocked"] = bool(
            stats["times_selected"] > 0
            and stats["consecutive_no_new_effects"] >= 1
        )
        scored.append(row)

    allowed = [
        row
        for row in scored
        if not bool(row.get("exact_known_failure"))
        and not bool(row.get("exact_outside_boundary"))
        and not bool(row.get("soft_stale_blocked"))
    ]
    if allowed:
        best = min(
            allowed,
            key=lambda row: (
                float(row["score"]),
                deterministic_args_tie_break(row["action_args"], tie_break_seed),
            ),
        )
        details = dict(best)
        details["conditional_refresh"] = (
            "action6_first_refresh_only_after_soft_stale_exhaustion"
        )
        details["refresh_action_selected"] = False
        details["blocked_stale_args"] = len(
            [row for row in scored if bool(row.get("soft_stale_blocked"))]
        )
        details["failure_like_args_filtered"] = len(
            [
                row
                for row in scored
                if bool(row.get("exact_known_failure"))
                or bool(row.get("exact_outside_boundary"))
            ]
        )
        return ProbeDecision(
            condition=CONDITIONAL_ACTION4_REFRESH_POLICY,
            action_name=memory.target_action,
            action_args=dict(best["action_args"]),
            decision_reason="conditional_refresh_effective_action6",
            candidate_policy_used=True,
            candidate_score=float(best["score"]),
            candidate_score_details=details,
        )

    valid_names = {str(getattr(action, "name", "")) for action in valid_actions}
    last_action = str(action_history[-1]) if action_history else ""
    if (
        memory.repositioning_action in valid_names
        and last_action != memory.repositioning_action
    ):
        return ProbeDecision(
            condition=CONDITIONAL_ACTION4_REFRESH_POLICY,
            action_name=memory.repositioning_action,
            decision_reason="conditional_action4_refresh_after_soft_stale_exhausted",
            candidate_policy_used=True,
            candidate_score_details={
                "conditional_refresh_triggered": True,
                "effective_action6_available": False,
                "refresh_action": memory.repositioning_action,
                "scored_action6_candidates": len(scored),
                "blocked_stale_args": len(
                    [row for row in scored if bool(row.get("soft_stale_blocked"))]
                ),
                "failure_like_args_filtered": len(
                    [
                        row
                        for row in scored
                        if bool(row.get("exact_known_failure"))
                        or bool(row.get("exact_outside_boundary"))
                    ]
                ),
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
            },
        )

    fallback_actions = [
        action
        for action in valid_actions
        if str(getattr(action, "name", ""))
        not in {memory.repositioning_action, memory.target_action}
    ]
    decision = select_baseline_decision(
        fallback_actions,
        action_counts=action_counts,
        condition=CONDITIONAL_ACTION4_REFRESH_POLICY,
        tie_break_seed=tie_break_seed,
        fallback_reason="conditional_refresh_no_effective_action6_or_refresh_available",
    )
    return ProbeDecision(
        condition=CONDITIONAL_ACTION4_REFRESH_POLICY,
        action_name=decision.action_name,
        action_args=dict(decision.action_args),
        decision_reason="conditional_refresh_no_effective_action6_fallback",
        candidate_policy_used=True,
        candidate_score=None,
        candidate_score_details={
            "conditional_refresh_triggered": False,
            "effective_action6_available": False,
            "refresh_action_available": memory.repositioning_action in valid_names,
            "scored_action6_candidates": len(scored),
            "fallback_action": decision.action_name,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        },
        fallback_reason=decision.fallback_reason,
    )


def select_conditional_movement_refresh_decision(
    *,
    memory: CandidatePolicyMemory,
    before_grid: Any,
    valid_actions: Sequence[Any],
    action_history: Sequence[str],
    used_action6_args: Sequence[Mapping[str, Any]],
    previous_steps: Sequence[ProbeStep],
    action_counts: Mapping[str, int],
    tie_break_seed: int = 0,
) -> ProbeDecision:
    effect_stats = action6_effect_stats(previous_steps)
    candidates = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == memory.target_action
    ]
    scored = []
    for action in candidates:
        action_args = dict(getattr(action, "action_args", {}) or {})
        if not action_args:
            continue
        row = score_action6_candidate(
            action_args,
            before_grid=before_grid,
            memory=memory,
            used_action6_args=used_action6_args,
        )
        stats = effect_stats.get(_args_key(action_args), default_action6_effect_stats())
        row["effect_memory"] = stats
        row["soft_stale_blocked"] = bool(
            stats["times_selected"] > 0
            and stats["consecutive_no_new_effects"] >= 1
        )
        scored.append(row)

    allowed = [
        row
        for row in scored
        if not bool(row.get("exact_known_failure"))
        and not bool(row.get("exact_outside_boundary"))
        and not bool(row.get("soft_stale_blocked"))
    ]
    if allowed:
        best = min(
            allowed,
            key=lambda row: (
                float(row["score"]),
                deterministic_args_tie_break(row["action_args"], tie_break_seed),
            ),
        )
        details = dict(best)
        details["movement_refresh"] = (
            "action6_first_movement_refresh_only_after_soft_stale_exhaustion"
        )
        details["refresh_action_selected"] = False
        details["movement_refresh_candidates"] = list(MOVEMENT_REFRESH_CANDIDATES)
        details["blocked_stale_args"] = len(
            [row for row in scored if bool(row.get("soft_stale_blocked"))]
        )
        details["failure_like_args_filtered"] = len(
            [
                row
                for row in scored
                if bool(row.get("exact_known_failure"))
                or bool(row.get("exact_outside_boundary"))
            ]
        )
        return ProbeDecision(
            condition=CONDITIONAL_MOVEMENT_REFRESH_POLICY,
            action_name=memory.target_action,
            action_args=dict(best["action_args"]),
            decision_reason="conditional_movement_refresh_effective_action6",
            candidate_policy_used=True,
            candidate_score=float(best["score"]),
            candidate_score_details=details,
        )

    refresh_action = select_movement_refresh_action(
        valid_actions=valid_actions,
        action_history=action_history,
        action_counts=action_counts,
    )
    if refresh_action:
        return ProbeDecision(
            condition=CONDITIONAL_MOVEMENT_REFRESH_POLICY,
            action_name=refresh_action,
            decision_reason="conditional_movement_refresh_after_soft_stale_exhausted",
            candidate_policy_used=True,
            candidate_score_details={
                "conditional_movement_refresh_triggered": True,
                "effective_action6_available": False,
                "refresh_action": refresh_action,
                "refresh_candidates_considered": list(MOVEMENT_REFRESH_CANDIDATES),
                "scored_action6_candidates": len(scored),
                "blocked_stale_args": len(
                    [row for row in scored if bool(row.get("soft_stale_blocked"))]
                ),
                "failure_like_args_filtered": len(
                    [
                        row
                        for row in scored
                        if bool(row.get("exact_known_failure"))
                        or bool(row.get("exact_outside_boundary"))
                    ]
                ),
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": TRUTH_STATUS,
            },
        )

    fallback_actions = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) != memory.target_action
    ]
    decision = select_baseline_decision(
        fallback_actions,
        action_counts=action_counts,
        condition=CONDITIONAL_MOVEMENT_REFRESH_POLICY,
        tie_break_seed=tie_break_seed,
        fallback_reason="movement_refresh_no_effective_action6_or_refresh_available",
    )
    return ProbeDecision(
        condition=CONDITIONAL_MOVEMENT_REFRESH_POLICY,
        action_name=decision.action_name,
        action_args=dict(decision.action_args),
        decision_reason="conditional_movement_refresh_no_effective_action6_fallback",
        candidate_policy_used=True,
        candidate_score=None,
        candidate_score_details={
            "conditional_movement_refresh_triggered": False,
            "effective_action6_available": False,
            "refresh_candidates_considered": list(MOVEMENT_REFRESH_CANDIDATES),
            "fallback_action": decision.action_name,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": TRUTH_STATUS,
        },
        fallback_reason=decision.fallback_reason,
    )


def select_movement_refresh_action(
    *,
    valid_actions: Sequence[Any],
    action_history: Sequence[str],
    action_counts: Mapping[str, int],
) -> str:
    valid_names = {str(getattr(action, "name", "")) for action in valid_actions}
    last_action = str(action_history[-1]) if action_history else ""
    candidates = [
        name
        for name in MOVEMENT_REFRESH_CANDIDATES
        if name in valid_names and name != last_action
    ]
    if not candidates:
        return ""
    return min(
        candidates,
        key=lambda name: (
            int(action_counts.get(name, 0) or 0),
            MOVEMENT_REFRESH_CANDIDATES.index(name),
        ),
    )


def select_baseline_decision(
    valid_actions: Sequence[Any],
    *,
    action_counts: Mapping[str, int],
    condition: str = BASELINE_POLICY,
    tie_break_seed: int = 0,
    fallback_reason: str = "",
) -> ProbeDecision:
    order = deterministic_action_order(tie_break_seed)
    valid_names = {str(getattr(action, "name", "")) for action in valid_actions}
    available_order = [name for name in order if name in valid_names]
    if not available_order:
        return ProbeDecision(
            condition=condition,
            action_name="",
            decision_reason="baseline_no_available_action",
            fallback_reason=fallback_reason,
        )
    selected_name = min(
        available_order,
        key=lambda name: (int(action_counts.get(name, 0) or 0), order.index(name)),
    )
    selected_args: Dict[str, Any] = {}
    if selected_name == "ACTION6":
        action6_candidates = [
            action
            for action in valid_actions
            if str(getattr(action, "name", "")) == "ACTION6"
        ]
        if action6_candidates:
            selected = select_deterministic_action6_candidate(
                action6_candidates,
                tie_break_seed=tie_break_seed,
            )
            selected_args = dict(getattr(selected, "action_args", {}) or {})
    return ProbeDecision(
        condition=condition,
        action_name=selected_name,
        action_args=selected_args,
        decision_reason="baseline_round_robin_neutral_exploration",
        fallback_reason=fallback_reason,
    )


def score_action6_candidate(
    action_args: Mapping[str, Any],
    *,
    before_grid: Any,
    memory: CandidatePolicyMemory,
    used_action6_args: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    args = dict(action_args)
    success_seeds = dedupe_args(
        [*memory.known_success_args, *memory.alternate_context_success_args]
    )
    failure_like = dedupe_args([*memory.known_failed_args, *memory.outside_boundary_args])
    success_distance = min_patch_distance(args, success_seeds, before_grid)
    failure_distance = min_patch_distance(args, failure_like, before_grid)
    exact_success = _args_key(args) in {_args_key(row) for row in success_seeds}
    exact_failure = _args_key(args) in {_args_key(row) for row in memory.known_failed_args}
    exact_boundary = _args_key(args) in {_args_key(row) for row in memory.outside_boundary_args}
    previously_used = _args_key(args) in {_args_key(row) for row in used_action6_args}
    score = float(success_distance) - 0.5 * float(failure_distance)
    if exact_success:
        score -= 20.0
    if exact_failure:
        score += 100.0
    elif exact_boundary:
        score += 40.0
    if previously_used:
        score += 10.0
    return {
        "action_args": args,
        "score": round(score, 4),
        "success_patch_distance": success_distance,
        "failure_patch_distance": failure_distance,
        "exact_known_success": exact_success,
        "exact_known_failure": exact_failure,
        "exact_outside_boundary": exact_boundary,
        "previously_used": previously_used,
        "candidate_rule_family": memory.candidate_rule_family,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def concrete_action_for_decision(
    valid_actions: Sequence[Any],
    decision: ProbeDecision,
) -> Any | None:
    matches = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == str(decision.action_name)
    ]
    if not decision.action_args:
        return matches[0] if matches else None
    desired_key = _args_key(decision.action_args)
    for action in matches:
        if _args_key(dict(getattr(action, "action_args", {}) or {})) == desired_key:
            return action
    return None


def measure_probe_metrics(
    before_grid: Any,
    after_grid: Any,
    action_args: Mapping[str, Any],
) -> Dict[str, Any]:
    measurements = {
        "local_patch_before_after": measure_required_observation(
            before_grid,
            after_grid,
            required_observation="local_patch_before_after",
            action_args=action_args,
        ),
        "object_positions_before_after": measure_required_observation(
            before_grid,
            after_grid,
            required_observation="object_positions_before_after",
            action_args=action_args,
        ),
        "contact_graph_before_after": measure_required_observation(
            before_grid,
            after_grid,
            required_observation="contact_graph_before_after",
            action_args=action_args,
        ),
    }
    before = np.asarray(before_grid, dtype=np.int32)
    after = np.asarray(after_grid, dtype=np.int32)
    changed_pixels = int(np.sum(before != after)) if before.shape == after.shape else int(max(before.size, after.size))
    measurements["changed_pixels"] = {
        "metric": "changed_pixels",
        "measurable": before.shape == after.shape,
        "changed_pixels": changed_pixels,
        "changed": changed_pixels > 0,
        "signal_source": "raw_changed_pixels",
    }
    return measurements


def summarize_probe_steps(
    condition: str,
    steps: Sequence[ProbeStep],
) -> Dict[str, Any]:
    final = steps[-1] if steps else None
    levels = [step.levels_after for step in steps]
    best_level = max(levels or [0])
    useful_action6 = len([step for step in steps if step.useful_action6])
    useful_repositioning = len([step for step in steps if step.useful_repositioning])
    useful_states = len([step for step in steps if step.useful_new_state])
    cycles = len([step for step in steps if step.dead_end_or_cycle])
    failure_like = len([step for step in steps if step.failure_like_action6_arg])
    action6_args = [
        dict(step.action_args)
        for step in steps
        if step.policy_selected_action == "ACTION6"
    ]
    repeated_action6_args = count_repeated_args(action6_args)
    sterile_repeated_action6_args = count_sterile_repeated_action6_steps(steps)
    conditional_refresh_triggers = len(
        [
            step
            for step in steps
            if step.decision_reason
            == "conditional_action4_refresh_after_soft_stale_exhausted"
        ]
    )
    action6_after_refresh = action6_steps_immediately_after_conditional_refresh(steps)
    useful_action6_after_refresh = len(
        [step for step in action6_after_refresh if step.useful_action6]
    )
    new_action6_after_refresh = count_new_action6_affordances_after_refresh(steps)
    movement_refresh_steps = [
        step
        for step in steps
        if step.decision_reason
        == "conditional_movement_refresh_after_soft_stale_exhausted"
    ]
    action6_after_movement_refresh = (
        action6_steps_immediately_after_conditional_movement_refresh(steps)
    )
    useful_action6_after_movement_refresh = len(
        [step for step in action6_after_movement_refresh if step.useful_action6]
    )
    new_action6_after_movement_refresh = (
        count_new_action6_affordances_after_movement_refresh(steps)
    )
    progress_proxy = (
        best_level * 100.0
        + useful_states * 5.0
        + useful_action6 * 3.0
        + useful_repositioning * 2.0
        - cycles * 1.0
        - failure_like * 2.0
    )
    return {
        "condition": condition,
        "policy_steps": len(steps),
        "final_levels_completed": int(final.levels_after if final else 0),
        "best_level_reached": int(best_level),
        "final_game_state": final.game_state_after if final else "",
        "progress_proxy": round(progress_proxy, 4),
        "actions_before_stagnation_proxy": len(steps),
        "dead_end_or_cycle_steps": cycles,
        "useful_new_states": useful_states,
        "useful_action6_steps": useful_action6,
        "neutral_or_unhelpful_action6_steps": len(
            [
                step
                for step in steps
                if step.policy_selected_action == "ACTION6" and not step.useful_action6
            ]
        ),
        "action6_steps": len(
            [step for step in steps if step.policy_selected_action == "ACTION6"]
        ),
        "action4_repositioning_steps": len(
            [step for step in steps if step.repositioning_action]
        ),
        "useful_repositioning_steps": useful_repositioning,
        "failure_like_action6_args_selected": failure_like,
        "success_like_action6_args_selected": len(
            [step for step in steps if step.success_like_action6_arg]
        ),
        "selected_action6_args": [
            dict(row)
            for row in action6_args
        ],
        "selected_failure_like_action6_args": [
            dict(step.action_args)
            for step in steps
            if step.failure_like_action6_arg
        ],
        "selected_success_like_action6_args": [
            dict(step.action_args)
            for step in steps
            if step.success_like_action6_arg
        ],
        "candidate_policy_used_steps": len(
            [step for step in steps if step.candidate_policy_used]
        ),
        "unique_action6_args_selected": len({_args_key(row) for row in action6_args}),
        "repeated_action6_args_selected": repeated_action6_args,
        "sterile_repeated_action6_args_selected": sterile_repeated_action6_args,
        "conditional_refresh_triggers": conditional_refresh_triggers,
        "action6_after_conditional_refresh_steps": len(action6_after_refresh),
        "useful_action6_after_conditional_refresh_steps": useful_action6_after_refresh,
        "new_action6_affordances_after_refresh": new_action6_after_refresh,
        "conditional_movement_refresh_triggers": len(movement_refresh_steps),
        "movement_refresh_actions_selected": [
            step.policy_selected_action for step in movement_refresh_steps
        ],
        "action6_after_conditional_movement_refresh_steps": len(
            action6_after_movement_refresh
        ),
        "useful_action6_after_conditional_movement_refresh_steps": (
            useful_action6_after_movement_refresh
        ),
        "new_action6_affordances_after_movement_refresh": (
            new_action6_after_movement_refresh
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def compare_probe_conditions(
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_condition = {str(row.get("condition", "")): dict(row) for row in summaries}
    baseline = by_condition.get(BASELINE_POLICY, {})
    candidate = by_condition.get(CANDIDATE_POLICY, {})
    pair = compare_condition_pair(baseline, candidate)
    return {
        "candidate_policy_better_than_baseline_on_any_axis": bool(
            pair["improved_axes"]
        ),
        "candidate_policy_improved_axes": list(pair["improved_axes"]),
        "candidate_progress_proxy_delta": pair["progress_proxy_delta"],
        "candidate_useful_action6_delta": pair["useful_action6_delta"],
        "candidate_failure_like_selection_delta": pair[
            "failure_like_selection_delta"
        ],
        "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "candidate_policy_counted_as_confirmation": False,
    }


def compare_probe_conditions_with_ablation(
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_condition = {str(row.get("condition", "")): dict(row) for row in summaries}
    baseline = by_condition.get(BASELINE_POLICY, {})
    action4_only = by_condition.get(ACTION4_ONLY_POLICY, {})
    patch_similarity_only = by_condition.get(PATCH_SIMILARITY_ONLY_POLICY, {})
    stale_guard = by_condition.get(PATCH_SIMILARITY_STALE_GUARD_POLICY, {})
    soft_stale_guard = by_condition.get(PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY, {})
    conditional_refresh = by_condition.get(CONDITIONAL_ACTION4_REFRESH_POLICY, {})
    movement_refresh = by_condition.get(CONDITIONAL_MOVEMENT_REFRESH_POLICY, {})
    candidate = by_condition.get(CANDIDATE_POLICY, {})
    candidate_vs_baseline = compare_condition_pair(baseline, candidate)
    candidate_vs_action4_only = compare_condition_pair(action4_only, candidate)
    candidate_vs_patch_similarity_only = compare_condition_pair(
        patch_similarity_only,
        candidate,
    )
    stale_guard_vs_patch_similarity_only = compare_condition_pair(
        patch_similarity_only,
        stale_guard,
    )
    soft_stale_guard_vs_patch_similarity_only = compare_condition_pair(
        patch_similarity_only,
        soft_stale_guard,
    )
    soft_stale_guard_vs_hard_stale_guard = compare_condition_pair(
        stale_guard,
        soft_stale_guard,
    )
    conditional_refresh_vs_patch_similarity_only = compare_condition_pair(
        patch_similarity_only,
        conditional_refresh,
    )
    conditional_refresh_vs_soft_stale_guard = compare_condition_pair(
        soft_stale_guard,
        conditional_refresh,
    )
    movement_refresh_vs_patch_similarity_only = compare_condition_pair(
        patch_similarity_only,
        movement_refresh,
    )
    movement_refresh_vs_soft_stale_guard = compare_condition_pair(
        soft_stale_guard,
        movement_refresh,
    )
    movement_refresh_vs_conditional_action4_refresh = compare_condition_pair(
        conditional_refresh,
        movement_refresh,
    )
    stale_guard_vs_baseline = compare_condition_pair(baseline, stale_guard)
    soft_stale_guard_vs_baseline = compare_condition_pair(
        baseline,
        soft_stale_guard,
    )
    conditional_refresh_vs_baseline = compare_condition_pair(
        baseline,
        conditional_refresh,
    )
    movement_refresh_vs_baseline = compare_condition_pair(
        baseline,
        movement_refresh,
    )
    action4_only_vs_baseline = compare_condition_pair(baseline, action4_only)
    patch_similarity_only_vs_baseline = compare_condition_pair(
        baseline,
        patch_similarity_only,
    )
    return {
        "candidate_vs_baseline": candidate_vs_baseline,
        "candidate_vs_action4_only": candidate_vs_action4_only,
        "candidate_vs_patch_similarity_only": candidate_vs_patch_similarity_only,
        "stale_guard_vs_patch_similarity_only": stale_guard_vs_patch_similarity_only,
        "soft_stale_guard_vs_patch_similarity_only": (
            soft_stale_guard_vs_patch_similarity_only
        ),
        "soft_stale_guard_vs_hard_stale_guard": (
            soft_stale_guard_vs_hard_stale_guard
        ),
        "conditional_refresh_vs_patch_similarity_only": (
            conditional_refresh_vs_patch_similarity_only
        ),
        "conditional_refresh_vs_soft_stale_guard": (
            conditional_refresh_vs_soft_stale_guard
        ),
        "movement_refresh_vs_patch_similarity_only": (
            movement_refresh_vs_patch_similarity_only
        ),
        "movement_refresh_vs_soft_stale_guard": movement_refresh_vs_soft_stale_guard,
        "movement_refresh_vs_conditional_action4_refresh": (
            movement_refresh_vs_conditional_action4_refresh
        ),
        "stale_guard_vs_baseline": stale_guard_vs_baseline,
        "soft_stale_guard_vs_baseline": soft_stale_guard_vs_baseline,
        "conditional_refresh_vs_baseline": conditional_refresh_vs_baseline,
        "movement_refresh_vs_baseline": movement_refresh_vs_baseline,
        "action4_only_vs_baseline": action4_only_vs_baseline,
        "patch_similarity_only_vs_baseline": patch_similarity_only_vs_baseline,
        "candidate_policy_better_than_baseline_on_any_axis": bool(
            candidate_vs_baseline["improved_axes"]
        ),
        "candidate_policy_better_than_action4_only_on_any_axis": bool(
            candidate_vs_action4_only["improved_axes"]
        ),
        "candidate_policy_better_than_patch_similarity_only_on_any_axis": bool(
            candidate_vs_patch_similarity_only["improved_axes"]
        ),
        "patch_similarity_attribution_signal": bool(
            candidate_vs_action4_only["improved_axes"]
        ),
        "repositioning_context_dependency_signal": bool(
            candidate_vs_patch_similarity_only["improved_axes"]
        ),
        "stale_guard_policy_better_than_patch_similarity_only_on_any_axis": bool(
            stale_guard_vs_patch_similarity_only["improved_axes"]
        ),
        "soft_stale_guard_policy_better_than_patch_similarity_only_on_any_axis": bool(
            soft_stale_guard_vs_patch_similarity_only["improved_axes"]
        ),
        "soft_stale_guard_policy_better_than_hard_stale_guard_on_any_axis": bool(
            soft_stale_guard_vs_hard_stale_guard["improved_axes"]
        ),
        "conditional_refresh_policy_better_than_patch_similarity_only_on_any_axis": bool(
            conditional_refresh_vs_patch_similarity_only["improved_axes"]
        ),
        "conditional_refresh_policy_better_than_soft_stale_guard_on_any_axis": bool(
            conditional_refresh_vs_soft_stale_guard["improved_axes"]
        ),
        "movement_refresh_policy_better_than_patch_similarity_only_on_any_axis": bool(
            movement_refresh_vs_patch_similarity_only["improved_axes"]
        ),
        "movement_refresh_policy_better_than_soft_stale_guard_on_any_axis": bool(
            movement_refresh_vs_soft_stale_guard["improved_axes"]
        ),
        "movement_refresh_policy_better_than_action4_refresh_on_any_axis": bool(
            movement_refresh_vs_conditional_action4_refresh["improved_axes"]
        ),
        "stale_guard_reduces_repetition_signal": bool(
            int(stale_guard.get("repeated_action6_args_selected", 0) or 0)
            < int(patch_similarity_only.get("repeated_action6_args_selected", 0) or 0)
        ),
        "candidate_policy_status": "EXPERIMENTAL_POLICY_CANDIDATE_ONLY",
        "ablation_policy_status": "EXPERIMENTAL_POLICY_ABLATION_ONLY",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "candidate_policy_counted_as_confirmation": False,
        "ablation_counted_as_confirmation": False,
        "patch_similarity_only_counted_as_confirmation": False,
        "stale_guard_counted_as_confirmation": False,
        "soft_stale_guard_counted_as_confirmation": False,
        "conditional_refresh_counted_as_confirmation": False,
        "movement_refresh_counted_as_confirmation": False,
    }


def compare_condition_pair(
    reference: Mapping[str, Any],
    tested: Mapping[str, Any],
) -> Dict[str, Any]:
    improved_axes = []
    if float(tested.get("progress_proxy", 0.0) or 0.0) > float(
        reference.get("progress_proxy", 0.0) or 0.0
    ):
        improved_axes.append("progress_proxy")
    if int(tested.get("useful_action6_steps", 0) or 0) > int(
        reference.get("useful_action6_steps", 0) or 0
    ):
        improved_axes.append("useful_action6_steps")
    if int(tested.get("failure_like_action6_args_selected", 0) or 0) < int(
        reference.get("failure_like_action6_args_selected", 0) or 0
    ):
        improved_axes.append("failure_like_action6_args_selected")
    if int(tested.get("dead_end_or_cycle_steps", 0) or 0) < int(
        reference.get("dead_end_or_cycle_steps", 0) or 0
    ):
        improved_axes.append("dead_end_or_cycle_steps")
    if int(tested.get("best_level_reached", 0) or 0) > int(
        reference.get("best_level_reached", 0) or 0
    ):
        improved_axes.append("best_level_reached")
    if int(tested.get("success_like_action6_args_selected", 0) or 0) > int(
        reference.get("success_like_action6_args_selected", 0) or 0
    ):
        improved_axes.append("success_like_action6_args_selected")
    return {
        "reference_condition": str(reference.get("condition", "")),
        "tested_condition": str(tested.get("condition", "")),
        "tested_better_on_any_axis": bool(improved_axes),
        "improved_axes": improved_axes,
        "progress_proxy_delta": round(
            float(tested.get("progress_proxy", 0.0) or 0.0)
            - float(reference.get("progress_proxy", 0.0) or 0.0),
            4,
        ),
        "useful_action6_delta": int(tested.get("useful_action6_steps", 0) or 0)
        - int(reference.get("useful_action6_steps", 0) or 0),
        "failure_like_selection_delta": int(
            tested.get("failure_like_action6_args_selected", 0) or 0
        )
        - int(reference.get("failure_like_action6_args_selected", 0) or 0),
        "success_like_selection_delta": int(
            tested.get("success_like_action6_args_selected", 0) or 0
        )
        - int(reference.get("success_like_action6_args_selected", 0) or 0),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
    }


def aggregate_probe_matrix(
    budget_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    total_runs = len(budget_runs)
    candidate_vs_baseline_positive = 0
    candidate_vs_action4_positive = 0
    candidate_vs_patch_similarity_only_positive = 0
    stale_guard_vs_patch_similarity_only_positive = 0
    soft_stale_guard_vs_patch_similarity_only_positive = 0
    soft_stale_guard_vs_hard_stale_guard_positive = 0
    conditional_refresh_vs_patch_similarity_only_positive = 0
    conditional_refresh_vs_soft_stale_guard_positive = 0
    movement_refresh_vs_patch_similarity_only_positive = 0
    movement_refresh_vs_soft_stale_guard_positive = 0
    movement_refresh_vs_conditional_action4_positive = 0
    patch_similarity_only_runs = 0
    stale_guard_runs = 0
    soft_stale_guard_runs = 0
    conditional_refresh_runs = 0
    movement_refresh_runs = 0
    progress_deltas = []
    action4_progress_deltas = []
    patch_only_progress_deltas = []
    stale_guard_progress_deltas = []
    soft_stale_guard_progress_deltas = []
    conditional_refresh_progress_deltas = []
    movement_refresh_progress_deltas = []
    stale_guard_repetition_deltas = []
    soft_stale_guard_repetition_deltas = []
    soft_stale_guard_sterile_repetition_deltas = []
    conditional_refresh_repetition_deltas = []
    conditional_refresh_sterile_repetition_deltas = []
    conditional_refresh_trigger_counts = []
    conditional_refresh_new_action6_after_refresh_counts = []
    movement_refresh_trigger_counts = []
    movement_refresh_new_action6_after_refresh_counts = []
    movement_refresh_action_counts: Counter[str] = Counter()
    failure_like_deltas = []
    for run in budget_runs:
        conditions_run = set(str(row) for row in run.get("conditions_run", []) or [])
        comparison = dict(run.get("comparison", {}) or {})
        candidate_vs_baseline = dict(
            comparison.get("candidate_vs_baseline", {}) or {}
        )
        candidate_vs_action4 = dict(
            comparison.get("candidate_vs_action4_only", {}) or {}
        )
        candidate_vs_patch_only = dict(
            comparison.get("candidate_vs_patch_similarity_only", {}) or {}
        )
        stale_guard_vs_patch_only = dict(
            comparison.get("stale_guard_vs_patch_similarity_only", {}) or {}
        )
        soft_stale_guard_vs_patch_only = dict(
            comparison.get("soft_stale_guard_vs_patch_similarity_only", {}) or {}
        )
        soft_stale_guard_vs_hard = dict(
            comparison.get("soft_stale_guard_vs_hard_stale_guard", {}) or {}
        )
        conditional_refresh_vs_patch_only = dict(
            comparison.get("conditional_refresh_vs_patch_similarity_only", {}) or {}
        )
        conditional_refresh_vs_soft = dict(
            comparison.get("conditional_refresh_vs_soft_stale_guard", {}) or {}
        )
        movement_refresh_vs_patch_only = dict(
            comparison.get("movement_refresh_vs_patch_similarity_only", {}) or {}
        )
        movement_refresh_vs_soft = dict(
            comparison.get("movement_refresh_vs_soft_stale_guard", {}) or {}
        )
        movement_refresh_vs_action4_refresh = dict(
            comparison.get("movement_refresh_vs_conditional_action4_refresh", {}) or {}
        )
        stale_guard_vs_baseline = dict(
            comparison.get("stale_guard_vs_baseline", {}) or {}
        )
        soft_stale_guard_vs_baseline = dict(
            comparison.get("soft_stale_guard_vs_baseline", {}) or {}
        )
        conditional_refresh_vs_baseline = dict(
            comparison.get("conditional_refresh_vs_baseline", {}) or {}
        )
        movement_refresh_vs_baseline = dict(
            comparison.get("movement_refresh_vs_baseline", {}) or {}
        )
        action4_vs_baseline = dict(
            comparison.get("action4_only_vs_baseline", {}) or {}
        )
        patch_only_vs_baseline = dict(
            comparison.get("patch_similarity_only_vs_baseline", {}) or {}
        )
        if candidate_vs_baseline.get("tested_better_on_any_axis"):
            candidate_vs_baseline_positive += 1
        if candidate_vs_action4.get("tested_better_on_any_axis"):
            candidate_vs_action4_positive += 1
        if PATCH_SIMILARITY_ONLY_POLICY in conditions_run:
            patch_similarity_only_runs += 1
            if candidate_vs_patch_only.get("tested_better_on_any_axis"):
                candidate_vs_patch_similarity_only_positive += 1
            patch_only_progress_deltas.append(
                float(
                    patch_only_vs_baseline.get("progress_proxy_delta", 0.0) or 0.0
                )
            )
        if PATCH_SIMILARITY_STALE_GUARD_POLICY in conditions_run:
            stale_guard_runs += 1
            if stale_guard_vs_patch_only.get("tested_better_on_any_axis"):
                stale_guard_vs_patch_similarity_only_positive += 1
            stale_guard_progress_deltas.append(
                float(stale_guard_vs_baseline.get("progress_proxy_delta", 0.0) or 0.0)
            )
            by_condition = {
                str(row.get("condition", "")): dict(row)
                for row in run.get("condition_summaries", []) or []
            }
            patch_row = by_condition.get(PATCH_SIMILARITY_ONLY_POLICY, {})
            stale_row = by_condition.get(PATCH_SIMILARITY_STALE_GUARD_POLICY, {})
            stale_guard_repetition_deltas.append(
                int(stale_row.get("repeated_action6_args_selected", 0) or 0)
                - int(patch_row.get("repeated_action6_args_selected", 0) or 0)
            )
        if PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY in conditions_run:
            soft_stale_guard_runs += 1
            if soft_stale_guard_vs_patch_only.get("tested_better_on_any_axis"):
                soft_stale_guard_vs_patch_similarity_only_positive += 1
            if soft_stale_guard_vs_hard.get("tested_better_on_any_axis"):
                soft_stale_guard_vs_hard_stale_guard_positive += 1
            soft_stale_guard_progress_deltas.append(
                float(
                    soft_stale_guard_vs_baseline.get("progress_proxy_delta", 0.0)
                    or 0.0
                )
            )
            by_condition = {
                str(row.get("condition", "")): dict(row)
                for row in run.get("condition_summaries", []) or []
            }
            patch_row = by_condition.get(PATCH_SIMILARITY_ONLY_POLICY, {})
            soft_row = by_condition.get(PATCH_SIMILARITY_SOFT_STALE_GUARD_POLICY, {})
            soft_stale_guard_repetition_deltas.append(
                int(soft_row.get("repeated_action6_args_selected", 0) or 0)
                - int(patch_row.get("repeated_action6_args_selected", 0) or 0)
            )
            soft_stale_guard_sterile_repetition_deltas.append(
                int(soft_row.get("sterile_repeated_action6_args_selected", 0) or 0)
                - int(patch_row.get("sterile_repeated_action6_args_selected", 0) or 0)
            )
        if CONDITIONAL_ACTION4_REFRESH_POLICY in conditions_run:
            conditional_refresh_runs += 1
            if conditional_refresh_vs_patch_only.get("tested_better_on_any_axis"):
                conditional_refresh_vs_patch_similarity_only_positive += 1
            if conditional_refresh_vs_soft.get("tested_better_on_any_axis"):
                conditional_refresh_vs_soft_stale_guard_positive += 1
            conditional_refresh_progress_deltas.append(
                float(
                    conditional_refresh_vs_baseline.get("progress_proxy_delta", 0.0)
                    or 0.0
                )
            )
            by_condition = {
                str(row.get("condition", "")): dict(row)
                for row in run.get("condition_summaries", []) or []
            }
            patch_row = by_condition.get(PATCH_SIMILARITY_ONLY_POLICY, {})
            conditional_row = by_condition.get(CONDITIONAL_ACTION4_REFRESH_POLICY, {})
            conditional_refresh_repetition_deltas.append(
                int(conditional_row.get("repeated_action6_args_selected", 0) or 0)
                - int(patch_row.get("repeated_action6_args_selected", 0) or 0)
            )
            conditional_refresh_sterile_repetition_deltas.append(
                int(
                    conditional_row.get("sterile_repeated_action6_args_selected", 0)
                    or 0
                )
                - int(patch_row.get("sterile_repeated_action6_args_selected", 0) or 0)
            )
            conditional_refresh_trigger_counts.append(
                int(conditional_row.get("conditional_refresh_triggers", 0) or 0)
            )
            conditional_refresh_new_action6_after_refresh_counts.append(
                int(
                    conditional_row.get(
                        "new_action6_affordances_after_refresh",
                        0,
                    )
                    or 0
                )
            )
        if CONDITIONAL_MOVEMENT_REFRESH_POLICY in conditions_run:
            movement_refresh_runs += 1
            if movement_refresh_vs_patch_only.get("tested_better_on_any_axis"):
                movement_refresh_vs_patch_similarity_only_positive += 1
            if movement_refresh_vs_soft.get("tested_better_on_any_axis"):
                movement_refresh_vs_soft_stale_guard_positive += 1
            if movement_refresh_vs_action4_refresh.get("tested_better_on_any_axis"):
                movement_refresh_vs_conditional_action4_positive += 1
            movement_refresh_progress_deltas.append(
                float(
                    movement_refresh_vs_baseline.get("progress_proxy_delta", 0.0)
                    or 0.0
                )
            )
            by_condition = {
                str(row.get("condition", "")): dict(row)
                for row in run.get("condition_summaries", []) or []
            }
            movement_row = by_condition.get(CONDITIONAL_MOVEMENT_REFRESH_POLICY, {})
            movement_refresh_trigger_counts.append(
                int(movement_row.get("conditional_movement_refresh_triggers", 0) or 0)
            )
            movement_refresh_new_action6_after_refresh_counts.append(
                int(
                    movement_row.get(
                        "new_action6_affordances_after_movement_refresh",
                        0,
                    )
                    or 0
                )
            )
            movement_refresh_action_counts.update(
                [
                    str(action)
                    for action in movement_row.get(
                        "movement_refresh_actions_selected",
                        [],
                    )
                    or []
                ]
            )
        progress_deltas.append(
            float(candidate_vs_baseline.get("progress_proxy_delta", 0.0) or 0.0)
        )
        action4_progress_deltas.append(
            float(action4_vs_baseline.get("progress_proxy_delta", 0.0) or 0.0)
        )
        failure_like_deltas.append(
            int(candidate_vs_baseline.get("failure_like_selection_delta", 0) or 0)
        )
    return {
        "budget_runs": total_runs,
        "candidate_beats_baseline_runs": candidate_vs_baseline_positive,
        "candidate_beats_action4_only_runs": candidate_vs_action4_positive,
        "patch_similarity_only_runs": patch_similarity_only_runs,
        "candidate_beats_patch_similarity_only_runs": (
            candidate_vs_patch_similarity_only_positive
        ),
        "stale_guard_runs": stale_guard_runs,
        "stale_guard_beats_patch_similarity_only_runs": (
            stale_guard_vs_patch_similarity_only_positive
        ),
        "soft_stale_guard_runs": soft_stale_guard_runs,
        "soft_stale_guard_beats_patch_similarity_only_runs": (
            soft_stale_guard_vs_patch_similarity_only_positive
        ),
        "soft_stale_guard_beats_hard_stale_guard_runs": (
            soft_stale_guard_vs_hard_stale_guard_positive
        ),
        "conditional_refresh_runs": conditional_refresh_runs,
        "conditional_refresh_beats_patch_similarity_only_runs": (
            conditional_refresh_vs_patch_similarity_only_positive
        ),
        "conditional_refresh_beats_soft_stale_guard_runs": (
            conditional_refresh_vs_soft_stale_guard_positive
        ),
        "movement_refresh_runs": movement_refresh_runs,
        "movement_refresh_beats_patch_similarity_only_runs": (
            movement_refresh_vs_patch_similarity_only_positive
        ),
        "movement_refresh_beats_soft_stale_guard_runs": (
            movement_refresh_vs_soft_stale_guard_positive
        ),
        "movement_refresh_beats_conditional_action4_refresh_runs": (
            movement_refresh_vs_conditional_action4_positive
        ),
        "candidate_beats_baseline_ratio": _safe_ratio(
            candidate_vs_baseline_positive,
            total_runs,
        ),
        "candidate_beats_action4_only_ratio": _safe_ratio(
            candidate_vs_action4_positive,
            total_runs,
        ),
        "candidate_beats_patch_similarity_only_ratio": _safe_ratio(
            candidate_vs_patch_similarity_only_positive,
            patch_similarity_only_runs,
        ),
        "stale_guard_beats_patch_similarity_only_ratio": _safe_ratio(
            stale_guard_vs_patch_similarity_only_positive,
            stale_guard_runs,
        ),
        "soft_stale_guard_beats_patch_similarity_only_ratio": _safe_ratio(
            soft_stale_guard_vs_patch_similarity_only_positive,
            soft_stale_guard_runs,
        ),
        "soft_stale_guard_beats_hard_stale_guard_ratio": _safe_ratio(
            soft_stale_guard_vs_hard_stale_guard_positive,
            soft_stale_guard_runs,
        ),
        "conditional_refresh_beats_patch_similarity_only_ratio": _safe_ratio(
            conditional_refresh_vs_patch_similarity_only_positive,
            conditional_refresh_runs,
        ),
        "conditional_refresh_beats_soft_stale_guard_ratio": _safe_ratio(
            conditional_refresh_vs_soft_stale_guard_positive,
            conditional_refresh_runs,
        ),
        "movement_refresh_beats_patch_similarity_only_ratio": _safe_ratio(
            movement_refresh_vs_patch_similarity_only_positive,
            movement_refresh_runs,
        ),
        "movement_refresh_beats_soft_stale_guard_ratio": _safe_ratio(
            movement_refresh_vs_soft_stale_guard_positive,
            movement_refresh_runs,
        ),
        "movement_refresh_beats_conditional_action4_refresh_ratio": _safe_ratio(
            movement_refresh_vs_conditional_action4_positive,
            movement_refresh_runs,
        ),
        "candidate_mean_progress_delta_vs_baseline": round(
            float(np.mean(progress_deltas)) if progress_deltas else 0.0,
            4,
        ),
        "action4_only_mean_progress_delta_vs_baseline": round(
            float(np.mean(action4_progress_deltas))
            if action4_progress_deltas
            else 0.0,
            4,
        ),
        "patch_similarity_only_mean_progress_delta_vs_baseline": round(
            float(np.mean(patch_only_progress_deltas))
            if patch_only_progress_deltas
            else 0.0,
            4,
        ),
        "stale_guard_mean_progress_delta_vs_baseline": round(
            float(np.mean(stale_guard_progress_deltas))
            if stale_guard_progress_deltas
            else 0.0,
            4,
        ),
        "soft_stale_guard_mean_progress_delta_vs_baseline": round(
            float(np.mean(soft_stale_guard_progress_deltas))
            if soft_stale_guard_progress_deltas
            else 0.0,
            4,
        ),
        "conditional_refresh_mean_progress_delta_vs_baseline": round(
            float(np.mean(conditional_refresh_progress_deltas))
            if conditional_refresh_progress_deltas
            else 0.0,
            4,
        ),
        "movement_refresh_mean_progress_delta_vs_baseline": round(
            float(np.mean(movement_refresh_progress_deltas))
            if movement_refresh_progress_deltas
            else 0.0,
            4,
        ),
        "stale_guard_mean_repeated_action6_delta_vs_patch_similarity_only": round(
            float(np.mean(stale_guard_repetition_deltas))
            if stale_guard_repetition_deltas
            else 0.0,
            4,
        ),
        "soft_stale_guard_mean_repeated_action6_delta_vs_patch_similarity_only": round(
            float(np.mean(soft_stale_guard_repetition_deltas))
            if soft_stale_guard_repetition_deltas
            else 0.0,
            4,
        ),
        "soft_stale_guard_mean_sterile_repetition_delta_vs_patch_similarity_only": round(
            float(np.mean(soft_stale_guard_sterile_repetition_deltas))
            if soft_stale_guard_sterile_repetition_deltas
            else 0.0,
            4,
        ),
        "conditional_refresh_mean_repeated_action6_delta_vs_patch_similarity_only": round(
            float(np.mean(conditional_refresh_repetition_deltas))
            if conditional_refresh_repetition_deltas
            else 0.0,
            4,
        ),
        "conditional_refresh_mean_sterile_repetition_delta_vs_patch_similarity_only": round(
            float(np.mean(conditional_refresh_sterile_repetition_deltas))
            if conditional_refresh_sterile_repetition_deltas
            else 0.0,
            4,
        ),
        "conditional_refresh_total_triggers": int(
            sum(conditional_refresh_trigger_counts)
        ),
        "conditional_refresh_runs_with_triggers": len(
            [value for value in conditional_refresh_trigger_counts if value > 0]
        ),
        "conditional_refresh_mean_triggers": round(
            float(np.mean(conditional_refresh_trigger_counts))
            if conditional_refresh_trigger_counts
            else 0.0,
            4,
        ),
        "conditional_refresh_mean_new_action6_affordances_after_refresh": round(
            float(np.mean(conditional_refresh_new_action6_after_refresh_counts))
            if conditional_refresh_new_action6_after_refresh_counts
            else 0.0,
            4,
        ),
        "movement_refresh_total_triggers": int(sum(movement_refresh_trigger_counts)),
        "movement_refresh_runs_with_triggers": len(
            [value for value in movement_refresh_trigger_counts if value > 0]
        ),
        "movement_refresh_mean_triggers": round(
            float(np.mean(movement_refresh_trigger_counts))
            if movement_refresh_trigger_counts
            else 0.0,
            4,
        ),
        "movement_refresh_mean_new_action6_affordances_after_refresh": round(
            float(np.mean(movement_refresh_new_action6_after_refresh_counts))
            if movement_refresh_new_action6_after_refresh_counts
            else 0.0,
            4,
        ),
        "movement_refresh_actions_selected_counts": dict(
            sorted(movement_refresh_action_counts.items())
        ),
        "candidate_mean_failure_like_selection_delta_vs_baseline": round(
            float(np.mean(failure_like_deltas)) if failure_like_deltas else 0.0,
            4,
        ),
        "robust_candidate_policy_utility_signal": bool(
            total_runs > 0 and candidate_vs_baseline_positive == total_runs
        ),
        "patch_similarity_attribution_signal_candidate_only": bool(
            total_runs > 0 and candidate_vs_action4_positive > 0
        ),
        "repositioning_context_dependency_signal_candidate_only": bool(
            patch_similarity_only_runs > 0
            and candidate_vs_patch_similarity_only_positive
            == patch_similarity_only_runs
        ),
        "stale_guard_repetition_reduction_signal_candidate_only": bool(
            stale_guard_runs > 0
            and stale_guard_repetition_deltas
            and max(stale_guard_repetition_deltas) < 0
        ),
        "stale_guard_repetition_nonincrease_signal_candidate_only": bool(
            stale_guard_runs > 0
            and stale_guard_repetition_deltas
            and max(stale_guard_repetition_deltas) <= 0
            and min(stale_guard_repetition_deltas) < 0
        ),
        "soft_stale_guard_preserves_patch_only_progress_signal_candidate_only": bool(
            soft_stale_guard_runs > 0
            and soft_stale_guard_progress_deltas
            and min(soft_stale_guard_progress_deltas)
            >= min(patch_only_progress_deltas or [0.0])
        ),
        "soft_stale_guard_sterile_repetition_nonincrease_signal_candidate_only": bool(
            soft_stale_guard_runs > 0
            and soft_stale_guard_sterile_repetition_deltas
            and max(soft_stale_guard_sterile_repetition_deltas) <= 0
        ),
        "conditional_refresh_preserves_soft_stale_progress_signal_candidate_only": bool(
            conditional_refresh_runs > 0
            and conditional_refresh_progress_deltas
            and soft_stale_guard_progress_deltas
            and min(conditional_refresh_progress_deltas)
            >= min(soft_stale_guard_progress_deltas)
        ),
        "conditional_refresh_triggered_in_any_run_candidate_only": bool(
            conditional_refresh_trigger_counts
            and max(conditional_refresh_trigger_counts) > 0
        ),
        "movement_refresh_preserves_soft_stale_progress_signal_candidate_only": bool(
            movement_refresh_runs > 0
            and movement_refresh_progress_deltas
            and soft_stale_guard_progress_deltas
            and min(movement_refresh_progress_deltas)
            >= min(soft_stale_guard_progress_deltas)
        ),
        "movement_refresh_triggered_in_any_run_candidate_only": bool(
            movement_refresh_trigger_counts
            and max(movement_refresh_trigger_counts) > 0
        ),
        "policy_probe_result_is_not_scientific_verdict": True,
        "candidate_policy_counted_as_confirmation": False,
        "ablation_counted_as_confirmation": False,
        "patch_similarity_only_counted_as_confirmation": False,
        "stale_guard_counted_as_confirmation": False,
        "soft_stale_guard_counted_as_confirmation": False,
        "conditional_refresh_counted_as_confirmation": False,
        "movement_refresh_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def classify_action6_args(
    action_args: Mapping[str, Any],
    memory: CandidatePolicyMemory,
) -> str:
    key = _args_key(action_args)
    if key in {_args_key(row) for row in memory.known_failed_args}:
        return "known_failure"
    if key in {_args_key(row) for row in memory.outside_boundary_args}:
        return "outside_boundary"
    if key in {_args_key(row) for row in memory.alternate_context_success_args}:
        return "alternate_context_success"
    if key in {_args_key(row) for row in memory.known_success_args}:
        return "known_success"
    return "unknown_action6_args"


def min_patch_distance(
    action_args: Mapping[str, Any],
    seed_args: Sequence[Mapping[str, Any]],
    grid: Any,
) -> float:
    if not seed_args:
        return 0.0
    candidate_patch = local_patch_signature(grid, action_args)
    return min(
        patch_distance(candidate_patch, local_patch_signature(grid, seed))
        for seed in seed_args
    )


def local_patch_signature(
    grid: Any,
    action_args: Mapping[str, Any],
    *,
    radius: int = 1,
) -> list[list[int]]:
    array = np.asarray(grid, dtype=np.int32)
    x = _safe_int(action_args.get("x"))
    y = _safe_int(action_args.get("y"))
    if x is None or y is None or array.ndim != 2:
        return []
    y0 = max(0, y - radius)
    y1 = min(array.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(array.shape[1], x + radius + 1)
    return array[y0:y1, x0:x1].tolist()


def patch_distance(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> float:
    left_rows = [list(row) for row in left]
    right_rows = [list(row) for row in right]
    max_rows = max(len(left_rows), len(right_rows))
    max_cols = max(
        [len(row) for row in left_rows] + [len(row) for row in right_rows] + [0]
    )
    distance = 0
    for y in range(max_rows):
        for x in range(max_cols):
            l_val = left_rows[y][x] if y < len(left_rows) and x < len(left_rows[y]) else None
            r_val = right_rows[y][x] if y < len(right_rows) and x < len(right_rows[y]) else None
            if l_val != r_val:
                distance += 1
    return float(distance)


def error_step(
    step_index: int,
    *,
    condition: str,
    decision: ProbeDecision,
    before: Any,
    before_signature: str,
    error: str,
) -> ProbeStep:
    return ProbeStep(
        step=step_index,
        condition=condition,
        policy_selected_action=decision.action_name,
        action_args=dict(decision.action_args),
        decision_reason=decision.decision_reason,
        candidate_policy_used=decision.candidate_policy_used,
        candidate_score=decision.candidate_score,
        candidate_score_details=decision.candidate_score_details,
        action6_arg_class="",
        failure_like_action6_arg=False,
        success_like_action6_arg=False,
        repositioning_action=False,
        local_patch_signal=0.0,
        object_positions_signal=0.0,
        changed_pixels=0.0,
        contact_graph_signal=0.0,
        useful_action6=False,
        useful_repositioning=False,
        useful_new_state=False,
        dead_end_or_cycle=False,
        state_signature_before=before_signature,
        state_signature_after=before_signature,
        levels_before=int(before.levels_completed),
        levels_after=int(before.levels_completed),
        game_state_before=str(before.game_state),
        game_state_after=str(before.game_state),
        measurements={},
        env_actions=0,
        error=error,
    )


def state_signature(grid: Any, levels_completed: int, game_state: str) -> str:
    array = np.asarray(grid, dtype=np.int32)
    digest = hashlib.sha1(array.tobytes()).hexdigest()[:16]
    return f"{tuple(array.shape)}:{digest}:{int(levels_completed)}:{game_state}"


def dedupe_args(values: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    by_key = {_args_key(dict(value)): dict(value) for value in values if value}
    return tuple(by_key[key] for key in sorted(by_key))


def count_repeated_args(values: Sequence[Mapping[str, Any]]) -> int:
    seen: set[str] = set()
    repeats = 0
    for value in values:
        key = _args_key(value)
        if key in seen:
            repeats += 1
        else:
            seen.add(key)
    return repeats


def count_sterile_repeated_action6_steps(steps: Sequence[ProbeStep]) -> int:
    seen: set[str] = set()
    sterile = 0
    for step in steps:
        if step.policy_selected_action != "ACTION6":
            continue
        key = _args_key(step.action_args)
        if key in seen and not action6_step_has_new_effect(step):
            sterile += 1
        seen.add(key)
    return sterile


def action6_steps_immediately_after_conditional_refresh(
    steps: Sequence[ProbeStep],
) -> Tuple[ProbeStep, ...]:
    rows = []
    for index, step in enumerate(steps[1:], start=1):
        previous = steps[index - 1]
        if (
            previous.decision_reason
            == "conditional_action4_refresh_after_soft_stale_exhausted"
            and step.policy_selected_action == "ACTION6"
        ):
            rows.append(step)
    return tuple(rows)


def count_new_action6_affordances_after_refresh(steps: Sequence[ProbeStep]) -> int:
    seen_before: set[str] = set()
    new_after_refresh = 0
    refresh_pending = False
    for step in steps:
        if step.policy_selected_action == "ACTION6":
            key = _args_key(step.action_args)
            if refresh_pending and step.useful_action6 and key not in seen_before:
                new_after_refresh += 1
            seen_before.add(key)
            refresh_pending = False
            continue
        refresh_pending = (
            step.decision_reason
            == "conditional_action4_refresh_after_soft_stale_exhausted"
        )
    return new_after_refresh


def action6_steps_immediately_after_conditional_movement_refresh(
    steps: Sequence[ProbeStep],
) -> Tuple[ProbeStep, ...]:
    rows = []
    for index, step in enumerate(steps[1:], start=1):
        previous = steps[index - 1]
        if (
            previous.decision_reason
            == "conditional_movement_refresh_after_soft_stale_exhausted"
            and step.policy_selected_action == "ACTION6"
        ):
            rows.append(step)
    return tuple(rows)


def count_new_action6_affordances_after_movement_refresh(
    steps: Sequence[ProbeStep],
) -> int:
    seen_before: set[str] = set()
    new_after_refresh = 0
    refresh_pending = False
    for step in steps:
        if step.policy_selected_action == "ACTION6":
            key = _args_key(step.action_args)
            if refresh_pending and step.useful_action6 and key not in seen_before:
                new_after_refresh += 1
            seen_before.add(key)
            refresh_pending = False
            continue
        refresh_pending = (
            step.decision_reason
            == "conditional_movement_refresh_after_soft_stale_exhausted"
        )
    return new_after_refresh


def action6_effect_stats(steps: Sequence[ProbeStep]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        if step.policy_selected_action != "ACTION6":
            continue
        key = _args_key(step.action_args)
        row = stats.setdefault(key, default_action6_effect_stats())
        useful_effect = action6_step_has_new_effect(step)
        row["arg"] = dict(step.action_args)
        row["times_selected"] = int(row["times_selected"]) + 1
        row["new_state_count"] = int(row["new_state_count"]) + int(
            bool(step.useful_new_state)
        )
        row["last_effect_signature"] = action6_effect_signature(step)
        row["last_changed_pixels"] = float(step.changed_pixels)
        row["last_local_patch_signal"] = float(step.local_patch_signal)
        row["last_object_positions_signal"] = float(step.object_positions_signal)
        row["last_effect_useful"] = bool(useful_effect)
        if useful_effect:
            row["useful_effect_count"] = int(row["useful_effect_count"]) + 1
            row["consecutive_no_new_effects"] = 0
        else:
            row["consecutive_no_new_effects"] = int(
                row["consecutive_no_new_effects"]
            ) + 1
    return stats


def default_action6_effect_stats() -> Dict[str, Any]:
    return {
        "arg": {},
        "times_selected": 0,
        "new_state_count": 0,
        "useful_effect_count": 0,
        "consecutive_no_new_effects": 0,
        "last_effect_signature": "",
        "last_changed_pixels": 0.0,
        "last_local_patch_signal": 0.0,
        "last_object_positions_signal": 0.0,
        "last_effect_useful": False,
    }


def action6_step_has_new_effect(step: ProbeStep) -> bool:
    return bool(
        step.useful_new_state
        or float(step.local_patch_signal) > 0.0
        or float(step.object_positions_signal) > 0.0
        or step.state_signature_after != step.state_signature_before
    )


def action6_effect_signature(step: ProbeStep) -> str:
    return "|".join(
        [
            str(step.state_signature_after),
            f"local={float(step.local_patch_signal):.4f}",
            f"objects={float(step.object_positions_signal):.4f}",
            f"changed={float(step.changed_pixels):.4f}",
        ]
    )


def deterministic_action_order(tie_break_seed: int = 0) -> list[str]:
    order = ["ACTION3", "ACTION4", "ACTION6", "ACTION1", "ACTION2", "ACTION5", "ACTION7"]
    if not order:
        return []
    offset = int(tie_break_seed) % len(order)
    return order[offset:] + order[:offset]


def deterministic_args_tie_break(
    args: Mapping[str, Any],
    tie_break_seed: int = 0,
) -> str:
    key = _args_key(args)
    if int(tie_break_seed) == 0:
        return key
    return hashlib.sha1(f"{int(tie_break_seed)}:{key}".encode("utf-8")).hexdigest()


def select_deterministic_action6_candidate(
    actions: Sequence[Any],
    *,
    tie_break_seed: int = 0,
) -> Any:
    rows = list(actions)
    if not rows:
        return None
    if int(tie_break_seed) == 0:
        return rows[0]
    return min(
        rows,
        key=lambda action: deterministic_args_tie_break(
            dict(getattr(action, "action_args", {}) or {}),
            tie_break_seed,
        ),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _args_key(args: Mapping[str, Any]) -> str:
    return json.dumps({str(key): args[key] for key in sorted(args)}, sort_keys=True)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_bp35_sage_candidate_policy_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P1 bp35 closed-loop SAGE candidate-policy probe.",
    )
    parser.add_argument(
        "--scope-consolidation",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_SCOPE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=None,
        help="Run matrix mode with these budgets when --matrix is set.",
    )
    parser.add_argument(
        "--tie-break-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Deterministic tie-break variants for --matrix mode.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run P1.2/P1.3 multi-budget ablation matrix.",
    )
    parser.add_argument(
        "--include-patch-similarity-only",
        action="store_true",
        help="Include P1.5 patch-similarity-only no-repositioning ablation.",
    )
    parser.add_argument(
        "--include-stale-guard",
        action="store_true",
        help="Include P1.6 patch-similarity stale/repetition guard ablation.",
    )
    parser.add_argument(
        "--include-soft-stale-guard",
        action="store_true",
        help="Include P1.7 effect-aware soft stale guard ablation.",
    )
    parser.add_argument(
        "--include-conditional-refresh",
        action="store_true",
        help="Include P1.8 conditional ACTION4 refresh ablation.",
    )
    parser.add_argument(
        "--include-movement-refresh",
        action="store_true",
        help="Include P1.10 conditional movement refresh selector ablation.",
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = args.out
    if args.matrix:
        if output_path == DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_PROBE_OUTPUT_PATH:
            output_path = (
                DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MOVEMENT_REFRESH_OUTPUT_PATH
                if args.include_movement_refresh
                else DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_CONDITIONAL_REFRESH_OUTPUT_PATH
                if args.include_conditional_refresh
                else DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_SOFT_STALE_GUARD_OUTPUT_PATH
                if args.include_soft_stale_guard
                else DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_STALE_GUARD_OUTPUT_PATH
                if args.include_stale_guard
                else (
                    DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_ABLATION_OUTPUT_PATH
                    if args.include_patch_similarity_only
                    else DEFAULT_P1_BP35_SAGE_CANDIDATE_POLICY_MATRIX_OUTPUT_PATH
                )
            )
        payload = run_bp35_sage_candidate_policy_probe_matrix(
            scope_consolidation_path=args.scope_consolidation,
            environments_dir=args.environments_dir,
            budgets=tuple(args.budgets or DEFAULT_BUDGETS),
            tie_break_seeds=tuple(args.tie_break_seeds or DEFAULT_TIE_BREAK_SEEDS),
            game_id=args.game_id,
            include_patch_similarity_only=bool(args.include_patch_similarity_only),
            include_stale_guard=bool(args.include_stale_guard),
            include_soft_stale_guard=bool(args.include_soft_stale_guard),
            include_conditional_refresh=bool(args.include_conditional_refresh),
            include_movement_refresh=bool(args.include_movement_refresh),
        )
    else:
        payload = run_bp35_sage_candidate_policy_probe(
            scope_consolidation_path=args.scope_consolidation,
            environments_dir=args.environments_dir,
            budget=args.budget,
            game_id=args.game_id,
        )
    write_bp35_sage_candidate_policy_probe(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "summary": payload["summary"],
                "comparison": payload.get("comparison", payload.get("aggregate", {})),
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
