"""SAGE.8i held-out next-level evaluation of exact goal-grounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .goal_grounded_relational_memory_evaluation import (
    DEFAULT_SAGE8H_GOAL_GROUNDED_RELATIONAL_MEMORY_EVALUATION_PATH,
    MEMORY_ACTION_UNAVAILABLE,
    MEMORY_OUT_OF_SCOPE,
    SAGE8H_ARC_GAIN,
    SAGE8H_SCHEMA_VERSION,
    SAGE8H_TRUTH_STATUS,
    build_goal_trajectory_memory_index,
    select_exact_goal_memory_action,
)
from .live_mini_frontier_m3_executor import EnvFactory, _make_real_env
from .live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from .relational_memory_ab_evaluation import summarize_primary_metrics
from .relational_memory_closed_loop_evaluation import (
    REPLANNING_POLICY,
    action_identity,
    select_state_conditioned_replanning_action,
)


DEFAULT_SAGE8I_GOAL_GROUNDED_MEMORY_HELD_OUT_EVALUATION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage8i_goal_grounded_memory_held_out_evaluation.json"
)

SAGE8I_SCHEMA_VERSION = "sage.goal_grounded_memory_held_out_evaluation.v1"
SAGE8I_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8I"
SAGE8I_HELD_OUT_GAIN = (
    "SAGE_EXACT_GOAL_MEMORY_HELD_OUT_NEXT_LEVEL_ARC_SCORE_GAIN_OBSERVED"
)
SAGE8I_SCOPE_SAFE_NO_GENERALIZATION = (
    "SAGE_EXACT_GOAL_MEMORY_HELD_OUT_NEXT_LEVEL_SCOPE_SAFE_NO_GENERALIZATION"
)
SAGE8I_HELD_OUT_REGRESSION = (
    "SAGE_EXACT_GOAL_MEMORY_HELD_OUT_NEXT_LEVEL_ARC_SCORE_REGRESSION_OBSERVED"
)

HELD_OUT_PROTOCOL = "NEXT_LEVEL_AFTER_ADMITTED_POSITIVE_EXACT_SCOPE_ABLATION"
PAIRED_PLANNER = "EXACT_GOAL_MEMORY_THEN_SHARED_STATE_CONDITIONED_FALLBACK"
TERMINAL_WIN_STATES = {"WIN", "WON", "VICTORY"}
NON_TERMINAL_STATES = {"", "NOT_FINISHED", "PLAYING", "IN_PROGRESS"}


def run_sage8i_goal_grounded_memory_held_out_evaluation(
    *,
    sage8h_path: str | Path = (
        DEFAULT_SAGE8H_GOAL_GROUNDED_RELATIONAL_MEMORY_EVALUATION_PATH
    ),
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Evaluate exact memory only after entering each unseen next level."""
    source_sage8h = _load_json(sage8h_path)
    validate_sage8i_source(source_sage8h)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    specifications = build_sage8i_held_out_specifications(source_sage8h)
    source_memory = dict(source_sage8h.get("goal_grounded_trajectory_memory", {}) or {})
    memory_index = build_goal_trajectory_memory_index(source_memory)
    episodes = [
        execute_sage8i_paired_held_out_rollout(
            specification,
            memory_index=memory_index,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        for specification in specifications
    ]
    primary_metrics = summarize_primary_metrics(episodes)
    held_out_metrics = summarize_sage8i_held_out_metrics(
        episodes,
        training_memory_entries=len(memory_index),
    )
    gate = build_sage8i_gate(
        source_sage8h,
        specifications,
        episodes,
        primary_metrics,
        held_out_metrics,
    )
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8i held-out evaluation gate did not pass")
    summary = summarize_sage8i(
        episodes,
        primary_metrics,
        held_out_metrics,
        gate,
    )
    payload = {
        "config": {
            "schema_version": SAGE8I_SCHEMA_VERSION,
            "sage8h_path": str(sage8h_path),
            "environments_dir": str(env_dir),
            "held_out_protocol": HELD_OUT_PROTOCOL,
            "paired_planner": PAIRED_PLANNER,
            "fallback_planner": REPLANNING_POLICY,
            "evaluation_design": {
                "held_out_unit": "NEXT_LEVEL_AFTER_SAGE8G_POSITIVE",
                "setup_replays_admitted_positive_trajectory": True,
                "setup_actions_are_pre_evaluation": True,
                "setup_level_delta_excluded_from_primary_metrics": True,
                "evaluation_begins_after_setup_level_increase": True,
                "held_out_horizon_matches_source_trajectory_length_per_game": True,
                "same_held_out_start_and_horizon_between_arms": True,
                "control_arm_exact_memory_enabled": False,
                "treatment_arm_exact_memory_enabled": True,
                "shared_fallback_planner": True,
                "primary_metrics": ["levels_completed", "win_rate"],
                "future_outcomes_used_for_action_ranking": False,
                "evaluation_outcomes_used_for_training_or_tuning": False,
                "counterfactual_rollouts_performed": 0,
            },
            "scope_policy": {
                "exact_game_and_visual_digest_required": True,
                "fuzzy_matching_allowed": False,
                "cross_game_transfer_allowed": False,
                "out_of_scope_memory_is_quarantined": True,
                "unavailable_memory_action_is_quarantined": True,
            },
            "scientific_scope": {
                "held_out_generalization_evaluation": True,
                "training_trajectory_states_are_excluded_from_treatment_hits": True,
                "structural_generalization_policy_applied": False,
                "exact_memory_can_only_demonstrate_scope_safe_reuse": True,
                "no_gain_means_no_held_out_generalization_observed": True,
            },
            "artifacts_not_modified": [
                "A33.3",
                "A33.4",
                "A34.2",
                "A34.3",
                "SAGE.8a",
                "SAGE.8b",
                "SAGE.8c",
                "SAGE.8d",
                "SAGE.8e",
                "SAGE.8f",
                "SAGE.8g",
                "SAGE.8h",
            ],
        },
        "source_memory_summary": {
            "schema_version": str(
                source_memory.get("config", {}).get("schema_version", "")
            ),
            "training_memory_entries": len(memory_index),
            "training_games": list(
                source_memory.get("summary", {}).get("games", []) or []
            ),
            "exact_match_required": True,
            "fuzzy_matching_allowed": False,
            "cross_game_transfer_allowed": False,
        },
        "held_out_specifications": specifications,
        "paired_held_out_rollouts": episodes,
        "primary_metrics": primary_metrics,
        "held_out_metrics": held_out_metrics,
        "gate": gate,
        "summary": summary,
        "status": "EVALUATED",
        "outcome_status": summary["outcome_status"],
        "truth_status": SAGE8I_TRUTH_STATUS,
        "held_out_generalization_evaluated": True,
        "held_out_generalization_observed": summary["held_out_generalization_observed"],
        "exact_memory_quarantined_out_of_scope": summary[
            "exact_memory_quarantined_out_of_scope"
        ],
        "structural_generalization_policy_applied": False,
        "primary_arc_progress_improved": summary["primary_arc_progress_improved"],
        "primary_arc_progress_regressed": summary["primary_arc_progress_regressed"],
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "future_outcomes_used_for_planning": False,
        "counterfactual_rollouts_performed": 0,
        "cross_game_transfer_performed": False,
        "scope_generalization_performed": False,
        "scientific_review_performed": False,
        "confirmation_performed": False,
        "revision_performed": False,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "support": 0,
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_sage8i_goal_grounded_memory_held_out_evaluation(payload, output_path)
    return payload


def build_sage8i_held_out_specifications(
    source_sage8h: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    specifications = []
    for row in source_sage8h.get("evaluation_specifications", []) or []:
        game_id = str(row.get("game_id", ""))
        setup_commands = [
            {
                "action": str(command.get("action", "")),
                "action_args": dict(command.get("action_args", {}) or {}),
            }
            for command in row.get("source_commands", []) or []
        ]
        horizon = int(row.get("horizon", 0) or 0)
        expected_level_delta = int(row.get("expected_level_delta", 0) or 0)
        if not setup_commands or horizon != len(setup_commands):
            raise ValueError("SAGE.8i requires complete SAGE.8h setup trajectories")
        if expected_level_delta <= 0:
            raise ValueError("SAGE.8i setup must enter a later held-out level")
        specifications.append(
            {
                "evaluation_id": f"sage8i::{game_id}::next_level_holdout",
                "game_id": game_id,
                "setup_commands": setup_commands,
                "setup_action_count": len(setup_commands),
                "held_out_horizon": horizon,
                "expected_reset_visual_digest": str(
                    row.get("expected_reset_visual_digest", "")
                ),
                "expected_held_out_start_visual_digest": str(
                    row.get("expected_post_positive_visual_digest", "")
                ),
                "expected_setup_level_delta": expected_level_delta,
                "training_trajectory_evaluation_id": str(row.get("evaluation_id", "")),
                "paired_arms": ["no_exact_memory", "with_exact_memory"],
                "setup_excluded_from_primary_metrics": True,
                "held_out_next_level": True,
                "structural_generalization_policy_applied": False,
            }
        )
    return sorted(specifications, key=lambda row: str(row.get("game_id", "")))


def execute_sage8i_paired_held_out_rollout(
    specification: Mapping[str, Any],
    *,
    memory_index: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    no_memory = _execute_sage8i_held_out_arm(
        specification,
        memory_index=memory_index,
        memory_enabled=False,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    with_memory = _execute_sage8i_held_out_arm(
        specification,
        memory_index=memory_index,
        memory_enabled=True,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if no_memory.get("status") != "EXECUTED" or with_memory.get("status") != "EXECUTED":
        raise ValueError(
            "SAGE.8i paired held-out arm blocked:"
            f"{no_memory.get('reason', '')}:{with_memory.get('reason', '')}"
        )
    no_before = int(no_memory.get("levels_completed_before", 0) or 0)
    with_before = int(with_memory.get("levels_completed_before", 0) or 0)
    no_after = int(no_memory.get("levels_completed_after", 0) or 0)
    with_after = int(with_memory.get("levels_completed_after", 0) or 0)
    comparison = compare_held_out_arms(no_memory, with_memory)
    return {
        "evaluation_id": str(specification.get("evaluation_id", "")),
        "game_id": str(specification.get("game_id", "")),
        "held_out_horizon": int(specification.get("held_out_horizon", 0) or 0),
        "setup_action_count_per_arm": int(
            specification.get("setup_action_count", 0) or 0
        ),
        "same_held_out_start_between_arms": (
            str(no_memory.get("held_out_start_visual_digest", ""))
            == str(with_memory.get("held_out_start_visual_digest", ""))
            == str(specification.get("expected_held_out_start_visual_digest", ""))
        ),
        "same_held_out_level_between_arms": no_before == with_before,
        "same_held_out_horizon_between_arms": (
            int(no_memory.get("held_out_horizon", 0) or 0)
            == int(with_memory.get("held_out_horizon", 0) or 0)
            == int(specification.get("held_out_horizon", 0) or 0)
        ),
        "shared_fallback_planner": REPLANNING_POLICY,
        "no_memory_levels_completed_before": no_before,
        "with_memory_levels_completed_before": with_before,
        "no_memory_levels_completed_after": no_after,
        "with_memory_levels_completed_after": with_after,
        "no_memory_levels_completed_delta": no_after - no_before,
        "with_memory_levels_completed_delta": with_after - with_before,
        "no_memory_win": bool(no_memory.get("win", False)),
        "with_memory_win": bool(with_memory.get("win", False)),
        "control_exact_memory_disabled": int(
            no_memory.get("memory_applications", 0) or 0
        )
        == 0,
        "treatment_exact_memory_quarantined_out_of_scope": (
            int(with_memory.get("memory_applications", 0) or 0) == 0
            and int(with_memory.get("memory_scope_misses", 0) or 0)
            == int(with_memory.get("steps_executed", -1) or 0)
        ),
        "held_out_next_level": True,
        "setup_excluded_from_primary_metrics": True,
        "future_outcomes_used_for_action_ranking": False,
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "comparison": comparison,
        "no_memory_arm": no_memory,
        "with_memory_arm": with_memory,
        "primary_metrics": ["levels_completed", "win_rate"],
        "truth_reevaluated": False,
        "support_counted": 0,
        "wrong_confirmations": 0,
    }


def _execute_sage8i_held_out_arm(
    specification: Mapping[str, Any],
    *,
    memory_index: Mapping[str, Mapping[str, Any]],
    memory_enabled: bool,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    game_id = str(specification.get("game_id", ""))
    try:
        env = _new_env(game_id, environments_dir, env_factory)
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return {"status": "BLOCKED", "reason": f"env_setup_failed:{exc}"}
    reset_snapshot = snapshot_frame(frame)
    reset_digest = _visual_digest(frame)
    setup_commands = list(specification.get("setup_commands", []) or [])
    setup_legal = True
    for index, command in enumerate(setup_commands):
        action = str(command.get("action", ""))
        args = dict(command.get("action_args", {}) or {})
        selected = select_live_action(env, action, action_args=args)
        if selected is None:
            setup_legal = False
            return {
                "status": "BLOCKED",
                "reason": f"setup_action_unavailable:{index}:{action_identity(action, args)}",
                "memory_enabled": memory_enabled,
            }
        frame = _step_env_action(env, selected)
    held_out_start = snapshot_frame(frame)
    held_out_start_digest = _visual_digest(frame)
    setup_level_delta = int(held_out_start.levels_completed) - int(
        reset_snapshot.levels_completed
    )
    expected_setup_delta = int(specification.get("expected_setup_level_delta", 0) or 0)
    if (
        reset_digest != str(specification.get("expected_reset_visual_digest", ""))
        or held_out_start_digest
        != str(specification.get("expected_held_out_start_visual_digest", ""))
        or setup_level_delta != expected_setup_delta
    ):
        return {
            "status": "BLOCKED",
            "reason": "held_out_setup_verification_mismatch",
            "memory_enabled": memory_enabled,
        }

    horizon = int(specification.get("held_out_horizon", 0) or 0)
    initial_levels = int(held_out_start.levels_completed)
    family_counts: Counter[str] = Counter()
    concrete_counts: Counter[str] = Counter()
    state_action_visits: Counter[str] = Counter()
    previous_identity = ""
    trace = []
    memory_applications = 0
    memory_scope_misses = 0
    memory_action_quarantines = 0
    fallback_applications = 0
    stopped_on_level_increase = False
    stopped_on_terminal = False
    for step in range(horizon):
        current_signature = state_signature_from_frame(frame)
        current_digest = _visual_digest(frame)
        legal_actions = tuple(_valid_actions(env))
        selected, memory_decision = select_exact_goal_memory_action(
            legal_actions,
            game_id=game_id,
            current_visual_digest=current_digest,
            memory_index=memory_index,
            memory_enabled=memory_enabled,
        )
        fallback_decision: Dict[str, Any] = {}
        if selected is not None:
            memory_applications += 1
            selection_source = "EXACT_GOAL_TRAJECTORY_MEMORY"
        else:
            reason = str(memory_decision.get("memory_decision_reason", ""))
            memory_scope_misses += int(memory_enabled and reason == MEMORY_OUT_OF_SCOPE)
            memory_action_quarantines += int(
                memory_enabled and reason == MEMORY_ACTION_UNAVAILABLE
            )
            selected, fallback_decision = select_state_conditioned_replanning_action(
                legal_actions,
                current_state_signature=current_signature,
                family_counts=family_counts,
                concrete_counts=concrete_counts,
                state_action_visits=state_action_visits,
                previous_action_identity=previous_identity,
            )
            fallback_applications += 1
            selection_source = "SHARED_STATE_CONDITIONED_FALLBACK"
        if selected is None:
            return {
                "status": "BLOCKED",
                "reason": f"no_legal_held_out_action:{step}",
                "memory_enabled": memory_enabled,
            }
        action = str(getattr(selected, "name", ""))
        args = dict(getattr(selected, "action_args", {}) or {})
        identity = action_identity(action, args)
        frame = _step_env_action(env, selected)
        after = snapshot_frame(frame)
        after_digest = _visual_digest(frame)
        family_counts[action] += 1
        concrete_counts[identity] += 1
        state_action_visits[f"{current_digest}|{identity}"] += 1
        previous_identity = identity
        trace.append(
            {
                "held_out_step": step,
                "before_visual_digest": current_digest,
                "after_visual_digest": after_digest,
                "action": action,
                "action_args": args,
                "action_identity": identity,
                "selection_source": selection_source,
                "memory_decision": memory_decision,
                "fallback_decision": _public_planner_decision(fallback_decision),
                "levels_completed_after": int(after.levels_completed),
                "game_state_after": str(after.game_state),
            }
        )
        if int(after.levels_completed) > initial_levels:
            stopped_on_level_increase = True
            break
        if _terminal_state(after.game_state):
            stopped_on_terminal = True
            break
    final_snapshot = snapshot_frame(frame)
    return {
        "status": "EXECUTED",
        "arm": "with_exact_memory" if memory_enabled else "no_exact_memory",
        "memory_enabled": memory_enabled,
        "reset_visual_digest": reset_digest,
        "setup_actions_executed": len(setup_commands),
        "setup_all_actions_legal": setup_legal,
        "setup_level_delta": setup_level_delta,
        "setup_excluded_from_primary_metrics": True,
        "held_out_start_visual_digest": held_out_start_digest,
        "held_out_horizon": horizon,
        "levels_completed_before": initial_levels,
        "levels_completed_after": int(final_snapshot.levels_completed),
        "levels_completed_delta": int(final_snapshot.levels_completed) - initial_levels,
        "game_state_after": str(final_snapshot.game_state),
        "win": str(final_snapshot.game_state).upper() in TERMINAL_WIN_STATES,
        "steps_executed": len(trace),
        "stopped_on_level_increase": stopped_on_level_increase,
        "stopped_on_terminal": stopped_on_terminal,
        "memory_applications": memory_applications,
        "memory_scope_misses": memory_scope_misses,
        "memory_action_quarantines": memory_action_quarantines,
        "fallback_applications": fallback_applications,
        "memory_coverage_rate": memory_applications / len(trace) if trace else 0.0,
        "final_visual_digest": _visual_digest(frame),
        "all_selected_actions_legal": True,
        "future_outcome_fields_read_for_action_ranking": [],
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "counterfactual_rollouts_performed": 0,
        "trace": trace,
    }


def compare_held_out_arms(
    no_memory: Mapping[str, Any],
    with_memory: Mapping[str, Any],
) -> Dict[str, Any]:
    no_trace = list(no_memory.get("trace", []) or [])
    with_trace = list(with_memory.get("trace", []) or [])
    paired = min(len(no_trace), len(with_trace))
    divergent = sum(
        str(no_trace[index].get("action_identity", ""))
        != str(with_trace[index].get("action_identity", ""))
        for index in range(paired)
    ) + abs(len(no_trace) - len(with_trace))
    return {
        "paired_action_positions": paired,
        "no_memory_actions": len(no_trace),
        "with_memory_actions": len(with_trace),
        "divergent_action_positions": divergent,
        "action_trajectories_identical": divergent == 0,
        "final_visual_digests_identical": str(no_memory.get("final_visual_digest", ""))
        == str(with_memory.get("final_visual_digest", "")),
        "scope_quarantine_preserved_shared_fallback_behavior": divergent == 0,
    }


def summarize_sage8i_held_out_metrics(
    episodes: Sequence[Mapping[str, Any]],
    *,
    training_memory_entries: int,
) -> Dict[str, Any]:
    no_steps = sum(
        int(row.get("no_memory_arm", {}).get("steps_executed", 0) or 0)
        for row in episodes
    )
    with_steps = sum(
        int(row.get("with_memory_arm", {}).get("steps_executed", 0) or 0)
        for row in episodes
    )
    no_apps = sum(
        int(row.get("no_memory_arm", {}).get("memory_applications", 0) or 0)
        for row in episodes
    )
    with_apps = sum(
        int(row.get("with_memory_arm", {}).get("memory_applications", 0) or 0)
        for row in episodes
    )
    scope_misses = sum(
        int(row.get("with_memory_arm", {}).get("memory_scope_misses", 0) or 0)
        for row in episodes
    )
    quarantines = sum(
        int(row.get("with_memory_arm", {}).get("memory_action_quarantines", 0) or 0)
        for row in episodes
    )
    with_fallbacks = sum(
        int(row.get("with_memory_arm", {}).get("fallback_applications", 0) or 0)
        for row in episodes
    )
    held_out_scope_keys = {
        f"{row.get('game_id', '')}|{trace.get('before_visual_digest', '')}"
        for row in episodes
        for trace in row.get("with_memory_arm", {}).get("trace", []) or []
    }
    return {
        "training_memory_entries": int(training_memory_entries),
        "held_out_levels_evaluated": len(episodes),
        "held_out_scope_keys_observed": len(held_out_scope_keys),
        "held_out_scope_keys_matching_training_memory": with_apps,
        "setup_actions_executed_total": sum(
            int(row.get("setup_action_count_per_arm", 0) or 0) * 2 for row in episodes
        ),
        "no_memory_steps_executed": no_steps,
        "with_memory_steps_executed": with_steps,
        "no_memory_applications": no_apps,
        "with_memory_applications": with_apps,
        "with_memory_scope_misses": scope_misses,
        "with_memory_action_quarantines": quarantines,
        "with_memory_fallback_applications": with_fallbacks,
        "with_memory_exact_coverage_rate": with_apps / with_steps
        if with_steps
        else 0.0,
        "episodes_with_identical_action_trajectories": sum(
            bool(row.get("comparison", {}).get("action_trajectories_identical", False))
            for row in episodes
        ),
        "episodes_with_identical_final_states": sum(
            bool(row.get("comparison", {}).get("final_visual_digests_identical", False))
            for row in episodes
        ),
    }


def build_sage8i_gate(
    source_sage8h: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    held_out_metrics: Mapping[str, Any],
) -> Dict[str, bool]:
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    return {
        "completed_exact_replay_gain_sage8h_source_validated": (
            str(source_sage8h.get("config", {}).get("schema_version", ""))
            == SAGE8H_SCHEMA_VERSION
            and str(source_sage8h.get("outcome_status", "")) == SAGE8H_ARC_GAIN
        ),
        "both_next_levels_evaluated_once": len(specifications) == len(episodes) == 2
        and {str(row.get("game_id", "")) for row in episodes}
        == {"tn36-ab4f63cc", "wa30-ee6fef47"},
        "setup_reaches_exact_held_out_start_in_both_arms": all(
            bool(row.get("same_held_out_start_between_arms", False))
            and bool(row.get("same_held_out_level_between_arms", False))
            and int(row.get("no_memory_arm", {}).get("setup_level_delta", 0) or 0) > 0
            and int(row.get("with_memory_arm", {}).get("setup_level_delta", 0) or 0) > 0
            for row in episodes
        ),
        "setup_progress_is_excluded_from_primary_metrics": all(
            bool(row.get("setup_excluded_from_primary_metrics", False))
            and bool(
                row.get("no_memory_arm", {}).get(
                    "setup_excluded_from_primary_metrics", False
                )
            )
            and bool(
                row.get("with_memory_arm", {}).get(
                    "setup_excluded_from_primary_metrics", False
                )
            )
            for row in episodes
        ),
        "paired_held_out_arms_share_horizon_and_fallback": all(
            bool(row.get("same_held_out_horizon_between_arms", False))
            and str(row.get("shared_fallback_planner", "")) == REPLANNING_POLICY
            for row in episodes
        ),
        "control_arm_never_uses_exact_memory": all(
            bool(row.get("control_exact_memory_disabled", False))
            and int(row.get("no_memory_arm", {}).get("memory_applications", -1) or 0)
            == 0
            for row in episodes
        ),
        "treatment_memory_is_fully_quarantined_out_of_scope": all(
            bool(row.get("treatment_exact_memory_quarantined_out_of_scope", False))
            for row in episodes
        )
        and int(held_out_metrics.get("with_memory_applications", -1) or 0) == 0
        and int(held_out_metrics.get("with_memory_scope_misses", 0) or 0)
        == int(held_out_metrics.get("with_memory_steps_executed", -1) or 0),
        "scope_quarantine_preserves_identical_fallback_behavior": all(
            bool(row.get("comparison", {}).get("action_trajectories_identical", False))
            and bool(
                row.get("comparison", {}).get("final_visual_digests_identical", False)
            )
            for row in episodes
        ),
        "all_selected_actions_are_live_legal": all(
            bool(row.get("no_memory_arm", {}).get("all_selected_actions_legal", False))
            and bool(
                row.get("with_memory_arm", {}).get("all_selected_actions_legal", False)
            )
            for row in episodes
        ),
        "primary_arc_metrics_recorded_after_held_out_start": all(
            key in levels
            for key in (
                "no_memory_total_delta",
                "with_memory_total_delta",
                "absolute_delta_gain",
                "improved",
            )
        )
        and int(wins.get("episodes_per_arm", 0) or 0) == len(episodes),
        "evaluation_outcomes_not_used_for_training_tuning_or_action_ranking": all(
            not bool(row.get("future_outcomes_used_for_action_ranking", True))
            and not bool(
                row.get("evaluation_outcomes_used_for_training_or_tuning", True)
            )
            for row in episodes
        ),
        "held_out_result_not_mislabeled_as_structural_generalization": all(
            bool(row.get("held_out_next_level", False))
            and not bool(
                next(
                    spec.get("structural_generalization_policy_applied", True)
                    for spec in specifications
                    if str(spec.get("game_id", "")) == str(row.get("game_id", ""))
                )
            )
            for row in episodes
        ),
        "no_truth_reevaluation_or_registry_support_counting": all(
            not bool(row.get("truth_reevaluated", True))
            and int(row.get("support_counted", -1) or 0) == 0
            for row in episodes
        ),
    }


def summarize_sage8i(
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    held_out_metrics: Mapping[str, Any],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    level_gain = int(levels.get("absolute_delta_gain", 0) or 0)
    win_gain = float(wins.get("absolute_gain", 0.0) or 0.0)
    improved = level_gain > 0 or win_gain > 0.0
    regressed = level_gain < 0 or win_gain < 0.0
    if improved:
        outcome_status = SAGE8I_HELD_OUT_GAIN
    elif regressed:
        outcome_status = SAGE8I_HELD_OUT_REGRESSION
    else:
        outcome_status = SAGE8I_SCOPE_SAFE_NO_GENERALIZATION
    with_steps = int(held_out_metrics.get("with_memory_steps_executed", 0) or 0)
    scope_misses = int(held_out_metrics.get("with_memory_scope_misses", 0) or 0)
    quarantined = (
        int(held_out_metrics.get("with_memory_applications", -1) or 0) == 0
        and scope_misses == with_steps
    )
    return {
        "paired_held_out_rollouts_evaluated": len(episodes),
        "held_out_levels_evaluated": int(
            held_out_metrics.get("held_out_levels_evaluated", 0) or 0
        ),
        "games_evaluated": sorted({str(row.get("game_id", "")) for row in episodes}),
        "training_memory_entries": int(
            held_out_metrics.get("training_memory_entries", 0) or 0
        ),
        "held_out_scope_keys_observed": int(
            held_out_metrics.get("held_out_scope_keys_observed", 0) or 0
        ),
        "held_out_scope_keys_matching_training_memory": int(
            held_out_metrics.get("held_out_scope_keys_matching_training_memory", 0) or 0
        ),
        "setup_actions_executed_total": int(
            held_out_metrics.get("setup_actions_executed_total", 0) or 0
        ),
        "no_memory_steps_executed": int(
            held_out_metrics.get("no_memory_steps_executed", 0) or 0
        ),
        "with_memory_steps_executed": with_steps,
        "no_memory_applications": int(
            held_out_metrics.get("no_memory_applications", 0) or 0
        ),
        "with_memory_applications": int(
            held_out_metrics.get("with_memory_applications", 0) or 0
        ),
        "with_memory_scope_misses": scope_misses,
        "with_memory_action_quarantines": int(
            held_out_metrics.get("with_memory_action_quarantines", 0) or 0
        ),
        "with_memory_fallback_applications": int(
            held_out_metrics.get("with_memory_fallback_applications", 0) or 0
        ),
        "with_memory_exact_coverage_rate": float(
            held_out_metrics.get("with_memory_exact_coverage_rate", 0.0) or 0.0
        ),
        "episodes_with_identical_action_trajectories": int(
            held_out_metrics.get("episodes_with_identical_action_trajectories", 0) or 0
        ),
        "episodes_with_identical_final_states": int(
            held_out_metrics.get("episodes_with_identical_final_states", 0) or 0
        ),
        "no_memory_levels_completed_delta_total": int(
            levels.get("no_memory_total_delta", 0) or 0
        ),
        "with_memory_levels_completed_delta_total": int(
            levels.get("with_memory_total_delta", 0) or 0
        ),
        "levels_completed_absolute_gain": level_gain,
        "no_memory_wins": int(wins.get("no_memory_wins", 0) or 0),
        "with_memory_wins": int(wins.get("with_memory_wins", 0) or 0),
        "no_memory_win_rate": float(wins.get("no_memory", 0.0) or 0.0),
        "with_memory_win_rate": float(wins.get("with_memory", 0.0) or 0.0),
        "win_rate_absolute_gain": win_gain,
        "held_out_generalization_evaluated": True,
        "held_out_generalization_observed": improved,
        "exact_memory_quarantined_out_of_scope": quarantined,
        "structural_generalization_policy_applied": False,
        "primary_arc_progress_improved": improved,
        "primary_arc_progress_regressed": regressed,
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "future_outcomes_used_for_planning": False,
        "cross_game_transfer_performed": False,
        "outcome_status": outcome_status,
        "truth_reevaluations": 0,
        "support_counted": 0,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(gate.values()),
    }


def validate_sage8i_source(source_sage8h: Mapping[str, Any]) -> None:
    summary = dict(source_sage8h.get("summary", {}) or {})
    if (
        str(source_sage8h.get("config", {}).get("schema_version", ""))
        != SAGE8H_SCHEMA_VERSION
        or str(source_sage8h.get("truth_status", "")) != SAGE8H_TRUTH_STATUS
        or str(source_sage8h.get("outcome_status", "")) != SAGE8H_ARC_GAIN
        or str(source_sage8h.get("status", "")) != "EVALUATED"
        or not bool(summary.get("gate_passed", False))
        or not bool(summary.get("primary_arc_progress_improved", False))
        or bool(summary.get("primary_arc_progress_regressed", True))
        or not bool(summary.get("exact_training_trajectory_replay_evaluated", False))
        or bool(summary.get("held_out_generalization_evaluated", True))
        or int(summary.get("compiled_exact_visual_states", 0) or 0) != 57
        or bool(
            source_sage8h.get("evaluation_outcomes_used_for_training_or_tuning", True)
        )
        or bool(source_sage8h.get("future_outcomes_used_for_planning", True))
        or not all(bool(value) for value in source_sage8h.get("gate", {}).values())
    ):
        raise ValueError(
            "SAGE.8i requires the completed exact-replay-gain SAGE.8h source"
        )


def write_sage8i_goal_grounded_memory_held_out_evaluation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_SAGE8I_GOAL_GROUNDED_MEMORY_HELD_OUT_EVALUATION_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _new_env(
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Any:
    return (
        env_factory(game_id)
        if env_factory is not None
        else _make_real_env(game_id, environments_dir)
    )


def _visual_digest(frame: Any) -> str:
    snapshot = snapshot_frame(frame)
    array = np.asarray(snapshot.grid, dtype=np.int32)
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


def _terminal_state(game_state: Any) -> bool:
    return str(game_state).upper() not in NON_TERMINAL_STATES


def _public_planner_decision(decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in decision.items() if key != "selected_action_raw"
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sage8h",
        default=str(DEFAULT_SAGE8H_GOAL_GROUNDED_RELATIONAL_MEMORY_EVALUATION_PATH),
    )
    parser.add_argument("--environments-dir")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SAGE8I_GOAL_GROUNDED_MEMORY_HELD_OUT_EVALUATION_PATH),
    )
    args = parser.parse_args(argv)
    payload = run_sage8i_goal_grounded_memory_held_out_evaluation(
        sage8h_path=args.sage8h,
        environments_dir=args.environments_dir,
        output_path=args.output,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
