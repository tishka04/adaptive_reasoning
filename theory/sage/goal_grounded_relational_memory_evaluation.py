"""SAGE.8h exact goal-grounded memory paired closed-loop evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

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
from .target_goal_signal_active_acquisition import (
    DEFAULT_SAGE8G_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_PATH,
    SAGE8G_SCHEMA_VERSION,
    SAGE8G_TARGET_READY,
    SAGE8G_TRUTH_STATUS,
)


DEFAULT_SAGE8H_GOAL_GROUNDED_RELATIONAL_MEMORY_EVALUATION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage8h_goal_grounded_relational_memory_evaluation.json"
)

SAGE8H_SCHEMA_VERSION = "sage.goal_grounded_relational_memory_evaluation.v1"
SAGE8H_MEMORY_SCHEMA_VERSION = "sage.exact_goal_trajectory_action_memory.v1"
SAGE8H_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8H"
SAGE8H_ARC_GAIN = "SAGE_GOAL_GROUNDED_MEMORY_EXACT_REPLAY_ARC_SCORE_GAIN_OBSERVED"
SAGE8H_NO_GAIN = "SAGE_GOAL_GROUNDED_MEMORY_EXACT_REPLAY_NO_GAIN_OBSERVED"
SAGE8H_REGRESSION = (
    "SAGE_GOAL_GROUNDED_MEMORY_EXACT_REPLAY_ARC_SCORE_REGRESSION_OBSERVED"
)

MEMORY_ID = "EXACT_TARGET_GOAL_TRAJECTORY_ACTION_MEMORY"
MEMORY_SCOPE = "EXACT_GAME_AND_VISUAL_DIGEST_ONLY"
PAIRED_PLANNER = "EXACT_GOAL_MEMORY_THEN_SHARED_STATE_CONDITIONED_FALLBACK"
MEMORY_APPLIED = "EXACT_GOAL_TRAJECTORY_MEMORY_APPLIED"
MEMORY_DISABLED = "GOAL_TRAJECTORY_MEMORY_DISABLED_CONTROL_ARM"
MEMORY_OUT_OF_SCOPE = "NO_EXACT_GOAL_TRAJECTORY_MEMORY_STATE"
MEMORY_ACTION_UNAVAILABLE = "EXACT_GOAL_TRAJECTORY_ACTION_UNAVAILABLE"
TERMINAL_WIN_STATES = {"WIN", "WON", "VICTORY"}
NON_TERMINAL_STATES = {"", "NOT_FINISHED", "PLAYING", "IN_PROGRESS"}


def run_sage8h_goal_grounded_relational_memory_evaluation(
    *,
    sage8g_path: str | Path = (
        DEFAULT_SAGE8G_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_PATH
    ),
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Compile admitted positive trajectories, then run paired reset rollouts."""
    source_sage8g = _load_json(sage8g_path)
    validate_sage8h_source(source_sage8g)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    specifications = build_sage8h_evaluation_specifications(source_sage8g)
    trajectory_memory = compile_exact_goal_trajectory_memory(
        specifications,
        environments_dir=env_dir,
        env_factory=env_factory,
    )
    memory_index = build_goal_trajectory_memory_index(trajectory_memory)
    episodes = [
        execute_sage8h_paired_rollout(
            specification,
            memory_index=memory_index,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        for specification in specifications
    ]
    primary_metrics = summarize_primary_metrics(episodes)
    memory_metrics = summarize_goal_memory_metrics(episodes, trajectory_memory)
    gate = build_sage8h_gate(
        source_sage8g,
        specifications,
        trajectory_memory,
        episodes,
        primary_metrics,
        memory_metrics,
    )
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8h goal-grounded memory evaluation gate did not pass")
    summary = summarize_sage8h(
        episodes,
        primary_metrics,
        memory_metrics,
        gate,
    )
    payload = {
        "config": {
            "schema_version": SAGE8H_SCHEMA_VERSION,
            "sage8g_path": str(sage8g_path),
            "environments_dir": str(env_dir),
            "memory_id": MEMORY_ID,
            "memory_scope": MEMORY_SCOPE,
            "paired_planner": PAIRED_PLANNER,
            "fallback_planner": REPLANNING_POLICY,
            "evaluation_design": {
                "paired_reset_rollouts": True,
                "same_reset_and_horizon_between_arms": True,
                "control_arm_goal_memory_enabled": False,
                "treatment_arm_goal_memory_enabled": True,
                "shared_fallback_planner": True,
                "stop_on_observed_level_increase_or_terminal": True,
                "primary_metrics": ["levels_completed", "win_rate"],
                "future_outcomes_used_for_action_ranking": False,
                "evaluation_outcomes_used_for_training_or_tuning": False,
                "counterfactual_rollouts_performed": 0,
                "exact_scope_only": True,
                "fuzzy_matching_allowed": False,
                "cross_game_transfer_allowed": False,
            },
            "scientific_scope": {
                "evaluation_is_exact_training_trajectory_replay": True,
                "held_out_generalization_evaluation": False,
                "primary_gain_is_replay_conversion_not_generalization": True,
                "unseen_level_claim_allowed": False,
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
            ],
        },
        "evaluation_specifications": specifications,
        "goal_grounded_trajectory_memory": trajectory_memory,
        "paired_rollouts": episodes,
        "primary_metrics": primary_metrics,
        "memory_metrics": memory_metrics,
        "gate": gate,
        "summary": summary,
        "status": "EVALUATED",
        "outcome_status": summary["outcome_status"],
        "truth_status": SAGE8H_TRUTH_STATUS,
        "primary_arc_progress_improved": summary["primary_arc_progress_improved"],
        "primary_arc_progress_regressed": summary["primary_arc_progress_regressed"],
        "exact_replay_conversion_evaluated": True,
        "held_out_generalization_evaluated": False,
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
        write_sage8h_goal_grounded_relational_memory_evaluation(payload, output_path)
    return payload


def build_sage8h_evaluation_specifications(
    source_sage8g: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Build one reset-rooted paired rollout per admitted target trajectory."""
    specifications = []
    for row in source_sage8g.get("target_acquisitions", []) or []:
        game_id = str(row.get("game_id", ""))
        discovery = dict(row.get("discovery", {}) or {})
        admission = dict(row.get("admission", {}) or {})
        proof = dict(discovery.get("proof", {}) or {})
        commands = [
            {
                "action": str(command.get("action", "")),
                "action_args": dict(command.get("action_args", {}) or {}),
            }
            for command in discovery.get("commands", []) or []
        ]
        if not bool(admission.get("planner_activation_authorized", False)):
            continue
        if not commands or int(proof.get("positive_step", 0) or 0) != len(commands):
            raise ValueError("SAGE.8h requires a complete reset-to-positive sequence")
        specifications.append(
            {
                "evaluation_id": f"sage8h::{game_id}::exact_goal_replay",
                "game_id": game_id,
                "horizon": len(commands),
                "source_protocol": str(discovery.get("protocol", "")),
                "source_commands": commands,
                "expected_reset_visual_digest": str(
                    proof.get("reset_visual_digest", "")
                ),
                "expected_pre_positive_visual_digest": str(
                    proof.get("before_visual_digest", "")
                ),
                "expected_post_positive_visual_digest": str(
                    proof.get("after_visual_digest", "")
                ),
                "expected_level_delta": int(proof.get("level_delta", 0) or 0),
                "paired_arms": ["no_goal_memory", "with_goal_memory"],
                "same_reset_between_arms": True,
                "same_horizon_between_arms": True,
                "exact_training_trajectory_replay": True,
                "held_out_generalization_episode": False,
            }
        )
    return sorted(specifications, key=lambda row: str(row.get("game_id", "")))


def compile_exact_goal_trajectory_memory(
    specifications: Sequence[Mapping[str, Any]],
    *,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Compile exact game/state actions without reading evaluation outcomes."""
    raw_entries: list[Dict[str, Any]] = []
    replay_audits = []
    for specification in specifications:
        game_id = str(specification.get("game_id", ""))
        env = _new_env(game_id, environments_dir, env_factory)
        frame = _reset_env(env)
        reset_digest = _visual_digest(frame)
        expected_reset = str(specification.get("expected_reset_visual_digest", ""))
        if reset_digest != expected_reset:
            raise ValueError("SAGE.8h trajectory compiler reset digest mismatch")
        commands = list(specification.get("source_commands", []) or [])
        for index, command in enumerate(commands):
            digest = _visual_digest(frame)
            action = str(command.get("action", ""))
            args = dict(command.get("action_args", {}) or {})
            selected = select_live_action(env, action, action_args=args)
            if selected is None:
                raise ValueError(
                    f"SAGE.8h trajectory action unavailable:{game_id}:{index}"
                )
            raw_entries.append(
                {
                    "game_id": game_id,
                    "visual_digest": digest,
                    "scope_key": f"{game_id}|{digest}",
                    "action": action,
                    "action_args": args,
                    "action_identity": action_identity(action, args),
                    "source_trajectory_step": index,
                    "distance_to_observed_positive": len(commands) - index,
                    "source_evaluation_id": str(specification.get("evaluation_id", "")),
                }
            )
            frame = _step_env_action(env, selected)
        final_digest = _visual_digest(frame)
        expected_final = str(
            specification.get("expected_post_positive_visual_digest", "")
        )
        replay_audits.append(
            {
                "evaluation_id": str(specification.get("evaluation_id", "")),
                "game_id": game_id,
                "reset_visual_digest": reset_digest,
                "reset_visual_digest_match": reset_digest == expected_reset,
                "final_visual_digest": final_digest,
                "final_visual_digest_match": final_digest == expected_final,
                "actions_replayed": len(commands),
                "all_source_actions_legal": True,
            }
        )
        if final_digest != expected_final:
            raise ValueError("SAGE.8h trajectory compiler final digest mismatch")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_entries:
        grouped[str(row.get("scope_key", ""))].append(row)
    entries = []
    for scope_key, rows in sorted(grouped.items()):
        by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_identity[str(row.get("action_identity", ""))].append(row)
        candidates = []
        for identity, action_rows in sorted(by_identity.items()):
            representative = action_rows[0]
            candidates.append(
                {
                    "action": str(representative.get("action", "")),
                    "action_args": dict(representative.get("action_args", {}) or {}),
                    "action_identity": identity,
                    "demonstration_count": len(action_rows),
                    "minimum_distance_to_observed_positive": min(
                        int(row.get("distance_to_observed_positive", 0) or 0)
                        for row in action_rows
                    ),
                }
            )
        representative = rows[0]
        entries.append(
            {
                "memory_id": MEMORY_ID,
                "scope": MEMORY_SCOPE,
                "scope_key": scope_key,
                "game_id": str(representative.get("game_id", "")),
                "visual_digest": str(representative.get("visual_digest", "")),
                "action_candidates": candidates,
                "demonstration_count": len(rows),
                "source_trajectory_steps": sorted(
                    int(row.get("source_trajectory_step", 0) or 0) for row in rows
                ),
                "source_evaluation_ids": sorted(
                    {str(row.get("source_evaluation_id", "")) for row in rows}
                ),
                "exact_match_required": True,
                "fuzzy_match_allowed": False,
                "cross_game_transfer_allowed": False,
                "truth_status": SAGE8H_TRUTH_STATUS,
                "support": 0,
            }
        )

    ambiguous = sum(len(row.get("action_candidates", []) or []) > 1 for row in entries)
    gate = {
        "all_source_trajectories_replayed_exactly": bool(replay_audits)
        and all(
            bool(row.get("reset_visual_digest_match", False))
            and bool(row.get("final_visual_digest_match", False))
            and bool(row.get("all_source_actions_legal", False))
            for row in replay_audits
        ),
        "every_source_transition_compiled": len(entries) == len(raw_entries),
        "no_ambiguous_exact_states": ambiguous == 0,
        "all_entries_are_exact_game_state_scope": all(
            bool(row.get("exact_match_required", False))
            and not bool(row.get("fuzzy_match_allowed", True))
            and not bool(row.get("cross_game_transfer_allowed", True))
            for row in entries
        ),
        "no_truth_or_support_mutation": all(
            str(row.get("truth_status", "")) == SAGE8H_TRUTH_STATUS
            and int(row.get("support", -1) or 0) == 0
            for row in entries
        ),
    }
    return {
        "config": {
            "schema_version": SAGE8H_MEMORY_SCHEMA_VERSION,
            "memory_id": MEMORY_ID,
            "scope": MEMORY_SCOPE,
            "training_source": "SAGE.8g_ADMITTED_POSITIVE_COMMAND_SEQUENCES",
            "training_fields_read": [
                "game_id",
                "commands.action",
                "commands.action_args",
                "reset_visual_digest",
                "after_visual_digest",
            ],
            "evaluation_outcome_fields_read": [],
            "evaluation_outcomes_used_for_training_or_tuning": False,
            "exact_match_required": True,
            "fuzzy_matching_allowed": False,
            "cross_game_transfer_allowed": False,
        },
        "entries": entries,
        "source_replay_audits": replay_audits,
        "gate": gate,
        "summary": {
            "source_positive_trajectories": len(specifications),
            "source_trajectories_exactly_replayed": sum(
                bool(row.get("final_visual_digest_match", False))
                for row in replay_audits
            ),
            "demonstration_transitions": len(raw_entries),
            "exact_visual_states": len(entries),
            "ambiguous_exact_states": ambiguous,
            "games": sorted({str(row.get("game_id", "")) for row in specifications}),
            "evaluation_outcomes_used_for_training_or_tuning": False,
            "gate_passed": bool(gate) and all(gate.values()),
        },
        "truth_status": SAGE8H_TRUTH_STATUS,
        "support": 0,
        "wrong_confirmations": 0,
        "scope_generalization_performed": False,
    }


def build_goal_trajectory_memory_index(
    trajectory_memory: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("scope_key", "")): dict(row)
        for row in trajectory_memory.get("entries", []) or []
        if str(row.get("scope_key", ""))
    }


def execute_sage8h_paired_rollout(
    specification: Mapping[str, Any],
    *,
    memory_index: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    no_memory = _execute_sage8h_arm(
        specification,
        memory_index=memory_index,
        memory_enabled=False,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    with_memory = _execute_sage8h_arm(
        specification,
        memory_index=memory_index,
        memory_enabled=True,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if no_memory.get("status") != "EXECUTED" or with_memory.get("status") != "EXECUTED":
        raise ValueError(
            "SAGE.8h paired arm blocked:"
            f"{no_memory.get('reason', '')}:{with_memory.get('reason', '')}"
        )
    no_before = int(no_memory.get("levels_completed_before", 0) or 0)
    with_before = int(with_memory.get("levels_completed_before", 0) or 0)
    no_after = int(no_memory.get("levels_completed_after", 0) or 0)
    with_after = int(with_memory.get("levels_completed_after", 0) or 0)
    divergence = compare_sage8h_action_trajectories(no_memory, with_memory)
    return {
        "evaluation_id": str(specification.get("evaluation_id", "")),
        "game_id": str(specification.get("game_id", "")),
        "horizon": int(specification.get("horizon", 0) or 0),
        "same_reset_between_arms": (
            str(no_memory.get("reset_visual_digest", ""))
            == str(with_memory.get("reset_visual_digest", ""))
            == str(specification.get("expected_reset_visual_digest", ""))
        ),
        "same_horizon_between_arms": (
            int(no_memory.get("horizon", 0) or 0)
            == int(with_memory.get("horizon", 0) or 0)
            == int(specification.get("horizon", 0) or 0)
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
        "relational_goal_memory_applied": int(
            with_memory.get("memory_applications", 0) or 0
        )
        > 0,
        "control_goal_memory_disabled": int(
            no_memory.get("memory_applications", 0) or 0
        )
        == 0,
        "treatment_exact_source_sequence_replayed": bool(
            with_memory.get("exact_source_sequence_replayed", False)
        ),
        "treatment_source_positive_final_digest_reproduced": bool(
            with_memory.get("source_positive_final_digest_reproduced", False)
        ),
        "exact_training_trajectory_replay": True,
        "held_out_generalization_episode": False,
        "future_outcomes_used_for_action_ranking": False,
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "divergence": divergence,
        "no_memory_arm": no_memory,
        "with_memory_arm": with_memory,
        "primary_metrics": ["levels_completed", "win_rate"],
        "truth_reevaluated": False,
        "support_counted": 0,
        "wrong_confirmations": 0,
    }


def _execute_sage8h_arm(
    specification: Mapping[str, Any],
    *,
    memory_index: Mapping[str, Mapping[str, Any]],
    memory_enabled: bool,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    game_id = str(specification.get("game_id", ""))
    horizon = int(specification.get("horizon", 0) or 0)
    try:
        env = _new_env(game_id, environments_dir, env_factory)
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return {"status": "BLOCKED", "reason": f"env_setup_failed:{exc}"}
    reset_snapshot = snapshot_frame(frame)
    reset_digest = _visual_digest(frame)
    initial_levels = int(reset_snapshot.levels_completed)
    family_counts: Counter[str] = Counter()
    concrete_counts: Counter[str] = Counter()
    state_action_visits: Counter[str] = Counter()
    previous_identity = ""
    trace = []
    executed_commands = []
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
                "reason": f"no_legal_action:{step}",
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
        command = {"action": action, "action_args": args}
        executed_commands.append(command)
        trace.append(
            {
                "rollout_step": step,
                "before_visual_digest": current_digest,
                "after_visual_digest": after_digest,
                **command,
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
    source_commands = list(specification.get("source_commands", []) or [])
    final_digest = _visual_digest(frame)
    return {
        "status": "EXECUTED",
        "arm": "with_goal_memory" if memory_enabled else "no_goal_memory",
        "memory_enabled": memory_enabled,
        "horizon": horizon,
        "reset_visual_digest": reset_digest,
        "reset_visual_digest_matches_source": reset_digest
        == str(specification.get("expected_reset_visual_digest", "")),
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
        "exact_source_sequence_replayed": executed_commands == source_commands,
        "source_positive_final_digest_reproduced": final_digest
        == str(specification.get("expected_post_positive_visual_digest", "")),
        "final_visual_digest": final_digest,
        "all_selected_actions_legal": True,
        "future_outcome_fields_read_for_action_ranking": [],
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "counterfactual_rollouts_performed": 0,
        "trace": trace,
    }


def select_exact_goal_memory_action(
    valid_actions: Sequence[Any],
    *,
    game_id: str,
    current_visual_digest: str,
    memory_index: Mapping[str, Mapping[str, Any]],
    memory_enabled: bool,
) -> tuple[Any | None, Dict[str, Any]]:
    scope_key = f"{game_id}|{current_visual_digest}"
    base = {
        "memory_id": MEMORY_ID,
        "memory_scope": MEMORY_SCOPE,
        "scope_key": scope_key,
        "current_game_id": game_id,
        "current_visual_digest": current_visual_digest,
        "exact_match_required": True,
        "fuzzy_match_performed": False,
        "cross_game_transfer_performed": False,
        "future_outcome_fields_read": [],
    }
    if not memory_enabled:
        return None, {
            **base,
            "memory_enabled": False,
            "memory_applied": False,
            "memory_decision_reason": MEMORY_DISABLED,
        }
    entry = memory_index.get(scope_key)
    if entry is None:
        return None, {
            **base,
            "memory_enabled": True,
            "memory_applied": False,
            "memory_decision_reason": MEMORY_OUT_OF_SCOPE,
        }
    candidates = list(entry.get("action_candidates", []) or [])
    by_identity = {
        action_identity(
            str(getattr(action, "name", "")),
            dict(getattr(action, "action_args", {}) or {}),
        ): action
        for action in valid_actions
        if str(getattr(action, "name", "")) != "RESET"
    }
    for candidate in sorted(
        candidates,
        key=lambda row: (
            int(row.get("minimum_distance_to_observed_positive", 0) or 0),
            str(row.get("action_identity", "")),
        ),
    ):
        identity = str(candidate.get("action_identity", ""))
        selected = by_identity.get(identity)
        if selected is not None:
            return selected, {
                **base,
                "memory_enabled": True,
                "memory_applied": True,
                "memory_decision_reason": MEMORY_APPLIED,
                "selected_action_identity": identity,
                "candidate_count": len(candidates),
            }
    return None, {
        **base,
        "memory_enabled": True,
        "memory_applied": False,
        "memory_decision_reason": MEMORY_ACTION_UNAVAILABLE,
        "candidate_count": len(candidates),
    }


def compare_sage8h_action_trajectories(
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
        "trajectories_diverged": divergent > 0,
        "divergence_allowed_by_memory_ablation": True,
    }


def summarize_goal_memory_metrics(
    episodes: Sequence[Mapping[str, Any]],
    trajectory_memory: Mapping[str, Any],
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
    with_fallbacks = sum(
        int(row.get("with_memory_arm", {}).get("fallback_applications", 0) or 0)
        for row in episodes
    )
    return {
        "compiled_demonstration_transitions": int(
            trajectory_memory.get("summary", {}).get("demonstration_transitions", 0)
            or 0
        ),
        "compiled_exact_visual_states": int(
            trajectory_memory.get("summary", {}).get("exact_visual_states", 0) or 0
        ),
        "no_memory_steps_executed": no_steps,
        "with_memory_steps_executed": with_steps,
        "no_memory_applications": no_apps,
        "with_memory_applications": with_apps,
        "with_memory_fallback_applications": with_fallbacks,
        "with_memory_exact_coverage_rate": with_apps / with_steps
        if with_steps
        else 0.0,
        "exact_source_sequences_replayed": sum(
            bool(
                row.get("with_memory_arm", {}).get(
                    "exact_source_sequence_replayed", False
                )
            )
            for row in episodes
        ),
        "source_positive_final_digests_reproduced": sum(
            bool(
                row.get("with_memory_arm", {}).get(
                    "source_positive_final_digest_reproduced", False
                )
            )
            for row in episodes
        ),
        "episodes_with_divergent_action_trajectories": sum(
            bool(row.get("divergence", {}).get("trajectories_diverged", False))
            for row in episodes
        ),
    }


def build_sage8h_gate(
    source_sage8g: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
    trajectory_memory: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    memory_metrics: Mapping[str, Any],
) -> Dict[str, bool]:
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    entries = list(trajectory_memory.get("entries", []) or [])
    return {
        "completed_sage8g_source_validated": (
            str(source_sage8g.get("config", {}).get("schema_version", ""))
            == SAGE8G_SCHEMA_VERSION
            and str(source_sage8g.get("outcome_status", "")) == SAGE8G_TARGET_READY
        ),
        "both_admitted_target_games_evaluated_once": len(specifications)
        == len(episodes)
        == 2
        and {str(row.get("game_id", "")) for row in episodes}
        == {"tn36-ab4f63cc", "wa30-ee6fef47"},
        "trajectory_memory_compiler_gate_passed": bool(
            trajectory_memory.get("summary", {}).get("gate_passed", False)
        ),
        "all_memory_entries_are_exact_without_transfer": bool(entries)
        and all(
            bool(row.get("exact_match_required", False))
            and not bool(row.get("fuzzy_match_allowed", True))
            and not bool(row.get("cross_game_transfer_allowed", True))
            for row in entries
        ),
        "paired_arms_share_reset_horizon_and_fallback": all(
            bool(row.get("same_reset_between_arms", False))
            and bool(row.get("same_horizon_between_arms", False))
            and str(row.get("shared_fallback_planner", "")) == REPLANNING_POLICY
            for row in episodes
        ),
        "control_arm_never_uses_goal_memory": all(
            bool(row.get("control_goal_memory_disabled", False))
            and int(row.get("no_memory_arm", {}).get("memory_applications", -1) or 0)
            == 0
            for row in episodes
        ),
        "treatment_replays_every_exact_source_sequence": all(
            bool(row.get("treatment_exact_source_sequence_replayed", False))
            and bool(
                row.get("treatment_source_positive_final_digest_reproduced", False)
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
        "primary_arc_metrics_recorded": all(
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
        )
        and not bool(
            trajectory_memory.get("config", {}).get(
                "evaluation_outcomes_used_for_training_or_tuning", True
            )
        ),
        "exact_replay_gain_not_mislabeled_as_held_out_generalization": all(
            bool(row.get("exact_training_trajectory_replay", False))
            and not bool(row.get("held_out_generalization_episode", True))
            for row in episodes
        ),
        "memory_metrics_have_full_treatment_accounting": int(
            memory_metrics.get("with_memory_applications", 0) or 0
        )
        + int(memory_metrics.get("with_memory_fallback_applications", 0) or 0)
        == int(memory_metrics.get("with_memory_steps_executed", -1) or 0),
        "no_truth_reevaluation_or_registry_support_counting": all(
            not bool(row.get("truth_reevaluated", True))
            and int(row.get("support_counted", -1) or 0) == 0
            for row in episodes
        ),
    }


def summarize_sage8h(
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    memory_metrics: Mapping[str, Any],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    level_gain = int(levels.get("absolute_delta_gain", 0) or 0)
    win_gain = float(wins.get("absolute_gain", 0.0) or 0.0)
    improved = level_gain > 0 or win_gain > 0.0
    regressed = level_gain < 0 or win_gain < 0.0
    if improved:
        outcome_status = SAGE8H_ARC_GAIN
    elif regressed:
        outcome_status = SAGE8H_REGRESSION
    else:
        outcome_status = SAGE8H_NO_GAIN
    return {
        "paired_rollouts_evaluated": len(episodes),
        "games_evaluated": sorted({str(row.get("game_id", "")) for row in episodes}),
        "compiled_demonstration_transitions": int(
            memory_metrics.get("compiled_demonstration_transitions", 0) or 0
        ),
        "compiled_exact_visual_states": int(
            memory_metrics.get("compiled_exact_visual_states", 0) or 0
        ),
        "no_memory_steps_executed": int(
            memory_metrics.get("no_memory_steps_executed", 0) or 0
        ),
        "with_memory_steps_executed": int(
            memory_metrics.get("with_memory_steps_executed", 0) or 0
        ),
        "no_memory_applications": int(
            memory_metrics.get("no_memory_applications", 0) or 0
        ),
        "with_memory_applications": int(
            memory_metrics.get("with_memory_applications", 0) or 0
        ),
        "with_memory_fallback_applications": int(
            memory_metrics.get("with_memory_fallback_applications", 0) or 0
        ),
        "with_memory_exact_coverage_rate": float(
            memory_metrics.get("with_memory_exact_coverage_rate", 0.0) or 0.0
        ),
        "exact_source_sequences_replayed": int(
            memory_metrics.get("exact_source_sequences_replayed", 0) or 0
        ),
        "source_positive_final_digests_reproduced": int(
            memory_metrics.get("source_positive_final_digests_reproduced", 0) or 0
        ),
        "episodes_with_divergent_action_trajectories": int(
            memory_metrics.get("episodes_with_divergent_action_trajectories", 0) or 0
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
        "primary_arc_progress_improved": improved,
        "primary_arc_progress_regressed": regressed,
        "exact_training_trajectory_replay_evaluated": True,
        "held_out_generalization_evaluated": False,
        "primary_gain_is_replay_conversion_not_generalization": True,
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "future_outcomes_used_for_planning": False,
        "cross_game_transfer_performed": False,
        "outcome_status": outcome_status,
        "truth_reevaluations": 0,
        "support_counted": 0,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(gate.values()),
    }


def validate_sage8h_source(source_sage8g: Mapping[str, Any]) -> None:
    summary = dict(source_sage8g.get("summary", {}) or {})
    if (
        str(source_sage8g.get("config", {}).get("schema_version", ""))
        != SAGE8G_SCHEMA_VERSION
        or str(source_sage8g.get("truth_status", "")) != SAGE8G_TRUTH_STATUS
        or str(source_sage8g.get("outcome_status", "")) != SAGE8G_TARGET_READY
        or not bool(summary.get("gate_passed", False))
        or int(summary.get("target_games_with_exact_positive_replay", 0) or 0) != 2
        or int(summary.get("exact_target_goal_signal_entries", 0) or 0) != 2
        or not bool(source_sage8g.get("planner_activation_authorized", False))
        or not bool(
            source_sage8g.get("paired_closed_loop_evaluation_authorized", False)
        )
        or bool(source_sage8g.get("paired_closed_loop_evaluation_performed", True))
        or bool(source_sage8g.get("cross_game_transfer_performed", True))
        or list(source_sage8g.get("game_source_files_opened", ["drift"])) != []
        or not all(bool(value) for value in source_sage8g.get("gate", {}).values())
    ):
        raise ValueError(
            "SAGE.8h requires the completed exact-target-admitted SAGE.8g source"
        )


def write_sage8h_goal_grounded_relational_memory_evaluation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_SAGE8H_GOAL_GROUNDED_RELATIONAL_MEMORY_EVALUATION_PATH
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
        "--sage8g",
        default=str(DEFAULT_SAGE8G_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_PATH),
    )
    parser.add_argument("--environments-dir")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SAGE8H_GOAL_GROUNDED_RELATIONAL_MEMORY_EVALUATION_PATH),
    )
    args = parser.parse_args(argv)
    payload = run_sage8h_goal_grounded_relational_memory_evaluation(
        sage8g_path=args.sage8g,
        environments_dir=args.environments_dir,
        output_path=args.output,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
