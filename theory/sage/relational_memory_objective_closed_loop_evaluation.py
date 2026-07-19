"""SAGE.8e scope-safe learned-objective closed-loop memory evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
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
    build_sage8b_evaluation_specifications,
    summarize_primary_metrics,
    summarize_secondary_metrics,
)
from .relational_memory_closed_loop_evaluation import (
    DEFAULT_SAGE8D_RELATIONAL_MEMORY_CLOSED_LOOP_EVALUATION_PATH,
    SAGE8D_LOCAL_ONLY,
    SAGE8D_SCHEMA_VERSION,
    SAGE8D_TRUTH_STATUS,
    action_identity,
    compare_replanned_trajectories,
    select_state_conditioned_replanning_action,
    validate_sage8d_sources,
)
from .relational_memory_multi_action_evaluation import (
    DEFAULT_CONTINUATION_HORIZON,
    DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH,
)
from .relational_memory_policy import (
    DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
    LOWER_EFFECT_COMPARATOR_MATCH,
    PolicyActionOption,
    apply_relational_memory_policy,
)


DEFAULT_SAGE8E_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_EVALUATION_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage8e_relational_memory_objective_closed_loop_evaluation.json"
)

SAGE8E_SCHEMA_VERSION = "sage.relational_memory_objective_closed_loop_evaluation.v1"
SAGE8E_OBJECTIVE_SCHEMA_VERSION = "sage.scope_safe_progression_objective.v1"
SAGE8E_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8E"
SAGE8E_ARC_GAIN = "SAGE_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_ARC_SCORE_GAIN_OBSERVED"
SAGE8E_LOCAL_ONLY = "SAGE_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_LOCAL_GAIN_WITHOUT_ARC_SCORE_CONVERSION"
SAGE8E_NO_GAIN = "SAGE_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_NO_GAIN_OBSERVED"
SAGE8E_ARC_REGRESSION = (
    "SAGE_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_ARC_SCORE_REGRESSION_OBSERVED"
)

OBJECTIVE_ID = "REGISTERED_CONTEXT_REACHABILITY_PREFIX_IMITATION"
OBJECTIVE_SCOPE = "EXACT_GAME_AND_VISUAL_DIGEST_ONLY"
OBJECTIVE_PLANNER = "SCOPE_SAFE_LEARNED_OBJECTIVE_THEN_STATE_CONDITIONED_FALLBACK"
OBJECTIVE_APPLIED = "EXACT_LEARNED_OBJECTIVE_STATE_ACTION_APPLIED"
OBJECTIVE_OUT_OF_SCOPE = "NO_EXACT_LEARNED_OBJECTIVE_STATE"
OBJECTIVE_ACTION_UNAVAILABLE = "LEARNED_OBJECTIVE_ACTION_UNAVAILABLE"


def run_sage8e_relational_memory_objective_closed_loop_evaluation(
    *,
    sage8d_path: str | Path = (
        DEFAULT_SAGE8D_RELATIONAL_MEMORY_CLOSED_LOOP_EVALUATION_PATH
    ),
    sage8c_path: str | Path = (
        DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH
    ),
    sage8b_path: str | Path = DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
    policy_path: str | Path = DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
    a34_2_path: str | Path = (
        DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH
    ),
    a34_3_path: str | Path = DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
    environments_dir: str | Path | None = None,
    continuation_horizon: int = DEFAULT_CONTINUATION_HORIZON,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Evaluate a preregistered exact-state learned objective with memory."""
    source_sage8d = _load_json(sage8d_path)
    source_sage8c = _load_json(sage8c_path)
    source_sage8b = _load_json(sage8b_path)
    policy_source = _load_json(policy_path)
    a34_2 = _load_json(a34_2_path)
    a34_3 = _load_json(a34_3_path)
    validate_sage8e_sources(
        source_sage8d,
        source_sage8c,
        source_sage8b,
        policy_source,
        a34_2,
        a34_3,
    )
    _validate_objective_closed_loop_config(continuation_horizon)
    base_specifications = build_sage8b_evaluation_specifications(
        policy_source, a34_2, a34_3
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    objective_model = compile_scope_safe_progression_objective(
        base_specifications,
        environments_dir=env_dir,
        env_factory=env_factory,
    )
    specifications = tuple(
        build_sage8e_rollout_specification(
            row,
            continuation_horizon=continuation_horizon,
            objective_model=objective_model,
        )
        for row in base_specifications
    )
    objective_index = build_objective_index(objective_model)
    episodes = tuple(
        execute_sage8e_paired_objective_rollout(
            specification,
            policy=policy_source["policy"],
            objective_index=objective_index,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        for specification in specifications
    )
    primary_metrics = summarize_primary_metrics(episodes)
    secondary_metrics = summarize_secondary_metrics(episodes)
    objective_metrics = summarize_objective_planner_metrics(episodes, objective_model)
    gate = build_sage8e_gate(
        specifications,
        episodes,
        objective_model,
        primary_metrics,
        objective_metrics,
        continuation_horizon=continuation_horizon,
    )
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8e learned-objective evaluation gate did not pass")
    summary = summarize_sage8e(
        episodes,
        primary_metrics,
        secondary_metrics,
        objective_metrics,
        gate,
    )
    payload = {
        "config": {
            "schema_version": SAGE8E_SCHEMA_VERSION,
            "sage8d_path": str(sage8d_path),
            "sage8c_path": str(sage8c_path),
            "sage8b_path": str(sage8b_path),
            "policy_path": str(policy_path),
            "a34_2_path": str(a34_2_path),
            "a34_3_path": str(a34_3_path),
            "environments_dir": str(env_dir),
            "continuation_horizon": int(continuation_horizon),
            "objective_planner": OBJECTIVE_PLANNER,
            "objective_id": OBJECTIVE_ID,
            "objective_scope": OBJECTIVE_SCOPE,
            "inputs_read": [
                "SAGE.8d_MOTIVATION_AND_SCOPE_AUDIT_ONLY",
                "SAGE.8c_SOURCE_VALIDATION_ONLY",
                "SAGE.8b_SOURCE_VALIDATION_ONLY",
                "SAGE.8a_POLICY",
                "A34.2_REPLAY_ACTION_HISTORIES",
                "A34.3_REPLAY_ACTION_HISTORIES",
            ],
            "objective_training_design": {
                "training_sources": [
                    "A34.2_CONTEXT_REPLAY_ACTIONS",
                    "A34.3_CONTEXT_REPLAY_ACTIONS",
                ],
                "training_target": (
                    "NEXT_ACTION_ON_PREFIXES_REACHING_REGISTERED_RELATIONAL_CONTEXTS"
                ),
                "training_scope": OBJECTIVE_SCOPE,
                "exact_visual_digest_match_required": True,
                "fuzzy_or_cross_game_generalization": False,
                "training_outcome_fields_read": [],
                "sage8b_outcomes_used_for_training_or_tuning": False,
                "sage8c_outcomes_used_for_training_or_tuning": False,
                "sage8d_outcomes_used_for_training_or_tuning": False,
                "objective_is_arc_progress_truth_claim": False,
                "objective_is_preregistered_progress_proxy": True,
            },
            "evaluation_design": {
                "paired_exact_context_replays": True,
                "same_prefix_between_arms": True,
                "same_objective_planner_between_arms": True,
                "only_initial_memory_intervention_is_forced": True,
                "continuation_replanned_after_every_live_step": True,
                "exact_scope_objective_or_quarantine": True,
                "out_of_scope_falls_back_to_sage8d_planner": True,
                "future_action_schedule_preselected": False,
                "future_outcome_fields_read_for_planning": [],
                "counterfactual_rollouts_performed": 0,
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
                "SAGE.8c",
                "SAGE.8d",
            ],
        },
        "objective_model": objective_model,
        "rollout_specifications": [
            _public_specification(row) for row in specifications
        ],
        "paired_rollouts": [dict(row) for row in episodes],
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "objective_metrics": objective_metrics,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "EVALUATED",
        "truth_status": SAGE8E_TRUTH_STATUS,
        "comparative_evaluation_performed": True,
        "closed_loop_live_rollout_performed": True,
        "learned_objective_compiled_before_evaluation": True,
        "scope_safe_objective_planning_performed": True,
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
        write_sage8e_relational_memory_objective_closed_loop_evaluation(
            payload, output_path
        )
    return payload


def compile_scope_safe_progression_objective(
    specifications: Sequence[Mapping[str, Any]],
    *,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Learn exact-state next-action priors from pre-evaluation replay histories."""
    state_actions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    action_payloads: dict[str, Dict[str, Any]] = {}
    state_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    endpoint_digests: dict[str, set[str]] = defaultdict(set)
    transitions = 0
    verified = 0
    for specification in specifications:
        request = dict(specification.get("request", {}) or {})
        game_id = str(specification.get("game_id", ""))
        try:
            env = (
                env_factory(game_id)
                if env_factory is not None
                else _make_real_env(game_id, environments_dir)
            )
            frame = _reset_env(env)
        except Exception as exc:  # pragma: no cover - integration failure path
            raise ValueError(f"SAGE.8e objective replay setup failed: {exc}") from exc
        names = [str(value) for value in request.get("context_replay", []) or []]
        raw_args = list(request.get("context_replay_args", []) or [])
        source_id = str(specification.get("source_request_id", ""))
        for index, name in enumerate(names):
            args = (
                dict(raw_args[index])
                if index < len(raw_args) and isinstance(raw_args[index], Mapping)
                else {}
            )
            digest = _visual_digest(frame)
            identity = action_identity(name, args)
            state_actions[(game_id, digest)][identity] += 1
            state_sources[(game_id, digest)].add(source_id)
            action_payloads[identity] = {"action": name, "action_args": args}
            selected = select_live_action(env, name, action_args=args)
            if selected is None:
                raise ValueError(
                    f"SAGE.8e objective replay action unavailable:{source_id}:{index}"
                )
            frame = _step_env_action(env, selected)
            transitions += 1
        final_signature = state_signature_from_frame(frame)
        expected = str(specification.get("context_snapshot_hash", ""))
        if final_signature != expected:
            raise ValueError("SAGE.8e objective replay context hash mismatch")
        endpoint_digests[game_id].add(_visual_digest(frame))
        verified += 1
    entries = []
    for game_id, digest in sorted(state_actions):
        counts = state_actions[(game_id, digest)]
        candidates = []
        for identity, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        ):
            candidates.append(
                {
                    **action_payloads[identity],
                    "action_identity": identity,
                    "demonstration_count": int(count),
                }
            )
        entries.append(
            {
                "game_id": game_id,
                "visual_digest": digest,
                "scope_key": f"{game_id}|{digest}",
                "objective_id": OBJECTIVE_ID,
                "scope": OBJECTIVE_SCOPE,
                "action_candidates": candidates,
                "demonstration_transitions": int(sum(counts.values())),
                "source_request_ids": sorted(state_sources[(game_id, digest)]),
                "exact_match_required": True,
                "fuzzy_match_allowed": False,
                "cross_game_transfer_allowed": False,
                "truth_status": SAGE8E_TRUTH_STATUS,
                "support": 0,
            }
        )
    summary = {
        "source_replays": len(specifications),
        "source_replays_exactly_verified": verified,
        "games": sorted({str(row.get("game_id", "")) for row in specifications}),
        "demonstration_transitions": transitions,
        "exact_visual_states": len(entries),
        "ambiguous_visual_states": sum(
            len(row.get("action_candidates", []) or []) > 1 for row in entries
        ),
        "endpoint_visual_states": sum(
            len(values) for values in endpoint_digests.values()
        ),
        "training_outcome_fields_read": 0,
        "evaluation_outcomes_used_for_training_or_tuning": 0,
        "fuzzy_scope_entries": 0,
        "cross_game_scope_entries": 0,
        "gate_passed": bool(entries)
        and len(specifications) == verified == 11
        and transitions > 0,
    }
    return {
        "config": {
            "schema_version": SAGE8E_OBJECTIVE_SCHEMA_VERSION,
            "objective_id": OBJECTIVE_ID,
            "scope": OBJECTIVE_SCOPE,
            "training_sources": [
                "A34.2_CONTEXT_REPLAY_ACTIONS",
                "A34.3_CONTEXT_REPLAY_ACTIONS",
            ],
            "training_fields_read": [
                "game_id",
                "context_replay",
                "context_replay_args",
                "current_visual_digest",
            ],
            "training_outcome_fields_read": [],
            "evaluation_outcomes_used_for_training_or_tuning": [],
            "objective_is_arc_progress_truth_claim": False,
        },
        "state_action_entries": entries,
        "endpoint_visual_digests_by_game": {
            game_id: sorted(values)
            for game_id, values in sorted(endpoint_digests.items())
        },
        "summary": summary,
        "truth_status": SAGE8E_TRUTH_STATUS,
        "support": 0,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
    }


def build_objective_index(
    objective_model: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("scope_key", "")): dict(row)
        for row in objective_model.get("state_action_entries", []) or []
        if isinstance(row, Mapping) and str(row.get("scope_key", ""))
    }


def build_sage8e_rollout_specification(
    base: Mapping[str, Any],
    *,
    continuation_horizon: int,
    objective_model: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach the preregistered objective model without reading evaluation outcomes."""
    return {
        **dict(base),
        "evaluation_id": str(base.get("evaluation_id", "")).replace(
            "sage8b::paired", "sage8e::objective_closed_loop"
        ),
        "continuation_horizon": int(continuation_horizon),
        "objective_planner": OBJECTIVE_PLANNER,
        "objective_id": OBJECTIVE_ID,
        "objective_scope": OBJECTIVE_SCOPE,
        "objective_model_exact_states": int(
            objective_model.get("summary", {}).get("exact_visual_states", 0) or 0
        ),
        "future_action_schedule_preselected": False,
        "same_planner_for_both_arms": True,
        "future_outcome_fields_read_for_planning": [],
        "evaluation_outcomes_used_for_tuning": [],
    }


def execute_sage8e_paired_objective_rollout(
    specification: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    objective_index: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute identical scope-safe objective planners after the memory intervention."""
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
        raise ValueError("SAGE.8e requires an applied SAGE.8a policy decision")
    request = dict(specification.get("request", {}) or {})
    horizon = int(specification.get("continuation_horizon", 0) or 0)
    no_memory = _execute_objective_closed_loop_arm(
        request,
        initial_action=proposed.name,
        initial_action_args=proposed.action_args,
        continuation_horizon=horizon,
        objective_index=objective_index,
        arm="sage8e_no_memory",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    with_memory = _execute_objective_closed_loop_arm(
        request,
        initial_action=str(decision.get("selected_action", "")),
        initial_action_args=dict(decision.get("selected_action_args", {}) or {}),
        continuation_horizon=horizon,
        objective_index=objective_index,
        arm="sage8e_with_relational_memory",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if any(
        str(arm.get("status", "")) != "EXECUTED" for arm in (no_memory, with_memory)
    ):
        reasons = [str(arm.get("reason", "")) for arm in (no_memory, with_memory)]
        raise ValueError(f"SAGE.8e paired objective arm blocked: {reasons}")
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
        "same_objective_planner_between_arms": (
            no_memory.get("objective_planner")
            == with_memory.get("objective_planner")
            == OBJECTIVE_PLANNER
        ),
        "continuation_horizon": horizon,
        "objective_planner": OBJECTIVE_PLANNER,
        "objective_id": OBJECTIVE_ID,
        "objective_scope": OBJECTIVE_SCOPE,
        "future_action_schedule_preselected": False,
        "future_outcome_fields_read_for_planning": [],
        "evaluation_outcomes_used_for_tuning": [],
        "no_memory_action": proposed.name,
        "no_memory_action_args": dict(proposed.action_args),
        "with_memory_action": str(decision.get("selected_action", "")),
        "with_memory_action_args": dict(decision.get("selected_action_args", {}) or {}),
        "no_memory_arm": dict(no_memory),
        "with_memory_arm": dict(with_memory),
        "trajectory_comparison": compare_replanned_trajectories(no_memory, with_memory),
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


def _execute_objective_closed_loop_arm(
    request: Mapping[str, Any],
    *,
    initial_action: str,
    initial_action_args: Mapping[str, Any],
    continuation_horizon: int,
    objective_index: Mapping[str, Mapping[str, Any]],
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
    replay_names = [str(value) for value in request.get("context_replay", []) or []]
    replay_args = list(request.get("context_replay_args", []) or [])
    concrete_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for index, name in enumerate(replay_names):
        args = (
            dict(replay_args[index])
            if index < len(replay_args) and isinstance(replay_args[index], Mapping)
            else {}
        )
        selected = select_live_action(env, name, action_args=args)
        if selected is None:
            return {
                "status": "BLOCKED",
                "reason": f"prefix_action_unavailable:{name}",
                "arm": arm,
            }
        frame = _step_env_action(env, selected)
        _increment_action_counts(name, args, concrete_counts, family_counts)
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
    _increment_action_counts(
        initial_action, initial_action_args, concrete_counts, family_counts
    )
    previous_identity = action_identity(initial_action, initial_action_args)
    trace = [
        _trace_row(
            0,
            phase="initial_decision",
            action=initial_action,
            action_args=initial_action_args,
            frame=frame,
            planner_decision=None,
        )
    ]
    state_action_visits: Counter[str] = Counter()
    continuation_steps_executed = 0
    replanning_decisions = 0
    objective_scope_matches = 0
    objective_applications = 0
    objective_quarantines = 0
    stopped_on_terminal = _signature_terminal(
        _signature_payload(trace[-1]["signature"])
    )
    if not stopped_on_terminal:
        for index in range(1, continuation_horizon + 1):
            current_signature = state_signature_from_frame(frame)
            selected, planner_decision = select_scope_safe_objective_action(
                _valid_actions(env),
                game_id=game_id,
                current_state_signature=current_signature,
                objective_index=objective_index,
                family_counts=family_counts,
                concrete_counts=concrete_counts,
                state_action_visits=state_action_visits,
                previous_action_identity=previous_identity,
            )
            if selected is None:
                return {
                    "status": "BLOCKED",
                    "reason": f"no_non_reset_legal_action:{index}",
                    "arm": arm,
                    "continuation_steps_executed": continuation_steps_executed,
                }
            action = str(getattr(selected, "name", ""))
            action_args = dict(getattr(selected, "action_args", {}) or {})
            objective_scope_matches += int(
                bool(planner_decision.get("objective_scope_match", False))
            )
            objective_applications += int(
                bool(planner_decision.get("learned_objective_applied", False))
            )
            objective_quarantines += int(
                bool(planner_decision.get("learned_objective_quarantined", False))
            )
            frame = _step_env_action(env, selected)
            continuation_steps_executed += 1
            replanning_decisions += 1
            identity = action_identity(action, action_args)
            visual_digest = str(planner_decision.get("current_visual_digest", ""))
            state_action_visits[f"{visual_digest}|{identity}"] += 1
            _increment_action_counts(
                action, action_args, concrete_counts, family_counts
            )
            previous_identity = identity
            trace.append(
                _trace_row(
                    index,
                    phase="scope_safe_objective_replanning",
                    action=action,
                    action_args=action_args,
                    frame=frame,
                    planner_decision=planner_decision,
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
        "objective_planner": OBJECTIVE_PLANNER,
        "objective_id": OBJECTIVE_ID,
        "objective_scope": OBJECTIVE_SCOPE,
        "continuation_steps_requested": int(continuation_horizon),
        "continuation_steps_executed": continuation_steps_executed,
        "replanning_decisions": replanning_decisions,
        "objective_scope_matches": objective_scope_matches,
        "objective_applications": objective_applications,
        "objective_quarantines": objective_quarantines,
        "stopped_on_terminal": stopped_on_terminal,
        "future_action_schedule_preselected": False,
        "future_outcome_fields_read_for_planning": [],
        "evaluation_outcomes_used_for_tuning": [],
        "counterfactual_rollouts_performed": 0,
        "all_selected_actions_legal": True,
        "final_action_family_counts": dict(sorted(family_counts.items())),
        "final_concrete_action_counts": dict(sorted(concrete_counts.items())),
        "trace": trace,
    }


def select_scope_safe_objective_action(
    valid_actions: Sequence[Any],
    *,
    game_id: str,
    current_state_signature: str,
    objective_index: Mapping[str, Mapping[str, Any]],
    family_counts: Mapping[str, int],
    concrete_counts: Mapping[str, int],
    state_action_visits: Mapping[str, int],
    previous_action_identity: str,
) -> tuple[Any | None, Dict[str, Any]]:
    """Apply an exact-state learned action or quarantine to the SAGE.8d fallback."""
    state_payload = _signature_payload(current_state_signature)
    visual_digest = str(state_payload.get("digest", ""))
    scope_key = f"{game_id}|{visual_digest}"
    entry = dict(objective_index.get(scope_key, {}) or {})
    legal_by_identity = {
        action_identity(
            str(getattr(action, "name", "")),
            dict(getattr(action, "action_args", {}) or {}),
        ): action
        for action in valid_actions
        if str(getattr(action, "name", ""))
        and str(getattr(action, "name", "")) != "RESET"
    }
    learned = [
        dict(row)
        for row in entry.get("action_candidates", []) or []
        if isinstance(row, Mapping)
    ]
    legal_learned = [
        row
        for row in learned
        if str(row.get("action_identity", "")) in legal_by_identity
    ]
    if entry and legal_learned:
        max_count = max(
            int(row.get("demonstration_count", 0) or 0) for row in legal_learned
        )
        top = [
            row
            for row in legal_learned
            if int(row.get("demonstration_count", 0) or 0) == max_count
        ]
        top_actions = [
            legal_by_identity[str(row.get("action_identity", ""))] for row in top
        ]
        selected, fallback = select_state_conditioned_replanning_action(
            top_actions,
            current_state_signature=current_state_signature,
            family_counts=family_counts,
            concrete_counts=concrete_counts,
            state_action_visits=state_action_visits,
            previous_action_identity=previous_action_identity,
        )
        selected_identity = str(fallback.get("selected_action_identity", ""))
        selected_prior = next(
            row
            for row in top
            if str(row.get("action_identity", "")) == selected_identity
        )
        return selected, {
            **fallback,
            "objective_planner": OBJECTIVE_PLANNER,
            "objective_id": OBJECTIVE_ID,
            "objective_scope": OBJECTIVE_SCOPE,
            "objective_scope_key": scope_key,
            "objective_scope_match": True,
            "learned_objective_applied": True,
            "learned_objective_quarantined": False,
            "objective_decision_reason": OBJECTIVE_APPLIED,
            "learned_candidate_count": len(learned),
            "legal_learned_candidate_count": len(legal_learned),
            "selected_demonstration_count": int(
                selected_prior.get("demonstration_count", 0) or 0
            ),
            "objective_is_arc_progress_truth_claim": False,
            "future_outcome_fields_read": [],
            "evaluation_outcomes_used_for_tuning": [],
            "counterfactual_rollouts_performed": 0,
        }
    selected, fallback = select_state_conditioned_replanning_action(
        valid_actions,
        current_state_signature=current_state_signature,
        family_counts=family_counts,
        concrete_counts=concrete_counts,
        state_action_visits=state_action_visits,
        previous_action_identity=previous_action_identity,
    )
    reason = OBJECTIVE_ACTION_UNAVAILABLE if entry else OBJECTIVE_OUT_OF_SCOPE
    return selected, {
        **fallback,
        "objective_planner": OBJECTIVE_PLANNER,
        "objective_id": OBJECTIVE_ID,
        "objective_scope": OBJECTIVE_SCOPE,
        "objective_scope_key": scope_key,
        "objective_scope_match": bool(entry),
        "learned_objective_applied": False,
        "learned_objective_quarantined": True,
        "objective_decision_reason": reason,
        "learned_candidate_count": len(learned),
        "legal_learned_candidate_count": len(legal_learned),
        "selected_demonstration_count": 0,
        "objective_is_arc_progress_truth_claim": False,
        "future_outcome_fields_read": [],
        "evaluation_outcomes_used_for_tuning": [],
        "counterfactual_rollouts_performed": 0,
    }


def summarize_objective_planner_metrics(
    episodes: Sequence[Mapping[str, Any]],
    objective_model: Mapping[str, Any],
) -> Dict[str, Any]:
    no_replans = sum(
        int(row.get("no_memory_arm", {}).get("replanning_decisions", 0) or 0)
        for row in episodes
    )
    memory_replans = sum(
        int(row.get("with_memory_arm", {}).get("replanning_decisions", 0) or 0)
        for row in episodes
    )
    no_applied = sum(
        int(row.get("no_memory_arm", {}).get("objective_applications", 0) or 0)
        for row in episodes
    )
    memory_applied = sum(
        int(row.get("with_memory_arm", {}).get("objective_applications", 0) or 0)
        for row in episodes
    )
    no_quarantined = sum(
        int(row.get("no_memory_arm", {}).get("objective_quarantines", 0) or 0)
        for row in episodes
    )
    memory_quarantined = sum(
        int(row.get("with_memory_arm", {}).get("objective_quarantines", 0) or 0)
        for row in episodes
    )
    no_terminals = sum(bool(row.get("no_memory_terminal", False)) for row in episodes)
    memory_terminals = sum(
        bool(row.get("with_memory_terminal", False)) for row in episodes
    )
    divergence = sum(
        int(
            row.get("trajectory_comparison", {}).get(
                "divergent_replanning_positions", 0
            )
            or 0
        )
        for row in episodes
    )
    model_summary = dict(objective_model.get("summary", {}) or {})
    return {
        "episodes_per_arm": len(episodes),
        "objective_model_exact_visual_states": int(
            model_summary.get("exact_visual_states", 0) or 0
        ),
        "objective_model_demonstration_transitions": int(
            model_summary.get("demonstration_transitions", 0) or 0
        ),
        "no_memory_replanning_decisions": no_replans,
        "with_memory_replanning_decisions": memory_replans,
        "no_memory_objective_applications": no_applied,
        "with_memory_objective_applications": memory_applied,
        "no_memory_objective_quarantines": no_quarantined,
        "with_memory_objective_quarantines": memory_quarantined,
        "no_memory_objective_coverage_rate": no_applied / no_replans
        if no_replans
        else 0.0,
        "with_memory_objective_coverage_rate": memory_applied / memory_replans
        if memory_replans
        else 0.0,
        "no_memory_terminal_episodes": no_terminals,
        "with_memory_terminal_episodes": memory_terminals,
        "episodes_with_divergent_replanned_trajectories": sum(
            bool(
                row.get("trajectory_comparison", {}).get("trajectories_diverged", False)
            )
            for row in episodes
        ),
        "divergent_replanning_positions": divergence,
        "same_objective_planner_all_episodes": all(
            bool(row.get("same_objective_planner_between_arms", False))
            for row in episodes
        ),
        "all_actions_legal": all(
            bool(row.get("no_memory_arm", {}).get("all_selected_actions_legal", False))
            and bool(
                row.get("with_memory_arm", {}).get("all_selected_actions_legal", False)
            )
            for row in episodes
        ),
        "training_outcome_fields_read": 0,
        "evaluation_outcomes_used_for_training_or_tuning": 0,
        "future_outcome_fields_read_for_planning": [],
        "counterfactual_rollouts_performed": 0,
        "scope_generalization_performed": False,
    }


def build_sage8e_gate(
    specifications: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    objective_model: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
    objective_metrics: Mapping[str, Any],
    *,
    continuation_horizon: int,
) -> Dict[str, bool]:
    expected_hashes = {
        str(row.get("context_snapshot_hash", "")) for row in specifications
    }
    observed_hashes = {str(row.get("context_snapshot_hash", "")) for row in episodes}
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    model_summary = dict(objective_model.get("summary", {}) or {})
    planner_steps = [
        step
        for row in episodes
        for arm_name in ("no_memory_arm", "with_memory_arm")
        for step in list(row.get(arm_name, {}).get("trace", []) or [])[1:]
    ]
    return {
        "all_eleven_exact_contexts_evaluated_once": len(specifications)
        == len(episodes)
        == len(expected_hashes)
        == 11
        and expected_hashes == observed_hashes,
        "objective_model_compiled_from_all_exact_replays": bool(
            model_summary.get("gate_passed", False)
        )
        and int(model_summary.get("source_replays", 0) or 0) == 11
        and int(model_summary.get("source_replays_exactly_verified", 0) or 0) == 11
        and int(model_summary.get("exact_visual_states", 0) or 0) > 0,
        "objective_model_never_read_or_tuned_on_outcomes": int(
            model_summary.get("training_outcome_fields_read", -1) or 0
        )
        == 0
        and int(
            model_summary.get("evaluation_outcomes_used_for_training_or_tuning", -1)
            or 0
        )
        == 0,
        "objective_model_scope_is_exact_and_not_generalized": not bool(
            objective_model.get("scope_generalization_performed", True)
        )
        and int(model_summary.get("fuzzy_scope_entries", -1) or 0) == 0
        and int(model_summary.get("cross_game_scope_entries", -1) or 0) == 0,
        "fixed_positive_objective_closed_loop_horizon": continuation_horizon > 1
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
        "all_prefix_replays_exact_and_paired": all(
            bool(row.get("replay_exact_both_arms", False))
            and bool(row.get("same_prefix_between_arms", False))
            for row in episodes
        ),
        "same_objective_planner_used_between_arms": all(
            bool(row.get("same_objective_planner_between_arms", False))
            and not bool(row.get("future_action_schedule_preselected", True))
            for row in episodes
        ),
        "learned_objective_was_exercised_and_out_of_scope_was_quarantined": int(
            objective_metrics.get("no_memory_objective_applications", 0) or 0
        )
        + int(objective_metrics.get("with_memory_objective_applications", 0) or 0)
        > 0
        and int(objective_metrics.get("no_memory_objective_quarantines", 0) or 0)
        + int(objective_metrics.get("with_memory_objective_quarantines", 0) or 0)
        > 0,
        "every_continuation_action_was_replanned_from_current_state": bool(
            planner_steps
        )
        and all(
            str(step.get("phase", "")) == "scope_safe_objective_replanning"
            and bool(
                step.get("planner_decision", {}).get(
                    "current_state_observation_used", False
                )
            )
            for step in planner_steps
        ),
        "planner_never_read_future_outcomes_or_simulated": all(
            not list(
                step.get("planner_decision", {}).get("future_outcome_fields_read", [])
                or []
            )
            and not list(
                step.get("planner_decision", {}).get(
                    "evaluation_outcomes_used_for_tuning", []
                )
                or []
            )
            and int(
                step.get("planner_decision", {}).get(
                    "counterfactual_rollouts_performed", -1
                )
                or 0
            )
            == 0
            for step in planner_steps
        ),
        "all_objective_closed_loop_actions_legal": bool(
            objective_metrics.get("all_actions_legal", False)
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


def summarize_sage8e(
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    secondary_metrics: Mapping[str, Any],
    objective_metrics: Mapping[str, Any],
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
        outcome = SAGE8E_ARC_GAIN
    elif primary_regressed:
        outcome = SAGE8E_ARC_REGRESSION
    elif secondary_improved:
        outcome = SAGE8E_LOCAL_ONLY
    else:
        outcome = SAGE8E_NO_GAIN
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
        "objective_model_exact_visual_states": int(
            objective_metrics.get("objective_model_exact_visual_states", 0) or 0
        ),
        "objective_model_demonstration_transitions": int(
            objective_metrics.get("objective_model_demonstration_transitions", 0) or 0
        ),
        "no_memory_replanning_decisions": int(
            objective_metrics.get("no_memory_replanning_decisions", 0) or 0
        ),
        "with_memory_replanning_decisions": int(
            objective_metrics.get("with_memory_replanning_decisions", 0) or 0
        ),
        "no_memory_objective_applications": int(
            objective_metrics.get("no_memory_objective_applications", 0) or 0
        ),
        "with_memory_objective_applications": int(
            objective_metrics.get("with_memory_objective_applications", 0) or 0
        ),
        "no_memory_objective_quarantines": int(
            objective_metrics.get("no_memory_objective_quarantines", 0) or 0
        ),
        "with_memory_objective_quarantines": int(
            objective_metrics.get("with_memory_objective_quarantines", 0) or 0
        ),
        "no_memory_objective_coverage_rate": float(
            objective_metrics.get("no_memory_objective_coverage_rate", 0.0) or 0.0
        ),
        "with_memory_objective_coverage_rate": float(
            objective_metrics.get("with_memory_objective_coverage_rate", 0.0) or 0.0
        ),
        "episodes_with_divergent_replanned_trajectories": int(
            objective_metrics.get("episodes_with_divergent_replanned_trajectories", 0)
            or 0
        ),
        "divergent_replanning_positions": int(
            objective_metrics.get("divergent_replanning_positions", 0) or 0
        ),
        "no_memory_terminal_episodes": int(
            objective_metrics.get("no_memory_terminal_episodes", 0) or 0
        ),
        "with_memory_terminal_episodes": int(
            objective_metrics.get("with_memory_terminal_episodes", 0) or 0
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
        "objective_is_arc_progress_truth_claim": False,
        "local_signal_counted_as_arc_progress": False,
        "training_or_tuning_used_evaluation_outcomes": False,
        "future_outcomes_used_for_planning": False,
        "counterfactual_rollouts_performed": 0,
        "truth_reevaluations": 0,
        "support_counted": 0,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": gate_passed,
        "outcome_status": outcome,
    }


def validate_sage8e_sources(
    source_sage8d: Mapping[str, Any],
    source_sage8c: Mapping[str, Any],
    source_sage8b: Mapping[str, Any],
    policy_source: Mapping[str, Any],
    a34_2: Mapping[str, Any],
    a34_3: Mapping[str, Any],
) -> None:
    validate_sage8d_sources(source_sage8c, source_sage8b, policy_source, a34_2, a34_3)
    summary = dict(source_sage8d.get("summary", {}) or {})
    if (
        str(source_sage8d.get("config", {}).get("schema_version", ""))
        != SAGE8D_SCHEMA_VERSION
        or str(source_sage8d.get("outcome_status", "")) != SAGE8D_LOCAL_ONLY
        or str(source_sage8d.get("truth_status", "")) != SAGE8D_TRUTH_STATUS
        or str(source_sage8d.get("status", "")) != "EVALUATED"
        or not bool(source_sage8d.get("closed_loop_live_rollout_performed", False))
        or not bool(source_sage8d.get("state_conditioned_replanning_performed", False))
        or not bool(summary.get("gate_passed", False))
        or bool(summary.get("primary_arc_progress_improved", True))
        or bool(summary.get("primary_arc_progress_regressed", True))
        or bool(summary.get("future_outcomes_used_for_planning", True))
        or int(summary.get("counterfactual_rollouts_performed", -1) or 0) != 0
        or not all(bool(value) for value in source_sage8d.get("gate", {}).values())
    ):
        raise ValueError(
            "SAGE.8e requires the completed local-only outcome-blind SAGE.8d evaluation"
        )
    source_hashes = {
        str(row.get("context_snapshot_hash", ""))
        for row in source_sage8d.get("paired_rollouts", []) or []
    }
    policy_hashes = {
        str(value)
        for entry in policy_source.get("policy_entries", []) or []
        for value in entry.get("context_snapshot_hashes", []) or []
    }
    if len(source_hashes) != 11 or source_hashes != policy_hashes:
        raise ValueError("SAGE.8d and SAGE.8a context scopes must match exactly")


def write_sage8e_relational_memory_objective_closed_loop_evaluation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_SAGE8E_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_EVALUATION_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _increment_action_counts(
    action: str,
    action_args: Mapping[str, Any],
    concrete_counts: Counter[str],
    family_counts: Counter[str],
) -> None:
    family_counts[str(action)] += 1
    concrete_counts[action_identity(str(action), action_args)] += 1


def _validate_objective_closed_loop_config(horizon: int) -> None:
    if int(horizon) <= 1:
        raise ValueError("SAGE.8e continuation_horizon must be greater than one")


def _visual_digest(frame: Any) -> str:
    return str(_signature_payload(state_signature_from_frame(frame)).get("digest", ""))


def _trace_row(
    index: int,
    *,
    phase: str,
    action: str,
    action_args: Mapping[str, Any],
    frame: Any,
    planner_decision: Mapping[str, Any] | None,
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
        "planner_decision": dict(planner_decision or {}),
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
        "--sage8d",
        default=str(DEFAULT_SAGE8D_RELATIONAL_MEMORY_CLOSED_LOOP_EVALUATION_PATH),
    )
    parser.add_argument(
        "--sage8c",
        default=str(DEFAULT_SAGE8C_RELATIONAL_MEMORY_MULTI_ACTION_EVALUATION_PATH),
    )
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
        "--out",
        default=str(
            DEFAULT_SAGE8E_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_EVALUATION_PATH
        ),
    )
    args = parser.parse_args(argv)
    payload = run_sage8e_relational_memory_objective_closed_loop_evaluation(
        sage8d_path=args.sage8d,
        sage8c_path=args.sage8c,
        sage8b_path=args.sage8b,
        policy_path=args.policy,
        a34_2_path=args.a34_2,
        a34_3_path=args.a34_3,
        environments_dir=args.environments_dir,
        continuation_horizon=args.continuation_horizon,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
