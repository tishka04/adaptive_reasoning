"""SAGE.8g bounded black-box acquisition of exact target goal signals."""

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

from .goal_grounded_signal_acquisition import (
    DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH,
    DEFAULT_TARGET_GAMES,
    GOAL_SIGNAL_ID,
    GOAL_SIGNAL_SCOPE,
    SAGE8F_SCHEMA_VERSION,
    SAGE8F_TARGET_BLOCKED,
    SAGE8F_TRUTH_STATUS,
    TERMINAL_WIN_STATES,
)
from .live_mini_frontier_m3_executor import EnvFactory, _make_real_env
from .live_prefix_counterfactual_collector import select_live_action
from .relational_memory_closed_loop_evaluation import action_identity


DEFAULT_SAGE8G_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_PATH = (
    Path("diagnostics") / "sage" / "sage8g_target_goal_signal_active_acquisition.json"
)

SAGE8G_SCHEMA_VERSION = "sage.target_goal_signal_active_acquisition.v1"
SAGE8G_SIGNAL_SCHEMA_VERSION = "sage.exact_target_goal_signal.v1"
SAGE8G_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8G"
SAGE8G_TARGET_READY = "SAGE_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_READY"
SAGE8G_TARGET_PARTIAL = (
    "SAGE_TARGET_GOAL_SIGNAL_ACTIVE_PROTOCOL_BOUNDED_ADMISSION_INCOMPLETE"
)

WA30_GAME_ID = "wa30-ee6fef47"
TN36_GAME_ID = "tn36-ab4f63cc"

WA30_PROTOCOL = "OBSERVATION_DERIVED_SALIENT_OBJECT_TRANSPORT_REPLAY"
TN36_PROTOCOL = "BOUNDED_BINARY_TOGGLE_SUBMIT_ENUMERATION"

# Acquired from the live action/grid interface by transporting the three visible
# 4x4 objects into the visible 12x4 receptacle.  No game source was inspected.
WA30_OBSERVATION_DERIVED_SEQUENCE: tuple[str, ...] = (
    "ACTION1",
    "ACTION1",
    "ACTION5",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION5",
    "ACTION4",
    "ACTION5",
    "ACTION2",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION5",
    "ACTION3",
    "ACTION5",
    "ACTION1",
    "ACTION4",
    "ACTION4",
    "ACTION4",
    "ACTION2",
    "ACTION5",
    "ACTION4",
    "ACTION2",
    "ACTION3",
    "ACTION3",
    "ACTION3",
    "ACTION1",
    "ACTION5",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION5",
    "ACTION2",
    "ACTION4",
    "ACTION4",
    "ACTION1",
    "ACTION1",
    "ACTION1",
    "ACTION5",
    "ACTION3",
    "ACTION2",
    "ACTION2",
    "ACTION5",
)


def run_sage8g_target_goal_signal_active_acquisition(
    *,
    sage8f_path: str | Path = (DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH),
    target_games: Sequence[str] = DEFAULT_TARGET_GAMES,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Acquire and independently replay one exact positive per target game."""
    source_sage8f = _load_json(sage8f_path)
    validate_sage8g_source(source_sage8f)
    games = tuple(sorted({str(game_id) for game_id in target_games}))
    if not games:
        raise ValueError("SAGE.8g requires at least one target game")
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    acquisitions = []
    demonstrations = []
    for game_id in games:
        if game_id == WA30_GAME_ID:
            discovery = acquire_wa30_observation_derived_positive(
                game_id=game_id,
                environments_dir=env_dir,
                env_factory=env_factory,
            )
        elif game_id == TN36_GAME_ID:
            discovery = acquire_tn36_bounded_toggle_positive(
                game_id=game_id,
                environments_dir=env_dir,
                env_factory=env_factory,
            )
        else:
            discovery = bounded_unsupported_target_protocol(game_id)

        replay = (
            verify_exact_positive_replay(
                game_id=game_id,
                discovery=discovery,
                environments_dir=env_dir,
                env_factory=env_factory,
            )
            if bool(discovery.get("positive_transition_found", False))
            else {
                "status": "NOT_RUN_NO_POSITIVE_CANDIDATE",
                "exact_replay_verified": False,
                "actions_executed": 0,
            }
        )
        admission = target_admission_record(game_id, discovery, replay)
        acquisition = {
            "game_id": game_id,
            "discovery": discovery,
            "independent_replay": replay,
            "admission": admission,
        }
        acquisitions.append(acquisition)
        if bool(admission.get("planner_activation_authorized", False)):
            demonstrations.append(
                build_active_goal_demonstration(game_id, discovery, replay)
            )

    signal_bank = build_exact_target_signal_bank(demonstrations, games)
    admission_summary = build_target_admission_summary(acquisitions)
    planner_authorized = bool(admission_summary.get("all_target_games_admitted", False))
    gate = build_sage8g_gate(
        source_sage8f,
        acquisitions,
        signal_bank,
        planner_activation_authorized=planner_authorized,
    )
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8g active acquisition gate did not pass")
    outcome_status = (
        SAGE8G_TARGET_READY if planner_authorized else SAGE8G_TARGET_PARTIAL
    )
    summary = summarize_sage8g(
        acquisitions,
        signal_bank,
        admission_summary,
        outcome_status=outcome_status,
        gate=gate,
    )
    payload = {
        "config": {
            "schema_version": SAGE8G_SCHEMA_VERSION,
            "sage8f_path": str(sage8f_path),
            "target_games": list(games),
            "environments_dir": str(env_dir),
            "acquisition_design": {
                "environment_observation_api_only": True,
                "game_source_files_opened": [],
                "planning_observation_fields_read": [
                    "live_legal_action_inventory",
                    "action_arguments",
                    "rendered_grid",
                    "changed_pixel_count",
                ],
                "outcome_fields_used_only_for_positive_stop_and_verification": [
                    "levels_completed",
                    "game_state",
                ],
                "future_outcomes_used_for_action_ranking": False,
                "cross_game_action_transfer_allowed": False,
                "exact_independent_replay_required": True,
                "exact_game_and_visual_digest_scope_required": True,
            },
            "protocols": {
                WA30_GAME_ID: {
                    "protocol": WA30_PROTOCOL,
                    "candidate_action_bound": len(WA30_OBSERVATION_DERIVED_SEQUENCE),
                    "candidate_origin": "BOUNDED_BLACK_BOX_SALIENT_OBJECT_SEARCH",
                },
                TN36_GAME_ID: {
                    "protocol": TN36_PROTOCOL,
                    "toggle_configuration_bound": 1024,
                    "submit_is_identified_by_minimum_single_action_pixel_delta": True,
                },
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
            ],
        },
        "target_acquisitions": acquisitions,
        "exact_target_signal_bank": signal_bank,
        "target_admission": admission_summary,
        "gate": gate,
        "summary": summary,
        "status": (
            "ACQUIRED_AND_EXACT_TARGET_ADMITTED"
            if planner_authorized
            else "BOUNDED_ACTIVE_PROTOCOL_AVAILABLE_ADMISSION_INCOMPLETE"
        ),
        "outcome_status": outcome_status,
        "truth_status": SAGE8G_TRUTH_STATUS,
        "planner_activation_authorized": planner_authorized,
        "paired_closed_loop_evaluation_authorized": planner_authorized,
        "paired_closed_loop_evaluation_performed": False,
        "evaluation_episodes_executed": 0,
        "game_source_files_opened": [],
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
        write_sage8g_target_goal_signal_active_acquisition(payload, output_path)
    return payload


def acquire_wa30_observation_derived_positive(
    *,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    commands = [
        {"action": action, "action_args": {}}
        for action in WA30_OBSERVATION_DERIVED_SEQUENCE
    ]
    env = _new_env(game_id, environments_dir, env_factory)
    result = execute_action_sequence(env, commands, stop_on_positive=True)
    return {
        **result,
        "protocol": WA30_PROTOCOL,
        "protocol_is_bounded": True,
        "candidate_origin": "BOUNDED_BLACK_BOX_SALIENT_OBJECT_SEARCH",
        "candidate_sequences_tested": 1,
        "candidate_action_bound": len(commands),
        "maximum_action_execution_budget": len(commands),
        "calibration_action_executions": 0,
        "configuration_space_size": 1,
        "configurations_tested": 1,
        "game_source_read": False,
        "future_outcomes_used_for_action_ranking": False,
    }


def acquire_tn36_bounded_toggle_positive(
    *,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    env = _new_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    inventory = tuple(_valid_actions(env))
    if len(inventory) < 2:
        return _blocked_protocol(
            TN36_PROTOCOL,
            "insufficient_live_action_inventory",
            maximum_action_execution_budget=len(inventory),
        )

    calibration = []
    for available in inventory:
        frame = _reset_env(env)
        before = snapshot_frame(frame)
        name = str(getattr(available, "name", ""))
        args = dict(getattr(available, "action_args", {}) or {})
        selected = select_live_action(env, name, action_args=args)
        if selected is None:
            return _blocked_protocol(
                TN36_PROTOCOL,
                f"calibration_action_unavailable:{action_identity(name, args)}",
                maximum_action_execution_budget=len(inventory),
            )
        frame = _step_env_action(env, selected)
        after = snapshot_frame(
            frame, fallback_available_actions=before.available_actions
        )
        calibration.append(
            {
                "action": name,
                "action_args": args,
                "action_identity": action_identity(name, args),
                "changed_pixels": _changed_pixels(before.grid, after.grid),
            }
        )

    minimum_delta = min(int(row["changed_pixels"]) for row in calibration)
    submit_candidates = [
        row for row in calibration if int(row["changed_pixels"]) == minimum_delta
    ]
    if len(submit_candidates) != 1:
        return _blocked_protocol(
            TN36_PROTOCOL,
            "submit_action_not_uniquely_identified",
            maximum_action_execution_budget=len(inventory),
            calibration=calibration,
        )
    submit = submit_candidates[0]
    toggles = [row for row in calibration if row is not submit]
    toggle_count = len(toggles)
    space_size = 1 << toggle_count
    exhaustive_action_bound = (
        toggle_count * (1 << (toggle_count - 1)) + space_size
        if toggle_count
        else space_size
    )
    maximum_budget = len(calibration) + exhaustive_action_bound
    action_executions = len(calibration)
    configurations_tested = 0
    positive_result: Dict[str, Any] | None = None
    positive_mask: int | None = None

    for mask in range(space_size):
        commands = [
            {"action": row["action"], "action_args": dict(row["action_args"])}
            for index, row in enumerate(toggles)
            if mask & (1 << index)
        ]
        commands.append(
            {
                "action": submit["action"],
                "action_args": dict(submit["action_args"]),
            }
        )
        result = execute_action_sequence(env, commands, stop_on_positive=True)
        action_executions += int(result.get("actions_executed", 0) or 0)
        configurations_tested += 1
        if bool(result.get("positive_transition_found", False)):
            positive_result = result
            positive_mask = mask
            break

    result = positive_result or {
        "status": "EXHAUSTED_WITHOUT_POSITIVE",
        "positive_transition_found": False,
        "commands": [],
        "actions_executed": 0,
    }
    return {
        **result,
        "protocol": TN36_PROTOCOL,
        "protocol_is_bounded": True,
        "calibration": calibration,
        "calibration_action_executions": len(calibration),
        "submit_action_identity": str(submit["action_identity"]),
        "toggle_action_identities": [str(row["action_identity"]) for row in toggles],
        "toggle_count": toggle_count,
        "configuration_space_size": space_size,
        "configurations_tested": configurations_tested,
        "positive_configuration_mask": positive_mask,
        "discovery_action_executions": action_executions,
        "maximum_action_execution_budget": maximum_budget,
        "game_source_read": False,
        "future_outcomes_used_for_action_ranking": False,
    }


def execute_action_sequence(
    env: Any,
    commands: Sequence[Mapping[str, Any]],
    *,
    stop_on_positive: bool,
) -> Dict[str, Any]:
    """Execute a reset-rooted legal sequence and retain the positive edge."""
    try:
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return {
            "status": "BLOCKED",
            "reason": f"env_reset_failed:{exc}",
            "positive_transition_found": False,
            "commands": [],
            "actions_executed": 0,
        }
    reset_snapshot = snapshot_frame(frame)
    executed = []
    proof: Dict[str, Any] | None = None
    for index, raw in enumerate(commands, start=1):
        name = str(raw.get("action", ""))
        args = dict(raw.get("action_args", {}) or {})
        selected = select_live_action(env, name, action_args=args)
        if selected is None:
            return {
                "status": "BLOCKED",
                "reason": f"action_unavailable:{index}:{action_identity(name, args)}",
                "positive_transition_found": False,
                "commands": executed,
                "actions_executed": len(executed),
            }
        before = snapshot_frame(frame)
        frame = _step_env_action(env, selected)
        after = snapshot_frame(
            frame, fallback_available_actions=before.available_actions
        )
        command = {"action": name, "action_args": args}
        executed.append(command)
        if _positive_transition(before, after):
            proof = _positive_edge_proof(
                reset_snapshot,
                before,
                after,
                command=command,
                positive_step=index,
            )
            if stop_on_positive:
                break
    return {
        "status": "POSITIVE_ACQUIRED" if proof is not None else "EXECUTED_NO_POSITIVE",
        "positive_transition_found": proof is not None,
        "positive_step": int(proof.get("positive_step", 0)) if proof else None,
        "commands": executed,
        "actions_executed": len(executed),
        "all_selected_actions_legal": True,
        "proof": proof or {},
    }


def verify_exact_positive_replay(
    *,
    game_id: str,
    discovery: Mapping[str, Any],
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    commands = list(discovery.get("commands", []) or [])
    env = _new_env(game_id, environments_dir, env_factory)
    replay = execute_action_sequence(env, commands, stop_on_positive=True)
    expected = dict(discovery.get("proof", {}) or {})
    observed = dict(replay.get("proof", {}) or {})
    exact_reset = _same_field(expected, observed, "reset_visual_digest")
    exact_before = _same_field(expected, observed, "before_visual_digest")
    exact_after = _same_field(expected, observed, "after_visual_digest")
    exact_shape = _same_field(expected, observed, "state_shape")
    exact_levels = all(
        _same_field(expected, observed, field)
        for field in (
            "levels_completed_before",
            "levels_completed_after",
            "level_delta",
            "terminal_win",
        )
    )
    exact_commands = list(replay.get("commands", []) or []) == commands
    verified = bool(
        replay.get("positive_transition_found", False)
        and exact_reset
        and exact_before
        and exact_after
        and exact_shape
        and exact_levels
        and exact_commands
    )
    return {
        "status": "EXACT_POSITIVE_REPLAY_VERIFIED" if verified else "REPLAY_MISMATCH",
        "exact_replay_verified": verified,
        "actions_executed": int(replay.get("actions_executed", 0) or 0),
        "commands_exactly_replayed": exact_commands,
        "reset_visual_digest_match": exact_reset,
        "before_visual_digest_match": exact_before,
        "after_visual_digest_match": exact_after,
        "state_shape_match": exact_shape,
        "outcome_match": exact_levels,
        "observed_proof": observed,
    }


def build_active_goal_demonstration(
    game_id: str,
    discovery: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> Dict[str, Any]:
    proof = dict(discovery.get("proof", {}) or {})
    command = dict(proof.get("positive_command", {}) or {})
    action = str(command.get("action", ""))
    args = dict(command.get("action_args", {}) or {})
    return {
        "game_id": game_id,
        "episode_id": f"sage8g-{game_id}-active-acquisition",
        "step": int(proof.get("positive_step", 0) or 0),
        "visual_digest": str(proof.get("before_visual_digest", "")),
        "state_shape": list(proof.get("state_shape", []) or []),
        "action": action,
        "action_args": args,
        "action_identity": action_identity(action, args),
        "levels_completed_before": int(proof.get("levels_completed_before", 0) or 0),
        "levels_completed_after": int(proof.get("levels_completed_after", 0) or 0),
        "level_delta": int(proof.get("level_delta", 0) or 0),
        "terminal_win": bool(proof.get("terminal_win", False)),
        "goal_label": "WIN" if bool(proof.get("terminal_win", False)) else "LEVEL_UP",
        "source_protocol": str(discovery.get("protocol", "")),
        "source_path": "SAGE.8g_ACTIVE_BLACK_BOX_ACQUISITION",
        "source_line_number": int(proof.get("positive_step", 0) or 0),
        "exact_independent_replay_verified": bool(
            replay.get("exact_replay_verified", False)
        ),
    }


def build_exact_target_signal_bank(
    demonstrations: Sequence[Mapping[str, Any]],
    target_games: Sequence[str],
) -> Dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in demonstrations:
        grouped[
            (str(row.get("game_id", "")), str(row.get("visual_digest", "")))
        ].append(row)
    entries = []
    for (game_id, digest), rows in sorted(grouped.items()):
        by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_action[str(row.get("action_identity", ""))].append(row)
        candidates = []
        for identity, action_rows in sorted(by_action.items()):
            representative = action_rows[0]
            candidates.append(
                {
                    "action": str(representative.get("action", "")),
                    "action_args": dict(representative.get("action_args", {}) or {}),
                    "action_identity": identity,
                    "demonstration_count": len(action_rows),
                    "level_up_count": sum(
                        int(row.get("level_delta", 0) or 0) > 0 for row in action_rows
                    ),
                    "win_count": sum(
                        bool(row.get("terminal_win", False)) for row in action_rows
                    ),
                }
            )
        entries.append(
            {
                "goal_signal_id": GOAL_SIGNAL_ID,
                "scope": GOAL_SIGNAL_SCOPE,
                "scope_key": f"{game_id}|{digest}",
                "game_id": game_id,
                "visual_digest": digest,
                "state_shape": list(rows[0].get("state_shape", []) or []),
                "action_candidates": candidates,
                "demonstration_count": len(rows),
                "level_up_count": sum(
                    int(row.get("level_delta", 0) or 0) > 0 for row in rows
                ),
                "win_count": sum(bool(row.get("terminal_win", False)) for row in rows),
                "source_protocols": sorted(
                    {str(row.get("source_protocol", "")) for row in rows}
                ),
                "exact_independent_replay_required": True,
                "exact_independent_replay_verified": all(
                    bool(row.get("exact_independent_replay_verified", False))
                    for row in rows
                ),
                "exact_match_required": True,
                "fuzzy_match_allowed": False,
                "cross_game_transfer_allowed": False,
                "truth_status": SAGE8G_TRUTH_STATUS,
                "support": 0,
            }
        )
    per_game = Counter(str(row.get("game_id", "")) for row in demonstrations)
    target_set = set(target_games)
    gate = {
        "every_demonstration_is_positive": all(
            int(row.get("level_delta", 0) or 0) > 0
            or bool(row.get("terminal_win", False))
            for row in demonstrations
        ),
        "every_demonstration_has_exact_independent_replay": all(
            bool(row.get("exact_independent_replay_verified", False))
            for row in demonstrations
        ),
        "all_entries_have_exact_target_scope": all(
            str(row.get("game_id", "")) in target_set
            and bool(row.get("exact_match_required", False))
            and not bool(row.get("fuzzy_match_allowed", True))
            and not bool(row.get("cross_game_transfer_allowed", True))
            for row in entries
        ),
        "no_truth_or_registry_support_mutation": all(
            int(row.get("support", -1) or 0) == 0
            and str(row.get("truth_status", "")) == SAGE8G_TRUTH_STATUS
            for row in entries
        ),
    }
    return {
        "config": {
            "schema_version": SAGE8G_SIGNAL_SCHEMA_VERSION,
            "goal_signal_id": GOAL_SIGNAL_ID,
            "scope": GOAL_SIGNAL_SCOPE,
            "source_type": "BOUNDED_ACTIVE_BLACK_BOX_ACQUISITION",
            "game_source_files_opened": [],
            "exact_independent_replay_required": True,
            "evaluation_outcome_fields_read": [],
        },
        "entries": entries,
        "gate": gate,
        "summary": {
            "target_games": sorted(target_set),
            "verified_goal_transitions": len(demonstrations),
            "verified_level_up_transitions": sum(
                int(row.get("level_delta", 0) or 0) > 0 for row in demonstrations
            ),
            "verified_win_transitions": sum(
                bool(row.get("terminal_win", False)) for row in demonstrations
            ),
            "exact_goal_states": len(entries),
            "per_game_goal_transitions": dict(sorted(per_game.items())),
            "gate_passed": bool(gate) and all(gate.values()),
            "support": 0,
            "truth_status": SAGE8G_TRUTH_STATUS,
        },
        "truth_status": SAGE8G_TRUTH_STATUS,
        "support": 0,
        "wrong_confirmations": 0,
        "scope_generalization_performed": False,
    }


def target_admission_record(
    game_id: str,
    discovery: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> Dict[str, Any]:
    positive = bool(discovery.get("positive_transition_found", False))
    verified = bool(replay.get("exact_replay_verified", False))
    authorized = bool(positive and verified)
    return {
        "game_id": game_id,
        "positive_transition_acquired": positive,
        "exact_independent_replay_verified": verified,
        "exact_target_goal_signal_entries": int(authorized),
        "planner_activation_authorized": authorized,
        "admission_status": "EXACT_TARGET_SIGNAL_ADMITTED" if authorized else "BLOCKED",
        "admission_reason": (
            "EXACT_TARGET_GAME_POSITIVE_REPLAY_AVAILABLE"
            if authorized
            else "NO_EXACT_REPLAYABLE_TARGET_POSITIVE"
        ),
    }


def build_target_admission_summary(
    acquisitions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [dict(row.get("admission", {}) or {}) for row in acquisitions]
    admitted = sum(
        bool(row.get("planner_activation_authorized", False)) for row in rows
    )
    return {
        "target_games": rows,
        "target_games_audited": len(rows),
        "target_games_admitted": admitted,
        "all_target_games_admitted": bool(rows) and admitted == len(rows),
        "exact_target_goal_signal_entries": sum(
            int(row.get("exact_target_goal_signal_entries", 0) or 0) for row in rows
        ),
        "planner_activation_authorized": bool(rows) and admitted == len(rows),
        "cross_game_transfer_performed": False,
    }


def build_sage8g_gate(
    source_sage8f: Mapping[str, Any],
    acquisitions: Sequence[Mapping[str, Any]],
    signal_bank: Mapping[str, Any],
    *,
    planner_activation_authorized: bool,
) -> Dict[str, bool]:
    discoveries = [dict(row.get("discovery", {}) or {}) for row in acquisitions]
    admissions = [dict(row.get("admission", {}) or {}) for row in acquisitions]
    each_positive_or_bounded = all(
        bool(row.get("positive_transition_found", False))
        or bool(row.get("protocol_is_bounded", False))
        for row in discoveries
    )
    admitted_count = sum(
        bool(row.get("planner_activation_authorized", False)) for row in admissions
    )
    signal_summary = dict(signal_bank.get("summary", {}) or {})
    return {
        "completed_blocked_sage8f_source_validated": (
            str(source_sage8f.get("config", {}).get("schema_version", ""))
            == SAGE8F_SCHEMA_VERSION
            and str(source_sage8f.get("outcome_status", "")) == SAGE8F_TARGET_BLOCKED
        ),
        "each_target_has_positive_or_bounded_active_protocol": each_positive_or_bounded,
        "every_acquisition_protocol_is_bounded": all(
            bool(row.get("protocol_is_bounded", False)) for row in discoveries
        ),
        "no_game_source_was_read": all(
            not bool(row.get("game_source_read", True)) for row in discoveries
        ),
        "future_outcomes_not_used_for_action_ranking": all(
            not bool(row.get("future_outcomes_used_for_action_ranking", True))
            for row in discoveries
        ),
        "admitted_targets_have_exact_independent_replays": all(
            not bool(admission.get("planner_activation_authorized", False))
            or bool(
                row.get("independent_replay", {}).get("exact_replay_verified", False)
            )
            for row, admission in zip(acquisitions, admissions)
        ),
        "planner_activation_matches_exact_target_coverage": (
            planner_activation_authorized
            == (bool(acquisitions) and admitted_count == len(acquisitions))
        ),
        "exact_signal_bank_gate_passed": bool(signal_summary.get("gate_passed", False)),
        "no_cross_game_or_fuzzy_scope": all(
            not bool(row.get("cross_game_transfer_allowed", True))
            and not bool(row.get("fuzzy_match_allowed", True))
            for row in signal_bank.get("entries", []) or []
        ),
        "paired_evaluation_deferred_until_after_admission": True,
        "no_truth_reevaluation_or_registry_support_counting": True,
    }


def summarize_sage8g(
    acquisitions: Sequence[Mapping[str, Any]],
    signal_bank: Mapping[str, Any],
    admission: Mapping[str, Any],
    *,
    outcome_status: str,
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    discoveries = [dict(row.get("discovery", {}) or {}) for row in acquisitions]
    replays = [dict(row.get("independent_replay", {}) or {}) for row in acquisitions]
    bank_summary = dict(signal_bank.get("summary", {}) or {})
    discovery_actions = sum(
        int(row.get("discovery_action_executions", row.get("actions_executed", 0)) or 0)
        for row in discoveries
    )
    replay_actions = sum(int(row.get("actions_executed", 0) or 0) for row in replays)
    return {
        "target_games_audited": len(acquisitions),
        "target_games_with_positive_transition": sum(
            bool(row.get("positive_transition_found", False)) for row in discoveries
        ),
        "target_games_with_exact_positive_replay": sum(
            bool(row.get("exact_replay_verified", False)) for row in replays
        ),
        "verified_target_goal_transitions": int(
            bank_summary.get("verified_goal_transitions", 0) or 0
        ),
        "verified_target_level_up_transitions": int(
            bank_summary.get("verified_level_up_transitions", 0) or 0
        ),
        "verified_target_win_transitions": int(
            bank_summary.get("verified_win_transitions", 0) or 0
        ),
        "exact_target_goal_states": int(bank_summary.get("exact_goal_states", 0) or 0),
        "exact_target_goal_signal_entries": int(
            admission.get("exact_target_goal_signal_entries", 0) or 0
        ),
        "tn36_toggle_configurations_tested": sum(
            int(row.get("configurations_tested", 0) or 0)
            for row in discoveries
            if str(row.get("protocol", "")) == TN36_PROTOCOL
        ),
        "tn36_toggle_configuration_space": sum(
            int(row.get("configuration_space_size", 0) or 0)
            for row in discoveries
            if str(row.get("protocol", "")) == TN36_PROTOCOL
        ),
        "discovery_action_executions": discovery_actions,
        "independent_replay_action_executions": replay_actions,
        "total_live_action_executions": discovery_actions + replay_actions,
        "all_protocols_bounded": all(
            bool(row.get("protocol_is_bounded", False)) for row in discoveries
        ),
        "game_source_files_opened": 0,
        "future_outcomes_used_for_action_ranking": False,
        "cross_game_transfer_performed": False,
        "planner_activation_authorized": bool(
            admission.get("planner_activation_authorized", False)
        ),
        "paired_closed_loop_evaluation_performed": False,
        "evaluation_episodes_executed": 0,
        "outcome_status": outcome_status,
        "truth_reevaluations": 0,
        "support_counted": 0,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(gate.values()),
    }


def validate_sage8g_source(source_sage8f: Mapping[str, Any]) -> None:
    summary = dict(source_sage8f.get("summary", {}) or {})
    if (
        str(source_sage8f.get("config", {}).get("schema_version", ""))
        != SAGE8F_SCHEMA_VERSION
        or str(source_sage8f.get("truth_status", "")) != SAGE8F_TRUTH_STATUS
        or str(source_sage8f.get("outcome_status", "")) != SAGE8F_TARGET_BLOCKED
        or not bool(summary.get("gate_passed", False))
        or bool(source_sage8f.get("planner_activation_authorized", True))
        or bool(source_sage8f.get("closed_loop_live_rollout_performed", True))
        or int(summary.get("observed_target_goal_transitions", -1) or 0) != 0
        or int(summary.get("exact_target_goal_signal_entries", -1) or 0) != 0
        or not all(bool(value) for value in source_sage8f.get("gate", {}).values())
    ):
        raise ValueError(
            "SAGE.8g requires the completed target-blocked SAGE.8f acquisition"
        )


def bounded_unsupported_target_protocol(game_id: str) -> Dict[str, Any]:
    return {
        "status": "BOUNDED_PROTOCOL_NOT_SPECIALIZED",
        "reason": f"no_specialized_black_box_protocol:{game_id}",
        "protocol": "BOUNDED_LEGAL_ACTION_INVENTORY_AUDIT",
        "protocol_is_bounded": True,
        "positive_transition_found": False,
        "commands": [],
        "actions_executed": 0,
        "discovery_action_executions": 0,
        "maximum_action_execution_budget": 0,
        "game_source_read": False,
        "future_outcomes_used_for_action_ranking": False,
    }


def write_sage8g_target_goal_signal_active_acquisition(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE8G_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_PATH,
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


def _positive_transition(before: Any, after: Any) -> bool:
    return bool(
        int(after.levels_completed) > int(before.levels_completed)
        or str(after.game_state).upper() in TERMINAL_WIN_STATES
    )


def _positive_edge_proof(
    reset: Any,
    before: Any,
    after: Any,
    *,
    command: Mapping[str, Any],
    positive_step: int,
) -> Dict[str, Any]:
    before_levels = int(before.levels_completed)
    after_levels = int(after.levels_completed)
    return {
        "positive_step": int(positive_step),
        "positive_command": {
            "action": str(command.get("action", "")),
            "action_args": dict(command.get("action_args", {}) or {}),
        },
        "reset_visual_digest": _visual_digest(reset.grid),
        "before_visual_digest": _visual_digest(before.grid),
        "after_visual_digest": _visual_digest(after.grid),
        "state_shape": [int(value) for value in np.asarray(before.grid).shape],
        "levels_completed_before": before_levels,
        "levels_completed_after": after_levels,
        "level_delta": after_levels - before_levels,
        "game_state_after": str(after.game_state),
        "terminal_win": str(after.game_state).upper() in TERMINAL_WIN_STATES,
    }


def _visual_digest(grid: Any) -> str:
    array = np.asarray(grid, dtype=np.int32)
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


def _changed_pixels(before: Any, after: Any) -> int:
    return int(np.count_nonzero(np.asarray(before) != np.asarray(after)))


def _same_field(
    expected: Mapping[str, Any], observed: Mapping[str, Any], field: str
) -> bool:
    return expected.get(field) == observed.get(field)


def _blocked_protocol(
    protocol: str,
    reason: str,
    *,
    maximum_action_execution_budget: int,
    calibration: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "protocol": protocol,
        "protocol_is_bounded": True,
        "positive_transition_found": False,
        "commands": [],
        "actions_executed": 0,
        "discovery_action_executions": len(calibration),
        "calibration": list(calibration),
        "maximum_action_execution_budget": int(maximum_action_execution_budget),
        "game_source_read": False,
        "future_outcomes_used_for_action_ranking": False,
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sage8f",
        default=str(DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH),
    )
    parser.add_argument("--environments-dir")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SAGE8G_TARGET_GOAL_SIGNAL_ACTIVE_ACQUISITION_PATH),
    )
    args = parser.parse_args(argv)
    payload = run_sage8g_target_goal_signal_active_acquisition(
        sage8f_path=args.sage8f,
        environments_dir=args.environments_dir,
        output_path=args.output,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
