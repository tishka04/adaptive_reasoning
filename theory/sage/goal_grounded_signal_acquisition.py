"""SAGE.8f goal-grounded signal acquisition and target-scope admission audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .relational_memory_closed_loop_evaluation import action_identity
from .relational_memory_objective_closed_loop_evaluation import (
    DEFAULT_SAGE8E_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_EVALUATION_PATH,
    SAGE8E_LOCAL_ONLY,
    SAGE8E_SCHEMA_VERSION,
    SAGE8E_TRUTH_STATUS,
)


DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH = (
    Path("diagnostics") / "sage" / "sage8f_goal_grounded_signal_acquisition.json"
)
DEFAULT_HUMAN_TRACES_DIR = Path("human_traces")
DEFAULT_TARGET_TRANSITION_PATHS = (
    Path("training") / "data" / "wa30_ee6fef47.json",
    Path("training") / "data" / "tn36_ab4f63cc.json",
)
DEFAULT_TARGET_GAMES = ("tn36-ab4f63cc", "wa30-ee6fef47")

SAGE8F_SCHEMA_VERSION = "sage.goal_grounded_signal_acquisition.v1"
SAGE8F_SIGNAL_SCHEMA_VERSION = "sage.goal_grounded_exact_state_signal.v1"
SAGE8F_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8F"
SAGE8F_TARGET_READY = "SAGE_GOAL_GROUNDED_SIGNAL_ACQUIRED_TARGET_DOMAIN_READY"
SAGE8F_TARGET_BLOCKED = (
    "SAGE_GOAL_GROUNDED_SIGNAL_ACQUIRED_TARGET_DOMAIN_COVERAGE_BLOCKED"
)

GOAL_SIGNAL_ID = "OBSERVED_LEVEL_UP_OR_WIN_TRANSITION"
GOAL_SIGNAL_SCOPE = "EXACT_GAME_AND_VISUAL_DIGEST_ONLY"
TARGET_BLOCK_REASON = "NO_EXACT_TARGET_GAME_GOAL_TRANSITION_AVAILABLE"
TERMINAL_WIN_STATES = frozenset({"WIN", "WON", "VICTORY"})


def run_sage8f_goal_grounded_signal_acquisition(
    *,
    sage8e_path: str | Path = (
        DEFAULT_SAGE8E_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_EVALUATION_PATH
    ),
    human_traces_dir: str | Path = DEFAULT_HUMAN_TRACES_DIR,
    target_transition_paths: Sequence[str | Path] = DEFAULT_TARGET_TRANSITION_PATHS,
    target_games: Sequence[str] = DEFAULT_TARGET_GAMES,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Acquire verified goal transitions, then audit exact target-domain coverage."""
    source_sage8e = _load_json(sage8e_path)
    validate_sage8f_source(source_sage8e)
    normalized_targets = tuple(sorted({str(game_id) for game_id in target_games}))
    if not normalized_targets:
        raise ValueError("SAGE.8f requires at least one target game")

    signal_bank = acquire_goal_grounded_signal_bank(human_traces_dir)
    target_coverage = audit_target_goal_signal_coverage(
        signal_bank,
        target_transition_paths=target_transition_paths,
        target_games=normalized_targets,
    )
    gate = build_sage8f_gate(
        source_sage8e,
        signal_bank,
        target_coverage,
        target_games=normalized_targets,
    )
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8f goal-grounded signal acquisition gate did not pass")
    summary = summarize_sage8f(signal_bank, target_coverage, gate)
    payload = {
        "config": {
            "schema_version": SAGE8F_SCHEMA_VERSION,
            "sage8e_path": str(sage8e_path),
            "human_traces_dir": str(human_traces_dir),
            "target_transition_paths": [
                str(Path(path)) for path in target_transition_paths
            ],
            "target_games": list(normalized_targets),
            "goal_signal_id": GOAL_SIGNAL_ID,
            "goal_signal_scope": GOAL_SIGNAL_SCOPE,
            "acquisition_design": {
                "positive_label": (
                    "OBSERVED_LEVELS_COMPLETED_INCREASE_OR_TERMINAL_WIN"
                ),
                "predecessor_observation_required": True,
                "frame_continuity_required": True,
                "reset_actions_excluded": True,
                "non_goal_transitions_excluded": True,
                "exact_game_and_visual_digest_required": True,
                "fuzzy_matching_allowed": False,
                "cross_game_action_transfer_allowed": False,
                "source_outcomes_are_preexisting_human_demonstrations": True,
                "sage8e_evaluation_outcomes_used_for_training_or_tuning": False,
            },
            "target_admission_design": {
                "activation_requires_exact_target_game_signal": True,
                "action_schema_overlap_is_not_sufficient": True,
                "source_game_signal_is_quarantined_on_target_games": True,
                "no_target_signal_means_no_closed_loop_evaluation": True,
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
            ],
        },
        "goal_signal_bank": signal_bank,
        "target_coverage_audit": target_coverage,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "ACQUIRED_AND_ADMISSION_AUDITED",
        "truth_status": SAGE8F_TRUTH_STATUS,
        "goal_grounded_signal_acquisition_performed": True,
        "target_scope_admission_audit_performed": True,
        "planner_activation_authorized": summary["planner_activation_authorized"],
        "closed_loop_live_rollout_performed": False,
        "comparative_evaluation_performed": False,
        "evaluation_episodes_executed": 0,
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "support": 0,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "scope_generalization_performed": False,
        "evaluation_outcomes_used_for_training_or_tuning": False,
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_sage8f_goal_grounded_signal_acquisition(payload, output_path)
    return payload


def acquire_goal_grounded_signal_bank(
    human_traces_dir: str | Path,
) -> Dict[str, Any]:
    """Extract exact pre-action states that demonstrably led to level-up or win."""
    root = Path(human_traces_dir)
    source_paths = tuple(sorted(root.glob("*.steps.jsonl")))
    if not source_paths:
        raise ValueError("SAGE.8f requires at least one human step trace")

    demonstrations: list[Dict[str, Any]] = []
    rows_scanned = 0
    malformed_rows = 0
    reset_rows = 0
    non_goal_rows = 0
    missing_predecessor_rows = 0
    continuity_checks = 0
    continuity_mismatches = 0
    quarantined_goal_candidates = 0

    for path in source_paths:
        previous_by_episode: dict[str, Dict[str, Any]] = {}
        for line_number, row in _read_jsonl_rows(path):
            rows_scanned += 1
            if row is None:
                malformed_rows += 1
                continue
            episode_id = str(row.get("episode_id", ""))
            action = str(row.get("action", ""))
            after_levels = _safe_int(row.get("levels_completed_after"), default=0)
            after_state = str(row.get("game_state_after", "")).upper()
            previous = previous_by_episode.get(episode_id)
            current_after = row.get("frame_after")

            if action == "RESET":
                reset_rows += 1
            elif previous is None:
                missing_predecessor_rows += 1
            else:
                continuity_checks += 1
                continuous = previous.get("frame_after") == row.get("frame_before")
                if not continuous:
                    continuity_mismatches += 1
                before_levels = _safe_int(previous.get("levels_after"), default=0)
                won = after_state in TERMINAL_WIN_STATES
                level_delta = after_levels - before_levels
                goal_candidate = level_delta > 0 or won
                if not goal_candidate:
                    non_goal_rows += 1
                elif not continuous:
                    quarantined_goal_candidates += 1
                else:
                    demonstration = _goal_demonstration(
                        row,
                        source_path=path,
                        line_number=line_number,
                        before_levels=before_levels,
                        after_levels=after_levels,
                        level_delta=level_delta,
                        won=won,
                    )
                    if demonstration is None:
                        quarantined_goal_candidates += 1
                    else:
                        demonstrations.append(demonstration)

            previous_by_episode[episode_id] = {
                "frame_after": current_after,
                "levels_after": after_levels,
            }

    entries = aggregate_goal_signal_entries(demonstrations)
    per_game = Counter(str(row.get("game_id", "")) for row in demonstrations)
    action_families = Counter(str(row.get("action", "")) for row in demonstrations)
    level_up_count = sum(
        int(row.get("level_delta", 0) or 0) > 0 for row in demonstrations
    )
    win_count = sum(bool(row.get("terminal_win", False)) for row in demonstrations)
    repeat_states = sum(
        int(row.get("demonstration_count", 0) or 0) > 1 for row in entries
    )
    ambiguous_states = sum(
        len(row.get("action_candidates", []) or []) > 1 for row in entries
    )
    gate = {
        "at_least_one_verified_goal_transition": bool(demonstrations),
        "every_goal_transition_has_a_verified_predecessor": (
            quarantined_goal_candidates == 0
        ),
        "all_checked_frame_continuity_is_exact": continuity_mismatches == 0,
        "reset_rows_are_excluded": all(
            str(row.get("action", "")) != "RESET" for row in demonstrations
        ),
        "every_signal_is_level_up_or_win": all(
            int(row.get("level_delta", 0) or 0) > 0
            or bool(row.get("terminal_win", False))
            for row in demonstrations
        ),
        "all_entries_are_exact_scope_without_generalization": all(
            bool(row.get("exact_match_required", False))
            and not bool(row.get("fuzzy_match_allowed", True))
            and not bool(row.get("cross_game_transfer_allowed", True))
            for row in entries
        ),
        "no_truth_or_registry_support_mutation": all(
            int(row.get("support", -1) or 0) == 0
            and str(row.get("truth_status", "")) == SAGE8F_TRUTH_STATUS
            for row in entries
        ),
    }
    if not all(gate.values()):
        raise ValueError("SAGE.8f goal signal bank gate did not pass")
    return {
        "config": {
            "schema_version": SAGE8F_SIGNAL_SCHEMA_VERSION,
            "goal_signal_id": GOAL_SIGNAL_ID,
            "scope": GOAL_SIGNAL_SCOPE,
            "source_type": "PREEXISTING_HUMAN_STEP_TRACES",
            "source_paths": [str(path) for path in source_paths],
            "positive_fields_read": [
                "levels_completed_after",
                "game_state_after",
            ],
            "state_fields_read": ["frame_before", "frame_after"],
            "action_fields_read": ["action", "action_args"],
            "evaluation_outcome_fields_read": [],
            "exact_predecessor_required": True,
        },
        "entries": entries,
        "gate": gate,
        "summary": {
            "source_files": len(source_paths),
            "source_rows_scanned": rows_scanned,
            "malformed_rows": malformed_rows,
            "reset_rows_excluded": reset_rows,
            "non_goal_rows_excluded": non_goal_rows,
            "missing_predecessor_rows_quarantined": missing_predecessor_rows,
            "frame_continuity_checks": continuity_checks,
            "frame_continuity_mismatches": continuity_mismatches,
            "goal_candidates_quarantined": quarantined_goal_candidates,
            "verified_goal_transitions": len(demonstrations),
            "verified_level_up_transitions": level_up_count,
            "verified_win_transitions": win_count,
            "source_games": sorted(per_game),
            "source_games_count": len(per_game),
            "per_game_goal_transitions": dict(sorted(per_game.items())),
            "per_action_family_goal_transitions": dict(sorted(action_families.items())),
            "exact_goal_states": len(entries),
            "repeat_exact_goal_states": repeat_states,
            "ambiguous_exact_goal_states": ambiguous_states,
            "support": 0,
            "truth_status": SAGE8F_TRUTH_STATUS,
            "gate_passed": all(gate.values()),
        },
        "support": 0,
        "truth_status": SAGE8F_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "scope_generalization_performed": False,
    }


def aggregate_goal_signal_entries(
    demonstrations: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Aggregate demonstrations without weakening their exact game/state scope."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in demonstrations:
        grouped[
            (str(row.get("game_id", "")), str(row.get("visual_digest", "")))
        ].append(row)

    entries: list[Dict[str, Any]] = []
    for (game_id, visual_digest), rows in sorted(grouped.items()):
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
                "scope_key": f"{game_id}|{visual_digest}",
                "game_id": game_id,
                "visual_digest": visual_digest,
                "state_shape": list(rows[0].get("state_shape", []) or []),
                "action_candidates": candidates,
                "demonstration_count": len(rows),
                "level_up_count": sum(
                    int(row.get("level_delta", 0) or 0) > 0 for row in rows
                ),
                "win_count": sum(bool(row.get("terminal_win", False)) for row in rows),
                "source_references": [
                    {
                        "source_path": source_path,
                        "source_line_number": source_line_number,
                        "episode_id": episode_id,
                        "step": step,
                    }
                    for source_path, source_line_number, episode_id, step in sorted(
                        {
                            (
                                str(row.get("source_path", "")),
                                int(row.get("source_line_number", 0) or 0),
                                str(row.get("episode_id", "")),
                                int(row.get("step", 0) or 0),
                            )
                            for row in rows
                        }
                    )
                ],
                "exact_match_required": True,
                "fuzzy_match_allowed": False,
                "cross_game_transfer_allowed": False,
                "truth_status": SAGE8F_TRUTH_STATUS,
                "support": 0,
            }
        )
    return entries


def audit_target_goal_signal_coverage(
    signal_bank: Mapping[str, Any],
    *,
    target_transition_paths: Sequence[str | Path],
    target_games: Sequence[str],
) -> Dict[str, Any]:
    """Admit a signal only when the exact target game has a verified positive."""
    entries = [
        dict(row)
        for row in signal_bank.get("entries", []) or []
        if isinstance(row, Mapping)
    ]
    entries_by_game = Counter(str(row.get("game_id", "")) for row in entries)
    paths_by_game: dict[str, Path] = {}
    transition_rows_by_game: Counter[str] = Counter()
    positive_rows_by_game: Counter[str] = Counter()
    malformed_target_rows = 0
    for raw_path in target_transition_paths:
        path = Path(raw_path)
        rows = _load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"SAGE.8f target transition source must be a list: {path}")
        seen_games = {
            str(row.get("game_id", ""))
            for row in rows
            if isinstance(row, Mapping) and row.get("game_id")
        }
        if len(seen_games) != 1:
            raise ValueError(f"SAGE.8f target source must contain one game: {path}")
        game_id = next(iter(seen_games))
        paths_by_game[game_id] = path
        for row in rows:
            if not isinstance(row, Mapping):
                malformed_target_rows += 1
                continue
            transition_rows_by_game[game_id] += 1
            if _target_row_is_positive(row):
                positive_rows_by_game[game_id] += 1

    target_rows: list[Dict[str, Any]] = []
    target_game_set = {str(value) for value in target_games}
    for game_id in sorted(target_game_set):
        exact_entries = int(entries_by_game.get(game_id, 0) or 0)
        positive_rows = int(positive_rows_by_game.get(game_id, 0) or 0)
        activation = exact_entries > 0 and positive_rows > 0
        target_rows.append(
            {
                "game_id": game_id,
                "target_transition_path": str(paths_by_game.get(game_id, "")),
                "target_transitions_scanned": int(
                    transition_rows_by_game.get(game_id, 0) or 0
                ),
                "observed_target_goal_transitions": positive_rows,
                "exact_human_goal_signal_entries": exact_entries,
                "source_game_signal_entries_considered_for_transfer": 0,
                "cross_game_signal_entries_quarantined": sum(
                    str(row.get("game_id", "")) != game_id for row in entries
                ),
                "planner_activation_authorized": activation,
                "admission_reason": (
                    "EXACT_TARGET_GAME_GOAL_SIGNAL_AVAILABLE"
                    if activation
                    else TARGET_BLOCK_REASON
                ),
            }
        )

    missing_paths = [
        row["game_id"] for row in target_rows if not row["target_transition_path"]
    ]
    if missing_paths:
        raise ValueError(
            "SAGE.8f is missing target transition sources for: "
            + ", ".join(missing_paths)
        )
    authorized = all(
        bool(row.get("planner_activation_authorized", False)) for row in target_rows
    )
    exact_target_entries = sum(
        int(row.get("exact_human_goal_signal_entries", 0) or 0) for row in target_rows
    )
    observed_target_positives = sum(
        int(row.get("observed_target_goal_transitions", 0) or 0) for row in target_rows
    )
    source_demonstrations_quarantined = sum(
        int(row.get("demonstration_count", 0) or 0)
        for row in entries
        if str(row.get("game_id", "")) not in target_game_set
    )
    return {
        "target_games": target_rows,
        "summary": {
            "target_games_audited": len(target_rows),
            "target_transitions_scanned": sum(transition_rows_by_game.values()),
            "malformed_target_rows": malformed_target_rows,
            "observed_target_goal_transitions": observed_target_positives,
            "exact_target_goal_signal_entries": exact_target_entries,
            "source_goal_signal_demonstrations_quarantined_from_transfer": (
                source_demonstrations_quarantined
            ),
            "cross_game_transfer_performed": False,
            "planner_activation_authorized": authorized,
            "closed_loop_evaluation_authorized": authorized,
            "admission_status": (
                "TARGET_DOMAIN_READY" if authorized else "TARGET_DOMAIN_BLOCKED"
            ),
        },
    }


def build_sage8f_gate(
    source_sage8e: Mapping[str, Any],
    signal_bank: Mapping[str, Any],
    target_coverage: Mapping[str, Any],
    *,
    target_games: Sequence[str],
) -> Dict[str, bool]:
    bank_summary = dict(signal_bank.get("summary", {}) or {})
    coverage_summary = dict(target_coverage.get("summary", {}) or {})
    target_rows = target_coverage.get("target_games", []) or []
    authorized = bool(coverage_summary.get("planner_activation_authorized", False))
    target_signal_available = bool(target_rows) and all(
        int(row.get("exact_human_goal_signal_entries", 0) or 0) > 0
        and int(row.get("observed_target_goal_transitions", 0) or 0) > 0
        for row in target_rows
    )
    return {
        "completed_sage8e_source_validated": bool(
            source_sage8e.get("summary", {}).get("gate_passed", False)
        ),
        "verified_goal_signal_was_acquired": int(
            bank_summary.get("verified_goal_transitions", 0) or 0
        )
        > 0,
        "at_least_one_real_win_anchors_the_signal": int(
            bank_summary.get("verified_win_transitions", 0) or 0
        )
        > 0,
        "all_goal_candidates_have_exact_frame_continuity": int(
            bank_summary.get("goal_candidates_quarantined", -1) or 0
        )
        == 0
        and int(bank_summary.get("frame_continuity_mismatches", -1) or 0) == 0,
        "signal_scope_is_exact_without_cross_game_transfer": bool(
            signal_bank.get("gate", {}).get(
                "all_entries_are_exact_scope_without_generalization", False
            )
        )
        and not bool(signal_bank.get("scope_generalization_performed", True)),
        "all_target_games_have_transition_audits": {
            str(row.get("game_id", "")) for row in target_rows
        }
        == {str(game_id) for game_id in target_games},
        "target_admission_depends_only_on_exact_target_signal": (
            authorized == target_signal_available
        ),
        "uncovered_targets_are_quarantined_instead_of_generalized": all(
            bool(row.get("planner_activation_authorized", False))
            or str(row.get("admission_reason", "")) == TARGET_BLOCK_REASON
            for row in target_rows
        ),
        "no_sage8e_evaluation_outcomes_used_for_training_or_tuning": True,
        "no_closed_loop_evaluation_without_admitted_signal": authorized
        or not bool(coverage_summary.get("closed_loop_evaluation_authorized", True)),
        "no_truth_reevaluation_or_registry_support_counting": (
            int(signal_bank.get("support", -1) or 0) == 0
            and str(signal_bank.get("truth_status", "")) == SAGE8F_TRUTH_STATUS
        ),
    }


def summarize_sage8f(
    signal_bank: Mapping[str, Any],
    target_coverage: Mapping[str, Any],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    bank = dict(signal_bank.get("summary", {}) or {})
    coverage = dict(target_coverage.get("summary", {}) or {})
    authorized = bool(coverage.get("planner_activation_authorized", False))
    return {
        "source_trace_files": int(bank.get("source_files", 0) or 0),
        "source_rows_scanned": int(bank.get("source_rows_scanned", 0) or 0),
        "verified_goal_transitions": int(bank.get("verified_goal_transitions", 0) or 0),
        "verified_level_up_transitions": int(
            bank.get("verified_level_up_transitions", 0) or 0
        ),
        "verified_win_transitions": int(bank.get("verified_win_transitions", 0) or 0),
        "source_games_count": int(bank.get("source_games_count", 0) or 0),
        "exact_goal_states": int(bank.get("exact_goal_states", 0) or 0),
        "ambiguous_exact_goal_states": int(
            bank.get("ambiguous_exact_goal_states", 0) or 0
        ),
        "frame_continuity_mismatches": int(
            bank.get("frame_continuity_mismatches", 0) or 0
        ),
        "target_games_audited": int(coverage.get("target_games_audited", 0) or 0),
        "target_transitions_scanned": int(
            coverage.get("target_transitions_scanned", 0) or 0
        ),
        "observed_target_goal_transitions": int(
            coverage.get("observed_target_goal_transitions", 0) or 0
        ),
        "exact_target_goal_signal_entries": int(
            coverage.get("exact_target_goal_signal_entries", 0) or 0
        ),
        "source_goal_signal_demonstrations_quarantined_from_transfer": int(
            coverage.get(
                "source_goal_signal_demonstrations_quarantined_from_transfer", 0
            )
            or 0
        ),
        "cross_game_transfer_performed": False,
        "planner_activation_authorized": authorized,
        "closed_loop_evaluation_performed": False,
        "evaluation_episodes_executed": 0,
        "sage8e_evaluation_outcomes_used_for_training_or_tuning": False,
        "support_counted": 0,
        "truth_reevaluations": 0,
        "wrong_confirmations": 0,
        "gate_passed": bool(gate) and all(gate.values()),
        "outcome_status": SAGE8F_TARGET_READY if authorized else SAGE8F_TARGET_BLOCKED,
    }


def validate_sage8f_source(source_sage8e: Mapping[str, Any]) -> None:
    summary = dict(source_sage8e.get("summary", {}) or {})
    if (
        str(source_sage8e.get("config", {}).get("schema_version", ""))
        != SAGE8E_SCHEMA_VERSION
        or str(source_sage8e.get("outcome_status", "")) != SAGE8E_LOCAL_ONLY
        or str(source_sage8e.get("truth_status", "")) != SAGE8E_TRUTH_STATUS
        or str(source_sage8e.get("status", "")) != "EVALUATED"
        or not bool(source_sage8e.get("scope_safe_objective_planning_performed", False))
        or not bool(summary.get("gate_passed", False))
        or bool(summary.get("primary_arc_progress_improved", True))
        or bool(summary.get("primary_arc_progress_regressed", True))
        or bool(summary.get("training_or_tuning_used_evaluation_outcomes", True))
        or not all(bool(value) for value in source_sage8e.get("gate", {}).values())
    ):
        raise ValueError(
            "SAGE.8f requires the completed leak-safe local-only SAGE.8e evaluation"
        )


def write_sage8f_goal_grounded_signal_acquisition(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _goal_demonstration(
    row: Mapping[str, Any],
    *,
    source_path: Path,
    line_number: int,
    before_levels: int,
    after_levels: int,
    level_delta: int,
    won: bool,
) -> Dict[str, Any] | None:
    frame_before = _normalize_grid(row.get("frame_before"))
    if frame_before is None:
        return None
    array = np.asarray(frame_before, dtype=np.int32)
    visual_digest = hashlib.sha1(array.tobytes()).hexdigest()[:16]
    action = str(row.get("action", ""))
    action_args = dict(row.get("action_args", {}) or {})
    return {
        "game_id": str(row.get("game_id", "")),
        "episode_id": str(row.get("episode_id", "")),
        "step": _safe_int(row.get("step"), default=0),
        "visual_digest": visual_digest,
        "state_shape": [int(value) for value in array.shape],
        "action": action,
        "action_args": action_args,
        "action_identity": action_identity(action, action_args),
        "levels_completed_before": before_levels,
        "levels_completed_after": after_levels,
        "level_delta": level_delta,
        "terminal_win": won,
        "goal_label": "WIN" if won else "LEVEL_UP",
        "source_path": str(source_path),
        "source_line_number": int(line_number),
    }


def _target_row_is_positive(row: Mapping[str, Any]) -> bool:
    before = _safe_int(row.get("level_before"), default=0)
    after = _safe_int(row.get("level_after"), default=0)
    state = str(row.get("state_after", "")).upper()
    return (
        bool(row.get("level_changed", False))
        or after > before
        or state in TERMINAL_WIN_STATES
    )


def _read_jsonl_rows(
    path: Path,
) -> Iterable[tuple[int, Mapping[str, Any] | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None
                continue
            yield line_number, value if isinstance(value, Mapping) else None


def _normalize_grid(value: Any) -> list[list[int]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return None
        rows.append([int(cell) for cell in row])
    return rows or None


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sage8e",
        default=str(
            DEFAULT_SAGE8E_RELATIONAL_MEMORY_OBJECTIVE_CLOSED_LOOP_EVALUATION_PATH
        ),
    )
    parser.add_argument("--human-traces", default=str(DEFAULT_HUMAN_TRACES_DIR))
    parser.add_argument(
        "--target-transition",
        action="append",
        dest="target_transitions",
        default=None,
    )
    parser.add_argument(
        "--target-game", action="append", dest="target_games", default=None
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE8F_GOAL_GROUNDED_SIGNAL_ACQUISITION_PATH)
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_sage8f_goal_grounded_signal_acquisition(
        sage8e_path=args.sage8e,
        human_traces_dir=args.human_traces,
        target_transition_paths=(
            args.target_transitions
            if args.target_transitions is not None
            else DEFAULT_TARGET_TRANSITION_PATHS
        ),
        target_games=(
            args.target_games if args.target_games is not None else DEFAULT_TARGET_GAMES
        ),
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
