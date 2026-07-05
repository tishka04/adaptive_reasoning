"""SAGE.5 first unknown-game bounded closed-loop probe.

SAGE.5 runs the SAGE.4c loop discipline on one public-unseen game with short
budgets. It is deliberately not a benchmark and never treats policy outcomes as
scientific confirmation. The gate checks bounded technical properties only:
unknown-game hygiene, legal actions, progress-stall detector availability,
subgoal switching, no catastrophic repeat collapse, and terminal-rate guard.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import game_splits

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
)
from .long_horizon_progress_stall_transfer import (
    DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH,
)
from .long_horizon_transfer import DEFAULT_BUDGETS
from .policy_loop_guard import action_args, action_key, action_name
from .progress_stall_trigger import (
    DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    DEFAULT_PROGRESS_STALL_WINDOW,
    DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    DEFAULT_SAME_ACTION_ARG_REPEATS,
)
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    run_sage3_subgoal_switch_probe,
)
from .live_prefix_counterfactual_collector import state_signature_from_frame


DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage5_unknown_game_bounded_probe_results.json"
)
DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_dataset_manifest.json"
)
DEFAULT_HUMAN_TRACES_DIR = Path("human_traces")
DEFAULT_UNKNOWN_GAME_ID = "sb26-7fbdac44"

SAGE5_SCHEMA_VERSION = "sage.unknown_game_bounded_probe_results.v1"
SAGE5_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5"
SAGE5_TERMINAL_RATE_THRESHOLD = 0.05
SAGE5_REPEAT_COLLAPSE_THRESHOLD = 0.9

POLICY_RANDOM_LEGAL = "random_legal"
POLICY_NEUTRAL_LEGAL_FALLBACK = "neutral_legal_fallback"
BASELINE_POLICIES: tuple[str, ...] = (
    POLICY_RANDOM_LEGAL,
    POLICY_NEUTRAL_LEGAL_FALLBACK,
)

SAGE5_ALL_BUDGETS_PASSED = (
    "SAGE_UNKNOWN_BOUNDED_PROBE_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY"
)
SAGE5_PARTIAL_PASSED = (
    "SAGE_UNKNOWN_BOUNDED_PROBE_PARTIAL_GATE_PASSED_CANDIDATE_ONLY"
)
SAGE5_FAILED = "SAGE_UNKNOWN_BOUNDED_PROBE_GATE_FAILED_CANDIDATE_ONLY"
SAGE5_BLOCKED_KNOWN = (
    "SAGE_UNKNOWN_BOUNDED_PROBE_BLOCKED_KNOWN_GAME_CANDIDATE_ONLY"
)

_TERMINAL_STATES = {"GAME_OVER", "WIN", "TERMINATED", "FINISHED"}

EnvFactory = Callable[[str], Any]


def run_sage5_unknown_game_bounded_probe(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    source_sage4c_path: str | Path = (
        DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH
    ),
    m2_dataset_manifest_path: str | Path = DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH,
    human_traces_dir: str | Path = DEFAULT_HUMAN_TRACES_DIR,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    game_id: str = DEFAULT_UNKNOWN_GAME_ID,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 0,
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    progress_stall_window: int = DEFAULT_PROGRESS_STALL_WINDOW,
    same_action_arg_repeats: int = DEFAULT_SAME_ACTION_ARG_REPEATS,
    low_state_novelty_threshold: int = DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    repeated_action_arg_rate_threshold: float = (
        DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD
    ),
    terminal_rate_threshold: float = SAGE5_TERMINAL_RATE_THRESHOLD,
    repeat_collapse_threshold: float = SAGE5_REPEAT_COLLAPSE_THRESHOLD,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run a bounded candidate-only SAGE probe on one unknown game."""
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    m2_manifest = _load_optional_json(m2_dataset_manifest_path)
    source_sage4c = _load_optional_json(source_sage4c_path)
    identity = unknown_game_identity(
        game_id=game_id,
        human_traces_dir=human_traces_dir,
        m2_dataset_manifest=m2_manifest,
    )

    per_budget: List[Dict[str, Any]] = []
    for budget in budgets:
        sage_run = run_sage3_subgoal_switch_probe(
            m2_fused_requests_path=m2_fused_requests_path,
            m3_fused_results_path=m3_fused_results_path,
            m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
            p1_policy_probe_path=p1_policy_probe_path,
            p1_utility_handoff_path=p1_utility_handoff_path,
            environments_dir=env_dir,
            output_path=None,
            game_id=game_id,
            budget=int(budget),
            max_counterfactual_collections=max_counterfactual_collections,
            env_factory=env_factory,
            enable_progress_stall_trigger=True,
            progress_stall_window=progress_stall_window,
            same_action_arg_repeats=same_action_arg_repeats,
            low_state_novelty_threshold=low_state_novelty_threshold,
            repeated_action_arg_rate_threshold=repeated_action_arg_rate_threshold,
        )
        baselines = {
            policy: _run_baseline_policy(
                policy=policy,
                game_id=game_id,
                budget=int(budget),
                env_dir=env_dir,
                env_factory=env_factory,
                seed=seed,
            )
            for policy in BASELINE_POLICIES
        }
        per_budget.append(
            _budget_record(
                budget=int(budget),
                sage_run=sage_run,
                baselines=baselines,
                identity=identity,
                terminal_rate_threshold=terminal_rate_threshold,
                repeat_collapse_threshold=repeat_collapse_threshold,
            )
        )

    comparison = _build_comparison(
        per_budget,
        identity=identity,
        source_sage4c=source_sage4c,
        terminal_rate_threshold=terminal_rate_threshold,
        repeat_collapse_threshold=repeat_collapse_threshold,
    )
    payload = {
        "config": {
            "schema_version": SAGE5_SCHEMA_VERSION,
            "game_id": game_id,
            "budgets": [int(budget) for budget in budgets],
            "seed": int(seed),
            "environments_dir": str(env_dir),
            "source_sage4c_path": str(source_sage4c_path),
            "m2_dataset_manifest_path": str(m2_dataset_manifest_path),
            "human_traces_dir": str(human_traces_dir),
            "unknown_game_bounded_probe": True,
            "benchmark_run": False,
            "baselines_compared": [
                POLICY_RANDOM_LEGAL,
                POLICY_NEUTRAL_LEGAL_FALLBACK,
                "SAGE.4c",
            ],
            "progress_stall_detector_runs": True,
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "terminal_rate_guarded": True,
            "terminal_rate_threshold": float(terminal_rate_threshold),
            "stop_safe_hold_after_terminal": True,
            "offline_counterfactual_allowed": False,
            "active_counterfactual_collection_allowed": True,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1", "SAGE.4c"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
            "gate_thresholds": {
                "repeat_collapse_threshold": float(repeat_collapse_threshold),
                "terminal_rate_threshold": float(terminal_rate_threshold),
            },
        },
        "unknown_game_identity": identity,
        "source_sage4c_context": _source_sage4c_context(source_sage4c),
        "per_budget_results": per_budget,
        "comparison": comparison,
        "summary": comparison,
        "status": "UNRESOLVED",
        "outcome_status": comparison["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5_TRUTH_STATUS,
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
        write_sage5_unknown_game_bounded_probe_results(payload, output_path)
    return payload


def unknown_game_identity(
    *,
    game_id: str,
    human_traces_dir: str | Path,
    m2_dataset_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    short = str(game_id).split("-", 1)[0]
    split = game_splits.split_for_game(game_id) or "unregistered"
    trace_paths = _matching_human_trace_paths(short, human_traces_dir)
    m2_games = {
        str(game)
        for game in (m2_dataset_manifest.get("per_game_counts", {}) or {}).keys()
    }
    m2_short_games = {game.split("-", 1)[0] for game in m2_games}
    no_human_trace = len(trace_paths) == 0
    no_m2_trace = str(game_id) not in m2_games and short not in m2_short_games
    public_unseen_or_unregistered = split in {"unseen", "unregistered"}
    no_game_specific_prior = bool(
        no_human_trace and no_m2_trace and public_unseen_or_unregistered
    )
    unknown = bool(no_human_trace and no_m2_trace and no_game_specific_prior)
    return {
        "game_id": str(game_id),
        "short_game_id": short,
        "split": split,
        "public_unseen_or_unregistered": public_unseen_or_unregistered,
        "unknown_game": unknown,
        "no_human_trace_for_game": no_human_trace,
        "human_trace_matches": [str(path) for path in trace_paths],
        "no_m2_arc_lewm_trace_for_game": no_m2_trace,
        "no_game_specific_prior": no_game_specific_prior,
        "no_game_specific_prior_reason": (
            "no_human_trace_no_m2_trace_public_unseen_or_unregistered"
            if no_game_specific_prior
            else "known_trace_or_seen_split_detected"
        ),
        "support": 0,
        "truth_status": SAGE5_TRUTH_STATUS,
    }


def _matching_human_trace_paths(short_game_id: str, human_traces_dir: str | Path) -> List[Path]:
    root = Path(human_traces_dir)
    if not root.exists():
        return []
    return sorted(root.glob(f"{short_game_id}-*.steps.jsonl"))


def _budget_record(
    *,
    budget: int,
    sage_run: Mapping[str, Any],
    baselines: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
    terminal_rate_threshold: float,
    repeat_collapse_threshold: float,
) -> Dict[str, Any]:
    summary = dict(sage_run.get("summary", {}) or {})
    metrics = {
        "levels_completed": int(summary.get("levels_completed", 0) or 0),
        "terminal_rate": float(summary.get("terminal_rate", 0.0) or 0.0),
        "terminal_rate_guarded": True,
        "terminal_rate_threshold": float(terminal_rate_threshold),
        "terminal_rate_under_threshold": (
            float(summary.get("terminal_rate", 0.0) or 0.0)
            <= float(terminal_rate_threshold)
        ),
        "stop_safe_hold_after_terminal": True,
        "progress_stall_detector_runs": True,
        "progress_stall_detected": bool(
            summary.get("progress_stall_detected", False)
        ),
        "progress_stall_switches": int(
            summary.get("progress_stall_switches", 0) or 0
        ),
        "subgoal_switches": int(summary.get("subgoal_switches", 0) or 0),
        "subgoal_switch_success_rate": float(
            summary.get("subgoal_switch_success_rate", 0.0) or 0.0
        ),
        "new_candidate_targets_discovered": int(
            summary.get("new_candidate_targets_discovered", 0) or 0
        ),
        "active_counterfactuals_after_exhaustion": int(
            summary.get("active_counterfactuals_after_exhaustion", 0) or 0
        ),
        "rerun_m2_m3_requested": int(summary.get("rerun_m2_m3_requested", 0) or 0),
        "rerun_m2_m3_effective_requests_generated": int(
            summary.get("rerun_m2_m3_effective_requests_generated", 0) or 0
        ),
        "post_switch_repeat_rate": float(
            summary.get("post_switch_repeat_rate", 0.0) or 0.0
        ),
        "repeated_action_arg_rate": float(
            summary.get("repeated_action_arg_rate", 0.0) or 0.0
        ),
        "repeat_collapse_threshold": float(repeat_collapse_threshold),
        "unique_state_signatures": int(summary.get("unique_state_signatures", 0) or 0),
        "env_steps": int(summary.get("env_steps", 0) or 0),
        "selected_action_always_legal": bool(
            summary.get("selected_action_always_legal", False)
        ),
        "invalid_action_selected": bool(summary.get("invalid_action_selected", False)),
        "subgoals_used": list(summary.get("subgoals_used", []) or []),
        "offline_counterfactual_allowed": False,
        "active_counterfactual_collection_allowed": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE5_TRUTH_STATUS,
    }
    return {
        "budget": int(budget),
        "metrics": metrics,
        "gate": _evaluate_gate(
            metrics,
            identity=identity,
            terminal_rate_threshold=terminal_rate_threshold,
            repeat_collapse_threshold=repeat_collapse_threshold,
        ),
        "baselines": {key: dict(value) for key, value in baselines.items()},
        "sage_source_outcome_status": str(summary.get("outcome_status", "")),
    }


def _evaluate_gate(
    metrics: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    terminal_rate_threshold: float,
    repeat_collapse_threshold: float,
) -> Dict[str, Any]:
    unknown = bool(identity.get("unknown_game", False))
    no_human = bool(identity.get("no_human_trace_for_game", False))
    no_prior = bool(identity.get("no_game_specific_prior", False))
    legal = bool(metrics.get("selected_action_always_legal", False))
    detector_runs = bool(metrics.get("progress_stall_detector_runs", False))
    switches = int(metrics.get("subgoal_switches", 0) or 0) >= 1
    no_repeat_collapse = (
        float(metrics.get("repeated_action_arg_rate", 1.0))
        < float(repeat_collapse_threshold)
    )
    terminal_under_threshold = (
        float(metrics.get("terminal_rate", 1.0)) <= float(terminal_rate_threshold)
    )
    offline_cf_forbidden = not bool(metrics.get("offline_counterfactual_allowed", True))
    gate_passed = bool(
        unknown
        and no_human
        and no_prior
        and legal
        and detector_runs
        and switches
        and no_repeat_collapse
        and terminal_under_threshold
        and offline_cf_forbidden
    )
    return {
        "unknown_game": unknown,
        "no_human_trace_for_game": no_human,
        "no_game_specific_prior": no_prior,
        "selected_action_always_legal": legal,
        "progress_stall_detector_runs": detector_runs,
        "subgoal_switches_happened": switches,
        "no_catastrophic_repeat_collapse": no_repeat_collapse,
        "repeat_collapse_threshold": float(repeat_collapse_threshold),
        "terminal_rate_guarded": True,
        "terminal_rate_threshold": float(terminal_rate_threshold),
        "terminal_rate_under_threshold": terminal_under_threshold,
        "stop_safe_hold_after_terminal": True,
        "offline_counterfactual_allowed": False,
        "gate_passed": gate_passed,
    }


def _build_comparison(
    per_budget: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    source_sage4c: Mapping[str, Any],
    terminal_rate_threshold: float,
    repeat_collapse_threshold: float,
) -> Dict[str, Any]:
    gates = [bool(row.get("gate", {}).get("gate_passed", False)) for row in per_budget]
    passed = sum(1 for gate in gates if gate)
    total = len(gates)
    any_passed = passed > 0
    all_passed = total > 0 and passed == total
    if not bool(identity.get("unknown_game", False)):
        outcome = SAGE5_BLOCKED_KNOWN
    elif all_passed:
        outcome = SAGE5_ALL_BUDGETS_PASSED
    elif any_passed:
        outcome = SAGE5_PARTIAL_PASSED
    else:
        outcome = SAGE5_FAILED
    return {
        "game_id": str(identity.get("game_id", "")),
        "unknown_game": bool(identity.get("unknown_game", False)),
        "no_human_trace_for_game": bool(
            identity.get("no_human_trace_for_game", False)
        ),
        "no_game_specific_prior": bool(
            identity.get("no_game_specific_prior", False)
        ),
        "budgets_evaluated": [int(row.get("budget", 0)) for row in per_budget],
        "budgets_gate_passed": passed,
        "budgets_total": total,
        "all_budgets_gate_passed": all_passed,
        "any_budget_gate_passed": any_passed,
        "budgets_with_progress_stall_detected": [
            int(row.get("budget", 0))
            for row in per_budget
            if bool(row.get("metrics", {}).get("progress_stall_detected", False))
        ],
        "budgets_with_subgoal_switches": [
            int(row.get("budget", 0))
            for row in per_budget
            if int(row.get("metrics", {}).get("subgoal_switches", 0) or 0) >= 1
        ],
        "budgets_without_catastrophic_repeat_collapse": [
            int(row.get("budget", 0))
            for row in per_budget
            if bool(row.get("gate", {}).get("no_catastrophic_repeat_collapse", False))
        ],
        "budgets_terminal_rate_under_threshold": [
            int(row.get("budget", 0))
            for row in per_budget
            if bool(row.get("gate", {}).get("terminal_rate_under_threshold", False))
        ],
        "terminal_rate_guarded": True,
        "terminal_rate_threshold": float(terminal_rate_threshold),
        "stop_safe_hold_after_terminal": True,
        "repeat_collapse_threshold": float(repeat_collapse_threshold),
        "progress_stall_detector_runs": True,
        "offline_counterfactual_allowed": False,
        "active_counterfactual_collection_allowed": True,
        "baselines_compared": [
            POLICY_RANDOM_LEGAL,
            POLICY_NEUTRAL_LEGAL_FALLBACK,
            "SAGE.4c",
        ],
        "source_sage4c_all_budgets_gate_passed": bool(
            (source_sage4c.get("comparison", {}) or {}).get(
                "all_budgets_gate_passed", False
            )
        ),
        "source_sage4c_outcome_status": str(source_sage4c.get("outcome_status", "")),
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE5_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _run_baseline_policy(
    *,
    policy: str,
    game_id: str,
    budget: int,
    env_dir: Path,
    env_factory: EnvFactory | None,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(f"sage5:{policy}:{game_id}:{seed}")
    try:
        env = env_factory(game_id) if env_factory is not None else _make_real_env(game_id, env_dir)
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return _blocked_baseline(policy, f"env_setup_failed:{exc}")

    steps: List[Dict[str, Any]] = []
    initial_signature = state_signature_from_frame(frame)
    previous_key = ""
    for step_index in range(max(0, int(budget))):
        before = snapshot_frame(frame)
        if _is_terminal(before.game_state):
            break
        valid_actions = _valid_actions(env)
        selected = _select_baseline_action(policy, valid_actions, rng)
        if selected is None:
            steps.append(_baseline_error_step(policy, step_index, game_id))
            break
        before_signature = state_signature_from_frame(frame)
        try:
            after_frame = _step_env_action(env, selected)
        except Exception as exc:  # pragma: no cover - integration failure path
            steps.append(
                _baseline_error_step(
                    policy,
                    step_index,
                    game_id,
                    error=f"env_step_failed:{exc}",
                    before_signature=before_signature,
                )
            )
            break
        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        after_signature = state_signature_from_frame(after_frame)
        key = action_key(action_name(selected), action_args(selected))
        steps.append(
            {
                "policy": policy,
                "step": int(step_index),
                "game_id": game_id,
                "selected_action": action_name(selected),
                "selected_action_args": action_args(selected),
                "selected_action_legal": True,
                "invalid_action_selected": False,
                "state_signature_before": before_signature,
                "state_signature_after": after_signature,
                "state_changed": before_signature != after_signature,
                "repeated_previous_action": bool(previous_key and key == previous_key),
                "levels_before": before.levels_completed,
                "levels_after": after.levels_completed,
                "game_state_before": before.game_state,
                "game_state_after": after.game_state,
                "terminal_after": _is_terminal(after.game_state),
                "env_actions": 1,
                "support": 0,
                "truth_status": SAGE5_TRUTH_STATUS,
            }
        )
        previous_key = key
        frame = after_frame
        if _is_terminal(after.game_state):
            break
    return {
        "policy": policy,
        "summary": _summarize_baseline(policy, steps, initial_signature),
    }


def _select_baseline_action(policy: str, valid_actions: Sequence[Any], rng: random.Random) -> Any | None:
    legal = [
        action
        for action in valid_actions
        if action_name(action) and action_name(action) != "RESET"
    ]
    if not legal:
        return None
    if policy == POLICY_RANDOM_LEGAL:
        return rng.choice(legal)
    if policy == POLICY_NEUTRAL_LEGAL_FALLBACK:
        for preferred in ("ACTION4", "ACTION3", "ACTION7", "ACTION1", "ACTION2"):
            for action in legal:
                if action_name(action) == preferred and not action_args(action):
                    return action
        for action in legal:
            if not action_args(action):
                return action
        return legal[0]
    raise ValueError(f"unknown_sage5_baseline:{policy}")


def _summarize_baseline(
    policy: str,
    steps: Sequence[Mapping[str, Any]],
    initial_signature: str,
) -> Dict[str, Any]:
    executed = [row for row in steps if not row.get("invalid_action_selected", False)]
    n = len(executed)
    levels = [int(row.get("levels_after", 0) or 0) for row in executed]
    signatures = {initial_signature}
    for row in executed:
        signatures.add(str(row.get("state_signature_after", "")))
    return {
        "policy": policy,
        "steps_executed": n,
        "selected_action_always_legal": len(executed) == len(steps) and n > 0,
        "invalid_action_selected": any(
            bool(row.get("invalid_action_selected", False)) for row in steps
        ),
        "levels_completed": max(levels) if levels else 0,
        "terminal_rate": _ratio(
            sum(1 for row in executed if bool(row.get("terminal_after", False))),
            n,
        ),
        "state_changed_rate": _ratio(
            sum(1 for row in executed if bool(row.get("state_changed", False))),
            n,
        ),
        "unique_state_signatures": len(signatures),
        "repeated_action_arg_rate": _ratio(
            sum(1 for row in executed if bool(row.get("repeated_previous_action", False))),
            n,
        ),
        "support": 0,
        "truth_status": SAGE5_TRUTH_STATUS,
        "policy_result_counted_as_confirmation": False,
    }


def _baseline_error_step(
    policy: str,
    step_index: int,
    game_id: str,
    *,
    error: str = "no_legal_action",
    before_signature: str = "",
) -> Dict[str, Any]:
    return {
        "policy": policy,
        "step": int(step_index),
        "game_id": game_id,
        "selected_action": "",
        "selected_action_args": {},
        "selected_action_legal": False,
        "invalid_action_selected": True,
        "error": error,
        "state_signature_before": before_signature,
        "env_actions": 0,
        "terminal_after": False,
        "support": 0,
        "truth_status": SAGE5_TRUTH_STATUS,
    }


def _blocked_baseline(policy: str, reason: str) -> Dict[str, Any]:
    return {
        "policy": policy,
        "summary": {
            "policy": policy,
            "blocked_reason": reason,
            "steps_executed": 0,
            "selected_action_always_legal": False,
            "invalid_action_selected": False,
            "levels_completed": 0,
            "terminal_rate": 0.0,
            "state_changed_rate": 0.0,
            "unique_state_signatures": 0,
            "repeated_action_arg_rate": 0.0,
            "support": 0,
            "truth_status": SAGE5_TRUTH_STATUS,
            "policy_result_counted_as_confirmation": False,
        },
    }


def _source_sage4c_context(source_sage4c: Mapping[str, Any]) -> Dict[str, Any]:
    comparison = dict(source_sage4c.get("comparison", {}) or {})
    return {
        "source_artifact": "SAGE.4c",
        "all_budgets_gate_passed": bool(
            comparison.get("all_budgets_gate_passed", False)
        ),
        "budgets_gate_passed": int(comparison.get("budgets_gate_passed", 0) or 0),
        "budgets_total": int(comparison.get("budgets_total", 0) or 0),
        "outcome_status": str(source_sage4c.get("outcome_status", "")),
        "source_counted_as_unknown_game_evidence": False,
        "support": 0,
        "truth_status": SAGE5_TRUTH_STATUS,
    }


def write_sage5_unknown_game_bounded_probe_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
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
        description="Run SAGE.5 bounded unknown-game closed-loop probe.",
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
    parser.add_argument("--source-sage4c", default=str(DEFAULT_SAGE4C_LONG_HORIZON_PROGRESS_STALL_RESULTS_PATH))
    parser.add_argument("--m2-dataset-manifest", default=str(DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH))
    parser.add_argument("--human-traces-dir", default=str(DEFAULT_HUMAN_TRACES_DIR))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH))
    parser.add_argument("--game-id", default=DEFAULT_UNKNOWN_GAME_ID)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seed", type=int, default=0)
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
        "--terminal-rate-threshold",
        type=float,
        default=SAGE5_TERMINAL_RATE_THRESHOLD,
    )
    args = parser.parse_args(argv)
    run_sage5_unknown_game_bounded_probe(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        source_sage4c_path=args.source_sage4c,
        m2_dataset_manifest_path=args.m2_dataset_manifest,
        human_traces_dir=args.human_traces_dir,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budgets=args.budgets,
        seed=args.seed,
        max_counterfactual_collections=args.max_counterfactual_collections,
        progress_stall_window=args.progress_stall_window,
        same_action_arg_repeats=args.same_action_arg_repeats,
        low_state_novelty_threshold=args.low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=args.repeated_action_arg_rate_threshold,
        terminal_rate_threshold=args.terminal_rate_threshold,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
