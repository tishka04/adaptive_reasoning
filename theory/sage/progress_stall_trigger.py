"""SAGE.4b long-horizon progress-stall trigger.

This module detects repeat-collapse/progress-stall patterns that do not pass
through the bp35-specific ``candidate_policy_live_target_fallback`` signal.
It is a trigger repair, not a policy verdict.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .known_game_closed_loop_scaffold import (
    DEFAULT_M2_FUSED_REQUESTS_PATH,
    DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH,
    DEFAULT_M3_FUSED_RESULTS_PATH,
    DEFAULT_P1_POLICY_PROBE_PATH,
    DEFAULT_P1_UTILITY_HANDOFF_PATH,
)
from .policy_loop_guard import action_key


DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH = (
    Path("diagnostics") / "sage" / "sage4b_progress_stall_trigger_results.json"
)
SAGE4B_SCHEMA_VERSION = "sage.progress_stall_trigger_results.v1"
SAGE4B_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_4B"
PROGRESS_STALL_TRIGGER_REASON = "progress_stall_or_repeat_collapse"
DEFAULT_GAME_ID = "ar25-e3c63847"
DEFAULT_BUDGET = 150
DEFAULT_PROGRESS_STALL_WINDOW = 8
DEFAULT_SAME_ACTION_ARG_REPEATS = 4
DEFAULT_LOW_STATE_NOVELTY_THRESHOLD = 3
DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD = 0.75
POST_TRIGGER_REPEAT_FAILURE_THRESHOLD = 0.9


@dataclass(frozen=True)
class ProgressStallTriggerConfig:
    """Configuration for long-horizon stall detection."""

    window_size: int = DEFAULT_PROGRESS_STALL_WINDOW
    same_action_arg_repeats: int = DEFAULT_SAME_ACTION_ARG_REPEATS
    low_state_novelty_threshold: int = DEFAULT_LOW_STATE_NOVELTY_THRESHOLD
    repeated_action_arg_rate_threshold: float = (
        DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD
    )


@dataclass(frozen=True)
class ProgressStallTriggerResult:
    """Candidate-only trigger diagnostics for one proposed action."""

    switch_required: bool
    trigger_reason: str
    same_action_args_repeated: bool
    same_action_args_repeat_count: int
    low_state_novelty: bool
    unique_state_signatures_in_window: int
    no_level_progress: bool
    levels_completed_delta: int
    repeated_action_arg_rate_window: float
    repeated_action_arg_rate_over_threshold: bool
    no_new_candidate_targets_discovered: bool = True
    active_counterfactuals_after_exhaustion_in_window: int = 0
    window_size: int = DEFAULT_PROGRESS_STALL_WINDOW
    support: int = 0
    truth_status: str = SAGE4B_TRUTH_STATUS
    revision_status: str = "CANDIDATE_ONLY"
    policy_result_counted_as_confirmation: bool = False
    a32_write_performed: bool = False
    a33_write_performed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "switch_required": self.switch_required,
            "trigger_reason": self.trigger_reason,
            "same_action_args_repeated": self.same_action_args_repeated,
            "same_action_args_repeat_count": self.same_action_args_repeat_count,
            "low_state_novelty": self.low_state_novelty,
            "unique_state_signatures_in_window": self.unique_state_signatures_in_window,
            "no_level_progress": self.no_level_progress,
            "levels_completed_delta": self.levels_completed_delta,
            "repeated_action_arg_rate_window": self.repeated_action_arg_rate_window,
            "repeated_action_arg_rate_over_threshold": (
                self.repeated_action_arg_rate_over_threshold
            ),
            "no_new_candidate_targets_discovered": (
                self.no_new_candidate_targets_discovered
            ),
            "active_counterfactuals_after_exhaustion_in_window": (
                self.active_counterfactuals_after_exhaustion_in_window
            ),
            "window_size": self.window_size,
            "support": self.support,
            "truth_status": self.truth_status,
            "revision_status": self.revision_status,
            "policy_result_counted_as_confirmation": (
                self.policy_result_counted_as_confirmation
            ),
            "a32_write_performed": self.a32_write_performed,
            "a33_write_performed": self.a33_write_performed,
        }


def evaluate_progress_stall_trigger(
    *,
    steps: Sequence[Mapping[str, Any]],
    proposed_action: str,
    proposed_action_args: Mapping[str, Any] | None = None,
    config: ProgressStallTriggerConfig | None = None,
) -> ProgressStallTriggerResult:
    """Detect whether a generic long-horizon progress stall requires a switch."""
    cfg = config or ProgressStallTriggerConfig()
    executed = [
        row for row in steps if not bool(row.get("invalid_action_selected", False))
    ]
    proposed_key = action_key(proposed_action, dict(proposed_action_args or {}))
    action_keys = [
        action_key(
            str(row.get("selected_action", "")),
            dict(row.get("selected_action_args", {}) or {}),
        )
        for row in executed
    ]
    action_keys = [key for key in action_keys if key]
    action_keys_with_proposed = action_keys + ([proposed_key] if proposed_key else [])
    repeat_count = _trailing_count(action_keys_with_proposed, proposed_key)
    same_action_repeated = repeat_count >= max(1, int(cfg.same_action_arg_repeats))

    window_count = max(1, int(cfg.window_size))
    action_window = action_keys_with_proposed[-window_count:]
    repeated_pairs = sum(
        1
        for previous, current in zip(action_window, action_window[1:])
        if previous and previous == current
    )
    repeated_rate = _ratio(repeated_pairs, max(1, len(action_window) - 1))
    repeated_over_threshold = (
        repeated_rate > float(cfg.repeated_action_arg_rate_threshold)
    )

    recent_steps = executed[-window_count:]
    signatures = {
        str(row.get("state_signature_after", ""))
        for row in recent_steps
        if str(row.get("state_signature_after", ""))
    }
    unique_signatures = len(signatures)
    enough_state_window = len(recent_steps) >= max(1, window_count // 2)
    low_state_novelty = (
        enough_state_window
        and unique_signatures <= int(cfg.low_state_novelty_threshold)
    )

    levels = [int(row.get("levels_after", 0) or 0) for row in recent_steps]
    if levels:
        levels_delta = max(levels) - min(levels)
    else:
        levels_delta = 0
    no_level_progress = len(recent_steps) >= 1 and levels_delta == 0

    switch_required = bool(
        no_level_progress
        and (
            same_action_repeated
            or repeated_over_threshold
            or (low_state_novelty and repeated_rate > 0.0)
        )
    )
    return ProgressStallTriggerResult(
        switch_required=switch_required,
        trigger_reason=(
            PROGRESS_STALL_TRIGGER_REASON if switch_required else "no_progress_stall"
        ),
        same_action_args_repeated=same_action_repeated,
        same_action_args_repeat_count=repeat_count,
        low_state_novelty=low_state_novelty,
        unique_state_signatures_in_window=unique_signatures,
        no_level_progress=no_level_progress,
        levels_completed_delta=levels_delta,
        repeated_action_arg_rate_window=repeated_rate,
        repeated_action_arg_rate_over_threshold=repeated_over_threshold,
        window_size=window_count,
    )


def run_sage4b_progress_stall_trigger_probe(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    budget: int = DEFAULT_BUDGET,
    max_counterfactual_collections: int = 8,
    progress_stall_window: int = DEFAULT_PROGRESS_STALL_WINDOW,
    same_action_arg_repeats: int = DEFAULT_SAME_ACTION_ARG_REPEATS,
    low_state_novelty_threshold: int = DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    repeated_action_arg_rate_threshold: float = (
        DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD
    ),
    env_factory: Any | None = None,
) -> Dict[str, Any]:
    """Run SAGE.3 with the SAGE.4b progress-stall trigger enabled."""
    from .subgoal_switcher import run_sage3_subgoal_switch_probe

    run = run_sage3_subgoal_switch_probe(
        m2_fused_requests_path=m2_fused_requests_path,
        m3_fused_results_path=m3_fused_results_path,
        m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
        p1_policy_probe_path=p1_policy_probe_path,
        p1_utility_handoff_path=p1_utility_handoff_path,
        environments_dir=environments_dir,
        output_path=None,
        game_id=game_id,
        budget=budget,
        max_counterfactual_collections=max_counterfactual_collections,
        env_factory=env_factory,
        enable_progress_stall_trigger=True,
        progress_stall_window=progress_stall_window,
        same_action_arg_repeats=same_action_arg_repeats,
        low_state_novelty_threshold=low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=repeated_action_arg_rate_threshold,
    )
    summary = build_sage4b_summary(run)
    payload = {
        "config": {
            "schema_version": SAGE4B_SCHEMA_VERSION,
            "game_id": game_id,
            "budget": int(budget),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "post_trigger_repeat_failure_threshold": (
                POST_TRIGGER_REPEAT_FAILURE_THRESHOLD
            ),
            "benchmark_run": False,
            "inputs_read": ["M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "progress_stall_run": {
            "steps": run.get("steps", []),
            "subgoal_switch_events": run.get("subgoal_switch_events", []),
            "active_counterfactual_collections": run.get(
                "active_counterfactual_collections", []
            ),
            "source_summary": run.get("summary", {}),
        },
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE4B_TRUTH_STATUS,
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
        write_sage4b_progress_stall_results(payload, output_path)
    return payload


def build_sage4b_summary(run: Mapping[str, Any]) -> Dict[str, Any]:
    source = dict(run.get("summary", {}) or {})
    steps = list(run.get("steps", []) or [])
    progress_steps = [
        row
        for row in steps
        if str(row.get("trigger_reason", "")) == PROGRESS_STALL_TRIGGER_REASON
    ]
    first_trigger_step = (
        min(int(row.get("step", 0) or 0) for row in progress_steps)
        if progress_steps
        else None
    )
    before_rows = [
        row
        for row in steps
        if first_trigger_step is not None
        and int(row.get("step", 0) or 0) < first_trigger_step
        and int(row.get("env_actions", 0) or 0) > 0
    ]
    after_rows = [
        row
        for row in steps
        if first_trigger_step is not None
        and int(row.get("step", 0) or 0) >= first_trigger_step
        and int(row.get("env_actions", 0) or 0) > 0
    ]
    before_rate = repeated_action_arg_rate(before_rows)
    after_rate = repeated_action_arg_rate(after_rows)
    progress_stall_detected = bool(progress_steps)
    repeat_interrupted = bool(
        progress_stall_detected
        and before_rate > after_rate
        and after_rate < POST_TRIGGER_REPEAT_FAILURE_THRESHOLD
    )
    useful_action = bool(
        int(source.get("active_counterfactuals_after_exhaustion", 0) or 0) > 0
        or int(source.get("rerun_m2_m3_requested", 0) or 0) > 0
    )
    gate_passed = bool(
        source.get("selected_action_always_legal", False)
        and progress_stall_detected
        and int(source.get("subgoal_switches", 0) or 0) > 0
        and repeat_interrupted
        and useful_action
    )
    if gate_passed:
        outcome = "SAGE_PROGRESS_STALL_TRIGGER_REPAIRED_CANDIDATE_ONLY"
    elif progress_stall_detected:
        outcome = "SAGE_PROGRESS_STALL_TRIGGER_PARTIAL_REPAIR_CANDIDATE_ONLY"
    else:
        outcome = "SAGE_PROGRESS_STALL_TRIGGER_NOT_DETECTED_CANDIDATE_ONLY"
    return {
        "progress_stall_trigger_enabled": True,
        "progress_stall_detected": progress_stall_detected,
        "first_progress_stall_trigger_step": first_trigger_step,
        "subgoal_switches": int(source.get("subgoal_switches", 0) or 0),
        "progress_stall_switches": len(progress_steps),
        "repeat_collapse_interrupted": repeat_interrupted,
        "repeated_action_arg_rate_before_trigger": before_rate,
        "repeated_action_arg_rate_after_trigger": after_rate,
        "post_trigger_repeat_failure_threshold": POST_TRIGGER_REPEAT_FAILURE_THRESHOLD,
        "selected_action_always_legal": bool(
            source.get("selected_action_always_legal", False)
        ),
        "invalid_action_selected": bool(source.get("invalid_action_selected", False)),
        "active_counterfactual_collection_attempted": bool(
            int(source.get("active_counterfactuals_after_exhaustion", 0) or 0) > 0
        ),
        "active_counterfactuals_after_exhaustion": int(
            source.get("active_counterfactuals_after_exhaustion", 0) or 0
        ),
        "rerun_m2_m3_requested": int(source.get("rerun_m2_m3_requested", 0) or 0),
        "rerun_m2_m3_effective_requests_generated": int(
            source.get("rerun_m2_m3_effective_requests_generated", 0) or 0
        ),
        "new_candidate_targets_discovered": int(
            source.get("new_candidate_targets_discovered", 0) or 0
        ),
        "levels_completed": int(source.get("levels_completed", 0) or 0),
        "unique_state_signatures": int(source.get("unique_state_signatures", 0) or 0),
        "gate_passed": gate_passed,
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "policy_result_counted_as_confirmation": False,
        "support": 0,
        "truth_status": SAGE4B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def repeated_action_arg_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    repeats = 0
    previous = ""
    for row in rows:
        key = action_key(
            str(row.get("selected_action", "")),
            dict(row.get("selected_action_args", {}) or {}),
        )
        if previous and key == previous:
            repeats += 1
        previous = key
    return _ratio(repeats, len(rows))


def write_sage4b_progress_stall_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trailing_count(values: Sequence[str], wanted: str) -> int:
    if not wanted:
        return 0
    count = 0
    for value in reversed(list(values)):
        if value != wanted:
            break
        count += 1
    return count


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the SAGE.4b long-horizon progress-stall trigger probe.",
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
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE4B_PROGRESS_STALL_RESULTS_PATH))
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--max-counterfactual-collections",
        type=int,
        default=8,
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
    args = parser.parse_args(argv)
    run_sage4b_progress_stall_trigger_probe(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budget=args.budget,
        max_counterfactual_collections=args.max_counterfactual_collections,
        progress_stall_window=args.progress_stall_window,
        same_action_arg_repeats=args.same_action_arg_repeats,
        low_state_novelty_threshold=args.low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=args.repeated_action_arg_rate_threshold,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
