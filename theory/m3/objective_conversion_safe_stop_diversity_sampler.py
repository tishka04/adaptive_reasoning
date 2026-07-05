"""M3.G3 safe-stop diversity acquisition for objective conversion.

M3.G2 showed that varying lambda/budget/seed for the P3.G1 safe-stop policy
collapses back onto one replay-exact safe-stop. This module treats that as a
diversity diagnostic, not as a failure of the ACTION6-led conversion signal.

G3 samples candidate safe-stop endpoints with explicit novelty constraints and
does not retest objective-conversion sequences. It only answers whether enough
replay-exact, non-terminal, terminal-safe, structurally distinct safe-stops
exist to feed a later M3.G4 validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_conversion_multi_safe_stop_validation import (
    DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH,
)


DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "objective_conversion_safe_stop_diversity_sampler.json"
)
SAFE_STOP_DIVERSITY_SCHEMA_VERSION = (
    "m3.objective_conversion_safe_stop_diversity_sampler.v1"
)

DEFAULT_GAME_ID = "bp35-0a0ad940"
DEFAULT_MIN_DIVERSE_SAFE_STOPS = 3
DEFAULT_MAX_CANDIDATE_PLANS = 32

SAFE_STOP_CANDIDATE_MEASURED = "SAFE_STOP_CANDIDATE_MEASURED"
SAFE_STOP_CANDIDATE_BLOCKED = "SAFE_STOP_CANDIDATE_BLOCKED"
ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE = "ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE"
DUPLICATE_OR_NEAR_DUPLICATE_REJECTED = "DUPLICATE_OR_NEAR_DUPLICATE_REJECTED"
TERMINAL_OR_UNSAFE_REJECTED = "TERMINAL_OR_UNSAFE_REJECTED"
REPLAY_INEXACT_REJECTED = "REPLAY_INEXACT_REJECTED"
HOLD_BASELINE_UNMEASURABLE_REJECTED = "HOLD_BASELINE_UNMEASURABLE_REJECTED"
BLOCKED_SAFE_STOP_CANDIDATE_REJECTED = "BLOCKED_SAFE_STOP_CANDIDATE_REJECTED"

SUFFICIENT_FOR_M3_G4 = "SUFFICIENT_FOR_M3_G4"
INSUFFICIENT_DIVERSITY = "INSUFFICIENT_DIVERSITY"


@dataclass(frozen=True)
class SafeStopCandidatePlan:
    """A candidate endpoint plan whose terminal-safe replay is tested by G3."""

    plan_id: str
    planned_prefix: Tuple[Dict[str, Any], ...]
    sampling_family: str
    anti_attractor_rationale: str
    tie_break_seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "planned_prefix": [dict(step) for step in self.planned_prefix],
            "planned_prefix_len": len(self.planned_prefix),
            "sampling_family": self.sampling_family,
            "anti_attractor_rationale": self.anti_attractor_rationale,
            "tie_break_seed": int(self.tie_break_seed),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        }


def run_objective_conversion_safe_stop_diversity_sampling(
    *,
    source_g2_path: str
    | Path = DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    game_id: str = DEFAULT_GAME_ID,
    min_diverse_safe_stops: int = DEFAULT_MIN_DIVERSE_SAFE_STOPS,
    max_candidate_plans: int | None = DEFAULT_MAX_CANDIDATE_PLANS,
    candidate_plans: Sequence[SafeStopCandidatePlan | Mapping[str, Any]]
    | None = None,
    candidate_executor: Callable[[SafeStopCandidatePlan], Mapping[str, Any]]
    | None = None,
) -> Dict[str, Any]:
    source_payload = _load_json(source_g2_path)
    _validate_source_g2_payload(source_payload)

    plans = normalize_candidate_plans(
        candidate_plans
        if candidate_plans is not None
        else generate_safe_stop_candidate_plans(source_payload)
    )
    if max_candidate_plans is not None:
        plans = plans[: max(0, int(max_candidate_plans))]

    if candidate_executor is None:
        candidate_executor = _make_default_candidate_executor(
            environments_dir=environments_dir,
            game_id=game_id,
        )

    raw_records = [dict(candidate_executor(plan)) for plan in plans]
    classified_records, accepted = classify_safe_stop_candidates(
        raw_records,
        min_diverse_safe_stops=int(min_diverse_safe_stops),
    )
    diversity_status = (
        SUFFICIENT_FOR_M3_G4
        if len(accepted) >= int(min_diverse_safe_stops)
        else INSUFFICIENT_DIVERSITY
    )

    return {
        "config": {
            "schema_version": SAFE_STOP_DIVERSITY_SCHEMA_VERSION,
            "source_g2_path": str(source_g2_path),
            "environments_dir": (
                None if environments_dir is None else str(environments_dir)
            ),
            "game_id": game_id,
            "inputs_read": ["M3.G2"],
            "artifacts_not_modified": ["M2", "M3.G2", "A32", "A33"],
            "stage_produces": "safe_stop_diversity_candidates_only",
            "objective_conversion_sequences_tested": False,
            "support_events_counted_as_scientific_support": False,
        },
        "source_g2_diversity_diagnostic": source_g2_diversity_diagnostic(
            source_payload
        ),
        "safe_stop_candidate_plans": [plan.to_dict() for plan in plans],
        "safe_stop_candidate_records": classified_records,
        "accepted_diverse_safe_stops": accepted,
        "rejected_safe_stop_candidates": [
            record
            for record in classified_records
            if str(record.get("acceptance_status", ""))
            != ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE
        ],
        "summary": summarize_safe_stop_diversity(
            plans=plans,
            records=classified_records,
            accepted=accepted,
            min_diverse_safe_stops=int(min_diverse_safe_stops),
            diversity_status=diversity_status,
        ),
        "diversity_status": diversity_status,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "safe_stop_diversity_counted_as_confirmation": False,
        "objective_conversion_sequences_tested": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "diversity_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def generate_safe_stop_candidate_plans(
    source_payload: Mapping[str, Any],
) -> Tuple[SafeStopCandidatePlan, ...]:
    base_prefix = source_unique_prefix(source_payload)
    plans: List[SafeStopCandidatePlan] = []

    for length in _candidate_lengths(len(base_prefix)):
        plans.append(
            SafeStopCandidatePlan(
                plan_id=f"m3_g3::base_truncation::{length:02d}",
                planned_prefix=tuple(dict(step) for step in base_prefix[:length]),
                sampling_family="base_prefix_truncation",
                anti_attractor_rationale=(
                    "Stop before the deterministic P3.G1 attractor endpoint."
                ),
                tie_break_seed=length,
            )
        )

    for length in (4, 6, 8, 10, 12, 14):
        plans.append(
            SafeStopCandidatePlan(
                plan_id=f"m3_g3::alternating_action4_first::{length:02d}",
                planned_prefix=alternating_prefix("ACTION4", "ACTION3", length),
                sampling_family="phase_shifted_relation_prefix",
                anti_attractor_rationale=(
                    "Invert the ACTION3/ACTION4 phase to avoid the known attractor."
                ),
                tie_break_seed=length,
            )
        )

    for action in ("ACTION3", "ACTION4"):
        for length in (3, 5, 7):
            plans.append(
                SafeStopCandidatePlan(
                    plan_id=f"m3_g3::{action.lower()}_burst::{length:02d}",
                    planned_prefix=tuple(
                        {"action": action, "action_args": {}} for _ in range(length)
                    ),
                    sampling_family="single_action_burst",
                    anti_attractor_rationale=(
                        "Sample a relation-progress endpoint outside strict alternation."
                    ),
                    tie_break_seed=length,
                )
            )

    action6_prefixes = (
        ("ACTION6",),
        ("ACTION6", "ACTION3"),
        ("ACTION6", "ACTION4"),
        ("ACTION3", "ACTION6"),
        ("ACTION4", "ACTION6"),
        ("ACTION6", "ACTION3", "ACTION4"),
        ("ACTION6", "ACTION4", "ACTION3"),
    )
    for index, actions in enumerate(action6_prefixes, start=1):
        plans.append(
            SafeStopCandidatePlan(
                plan_id=f"m3_g3::action6_perturbation::{index:02d}",
                planned_prefix=tuple(
                    {"action": action, "action_args": {}} for action in actions
                ),
                sampling_family="action6_perturbation",
                anti_attractor_rationale=(
                    "Probe safe endpoints that include ACTION6 before validation."
                ),
                tie_break_seed=index,
            )
        )

    return tuple(deduplicate_plans(plans))


def normalize_candidate_plans(
    plans: Sequence[SafeStopCandidatePlan | Mapping[str, Any]],
) -> Tuple[SafeStopCandidatePlan, ...]:
    normalized: List[SafeStopCandidatePlan] = []
    for index, plan in enumerate(plans, start=1):
        if isinstance(plan, SafeStopCandidatePlan):
            normalized.append(plan)
            continue
        normalized.append(
            SafeStopCandidatePlan(
                plan_id=str(plan.get("plan_id", f"m3_g3::plan::{index:03d}")),
                planned_prefix=tuple(
                    dict(step) for step in plan.get("planned_prefix", []) or []
                ),
                sampling_family=str(plan.get("sampling_family", "external")),
                anti_attractor_rationale=str(
                    plan.get("anti_attractor_rationale", "")
                ),
                tie_break_seed=int(plan.get("tie_break_seed", index) or index),
            )
        )
    return tuple(normalized)


def classify_safe_stop_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    min_diverse_safe_stops: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen_state_hashes: set[str] = set()
    seen_prefix_hashes: set[str] = set()
    seen_relation_signatures: set[str] = set()
    classified: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []

    for record in records:
        row = {
            **dict(record),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            "wrong_confirmations": 0,
            "safe_stop_candidate_counted_as_confirmation": False,
        }
        acceptance_status = ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE
        rejection_reason = ""
        if not bool(row.get("execution_performed", False)):
            acceptance_status = BLOCKED_SAFE_STOP_CANDIDATE_REJECTED
            rejection_reason = str(row.get("blocked_reason", "blocked_candidate"))
        elif not bool(row.get("replay_exact", False)):
            acceptance_status = REPLAY_INEXACT_REJECTED
            rejection_reason = "candidate_prefix_not_replay_exact"
        elif not bool(row.get("non_terminal", False)) or not bool(
            row.get("terminal_safe", False)
        ):
            acceptance_status = TERMINAL_OR_UNSAFE_REJECTED
            rejection_reason = "terminal_or_terminal_unsafe_endpoint"
        elif not bool(row.get("hold_baseline_measurable", False)):
            acceptance_status = HOLD_BASELINE_UNMEASURABLE_REJECTED
            rejection_reason = "hold_baseline_unmeasurable"
        elif (
            str(row.get("safe_stop_state_hash", "")) in seen_state_hashes
            or str(row.get("captured_prefix_hash", "")) in seen_prefix_hashes
            or str(row.get("relation_state_signature", ""))
            in seen_relation_signatures
        ):
            acceptance_status = DUPLICATE_OR_NEAR_DUPLICATE_REJECTED
            rejection_reason = "state_prefix_or_relation_signature_not_novel"

        row["acceptance_status"] = acceptance_status
        row["rejection_reason"] = rejection_reason
        row["candidate_needed_for_m3_g4"] = (
            len(accepted) < int(min_diverse_safe_stops)
            and acceptance_status == ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE
        )
        classified.append(row)

        if acceptance_status == ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE:
            seen_state_hashes.add(str(row.get("safe_stop_state_hash", "")))
            seen_prefix_hashes.add(str(row.get("captured_prefix_hash", "")))
            seen_relation_signatures.add(str(row.get("relation_state_signature", "")))
            accepted.append(accepted_safe_stop_public_record(row, len(accepted) + 1))

    return classified, accepted


def accepted_safe_stop_public_record(
    row: Mapping[str, Any],
    index: int,
) -> Dict[str, Any]:
    return {
        "safe_stop_id": f"m3_g3::safe_stop::{index:03d}",
        "source_plan_id": str(row.get("plan_id", "")),
        "sampling_family": str(row.get("sampling_family", "")),
        "captured_prefix": [dict(step) for step in row.get("captured_prefix", []) or []],
        "captured_prefix_len": int(row.get("captured_prefix_len", 0) or 0),
        "captured_prefix_hash": str(row.get("captured_prefix_hash", "")),
        "safe_stop_state_hash": str(row.get("safe_stop_state_hash", "")),
        "relation_state_signature": str(row.get("relation_state_signature", "")),
        "terminal_horizon_estimate": dict(row.get("terminal_horizon_estimate", {}) or {}),
        "hold_baseline_terminal_adjusted_progress": float(
            row.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
        ),
        "hold_baseline_levels_completed": int(
            row.get("hold_baseline_levels_completed", 0) or 0
        ),
        "hold_baseline_terminal": bool(row.get("hold_baseline_terminal", False)),
        "replay_exact": bool(row.get("replay_exact", False)),
        "non_terminal": bool(row.get("non_terminal", False)),
        "terminal_safe": bool(row.get("terminal_safe", False)),
        "hold_baseline_measurable": bool(
            row.get("hold_baseline_measurable", False)
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "safe_stop_record_counted_as_confirmation": False,
    }


def _make_default_candidate_executor(
    *,
    environments_dir: str | Path | None,
    game_id: str,
) -> Callable[[SafeStopCandidatePlan], Mapping[str, Any]]:
    def executor(plan: SafeStopCandidatePlan) -> Mapping[str, Any]:
        return execute_safe_stop_candidate_plan(
            plan,
            environments_dir=environments_dir,
            game_id=game_id,
        )

    return executor


def execute_safe_stop_candidate_plan(
    plan: SafeStopCandidatePlan,
    *,
    environments_dir: str | Path | None,
    game_id: str,
    terminal_budget_estimate: int = 64,
) -> Dict[str, Any]:
    from theory.m2.m3_execution_smoke import _make_env, _reset_env
    from theory.m1.polymorphic_a25_adapter import _step_env_action
    from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
    from theory.p1.bp35_sage_candidate_policy_probe import (
        measure_probe_metrics,
        state_signature,
    )
    from theory.p3.abstract_mechanic_policy_probe import (
        action_has_actor_effect,
        action_has_relation_effect,
        concrete_named_action,
        deterministic_action_args,
        is_game_over,
    )
    from theory.p3.objective_aware_abstract_policy_probe import (
        DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH,
        summarize_objective_aware_steps,
    )
    from theory.p3.terminal_horizon_estimator import estimate_terminal_horizon
    from theory.real_env_option_adapter import snapshot_frame

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    adapter_payload = _load_json(DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH)
    adapter = dict(adapter_payload.get("objective_aware_policy_adapter", {}) or {})

    def execute_once() -> Dict[str, Any]:
        env = _make_env(game_id, env_dir)
        frame = _reset_env(env)
        grid_history: List[Any] = [snapshot_frame(frame).grid]
        executed_steps: List[Dict[str, Any]] = []
        captured_prefix: List[Dict[str, Any]] = []
        seen_states: set[str] = set()
        initial = snapshot_frame(frame)
        seen_states.add(
            state_signature(initial.grid, initial.levels_completed, initial.game_state)
        )
        for index, step in enumerate(plan.planned_prefix):
            before = snapshot_frame(frame)
            if is_game_over(before.game_state):
                return {
                    "blocked": True,
                    "blocked_reason": "terminal_reached_before_prefix_complete",
                    "executed_steps": executed_steps,
                    "captured_prefix": captured_prefix,
                    "grid_history": grid_history,
                }
            action_name = str(step.get("action", ""))
            requested_args = dict(step.get("action_args", {}) or {})
            action_args = (
                requested_args
                if requested_args
                else deterministic_action_args(
                    list(_valid_actions(env)),
                    action_name,
                    int(plan.tie_break_seed) + index,
                )
            )
            selected = concrete_named_action(
                list(_valid_actions(env)),
                action_name,
                action_args,
            )
            if selected is None:
                return {
                    "blocked": True,
                    "blocked_reason": f"action_unavailable::{action_name}",
                    "executed_steps": executed_steps,
                    "captured_prefix": captured_prefix,
                    "grid_history": grid_history,
                }
            after_frame = _step_env_action(env, selected)
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
            actual_action = str(getattr(selected, "name", action_name))
            actual_args = dict(getattr(selected, "action_args", {}) or action_args)
            measurements = measure_probe_metrics(before.grid, after.grid, actual_args)
            after_signature = state_signature(
                after.grid,
                after.levels_completed,
                after.game_state,
            )
            changed_pixels = float(
                measurements["changed_pixels"].get("changed_pixels", 0) or 0
            )
            relation_expected = action_has_relation_effect(adapter, actual_action)
            actor_expected = action_has_actor_effect(adapter, actual_action)
            terminal = is_game_over(after.game_state)
            cycle = after_signature in seen_states
            useful_new_state = bool(
                changed_pixels > 0
                and not cycle
                and after.levels_completed >= before.levels_completed
                and not terminal
            )
            captured_prefix.append(
                {"action": actual_action, "action_args": dict(actual_args)}
            )
            executed_steps.append(
                {
                    "step": index,
                    "policy_selected_action": actual_action,
                    "action_args": dict(actual_args),
                    "changed_pixels": changed_pixels,
                    "actor_relation_delta_count": int(
                        relation_expected and changed_pixels > 0
                    ),
                    "action_effect_usefulness": int(
                        actor_expected and changed_pixels > 0
                    ),
                    "new_relation_state": int(relation_expected and useful_new_state),
                    "useful_new_state": useful_new_state,
                    "dead_end_or_cycle": cycle,
                    "state_signature_after": after_signature,
                    "levels_after": int(after.levels_completed),
                    "game_state_after": str(after.game_state),
                    "terminal_state_after": terminal,
                    "measurements": measurements,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                    "wrong_confirmations": 0,
                }
            )
            frame = after_frame
            grid_history.append(after.grid)
            seen_states.add(after_signature)
            if terminal:
                break
        return {
            "blocked": False,
            "blocked_reason": "",
            "executed_steps": executed_steps,
            "captured_prefix": captured_prefix,
            "grid_history": grid_history,
        }

    first = execute_once()
    if first.get("blocked"):
        return blocked_candidate_record(plan, str(first.get("blocked_reason", "")))
    second = execute_once()

    steps = [dict(step) for step in first.get("executed_steps", []) or []]
    captured_prefix = [dict(step) for step in first.get("captured_prefix", []) or []]
    final = steps[-1] if steps else {}
    prefix_hash = prefix_hash_for_steps(captured_prefix)
    state_hash = str(final.get("state_signature_after", "safe_stop::initial"))
    replay_exact = bool(
        prefix_hash_for_steps(second.get("captured_prefix", []) or []) == prefix_hash
        and str(
            (second.get("executed_steps", []) or [{}])[-1].get(
                "state_signature_after",
                "safe_stop::initial",
            )
        )
        == state_hash
    )
    terminal = bool(final.get("terminal_state_after", False))
    horizon = estimate_terminal_horizon(
        observation=(first.get("grid_history", []) or [None])[-1],
        history=(first.get("grid_history", []) or [])[:-1],
        policy_state={"env_actions_executed": len(steps)},
        terminal_budget_estimate=terminal_budget_estimate,
    ).to_dict()
    baseline = summarize_objective_aware_steps(
        condition="m3_g3_safe_stop_candidate",
        steps=steps,
        budget=len(steps),
        tie_break_seed=int(plan.tie_break_seed),
        stop_event={
            "stop_triggered": True,
            "trigger_reason": "m3_g3_candidate_endpoint",
            "terminal_horizon_source": str(horizon.get("source", "")),
            "estimated_moves_remaining": horizon.get("estimated_moves_remaining"),
        },
    )
    hold_taps = float(baseline.get("terminal_adjusted_progress", 0.0) or 0.0)
    terminal_safe = bool(
        not terminal
        and (
            horizon.get("estimated_moves_remaining") is None
            or int(horizon.get("estimated_moves_remaining") or 0) > 0
        )
    )
    return {
        **plan.to_dict(),
        "status": SAFE_STOP_CANDIDATE_MEASURED,
        "execution_performed": True,
        "blocked_reason": "",
        "captured_prefix": captured_prefix,
        "captured_prefix_len": len(captured_prefix),
        "captured_prefix_hash": prefix_hash,
        "safe_stop_state_hash": state_hash,
        "relation_state_signature": relation_state_signature(steps, horizon),
        "terminal_horizon_estimate": horizon,
        "hold_baseline_terminal_adjusted_progress": hold_taps,
        "hold_baseline_levels_completed": int(
            baseline.get("final_levels_completed", 0) or 0
        ),
        "hold_baseline_terminal": bool(
            baseline.get("terminal_state_after_rollout", False)
        ),
        "replay_exact": replay_exact,
        "non_terminal": not terminal,
        "terminal_safe": terminal_safe,
        "hold_baseline_measurable": isinstance(hold_taps, float) and not terminal,
        "endpoint_summary": baseline,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def blocked_candidate_record(
    plan: SafeStopCandidatePlan,
    reason: str,
) -> Dict[str, Any]:
    return {
        **plan.to_dict(),
        "status": SAFE_STOP_CANDIDATE_BLOCKED,
        "execution_performed": False,
        "blocked_reason": reason,
        "captured_prefix": [],
        "captured_prefix_len": 0,
        "captured_prefix_hash": "",
        "safe_stop_state_hash": "",
        "relation_state_signature": "",
        "terminal_horizon_estimate": {},
        "hold_baseline_terminal_adjusted_progress": 0.0,
        "hold_baseline_levels_completed": 0,
        "hold_baseline_terminal": False,
        "replay_exact": False,
        "non_terminal": False,
        "terminal_safe": False,
        "hold_baseline_measurable": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def relation_state_signature(
    steps: Sequence[Mapping[str, Any]],
    horizon: Mapping[str, Any],
) -> str:
    return "relation::" + json.dumps(
        {
            "actor_relation_delta_count": sum(
                int(step.get("actor_relation_delta_count", 0) or 0) for step in steps
            ),
            "new_relation_states": sum(
                int(step.get("new_relation_state", 0) or 0) for step in steps
            ),
            "action_effect_usefulness": sum(
                int(step.get("action_effect_usefulness", 0) or 0) for step in steps
            ),
            "terminal_remaining": horizon.get("estimated_moves_remaining"),
            "horizon_source": horizon.get("source"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def summarize_safe_stop_diversity(
    *,
    plans: Sequence[SafeStopCandidatePlan],
    records: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    min_diverse_safe_stops: int,
    diversity_status: str,
) -> Dict[str, Any]:
    unique_states = {
        str(record.get("safe_stop_state_hash", ""))
        for record in records
        if str(record.get("safe_stop_state_hash", ""))
    }
    duplicate_rejected = [
        record
        for record in records
        if str(record.get("acceptance_status", ""))
        == DUPLICATE_OR_NEAR_DUPLICATE_REJECTED
    ]
    return {
        "safe_stop_candidates_planned": len(plans),
        "safe_stop_candidates_executed": len(
            [record for record in records if bool(record.get("execution_performed", False))]
        ),
        "unique_safe_stop_candidates": len(unique_states),
        "accepted_diverse_safe_stops": len(accepted),
        "min_diverse_safe_stops_required": int(min_diverse_safe_stops),
        "duplicate_or_near_duplicate_rejected": len(duplicate_rejected),
        "terminal_or_unsafe_rejected": len(
            [
                record
                for record in records
                if str(record.get("acceptance_status", ""))
                == TERMINAL_OR_UNSAFE_REJECTED
            ]
        ),
        "replay_inexact_rejected": len(
            [
                record
                for record in records
                if str(record.get("acceptance_status", ""))
                == REPLAY_INEXACT_REJECTED
            ]
        ),
        "blocked_candidates_rejected": len(
            [
                record
                for record in records
                if str(record.get("acceptance_status", ""))
                == BLOCKED_SAFE_STOP_CANDIDATE_REJECTED
            ]
        ),
        "diversity_status": diversity_status,
        "ready_for_m3_g4": diversity_status == SUFFICIENT_FOR_M3_G4,
        "objective_conversion_sequences_tested": False,
        "execution_performed": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "safe_stop_diversity_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "diversity_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def source_g2_diversity_diagnostic(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    return {
        "source_validation_outcome_status": str(
            summary.get("validation_outcome_status", "")
        ),
        "source_safe_stop_capture_specs_planned": int(
            summary.get("safe_stop_capture_specs_planned", 0) or 0
        ),
        "source_unique_safe_stop_captures": int(
            summary.get("unique_safe_stop_captures", 0) or 0
        ),
        "source_duplicate_safe_stop_captures": int(
            summary.get("duplicate_safe_stop_captures", 0) or 0
        ),
        "source_safe_stop_context_diversity": str(
            summary.get("safe_stop_context_diversity", "")
        ),
        "source_g2_counted_as_signal_refutation": False,
        "source_g2_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def source_unique_prefix(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    for capture in payload.get("safe_stop_captures", []) or []:
        if not isinstance(capture, Mapping):
            continue
        if bool(capture.get("is_duplicate_safe_stop", False)):
            continue
        prefix = capture.get("captured_prefix", []) or []
        if prefix:
            return tuple(dict(step) for step in prefix)
    for capture in payload.get("safe_stop_captures", []) or []:
        if isinstance(capture, Mapping) and capture.get("captured_prefix"):
            return tuple(dict(step) for step in capture.get("captured_prefix", []) or [])
    return tuple()


def alternating_prefix(action_a: str, action_b: str, length: int) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        {"action": action_a if index % 2 == 0 else action_b, "action_args": {}}
        for index in range(max(0, int(length)))
    )


def deduplicate_plans(
    plans: Sequence[SafeStopCandidatePlan],
) -> Tuple[SafeStopCandidatePlan, ...]:
    seen: set[str] = set()
    unique: List[SafeStopCandidatePlan] = []
    for plan in plans:
        key = json.dumps(list(plan.planned_prefix), sort_keys=True)
        if not plan.planned_prefix or key in seen:
            continue
        seen.add(key)
        unique.append(plan)
    return tuple(unique)


def prefix_hash_for_steps(steps: Sequence[Mapping[str, Any]]) -> str:
    raw = [
        {
            "action": str(step.get("action", "")),
            "action_args": dict(step.get("action_args", {}) or {}),
        }
        for step in steps
    ]
    return "prefix::" + json.dumps(raw, sort_keys=True, separators=(",", ":"))


def _candidate_lengths(max_length: int) -> Tuple[int, ...]:
    if max_length <= 0:
        return tuple()
    base = [4, 6, 8, 10, 12, 14, max_length]
    return tuple(sorted({length for length in base if 1 <= length <= max_length}))


def _validate_source_g2_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G2 source support must remain 0")
    if bool(payload.get("revision_performed", False)) or bool(
        summary.get("revision_performed", False)
    ):
        raise ValueError("M3.G2 source cannot be revised")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("M3.G2 source must not write A32/A33")
    if bool(payload.get("validation_outcome_status_counted_as_scientific_verdict", False)):
        raise ValueError("M3.G2 source status cannot be scientific verdict")
    if not payload.get("safe_stop_captures"):
        raise ValueError("M3.G3 requires M3.G2 safe-stop captures")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_objective_conversion_safe_stop_diversity_sampling(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3.G3 safe-stop diversity acquisition.",
    )
    parser.add_argument(
        "--source-g2",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument(
        "--min-diverse-safe-stops",
        type=int,
        default=DEFAULT_MIN_DIVERSE_SAFE_STOPS,
    )
    parser.add_argument(
        "--max-candidate-plans",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_PLANS,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_safe_stop_diversity_sampling(
        source_g2_path=args.source_g2,
        environments_dir=args.environments_dir,
        game_id=args.game_id,
        min_diverse_safe_stops=args.min_diverse_safe_stops,
        max_candidate_plans=args.max_candidate_plans,
    )
    write_objective_conversion_safe_stop_diversity_sampling(payload, args.out)
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
