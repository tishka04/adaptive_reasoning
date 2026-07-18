"""SAGE.5e distributed live mini-frontier generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableSet, Sequence

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.m2.schema import M2_READY_FOR_M3_STATUS, M2_TRUTH_STATUS
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .known_game_closed_loop_scaffold import (
    DEFAULT_M2_FUSED_REQUESTS_PATH,
    DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH,
    DEFAULT_M3_FUSED_RESULTS_PATH,
    DEFAULT_P1_POLICY_PROBE_PATH,
    DEFAULT_P1_UTILITY_HANDOFF_PATH,
)
from .live_mini_frontier_generation import (
    DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH,
    generate_live_mini_frontier,
)
from .live_mini_frontier_m3_executor import (
    execute_live_prefix_mini_frontier_request,
)
from .live_prefix_counterfactual_collector import (
    LivePrefixAction,
    state_signature_from_frame,
)
from .policy_loop_guard import action_args, action_name
from .progress_stall_trigger import (
    DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    DEFAULT_PROGRESS_STALL_WINDOW,
    DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    DEFAULT_SAME_ACTION_ARG_REPEATS,
)
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    SUBGOAL_RERUN,
    run_sage3_subgoal_switch_probe,
)
from .switch_attribution_placeholder_audit import (
    DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH,
)
from .unknown_game_bounded_probe import (
    DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    DEFAULT_UNKNOWN_GAME_ID,
)


DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage5e_distributed_live_mini_frontier_results.json"
)
SAGE5E_SCHEMA_VERSION = "sage.distributed_live_mini_frontier_generation.v1"
SAGE5E_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5E"
SAGE5E_GENERATED_AND_EXECUTED = (
    "SAGE_DISTRIBUTED_LIVE_MINI_FRONTIER_GENERATED_AND_EXECUTED_CANDIDATE_ONLY"
)
SAGE5E_GENERATION_NOT_DISTRIBUTED = (
    "SAGE_DISTRIBUTED_LIVE_MINI_FRONTIER_GENERATION_NOT_DISTRIBUTED_CANDIDATE_ONLY"
)
DEFAULT_REQUESTS_PER_BUDGET = 8
DEFAULT_EXECUTE_REQUESTS_PER_BUDGET = 2

EnvFactory = Callable[[str], Any]


def run_sage5e_distributed_live_mini_frontier_generation(
    *,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    source_sage5_path: str | Path = DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH,
    source_sage5b_path: str | Path = DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH,
    source_sage5c_path: str | Path = DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    game_id: str | None = None,
    budgets: Sequence[int] | None = None,
    requests_per_budget: int = DEFAULT_REQUESTS_PER_BUDGET,
    execute_requests_per_budget: int = DEFAULT_EXECUTE_REQUESTS_PER_BUDGET,
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    progress_stall_window: int = DEFAULT_PROGRESS_STALL_WINDOW,
    same_action_arg_repeats: int = DEFAULT_SAME_ACTION_ARG_REPEATS,
    low_state_novelty_threshold: int = DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    repeated_action_arg_rate_threshold: float = (
        DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD
    ),
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    source_sage5 = _load_optional_json(source_sage5_path)
    source_sage5b = _load_optional_json(source_sage5b_path)
    source_sage5c = _load_optional_json(source_sage5c_path)
    source_config = dict(source_sage5.get("config", {}) or {})
    resolved_game_id = str(
        game_id or source_config.get("game_id", "") or DEFAULT_UNKNOWN_GAME_ID
    )
    resolved_budgets = [
        int(value)
        for value in (
            budgets
            if budgets is not None
            else source_config.get("budgets", [50, 150, 300])
        )
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    seen_dedup_keys: set[str] = set()
    per_budget: List[Dict[str, Any]] = []
    all_hypotheses: List[Dict[str, Any]] = []
    all_requests: List[Dict[str, Any]] = []

    for budget in resolved_budgets:
        run = run_sage3_subgoal_switch_probe(
            m2_fused_requests_path=m2_fused_requests_path,
            m3_fused_results_path=m3_fused_results_path,
            m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
            p1_policy_probe_path=p1_policy_probe_path,
            p1_utility_handoff_path=p1_utility_handoff_path,
            environments_dir=env_dir,
            output_path=None,
            game_id=resolved_game_id,
            budget=int(budget),
            max_counterfactual_collections=max_counterfactual_collections,
            env_factory=env_factory,
            enable_progress_stall_trigger=True,
            progress_stall_window=progress_stall_window,
            same_action_arg_repeats=same_action_arg_repeats,
            low_state_novelty_threshold=low_state_novelty_threshold,
            repeated_action_arg_rate_threshold=repeated_action_arg_rate_threshold,
        )
        generation = generate_distributed_live_mini_frontiers_for_run(
            game_id=resolved_game_id,
            budget=int(budget),
            run=run,
            environments_dir=env_dir,
            requests_per_budget=requests_per_budget,
            seen_dedup_keys=seen_dedup_keys,
            env_factory=env_factory,
        )
        all_hypotheses.extend(generation["mini_frontier_hypotheses"])
        all_requests.extend(generation["mini_frontier_m3_requests"])
        per_budget.append(_budget_generation_record(int(budget), run, generation))

    execution_requests = select_sage5e_execution_requests(
        all_requests,
        per_budget_limit=execute_requests_per_budget,
    )
    experiments: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for request in execution_requests:
        row = execute_live_prefix_mini_frontier_request(
            request,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        row = _retag_execution_row(row)
        if row["execution_status"] == "EXECUTED":
            experiments.append(row)
        else:
            blocked.append(row)

    summary = _build_summary(
        budgets=resolved_budgets,
        per_budget=per_budget,
        all_hypotheses=all_hypotheses,
        all_requests=all_requests,
        execution_requests=execution_requests,
        experiments=experiments,
        blocked=blocked,
        source_sage5=source_sage5,
        source_sage5b=source_sage5b,
        source_sage5c=source_sage5c,
        requests_per_budget=requests_per_budget,
        execute_requests_per_budget=execute_requests_per_budget,
    )
    payload = {
        "config": {
            "schema_version": SAGE5E_SCHEMA_VERSION,
            "game_id": resolved_game_id,
            "budgets": resolved_budgets,
            "requests_per_budget": int(requests_per_budget),
            "execute_requests_per_budget": int(execute_requests_per_budget),
            "source_sage5_path": str(source_sage5_path),
            "source_sage5b_path": str(source_sage5b_path),
            "source_sage5c_path": str(source_sage5c_path),
            "environments_dir": str(env_dir),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "distributed_generation": True,
            "dedup_policy": (
                "context_hash,target_action,target_args,diff_signature"
            ),
            "benchmark_run": False,
            "inputs_read": ["SAGE.5", "SAGE.5b", "SAGE.5c"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5_context": _source_context(source_sage5, "SAGE.5"),
        "source_sage5b_context": _source_context(source_sage5b, "SAGE.5b"),
        "source_sage5c_context": _source_context(source_sage5c, "SAGE.5c"),
        "per_budget_results": per_budget,
        "mini_frontier_hypotheses": all_hypotheses,
        "mini_frontier_m3_requests": all_requests,
        "selected_execution_requests": [dict(row) for row in execution_requests],
        "controlled_experiments": experiments,
        "blocked_replay_events": blocked,
        "summary": summary,
        "comparison": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5E_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "generated_requests_counted_as_support": False,
        "mini_frontier_counted_as_evidence": False,
        "mini_frontier_execution_counted_as_evidence": False,
        "support_events_counted_as_support": False,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5e_distributed_live_mini_frontier_results(payload, output_path)
    return payload


def generate_distributed_live_mini_frontiers_for_run(
    *,
    game_id: str,
    budget: int,
    run: Mapping[str, Any],
    environments_dir: str | Path,
    requests_per_budget: int,
    seen_dedup_keys: MutableSet[str] | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Generate non-duplicate live mini-frontiers up to a per-budget quota."""
    seen = seen_dedup_keys if seen_dedup_keys is not None else set()
    env_dir = Path(environments_dir)
    try:
        env = env_factory(game_id) if env_factory is not None else _make_real_env(game_id, env_dir)
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return _empty_generation(blocked_reason=f"env_setup_failed:{exc}")

    prefix: List[LivePrefixAction] = []
    hypotheses: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    placeholders_seen = 0
    duplicates_skipped = 0
    quota_exhausted = False

    for row in run.get("steps", []) or []:
        if int(row.get("env_actions", 0) or 0) <= 0:
            continue
        selected_action = str(row.get("selected_action", ""))
        selected_args = dict(row.get("selected_action_args", {}) or {})
        before_frame = frame
        before = snapshot_frame(before_frame)
        valid_actions = _valid_actions(env)
        selected = _select_action(valid_actions, selected_action, selected_args)
        if selected is None:
            diagnostics.append(
                {
                    "step": int(row.get("step", 0) or 0),
                    "blocked_reason": "replay_action_not_available",
                    "selected_action": selected_action,
                    "selected_action_args": selected_args,
                }
            )
            break
        try:
            after_frame = _step_env_action(env, selected)
        except Exception as exc:  # pragma: no cover - integration failure path
            diagnostics.append(
                {
                    "step": int(row.get("step", 0) or 0),
                    "blocked_reason": f"replay_step_failed:{exc}",
                    "selected_action": selected_action,
                    "selected_action_args": selected_args,
                }
            )
            break
        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        is_placeholder = bool(row.get("placeholder_action_used", False)) or (
            str(row.get("selected_subgoal", "")) == SUBGOAL_RERUN
        )
        if is_placeholder:
            placeholders_seen += 1
            if len(requests) >= int(requests_per_budget):
                quota_exhausted = True
            else:
                frontier = generate_live_mini_frontier(
                    game_id=game_id,
                    budget=budget,
                    step=int(row.get("step", 0) or 0),
                    before_frame=before_frame,
                    after_frame=after_frame,
                    before_snapshot=before,
                    after_snapshot=after,
                    action_name=selected_action,
                    action_args=selected_args,
                    valid_actions_before=valid_actions,
                    prefix_actions=prefix,
                )
                frontier = _retag_frontier(frontier, game_id=game_id, budget=budget)
                dedup_key = live_mini_frontier_dedup_key(
                    frontier["hypothesis"],
                    frontier["m3_request"],
                )
                if dedup_key in seen:
                    duplicates_skipped += 1
                    diagnostics.append(
                        {
                            **dict(frontier["diagnostics"]),
                            "dedup_status": "duplicate_skipped",
                            "dedup_key": dedup_key,
                        }
                    )
                else:
                    seen.add(dedup_key)
                    _attach_dedup(frontier, dedup_key=dedup_key)
                    hypotheses.append(frontier["hypothesis"])
                    request = frontier["m3_request"]
                    if request["status"] == M2_READY_FOR_M3_STATUS:
                        requests.append(request)
                    diagnostics.append(
                        {
                            **dict(frontier["diagnostics"]),
                            "dedup_status": "selected_non_redundant",
                            "dedup_key": dedup_key,
                        }
                    )
        prefix.append(
            LivePrefixAction(name=selected_action, action_args=dict(selected_args))
        )
        frame = after_frame

    return {
        "placeholder_switches_seen": placeholders_seen,
        "mini_frontier_hypotheses": hypotheses,
        "mini_frontier_m3_requests": requests,
        "generation_diagnostics": diagnostics,
        "effective_requests_generated": len(requests),
        "requests_per_budget_quota": int(requests_per_budget),
        "duplicate_candidates_skipped": duplicates_skipped,
        "generation_budget_exhausted": quota_exhausted,
        "dedup_keys_selected": len({request["dedup_key"] for request in requests}),
        "support": 0,
        "truth_status": SAGE5E_TRUTH_STATUS,
    }


def live_mini_frontier_dedup_key(
    hypothesis: Mapping[str, Any],
    request: Mapping[str, Any],
) -> str:
    diff = hypothesis.get("diff_summary", {}) or {}
    signature = {
        "context_hash": str(request.get("context_snapshot_hash", "")),
        "target_action": str(request.get("target_action", "")),
        "target_args": dict(request.get("target_action_args", {}) or {}),
        "diff_signature": diff_signature(diff),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def diff_signature(diff: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "changed_cells": int(diff.get("changed_cells", 0) or 0),
        "changed_bbox": diff.get("changed_bbox"),
        "color_transitions": dict(diff.get("color_transitions", {}) or {}),
        "component_delta_by_color": dict(
            diff.get("component_delta_by_color", {}) or {}
        ),
        "terminal_after": bool(diff.get("terminal_after", False)),
        "levels_delta": int(diff.get("levels_delta", 0) or 0),
    }


def select_sage5e_execution_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    per_budget_limit: int = DEFAULT_EXECUTE_REQUESTS_PER_BUDGET,
) -> tuple[Dict[str, Any], ...]:
    selected: List[Dict[str, Any]] = []
    budgets = sorted({_budget_from_request(row) for row in requests})
    for budget in budgets:
        rows = [dict(row) for row in requests if _budget_from_request(row) == budget]
        rows.sort(key=lambda row: int(row.get("source_step", 0) or 0))
        budget_selected: List[Dict[str, Any]] = []

        def add_first(predicate: Callable[[Mapping[str, Any]], bool]) -> None:
            match = next((row for row in rows if predicate(row) and row not in budget_selected), None)
            if match is not None and len(budget_selected) < per_budget_limit:
                budget_selected.append(match)

        add_first(lambda row: str(row.get("target_action", "")) == "ACTION5")
        add_first(lambda row: str(row.get("target_action", "")) == "ACTION6")
        add_first(
            lambda row: str(row.get("hypothesis_family", ""))
            == "object_delta_candidate"
        )
        add_first(
            lambda row: str(row.get("hypothesis_family", ""))
            == "local_patch_change_candidate"
        )
        for row in rows:
            if len(budget_selected) >= per_budget_limit:
                break
            if row not in budget_selected:
                budget_selected.append(row)
        selected.extend(budget_selected[: max(0, int(per_budget_limit))])
    return tuple(selected)


def write_sage5e_distributed_live_mini_frontier_results(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _budget_generation_record(
    budget: int,
    run: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> Dict[str, Any]:
    summary = dict(run.get("summary", {}) or {})
    placeholders = int(generation.get("placeholder_switches_seen", 0) or 0)
    effective = int(generation.get("effective_requests_generated", 0) or 0)
    return {
        "budget": int(budget),
        "source_metrics": {
            "subgoal_switches": int(summary.get("subgoal_switches", 0) or 0),
            "placeholder_switches_seen": placeholders,
            "rerun_m2_m3_requested": int(summary.get("rerun_m2_m3_requested", 0) or 0),
            "levels_completed": int(summary.get("levels_completed", 0) or 0),
            "support": 0,
            "truth_status": SAGE5E_TRUTH_STATUS,
        },
        "generation_metrics": {
            "requests_per_budget_quota": int(
                generation.get("requests_per_budget_quota", 0) or 0
            ),
            "effective_requests_generated": effective,
            "mini_frontier_hypotheses_generated": len(
                generation.get("mini_frontier_hypotheses", []) or []
            ),
            "duplicate_candidates_skipped": int(
                generation.get("duplicate_candidates_skipped", 0) or 0
            ),
            "dedup_keys_selected": int(generation.get("dedup_keys_selected", 0) or 0),
            "generation_budget_exhausted": bool(
                generation.get("generation_budget_exhausted", False)
            ),
            "effective_request_ratio": _ratio(effective, placeholders),
            "unresolved_placeholder_switches_after_generation": max(
                0,
                placeholders - effective,
            ),
            "support": 0,
            "truth_status": SAGE5E_TRUTH_STATUS,
        },
        "generation_diagnostics": [
            dict(row) for row in generation.get("generation_diagnostics", []) or []
        ],
    }


def _build_summary(
    *,
    budgets: Sequence[int],
    per_budget: Sequence[Mapping[str, Any]],
    all_hypotheses: Sequence[Mapping[str, Any]],
    all_requests: Sequence[Mapping[str, Any]],
    execution_requests: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    source_sage5: Mapping[str, Any],
    source_sage5b: Mapping[str, Any],
    source_sage5c: Mapping[str, Any],
    requests_per_budget: int,
    execute_requests_per_budget: int,
) -> Dict[str, Any]:
    generation_by_budget = {
        int(row.get("budget", 0) or 0): int(
            (row.get("generation_metrics", {}) or {}).get(
                "effective_requests_generated",
                0,
            )
            or 0
        )
        for row in per_budget
    }
    budgets_with_generation = [
        budget for budget in budgets if int(generation_by_budget.get(int(budget), 0)) > 0
    ]
    execution_by_budget = _counts(_budget_from_request(row) for row in experiments)
    budgets_with_execution = [
        budget for budget in budgets if int(execution_by_budget.get(str(int(budget)), 0)) > 0
    ]
    placeholders = sum(
        int((row.get("source_metrics", {}) or {}).get("placeholder_switches_seen", 0) or 0)
        for row in per_budget
    )
    effective = len(all_requests)
    support_events = sum(int(row.get("support_events", 0) or 0) for row in experiments)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in experiments
    )
    duplicate_skips = sum(
        int((row.get("generation_metrics", {}) or {}).get("duplicate_candidates_skipped", 0) or 0)
        for row in per_budget
    )
    gate_passed = bool(
        len(budgets_with_generation) >= 2
        and len(budgets_with_execution) >= 2
        and effective > int((source_sage5c.get("summary", {}) or {}).get(
            "effective_requests_generated",
            0,
        ) or 0)
        and len(blocked) == 0
    )
    return {
        "source_sage5_outcome_status": str(source_sage5.get("outcome_status", "")),
        "source_sage5b_outcome_status": str(source_sage5b.get("outcome_status", "")),
        "source_sage5c_outcome_status": str(source_sage5c.get("outcome_status", "")),
        "source_sage5c_effective_requests_generated": int(
            (source_sage5c.get("summary", {}) or {}).get(
                "effective_requests_generated",
                0,
            )
            or 0
        ),
        "budgets_evaluated": [int(item) for item in budgets],
        "requests_per_budget": int(requests_per_budget),
        "execute_requests_per_budget": int(execute_requests_per_budget),
        "budgets_with_effective_generation": [int(item) for item in budgets_with_generation],
        "budgets_with_execution": [int(item) for item in budgets_with_execution],
        "all_budgets_have_effective_generation": len(budgets_with_generation)
        == len(budgets),
        "all_budgets_have_execution": len(budgets_with_execution) == len(budgets),
        "effective_requests_generated": effective,
        "mini_frontier_hypotheses_generated": len(all_hypotheses),
        "mini_frontier_m3_requests": len(all_requests),
        "rerun_m2_m3_requested": placeholders,
        "effective_request_ratio": _ratio(effective, placeholders),
        "duplicate_candidates_skipped": duplicate_skips,
        "dedup_key_count": len({str(row.get("dedup_key", "")) for row in all_requests}),
        "selected_execution_requests": len(execution_requests),
        "requests_executed": len(experiments),
        "requests_blocked": len(blocked),
        "live_prefix_replay_exact_events": sum(
            1 for row in experiments if bool(row.get("live_prefix_replay_exact", False))
        ),
        "context_snapshot_hash_verified_events": sum(
            1
            for row in experiments
            if bool(row.get("target_context_signature_verified", False))
            and bool(row.get("control_context_signature_verified", False))
        ),
        "support_events": support_events,
        "contradiction_events": contradiction_events,
        "neutral_events": sum(int(row.get("neutral_events", 0) or 0) for row in experiments),
        "blocked_replay_events": len(blocked),
        "execution_failures": sum(
            1
            for row in blocked
            if "failed" in str(row.get("blocked_reason", "")).lower()
        ),
        "families_generated": _counts(row.get("hypothesis_family", "") for row in all_requests),
        "target_actions_generated": _counts(row.get("target_action", "") for row in all_requests),
        "families_executed": _counts(row.get("hypothesis_family", "") for row in experiments),
        "target_actions_executed": _counts(row.get("target_action", "") for row in experiments),
        "gate_passed": gate_passed,
        "outcome_status": (
            SAGE5E_GENERATED_AND_EXECUTED
            if gate_passed
            else SAGE5E_GENERATION_NOT_DISTRIBUTED
        ),
        "support": 0,
        "truth_status": SAGE5E_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "policy_result_counted_as_confirmation": False,
        "generated_requests_counted_as_support": False,
        "mini_frontier_counted_as_evidence": False,
        "mini_frontier_execution_counted_as_evidence": False,
        "support_events_counted_as_support": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _retag_frontier(
    frontier: Mapping[str, Any],
    *,
    game_id: str,
    budget: int,
) -> Dict[str, Any]:
    hypothesis = dict(frontier["hypothesis"])
    request = dict(frontier["m3_request"])
    step = int(hypothesis.get("frontier_step", request.get("source_step", 0)) or 0)
    hypothesis_id = f"sage5e::distributed_live_mini_frontier::{budget:03d}::{step:04d}"
    transition_id = f"sage5e::{game_id}::budget_{budget:03d}::step_{step:04d}"
    hypothesis.update(
        {
            "hypothesis_id": hypothesis_id,
            "frontier_context_id": transition_id,
            "frontier_reason": (
                "distributed_live_rerun_m2_m3_placeholder_replaced_by_mini_frontier"
            ),
            "truth_status": SAGE5E_TRUTH_STATUS,
        }
    )
    source_generation = dict(hypothesis.get("source_generation", {}) or {})
    source_generation["sources"] = ["distributed_live_mini_frontier"]
    hypothesis["source_generation"] = source_generation
    request.update(
        {
            "request_id": f"m2m3::{hypothesis_id}",
            "source_hypothesis_id": hypothesis_id,
            "source_transition_id": transition_id,
            "generated_by": "SAGE.5e_distributed_live_mini_frontier",
            "truth_status": M2_TRUTH_STATUS,
        }
    )
    diagnostics = dict(frontier["diagnostics"])
    diagnostics.update(
        {
            "hypothesis_id": hypothesis_id,
            "request_id": request["request_id"],
            "truth_status": SAGE5E_TRUTH_STATUS,
        }
    )
    return {"hypothesis": hypothesis, "m3_request": request, "diagnostics": diagnostics}


def _attach_dedup(frontier: Mapping[str, Any], *, dedup_key: str) -> None:
    diff = dict(frontier["hypothesis"].get("diff_summary", {}) or {})
    signature = diff_signature(diff)
    frontier["hypothesis"]["dedup_key"] = dedup_key
    frontier["hypothesis"]["diff_signature"] = signature
    frontier["m3_request"]["dedup_key"] = dedup_key
    frontier["m3_request"]["diff_signature"] = signature


def _retag_execution_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "truth_status": SAGE5E_TRUTH_STATUS,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "support_events_counted_as_support": False,
            "observation_counted_as_confirmation": False,
        }
    )
    return result


def _source_context(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    summary = dict(payload.get("summary", payload.get("comparison", {})) or {})
    return {
        "source_artifact": label,
        "outcome_status": str(payload.get("outcome_status", "")),
        "support": 0,
        "source_counted_as_scientific_evidence": False,
        "summary": {
            key: summary.get(key)
            for key in (
                "effective_requests_generated",
                "effective_request_ratio",
                "placeholder_switch_ratio",
                "rerun_m2_m3_requested",
            )
            if key in summary
        },
        "truth_status": SAGE5E_TRUTH_STATUS,
    }


def _budget_from_request(row: Mapping[str, Any]) -> int:
    transition = str(row.get("source_transition_id", ""))
    marker = "::budget_"
    if marker in transition:
        tail = transition.split(marker, 1)[1]
        return int(tail.split("::", 1)[0].split("_", 1)[0])
    return 0


def _select_action(
    valid_actions: Sequence[Any],
    name: str,
    args: Mapping[str, Any],
) -> Any | None:
    expected_args = {str(key): str(value) for key, value in dict(args or {}).items()}
    fallback = None
    for action in valid_actions:
        if action_name(action) != str(name):
            continue
        if fallback is None:
            fallback = action
        actual_args = {
            str(key): str(value)
            for key, value in dict(action_args(action) or {}).items()
        }
        if actual_args == expected_args:
            return action
    return fallback if not expected_args else None


def _make_real_env(game_id: str, env_dir: Path) -> Any:
    _configure_offline_env(env_dir)
    return _make_env(game_id, env_dir)


def _empty_generation(*, blocked_reason: str = "") -> Dict[str, Any]:
    return {
        "placeholder_switches_seen": 0,
        "mini_frontier_hypotheses": [],
        "mini_frontier_m3_requests": [],
        "generation_diagnostics": [],
        "effective_requests_generated": 0,
        "duplicate_candidates_skipped": 0,
        "generation_budget_exhausted": False,
        "dedup_keys_selected": 0,
        "blocked_reason": blocked_reason,
        "support": 0,
        "truth_status": SAGE5E_TRUTH_STATUS,
    }


def _load_optional_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.5e distributed live mini-frontier generation.",
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
    parser.add_argument("--source-sage5", default=str(DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH))
    parser.add_argument("--source-sage5b", default=str(DEFAULT_SAGE5B_SWITCH_AUDIT_RESULTS_PATH))
    parser.add_argument("--source-sage5c", default=str(DEFAULT_SAGE5C_LIVE_MINI_FRONTIER_RESULTS_PATH))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH))
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--requests-per-budget", type=int, default=DEFAULT_REQUESTS_PER_BUDGET)
    parser.add_argument(
        "--execute-requests-per-budget",
        type=int,
        default=DEFAULT_EXECUTE_REQUESTS_PER_BUDGET,
    )
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
    args = parser.parse_args(argv)
    run_sage5e_distributed_live_mini_frontier_generation(
        m2_fused_requests_path=args.m2_fused_requests,
        m3_fused_results_path=args.m3_fused_results,
        m3_counterfactual_feasibility_path=args.m3_counterfactual_feasibility,
        p1_policy_probe_path=args.p1_policy_probe,
        p1_utility_handoff_path=args.p1_utility_handoff,
        source_sage5_path=args.source_sage5,
        source_sage5b_path=args.source_sage5b,
        source_sage5c_path=args.source_sage5c,
        environments_dir=args.environments_dir,
        output_path=args.out,
        game_id=args.game_id,
        budgets=args.budgets,
        requests_per_budget=args.requests_per_budget,
        execute_requests_per_budget=args.execute_requests_per_budget,
        max_counterfactual_collections=args.max_counterfactual_collections,
        progress_stall_window=args.progress_stall_window,
        same_action_arg_repeats=args.same_action_arg_repeats,
        low_state_novelty_threshold=args.low_state_novelty_threshold,
        repeated_action_arg_rate_threshold=args.repeated_action_arg_rate_threshold,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
