"""SAGE.6a switch attribution and live mini-frontier for the second game."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from theory.m2.schema import M2_READY_FOR_M3_STATUS
from theory.non_ar25_active_micro_run import _env_dir

from .known_game_closed_loop_scaffold import (
    DEFAULT_M2_FUSED_REQUESTS_PATH,
    DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH,
    DEFAULT_M3_FUSED_RESULTS_PATH,
    DEFAULT_P1_POLICY_PROBE_PATH,
    DEFAULT_P1_UTILITY_HANDOFF_PATH,
)
from .live_mini_frontier_generation import (
    DEFAULT_MAX_GENERATED_REQUESTS,
    generate_live_mini_frontiers_for_run,
)
from .progress_stall_trigger import (
    DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    DEFAULT_PROGRESS_STALL_WINDOW,
    DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    DEFAULT_SAME_ACTION_ARG_REPEATS,
)
from .second_unknown_game_transfer import (
    DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH,
    DEFAULT_SELECTION_POLICY,
    SAGE6_ALL_BUDGETS_PASSED,
    SAGE6_SCHEMA_VERSION,
    SAGE6_TRUTH_STATUS,
    validate_sage5_transfer_source,
)
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    run_sage3_subgoal_switch_probe,
)
from .switch_attribution_placeholder_audit import (
    PLACEHOLDER_DEPENDENCY_THRESHOLD,
    _budget_audit,
)
from .unknown_game_bounded_probe import EnvFactory


DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH = (
    Path("diagnostics") / "sage" / "sage6a_switch_attribution_mini_frontier.json"
)

SAGE6A_SCHEMA_VERSION = "sage.second_unknown_game_switch_frontier.v1"
SAGE6A_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_6A"
SAGE6A_FRONTIER_GENERATED = (
    "SAGE_SECOND_UNKNOWN_GAME_SWITCH_FRONTIER_GENERATED_CANDIDATE_ONLY"
)
SAGE6A_NO_EFFECTIVE_FRONTIER = (
    "SAGE_SECOND_UNKNOWN_GAME_NO_EFFECTIVE_FRONTIER_CANDIDATE_ONLY"
)


def run_sage6a_switch_attribution_mini_frontier(
    *,
    source_sage6_path: str | Path = DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH,
    m2_fused_requests_path: str | Path = DEFAULT_M2_FUSED_REQUESTS_PATH,
    m3_fused_results_path: str | Path = DEFAULT_M3_FUSED_RESULTS_PATH,
    m3_counterfactual_feasibility_path: str | Path = (
        DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH
    ),
    p1_policy_probe_path: str | Path = DEFAULT_P1_POLICY_PROBE_PATH,
    p1_utility_handoff_path: str | Path = DEFAULT_P1_UTILITY_HANDOFF_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    budgets: Sequence[int] | None = None,
    max_generated_requests: int = DEFAULT_MAX_GENERATED_REQUESTS,
    max_counterfactual_collections: int = DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    progress_stall_window: int = DEFAULT_PROGRESS_STALL_WINDOW,
    same_action_arg_repeats: int = DEFAULT_SAME_ACTION_ARG_REPEATS,
    low_state_novelty_threshold: int = DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    repeated_action_arg_rate_threshold: float = (
        DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD
    ),
    placeholder_dependency_threshold: float = PLACEHOLDER_DEPENDENCY_THRESHOLD,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Reproduce SAGE.6 switches and replace bounded rerun placeholders."""
    source = _load_json(source_sage6_path)
    validate_sage6a_source(source)
    selected = dict(source.get("selected_second_unknown_game", {}) or {})
    game_id = str(selected.get("game_id", ""))
    resolved_budgets = [
        int(value)
        for value in (
            budgets
            if budgets is not None
            else source.get("summary", {}).get("budgets_evaluated", [])
        )
    ]
    if not resolved_budgets:
        raise ValueError("SAGE.6a requires at least one bounded budget")
    source_by_budget = {
        int(row.get("budget", 0) or 0): row
        for row in source.get("second_game_probe", {}).get("per_budget_results", [])
        or []
        if isinstance(row, Mapping)
    }
    if not set(resolved_budgets).issubset(set(source_by_budget)):
        raise ValueError("SAGE.6a budgets must be pre-registered by SAGE.6")

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    # Reproduce every SAGE.6 probe before replaying any live prefix.  Prefix
    # replay creates fresh environments and must not perturb a later bounded
    # probe through game-level process state.
    runs_by_budget: Dict[int, Dict[str, Any]] = {}
    for budget in resolved_budgets:
        runs_by_budget[int(budget)] = run_sage3_subgoal_switch_probe(
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

    remaining_generation_budget = max(0, int(max_generated_requests))
    per_budget: List[Dict[str, Any]] = []
    all_hypotheses: List[Dict[str, Any]] = []
    all_requests: List[Dict[str, Any]] = []
    for budget in resolved_budgets:
        run = runs_by_budget[int(budget)]
        attribution = _budget_audit(
            int(budget),
            run,
            placeholder_dependency_threshold=placeholder_dependency_threshold,
        )
        generation = generate_live_mini_frontiers_for_run(
            game_id=game_id,
            budget=int(budget),
            run=run,
            environments_dir=env_dir,
            max_generated_requests=remaining_generation_budget,
            env_factory=env_factory,
            id_prefix="sage6a",
            generator_label="SAGE.6a_second_unknown_game_live_mini_frontier",
            context_state_origin="sage6_second_game_live_prefix_frame_before",
            source_placeholder_id="sage6_placeholder::rerun_m2_m3",
            generation_truth_status=SAGE6A_TRUTH_STATUS,
        )
        generated = int(generation.get("effective_requests_generated", 0) or 0)
        remaining_generation_budget -= generated
        all_hypotheses.extend(generation.get("mini_frontier_hypotheses", []) or [])
        all_requests.extend(generation.get("mini_frontier_m3_requests", []) or [])
        per_budget.append(
            build_sage6a_budget_record(
                budget=int(budget),
                source_sage6_row=source_by_budget[int(budget)],
                run=run,
                attribution=attribution,
                generation=generation,
            )
        )

    summary = summarize_sage6a(
        source=source,
        game_id=game_id,
        per_budget=per_budget,
        hypotheses=all_hypotheses,
        requests=all_requests,
        max_generated_requests=max_generated_requests,
        placeholder_dependency_threshold=placeholder_dependency_threshold,
    )
    gate = build_sage6a_gate(
        source=source,
        game_id=game_id,
        per_budget=per_budget,
        hypotheses=all_hypotheses,
        requests=all_requests,
        max_generated_requests=max_generated_requests,
    )
    outcome = (
        SAGE6A_FRONTIER_GENERATED
        if all(gate.values()) and all_requests
        else SAGE6A_NO_EFFECTIVE_FRONTIER
    )
    summary["gate_passed"] = outcome == SAGE6A_FRONTIER_GENERATED
    summary["outcome_status"] = outcome
    payload = {
        "config": {
            "schema_version": SAGE6A_SCHEMA_VERSION,
            "source_sage6_path": str(source_sage6_path),
            "game_id": game_id,
            "budgets": resolved_budgets,
            "environments_dir": str(env_dir),
            "max_generated_requests": int(max_generated_requests),
            "generation_budget_distribution": (
                "FIXED_BUDGET_ORDER_WITH_GLOBAL_REQUEST_CAP"
            ),
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "placeholder_dependency_threshold": float(placeholder_dependency_threshold),
            "inputs_read": ["SAGE.6", "M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": [
                "SAGE.6",
                "SAGE.5",
                "A32.5",
                "A33.1",
                "A33.2",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
            "scientific_policy": {
                "switch_attribution_is_not_scientific_support": True,
                "generated_hypotheses_are_candidate_only": True,
                "generated_requests_are_not_evidence": True,
                "source_scoped_mechanics_are_quarantined": True,
                "a32_a33_write_performed": False,
            },
        },
        "source_sage6_context": build_source_sage6_context(source),
        "per_budget_results": per_budget,
        "mini_frontier_hypotheses": all_hypotheses,
        "mini_frontier_m3_requests": all_requests,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE6A_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "switch_attribution_counted_as_support": False,
        "generated_requests_counted_as_support": False,
        "mini_frontier_counted_as_evidence": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage6a_switch_attribution_mini_frontier(payload, output_path)
    return payload


def build_sage6a_budget_record(
    *,
    budget: int,
    source_sage6_row: Mapping[str, Any],
    run: Mapping[str, Any],
    attribution: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> Dict[str, Any]:
    source_metrics = dict(source_sage6_row.get("metrics", {}) or {})
    rerun_summary = dict(run.get("summary", {}) or {})
    switch_attribution = dict(attribution.get("switch_attribution", {}) or {})
    placeholder_audit = dict(attribution.get("placeholder_audit", {}) or {})
    placeholder_count = int(
        switch_attribution.get("placeholder_rerun_m2_m3_switches", 0) or 0
    )
    effective = int(generation.get("effective_requests_generated", 0) or 0)
    hypotheses = list(generation.get("mini_frontier_hypotheses", []) or [])
    requests = list(generation.get("mini_frontier_m3_requests", []) or [])
    source_counted_switches = int(rerun_summary.get("subgoal_switches", 0) or 0)
    switch_events_observed = int(switch_attribution.get("total_switches", 0) or 0)
    switch_attribution["total_switch_events_observed"] = switch_events_observed
    switch_attribution["total_switches"] = source_counted_switches
    switch_attribution["terminal_guard_events_outside_source_switch_count"] = max(
        0, switch_events_observed - source_counted_switches
    )
    placeholder_ratio = _ratio(placeholder_count, source_counted_switches)
    placeholder_dependency_under_threshold = placeholder_ratio <= float(
        switch_attribution.get("placeholder_dependency_threshold", 0.0)
    )
    switch_attribution["placeholder_switch_ratio"] = placeholder_ratio
    switch_attribution["placeholder_dependency_under_threshold"] = (
        placeholder_dependency_under_threshold
    )
    placeholder_audit["placeholder_dependency_high"] = (
        not placeholder_dependency_under_threshold
    )
    return {
        "budget": int(budget),
        "source_sage6_metrics": {
            "subgoal_switches": int(source_metrics.get("subgoal_switches", 0) or 0),
            "progress_stall_detected": bool(
                source_metrics.get("progress_stall_detected", False)
            ),
            "progress_stall_switches": int(
                source_metrics.get("progress_stall_switches", 0) or 0
            ),
            "unique_state_signatures": int(
                source_metrics.get("unique_state_signatures", 0) or 0
            ),
            "terminal_rate": float(source_metrics.get("terminal_rate", 0.0) or 0.0),
            "support": 0,
        },
        "rerun_reproducibility": {
            "subgoal_switches": int(rerun_summary.get("subgoal_switches", 0) or 0),
            "progress_stall_detected": bool(
                rerun_summary.get("progress_stall_detected", False)
            ),
            "progress_stall_switches": int(
                rerun_summary.get("progress_stall_switches", 0) or 0
            ),
            "unique_state_signatures": int(
                rerun_summary.get("unique_state_signatures", 0) or 0
            ),
            "terminal_rate": float(rerun_summary.get("terminal_rate", 0.0) or 0.0),
            "selected_action_always_legal": bool(
                rerun_summary.get("selected_action_always_legal", False)
            ),
            "switch_count_matches_source": int(
                rerun_summary.get("subgoal_switches", 0) or 0
            )
            == int(source_metrics.get("subgoal_switches", 0) or 0),
            "progress_stall_matches_source": bool(
                rerun_summary.get("progress_stall_detected", False)
            )
            == bool(source_metrics.get("progress_stall_detected", False)),
            "state_signature_count_matches_source": int(
                rerun_summary.get("unique_state_signatures", 0) or 0
            )
            == int(source_metrics.get("unique_state_signatures", 0) or 0),
        },
        "switch_attribution": switch_attribution,
        "placeholder_audit": placeholder_audit,
        "mini_frontier_generation": {
            "placeholder_switches_seen": int(
                generation.get("placeholder_switches_seen", 0) or 0
            ),
            "placeholder_count_matches_attribution": int(
                generation.get("placeholder_switches_seen", 0) or 0
            )
            == placeholder_count,
            "hypotheses_generated": len(hypotheses),
            "effective_requests_generated": effective,
            "unresolved_placeholders_after_generation": max(
                0, placeholder_count - effective
            ),
            "effective_request_ratio": _ratio(effective, placeholder_count),
            "generation_budget_exhausted": bool(
                generation.get("generation_budget_exhausted", False)
            ),
            "hypothesis_ids": [str(row.get("hypothesis_id", "")) for row in hypotheses],
            "request_ids": [str(row.get("request_id", "")) for row in requests],
        },
        "generation_diagnostics": list(
            generation.get("generation_diagnostics", []) or []
        )[:5],
        "support": 0,
        "truth_status": SAGE6A_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def summarize_sage6a(
    *,
    source: Mapping[str, Any],
    game_id: str,
    per_budget: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    max_generated_requests: int,
    placeholder_dependency_threshold: float,
) -> Dict[str, Any]:
    attribution_rows = [
        dict(row.get("switch_attribution", {}) or {}) for row in per_budget
    ]
    total_switches = sum(
        int(row.get("total_switches", 0) or 0) for row in attribution_rows
    )
    progress_stall = sum(
        int(row.get("switches_due_to_progress_stall_or_repeat_collapse", 0) or 0)
        for row in attribution_rows
    )
    success_like = sum(
        int(
            row.get("switches_due_to_success_like_targets_exhausted_or_loop_guard", 0)
            or 0
        )
        for row in attribution_rows
    )
    terminal = sum(
        int(row.get("switches_due_to_terminal_guard", 0) or 0)
        for row in attribution_rows
    )
    placeholders = sum(
        int(row.get("placeholder_rerun_m2_m3_switches", 0) or 0)
        for row in attribution_rows
    )
    unresolved = max(0, placeholders - len(requests))
    source_switches = sum(
        int(row.get("source_sage6_metrics", {}).get("subgoal_switches", 0) or 0)
        for row in per_budget
    )
    hypothesis_families = Counter(
        str(row.get("hypothesis_family", "")) for row in hypotheses
    )
    target_actions = Counter(str(row.get("target_action", "")) for row in requests)
    requests_by_budget = Counter(
        _budget_from_sage6a_id(str(row.get("request_id", ""))) for row in requests
    )
    return {
        "game_id": game_id,
        "source_sage6_outcome_status": str(source.get("outcome_status", "")),
        "budgets_evaluated": [int(row.get("budget", 0) or 0) for row in per_budget],
        "source_switches_expected": source_switches,
        "switches_reproduced": total_switches,
        "total_switch_events_observed": sum(
            int(row.get("total_switch_events_observed", 0) or 0)
            for row in attribution_rows
        ),
        "source_switch_count_reproduced_exactly": total_switches == source_switches,
        "switches_due_to_progress_stall_or_repeat_collapse": progress_stall,
        "switches_due_to_success_like_targets_exhausted_or_loop_guard": success_like,
        "switches_due_to_terminal_guard": terminal,
        "terminal_guard_events_outside_source_switch_count": sum(
            int(row.get("terminal_guard_events_outside_source_switch_count", 0) or 0)
            for row in attribution_rows
        ),
        "true_exploratory_switches": sum(
            int(row.get("true_exploratory_switches", 0) or 0)
            for row in attribution_rows
        ),
        "active_counterfactual_switches": sum(
            int(row.get("active_counterfactual_switches", 0) or 0)
            for row in attribution_rows
        ),
        "reposition_switches": sum(
            int(row.get("reposition_switches", 0) or 0) for row in attribution_rows
        ),
        "new_candidate_target_switches": sum(
            int(row.get("new_candidate_target_switches", 0) or 0)
            for row in attribution_rows
        ),
        "placeholder_rerun_m2_m3_switches": placeholders,
        "source_placeholder_switch_ratio": _ratio(placeholders, total_switches),
        "placeholder_dependency_threshold": float(placeholder_dependency_threshold),
        "source_placeholder_dependency_under_threshold": _ratio(
            placeholders, total_switches
        )
        <= float(placeholder_dependency_threshold),
        "max_generated_requests": int(max_generated_requests),
        "mini_frontier_hypotheses_generated": len(hypotheses),
        "effective_requests_generated": len(requests),
        "effective_request_ratio": _ratio(len(requests), placeholders),
        "unresolved_placeholder_switches_after_generation": unresolved,
        "residual_placeholder_switch_ratio": _ratio(unresolved, total_switches),
        "hypothesis_families": dict(sorted(hypothesis_families.items())),
        "target_actions": dict(sorted(target_actions.items())),
        "requests_by_budget": {
            str(key): value for key, value in sorted(requests_by_budget.items())
        },
        "unique_context_snapshot_hashes": len(
            {str(row.get("context_snapshot_hash", "")) for row in requests}
        ),
        "all_requests_ready_for_m3": bool(requests)
        and all(
            str(row.get("status", "")) == M2_READY_FOR_M3_STATUS for row in requests
        ),
        "all_requests_live_prefix_replayable": bool(requests)
        and all(
            str(row.get("replayability", "")) == "LIVE_PREFIX_REPLAY_CONTEXT"
            for row in requests
        ),
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "support": 0,
        "truth_status": SAGE6A_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def build_sage6a_gate(
    *,
    source: Mapping[str, Any],
    game_id: str,
    per_budget: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    max_generated_requests: int,
) -> Dict[str, bool]:
    attribution_total = sum(
        int(row.get("switch_attribution", {}).get("total_switches", 0) or 0)
        for row in per_budget
    )
    source_total = sum(
        int(row.get("source_sage6_metrics", {}).get("subgoal_switches", 0) or 0)
        for row in per_budget
    )
    request_ids = [str(row.get("request_id", "")) for row in requests]
    hypothesis_ids = [str(row.get("hypothesis_id", "")) for row in hypotheses]
    return {
        "source_sage6_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        ),
        "source_switch_count_reproduced": attribution_total == source_total,
        "all_budget_reruns_reproducible": all(
            bool(
                row.get("rerun_reproducibility", {}).get(
                    "switch_count_matches_source", False
                )
            )
            and bool(
                row.get("rerun_reproducibility", {}).get(
                    "progress_stall_matches_source", False
                )
            )
            and bool(
                row.get("rerun_reproducibility", {}).get(
                    "state_signature_count_matches_source", False
                )
            )
            for row in per_budget
        ),
        "placeholder_counts_align": all(
            bool(
                row.get("mini_frontier_generation", {}).get(
                    "placeholder_count_matches_attribution", False
                )
            )
            for row in per_budget
        ),
        "effective_requests_generated": bool(requests),
        "generation_cap_respected": len(requests) <= int(max_generated_requests),
        "request_ids_unique": "" not in request_ids
        and len(request_ids) == len(set(request_ids)),
        "hypothesis_ids_unique": "" not in hypothesis_ids
        and len(hypothesis_ids) == len(set(hypothesis_ids)),
        "all_requests_ready_for_m3": all(
            str(row.get("status", "")) == M2_READY_FOR_M3_STATUS for row in requests
        ),
        "all_requests_live_prefix_replayable": all(
            str(row.get("replayability", "")) == "LIVE_PREFIX_REPLAY_CONTEXT"
            for row in requests
        ),
        "all_requests_scoped_to_selected_game": all(
            str(row.get("game_id", "")) == game_id for row in requests
        ),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0 for row in hypotheses
        )
        and all(int(row.get("support", 0) or 0) == 0 for row in requests),
        "source_registry_quarantine_preserved": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        )
        == 0
        and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
        and not bool(source.get("scope_generalization_performed", False)),
    }


def build_source_sage6_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_outcome_status": str(source.get("outcome_status", "")),
        "source_game_id": str(source.get("summary", {}).get("source_game_id", "")),
        "selected_second_game_id": str(
            source.get("summary", {}).get("selected_second_game_id", "")
        ),
        "source_switches": int(
            source.get("summary", {}).get("subgoal_switches_total", 0) or 0
        ),
        "budgets": list(source.get("summary", {}).get("budgets_evaluated", []) or []),
        "progress_stall_budgets": list(
            source.get("summary", {}).get("budgets_with_progress_stall_detected", [])
            or []
        ),
        "selection_policy": str(source.get("summary", {}).get("selection_policy", "")),
        "source_scoped_mechanics_reused": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            source.get("cross_game_mechanics_imported", 0) or 0
        ),
        "scope_generalization_performed": bool(
            source.get("scope_generalization_performed", False)
        ),
        "source_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE6A_TRUTH_STATUS,
    }


def validate_sage6a_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    selected = dict(source.get("selected_second_unknown_game", {}) or {})
    transfer_guard = dict(source.get("cross_game_transfer_guard", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6_SCHEMA_VERSION:
        raise ValueError("SAGE.6 schema version is not supported by SAGE.6a")
    if str(source.get("outcome_status", "")) != SAGE6_ALL_BUDGETS_PASSED:
        raise ValueError("SAGE.6a requires a fully passed SAGE.6 transfer")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.6 source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE6_TRUTH_STATUS:
        raise ValueError("SAGE.6 truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.6 source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.6 source support must remain 0")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.6 cannot perform a scientific verdict")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.6 wrong_confirmations must remain 0")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.6 cannot write A32/A33")
    if bool(source.get("policy_result_counted_as_confirmation", False)):
        raise ValueError("SAGE.6 policy result cannot count as confirmation")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6 cannot import source-game mechanics")
    if bool(source.get("scope_generalization_performed", False)):
        raise ValueError("SAGE.6 cannot generalize the source registry scope")
    if (
        str(config.get("selection_policy", "")) != DEFAULT_SELECTION_POLICY
        or not bool(config.get("selection_occurs_before_second_game_execution", False))
        or bool(config.get("outcome_based_game_selection_allowed", False))
    ):
        raise ValueError("SAGE.6 selection must remain pre-execution and fixed-order")
    if not bool(summary.get("gate_passed", False)) or not bool(
        summary.get("all_budgets_gate_passed", False)
    ):
        raise ValueError("SAGE.6 transfer gate must pass before SAGE.6a")
    game_id = str(selected.get("game_id", ""))
    if not game_id or game_id != str(summary.get("selected_second_game_id", "")):
        raise ValueError("SAGE.6 selected game identity must align")
    if (
        not bool(selected.get("unknown_game", False))
        or not bool(selected.get("selected_before_execution", False))
        or bool(selected.get("selected_from_outcome_metrics", False))
    ):
        raise ValueError("SAGE.6 selected game must preserve unknown-game hygiene")
    if (
        not bool(transfer_guard.get("quarantine_passed", False))
        or int(transfer_guard.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(transfer_guard.get("cross_game_mechanics_imported", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6 source registry quarantine must pass")
    gate = dict(source.get("gate", {}) or {})
    if not gate or not all(bool(value) for value in gate.values()):
        raise ValueError("every SAGE.6 source gate must pass")
    second_probe = dict(source.get("second_game_probe", {}) or {})
    validate_sage5_transfer_source(second_probe)
    if str(second_probe.get("config", {}).get("game_id", "")) != game_id:
        raise ValueError("SAGE.6 embedded probe must match the selected game")
    rows = [
        row
        for row in second_probe.get("per_budget_results", []) or []
        if isinstance(row, Mapping)
    ]
    if len(rows) != int(summary.get("budgets_total", 0) or 0):
        raise ValueError("SAGE.6 embedded budget count must be exact")
    if sum(
        int(row.get("metrics", {}).get("subgoal_switches", 0) or 0) for row in rows
    ) != int(summary.get("subgoal_switches_total", 0) or 0):
        raise ValueError("SAGE.6 source switch count must match its embedded probe")


def write_sage6a_switch_attribution_mini_frontier(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _budget_from_sage6a_id(request_id: str) -> int:
    parts = str(request_id).split("::")
    try:
        return int(parts[-2])
    except (ValueError, IndexError):
        return 0


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attribute SAGE.6 switches and generate the SAGE.6a frontier."
    )
    parser.add_argument(
        "--source-sage6", default=str(DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH)
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH))
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument(
        "--max-generated-requests",
        type=int,
        default=DEFAULT_MAX_GENERATED_REQUESTS,
    )
    args = parser.parse_args(argv)
    run_sage6a_switch_attribution_mini_frontier(
        source_sage6_path=args.source_sage6,
        environments_dir=args.environments_dir,
        output_path=args.out,
        budgets=args.budgets,
        max_generated_requests=args.max_generated_requests,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
