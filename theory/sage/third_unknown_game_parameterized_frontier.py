"""SAGE.7a parameterized live-prefix mini-frontier for the third game."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.m2.schema import M2_READY_FOR_M3_STATUS
from theory.m2.validators import validate_m3_request
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .known_game_closed_loop_scaffold import (
    DEFAULT_M2_FUSED_REQUESTS_PATH,
    DEFAULT_M3_COUNTERFACTUAL_FEASIBILITY_PATH,
    DEFAULT_M3_FUSED_RESULTS_PATH,
    DEFAULT_P1_POLICY_PROBE_PATH,
    DEFAULT_P1_UTILITY_HANDOFF_PATH,
)
from .live_mini_frontier_generation import (
    _is_terminal,
    _make_real_env,
    _select_action,
    generate_live_mini_frontier,
)
from .live_prefix_counterfactual_collector import LivePrefixAction
from .policy_loop_guard import action_args, action_name
from .progress_stall_trigger import (
    DEFAULT_LOW_STATE_NOVELTY_THRESHOLD,
    DEFAULT_PROGRESS_STALL_WINDOW,
    DEFAULT_REPEATED_ACTION_ARG_RATE_THRESHOLD,
    DEFAULT_SAME_ACTION_ARG_REPEATS,
)
from .second_unknown_game_transfer import validate_sage5_transfer_source
from .subgoal_switcher import (
    DEFAULT_MAX_COUNTERFACTUAL_COLLECTIONS,
    SUBGOAL_RERUN,
    run_sage3_subgoal_switch_probe,
)
from .switch_attribution_placeholder_audit import (
    PLACEHOLDER_DEPENDENCY_THRESHOLD,
    _budget_audit,
)
from .third_unknown_game_transfer import (
    DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH,
    SAGE7_ALL_BUDGETS_PASSED,
    SAGE7_PARAMETERIZED_FRONTIER_REQUIRED,
    SAGE7_SCHEMA_VERSION,
    SAGE7_TRUTH_STATUS,
)
from .unknown_game_bounded_probe import EnvFactory


DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH = (
    Path("diagnostics")
    / "sage"
    / "sage7a_third_unknown_game_parameterized_frontier.json"
)

SAGE7A_SCHEMA_VERSION = "sage.third_unknown_game_parameterized_frontier.v1"
SAGE7A_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_7A"
SAGE7A_FRONTIER_GENERATED = (
    "SAGE_THIRD_UNKNOWN_GAME_PARAMETERIZED_FRONTIER_GENERATED_CANDIDATE_ONLY"
)
SAGE7A_NO_EFFECTIVE_FRONTIER = (
    "SAGE_THIRD_UNKNOWN_GAME_NO_PARAMETERIZED_FRONTIER_CANDIDATE_ONLY"
)
SAGE7A_PARAMETERIZED_EXECUTION_REQUIRED = (
    "SAGE7B_PARAMETERIZED_ACTION6_EXECUTION_REQUIRED_CANDIDATE_ONLY"
)
SAGE7A_NO_EXECUTION_REQUESTED = (
    "NO_PARAMETERIZED_ACTION6_EXECUTION_REQUESTED_CANDIDATE_ONLY"
)

DEFAULT_REQUESTS_PER_BUDGET = 6
DEFAULT_CONTROLS_PER_REQUEST = 2
PARAMETERIZED_CONTROL_POLICY = (
    "PRE_REGISTERED_SAME_ACTION_ALTERNATIVE_ARGS_AT_LIVE_PREFIX"
)
PARAMETERIZED_CONTROL_SELECTION_RULE = (
    "MAX_MANHATTAN_DISTANCE_FROM_TARGET_THEN_CANONICAL_ARGS"
)


def run_sage7a_parameterized_mini_frontier(
    *,
    source_sage7_path: str | Path = DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH,
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
    requests_per_budget: int = DEFAULT_REQUESTS_PER_BUDGET,
    controls_per_request: int = DEFAULT_CONTROLS_PER_REQUEST,
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
    """Materialize distributed ACTION6 target/control requests from SAGE.7."""
    source = _load_json(source_sage7_path)
    validate_sage7a_source(source)
    if int(requests_per_budget) <= 0:
        raise ValueError("SAGE.7a requests_per_budget must be positive")
    if int(controls_per_request) < 2:
        raise ValueError("SAGE.7a requires at least two parameterized controls")

    game_id = str(source.get("summary", {}).get("selected_third_game_id", ""))
    source_budgets = [
        int(value)
        for value in source.get("summary", {}).get("budgets_evaluated", []) or []
    ]
    resolved_budgets = [
        int(value) for value in (budgets if budgets is not None else source_budgets)
    ]
    if not resolved_budgets:
        raise ValueError("SAGE.7a requires at least one bounded budget")
    if not set(resolved_budgets).issubset(set(source_budgets)):
        raise ValueError("SAGE.7a budgets must be pre-registered by SAGE.7")
    source_by_budget = {
        int(row.get("budget", 0) or 0): row
        for row in source.get("third_game_probe", {}).get("per_budget_results", [])
        or []
        if isinstance(row, Mapping)
    }
    if not set(resolved_budgets).issubset(set(source_by_budget)):
        raise ValueError("SAGE.7a source budget records are incomplete")

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    # Complete all bounded reruns first.  The later prefix replays use fresh
    # environments and cannot alter the reproducibility observation.
    runs_by_budget: Dict[int, Dict[str, Any]] = {}
    for budget in resolved_budgets:
        runs_by_budget[budget] = run_sage3_subgoal_switch_probe(
            m2_fused_requests_path=m2_fused_requests_path,
            m3_fused_results_path=m3_fused_results_path,
            m3_counterfactual_feasibility_path=m3_counterfactual_feasibility_path,
            p1_policy_probe_path=p1_policy_probe_path,
            p1_utility_handoff_path=p1_utility_handoff_path,
            environments_dir=env_dir,
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

    per_budget: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    for budget in resolved_budgets:
        run = runs_by_budget[budget]
        attribution = _budget_audit(
            budget,
            run,
            placeholder_dependency_threshold=placeholder_dependency_threshold,
        )
        generation = generate_parameterized_frontiers_for_run(
            game_id=game_id,
            budget=budget,
            run=run,
            environments_dir=env_dir,
            requests_per_budget=requests_per_budget,
            controls_per_request=controls_per_request,
            env_factory=env_factory,
        )
        hypotheses.extend(generation["mini_frontier_hypotheses"])
        requests.extend(generation["mini_frontier_m3_requests"])
        per_budget.append(
            build_sage7a_budget_record(
                budget=budget,
                source_row=source_by_budget[budget],
                run=run,
                attribution=attribution,
                generation=generation,
            )
        )

    gate = build_sage7a_gate(
        source=source,
        game_id=game_id,
        budgets=resolved_budgets,
        per_budget=per_budget,
        hypotheses=hypotheses,
        requests=requests,
        requests_per_budget=requests_per_budget,
        controls_per_request=controls_per_request,
    )
    outcome = (
        SAGE7A_FRONTIER_GENERATED
        if all(gate.values()) and requests
        else SAGE7A_NO_EFFECTIVE_FRONTIER
    )
    summary = summarize_sage7a(
        source=source,
        game_id=game_id,
        per_budget=per_budget,
        hypotheses=hypotheses,
        requests=requests,
        requests_per_budget=requests_per_budget,
        controls_per_request=controls_per_request,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE7A_SCHEMA_VERSION,
            "source_sage7_path": str(source_sage7_path),
            "game_id": game_id,
            "budgets": resolved_budgets,
            "environments_dir": str(env_dir),
            "requests_per_budget": int(requests_per_budget),
            "controls_per_request": int(controls_per_request),
            "frontier_distribution": (
                "FIXED_EVENLY_SPACED_PLACEHOLDER_ORDINALS_PER_BUDGET"
            ),
            "parameterized_control_policy": PARAMETERIZED_CONTROL_POLICY,
            "parameterized_control_selection_rule": (
                PARAMETERIZED_CONTROL_SELECTION_RULE
            ),
            "control_selection_occurs_before_m3_execution": True,
            "outcome_metrics_used_for_control_selection": False,
            "parameterized_variants_are_distinct_action_families": False,
            "max_counterfactual_collections": int(max_counterfactual_collections),
            "progress_stall_trigger_enabled": True,
            "progress_stall_window": int(progress_stall_window),
            "same_action_arg_repeats": int(same_action_arg_repeats),
            "low_state_novelty_threshold": int(low_state_novelty_threshold),
            "repeated_action_arg_rate_threshold": float(
                repeated_action_arg_rate_threshold
            ),
            "placeholder_dependency_threshold": float(placeholder_dependency_threshold),
            "inputs_read": ["SAGE.7", "M2.15", "M3.7e", "M3.7f", "P1"],
            "artifacts_not_modified": [
                "SAGE.7",
                "SAGE.6",
                "A32",
                "A33.1",
                "A33.2",
                "A33.3",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
            "scientific_policy": {
                "generated_hypotheses_are_candidate_only": True,
                "generated_requests_are_not_evidence": True,
                "parameterized_controls_are_not_distinct_actions": True,
                "source_scoped_mechanics_are_quarantined": True,
                "a32_a33_write_performed": False,
            },
        },
        "source_sage7_context": build_source_sage7_context(source),
        "per_budget_results": per_budget,
        "mini_frontier_hypotheses": hypotheses,
        "mini_frontier_m3_requests": requests,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE7A_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "generated_requests_counted_as_support": False,
        "mini_frontier_counted_as_evidence": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage7a_parameterized_mini_frontier(payload, output_path)
    return payload


def generate_parameterized_frontiers_for_run(
    *,
    game_id: str,
    budget: int,
    run: Mapping[str, Any],
    environments_dir: str | Path,
    requests_per_budget: int,
    controls_per_request: int,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Replay one run and pre-register same-family ACTION6 controls."""
    rows = [
        row
        for row in run.get("steps", []) or []
        if isinstance(row, Mapping) and int(row.get("env_actions", 0) or 0) > 0
    ]
    placeholders = [row for row in rows if _is_placeholder_row(row)]
    selected_ordinals = stratified_placeholder_ordinals(
        len(placeholders), int(requests_per_budget)
    )
    selected_steps = {
        int(placeholders[index].get("step", 0) or 0): index
        for index in selected_ordinals
    }
    env_dir = Path(environments_dir)
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, env_dir)
        )
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return _blocked_generation(f"env_setup_failed:{exc}")

    prefix: List[LivePrefixAction] = []
    hypotheses: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    placeholders_seen = 0
    for row in rows:
        step = int(row.get("step", 0) or 0)
        selected_action = str(row.get("selected_action", ""))
        selected_args = dict(row.get("selected_action_args", {}) or {})
        before_frame = frame
        before = snapshot_frame(before_frame)
        valid_actions = _valid_actions(env)
        selected = _select_action(valid_actions, selected_action, selected_args)
        if selected is None:
            diagnostics.append(
                {
                    "step": step,
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
                    "step": step,
                    "blocked_reason": f"replay_step_failed:{exc}",
                    "selected_action": selected_action,
                    "selected_action_args": selected_args,
                }
            )
            break
        after = snapshot_frame(
            after_frame, fallback_available_actions=before.available_actions
        )
        if _is_placeholder_row(row):
            placeholder_ordinal = placeholders_seen
            placeholders_seen += 1
            if step in selected_steps:
                frontier = generate_live_mini_frontier(
                    game_id=game_id,
                    budget=int(budget),
                    step=step,
                    before_frame=before_frame,
                    after_frame=after_frame,
                    before_snapshot=before,
                    after_snapshot=after,
                    action_name=selected_action,
                    action_args=selected_args,
                    valid_actions_before=valid_actions,
                    prefix_actions=prefix,
                    id_prefix="sage7a",
                    generator_label=(
                        "SAGE.7a_third_unknown_game_parameterized_frontier"
                    ),
                    context_state_origin=("sage7_third_game_live_prefix_frame_before"),
                    source_placeholder_id="sage7_placeholder::rerun_m2_m3",
                    generation_truth_status=SAGE7A_TRUTH_STATUS,
                )
                hypothesis = dict(frontier["hypothesis"])
                request = dict(frontier["m3_request"])
                controls = select_parameterized_control_variants(
                    valid_actions=valid_actions,
                    target_action=selected_action,
                    target_action_args=selected_args,
                    limit=controls_per_request,
                )
                _parameterize_frontier(
                    hypothesis=hypothesis,
                    request=request,
                    controls=controls,
                    controls_per_request=controls_per_request,
                    placeholder_ordinal=placeholder_ordinal,
                )
                hypotheses.append(hypothesis)
                if request["status"] == M2_READY_FOR_M3_STATUS:
                    requests.append(request)
                diagnostics.append(
                    {
                        "step": step,
                        "placeholder_ordinal": placeholder_ordinal,
                        "hypothesis_id": hypothesis["hypothesis_id"],
                        "request_id": request["request_id"],
                        "request_status": request["status"],
                        "target_action": selected_action,
                        "target_action_args": selected_args,
                        "live_action_options": len(valid_actions),
                        "parameterized_controls_available": len(controls),
                        "parameterized_controls_pre_registered": [
                            dict(control) for control in controls
                        ],
                        "support": 0,
                        "truth_status": SAGE7A_TRUTH_STATUS,
                    }
                )
        prefix.append(
            LivePrefixAction(name=selected_action, action_args=dict(selected_args))
        )
        frame = after_frame
        if _is_terminal(after.game_state):
            continue

    return {
        "placeholder_switches_seen": placeholders_seen,
        "selected_placeholder_ordinals": selected_ordinals,
        "selected_placeholder_steps": sorted(selected_steps),
        "mini_frontier_hypotheses": hypotheses,
        "mini_frontier_m3_requests": requests,
        "generation_diagnostics": diagnostics,
        "effective_requests_generated": len(requests),
        "stratified_request_cap_reached": placeholders_seen > len(selected_ordinals),
        "support": 0,
        "truth_status": SAGE7A_TRUTH_STATUS,
    }


def stratified_placeholder_ordinals(total: int, requested: int) -> List[int]:
    """Choose deterministic, evenly spaced placeholder ordinals."""
    total = max(0, int(total))
    requested = max(0, min(int(requested), total))
    if requested == 0:
        return []
    if requested == 1:
        return [0]
    return [(index * (total - 1)) // (requested - 1) for index in range(requested)]


def select_parameterized_control_variants(
    *,
    valid_actions: Sequence[Any],
    target_action: str,
    target_action_args: Mapping[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Select live-legal same-family controls without reading an outcome."""
    target_key = _canonical_json(dict(target_action_args))
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for option in valid_actions:
        name = action_name(option)
        args = dict(action_args(option) or {})
        args_key = _canonical_json(args)
        if name != target_action or not args or args_key == target_key:
            continue
        identity = _canonical_json({"action": name, "action_args": args})
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(
            {
                "action": name,
                "action_args": args,
                "distance_from_target": _manhattan_distance(target_action_args, args),
                "parameterized_control_role": "same_action_alternative_args",
                "live_legal_at_context": True,
                "counted_as_distinct_action": False,
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(row["distance_from_target"]),
            _canonical_json(row["action_args"]),
        )
    )
    return candidates[: max(0, int(limit))]


def _parameterize_frontier(
    *,
    hypothesis: Dict[str, Any],
    request: Dict[str, Any],
    controls: Sequence[Mapping[str, Any]],
    controls_per_request: int,
    placeholder_ordinal: int,
) -> None:
    target_action = str(request.get("target_action", ""))
    target_args = dict(request.get("target_action_args", {}) or {})
    enough_controls = len(controls) == int(controls_per_request)
    protocol = {
        "action_family": target_action,
        "target_variant": {"action": target_action, "action_args": target_args},
        "control_variants": [dict(row) for row in controls],
        "controls_required": int(controls_per_request),
        "controls_pre_registered": len(controls),
        "selection_rule": PARAMETERIZED_CONTROL_SELECTION_RULE,
        "selection_timing": "BEFORE_M3_EXECUTION_AND_OUTCOME_OBSERVATION",
        "same_action_family_required": True,
        "target_args_must_differ_from_control_args": True,
        "variants_counted_as_distinct_actions": False,
        "variant_substitution_allowed": False,
    }
    hypothesis.update(
        {
            "frontier_reason": (
                "live_rerun_placeholder_replaced_by_parameterized_action6_frontier"
            ),
            "placeholder_ordinal": int(placeholder_ordinal),
            "parameterized_action_family": target_action,
            "parameterized_control_protocol": protocol,
            "parameterized_variants_counted_as_distinct_actions": False,
        }
    )
    request.update(
        {
            "suggested_control_actions": [target_action] if controls else [],
            "status": (
                M2_READY_FOR_M3_STATUS if enough_controls else "BLOCKED_NOT_TESTABLE"
            ),
            "blocking_reason": (
                None
                if enough_controls
                else "INSUFFICIENT_LIVE_PARAMETERIZED_ACTION6_CONTROLS"
            ),
            "parameterized_action_family": target_action,
            "parameterized_control_policy": PARAMETERIZED_CONTROL_POLICY,
            "parameterized_control_protocol": protocol,
            "pre_registered_parameterized_control_variants": [
                dict(row) for row in controls
            ],
            "parameterized_controls_counted_as_distinct_actions": False,
            "generated_by": ("SAGE.7a_third_unknown_game_parameterized_frontier"),
            "generated_request_counted_as_support": False,
        }
    )
    request["falsification_criterion"] = {
        "metric": str(request.get("metric", "")),
        "support_condition": (
            "target_variant_live_effect_differs_from_pre_registered_controls"
        ),
        "failure_condition": (
            "target_variant_live_effect_matches_all_pre_registered_controls_or_is_unavailable"
        ),
        "minimum_effect_size": 1,
    }
    validation = validate_m3_request(request)
    if not validation.valid:
        raise ValueError(
            f"invalid SAGE.7a request {request.get('request_id')}: {validation.errors}"
        )


def build_sage7a_budget_record(
    *,
    budget: int,
    source_row: Mapping[str, Any],
    run: Mapping[str, Any],
    attribution: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> Dict[str, Any]:
    source_metrics = dict(source_row.get("metrics", {}) or {})
    rerun = dict(run.get("summary", {}) or {})
    switch_attribution = dict(attribution.get("switch_attribution", {}) or {})
    placeholder_audit = dict(attribution.get("placeholder_audit", {}) or {})
    source_switches = int(source_metrics.get("subgoal_switches", 0) or 0)
    observed_events = int(switch_attribution.get("total_switches", 0) or 0)
    switch_attribution["total_switch_events_observed"] = observed_events
    switch_attribution["total_switches"] = int(rerun.get("subgoal_switches", 0) or 0)
    switch_attribution["terminal_guard_events_outside_source_switch_count"] = max(
        0, observed_events - switch_attribution["total_switches"]
    )
    placeholders = int(
        switch_attribution.get("placeholder_rerun_m2_m3_switches", 0) or 0
    )
    generated = int(generation.get("effective_requests_generated", 0) or 0)
    requests = list(generation.get("mini_frontier_m3_requests", []) or [])
    return {
        "budget": int(budget),
        "source_sage7_metrics": {
            "subgoal_switches": source_switches,
            "rerun_m2_m3_requested": int(
                source_metrics.get("rerun_m2_m3_requested", 0) or 0
            ),
            "progress_stall_detected": bool(
                source_metrics.get("progress_stall_detected", False)
            ),
            "unique_state_signatures": int(
                source_metrics.get("unique_state_signatures", 0) or 0
            ),
            "terminal_rate": float(source_metrics.get("terminal_rate", 0.0) or 0.0),
            "support": 0,
        },
        "rerun_reproducibility": {
            "subgoal_switches": int(rerun.get("subgoal_switches", 0) or 0),
            "progress_stall_detected": bool(
                rerun.get("progress_stall_detected", False)
            ),
            "unique_state_signatures": int(
                rerun.get("unique_state_signatures", 0) or 0
            ),
            "selected_action_always_legal": bool(
                rerun.get("selected_action_always_legal", False)
            ),
            "switch_count_matches_source": int(rerun.get("subgoal_switches", 0) or 0)
            == source_switches,
            "progress_stall_matches_source": bool(
                rerun.get("progress_stall_detected", False)
            )
            == bool(source_metrics.get("progress_stall_detected", False)),
            "state_signature_count_matches_source": int(
                rerun.get("unique_state_signatures", 0) or 0
            )
            == int(source_metrics.get("unique_state_signatures", 0) or 0),
        },
        "switch_attribution": switch_attribution,
        "placeholder_audit": placeholder_audit,
        "parameterized_frontier_generation": {
            "placeholder_switches_seen": int(
                generation.get("placeholder_switches_seen", 0) or 0
            ),
            "placeholder_count_matches_attribution": int(
                generation.get("placeholder_switches_seen", 0) or 0
            )
            == placeholders,
            "placeholder_count_matches_source": placeholders
            == int(source_metrics.get("rerun_m2_m3_requested", 0) or 0),
            "selected_placeholder_ordinals": list(
                generation.get("selected_placeholder_ordinals", []) or []
            ),
            "selected_placeholder_steps": list(
                generation.get("selected_placeholder_steps", []) or []
            ),
            "hypotheses_generated": len(
                generation.get("mini_frontier_hypotheses", []) or []
            ),
            "effective_requests_generated": generated,
            "unselected_placeholders": max(0, placeholders - generated),
            "effective_request_ratio": _ratio(generated, placeholders),
            "stratified_request_cap_reached": bool(
                generation.get("stratified_request_cap_reached", False)
            ),
            "all_requests_have_parameterized_controls": all(
                len(row.get("pre_registered_parameterized_control_variants", []) or [])
                > 0
                for row in requests
            ),
            "request_ids": [str(row.get("request_id", "")) for row in requests],
        },
        "generation_diagnostics": list(
            generation.get("generation_diagnostics", []) or []
        ),
        "support": 0,
        "truth_status": SAGE7A_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def summarize_sage7a(
    *,
    source: Mapping[str, Any],
    game_id: str,
    per_budget: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    requests_per_budget: int,
    controls_per_request: int,
    outcome: str,
) -> Dict[str, Any]:
    attribution = [dict(row.get("switch_attribution", {}) or {}) for row in per_budget]
    placeholders = sum(
        int(row.get("placeholder_rerun_m2_m3_switches", 0) or 0) for row in attribution
    )
    requests_by_budget = Counter(
        _budget_from_request_id(str(row.get("request_id", ""))) for row in requests
    )
    target_variants = {
        _canonical_json(
            {
                "action": row.get("target_action"),
                "action_args": row.get("target_action_args", {}),
            }
        )
        for row in requests
    }
    control_variants = {
        _canonical_json(
            {
                "action": control.get("action"),
                "action_args": control.get("action_args", {}),
            }
        )
        for row in requests
        for control in row.get("pre_registered_parameterized_control_variants", [])
        or []
    }
    generated = len(requests)
    ready = outcome == SAGE7A_FRONTIER_GENERATED and generated > 0
    return {
        "game_id": game_id,
        "source_sage7_outcome_status": str(source.get("outcome_status", "")),
        "budgets_evaluated": [int(row.get("budget", 0) or 0) for row in per_budget],
        "source_switches_expected": sum(
            int(row.get("source_sage7_metrics", {}).get("subgoal_switches", 0) or 0)
            for row in per_budget
        ),
        "switches_reproduced": sum(
            int(row.get("switch_attribution", {}).get("total_switches", 0) or 0)
            for row in per_budget
        ),
        "source_switch_count_reproduced_exactly": all(
            bool(
                row.get("rerun_reproducibility", {}).get(
                    "switch_count_matches_source", False
                )
            )
            for row in per_budget
        ),
        "total_switch_events_observed": sum(
            int(row.get("total_switch_events_observed", 0) or 0) for row in attribution
        ),
        "terminal_guard_events_outside_source_switch_count": sum(
            int(row.get("terminal_guard_events_outside_source_switch_count", 0) or 0)
            for row in attribution
        ),
        "true_exploratory_switches": sum(
            int(row.get("true_exploratory_switches", 0) or 0) for row in attribution
        ),
        "new_candidate_target_switches": sum(
            int(row.get("new_candidate_target_switches", 0) or 0) for row in attribution
        ),
        "placeholder_rerun_m2_m3_switches": placeholders,
        "requests_per_budget_target": int(requests_per_budget),
        "controls_per_request": int(controls_per_request),
        "mini_frontier_hypotheses_generated": len(hypotheses),
        "effective_requests_generated": generated,
        "effective_request_ratio": _ratio(generated, placeholders),
        "unselected_placeholder_switches": max(0, placeholders - generated),
        "requests_by_budget": {
            str(key): value for key, value in sorted(requests_by_budget.items())
        },
        "action_families": sorted(
            {str(row.get("parameterized_action_family", "")) for row in requests}
        ),
        "distinct_action_families": len(
            {str(row.get("parameterized_action_family", "")) for row in requests}
        ),
        "distinct_target_parameter_variants": len(target_variants),
        "distinct_control_parameter_variants": len(control_variants),
        "parameterized_control_variants_pre_registered": sum(
            len(row.get("pre_registered_parameterized_control_variants", []) or [])
            for row in requests
        ),
        "parameterized_variants_counted_as_distinct_actions": False,
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
        "ready_for_parameterized_m3_execution": ready,
        "required_next_step": (
            SAGE7A_PARAMETERIZED_EXECUTION_REQUIRED
            if ready
            else SAGE7A_NO_EXECUTION_REQUESTED
        ),
        "gate_passed": outcome == SAGE7A_FRONTIER_GENERATED,
        "outcome_status": outcome,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "support": 0,
        "truth_status": SAGE7A_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def build_sage7a_gate(
    *,
    source: Mapping[str, Any],
    game_id: str,
    budgets: Sequence[int],
    per_budget: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    requests_per_budget: int,
    controls_per_request: int,
) -> Dict[str, bool]:
    request_ids = [str(row.get("request_id", "")) for row in requests]
    hypothesis_ids = [str(row.get("hypothesis_id", "")) for row in hypotheses]
    expected_requests = len(budgets) * int(requests_per_budget)
    return {
        "source_sage7_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        ),
        "source_parameterized_frontier_required": bool(
            source.get("summary", {}).get(
                "ready_for_parameterized_mini_frontier", False
            )
        )
        and str(source.get("summary", {}).get("required_next_step", ""))
        == SAGE7_PARAMETERIZED_FRONTIER_REQUIRED,
        "source_switch_count_reproduced": all(
            bool(
                row.get("rerun_reproducibility", {}).get(
                    "switch_count_matches_source", False
                )
            )
            for row in per_budget
        ),
        "all_budget_reruns_reproducible": all(
            bool(
                row.get("rerun_reproducibility", {}).get(
                    "progress_stall_matches_source", False
                )
            )
            and bool(
                row.get("rerun_reproducibility", {}).get(
                    "state_signature_count_matches_source", False
                )
            )
            and bool(
                row.get("rerun_reproducibility", {}).get(
                    "selected_action_always_legal", False
                )
            )
            for row in per_budget
        ),
        "placeholder_counts_align": all(
            bool(
                row.get("parameterized_frontier_generation", {}).get(
                    "placeholder_count_matches_attribution", False
                )
            )
            and bool(
                row.get("parameterized_frontier_generation", {}).get(
                    "placeholder_count_matches_source", False
                )
            )
            for row in per_budget
        ),
        "distributed_request_target_met": len(requests) == expected_requests
        and all(
            int(
                row.get("parameterized_frontier_generation", {}).get(
                    "effective_requests_generated", 0
                )
                or 0
            )
            == int(requests_per_budget)
            for row in per_budget
        ),
        "request_ids_unique": "" not in request_ids
        and len(request_ids) == len(set(request_ids)),
        "hypothesis_ids_unique": "" not in hypothesis_ids
        and len(hypothesis_ids) == len(set(hypothesis_ids)),
        "all_requests_ready_for_m3": bool(requests)
        and all(
            str(row.get("status", "")) == M2_READY_FOR_M3_STATUS for row in requests
        ),
        "all_requests_live_prefix_replayable": bool(requests)
        and all(
            str(row.get("replayability", "")) == "LIVE_PREFIX_REPLAY_CONTEXT"
            for row in requests
        ),
        "all_requests_scoped_to_selected_game": all(
            str(row.get("game_id", "")) == game_id for row in requests
        ),
        "single_action6_family_preserved": bool(requests)
        and all(str(row.get("target_action", "")) == "ACTION6" for row in requests)
        and all(
            str(row.get("parameterized_action_family", "")) == "ACTION6"
            for row in requests
        ),
        "target_args_explicit": all(
            bool(row.get("target_action_args")) for row in requests
        ),
        "parameterized_controls_pre_registered": all(
            len(row.get("pre_registered_parameterized_control_variants", []) or [])
            == int(controls_per_request)
            for row in requests
        ),
        "parameterized_controls_live_legal_and_distinct": all(
            _request_controls_valid(row) for row in requests
        ),
        "parameterized_variants_not_relabelled_as_actions": not bool(
            source.get("action_surface_audit", {}).get(
                "parameterized_action_variants_counted_as_distinct_actions", True
            )
        )
        and all(
            not bool(
                row.get("parameterized_controls_counted_as_distinct_actions", True)
            )
            for row in requests
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


def build_source_sage7_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(source.get("summary", {}) or {})
    return {
        "source_outcome_status": str(source.get("outcome_status", "")),
        "selected_third_game_id": str(summary.get("selected_third_game_id", "")),
        "budgets": list(summary.get("budgets_evaluated", []) or []),
        "source_switches": int(summary.get("subgoal_switches_total", 0) or 0),
        "source_placeholders": int(summary.get("rerun_m2_m3_requested_total", 0) or 0),
        "source_effective_requests": int(
            summary.get("rerun_m2_m3_effective_requests_generated_total", 0) or 0
        ),
        "action_families": list(summary.get("action_families", []) or []),
        "parameterized_action_options_count": int(
            summary.get("parameterized_action_options_count", 0) or 0
        ),
        "ready_for_parameterized_mini_frontier": bool(
            summary.get("ready_for_parameterized_mini_frontier", False)
        ),
        "required_next_step": str(summary.get("required_next_step", "")),
        "source_scoped_mechanics_reused": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            source.get("cross_game_mechanics_imported", 0) or 0
        ),
        "source_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE7A_TRUTH_STATUS,
    }


def validate_sage7a_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    surface = dict(source.get("action_surface_audit", {}) or {})
    guard = dict(source.get("cross_game_transfer_guard", {}) or {})
    if str(config.get("schema_version", "")) != SAGE7_SCHEMA_VERSION:
        raise ValueError("SAGE.7 schema version is not supported by SAGE.7a")
    if str(source.get("outcome_status", "")) != SAGE7_ALL_BUDGETS_PASSED:
        raise ValueError("SAGE.7a requires the completed SAGE.7 transfer")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.7 source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE7_TRUTH_STATUS:
        raise ValueError("SAGE.7 truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.7 source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.7 source support must remain 0")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.7 cannot perform a scientific verdict")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.7 cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.7 wrong_confirmations must remain 0")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
        or bool(source.get("scope_generalization_performed", False))
    ):
        raise ValueError("SAGE.7 cannot import or generalize quarantined mechanics")
    if not bool(summary.get("gate_passed", False)) or not bool(
        summary.get("all_budgets_gate_passed", False)
    ):
        raise ValueError("SAGE.7 transfer gate must pass before SAGE.7a")
    if (
        not bool(summary.get("ready_for_parameterized_mini_frontier", False))
        or str(summary.get("required_next_step", ""))
        != SAGE7_PARAMETERIZED_FRONTIER_REQUIRED
    ):
        raise ValueError("SAGE.7 must explicitly require a parameterized frontier")
    game_id = str(summary.get("selected_third_game_id", ""))
    if not game_id or game_id != str(surface.get("game_id", "")):
        raise ValueError("SAGE.7 selected game and action surface must align")
    if (
        list(surface.get("action_families", []) or []) != ["ACTION6"]
        or int(surface.get("distinct_action_families", 0) or 0) != 1
    ):
        raise ValueError("SAGE.7a requires a single ACTION6 family")
    if int(surface.get("parameterized_action_options_count", 0) or 0) < 3 or not bool(
        surface.get("parameterized_control_design_required", False)
    ):
        raise ValueError("SAGE.7a requires at least three ACTION6 parameter variants")
    if bool(
        surface.get("parameterized_action_variants_counted_as_distinct_actions", True)
    ):
        raise ValueError("SAGE.7 variants cannot be relabelled as distinct actions")
    if (
        not bool(guard.get("quarantine_passed", False))
        or not bool(guard.get("a33_2_registry_read_only", False))
        or not bool(guard.get("a33_3_registry_read_only", False))
    ):
        raise ValueError("SAGE.7 registry quarantine must pass")
    gate = dict(source.get("gate", {}) or {})
    if not gate or not all(bool(value) for value in gate.values()):
        raise ValueError("every SAGE.7 source gate must pass")
    probe = dict(source.get("third_game_probe", {}) or {})
    validate_sage5_transfer_source(probe)
    if str(probe.get("config", {}).get("game_id", "")) != game_id:
        raise ValueError("SAGE.7 embedded probe must match the selected game")
    rows = [
        row
        for row in probe.get("per_budget_results", []) or []
        if isinstance(row, Mapping)
    ]
    if len(rows) != int(summary.get("budgets_total", 0) or 0):
        raise ValueError("SAGE.7 embedded budget count must be exact")
    if sum(
        int(row.get("metrics", {}).get("rerun_m2_m3_requested", 0) or 0) for row in rows
    ) != int(summary.get("rerun_m2_m3_requested_total", 0) or 0):
        raise ValueError("SAGE.7 placeholder count must match its embedded probe")


def write_sage7a_parameterized_mini_frontier(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _request_controls_valid(request: Mapping[str, Any]) -> bool:
    target_action = str(request.get("target_action", ""))
    target_key = _canonical_json(request.get("target_action_args", {}) or {})
    controls = request.get("pre_registered_parameterized_control_variants", []) or []
    return bool(controls) and all(
        str(row.get("action", "")) == target_action
        and bool(row.get("action_args"))
        and _canonical_json(row.get("action_args", {}) or {}) != target_key
        and bool(row.get("live_legal_at_context", False))
        and not bool(row.get("counted_as_distinct_action", True))
        for row in controls
    )


def _is_placeholder_row(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("placeholder_action_used", False))
        or str(row.get("selected_subgoal", "")) == SUBGOAL_RERUN
    )


def _manhattan_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    try:
        return abs(int(left.get("x")) - int(right.get("x"))) + abs(
            int(left.get("y")) - int(right.get("y"))
        )
    except (TypeError, ValueError):
        return 0


def _budget_from_request_id(request_id: str) -> int:
    parts = request_id.split("::")
    try:
        return int(parts[-2])
    except (IndexError, ValueError):
        return 0


def _blocked_generation(reason: str) -> Dict[str, Any]:
    return {
        "placeholder_switches_seen": 0,
        "selected_placeholder_ordinals": [],
        "selected_placeholder_steps": [],
        "mini_frontier_hypotheses": [],
        "mini_frontier_m3_requests": [],
        "generation_diagnostics": [{"blocked_reason": reason}],
        "effective_requests_generated": 0,
        "stratified_request_cap_reached": False,
        "support": 0,
        "truth_status": SAGE7A_TRUTH_STATUS,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(int(numerator) / int(denominator), 6) if int(denominator) else 0.0


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-sage7", default=str(DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH)
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE7A_PARAMETERIZED_FRONTIER_PATH)
    )
    parser.add_argument(
        "--requests-per-budget", type=int, default=DEFAULT_REQUESTS_PER_BUDGET
    )
    parser.add_argument(
        "--controls-per-request", type=int, default=DEFAULT_CONTROLS_PER_REQUEST
    )
    args = parser.parse_args(argv)
    payload = run_sage7a_parameterized_mini_frontier(
        source_sage7_path=args.source_sage7,
        output_path=args.out,
        requests_per_budget=args.requests_per_budget,
        controls_per_request=args.controls_per_request,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
