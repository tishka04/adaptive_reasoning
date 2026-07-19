"""SAGE.7 bounded transfer to the pre-registered third unknown game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import game_splits

from theory.a33.control_dependent_relational_registry import (
    A33_3_ENTRY_ADDED,
    A33_3_SCHEMA_VERSION,
    A33_3_TRUTH_STATUS,
    CONTROL_DEPENDENT_RELATIONAL_CONTRAST,
    DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH,
)
from theory.a33.scoped_unknown_game_registry import (
    A33_2_ENTRY_ADDED,
    A33_2_SCHEMA_VERSION,
    A33_2_TRUTH_STATUS,
    DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH,
)
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions

from .long_horizon_transfer import DEFAULT_BUDGETS
from .policy_loop_guard import action_args, action_name
from .second_unknown_game_transfer import (
    DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH,
    SAGE6_ALL_BUDGETS_PASSED,
    SAGE6_SCHEMA_VERSION,
    SAGE6_TRUTH_STATUS,
    _same_game,
    validate_sage5_transfer_source,
)
from .unknown_game_bounded_probe import (
    DEFAULT_HUMAN_TRACES_DIR,
    DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH,
    EnvFactory,
    _make_real_env,
    run_sage5_unknown_game_bounded_probe,
    unknown_game_identity,
)


DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH = (
    Path("diagnostics") / "sage" / "sage7_third_unknown_game_transfer_results.json"
)

SAGE7_SCHEMA_VERSION = "sage.third_unknown_game_transfer.v1"
SAGE7_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_7"
SAGE7_ALL_BUDGETS_PASSED = (
    "SAGE_THIRD_UNKNOWN_GAME_ALL_BUDGETS_GATE_PASSED_CANDIDATE_ONLY"
)
SAGE7_TRANSFER_GATE_FAILED = (
    "SAGE_THIRD_UNKNOWN_GAME_TRANSFER_GATE_FAILED_CANDIDATE_ONLY"
)
SAGE7_PARAMETERIZED_FRONTIER_REQUIRED = (
    "PARAMETERIZED_ACTION6_MINI_FRONTIER_REQUIRED_CANDIDATE_ONLY"
)
SAGE7_NO_PARAMETERIZED_FRONTIER_REQUIRED = (
    "NO_PARAMETERIZED_MINI_FRONTIER_REQUIRED_CANDIDATE_ONLY"
)

ELIGIBLE_UNKNOWN_GAME = "ELIGIBLE_UNKNOWN_GAME"
EXCLUDED_PRIOR_UNKNOWN_GAME = "EXCLUDED_PRIOR_UNKNOWN_GAME"
EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME = "EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME"
EXCLUDED_KNOWN_GAME = "EXCLUDED_KNOWN_GAME"

DEFAULT_SELECTION_POLICY = (
    "PUBLIC_UNSEEN_FIXED_ORDER_EXCLUDING_PRIOR_SCOPED_GAMES_BEFORE_EXECUTION"
)


def run_sage7_third_unknown_game_transfer(
    *,
    source_sage6_path: str | Path = DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH,
    source_a33_2_path: str | Path = (
        DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH
    ),
    source_a33_3_path: str | Path = (
        DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH
    ),
    m2_dataset_manifest_path: str | Path = (
        DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH
    ),
    human_traces_dir: str | Path = DEFAULT_HUMAN_TRACES_DIR,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    candidate_game_ids: Sequence[str] | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 0,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run the bounded SAGE discipline on the third preselected unknown game."""
    source_sage6 = _load_json(source_sage6_path)
    source_a33_2 = _load_json(source_a33_2_path)
    source_a33_3 = _load_json(source_a33_3_path)
    validate_sage6_transfer_source(source_sage6)
    validate_a33_2_quarantine_source(source_a33_2)
    validate_a33_3_quarantine_source(source_a33_3, source_sage6=source_sage6)

    first_unknown_game_id = str(
        source_sage6.get("source_loop_context", {}).get("source_game_id", "")
    )
    second_unknown_game_id = str(
        source_sage6.get("summary", {}).get("selected_second_game_id", "")
    )
    prior_unknown_game_ids = tuple(
        value
        for value in (first_unknown_game_id, second_unknown_game_id)
        if value
    )
    a33_2_game_ids = tuple(
        sorted(
            {
                str(row.get("game_id", ""))
                for row in source_a33_2.get("scoped_confirmed_mechanics", []) or []
                if isinstance(row, Mapping) and str(row.get("game_id", ""))
            }
        )
    )
    a33_3_game_ids = tuple(
        sorted(
            {
                str(row.get("game_id", ""))
                for row in source_a33_3.get(
                    "control_dependent_relational_contrasts", []
                )
                or []
                if isinstance(row, Mapping) and str(row.get("game_id", ""))
            }
        )
    )
    registry_game_ids = tuple(sorted({*a33_2_game_ids, *a33_3_game_ids}))
    manifest = _load_optional_json(m2_dataset_manifest_path)
    requested_candidates = tuple(
        candidate_game_ids
        if candidate_game_ids is not None
        else (
            game_splits.resolve_full_game_id(short_id)
            for short_id in game_splits.PUBLIC_UNSEEN
        )
    )
    selection_audit, selected_game_id = select_third_unknown_game(
        prior_unknown_game_ids=prior_unknown_game_ids,
        candidate_game_ids=requested_candidates,
        registry_game_ids=registry_game_ids,
        human_traces_dir=human_traces_dir,
        m2_dataset_manifest=manifest,
    )

    third_probe = run_sage5_unknown_game_bounded_probe(
        m2_dataset_manifest_path=m2_dataset_manifest_path,
        human_traces_dir=human_traces_dir,
        environments_dir=environments_dir,
        output_path=None,
        game_id=selected_game_id,
        budgets=budgets,
        seed=seed,
        env_factory=env_factory,
    )
    validate_sage5_transfer_source(third_probe)
    action_surface = audit_third_game_action_surface(
        game_id=selected_game_id,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if any(
        _same_game(selected_game_id, game_id) for game_id in prior_unknown_game_ids
    ):
        raise ValueError("SAGE.7 selected game must differ from prior unknown games")
    if any(_same_game(selected_game_id, game_id) for game_id in registry_game_ids):
        raise ValueError("SAGE.7 selected game cannot reuse a registry-scoped game")

    transfer_guard = build_third_game_transfer_guard(
        prior_unknown_game_ids=prior_unknown_game_ids,
        selected_game_id=selected_game_id,
        a33_2_game_ids=a33_2_game_ids,
        a33_3_game_ids=a33_3_game_ids,
        source_a33_2=source_a33_2,
        source_a33_3=source_a33_3,
    )
    gate = build_sage7_gate(
        source_sage6=source_sage6,
        source_a33_2=source_a33_2,
        source_a33_3=source_a33_3,
        selected_game_id=selected_game_id,
        third_probe=third_probe,
        action_surface=action_surface,
        transfer_guard=transfer_guard,
    )
    outcome = (
        SAGE7_ALL_BUDGETS_PASSED
        if all(bool(value) for value in gate.values())
        else SAGE7_TRANSFER_GATE_FAILED
    )
    summary = summarize_sage7_transfer(
        source_sage6=source_sage6,
        source_a33_2=source_a33_2,
        source_a33_3=source_a33_3,
        selection_audit=selection_audit,
        selected_game_id=selected_game_id,
        third_probe=third_probe,
        action_surface=action_surface,
        transfer_guard=transfer_guard,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE7_SCHEMA_VERSION,
            "source_sage6_path": str(source_sage6_path),
            "source_a33_2_path": str(source_a33_2_path),
            "source_a33_3_path": str(source_a33_3_path),
            "m2_dataset_manifest_path": str(m2_dataset_manifest_path),
            "human_traces_dir": str(human_traces_dir),
            "environments_dir": (
                str(environments_dir) if environments_dir is not None else None
            ),
            "candidate_game_ids_in_fixed_order": list(requested_candidates),
            "selection_policy": DEFAULT_SELECTION_POLICY,
            "selection_occurs_before_third_game_execution": True,
            "outcome_based_game_selection_allowed": False,
            "budgets": [int(value) for value in budgets],
            "seed": int(seed),
            "inputs_read": ["SAGE.6", "A33.2", "A33.3", "M2 manifest"],
            "artifacts_not_modified": [
                "SAGE.6",
                "A32.5",
                "A32.6",
                "A33.1",
                "A33.2",
                "A33.3",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
            "transfer_policy": {
                "third_game_must_differ_from_prior_unknown_games": True,
                "unknown_game_hygiene_required": True,
                "a33_2_and_a33_3_entries_are_quarantined": True,
                "cross_game_mechanic_reuse_allowed": False,
                "policy_results_remain_candidate_only": True,
                "scientific_support_remains_zero": True,
                "parameterized_action_variants_are_not_distinct_action_families": True,
            },
        },
        "source_loop_context": {
            "first_unknown_game_id": first_unknown_game_id,
            "second_unknown_game_id": second_unknown_game_id,
            "source_sage6_outcome_status": str(
                source_sage6.get("outcome_status", "")
            ),
            "source_sage6_all_budgets_gate_passed": bool(
                source_sage6.get("summary", {}).get("all_budgets_gate_passed", False)
            ),
            "source_a33_2_outcome_status": str(source_a33_2.get("outcome_status", "")),
            "source_a33_3_outcome_status": str(source_a33_3.get("outcome_status", "")),
            "source_a33_2_registration_performed": bool(
                source_a33_2.get("registration_performed", False)
            ),
            "source_a33_3_registration_performed": bool(
                source_a33_3.get("registration_performed", False)
            ),
            "quarantined_registry_game_ids": list(registry_game_ids),
            "source_loop_closed_before_transfer": True,
        },
        "selection_audit": selection_audit,
        "selected_third_unknown_game": {
            **dict(third_probe.get("unknown_game_identity", {}) or {}),
            "source_probe_truth_status": str(
                third_probe.get("unknown_game_identity", {}).get("truth_status", "")
            ),
            "truth_status": SAGE7_TRUTH_STATUS,
            "selected_by_policy": DEFAULT_SELECTION_POLICY,
            "selection_rank": next(
                int(row.get("eligible_rank", 0) or 0)
                for row in selection_audit
                if bool(row.get("selected", False))
            ),
            "selected_before_execution": True,
            "selected_from_outcome_metrics": False,
        },
        "action_surface_audit": action_surface,
        "cross_game_transfer_guard": transfer_guard,
        "third_game_probe": third_probe,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE7_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": True,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "parameterized_action_variants_counted_as_distinct_actions": False,
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage7_third_unknown_game_transfer(payload, output_path)
    return payload


def select_third_unknown_game(
    *,
    prior_unknown_game_ids: Sequence[str],
    candidate_game_ids: Sequence[str],
    registry_game_ids: Sequence[str],
    human_traces_dir: str | Path,
    m2_dataset_manifest: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    """Select the first unused eligible game in fixed order before execution."""
    audit: List[Dict[str, Any]] = []
    eligible_rank = 0
    for candidate_index, candidate in enumerate(candidate_game_ids, start=1):
        game_id = game_splits.resolve_full_game_id(str(candidate))
        identity = unknown_game_identity(
            game_id=game_id,
            human_traces_dir=human_traces_dir,
            m2_dataset_manifest=m2_dataset_manifest,
        )
        if any(_same_game(game_id, prior) for prior in prior_unknown_game_ids):
            eligibility_status = EXCLUDED_PRIOR_UNKNOWN_GAME
        elif any(_same_game(game_id, scoped) for scoped in registry_game_ids):
            eligibility_status = EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME
        elif not bool(identity.get("unknown_game", False)):
            eligibility_status = EXCLUDED_KNOWN_GAME
        else:
            eligibility_status = ELIGIBLE_UNKNOWN_GAME
            eligible_rank += 1
        audit.append(
            {
                "candidate_index": candidate_index,
                "game_id": game_id,
                "short_game_id": str(identity.get("short_game_id", "")),
                "split": str(identity.get("split", "")),
                "unknown_game": bool(identity.get("unknown_game", False)),
                "no_human_trace_for_game": bool(
                    identity.get("no_human_trace_for_game", False)
                ),
                "no_m2_arc_lewm_trace_for_game": bool(
                    identity.get("no_m2_arc_lewm_trace_for_game", False)
                ),
                "no_game_specific_prior": bool(
                    identity.get("no_game_specific_prior", False)
                ),
                "eligibility_status": eligibility_status,
                "eligible_rank": (
                    eligible_rank if eligibility_status == ELIGIBLE_UNKNOWN_GAME else 0
                ),
                "selected": False,
                "outcome_metrics_read_for_selection": False,
            }
        )
    selected = next(
        (row for row in audit if row["eligibility_status"] == ELIGIBLE_UNKNOWN_GAME),
        None,
    )
    if selected is None:
        raise ValueError("SAGE.7 found no eligible third unknown game")
    selected["selected"] = True
    return audit, str(selected["game_id"])


def audit_third_game_action_surface(
    *,
    game_id: str,
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
) -> Dict[str, Any]:
    """Record the live reset action surface without turning variants into actions."""
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    env = (
        env_factory(game_id)
        if env_factory is not None
        else _make_real_env(game_id, env_dir)
    )
    _reset_env(env)
    rows = [
        {
            "action": action_name(action),
            "action_args": action_args(action),
        }
        for action in _valid_actions(env)
        if action_name(action) and action_name(action) != "RESET"
    ]
    unique = {
        _canonical_json(row): row
        for row in rows
    }
    options = [unique[key] for key in sorted(unique)]
    action_families = sorted({str(row.get("action", "")) for row in options})
    parameterized = [row for row in options if dict(row.get("action_args", {}) or {})]
    return {
        "game_id": game_id,
        "audit_context": "LIVE_RESET_ACTION_SURFACE",
        "action_families": action_families,
        "distinct_action_families": len(action_families),
        "legal_action_options": options,
        "legal_action_options_count": len(options),
        "parameterized_action_options": parameterized,
        "parameterized_action_options_count": len(parameterized),
        "single_action_family_only": len(action_families) == 1,
        "parameterized_control_design_required": len(action_families) == 1
        and len(parameterized) >= 2,
        "parameterized_action_variants_counted_as_distinct_actions": False,
        "action_surface_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE7_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
    }


def build_third_game_transfer_guard(
    *,
    prior_unknown_game_ids: Sequence[str],
    selected_game_id: str,
    a33_2_game_ids: Sequence[str],
    a33_3_game_ids: Sequence[str],
    source_a33_2: Mapping[str, Any],
    source_a33_3: Mapping[str, Any],
) -> Dict[str, Any]:
    all_registry_games = tuple(sorted({*a33_2_game_ids, *a33_3_game_ids}))
    a33_2_entries = [
        row
        for row in source_a33_2.get("scoped_confirmed_mechanics", []) or []
        if isinstance(row, Mapping)
    ]
    a33_3_entries = [
        row
        for row in source_a33_3.get("control_dependent_relational_contrasts", [])
        or []
        if isinstance(row, Mapping)
    ]
    return {
        "prior_unknown_game_ids": list(prior_unknown_game_ids),
        "selected_game_id": selected_game_id,
        "different_from_all_prior_unknown_games": not any(
            _same_game(selected_game_id, game_id)
            for game_id in prior_unknown_game_ids
        ),
        "a33_2_scope_locked_entries_available": len(a33_2_entries),
        "a33_3_relational_entries_available": len(a33_3_entries),
        "a33_2_registry_game_ids": list(a33_2_game_ids),
        "a33_3_registry_game_ids": list(a33_3_game_ids),
        "all_quarantined_registry_game_ids": list(all_registry_games),
        "selected_game_outside_registry_scopes": not any(
            _same_game(selected_game_id, game_id) for game_id in all_registry_games
        ),
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "sb26_action5_prior_applied": False,
        "wa30_action2_relational_contrast_applied": False,
        "wa30_action1_universal_baseline_applied": False,
        "wa30_standalone_action2_effect_applied": False,
        "scope_generalization_performed": False,
        "a33_2_registry_read_only": True,
        "a33_3_registry_read_only": True,
        "quarantine_passed": True,
    }


def build_sage7_gate(
    *,
    source_sage6: Mapping[str, Any],
    source_a33_2: Mapping[str, Any],
    source_a33_3: Mapping[str, Any],
    selected_game_id: str,
    third_probe: Mapping[str, Any],
    action_surface: Mapping[str, Any],
    transfer_guard: Mapping[str, Any],
) -> Dict[str, bool]:
    identity = dict(third_probe.get("unknown_game_identity", {}) or {})
    summary = dict(third_probe.get("summary", {}) or {})
    return {
        "source_sage6_loop_passed": bool(
            source_sage6.get("summary", {}).get("gate_passed", False)
        ),
        "source_a33_2_registration_complete": bool(
            source_a33_2.get("registration_performed", False)
        ),
        "source_a33_3_registration_complete": bool(
            source_a33_3.get("registration_performed", False)
        ),
        "third_game_differs_from_prior_games": bool(
            transfer_guard.get("different_from_all_prior_unknown_games", False)
        ),
        "selected_game_unknown": bool(identity.get("unknown_game", False)),
        "selected_game_no_human_trace": bool(
            identity.get("no_human_trace_for_game", False)
        ),
        "selected_game_no_m2_trace": bool(
            identity.get("no_m2_arc_lewm_trace_for_game", False)
        ),
        "selected_game_no_specific_prior": bool(
            identity.get("no_game_specific_prior", False)
        ),
        "selected_game_matches_probe": str(identity.get("game_id", ""))
        == selected_game_id,
        "all_third_game_budgets_passed": bool(
            summary.get("all_budgets_gate_passed", False)
        ),
        "selected_actions_always_legal": all(
            bool(row.get("gate", {}).get("selected_action_always_legal", False))
            for row in third_probe.get("per_budget_results", []) or []
        ),
        "subgoal_switches_happened": all(
            bool(row.get("gate", {}).get("subgoal_switches_happened", False))
            for row in third_probe.get("per_budget_results", []) or []
        ),
        "no_catastrophic_repeat_collapse": all(
            bool(row.get("gate", {}).get("no_catastrophic_repeat_collapse", False))
            for row in third_probe.get("per_budget_results", []) or []
        ),
        "terminal_rate_under_threshold": all(
            bool(row.get("gate", {}).get("terminal_rate_under_threshold", False))
            for row in third_probe.get("per_budget_results", []) or []
        ),
        "live_action_surface_audited": int(
            action_surface.get("legal_action_options_count", 0) or 0
        )
        > 0,
        "parameterized_variants_not_relabelled_as_actions": not bool(
            action_surface.get(
                "parameterized_action_variants_counted_as_distinct_actions", True
            )
        ),
        "a33_2_and_a33_3_quarantined": bool(
            transfer_guard.get("quarantine_passed", False)
        )
        and bool(transfer_guard.get("a33_2_registry_read_only", False))
        and bool(transfer_guard.get("a33_3_registry_read_only", False))
        and int(transfer_guard.get("source_scoped_mechanics_reused", 0) or 0) == 0,
        "no_cross_game_mechanics_imported": int(
            transfer_guard.get("cross_game_mechanics_imported", 0) or 0
        )
        == 0,
    }


def summarize_sage7_transfer(
    *,
    source_sage6: Mapping[str, Any],
    source_a33_2: Mapping[str, Any],
    source_a33_3: Mapping[str, Any],
    selection_audit: Sequence[Mapping[str, Any]],
    selected_game_id: str,
    third_probe: Mapping[str, Any],
    action_surface: Mapping[str, Any],
    transfer_guard: Mapping[str, Any],
    outcome: str,
) -> Dict[str, Any]:
    per_budget = [
        row
        for row in third_probe.get("per_budget_results", []) or []
        if isinstance(row, Mapping)
    ]
    metrics = [dict(row.get("metrics", {}) or {}) for row in per_budget]
    terminal_rates = [float(row.get("terminal_rate", 0.0) or 0.0) for row in metrics]
    repeat_rates = [
        float(row.get("repeated_action_arg_rate", 0.0) or 0.0) for row in metrics
    ]
    unique_signatures = [
        int(row.get("unique_state_signatures", 0) or 0) for row in metrics
    ]
    rerun_requested = sum(
        int(row.get("rerun_m2_m3_requested", 0) or 0) for row in metrics
    )
    rerun_effective = sum(
        int(row.get("rerun_m2_m3_effective_requests_generated", 0) or 0)
        for row in metrics
    )
    ready_parameterized = bool(
        outcome == SAGE7_ALL_BUDGETS_PASSED
        and action_surface.get("parameterized_control_design_required", False)
        and rerun_requested > 0
        and rerun_effective == 0
    )
    return {
        "first_unknown_game_id": str(
            source_sage6.get("source_loop_context", {}).get("source_game_id", "")
        ),
        "second_unknown_game_id": str(
            source_sage6.get("summary", {}).get("selected_second_game_id", "")
        ),
        "selected_third_game_id": selected_game_id,
        "candidate_games_audited": len(selection_audit),
        "eligible_unknown_games": sum(
            str(row.get("eligibility_status", "")) == ELIGIBLE_UNKNOWN_GAME
            for row in selection_audit
        ),
        "known_or_seen_candidates_excluded": sum(
            str(row.get("eligibility_status", "")) == EXCLUDED_KNOWN_GAME
            for row in selection_audit
        ),
        "prior_or_registry_scope_candidates_excluded": sum(
            str(row.get("eligibility_status", ""))
            in {EXCLUDED_PRIOR_UNKNOWN_GAME, EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME}
            for row in selection_audit
        ),
        "selection_policy": DEFAULT_SELECTION_POLICY,
        "outcome_metrics_read_for_selection": False,
        "source_a33_2_entries_quarantined": int(
            source_a33_2.get("summary", {}).get(
                "scoped_confirmed_mechanics_registered", 0
            )
            or 0
        ),
        "source_a33_3_entries_quarantined": int(
            source_a33_3.get("summary", {}).get(
                "control_dependent_relational_contrasts_registered", 0
            )
            or 0
        ),
        "source_scoped_mechanics_reused": int(
            transfer_guard.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            transfer_guard.get("cross_game_mechanics_imported", 0) or 0
        ),
        "budgets_evaluated": [int(row.get("budget", 0) or 0) for row in per_budget],
        "budgets_gate_passed": sum(
            bool(row.get("gate", {}).get("gate_passed", False)) for row in per_budget
        ),
        "budgets_total": len(per_budget),
        "all_budgets_gate_passed": bool(
            third_probe.get("summary", {}).get("all_budgets_gate_passed", False)
        ),
        "env_steps_total": sum(int(row.get("env_steps", 0) or 0) for row in metrics),
        "subgoal_switches_total": sum(
            int(row.get("subgoal_switches", 0) or 0) for row in metrics
        ),
        "new_candidate_targets_discovered_total": sum(
            int(row.get("new_candidate_targets_discovered", 0) or 0)
            for row in metrics
        ),
        "rerun_m2_m3_requested_total": rerun_requested,
        "rerun_m2_m3_effective_requests_generated_total": rerun_effective,
        "budgets_with_progress_stall_detected": [
            int(per_budget[index].get("budget", 0) or 0)
            for index, row in enumerate(metrics)
            if bool(row.get("progress_stall_detected", False))
        ],
        "max_terminal_rate": max(terminal_rates, default=0.0),
        "max_repeated_action_arg_rate": max(repeat_rates, default=0.0),
        "min_unique_state_signatures": min(unique_signatures, default=0),
        "max_unique_state_signatures": max(unique_signatures, default=0),
        "levels_completed_max": max(
            (int(row.get("levels_completed", 0) or 0) for row in metrics),
            default=0,
        ),
        "action_families": list(action_surface.get("action_families", []) or []),
        "distinct_action_families": int(
            action_surface.get("distinct_action_families", 0) or 0
        ),
        "legal_action_options_count": int(
            action_surface.get("legal_action_options_count", 0) or 0
        ),
        "parameterized_action_options_count": int(
            action_surface.get("parameterized_action_options_count", 0) or 0
        ),
        "parameterized_action_variants_counted_as_distinct_actions": False,
        "ready_for_parameterized_mini_frontier": ready_parameterized,
        "required_next_step": (
            SAGE7_PARAMETERIZED_FRONTIER_REQUIRED
            if ready_parameterized
            else SAGE7_NO_PARAMETERIZED_FRONTIER_REQUIRED
        ),
        "gate_passed": outcome == SAGE7_ALL_BUDGETS_PASSED,
        "outcome_status": outcome,
        "support": 0,
        "truth_status": SAGE7_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_intake_requested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def validate_sage6_transfer_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6_SCHEMA_VERSION:
        raise ValueError("SAGE.6 schema version is not supported by SAGE.7")
    if str(source.get("outcome_status", "")) != SAGE6_ALL_BUDGETS_PASSED:
        raise ValueError("SAGE.7 requires the completed SAGE.6 transfer")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.6 source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE6_TRUTH_STATUS:
        raise ValueError("SAGE.6 source truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.6 source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.6 source support must remain 0")
    if bool(source.get("revision_performed", False)) or bool(
        source.get("confirmation_performed", False)
    ) or bool(source.get("refutation_performed", False)):
        raise ValueError("SAGE.6 source cannot create a scientific verdict")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.6 source cannot write A32/A33")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.6 source wrong_confirmations must remain 0")
    if bool(source.get("policy_result_counted_as_confirmation", False)):
        raise ValueError("SAGE.6 policy result cannot count as confirmation")
    if int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0 or int(
        source.get("cross_game_mechanics_imported", 0) or 0
    ) != 0:
        raise ValueError("SAGE.6 cannot import scoped cross-game mechanics")
    if bool(source.get("scope_generalization_performed", False)):
        raise ValueError("SAGE.6 cannot generalize registry scope")
    if not bool(summary.get("gate_passed", False)) or not bool(
        summary.get("all_budgets_gate_passed", False)
    ) or not all(bool(value) for value in source.get("gate", {}).values()):
        raise ValueError("SAGE.6 transfer gate must pass before SAGE.7")
    if str(summary.get("selected_second_game_id", "")) != "wa30-ee6fef47":
        raise ValueError("SAGE.7 expects pre-registered wa30 as the second game")
    if not bool(
        source.get("source_loop_context", {}).get(
            "source_loop_closed_before_transfer", False
        )
    ):
        raise ValueError("SAGE.6 source loop must be closed")
    if bool(config.get("outcome_based_game_selection_allowed", True)) or not bool(
        config.get("selection_occurs_before_second_game_execution", False)
    ):
        raise ValueError("SAGE.6 selection must remain pre-execution and outcome-free")
    probe = dict(source.get("second_game_probe", {}) or {})
    validate_sage5_transfer_source(probe)
    if str(probe.get("config", {}).get("game_id", "")) != str(
        summary.get("selected_second_game_id", "")
    ):
        raise ValueError("SAGE.6 embedded probe must match its selected game")


def validate_a33_2_quarantine_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != A33_2_SCHEMA_VERSION:
        raise ValueError("A33.2 schema version is not supported by SAGE.7")
    if str(source.get("outcome_status", "")) != A33_2_ENTRY_ADDED or str(
        source.get("status", "")
    ) != "REGISTERED":
        raise ValueError("SAGE.7 requires the completed A33.2 registry")
    if str(source.get("truth_status", "")) != A33_2_TRUTH_STATUS:
        raise ValueError("A33.2 truth must remain non-reevaluated")
    if not bool(source.get("registration_performed", False)) or not bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("A33.2 registry must be written before SAGE.7")
    if bool(source.get("revision_performed", False)) or bool(
        source.get("confirmation_performed", False)
    ) or bool(source.get("refutation_performed", False)):
        raise ValueError("A33.2 cannot create a new scientific verdict")
    if bool(source.get("scope_generalization_performed", False)) or int(
        source.get("wrong_confirmations", 0) or 0
    ) != 0:
        raise ValueError("A33.2 quarantine scope must remain valid")
    entries = [
        row
        for row in source.get("scoped_confirmed_mechanics", []) or []
        if isinstance(row, Mapping)
    ]
    if (
        len(entries)
        != int(summary.get("scoped_confirmed_mechanics_registered", 0) or 0)
        or len(entries) != 1
    ):
        raise ValueError("SAGE.7 requires exactly one A33.2 scoped entry")
    for entry in entries:
        if str(entry.get("game_id", "")) != "sb26-7fbdac44" or not all(
            bool(entry.get(flag, False))
            for flag in (
                "scope_game_locked",
                "scope_candidate_locked",
                "scope_contexts_locked",
                "scope_measurement_locked",
                "not_generalized_beyond_game",
                "not_generalized_beyond_candidate_scope",
                "not_generalized_to_other_actions",
            )
        ):
            raise ValueError("A33.2 sb26 entry scope must remain fully locked")


def validate_a33_3_quarantine_source(
    source: Mapping[str, Any],
    *,
    source_sage6: Mapping[str, Any],
) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != A33_3_SCHEMA_VERSION:
        raise ValueError("A33.3 schema version is not supported by SAGE.7")
    if str(source.get("outcome_status", "")) != A33_3_ENTRY_ADDED or str(
        source.get("status", "")
    ) != "REGISTERED":
        raise ValueError("SAGE.7 requires the completed A33.3 registry")
    if str(source.get("truth_status", "")) != A33_3_TRUTH_STATUS:
        raise ValueError("A33.3 truth must remain non-reevaluated")
    if not bool(source.get("registration_performed", False)) or not bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("A33.3 registry must be written before SAGE.7")
    if bool(source.get("revision_performed", False)) or bool(
        source.get("confirmation_performed", False)
    ) or bool(source.get("refutation_performed", False)):
        raise ValueError("A33.3 cannot create a new scientific verdict")
    if bool(source.get("scope_generalization_performed", False)) or int(
        source.get("wrong_confirmations", 0) or 0
    ) != 0:
        raise ValueError("A33.3 quarantine scope must remain valid")
    if bool(source.get("standalone_action2_effect_registered", False)) or bool(
        source.get("action1_universal_baseline_registered", False)
    ):
        raise ValueError("A33.3 excluded claims must remain unregistered")
    entries = [
        row
        for row in source.get("control_dependent_relational_contrasts", []) or []
        if isinstance(row, Mapping)
    ]
    if (
        len(entries)
        != int(summary.get("control_dependent_relational_contrasts_registered", 0) or 0)
        or len(entries) != 1
    ):
        raise ValueError("SAGE.7 requires exactly one A33.3 relational entry")
    expected_game = str(
        source_sage6.get("summary", {}).get("selected_second_game_id", "")
    )
    entry = entries[0]
    if str(entry.get("game_id", "")) != expected_game or str(
        entry.get("registry_entry_type", "")
    ) != CONTROL_DEPENDENT_RELATIONAL_CONTRAST:
        raise ValueError("A33.3 entry must remain the wa30 relational contrast")
    if str(entry.get("standalone_action2_effect_status", "")) != "unresolved":
        raise ValueError("A33.3 standalone ACTION2 effect must remain unresolved")
    if not all(
        bool(entry.get(flag, False))
        for flag in (
            "scope_game_locked",
            "scope_contexts_locked",
            "scope_target_action_locked",
            "scope_control_actions_locked",
            "scope_metric_locked",
            "scope_budgets_locked",
            "not_generalized_beyond_game",
            "not_generalized_beyond_exact_paired_contexts",
            "not_generalized_beyond_recorded_controls",
            "standalone_action2_effect_excluded",
            "action1_universal_baseline_excluded",
        )
    ):
        raise ValueError("A33.3 relational entry scope must remain fully locked")


def write_sage7_third_unknown_game_transfer(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _load_optional_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return _load_json(source)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SAGE.7 on the fixed-order third unknown game."
    )
    parser.add_argument(
        "--source-sage6", default=str(DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH)
    )
    parser.add_argument(
        "--source-a33-2",
        default=str(DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--source-a33-3",
        default=str(DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--m2-dataset-manifest",
        default=str(DEFAULT_M2_ARC_LEWM_DATASET_MANIFEST_PATH),
    )
    parser.add_argument("--human-traces-dir", default=str(DEFAULT_HUMAN_TRACES_DIR))
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH)
    )
    parser.add_argument("--candidate-game-ids", nargs="+", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_sage7_third_unknown_game_transfer(
        source_sage6_path=args.source_sage6,
        source_a33_2_path=args.source_a33_2,
        source_a33_3_path=args.source_a33_3,
        m2_dataset_manifest_path=args.m2_dataset_manifest,
        human_traces_dir=args.human_traces_dir,
        environments_dir=args.environments_dir,
        output_path=args.out,
        candidate_game_ids=args.candidate_game_ids,
        budgets=args.budgets,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
