"""SAGE.5i bounded expansion of the unknown-game scientific control surface."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from .a32_review_handoff import (
    DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH,
    a32_intake_recommendation,
    missing_revision_requirements,
)
from .controlled_followup_acquisition import (
    DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH,
    SAGE5H_TRUTH_STATUS,
    audit_context_control_surface,
    execute_followup_experiment,
)
from .distributed_live_mini_frontier_generation import (
    DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH,
)


DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH = (
    Path("diagnostics") / "sage" / "sage5i_control_surface_expansion.json"
)
SAGE5I_SCHEMA_VERSION = "sage.control_surface_expansion.v1"
SAGE5I_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_5I"
SAGE5I_CONTROL_DIVERSITY_ACQUIRED = (
    "SAGE_CONTROL_SURFACE_EXPANSION_CONTROL_DIVERSITY_ACQUIRED_CANDIDATE_ONLY"
)
SAGE5I_PARTIAL_CONTROL_DIVERSITY = (
    "SAGE_CONTROL_SURFACE_EXPANSION_PARTIAL_CONTROL_DIVERSITY_CANDIDATE_ONLY"
)
SAGE5I_ACTION_DISTINCT_EXHAUSTED = (
    "SAGE_CONTROL_SURFACE_EXPANSION_ACTION_DISTINCT_EXHAUSTED_CANDIDATE_ONLY"
)

EXPANSION_ACQUIRED = "ACTION_DISTINCT_CONTROL_ACQUIRED_CANDIDATE_ONLY"
EXPANSION_NOT_FOUND = "NO_ACTION_DISTINCT_CONTROL_IN_REPLAYABLE_FRONTIER"
EXPANSION_EXECUTION_INCOMPLETE = "ACTION_DISTINCT_CONTROL_EXECUTION_INCOMPLETE"
A32_PARAMETERIZED_PROTOCOL_REQUIRED = (
    "A32_PROTOCOL_DECISION_REQUIRED_PARAMETERIZED_CONTROLS_CANDIDATE_ONLY"
)
MIN_CONTROL_CONTEXTS = 2

EnvFactory = Callable[[str], Any]


def run_sage5i_control_surface_expansion(
    *,
    source_sage5e_path: str | Path = (
        DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH
    ),
    source_sage5g_path: str | Path = DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH,
    source_sage5h_path: str | Path = (
        DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH
    ),
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    min_control_contexts: int = MIN_CONTROL_CONTEXTS,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Scan every replayable SAGE.5e candidate context for a new control action."""
    source_sage5e = _load_json(source_sage5e_path)
    source_sage5g = _load_json(source_sage5g_path)
    source_sage5h = _load_json(source_sage5h_path)
    validate_sage5i_sources(source_sage5e, source_sage5g, source_sage5h)

    requests = [
        dict(row)
        for row in source_sage5e.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping)
    ]
    candidates = [
        dict(row)
        for row in source_sage5g.get("a32_review_candidate_items", []) or []
        if isinstance(row, Mapping)
    ]
    prior_assessments = {
        str(row.get("candidate_id", "")): dict(row)
        for row in source_sage5h.get("updated_candidate_assessments", []) or []
        if isinstance(row, Mapping) and str(row.get("candidate_id", ""))
    }

    all_audits: List[Dict[str, Any]] = []
    candidate_surface_results: List[Dict[str, Any]] = []
    execution_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    experiments: List[Dict[str, Any]] = []
    expansions: List[Dict[str, Any]] = []

    for candidate in candidates:
        candidate_requests = select_candidate_context_requests(candidate, requests)
        surface = audit_candidate_control_surface(
            candidate=candidate,
            candidate_requests=candidate_requests,
            environments_dir=environments_dir,
            env_factory=env_factory,
            all_audits=all_audits,
        )
        candidate_surface_results.append(surface)
        expansion = execute_candidate_control_expansion(
            candidate=candidate,
            candidate_requests=candidate_requests,
            surface=surface,
            min_control_contexts=max(1, int(min_control_contexts)),
            environments_dir=environments_dir,
            env_factory=env_factory,
            execution_cache=execution_cache,
            experiments=experiments,
        )
        expansions.append(expansion)

    updated_assessments = update_sage5i_candidate_assessments(
        candidates=candidates,
        prior_assessments=prior_assessments,
        surfaces=candidate_surface_results,
        expansions=expansions,
    )
    protocol_proposal = build_a32_parameterized_control_protocol_proposal(
        surfaces=candidate_surface_results,
        updated_assessments=updated_assessments,
    )
    executed = [
        row for row in experiments if str(row.get("execution_status", "")) == "EXECUTED"
    ]
    blocked = [
        row for row in experiments if str(row.get("execution_status", "")) != "EXECUTED"
    ]
    summary = summarize_sage5i(
        source_sage5e=source_sage5e,
        candidates=candidates,
        surfaces=candidate_surface_results,
        audits=all_audits,
        expansions=expansions,
        experiments=executed,
        blocked_experiments=blocked,
        updated_assessments=updated_assessments,
        protocol_proposal=protocol_proposal,
    )
    payload = {
        "config": {
            "schema_version": SAGE5I_SCHEMA_VERSION,
            "source_sage5e_path": str(source_sage5e_path),
            "source_sage5g_path": str(source_sage5g_path),
            "source_sage5h_path": str(source_sage5h_path),
            "environments_dir": (
                str(environments_dir) if environments_dir is not None else None
            ),
            "minimum_control_contexts": max(1, int(min_control_contexts)),
            "inputs_read": ["SAGE.5e", "SAGE.5g", "SAGE.5h"],
            "execution_performed": True,
            "expansion_policy": {
                "scan_all_matching_sage5e_requests": True,
                "exact_context_replay_required": True,
                "new_control_must_have_distinct_action_name": True,
                "parameter_variants_are_not_action_distinct_controls": True,
                "reset_is_not_a_scientific_control": True,
                "bounded_exhaustion_is_not_refutation": True,
                "a32_write_performed": False,
                "a33_write_performed": False,
            },
            "artifacts_not_modified": ["M2", "M3", "A32", "A33", "A40", "P2"],
        },
        "source_sage5h_summary": dict(source_sage5h.get("summary", {}) or {}),
        "candidate_control_surface_results": candidate_surface_results,
        "context_surface_audits": all_audits,
        "control_expansion_experiments": executed,
        "blocked_control_expansion_experiments": blocked,
        "candidate_control_expansions": expansions,
        "updated_candidate_assessments": updated_assessments,
        "a32_parameterized_control_protocol_proposal": protocol_proposal,
        "summary": summary,
        "comparison": summary,
        "status": "UNRESOLVED",
        "outcome_status": summary["outcome_status"],
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE5I_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "execution_performed": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "bounded_control_surface_exhaustion_counted_as_refutation": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "control_expansion_events_counted_as_scientific_support": False,
        "protocol_proposal_counted_as_revision": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage5i_control_surface_expansion(payload, output_path)
    return payload


def select_candidate_context_requests(
    candidate: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    selected = [
        dict(request)
        for request in requests
        if _request_matches_candidate(candidate, request)
    ]
    selected.sort(key=_request_sort_key)
    return tuple(selected)


def audit_candidate_control_surface(
    *,
    candidate: Mapping[str, Any],
    candidate_requests: Sequence[Mapping[str, Any]],
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
    all_audits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    existing_controls = sorted(
        {
            str(row.get("action", ""))
            for row in candidate.get("control_interventions", []) or []
            if str(row.get("action", ""))
        }
    )
    synthetic_followup = {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "action": str(candidate.get("action", "")),
        "action_args": candidate.get("action_args"),
        "excluded_control_actions": existing_controls,
    }
    candidate_audits: List[Dict[str, Any]] = []
    for request in candidate_requests:
        audit = audit_context_control_surface(
            request=request,
            followup=synthetic_followup,
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        audit.update(
            {
                "audit_id": f"sage5i::context_surface_audit::{len(all_audits) + 1:03d}",
                "candidate_id": str(candidate.get("candidate_id", "")),
                "candidate_key": str(candidate.get("candidate_key", "")),
                "source_truth_status": str(audit.get("truth_status", "")),
                "truth_status": SAGE5I_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
                "control_surface_audit_counted_as_refutation": False,
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
        candidate_audits.append(audit)
        all_audits.append(audit)

    third_action_contexts = [
        row
        for row in candidate_audits
        if row.get("eligible_distinct_control_actions", []) or []
    ]
    surface_signatures = Counter(
        ",".join(
            sorted(str(value) for value in row.get("available_action_names", []) or [])
        )
        for row in candidate_audits
    )
    variants = _unique_action_variants(candidate_audits)
    parameterized_options = parameterized_control_options(
        candidate=candidate,
        existing_controls=existing_controls,
        available_variants=variants,
    )
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "candidate_key": str(candidate.get("candidate_key", "")),
        "game_id": str(candidate.get("game_id", "")),
        "action": str(candidate.get("action", "")),
        "action_args": candidate.get("action_args"),
        "existing_control_actions": existing_controls,
        "matching_context_requests": len(candidate_requests),
        "unique_contexts": len(
            {
                str(row.get("context_snapshot_hash", ""))
                for row in candidate_requests
                if str(row.get("context_snapshot_hash", ""))
            }
        ),
        "contexts_scanned": len(candidate_audits),
        "replay_exact_contexts": sum(
            1
            for row in candidate_audits
            if bool(row.get("live_prefix_replay_exact", False))
        ),
        "budgets_scanned": sorted(
            {_budget_from_request(row) for row in candidate_requests}
        ),
        "action_surface_signature_counts": dict(sorted(surface_signatures.items())),
        "available_action_names": sorted(
            {
                str(value)
                for row in candidate_audits
                for value in row.get("available_action_names", []) or []
                if str(value)
            }
        ),
        "contexts_with_new_action_distinct_control": len(third_action_contexts),
        "new_action_distinct_control_candidates": sorted(
            {
                str(value)
                for row in third_action_contexts
                for value in row.get("eligible_distinct_control_actions", []) or []
                if str(value)
            }
        ),
        "eligible_contexts": [
            {
                "source_request_id": str(row.get("source_request_id", "")),
                "live_prefix_replay_exact": bool(
                    row.get("live_prefix_replay_exact", False)
                ),
                "eligible_distinct_control_actions": list(
                    row.get("eligible_distinct_control_actions", []) or []
                ),
            }
            for row in candidate_audits
            if row.get("eligible_distinct_control_actions", []) or []
        ],
        "available_action_variants": variants,
        "parameterized_control_options": parameterized_options,
        "parameterized_control_option_count": len(parameterized_options),
        "control_surface_exhausted_action_distinct": not third_action_contexts,
        "audit_ids": [str(row.get("audit_id", "")) for row in candidate_audits],
        "bounded_scope": "SAGE.5e_matching_replayable_candidate_contexts",
        "bounded_exhaustion_counted_as_refutation": False,
        "support": 0,
        "truth_status": SAGE5I_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
    }


def execute_candidate_control_expansion(
    *,
    candidate: Mapping[str, Any],
    candidate_requests: Sequence[Mapping[str, Any]],
    surface: Mapping[str, Any],
    min_control_contexts: int,
    environments_dir: str | Path | None,
    env_factory: EnvFactory | None,
    execution_cache: MutableMapping[Tuple[str, str, str], Dict[str, Any]],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    request_by_id = {str(row.get("request_id", "")): row for row in candidate_requests}
    eligible_by_action: Dict[str, List[str]] = defaultdict(list)
    for row in surface.get("eligible_contexts", []) or []:
        request_id = str(row.get("source_request_id", ""))
        if request_id not in request_by_id:
            continue
        if not bool(row.get("live_prefix_replay_exact", False)):
            continue
        for control in row.get("eligible_distinct_control_actions", []) or []:
            eligible_by_action[str(control)].append(request_id)

    selected_control = next(
        (
            action
            for action, request_ids in sorted(eligible_by_action.items())
            if len(set(request_ids)) >= min_control_contexts
        ),
        "",
    )
    if not selected_control:
        return {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "candidate_key": str(candidate.get("candidate_key", "")),
            "action": str(candidate.get("action", "")),
            "expansion_status": EXPANSION_NOT_FOUND,
            "new_control_action": "",
            "minimum_control_contexts": min_control_contexts,
            "controlled_experiment_ids": [],
            "raw_support_events": 0,
            "raw_contradiction_events": 0,
            "new_distinct_control_action_acquired": False,
            "support": 0,
            "truth_status": SAGE5I_TRUTH_STATUS,
            "revision_status": "CANDIDATE_ONLY",
        }

    selected_request_ids = list(dict.fromkeys(eligible_by_action[selected_control]))[
        :min_control_contexts
    ]
    results: List[Dict[str, Any]] = []
    for request_id in selected_request_ids:
        request = request_by_id[request_id]
        result = execute_followup_experiment(
            source_request=request,
            metric=str(candidate.get("predicted_metric", request.get("metric", ""))),
            control_action=selected_control,
            followup_id=f"sage5i::{candidate.get('candidate_id', '')}::control_expansion",
            purpose="action_distinct_control_surface_expansion",
            environments_dir=environments_dir,
            env_factory=env_factory,
            execution_cache=execution_cache,
            experiments=experiments,
        )
        result.update(
            {
                "source_truth_status": str(
                    result.get("truth_status", SAGE5H_TRUTH_STATUS)
                ),
                "truth_status": SAGE5I_TRUTH_STATUS,
                "control_expansion_event_counted_as_scientific_support": False,
            }
        )
        results.append(result)
    exact = [
        row
        for row in results
        if str(row.get("execution_status", "")) == "EXECUTED"
        and bool(row.get("live_prefix_replay_exact", False))
    ]
    support_events = sum(int(row.get("support_events", 0) or 0) for row in exact)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in exact
    )
    acquired = bool(
        len(exact) >= min_control_contexts
        and support_events >= min_control_contexts
        and contradiction_events == 0
    )
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "candidate_key": str(candidate.get("candidate_key", "")),
        "action": str(candidate.get("action", "")),
        "expansion_status": (
            EXPANSION_ACQUIRED if acquired else EXPANSION_EXECUTION_INCOMPLETE
        ),
        "new_control_action": selected_control,
        "minimum_control_contexts": min_control_contexts,
        "controlled_experiment_ids": [
            str(row.get("experiment_id", "")) for row in results
        ],
        "raw_support_events": support_events,
        "raw_contradiction_events": contradiction_events,
        "new_distinct_control_action_acquired": acquired,
        "support": 0,
        "truth_status": SAGE5I_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
    }


def parameterized_control_options(
    *,
    candidate: Mapping[str, Any],
    existing_controls: Sequence[str],
    available_variants: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    target_action = str(candidate.get("action", ""))
    target_args = candidate.get("action_args")
    options: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for variant in available_variants:
        name = str(variant.get("action", ""))
        args = dict(variant.get("action_args", {}) or {})
        role = ""
        if name == target_action and _canonical_json(args) != _canonical_json(
            target_args or {}
        ):
            role = "same_action_alternative_args"
        elif name in set(existing_controls) and args:
            role = "existing_control_parameter_variant"
        if not role:
            continue
        key = _canonical_json({"action": name, "action_args": args})
        if key in seen:
            continue
        seen.add(key)
        options.append(
            {
                "action": name,
                "action_args": args,
                "parameterized_control_role": role,
                "counted_as_action_distinct_control": False,
            }
        )
    return sorted(
        options,
        key=lambda row: (
            str(row.get("action", "")),
            _canonical_json(row.get("action_args", {})),
        ),
    )


def update_sage5i_candidate_assessments(
    *,
    candidates: Sequence[Mapping[str, Any]],
    prior_assessments: Mapping[str, Mapping[str, Any]],
    surfaces: Sequence[Mapping[str, Any]],
    expansions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    surface_by_id = {str(row.get("candidate_id", "")): row for row in surfaces}
    expansion_by_id = {str(row.get("candidate_id", "")): row for row in expansions}
    updated: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        prior = dict(prior_assessments.get(candidate_id, {}) or {})
        surface = dict(surface_by_id.get(candidate_id, {}) or {})
        expansion = dict(expansion_by_id.get(candidate_id, {}) or {})
        acquired = bool(expansion.get("new_distinct_control_action_acquired", False))
        raw_support = int(prior.get("raw_support_events_after", 0) or 0) + int(
            expansion.get("raw_support_events", 0) or 0
        )
        contexts = int(prior.get("independent_context_events_after", 0) or 0)
        controls = int(prior.get("distinct_control_actions_after", 0) or 0) + (
            1 if acquired else 0
        )
        contradictions = int(prior.get("contradiction_events_after", 0) or 0) + int(
            expansion.get("raw_contradiction_events", 0) or 0
        )
        missing = missing_revision_requirements(
            raw_support_events=raw_support,
            independent_context_events=contexts,
            distinct_control_actions=controls,
            contradiction_events=contradictions,
        )
        bounded_exhausted = bool(
            surface.get("control_surface_exhausted_action_distinct", False)
        )
        recommendation = a32_intake_recommendation(missing)
        if bounded_exhausted and "minimum_distinct_control_actions" in missing:
            recommendation = A32_PARAMETERIZED_PROTOCOL_REQUIRED
        updated.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": str(candidate.get("candidate_key", "")),
                "game_id": str(candidate.get("game_id", "")),
                "action": str(candidate.get("action", "")),
                "action_args": candidate.get("action_args"),
                "matching_contexts_scanned": int(
                    surface.get("contexts_scanned", 0) or 0
                ),
                "replay_exact_contexts_scanned": int(
                    surface.get("replay_exact_contexts", 0) or 0
                ),
                "contexts_with_action_distinct_control": int(
                    surface.get("contexts_with_new_action_distinct_control", 0) or 0
                ),
                "parameterized_control_option_count": int(
                    surface.get("parameterized_control_option_count", 0) or 0
                ),
                "new_distinct_control_action_acquired": acquired,
                "new_control_action": str(expansion.get("new_control_action", "")),
                "raw_support_events_after": raw_support,
                "independent_context_events_after": contexts,
                "distinct_control_actions_after": controls,
                "contradiction_events_after": contradictions,
                "bounded_action_distinct_surface_exhausted": bounded_exhausted,
                "missing_revision_requirements": list(missing),
                "a32_intake_recommendation": recommendation,
                "ready_for_A32_intake": not missing,
                "parameterized_controls_counted_as_distinct_actions": False,
                "bounded_exhaustion_counted_as_refutation": False,
                "candidate_assessment_counted_as_revision": False,
                "support": 0,
                "truth_status": SAGE5I_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return updated


def build_a32_parameterized_control_protocol_proposal(
    *,
    surfaces: Sequence[Mapping[str, Any]],
    updated_assessments: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    exhausted = [
        row
        for row in surfaces
        if bool(row.get("control_surface_exhausted_action_distinct", False))
    ]
    proposal_required = bool(exhausted)
    return {
        "proposal_id": "sage5i::a32_parameterized_control_protocol_proposal::001",
        "proposal_required": proposal_required,
        "proposal_status": (
            "A32_REVIEW_REQUIRED_DO_NOT_AUTO_RELAX_CRITERION"
            if proposal_required
            else "NOT_REQUIRED_ACTION_DISTINCT_CONTROL_FOUND"
        ),
        "bounded_evidence_scope": "SAGE.5e_matching_replayable_candidate_contexts",
        "candidates_affected": [str(row.get("candidate_id", "")) for row in exhausted],
        "historical_requirement": {
            "minimum_distinct_control_action_names": 2,
            "currently_satisfied": all(
                int(row.get("distinct_control_actions_after", 0) or 0) >= 2
                for row in updated_assessments
            ),
        },
        "parameterized_control_option_counts": {
            str(row.get("candidate_id", "")): int(
                row.get("parameterized_control_option_count", 0) or 0
            )
            for row in surfaces
        },
        "proposed_pre_registration_requirements": [
            "A32_must_explicitly_authorize_parameterized_intervention_controls",
            "action_args_must_differ_from_target_args",
            "parameterized_controls_must_run_in_replay_exact_distinct_contexts",
            "minimum_two_parameter_variants_if_action_name_is_reused",
            "zero_contradiction_events_required",
            "parameterized_variants_must_not_be_relabelled_as_distinct_action_names",
        ],
        "allowed_decisions": [
            "retain_strict_action_distinct_requirement_and_keep_unresolved",
            "authorize_pre_registered_parameterized_control_protocol",
            "reject_candidate_as_unidentifiable_in_current_action_surface",
        ],
        "recommendation": "A32_DECISION_REQUIRED_NO_AUTOMATIC_CONFIRMATION",
        "protocol_proposal_counted_as_revision": False,
        "support": 0,
        "truth_status": SAGE5I_TRUTH_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def summarize_sage5i(
    *,
    source_sage5e: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    surfaces: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    expansions: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked_experiments: Sequence[Mapping[str, Any]],
    updated_assessments: Sequence[Mapping[str, Any]],
    protocol_proposal: Mapping[str, Any],
) -> Dict[str, Any]:
    contexts_considered = sum(
        int(row.get("contexts_scanned", 0) or 0) for row in surfaces
    )
    exact_audits = sum(
        1 for row in audits if bool(row.get("live_prefix_replay_exact", False))
    )
    expanded = [
        row
        for row in expansions
        if bool(row.get("new_distinct_control_action_acquired", False))
    ]
    exhausted = [
        row
        for row in surfaces
        if bool(row.get("control_surface_exhausted_action_distinct", False))
    ]
    if len(expanded) == len(candidates) and candidates:
        outcome_status = SAGE5I_CONTROL_DIVERSITY_ACQUIRED
    elif expanded:
        outcome_status = SAGE5I_PARTIAL_CONTROL_DIVERSITY
    else:
        outcome_status = SAGE5I_ACTION_DISTINCT_EXHAUSTED
    surface_signatures = Counter(
        signature
        for surface in surfaces
        for signature, count in (
            surface.get("action_surface_signature_counts", {}) or {}
        ).items()
        for _ in range(int(count or 0))
    )
    all_audited_exact = (
        bool(contexts_considered) and exact_audits == contexts_considered
    )
    return {
        "source_sage5e_outcome_status": str(source_sage5e.get("outcome_status", "")),
        "source_requests_available": len(
            source_sage5e.get("mini_frontier_m3_requests", []) or []
        ),
        "candidates_evaluated": len(candidates),
        "candidate_contexts_considered": contexts_considered,
        "unique_candidate_contexts": len(
            {
                str(row.get("context_snapshot_hash", ""))
                for row in audits
                if str(row.get("context_snapshot_hash", ""))
            }
        ),
        "context_surface_audits": len(audits),
        "replay_exact_context_surface_audits": exact_audits,
        "action_surface_signature_counts": dict(sorted(surface_signatures.items())),
        "contexts_with_third_action_family": sum(
            int(row.get("contexts_with_new_action_distinct_control", 0) or 0)
            for row in surfaces
        ),
        "candidate_context_counts": {
            str(row.get("action", "")): int(row.get("contexts_scanned", 0) or 0)
            for row in surfaces
        },
        "parameterized_control_option_counts": {
            str(row.get("action", "")): int(
                row.get("parameterized_control_option_count", 0) or 0
            )
            for row in surfaces
        },
        "candidates_with_action_distinct_surface_exhausted": len(exhausted),
        "candidates_with_new_distinct_control": len(expanded),
        "control_expansion_experiments_executed": len(experiments),
        "control_expansion_experiments_blocked": len(blocked_experiments),
        "raw_support_events": sum(
            int(row.get("support_events", 0) or 0) for row in experiments
        ),
        "raw_contradiction_events": sum(
            int(row.get("contradiction_events", 0) or 0) for row in experiments
        ),
        "candidates_ready_for_A32_intake": sum(
            1
            for row in updated_assessments
            if bool(row.get("ready_for_A32_intake", False))
        ),
        "bounded_action_distinct_exhaustion_proven": bool(
            all_audited_exact and len(exhausted) == len(candidates) and candidates
        ),
        "bounded_scope_only": True,
        "all_candidate_contexts_audited_exact": all_audited_exact,
        "a32_parameterized_protocol_proposal_generated": bool(
            protocol_proposal.get("proposal_required", False)
        ),
        "gate_passed": all_audited_exact and len(surfaces) == len(candidates),
        "outcome_status": outcome_status,
        "bounded_control_surface_exhaustion_counted_as_refutation": False,
        "parameterized_controls_counted_as_distinct_actions": False,
        "control_expansion_events_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE5I_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "execution_performed": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_sage5i_sources(
    source_sage5e: Mapping[str, Any],
    source_sage5g: Mapping[str, Any],
    source_sage5h: Mapping[str, Any],
) -> None:
    for label, source in (
        ("SAGE.5e", source_sage5e),
        ("SAGE.5g", source_sage5g),
        ("SAGE.5h", source_sage5h),
    ):
        summary = dict(source.get("summary", {}) or {})
        if int(source.get("support", summary.get("support", 0)) or 0) != 0:
            raise ValueError(f"{label} support must remain 0")
        if str(source.get("revision_status", "CANDIDATE_ONLY")) != "CANDIDATE_ONLY":
            raise ValueError(f"{label} must remain candidate-only")
        if bool(source.get("revision_performed", False)):
            raise ValueError(f"{label} must not perform revision")
        if int(source.get("wrong_confirmations", 0) or 0) != 0:
            raise ValueError(f"{label} wrong_confirmations must remain 0")
        if bool(source.get("a32_write_performed", False)) or bool(
            source.get("a33_write_performed", False)
        ):
            raise ValueError(f"{label} cannot write A32/A33")
    if bool(source_sage5h.get("control_surface_block_counted_as_refutation", False)):
        raise ValueError("SAGE.5h control-surface block cannot count as refutation")
    if bool(source_sage5h.get("candidate_assessment_counted_as_revision", False)):
        raise ValueError("SAGE.5h candidate assessment cannot count as revision")


def write_sage5i_control_surface_expansion(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _request_matches_candidate(
    candidate: Mapping[str, Any],
    request: Mapping[str, Any],
) -> bool:
    if str(request.get("target_action", "")) != str(candidate.get("action", "")):
        return False
    request_args = request.get("target_action_args")
    candidate_args = candidate.get("action_args")
    if _canonical_json(request_args or {}) != _canonical_json(candidate_args or {}):
        return False
    predicted = dict(candidate.get("predicted_effect_signature", {}) or {})
    observed = dict(request.get("diff_signature", {}) or {})
    return bool(
        int(predicted.get("changed_cells", 0) or 0)
        == int(observed.get("changed_cells", 0) or 0)
        and dict(predicted.get("color_transitions", {}) or {})
        == dict(observed.get("color_transitions", {}) or {})
        and bool(predicted.get("terminal_after", False))
        == bool(observed.get("terminal_after", False))
        and int(predicted.get("levels_delta", 0) or 0)
        == int(observed.get("levels_delta", 0) or 0)
    )


def _unique_action_variants(
    audits: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    variants: Dict[str, Dict[str, Any]] = {}
    for audit in audits:
        for row in audit.get("available_action_variants", []) or []:
            variant = {
                "action": str(row.get("action", "")),
                "action_args": dict(row.get("action_args", {}) or {}),
            }
            variants[_canonical_json(variant)] = variant
    return sorted(
        variants.values(),
        key=lambda row: (
            str(row.get("action", "")),
            _canonical_json(row.get("action_args", {})),
        ),
    )


def _request_sort_key(request: Mapping[str, Any]) -> Tuple[int, int, str]:
    return (
        _budget_from_request(request),
        int(request.get("source_step", 0) or 0),
        str(request.get("request_id", "")),
    )


def _budget_from_request(request: Mapping[str, Any]) -> int:
    transition = str(request.get("source_transition_id", ""))
    marker = "::budget_"
    if marker in transition:
        return int(transition.split(marker, 1)[1].split("::", 1)[0].split("_", 1)[0])
    return 0


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand the SAGE unknown-game action-distinct control surface.",
    )
    parser.add_argument(
        "--source-sage5e",
        default=str(DEFAULT_SAGE5E_DISTRIBUTED_LIVE_MINI_FRONTIER_RESULTS_PATH),
    )
    parser.add_argument(
        "--source-sage5g", default=str(DEFAULT_SAGE5G_A32_REVIEW_HANDOFF_PATH)
    )
    parser.add_argument(
        "--source-sage5h",
        default=str(DEFAULT_SAGE5H_CONTROLLED_FOLLOWUP_ACQUISITION_PATH),
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--minimum-control-contexts", type=int, default=MIN_CONTROL_CONTEXTS
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE5I_CONTROL_SURFACE_EXPANSION_PATH)
    )
    args = parser.parse_args(argv)
    run_sage5i_control_surface_expansion(
        source_sage5e_path=args.source_sage5e,
        source_sage5g_path=args.source_sage5g,
        source_sage5h_path=args.source_sage5h,
        environments_dir=args.environments_dir,
        output_path=args.out,
        min_control_contexts=args.minimum_control_contexts,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
