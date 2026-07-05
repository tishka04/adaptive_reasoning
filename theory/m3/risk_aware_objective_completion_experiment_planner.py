"""M3.G5 planner for objective-readiness / commit-action experiments.

Consumes M2.G2 risk-aware objective-completion hypotheses and compiles
candidate-only controlled experiment requests. This planner does not execute
environment steps and does not create scientific verdicts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.m2.risk_aware_objective_completion_hypothesis_generator import (
    DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH,
    SUBSTRATE_ACTIONS_NOT_TARGETS,
)

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "risk_aware_objective_completion_experiment_requests.json"
)
RISK_AWARE_OBJECTIVE_REQUESTS_SCHEMA_VERSION = (
    "m3.risk_aware_objective_completion_experiment_requests.v1"
)

READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT = (
    "READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT"
)
BLOCKED_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT = (
    "BLOCKED_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT"
)
COMPILED_STATUS = "OBJECTIVE_READINESS_EXPERIMENT_REQUESTS_COMPILED_CANDIDATE_ONLY"

HOLD_OR_STOP_STATE_CONTROL = "hold_or_stop_state"
ACTION6_ONLY_CONTROL = "ACTION6"
FROZEN_CONTEXTUAL_SELECTOR_CONTROL = "frozen_contextual_selector"
STATIC_ACTION6_ACTION3_CONTROL = "static_ACTION6,ACTION3"
STATIC_ACTION6_ACTION4_CONTROL = "static_ACTION6,ACTION4"
RELATION_PROGRESS_POLICY_CONTROL = "relation_progress_policy"

PRIMARY_METRICS = (
    "objective_completion_signal",
    "levels_completed_after_rollout",
    "terminal_reentry_rate",
)
SAFETY_METRICS = (
    "terminal_adjusted_progress_after_stop",
    "terminal_state_after_rollout",
)
DIAGNOSTIC_METRICS = (
    "completion_ready_signature",
    "proxy_completion_divergence",
    "objective_readiness_precision",
    "relation_delta_after_stop",
    "new_relation_states",
    "changed_pixels",
    "global_configuration_signature",
    "actor_target_contact_graph",
    "terminal_horizon_remaining",
    "terminal_horizon_band",
    "hold_baseline_terminal_adjusted_progress",
    "hold_baseline_band",
)
CENTRAL_DECISION_RULE = (
    "candidate protocol improves objective_completion_signal or "
    "levels_completed_after_rollout over matched controls while keeping "
    "terminal_reentry_rate at or below frozen_contextual_selector"
)
FALSIFICATION_GUARD = (
    "no completion/level gain over frozen_contextual_selector, or terminal "
    "re-entry increases relative to frozen_contextual_selector"
)


@dataclass(frozen=True)
class RiskAwareObjectiveExperimentRequest:
    request_id: str
    source_hypothesis_id: str
    source_request_id: str
    game_id: str
    hypothesis_family: str
    hypothesis_tested: str
    requested_experiment_style: str
    falsification_signal: str
    substrate_selection_spec: Dict[str, Any]
    candidate_protocols: Tuple[Dict[str, Any], ...]
    control_conditions: Tuple[Dict[str, Any], ...]
    primary_metrics: Tuple[str, ...]
    safety_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    required_observables: Tuple[str, ...]
    decision_rule: Dict[str, str]
    falsification_criteria: Tuple[Dict[str, Any], ...]
    planning_rationale: str
    forbidden_interpretations: Tuple[str, ...]
    status: str = READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    execution_performed: bool = False
    policy_rollout_performed: bool = False
    environment_step_performed: bool = False
    revision_performed: bool = False
    wrong_confirmations: int = 0
    m2_hypothesis_counted_as_confirmation: bool = False
    experiment_request_counted_as_support: bool = False
    experiment_result_counted_as_scientific_verdict: bool = False
    a32_write_performed: bool = False
    a33_write_performed: bool = False
    a32_remains_only_verdict_location: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_hypothesis_id": self.source_hypothesis_id,
            "source_request_id": self.source_request_id,
            "game_id": self.game_id,
            "hypothesis_family": self.hypothesis_family,
            "hypothesis_tested": self.hypothesis_tested,
            "requested_experiment_style": self.requested_experiment_style,
            "falsification_signal": self.falsification_signal,
            "substrate_selection_spec": dict(self.substrate_selection_spec),
            "candidate_protocols": [
                dict(protocol) for protocol in self.candidate_protocols
            ],
            "control_conditions": [
                dict(condition) for condition in self.control_conditions
            ],
            "primary_metrics": list(self.primary_metrics),
            "safety_metrics": list(self.safety_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "required_observables": list(self.required_observables),
            "decision_rule": dict(self.decision_rule),
            "falsification_criteria": [
                dict(item) for item in self.falsification_criteria
            ],
            "planning_rationale": self.planning_rationale,
            "forbidden_interpretations": list(self.forbidden_interpretations),
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "execution_performed": self.execution_performed,
            "policy_rollout_performed": self.policy_rollout_performed,
            "environment_step_performed": self.environment_step_performed,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "m2_hypothesis_counted_as_confirmation": (
                self.m2_hypothesis_counted_as_confirmation
            ),
            "experiment_request_counted_as_support": (
                self.experiment_request_counted_as_support
            ),
            "experiment_result_counted_as_scientific_verdict": (
                self.experiment_result_counted_as_scientific_verdict
            ),
            "a32_write_performed": self.a32_write_performed,
            "a33_write_performed": self.a33_write_performed,
            "a32_remains_only_verdict_location": (
                self.a32_remains_only_verdict_location
            ),
        }


def run_risk_aware_objective_completion_experiment_planning(
    *,
    risk_aware_objective_hypotheses_path: str
    | Path = DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(risk_aware_objective_hypotheses_path)
    _validate_source_payload(payload)
    hypotheses = risk_aware_objective_hypotheses_from_payload(payload)
    requests = [
        build_risk_aware_objective_experiment_request(hypothesis)
        for hypothesis in hypotheses
        if is_ready_risk_aware_objective_hypothesis(hypothesis)
    ]
    skipped = [
        skipped_risk_aware_objective_hypothesis(hypothesis)
        for hypothesis in hypotheses
        if not is_ready_risk_aware_objective_hypothesis(hypothesis)
    ]
    for request in requests:
        validate_risk_aware_objective_experiment_request(request)

    return {
        "config": {
            "risk_aware_objective_hypotheses_path": str(
                risk_aware_objective_hypotheses_path
            ),
            "schema_version": RISK_AWARE_OBJECTIVE_REQUESTS_SCHEMA_VERSION,
            "inputs_read": ["M2.G2"],
            "artifacts_not_modified": ["M2", "A32", "A33"],
            "execution_performed": False,
            "policy_rollout_performed": False,
            "environment_step_performed": False,
            "central_experimental_unit": (
                "risk_aware_post_stop_substrate -> readiness_or_commit_protocol"
            ),
            "substrate_sources": [
                "diagnostics/m3/objective_conversion_diverse_safe_stop_validation.json",
                "diagnostics/p3/risk_targeted_contextual_post_stop_policy_validation.json",
            ],
            "controls": common_control_condition_names(),
        },
        "summary": summarize_risk_aware_objective_requests(
            hypotheses=hypotheses,
            requests=requests,
            skipped=skipped,
        ),
        "risk_aware_objective_experiment_requests": [
            request.to_dict() for request in requests
        ],
        "skipped_risk_aware_objective_hypotheses": skipped,
        "planner_outcome_status": COMPILED_STATUS if requests else "NO_REQUESTS_COMPILED",
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": False,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def build_risk_aware_objective_experiment_request(
    hypothesis: Mapping[str, Any],
) -> RiskAwareObjectiveExperimentRequest:
    family = str(hypothesis.get("hypothesis_family", ""))
    return RiskAwareObjectiveExperimentRequest(
        request_id=risk_aware_objective_request_id(hypothesis),
        source_hypothesis_id=str(hypothesis.get("hypothesis_id", "")),
        source_request_id=str(hypothesis.get("source_request_id", "")),
        game_id=str(hypothesis.get("game_id", "")),
        hypothesis_family=family,
        hypothesis_tested=str(
            hypothesis.get("claim", "") or hypothesis.get("predicted_effect", "")
        ),
        requested_experiment_style=str(
            hypothesis.get("requested_experiment_style", "")
            or hypothesis.get("suggested_experiment_style", "")
        ),
        falsification_signal=str(hypothesis.get("falsification_signal", "")),
        substrate_selection_spec=substrate_selection_spec_for_family(family),
        candidate_protocols=candidate_protocols_for_hypothesis(hypothesis),
        control_conditions=common_control_conditions(),
        primary_metrics=PRIMARY_METRICS,
        safety_metrics=SAFETY_METRICS,
        diagnostic_metrics=DIAGNOSTIC_METRICS,
        required_observables=tuple(
            str(value) for value in hypothesis.get("required_observables", []) or []
        ),
        decision_rule={
            "central": CENTRAL_DECISION_RULE,
            "falsification_guard": FALSIFICATION_GUARD,
            "completion_priority": (
                "objective_completion_signal and levels_completed_after_rollout "
                "dominate proxy progress"
            ),
        },
        falsification_criteria=falsification_criteria_for_hypothesis(hypothesis),
        planning_rationale=planning_rationale_for_family(family),
        forbidden_interpretations=tuple(
            str(value)
            for value in hypothesis.get("forbidden_interpretations", []) or []
        ),
    )


def candidate_protocols_for_hypothesis(
    hypothesis: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    family = str(hypothesis.get("hypothesis_family", ""))
    target = str(hypothesis.get("candidate_action", ""))
    style = str(
        hypothesis.get("requested_experiment_style", "")
        or hypothesis.get("suggested_experiment_style", "")
    )
    observables = [str(value) for value in hypothesis.get("required_observables", []) or []]
    if family == "objective_readiness_detection":
        return (
            {
                "protocol_id": "readiness_probe",
                "protocol_family": "objective_readiness_detection",
                "target_detector": target,
                "experiment_style": style,
                "readiness_features": observables,
                "candidate_state_partitions": [
                    "readiness_positive",
                    "matched_high_proxy_readiness_negative",
                    "matched_hold_band_control",
                ],
                "role": "detect_completion_ready_state_before_commit",
            },
        )
    if family == "post_conversion_commit_action_search":
        commit_actions = commit_action_candidates_for_hypothesis(hypothesis)
        return (
            {
                "protocol_id": "post_conversion_commit_matrix",
                "protocol_family": "post_conversion_commit_action_search",
                "target_commit_action": target,
                "commit_action_candidates": commit_actions,
                "optional_commit_actions_if_available": ["ACTION7"],
                "experiment_style": style,
                "pre_commit_state_requirements": [
                    "risk_aware_terminal_safe_post_stop_state",
                    "safe_conversion_state",
                    "hold_baseline_measurable",
                ],
                "role": "test_distinct_completion_commit_after_safe_conversion",
            },
        )
    if family == "goal_state_representation_beyond_safe_progress":
        return (
            {
                "protocol_id": "goal_representation_discriminator",
                "protocol_family": "goal_state_representation_beyond_safe_progress",
                "target_representation": target,
                "experiment_style": style,
                "representation_features": observables,
                "compare_against": [
                    "terminal_adjusted_progress",
                    "relation_delta_after_stop",
                    "changed_pixels",
                ],
                "role": "test_goal_feature_beyond_safe_proxy_progress",
            },
        )
    if family == "proxy_progress_vs_completion_discriminator":
        return (
            {
                "protocol_id": "proxy_completion_discriminator",
                "protocol_family": "proxy_progress_vs_completion_discriminator",
                "target_discriminator": target,
                "experiment_style": style,
                "proxy_features": observables,
                "state_strata": [
                    "high_hold_high_proxy",
                    "changed_pixels_high",
                    "relation_delta_high",
                    "completion_ready_candidate",
                ],
                "role": "separate_scoring_states_from_completion_ready_states",
            },
        )
    if family == "risk_aware_selector_completion_gap":
        return (
            {
                "protocol_id": "selector_gap_policy_ablation",
                "protocol_family": "risk_aware_selector_completion_gap",
                "target_policy_variant": target,
                "experiment_style": style,
                "policy_variants": [
                    "frozen_contextual_selector",
                    "selector_plus_commit_branch",
                    "completion_probability_selector_target",
                    "safe_conversion_then_objective_commit_policy",
                ],
                "role": "test_why_safe_risk_aware_selector_does_not_complete",
            },
        )
    return (
        {
            "protocol_id": "generic_m3_g5_protocol",
            "protocol_family": family,
            "target": target,
            "experiment_style": style,
            "role": "generic_risk_aware_objective_completion_probe",
        },
    )


def substrate_selection_spec_for_family(family: str) -> Dict[str, Any]:
    categories = [
        {
            "category": "risk_aware_post_stop_safe_contexts",
            "source": "P3.G4 accepted_risk_targeted_safe_stops",
            "acceptance": [
                "replay_exact",
                "non_terminal",
                "terminal_safe",
                "hold_baseline_measurable",
            ],
        },
        {
            "category": "selector_action6_fallback_contexts",
            "source": "P3.G4 policy_decision_records",
            "acceptance": ["selected_option == ACTION6", "terminal_rate == 0"],
        },
        {
            "category": "selector_extension_safe_contexts",
            "source": "P3.G4 policy_decision_records",
            "acceptance": [
                "selected_option in ACTION6,ACTION3|ACTION6,ACTION4",
                "candidate_terminal_reentry == false",
            ],
        },
        {
            "category": "static_extension_terminal_risk_contexts",
            "source": "P3.G4 risk_targeted_extension_risk_stats",
            "acceptance": [
                "static extension terminal record exists",
                "hold_high_ge_120 or horizon_mid_45_54",
            ],
        },
    ]
    if family == "post_conversion_commit_action_search":
        preferred = [
            "selector_action6_fallback_contexts",
            "selector_extension_safe_contexts",
        ]
    elif family == "proxy_progress_vs_completion_discriminator":
        preferred = [
            "static_extension_terminal_risk_contexts",
            "risk_aware_post_stop_safe_contexts",
        ]
    else:
        preferred = [row["category"] for row in categories]
    return {
        "base_state_family": "risk_aware_terminal_safe_post_stop_conversion_state",
        "substrate_categories": categories,
        "preferred_categories_for_family": preferred,
        "coverage_requirements": [
            "multiple_hold_baseline_band_values_when_available",
            "multiple_terminal_horizon_band_values_when_available",
            "include_hold_high_ge_120_and_horizon_mid_45_54_when_available",
            "include_contexts_where_static_extensions_are_terminal_when_available",
        ],
        "selection_counted_as_support": False,
    }


def common_control_conditions() -> Tuple[Dict[str, Any], ...]:
    return (
        {
            "condition_id": HOLD_OR_STOP_STATE_CONTROL,
            "condition_kind": "control",
            "condition_family": HOLD_OR_STOP_STATE_CONTROL,
            "role": "zero_action_control",
        },
        {
            "condition_id": ACTION6_ONLY_CONTROL,
            "condition_kind": "control",
            "condition_family": "ACTION6_only",
            "action_or_sequence": ["ACTION6"],
            "role": "safe_weak_conversion_control",
        },
        {
            "condition_id": FROZEN_CONTEXTUAL_SELECTOR_CONTROL,
            "condition_kind": "control",
            "condition_family": FROZEN_CONTEXTUAL_SELECTOR_CONTROL,
            "policy": "P3.G2_frozen_contextual_post_stop_selector",
            "role": "risk_aware_policy_control",
        },
        {
            "condition_id": STATIC_ACTION6_ACTION3_CONTROL,
            "condition_kind": "control",
            "condition_family": "static_extension_policy",
            "action_or_sequence": ["ACTION6", "ACTION3"],
            "role": "risky_static_extension_control",
        },
        {
            "condition_id": STATIC_ACTION6_ACTION4_CONTROL,
            "condition_kind": "control",
            "condition_family": "static_extension_policy",
            "action_or_sequence": ["ACTION6", "ACTION4"],
            "role": "risky_static_extension_control",
        },
        {
            "condition_id": RELATION_PROGRESS_POLICY_CONTROL,
            "condition_kind": "control",
            "condition_family": RELATION_PROGRESS_POLICY_CONTROL,
            "policy": "abstract_relation_progress_continuation",
            "role": "proxy_progress_control",
        },
    )


def common_control_condition_names() -> Tuple[str, ...]:
    return tuple(str(row["condition_id"]) for row in common_control_conditions())


def commit_action_candidates_for_hypothesis(
    hypothesis: Mapping[str, Any],
) -> list[str]:
    target = str(hypothesis.get("candidate_action", ""))
    controls = [
        str(action)
        for action in (
            (hypothesis.get("testability", {}) or {}).get(
                "suggested_control_actions", []
            )
            or []
        )
    ]
    ordered = [target, *controls, "ACTION1", "ACTION2", "ACTION5"]
    result: list[str] = []
    for action in ordered:
        if not action or action in result:
            continue
        if action in SUBSTRATE_ACTIONS_NOT_TARGETS:
            continue
        result.append(action)
    return result


def falsification_criteria_for_hypothesis(
    hypothesis: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    source = dict(hypothesis.get("falsification", {}) or {})
    criteria: list[Dict[str, Any]] = []
    if source.get("metric"):
        criteria.append(
            {
                "metric": str(source.get("metric", "")),
                "support_condition": str(source.get("support_condition", "")),
                "failure_condition": str(source.get("failure_condition", "")),
                "minimum_effect_size": source.get("minimum_effect_size", 1),
                "source": "m2_g2_hypothesis",
            }
        )
    criteria.append(
        {
            "metric": "objective_completion_signal",
            "support_condition": (
                "candidate_protocol_completion_signal > best_control_completion_signal "
                "and candidate_protocol_terminal_reentry_rate <= "
                "frozen_contextual_selector_terminal_reentry_rate"
            ),
            "failure_condition": FALSIFICATION_GUARD,
            "minimum_effect_size": 1,
            "source": "m3_g5_central_completion_guard",
        }
    )
    return tuple(criteria)


def planning_rationale_for_family(family: str) -> str:
    if family == "objective_readiness_detection":
        return (
            "Compile readiness probes that separate completion-ready states from "
            "safe proxy-progress states before any commit or extension decision."
        )
    if family == "post_conversion_commit_action_search":
        return (
            "Compile a post-conversion commit matrix over explicit commit actions, "
            "after the risk-aware selector has reached terminal-safe substrate states."
        )
    if family == "goal_state_representation_beyond_safe_progress":
        return (
            "Compile representation discriminators that test global/contact/region "
            "goal features against terminal-adjusted proxy progress."
        )
    if family == "proxy_progress_vs_completion_discriminator":
        return (
            "Compile discriminators for high proxy-progress states that may not be "
            "completion-ready despite safe terminal-adjusted progress."
        )
    if family == "risk_aware_selector_completion_gap":
        return (
            "Compile selector ablations that compare the frozen selector with "
            "completion-targeted or two-stage variants without changing support."
        )
    return "Compile M3.G5 risk-aware objective-completion controlled experiment."


def summarize_risk_aware_objective_requests(
    *,
    hypotheses: Sequence[Mapping[str, Any]],
    requests: Sequence[RiskAwareObjectiveExperimentRequest],
    skipped: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    families = sorted({request.hypothesis_family for request in requests})
    styles = sorted({request.requested_experiment_style for request in requests})
    substrate_categories = sorted(
        {
            str(category.get("category", ""))
            for request in requests
            for category in request.substrate_selection_spec.get(
                "substrate_categories", []
            )
        }
    )
    return {
        "risk_aware_objective_hypotheses_consumed": len(hypotheses),
        "risk_aware_objective_experiment_requests_generated": len(requests),
        "skipped_risk_aware_objective_hypotheses": len(skipped),
        "covered_hypothesis_families": families,
        "all_five_families_covered": set(families)
        >= {
            "objective_readiness_detection",
            "post_conversion_commit_action_search",
            "goal_state_representation_beyond_safe_progress",
            "proxy_progress_vs_completion_discriminator",
            "risk_aware_selector_completion_gap",
        },
        "covered_experiment_styles": styles,
        "substrate_categories": substrate_categories,
        "controls_per_request": list(common_control_condition_names()),
        "primary_metrics": list(PRIMARY_METRICS),
        "safety_metrics": list(SAFETY_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "central_decision_rule": CENTRAL_DECISION_RULE,
        "planner_outcome_status": COMPILED_STATUS if requests else "NO_REQUESTS_COMPILED",
        "action6_extension_retest_requests_generated": False,
        "substrate_actions_not_target_hypotheses": list(SUBSTRATE_ACTIONS_NOT_TARGETS),
        "execution_performed": False,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_request_counted_as_support": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def risk_aware_objective_hypotheses_from_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    for batch in payload.get("risk_aware_objective_hypothesis_batches", []) or []:
        if not isinstance(batch, Mapping):
            continue
        for hypothesis in batch.get("candidate_hypotheses", []) or []:
            if isinstance(hypothesis, Mapping):
                rows.append(dict(hypothesis))
    return tuple(rows)


def is_ready_risk_aware_objective_hypothesis(hypothesis: Mapping[str, Any]) -> bool:
    return bool(
        str(hypothesis.get("status", "")) == "UNRESOLVED"
        and int(hypothesis.get("support", 0) or 0) == 0
        and str(hypothesis.get("revision_status", "")) == "CANDIDATE_ONLY"
        and str(hypothesis.get("truth_status", "")) == "NOT_EVALUATED_BY_M2"
        and not bool(hypothesis.get("revision_performed", False))
        and int(hypothesis.get("wrong_confirmations", 0) or 0) == 0
        and bool(hypothesis.get("ready_for_m3_g5", False))
        and bool(hypothesis.get("ready_for_m3_g5_candidate_experiment_request", False))
        and bool((hypothesis.get("testability", {}) or {}).get("testable", False))
        and str(hypothesis.get("candidate_action", "")) not in SUBSTRATE_ACTIONS_NOT_TARGETS
        and list(hypothesis.get("required_observables", []) or [])
        and list(hypothesis.get("forbidden_interpretations", []) or [])
    )


def skipped_risk_aware_objective_hypothesis(
    hypothesis: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "source_hypothesis_id": str(hypothesis.get("hypothesis_id", "")),
        "hypothesis_family": str(hypothesis.get("hypothesis_family", "")),
        "reason": "not_ready_for_m3_g5_objective_completion_experiment",
        "status": BLOCKED_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def validate_risk_aware_objective_experiment_request(
    request: RiskAwareObjectiveExperimentRequest | Mapping[str, Any],
) -> None:
    data = (
        request.to_dict()
        if isinstance(request, RiskAwareObjectiveExperimentRequest)
        else dict(request)
    )
    if not str(data.get("request_id", "")):
        raise ValueError("request_id is required")
    if not str(data.get("source_hypothesis_id", "")):
        raise ValueError("source_hypothesis_id is required")
    if not data.get("candidate_protocols"):
        raise ValueError("candidate_protocols are required")
    for protocol in data.get("candidate_protocols", []) or []:
        target = str(
            protocol.get("target_commit_action", "")
            or protocol.get("target_detector", "")
            or protocol.get("target_representation", "")
            or protocol.get("target_discriminator", "")
            or protocol.get("target_policy_variant", "")
            or protocol.get("target", "")
        )
        if target in SUBSTRATE_ACTIONS_NOT_TARGETS:
            raise ValueError("ACTION6-led extension cannot be M3.G5 target protocol")
    controls = {
        str(control.get("condition_id", ""))
        for control in data.get("control_conditions", []) or []
    }
    for control in common_control_condition_names():
        if control not in controls:
            raise ValueError(f"{control} control is required")
    substrate = dict(data.get("substrate_selection_spec", {}) or {})
    categories = {
        str(row.get("category", ""))
        for row in substrate.get("substrate_categories", []) or []
        if isinstance(row, Mapping)
    }
    required_categories = {
        "risk_aware_post_stop_safe_contexts",
        "selector_action6_fallback_contexts",
        "selector_extension_safe_contexts",
        "static_extension_terminal_risk_contexts",
    }
    if not required_categories <= categories:
        raise ValueError("M3.G5 substrate categories are incomplete")
    if "objective_completion_signal" not in data.get("primary_metrics", []):
        raise ValueError("objective completion must be a primary metric")
    if str(data.get("status", "")) != READY_FOR_M3_G5_OBJECTIVE_COMPLETION_EXPERIMENT:
        raise ValueError("request must be ready for M3.G5 experiment")
    if str(data.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("request must remain candidate-only")
    if int(data.get("support", 0) or 0) != 0:
        raise ValueError("request support must remain 0")
    if str(data.get("truth_status", "")) != M3_REFINEMENT_TRUTH_STATUS:
        raise ValueError("request truth_status must remain M3-local")
    if bool(data.get("execution_performed", False)):
        raise ValueError("planner cannot execute requests")
    if bool(data.get("policy_rollout_performed", False)):
        raise ValueError("planner cannot run policy rollouts")
    if bool(data.get("environment_step_performed", False)):
        raise ValueError("planner cannot step the environment")
    if bool(data.get("revision_performed", False)):
        raise ValueError("request revision_performed must be false")
    if int(data.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("request wrong_confirmations must remain 0")
    if bool(data.get("m2_hypothesis_counted_as_confirmation", False)):
        raise ValueError("request cannot count M2 hypothesis as confirmation")
    if bool(data.get("experiment_request_counted_as_support", False)):
        raise ValueError("request cannot count as support")
    if bool(data.get("experiment_result_counted_as_scientific_verdict", False)):
        raise ValueError("experiment result cannot count as scientific verdict")
    if bool(data.get("a32_write_performed", False)) or bool(
        data.get("a33_write_performed", False)
    ):
        raise ValueError("planner must not write A32/A33")


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(summary.get("revision_performed", False)) or bool(
        payload.get("revision_performed", False)
    ):
        raise ValueError("source revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source summary wrong_confirmations must remain 0")
    for key in (
        "execution_performed",
        "policy_rollout_performed",
        "environment_step_performed",
    ):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"M2.G2 source must not have {key}")
    if bool(summary.get("action6_extension_retest_hypotheses_generated", True)):
        raise ValueError("M2.G2 source must not retest ACTION6-led extensions")
    if bool(payload.get("risk_aware_objective_hypotheses_counted_as_confirmation", False)):
        raise ValueError("risk-aware objective hypotheses cannot count as confirmation")
    if bool(payload.get("request_counted_as_scientific_verdict", False)):
        raise ValueError("P2 request cannot count as scientific verdict")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")
    for key in ("a32_write_performed", "a33_write_performed", "m3_write_performed"):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"source must not have {key}")


def risk_aware_objective_request_id(hypothesis: Mapping[str, Any]) -> str:
    source = str(hypothesis.get("hypothesis_id", "")).replace("::", "_")
    family = str(hypothesis.get("hypothesis_family", "risk_aware_objective"))
    return f"m3_g5::{source}::{family}"


def write_risk_aware_objective_experiment_requests(
    payload: Mapping[str, Any],
    output_path: str
    | Path = DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build M3.G5 risk-aware objective-completion experiment requests.",
    )
    parser.add_argument(
        "--hypotheses",
        type=Path,
        default=DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_risk_aware_objective_completion_experiment_planning(
        risk_aware_objective_hypotheses_path=args.hypotheses,
    )
    write_risk_aware_objective_experiment_requests(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "planner_outcome_status": payload["planner_outcome_status"],
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
