"""M2.G2 risk-aware objective-completion hypothesis generation from P2.G5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.risk_aware_objective_frontier_handoff_schema import (
    DEFAULT_P2_RISK_AWARE_OBJECTIVE_HANDOFF_REQUESTS_OUTPUT_PATH,
    HANDOFF_TARGET,
    HANDOFF_TYPE,
    TARGET_MODULES,
)
from theory.p2.risk_aware_post_stop_frontier_records import (
    OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION,
    RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER,
    RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION,
)

from .metric_registry import default_falsification_for_metric, is_metric_measurable
from .schema import (
    ContextSnapshot,
    FalsificationCriterion,
    FrontierConditionedHypothesis,
    HypothesisTestability,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_HYPOTHESIS_STATUS,
    M2_SCHEMA_VERSION,
    M2_TRUTH_STATUS,
    SourceGenerationAudit,
)
from .validators import validate_hypothesis


DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "risk_aware_objective_completion_hypotheses.json"
)
RISK_AWARE_OBJECTIVE_SCHEMA_VERSION = "m2.risk_aware_objective_completion.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
SUBSTRATE_ACTIONS_NOT_TARGETS = ("ACTION6", "ACTION6,ACTION3", "ACTION6,ACTION4")


def run_risk_aware_objective_completion_hypothesis_generator(
    *,
    risk_aware_objective_requests_path: str
    | Path = DEFAULT_P2_RISK_AWARE_OBJECTIVE_HANDOFF_REQUESTS_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(risk_aware_objective_requests_path)
    _validate_source_payload(payload)
    requests = [
        dict(row)
        for row in payload.get("risk_aware_objective_handoff_requests", []) or []
        if isinstance(row, Mapping)
    ]
    valid_requests = [
        request for request in requests if _is_valid_risk_aware_objective_request(request)
    ]
    rejected_requests = [
        {
            "request_id": str(request.get("request_id", "")),
            "reason": "invalid_risk_aware_objective_request_contract",
            "support": 0,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "truth_status": M2_TRUTH_STATUS,
        }
        for request in requests
        if not _is_valid_risk_aware_objective_request(request)
    ]
    hypotheses = [
        hypothesis
        for request in valid_requests
        for hypothesis in generate_risk_aware_objective_hypotheses_for_request(request)
    ]
    invalid = [
        {
            "hypothesis_id": hypothesis.hypothesis_id,
            "errors": list(validate_hypothesis(hypothesis).errors),
        }
        for hypothesis in hypotheses
        if not validate_hypothesis(hypothesis).valid
    ]
    if invalid:
        raise ValueError(
            f"invalid risk-aware objective hypotheses generated: {invalid}"
        )
    return build_risk_aware_objective_hypothesis_payload(
        source_path=str(risk_aware_objective_requests_path),
        source_payload=payload,
        requests=valid_requests,
        rejected_requests=rejected_requests,
        hypotheses=hypotheses,
    )


def generate_risk_aware_objective_hypotheses_for_request(
    request: Mapping[str, Any],
) -> tuple[FrontierConditionedHypothesis, ...]:
    request_id = str(request.get("request_id", ""))
    game_id = str(request.get("game_id", ""))
    source_frontier_id = str(request.get("source_frontier_id", ""))
    matrix = dict(request.get("suggested_initial_experiment_matrix", {}) or {})
    base_state_family = str(
        matrix.get(
            "base_state_family",
            "risk_aware_terminal_safe_post_stop_conversion_state",
        )
    )
    return tuple(
        _make_risk_aware_objective_hypothesis(
            request=request,
            index=index,
            template=template,
            base_state_family=base_state_family,
            game_id=game_id,
            request_id=request_id,
            source_frontier_id=source_frontier_id,
        )
        for index, template in enumerate(_hypothesis_templates(), start=1)
    )


def build_risk_aware_objective_hypothesis_payload(
    *,
    source_path: str,
    source_payload: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    rejected_requests: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[FrontierConditionedHypothesis],
) -> Dict[str, Any]:
    by_request: dict[str, list[FrontierConditionedHypothesis]] = {}
    for hypothesis in hypotheses:
        by_request.setdefault(hypothesis.source_request_id, []).append(hypothesis)

    batches = []
    for request in requests:
        request_id = str(request.get("request_id", ""))
        candidates = by_request.get(request_id, [])
        batches.append(
            {
                "source_request_id": request_id,
                "source_frontier_id": str(request.get("source_frontier_id", "")),
                "frontier_type": str(request.get("frontier_type", "")),
                "frontier_reason": str(request.get("frontier_reason", "")),
                "blocked_capability": str(request.get("blocked_capability", "")),
                "handoff_type": str(request.get("handoff_type", "")),
                "target": str(request.get("target", "")),
                "target_modules": list(request.get("target_modules", []) or []),
                "candidate_hypotheses": [
                    _risk_aware_objective_hypothesis_dict(hypothesis, request)
                    for hypothesis in candidates
                ],
            }
        )

    testable = [
        hypothesis for hypothesis in hypotheses if hypothesis.testability.testable
    ]
    invalid = [
        hypothesis
        for hypothesis in hypotheses
        if not validate_hypothesis(hypothesis).valid
    ]
    requested_families = {
        str(family)
        for request in requests
        for family in request.get("requested_hypothesis_families", []) or []
    }
    generated_families = {hypothesis.hypothesis_family for hypothesis in hypotheses}
    requested_styles = {
        str(style)
        for request in requests
        for style in request.get("requested_experiment_styles", []) or []
    }
    generated_styles = {
        _requested_style_for_hypothesis(hypothesis)
        for hypothesis in hypotheses
        if _requested_style_for_hypothesis(hypothesis)
    }
    return {
        "config": {
            "source_risk_aware_objective_requests_path": source_path,
            "schema_version": RISK_AWARE_OBJECTIVE_SCHEMA_VERSION,
            "base_m2_schema_version": M2_SCHEMA_VERSION,
            "generator_mode": "risk_aware_objective_completion_heuristic_only",
            "inputs_read": ["P2.G5"],
            "contextual_inputs_not_required": [
                "diagnostics/p2/risk_aware_post_stop_frontier_records.json",
                "diagnostics/p3/risk_targeted_contextual_post_stop_policy_validation.json",
                "diagnostics/p3/contextual_post_stop_conversion_policy_adapter.json",
            ],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["P2", "M3", "A32", "A33"],
            "execution_performed": False,
            "policy_rollout_performed": False,
            "environment_step_performed": False,
        },
        "risk_aware_objective_hypothesis_batches": batches,
        "rejected_risk_aware_objective_requests": [
            dict(row) for row in rejected_requests
        ],
        "summary": {
            "risk_aware_objective_requests_seen": len(
                source_payload.get("risk_aware_objective_handoff_requests", []) or []
            ),
            "risk_aware_objective_requests_consumed": len(requests),
            "risk_aware_objective_requests_rejected": len(rejected_requests),
            "hypothesis_batches": len(batches),
            "hypotheses_generated": len(hypotheses),
            "testable_hypotheses": len(testable),
            "ready_for_m3_g5_candidate_experiment_request": len(testable),
            "blocked_not_testable_hypotheses": len(hypotheses) - len(testable),
            "requested_hypothesis_families": sorted(requested_families),
            "generated_hypothesis_families": sorted(generated_families),
            "all_requested_hypothesis_families_covered": (
                requested_families <= generated_families
            ),
            "requested_experiment_styles": sorted(requested_styles),
            "generated_requested_experiment_styles": sorted(generated_styles),
            "all_hypotheses_have_falsification_signal": all(
                bool(_falsification_signal_for_hypothesis(hypothesis))
                for hypothesis in hypotheses
            ),
            "all_hypotheses_map_to_requested_experiment_style": (
                bool(hypotheses) and generated_styles <= requested_styles
            ),
            "action6_extension_retest_hypotheses_generated": False,
            "substrate_actions_not_target_hypotheses": list(
                SUBSTRATE_ACTIONS_NOT_TARGETS
            ),
            "execution_performed": False,
            "policy_rollout_performed": False,
            "environment_step_performed": False,
            "final_invalid_hypotheses": len(invalid),
            "m3_write_performed": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
            "support": 0,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "status": M2_HYPOTHESIS_STATUS,
        "support": 0,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "truth_status": M2_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "risk_aware_objective_hypotheses_counted_as_confirmation": False,
        "request_counted_as_scientific_verdict": False,
        "policy_result_counted_as_scientific_verdict": False,
        "execution_performed": False,
        "policy_rollout_performed": False,
        "environment_step_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _make_risk_aware_objective_hypothesis(
    *,
    request: Mapping[str, Any],
    index: int,
    template: Mapping[str, Any],
    base_state_family: str,
    game_id: str,
    request_id: str,
    source_frontier_id: str,
) -> FrontierConditionedHypothesis:
    metric = str(template["metric"])
    target_action = str(template["target_action"])
    controls = tuple(str(value) for value in template.get("control_actions", ()) or ())
    falsification = FalsificationCriterion(
        metric=metric,
        support_condition=str(template["support_condition"]),
        failure_condition=str(template["failure_condition"]),
        minimum_effect_size=template.get("minimum_effect_size", 1),
    )
    if not is_metric_measurable(metric):
        falsification = default_falsification_for_metric(metric)
    testability = HypothesisTestability(
        testable=True,
        recommended_test_type=str(template["requested_experiment_style"]),
        target_action=target_action,
        suggested_control_actions=controls,
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=metric,
        required_context_replay=tuple(
            str(value) for value in template.get("required_context_replay", ()) or ()
        ),
        required_action_args=None,
        expected_signal_type=str(template["expected_signal"]),
        measurable_by_existing_extractor=is_metric_measurable(metric),
        blocking_reason=None,
    )
    context_snapshot = ContextSnapshot(
        replay_actions=(),
        replay_action_args=None,
        frame_before_hash=None,
        live_state_signature=(
            f"risk_aware_objective_completion:{base_state_family}:"
            f"{template['family']}:{request.get('candidate_problem_statement', '')}"
        ),
        available_actions=tuple(str(action) for action in _commit_action_candidates(request)),
        local_patch=None,
        terminal_state=False,
    )
    audit = SourceGenerationAudit(
        sources=("heuristic",),
        raw_proposal_ids=(
            f"raw::risk_aware_objective_completion::{request_id}::{index:03d}",
        ),
        rationales=(
            "P2.G5 request is a candidate-only frontier handoff, not evidence.",
            str(template["why_prompted_by_frontier"]),
        ),
        normalization_warnings=(
            "risk_aware_objective_request_not_counted_as_support",
            "policy_result_not_counted_as_scientific_verdict",
            "action6_extension_sequences_are_substrate_not_target_hypotheses",
        ),
        priority_score=float(template.get("priority", 1.0)),
        priority_score_counted_as_support=False,
    )
    return FrontierConditionedHypothesis(
        hypothesis_id=(
            f"m2g2::{game_id or 'unknown_game'}::"
            f"risk_aware_objective_completion::H{index:03d}"
        ),
        source_request_id=request_id,
        game_id=game_id,
        frontier_context_id=source_frontier_id
        or "risk_aware_objective_completion::post_stop",
        frontier_reason=RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION,
        frontier_step=None,
        hypothesis_family=str(template["family"]),
        candidate_action=target_action,
        predicted_metric=metric,
        predicted_effect=str(template["expected_signal"]),
        rationale=(
            f"{template['claim']} Falsification signal: "
            f"{template['falsification_signal']}. Requested experiment style: "
            f"{template['requested_experiment_style']}. This is an M2.G2 "
            "candidate hypothesis about completion, not a policy decision."
        ),
        testability=testability,
        falsification=falsification,
        context_snapshot=context_snapshot,
        source_generation=audit,
        status=M2_HYPOTHESIS_STATUS,
        support=0,
        controlled_test_required=True,
        truth_status=M2_TRUTH_STATUS,
        revision_performed=False,
        wrong_confirmations=0,
        trace_support_counted_as_proof=False,
        prior_counted_as_proof=False,
    )


def _risk_aware_objective_hypothesis_dict(
    hypothesis: FrontierConditionedHypothesis,
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    row = hypothesis.to_dict()
    template = _template_by_index(_hypothesis_index(hypothesis.hypothesis_id))
    row["revision_status"] = CANDIDATE_REVISION_STATUS
    row["family"] = template["family"]
    row["claim"] = template["claim"]
    row["why_prompted_by_frontier"] = template["why_prompted_by_frontier"]
    row["falsification_signal"] = template["falsification_signal"]
    row["suggested_experiment_style"] = template["requested_experiment_style"]
    row["requested_experiment_style"] = template["requested_experiment_style"]
    row["required_observables"] = list(template["required_observables"])
    row["forbidden_interpretations"] = list(template["forbidden_interpretations"])
    row["substrate_actions_not_target_hypotheses"] = list(
        SUBSTRATE_ACTIONS_NOT_TARGETS
    )
    row["base_state_family"] = str(
        dict(request.get("suggested_initial_experiment_matrix", {}) or {}).get(
            "base_state_family",
            "risk_aware_terminal_safe_post_stop_conversion_state",
        )
    )
    row["ready_for_m3_g5"] = bool(hypothesis.testability.testable)
    row["ready_for_m3_g5_candidate_experiment_request"] = bool(
        hypothesis.testability.testable
    )
    row["ready_for_a32"] = False
    row["ready_for_a33"] = False
    row["risk_aware_objective_hypothesis_counted_as_confirmation"] = False
    row["request_counted_as_scientific_verdict"] = False
    row["policy_result_counted_as_scientific_verdict"] = False
    return row


def _hypothesis_templates() -> tuple[Dict[str, Any], ...]:
    common_forbidden = (
        "Do not interpret policy utility as objective completion.",
        "Do not increment support from this generated hypothesis.",
        "Do not treat ACTION6,ACTION3 or ACTION6,ACTION4 as newly confirmed mechanics.",
    )
    return (
        {
            "family": "objective_readiness_detection",
            "claim": "H_readiness_relation_saturation: objective completion becomes possible when new relation states stop increasing despite high safe progress.",
            "target_action": "READINESS_RELATION_SATURATION_DETECTOR",
            "control_actions": ("no_readiness_filter", "hold_or_stop_state"),
            "metric": "levels_completed_after_rollout",
            "expected_signal": "plateau-gated states complete more often than equally high proxy states without plateau",
            "falsification_signal": "plateau-gated states do not improve completion over non-plateau controls",
            "requested_experiment_style": "post_selector_objective_readiness_probe",
            "support_condition": "plateau_gated_levels_completed > non_plateau_levels_completed",
            "failure_condition": "plateau_gated_levels_completed <= non_plateau_levels_completed",
            "required_observables": (
                "new_relation_states",
                "relation_delta_after_stop",
                "hold_baseline_terminal_adjusted_progress",
                "objective_completion_signal",
            ),
            "why_prompted_by_frontier": "P3.G4 shows safe proxy progress without completion, so a readiness discriminator may require relation saturation rather than more extension.",
            "priority": 5.0,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "objective_readiness_detection",
            "claim": "H_readiness_horizon_window: there is a terminal-horizon window where the policy should stop extending and prepare completion.",
            "target_action": "HORIZON_READINESS_WINDOW_DETECTOR",
            "control_actions": ("horizon_agnostic_readiness", "ACTION6_only"),
            "metric": "terminal_state_after_rollout",
            "expected_signal": "horizon-window gating preserves terminal safety while improving completion attempts",
            "falsification_signal": "horizon-window gating neither reduces terminal re-entry nor improves completion readiness",
            "requested_experiment_style": "horizon_conditioned_completion_trigger_search",
            "support_condition": "horizon_window_terminal_rate < horizon_agnostic_terminal_rate and completion_attempt_signal > control",
            "failure_condition": "horizon_window_terminal_rate >= horizon_agnostic_terminal_rate or no_completion_attempt_gain",
            "required_observables": (
                "terminal_horizon_remaining",
                "terminal_horizon_band",
                "terminal_reentry_rate",
                "objective_completion_signal",
            ),
            "why_prompted_by_frontier": "P3.G4 risk reappears in hold_high and horizon_mid contexts, suggesting readiness is horizon-conditioned.",
            "priority": 4.9,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "objective_readiness_detection",
            "claim": "H_readiness_state_geometry: post-ACTION6 spatial geometry marks a pre-completion zone distinct from generic progress.",
            "target_action": "STATE_GEOMETRY_READINESS_DETECTOR",
            "control_actions": ("changed_pixels_greedy", "relation_delta_greedy"),
            "metric": "topology_before_after",
            "expected_signal": "specific global geometry predicts completion-ready states better than raw proxy progress",
            "falsification_signal": "geometry features fail to separate completion-ready states from safe-progress controls",
            "requested_experiment_style": "post_selector_objective_readiness_probe",
            "support_condition": "geometry_readiness_precision > proxy_progress_readiness_precision",
            "failure_condition": "geometry_readiness_precision <= proxy_progress_readiness_precision",
            "required_observables": (
                "global_configuration_signature",
                "actor_target_geometry",
                "changed_pixels",
                "objective_completion_signal",
            ),
            "why_prompted_by_frontier": "The selector sees progress but not completion, so readiness may be encoded in state geometry rather than scalar score.",
            "priority": 4.8,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "post_conversion_commit_action_search",
            "claim": "H_commit_after_safe_extension: after safe conversion, a distinct commit action may trigger completion.",
            "target_action": "ACTION1",
            "control_actions": ("ACTION2", "ACTION5", "hold_or_stop_state"),
            "metric": "levels_completed_after_rollout",
            "expected_signal": "commit action after safe conversion improves completed levels over selector-only continuation",
            "falsification_signal": "commit action does not beat selector-only or re-enters terminal",
            "requested_experiment_style": "post_conversion_commit_action_matrix",
            "support_condition": "post_conversion_ACTION1_levels_completed > selector_only_levels_completed",
            "failure_condition": "post_conversion_ACTION1_levels_completed <= selector_only_levels_completed or ACTION1_terminal_reentry",
            "required_observables": (
                "safe_conversion_state",
                "commit_action",
                "levels_completed_after_rollout",
                "terminal_reentry_rate",
            ),
            "why_prompted_by_frontier": "P3.G4 policy options stop at hold/ACTION6/extensions and never test a separate commit branch.",
            "priority": 4.7,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "post_conversion_commit_action_search",
            "claim": "H_commit_requires_no_more_relation_progress: commit helps only after relation progress stalls.",
            "target_action": "ACTION2",
            "control_actions": ("ACTION1", "ACTION5", "continue_extension"),
            "metric": "objective_progress_proxy",
            "expected_signal": "commit after relation plateau produces more objective signal than commit before plateau",
            "falsification_signal": "commit timing is unrelated to plateau or does not improve objective signal",
            "requested_experiment_style": "post_conversion_commit_action_matrix",
            "support_condition": "plateau_then_commit_objective_signal > immediate_commit_objective_signal",
            "failure_condition": "plateau_then_commit_objective_signal <= immediate_commit_objective_signal",
            "required_observables": (
                "relation_delta_after_stop",
                "new_relation_states",
                "commit_action",
                "objective_completion_signal",
            ),
            "why_prompted_by_frontier": "The frontier points to proxy progress continuing without completion; commit may require a no-more-progress condition.",
            "priority": 4.6,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "post_conversion_commit_action_search",
            "claim": "H_commit_is_horizon_sensitive: commit becomes necessary before the mid/near terminal horizon risk zone.",
            "target_action": "ACTION5",
            "control_actions": ("ACTION1", "ACTION2", "ACTION6_only"),
            "metric": "terminal_state_after_rollout",
            "expected_signal": "horizon-sensitive commit lowers terminal re-entry while preserving completion attempts",
            "falsification_signal": "horizon-sensitive commit does not reduce terminal re-entry or remains passive",
            "requested_experiment_style": "horizon_conditioned_completion_trigger_search",
            "support_condition": "horizon_commit_terminal_rate < static_extension_terminal_rate",
            "failure_condition": "horizon_commit_terminal_rate >= static_extension_terminal_rate or no_completion_attempt_gain",
            "required_observables": (
                "terminal_horizon_band",
                "hold_baseline_band",
                "commit_action",
                "terminal_reentry_rate",
            ),
            "why_prompted_by_frontier": "The selector avoided terminal extension risk in mid-horizon contexts but did not add a completion step.",
            "priority": 4.5,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "goal_state_representation_beyond_safe_progress",
            "claim": "H_goal_global_pattern_alignment: the objective is a global grid configuration, not additional relation progress.",
            "target_action": "GLOBAL_PATTERN_ALIGNMENT_METRIC",
            "control_actions": ("terminal_adjusted_progress_metric",),
            "metric": "topology_before_after",
            "expected_signal": "global pattern alignment predicts completion better than terminal-adjusted progress",
            "falsification_signal": "global pattern alignment does not outperform terminal-adjusted progress",
            "requested_experiment_style": "terminal_safe_progress_vs_completion_discriminator",
            "support_condition": "global_pattern_completion_prediction > proxy_progress_completion_prediction",
            "failure_condition": "global_pattern_completion_prediction <= proxy_progress_completion_prediction",
            "required_observables": (
                "global_configuration_signature",
                "topology_before_after",
                "objective_completion_signal",
            ),
            "why_prompted_by_frontier": "Safe progress and risk-aware utility are present, but levels remain incomplete.",
            "priority": 4.4,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "goal_state_representation_beyond_safe_progress",
            "claim": "H_goal_actor_target_contact: completion depends on a specific actor-target contact relation rather than total relation count.",
            "target_action": "ACTOR_TARGET_CONTACT_METRIC",
            "control_actions": ("relation_delta_count_metric",),
            "metric": "contact_graph_before_after",
            "expected_signal": "specific actor-target contact predicts completion better than aggregate relation delta",
            "falsification_signal": "specific contact relation does not separate completion states from proxy states",
            "requested_experiment_style": "terminal_safe_progress_vs_completion_discriminator",
            "support_condition": "actor_target_contact_completion_prediction > relation_delta_completion_prediction",
            "failure_condition": "actor_target_contact_completion_prediction <= relation_delta_completion_prediction",
            "required_observables": (
                "actor_target_contact_graph",
                "relation_delta_after_stop",
                "levels_completed_after_rollout",
            ),
            "why_prompted_by_frontier": "The current policy counts useful relation changes but may miss the exact contact needed by the task.",
            "priority": 4.3,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "goal_state_representation_beyond_safe_progress",
            "claim": "H_goal_region_or_boundary_condition: completion is tied to a region, boundary, or terrain condition rather than proximity progress.",
            "target_action": "REGION_BOUNDARY_GOAL_METRIC",
            "control_actions": ("distance_decreases_metric",),
            "metric": "object_shape_zone_before_after",
            "expected_signal": "region or boundary membership predicts objective completion better than proximity progress",
            "falsification_signal": "region or boundary features do not outperform proximity progress",
            "requested_experiment_style": "terminal_safe_progress_vs_completion_discriminator",
            "support_condition": "region_boundary_completion_prediction > proximity_completion_prediction",
            "failure_condition": "region_boundary_completion_prediction <= proximity_completion_prediction",
            "required_observables": (
                "object_shape_zone_before_after",
                "boundary_contact",
                "terrain_region_signature",
                "objective_completion_signal",
            ),
            "why_prompted_by_frontier": "The selector is safe around local progress, but completion may depend on a global zone condition.",
            "priority": 4.2,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "proxy_progress_vs_completion_discriminator",
            "claim": "H_proxy_completion_divergence_high_hold: high-hold states can score well on proxy progress while being poor or risky completion states.",
            "target_action": "HIGH_HOLD_PROXY_COMPLETION_DISCRIMINATOR",
            "control_actions": ("low_hold_completion_discriminator",),
            "metric": "levels_completed_after_rollout",
            "expected_signal": "high-hold proxy states underperform completion-specific states despite higher terminal-adjusted progress",
            "falsification_signal": "high-hold proxy states complete as well as completion-specific states",
            "requested_experiment_style": "terminal_safe_progress_vs_completion_discriminator",
            "support_condition": "completion_specific_levels_completed > high_hold_proxy_levels_completed",
            "failure_condition": "completion_specific_levels_completed <= high_hold_proxy_levels_completed",
            "required_observables": (
                "hold_baseline_terminal_adjusted_progress",
                "hold_baseline_band",
                "levels_completed_after_rollout",
            ),
            "why_prompted_by_frontier": "P3.G4 found the risky OOS region in high-hold/mid-horizon contexts.",
            "priority": 4.1,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "proxy_progress_vs_completion_discriminator",
            "claim": "H_changed_pixels_false_friend: many changed pixels after extension may not indicate objective readiness.",
            "target_action": "CHANGED_PIXELS_COMPLETION_DISCRIMINATOR",
            "control_actions": ("objective_readiness_detector",),
            "metric": "changed_pixels",
            "expected_signal": "changed-pixel magnitude diverges from completion readiness",
            "falsification_signal": "changed-pixel magnitude predicts completion as well as readiness features",
            "requested_experiment_style": "terminal_safe_progress_vs_completion_discriminator",
            "support_condition": "changed_pixels_completion_correlation < readiness_completion_correlation",
            "failure_condition": "changed_pixels_completion_correlation >= readiness_completion_correlation",
            "required_observables": (
                "changed_pixels",
                "objective_completion_signal",
                "global_configuration_signature",
            ),
            "why_prompted_by_frontier": "Changed pixels can help locate activity, but P3.G4 shows activity is not enough for completion.",
            "priority": 4.0,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "proxy_progress_vs_completion_discriminator",
            "claim": "H_relation_delta_false_friend: high relation delta can continue without ever triggering completion.",
            "target_action": "RELATION_DELTA_COMPLETION_DISCRIMINATOR",
            "control_actions": ("global_goal_metric",),
            "metric": "objective_progress_proxy",
            "expected_signal": "relation-delta progress diverges from completion probability",
            "falsification_signal": "relation-delta progress predicts completion as well as global goal features",
            "requested_experiment_style": "terminal_safe_progress_vs_completion_discriminator",
            "support_condition": "relation_delta_completion_correlation < global_goal_completion_correlation",
            "failure_condition": "relation_delta_completion_correlation >= global_goal_completion_correlation",
            "required_observables": (
                "relation_delta_after_stop",
                "new_relation_states",
                "objective_completion_signal",
            ),
            "why_prompted_by_frontier": "The safe selector optimizes terminal-adjusted progress but still completes zero levels.",
            "priority": 3.9,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "risk_aware_selector_completion_gap",
            "claim": "H_selector_missing_commit_branch: the selector is safe because it chooses hold/ACTION6/extensions, but it lacks a commit branch.",
            "target_action": "TWO_STAGE_SELECTOR_WITH_COMMIT_BRANCH",
            "control_actions": ("frozen_contextual_selector",),
            "metric": "levels_completed_after_rollout",
            "expected_signal": "adding a commit branch after safe conversion improves completion without raising terminal rate",
            "falsification_signal": "commit branch does not improve completion or increases terminal re-entry",
            "requested_experiment_style": "risk_aware_policy_ablation_with_completion_metrics",
            "support_condition": "selector_plus_commit_levels_completed > frozen_selector_levels_completed",
            "failure_condition": "selector_plus_commit_levels_completed <= frozen_selector_levels_completed or terminal_rate_increases",
            "required_observables": (
                "selected_option",
                "commit_action",
                "levels_completed_after_rollout",
                "terminal_reentry_rate",
            ),
            "why_prompted_by_frontier": "P3.G4 proves risk-aware selection utility but its action menu contains no explicit completion transition.",
            "priority": 3.8,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "risk_aware_selector_completion_gap",
            "claim": "H_selector_target_wrong_metric: the selector optimizes terminal-adjusted progress instead of completion probability.",
            "target_action": "COMPLETION_PROBABILITY_SELECTOR_TARGET",
            "control_actions": ("terminal_adjusted_progress_selector",),
            "metric": "levels_completed_after_rollout",
            "expected_signal": "completion-targeted selector improves levels completed at matched terminal rate",
            "falsification_signal": "completion-targeted selector fails to improve completion or loses terminal safety",
            "requested_experiment_style": "risk_aware_policy_ablation_with_completion_metrics",
            "support_condition": "completion_targeted_levels_completed > terminal_adjusted_progress_levels_completed and terminal_rate_matched",
            "failure_condition": "completion_targeted_levels_completed <= terminal_adjusted_progress_levels_completed or terminal_rate_higher",
            "required_observables": (
                "selector_score_components",
                "objective_completion_signal",
                "terminal_rate",
            ),
            "why_prompted_by_frontier": "The current selector gains terminal-adjusted progress but objective completion remains zero.",
            "priority": 3.7,
            "forbidden_interpretations": common_forbidden,
        },
        {
            "family": "risk_aware_selector_completion_gap",
            "claim": "H_selector_needs_two_stage_policy: safe conversion and objective commit must be modeled as separate stages.",
            "target_action": "SAFE_CONVERSION_THEN_OBJECTIVE_COMMIT_POLICY",
            "control_actions": ("single_stage_contextual_selector",),
            "metric": "final_game_state",
            "expected_signal": "two-stage policy reaches completion-ready or completed states more often than single-stage selector",
            "falsification_signal": "two-stage policy does not improve completion states or reintroduces terminal risk",
            "requested_experiment_style": "risk_aware_policy_ablation_with_completion_metrics",
            "support_condition": "two_stage_completion_ready_or_complete_rate > single_stage_completion_ready_or_complete_rate",
            "failure_condition": "two_stage_completion_ready_or_complete_rate <= single_stage_completion_ready_or_complete_rate or terminal_rate_higher",
            "required_observables": (
                "stage_transition",
                "safe_conversion_state",
                "completion_ready_state",
                "final_game_state",
            ),
            "why_prompted_by_frontier": "The frontier separates safety/utility from completion, suggesting a staged controller rather than another gate tweak.",
            "priority": 3.6,
            "forbidden_interpretations": common_forbidden,
        },
    )


def _template_by_index(index: int) -> Dict[str, Any]:
    return _hypothesis_templates()[index - 1]


def _hypothesis_index(hypothesis_id: str) -> int:
    marker = "::H"
    if marker not in hypothesis_id:
        return 1
    return int(hypothesis_id.rsplit(marker, 1)[-1])


def _requested_style_for_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> str:
    return str(
        _template_by_index(_hypothesis_index(hypothesis.hypothesis_id))[
            "requested_experiment_style"
        ]
    )


def _falsification_signal_for_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> str:
    return str(
        _template_by_index(_hypothesis_index(hypothesis.hypothesis_id))[
            "falsification_signal"
        ]
    )


def _commit_action_candidates(request: Mapping[str, Any]) -> tuple[str, ...]:
    matrix = dict(request.get("suggested_initial_experiment_matrix", {}) or {})
    actions = matrix.get("post_conversion_commit_action_candidates", []) or []
    if actions:
        return tuple(str(action) for action in actions)
    return ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6")


def _is_valid_risk_aware_objective_request(request: Mapping[str, Any]) -> bool:
    return bool(
        str(request.get("handoff_type", "")) == HANDOFF_TYPE
        and str(request.get("target", "")) == HANDOFF_TARGET
        and list(request.get("target_modules", []) or []) == TARGET_MODULES
        and str(request.get("frontier_type", ""))
        == RISK_AWARE_POST_STOP_NO_OBJECTIVE_COMPLETION_FRONTIER
        and str(request.get("frontier_reason", ""))
        == RISK_AWARE_UTILITY_WITHOUT_OBJECTIVE_COMPLETION
        and str(request.get("blocked_capability", ""))
        == OBJECTIVE_COMPLETION_AFTER_RISK_AWARE_SAFE_CONVERSION
        and bool(request.get("risk_aware_objective_review_accepted", False))
        and bool(
            request.get("ready_for_risk_aware_objective_hypothesis_generation", False)
        )
        and bool(request.get("ready_for_m2_or_m3_risk_aware_objective_branch", False))
        and not bool(request.get("ready_for_direct_downstream_write", False))
        and not bool(request.get("a33_ready", False))
        and list(request.get("blocked_capability_hypotheses", []) or [])
        and list(request.get("requested_hypothesis_families", []) or [])
        and list(request.get("requested_experiment_styles", []) or [])
        and list(request.get("scientific_questions", []) or [])
        and dict(request.get("suggested_initial_experiment_matrix", {}) or {})
        and int(request.get("support", 0) or 0) == 0
        and str(request.get("revision_status", "")) == CANDIDATE_REVISION_STATUS
        and str(request.get("truth_status", "")) == "NOT_EVALUATED_BY_P2"
        and not bool(request.get("revision_performed", False))
        and int(request.get("wrong_confirmations", 0) or 0) == 0
        and not bool(request.get("handoff_request_counted_as_confirmation", False))
        and not bool(request.get("policy_result_counted_as_scientific_verdict", False))
    )


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(payload.get("revision_performed", False)) or bool(
        summary.get("revision_performed", False)
    ):
        raise ValueError("source revision_performed must be false")
    if int(payload.get("wrong_confirmations", summary.get("wrong_confirmations", 0)) or 0) != 0:
        raise ValueError("source wrong_confirmations must remain 0")
    if bool(payload.get("a33_ready", False)) or bool(summary.get("a33_ready", False)):
        raise ValueError("risk-aware objective source cannot be A33-ready")
    if bool(summary.get("ready_for_direct_downstream_write", False)) or bool(
        payload.get("ready_for_direct_downstream_write", False)
    ):
        raise ValueError("risk-aware objective source cannot request direct write")
    for key in (
        "a40_write_performed",
        "m2_write_performed",
        "m3_write_performed",
        "a32_write_performed",
        "a33_write_performed",
    ):
        if bool(summary.get(key, False)) or bool(payload.get(key, False)):
            raise ValueError(f"source must not have {key}")
    if bool(payload.get("handoff_request_counted_as_confirmation", False)):
        raise ValueError("risk-aware objective handoff cannot be confirmation")
    if bool(payload.get("frontier_validation_counted_as_confirmation", False)):
        raise ValueError("risk-aware objective validation cannot be confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be scientific verdict")


def write_risk_aware_objective_completion_hypotheses(
    payload: Mapping[str, Any],
    output_path: str
    | Path = DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M2.G2 risk-aware objective-completion hypotheses.",
    )
    parser.add_argument(
        "--risk-aware-objective-requests",
        type=Path,
        default=DEFAULT_P2_RISK_AWARE_OBJECTIVE_HANDOFF_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_M2_RISK_AWARE_OBJECTIVE_HYPOTHESES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_risk_aware_objective_completion_hypothesis_generator(
        risk_aware_objective_requests_path=args.risk_aware_objective_requests,
    )
    write_risk_aware_objective_completion_hypotheses(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": CANDIDATE_REVISION_STATUS,
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
