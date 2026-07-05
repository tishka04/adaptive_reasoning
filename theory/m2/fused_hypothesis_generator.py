"""M2.15 - fused LLM + ARC-LeWM hypothesis generator.

M2.15 consumes candidate-only LLM proposals, ARC-LeWM latent signals and M3
grounding context. It produces new falsifiable M2 hypotheses and an M3 handoff.
It never produces policy actions, evidence, support or verdicts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .m3_handoff import run_m3_handoff_validation, write_m3_handoff_validation
from .normalizer import default_controls_for_action
from .schema import (
    ContextSnapshot,
    FalsificationCriterion,
    FrontierConditionedHypothesis,
    HypothesisTestability,
    M2_BLOCKED_FOR_M3_STATUS,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_HYPOTHESIS_STATUS,
    M2_READY_FOR_M3_STATUS,
    M2_SCHEMA_VERSION,
    M2_TO_M3_SCHEMA_VERSION,
    M2_TRUTH_STATUS,
    M3CandidateExperimentRequest,
    SourceGenerationAudit,
)
from .testability_compiler import write_m3_requests_payload
from .validators import validate_hypothesis, validate_m3_request


DEFAULT_LLM_HYPOTHESES_PATH = (
    Path("diagnostics") / "m2" / "state_conditioned_llm_hypotheses.json"
)
DEFAULT_LLM_M3_REQUESTS_PATH = (
    Path("diagnostics") / "m2" / "state_conditioned_llm_m3_candidate_requests.json"
)
DEFAULT_WM_INVARIANT_PACKET_PATH = (
    Path("diagnostics") / "m2" / "object_world_model_invariant_packet.json"
)
DEFAULT_ARC_LEWM_SIGNAL_REPORT_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_signal_report.json"
)
DEFAULT_ARC_LEWM_HYPOTHESES_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_hypotheses.json"
)
DEFAULT_ARC_LEWM_M3_REQUESTS_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_m3_candidate_requests_v2.json"
)
DEFAULT_M3_ARC_LEWM_SINGLE_RESULT_PATH = (
    Path("diagnostics") / "m3" / "arc_lewm_m2_candidate_experiment_results.json"
)
DEFAULT_M3_ARC_LEWM_REPLICATION_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "arc_lewm_terminal_risk_replication_results.json"
)
DEFAULT_M3G6_RESULTS_PATH = (
    Path("diagnostics") / "m3" / "risk_aware_objective_completion_experiment_results.json"
)

DEFAULT_FUSED_INPUT_PACKET_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "fused_llm_wm_input_packet.json"
)
DEFAULT_FUSED_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "fused_llm_wm_hypotheses.json"
)
DEFAULT_FUSED_M3_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "fused_llm_wm_m3_candidate_requests.json"
)
DEFAULT_FUSED_HANDOFF_VALIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "fused_llm_wm_handoff_validation.json"
)

FUSED_INPUT_PACKET_SCHEMA_VERSION = "m2.fused_llm_wm_input_packet.v1"
FUSED_HYPOTHESES_SCHEMA_VERSION = "m2.fused_llm_wm_hypotheses.v1"
FUSED_M3_SCHEMA_VERSION = "m2.fused_llm_wm_to_m3.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
OFFLINE_TRACE_REPLAYABILITY = "OFFLINE_TRACE_CONTEXT_ONLY"
OFFLINE_TRACE_CONTEXT_STATE_ORIGIN = "human_trace_frame_before"

FUSED_HYPOTHESIS_FAMILIES = (
    "terminal_risk_precondition",
    "terminal_safe_alternative_action",
    "wm_llm_disagreement_frontier",
    "objective_completion_vs_terminal_risk_tradeoff",
)
FORBIDDEN_FAMILIES = (
    "general_terminal_predictor",
    "universal_action_safety_model",
    "confirmed_goal_representation",
)


def run_fused_hypothesis_generation(
    *,
    llm_hypotheses_path: str | Path = DEFAULT_LLM_HYPOTHESES_PATH,
    llm_m3_requests_path: str | Path = DEFAULT_LLM_M3_REQUESTS_PATH,
    wm_invariant_packet_path: str | Path = DEFAULT_WM_INVARIANT_PACKET_PATH,
    arc_lewm_signal_report_path: str | Path = DEFAULT_ARC_LEWM_SIGNAL_REPORT_PATH,
    arc_lewm_hypotheses_path: str | Path = DEFAULT_ARC_LEWM_HYPOTHESES_PATH,
    arc_lewm_m3_requests_path: str | Path = DEFAULT_ARC_LEWM_M3_REQUESTS_PATH,
    m3_arc_lewm_single_result_path: str | Path = DEFAULT_M3_ARC_LEWM_SINGLE_RESULT_PATH,
    m3_arc_lewm_replication_results_path: str | Path = DEFAULT_M3_ARC_LEWM_REPLICATION_RESULTS_PATH,
    m3g6_results_path: str | Path = DEFAULT_M3G6_RESULTS_PATH,
    input_packet_output_path: str | Path | None = None,
    hypotheses_output_path: str | Path | None = None,
    m3_requests_output_path: str | Path | None = None,
    handoff_validation_output_path: str | Path | None = None,
) -> Dict[str, Any]:
    sources = {
        "llm_hypotheses": _load_json(llm_hypotheses_path),
        "llm_m3_requests": _load_json(llm_m3_requests_path),
        "wm_invariant_packet": _load_json(wm_invariant_packet_path),
        "arc_lewm_signal_report": _load_json(arc_lewm_signal_report_path),
        "arc_lewm_hypotheses": _load_json(arc_lewm_hypotheses_path),
        "arc_lewm_m3_requests": _load_json(arc_lewm_m3_requests_path),
        "m3_arc_lewm_single_result": _load_json(m3_arc_lewm_single_result_path),
        "m3_arc_lewm_replication_results": _load_json(
            m3_arc_lewm_replication_results_path
        ),
        "m3g6_results": _load_json(m3g6_results_path),
    }
    source_paths = {
        "llm_hypotheses": str(llm_hypotheses_path),
        "llm_m3_requests": str(llm_m3_requests_path),
        "wm_invariant_packet": str(wm_invariant_packet_path),
        "arc_lewm_signal_report": str(arc_lewm_signal_report_path),
        "arc_lewm_hypotheses": str(arc_lewm_hypotheses_path),
        "arc_lewm_m3_requests": str(arc_lewm_m3_requests_path),
        "m3_arc_lewm_single_result": str(m3_arc_lewm_single_result_path),
        "m3_arc_lewm_replication_results": str(m3_arc_lewm_replication_results_path),
        "m3g6_results": str(m3g6_results_path),
    }
    packet = build_fused_input_packet(sources, source_paths=source_paths)
    hypotheses = generate_fused_hypotheses(packet)
    validate_fused_hypotheses(hypotheses)

    input_packet_payload = packet
    hypothesis_payload = build_fused_hypothesis_payload(
        packet=packet,
        hypotheses=hypotheses,
    )
    m3_payload = build_fused_m3_requests_payload(
        hypotheses,
        source_hypothesis_path=str(
            hypotheses_output_path or DEFAULT_FUSED_HYPOTHESES_OUTPUT_PATH
        ),
    )

    if input_packet_output_path is not None:
        write_json(input_packet_payload, input_packet_output_path)
    if hypotheses_output_path is not None:
        write_json(hypothesis_payload, hypotheses_output_path)
    if m3_requests_output_path is not None:
        write_m3_requests_payload(m3_payload, m3_requests_output_path)
    handoff_payload: Dict[str, Any] | None = None
    if handoff_validation_output_path is not None:
        if m3_requests_output_path is None:
            raise ValueError("handoff validation requires m3_requests_output_path")
        handoff_payload = run_m3_handoff_validation(
            m3_requests_path=m3_requests_output_path,
            live_available_actions=("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"),
        )
        write_m3_handoff_validation(handoff_payload, handoff_validation_output_path)

    return {
        "input_packet": input_packet_payload,
        "hypotheses_payload": hypothesis_payload,
        "m3_payload": m3_payload,
        "handoff_validation_payload": handoff_payload,
    }


def build_fused_input_packet(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    source_paths = dict(source_paths or {})
    _validate_candidate_only_sources(sources)
    llm_hypotheses = _llm_hypotheses(sources.get("llm_hypotheses", {}))
    terminal_experiments = _terminal_risk_experiments(
        sources.get("m3_arc_lewm_replication_results", {})
    )
    summary = {
        "llm_hypotheses_consumed": len(llm_hypotheses),
        "terminal_risk_experiments_consumed": len(terminal_experiments),
        "terminal_risk_support_events_read": int(
            _summary_value(
                sources.get("m3_arc_lewm_replication_results", {}),
                "support_events",
                default=0,
            )
            or 0
        ),
        "terminal_risk_support_events_counted_as_support": False,
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "llm_output_counted_as_evidence": False,
        "world_model_score_counted_as_support": False,
        "world_model_counted_as_evidence": False,
        "m3_observation_counted_as_confirmation": False,
    }
    return {
        "config": {
            "schema_version": FUSED_INPUT_PACKET_SCHEMA_VERSION,
            "m2_schema_version": M2_SCHEMA_VERSION,
            "source_paths": source_paths,
            "inputs_read": [
                "M2.13a",
                "M2.14",
                "M3.7c",
                "M3.7d",
                "M3.G6",
            ],
            "artifacts_not_modified": ["M2 sources", "M3 sources", "A32", "A33"],
        },
        "scientific_question": (
            "Which new falsifiable hypotheses appear when LLM abductive "
            "proposals are fused with ARC-LeWM terminal-risk signals and M3 "
            "candidate-only grounding context?"
        ),
        "allowed_hypothesis_families": list(FUSED_HYPOTHESIS_FAMILIES),
        "forbidden_hypothesis_families": list(FORBIDDEN_FAMILIES),
        "fusion_roles": {
            "llm": "abductive_hypothesis_generator_not_evidence",
            "world_model": "latent_terminal_risk_context_provider_not_evidence",
            "m3_results": "candidate_only_grounding_context_not_verdict",
            "m2_15": "fusion_normalization_testability_compiler_not_policy",
        },
        "llm_context": {
            "summary": dict((sources.get("llm_hypotheses", {}) or {}).get("summary", {}) or {}),
            "m3_request_summary": dict(
                (sources.get("llm_m3_requests", {}) or {}).get("summary", {}) or {}
            ),
            "candidate_hypotheses": _sample_hypotheses(llm_hypotheses, cap=6),
        },
        "world_model_context": {
            "invariant_packet_summary": dict(
                (sources.get("wm_invariant_packet", {}) or {}).get("summary", {}) or {}
            ),
            "arc_lewm_signal_summary": dict(
                (sources.get("arc_lewm_signal_report", {}) or {}).get("summary", {}) or {}
            ),
            "arc_lewm_hypothesis_summary": dict(
                (sources.get("arc_lewm_hypotheses", {}) or {}).get("summary", {}) or {}
            ),
            "arc_lewm_m3_request_summary": dict(
                (sources.get("arc_lewm_m3_requests", {}) or {}).get("summary", {}) or {}
            ),
            "terminal_transition_count": _terminal_transition_count(
                sources.get("arc_lewm_signal_report", {})
            ),
        },
        "m3_grounding_context": {
            "arc_lewm_single_result_summary": dict(
                (sources.get("m3_arc_lewm_single_result", {}) or {}).get("summary", {}) or {}
            ),
            "terminal_risk_replication_summary": dict(
                (sources.get("m3_arc_lewm_replication_results", {}) or {}).get("summary", {}) or {}
            ),
            "risk_aware_objective_completion_summary": dict(
                (sources.get("m3g6_results", {}) or {}).get("summary", {}) or {}
            ),
            "terminal_risk_observations": [
                _terminal_observation_summary(row)
                for row in terminal_experiments[:14]
            ],
        },
        "candidate_only_contract": _candidate_only_contract(),
        "summary": summary,
        **_candidate_only_contract(),
    }


def generate_fused_hypotheses(
    packet: Mapping[str, Any],
) -> tuple[FrontierConditionedHypothesis, ...]:
    observations = list(
        ((packet.get("m3_grounding_context", {}) or {}).get("terminal_risk_observations", []) or [])
    )
    if not observations:
        return ()
    llm_items = list(((packet.get("llm_context", {}) or {}).get("candidate_hypotheses", []) or []))
    llm_actions = {str(item.get("candidate_action", "")) for item in llm_items}

    first = observations[0]
    disagreement = _first_observation_with_action(observations, llm_actions) or first
    tradeoff = _first_bp35_action6_observation(observations) or first
    hypotheses = [
        _make_fused_hypothesis(
            index=1,
            family="terminal_risk_precondition",
            observation=first,
            target_action=str(first.get("target_action", "")),
            target_action_args=_dict_or_none(first.get("target_action_args")),
            metric="terminal_state_after_rollout",
            expected_signal="terminal_risk_precondition_exceeds_matched_controls",
            predicted_effect=(
                "A specific action/context/action-set precondition marks a "
                "terminal-risk transition selected by ARC-LeWM."
            ),
            support_condition="target_action_terminal_rate > matched_control_terminal_rate",
            failure_condition="target_action_terminal_rate <= matched_control_terminal_rate",
            testable=True,
            source_llm_ids=_llm_ids(llm_items),
        ),
        _make_fused_hypothesis(
            index=2,
            family="terminal_safe_alternative_action",
            observation=first,
            target_action=str(first.get("control_action", "")),
            target_action_args=None,
            metric="terminal_state_after_rollout",
            expected_signal="alternative_action_terminal_rate_below_risk_target",
            predicted_effect=(
                "A matched dynamic control may be a terminal-safe alternative "
                "inside the same risk context while preserving future testability."
            ),
            support_condition="alternative_terminal_rate < risk_target_terminal_rate",
            failure_condition="alternative_terminal_rate >= risk_target_terminal_rate",
            testable=False,
            blocking_reason="BLOCKED_REQUIRES_COUNTERFACTUAL_ENV_REPLAY_FROM_OFFLINE_FRAME",
            source_llm_ids=_llm_ids(llm_items),
        ),
        _make_fused_hypothesis(
            index=3,
            family="wm_llm_disagreement_frontier",
            observation=disagreement,
            target_action=str(disagreement.get("target_action", "")),
            target_action_args=_dict_or_none(disagreement.get("target_action_args")),
            metric="terminal_state_after_rollout",
            expected_signal="llm_useful_action_overlaps_wm_terminal_risk_context",
            predicted_effect=(
                "An action proposed or favored by the LLM may be useful in one "
                "objective context yet terminal-risk under a different WM/M3 "
                "grounded context."
            ),
            support_condition="wm_terminal_risk_signal_reproduces_for_llm_action",
            failure_condition="wm_terminal_risk_signal_does_not_reproduce_for_llm_action",
            testable=True,
            source_llm_ids=_llm_ids(llm_items),
        ),
        _make_fused_hypothesis(
            index=4,
            family="objective_completion_vs_terminal_risk_tradeoff",
            observation=tradeoff,
            target_action=str(tradeoff.get("target_action", "")),
            target_action_args=_dict_or_none(tradeoff.get("target_action_args")),
            metric="terminal_state_after_rollout",
            expected_signal="proxy_completion_candidates_overlap_terminal_risk",
            predicted_effect=(
                "Proxy progress and objective-completion attempts may be "
                "contaminated by actions that ARC-LeWM marks as terminal-risk."
            ),
            support_condition="terminal_risk_rate_exceeds_controls_in_proxy_context",
            failure_condition="terminal_risk_rate_not_above_controls_in_proxy_context",
            testable=True,
            source_llm_ids=_llm_ids(llm_items),
        ),
    ]
    return tuple(hypotheses)


def build_fused_hypothesis_payload(
    *,
    packet: Mapping[str, Any],
    hypotheses: Sequence[FrontierConditionedHypothesis],
) -> Dict[str, Any]:
    ready = [hypothesis for hypothesis in hypotheses if hypothesis.testability.testable]
    blocked = [hypothesis for hypothesis in hypotheses if not hypothesis.testability.testable]
    return {
        "config": {
            "schema_version": FUSED_HYPOTHESES_SCHEMA_VERSION,
            "m2_schema_version": M2_SCHEMA_VERSION,
            "source_input_packet_path": str(DEFAULT_FUSED_INPUT_PACKET_OUTPUT_PATH),
            "generator_mode": "llm_wm_m3_grounding_fusion_candidate_only",
            "inputs_read": list((packet.get("config", {}) or {}).get("inputs_read", []) or []),
            "artifacts_not_modified": ["M2 source artifacts", "M3", "A32", "A33"],
            "environment_step_performed": False,
            "policy_rollout_performed": False,
        },
        "fused_hypothesis_batches": [
            {
                "allowed_hypothesis_families": list(FUSED_HYPOTHESIS_FAMILIES),
                "forbidden_hypothesis_families": list(FORBIDDEN_FAMILIES),
                "candidate_hypotheses": [
                    _fused_hypothesis_dict(hypothesis) for hypothesis in hypotheses
                ],
            }
        ],
        "summary": {
            "input_packet_consumed": True,
            "hypotheses_generated": len(hypotheses),
            "hypothesis_families_covered": sorted(
                {hypothesis.hypothesis_family for hypothesis in hypotheses}
            ),
            "ready_for_m3_candidate_experiment_request": len(ready),
            "blocked_not_testable_hypotheses": len(blocked),
            "blocked_by_reason": dict(
                sorted(
                    Counter(
                        hypothesis.testability.blocking_reason or "none"
                        for hypothesis in blocked
                    ).items()
                )
            ),
            "terminal_risk_support_events_read": int(
                (packet.get("summary", {}) or {}).get(
                    "terminal_risk_support_events_read", 0
                )
                or 0
            ),
            "terminal_risk_support_events_counted_as_support": False,
            "llm_output_counted_as_evidence": False,
            "world_model_score_counted_as_support": False,
            "world_model_counted_as_evidence": False,
            "m3_observation_counted_as_confirmation": False,
            "policy_generated": False,
            "general_terminal_predictor_generated": False,
            "universal_action_safety_model_generated": False,
            "confirmed_goal_representation_generated": False,
            **_candidate_only_contract(),
        },
        "contract": _candidate_only_contract(),
        **_candidate_only_contract(),
    }


def build_fused_m3_requests_payload(
    hypotheses: Sequence[FrontierConditionedHypothesis],
    *,
    source_hypothesis_path: str,
) -> Dict[str, Any]:
    requests = tuple(
        _m3_request_from_fused_hypothesis(hypothesis) for hypothesis in hypotheses
    )
    invalid = [
        {
            "request_id": request.request_id,
            "errors": list(validate_m3_request(request).errors),
        }
        for request in requests
        if not validate_m3_request(request).valid
    ]
    if invalid:
        raise ValueError(f"invalid fused M3 requests generated: {invalid}")
    ready = [request for request in requests if request.status == M2_READY_FOR_M3_STATUS]
    blocked = [
        request for request in requests if request.status == M2_BLOCKED_FOR_M3_STATUS
    ]
    return {
        "config": {
            "source_hypothesis_path": source_hypothesis_path,
            "schema_version": FUSED_M3_SCHEMA_VERSION,
            "base_schema_version": M2_TO_M3_SCHEMA_VERSION,
            "handoff_validator": "fused_llm_wm_candidate_handoff_v1",
        },
        "experiment_requests": [request.to_dict() for request in requests],
        "summary": {
            "source_hypotheses": len(hypotheses),
            "experiment_requests": len(requests),
            "ready_for_m3": len(ready),
            "blocked_not_testable": len(blocked),
            "blocked_by_reason": dict(
                sorted(
                    Counter(
                        request.blocking_reason or "none" for request in blocked
                    ).items()
                )
            ),
            "offline_trace_context_requests": sum(
                1
                for request in ready
                if request.replayability == OFFLINE_TRACE_REPLAYABILITY
            ),
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "support": 0,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "llm_output_counted_as_evidence": False,
            "world_model_score_counted_as_support": False,
            "world_model_counted_as_evidence": False,
            "m3_observation_counted_as_confirmation": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
    }


def validate_fused_hypotheses(
    hypotheses: Sequence[FrontierConditionedHypothesis],
) -> None:
    invalid: list[Dict[str, Any]] = []
    for hypothesis in hypotheses:
        result = validate_hypothesis(hypothesis)
        if not result.valid:
            invalid.append(
                {"hypothesis_id": hypothesis.hypothesis_id, "errors": list(result.errors)}
            )
        if hypothesis.hypothesis_family in set(FORBIDDEN_FAMILIES):
            invalid.append(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "errors": ["forbidden_hypothesis_family"],
                }
            )
        validate_fused_candidate_contract(hypothesis.to_dict())
    if invalid:
        raise ValueError(f"invalid fused hypotheses generated: {invalid}")


def validate_fused_candidate_contract(row: Mapping[str, Any]) -> None:
    if str(row.get("status", "")) != M2_HYPOTHESIS_STATUS:
        raise ValueError("fused hypothesis status must remain UNRESOLVED")
    if int(row.get("support", 0) or 0) != 0:
        raise ValueError("fused hypothesis support must remain 0")
    if str(row.get("truth_status", "")) != M2_TRUTH_STATUS:
        raise ValueError("fused hypothesis truth_status must be NOT_EVALUATED_BY_M2")
    if bool(row.get("revision_performed", False)):
        raise ValueError("fused hypothesis must not revise")
    if int(row.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("fused hypothesis wrong_confirmations must remain 0")
    if bool(row.get("ready_for_a32", False)) or bool(row.get("ready_for_a33", False)):
        raise ValueError("fused hypothesis must not be ready for A32/A33")
    for key in (
        "llm_output_counted_as_evidence",
        "world_model_score_counted_as_support",
        "world_model_counted_as_evidence",
        "m3_observation_counted_as_confirmation",
        "terminal_risk_support_events_counted_as_support",
    ):
        if bool(row.get(key, False)):
            raise ValueError(f"fused hypothesis must not set {key}")


def _make_fused_hypothesis(
    *,
    index: int,
    family: str,
    observation: Mapping[str, Any],
    target_action: str,
    target_action_args: Mapping[str, Any] | None,
    metric: str,
    expected_signal: str,
    predicted_effect: str,
    support_condition: str,
    failure_condition: str,
    testable: bool,
    source_llm_ids: Sequence[str],
    blocking_reason: str | None = None,
) -> FrontierConditionedHypothesis:
    source_transition_id = str(observation.get("source_transition_id", "") or "")
    available_actions = tuple(
        str(action)
        for action in observation.get("dynamic_available_actions", []) or []
        if str(action)
    )
    controls = default_controls_for_action(target_action, available_actions)
    falsification = FalsificationCriterion(
        metric=metric,
        support_condition=support_condition,
        failure_condition=failure_condition,
        minimum_effect_size=1,
    )
    source_ids = tuple(
        item
        for item in (
            f"m2_15::llm::{source_llm_ids[0]}" if source_llm_ids else "",
            f"m2_15::wm::{source_transition_id}",
            str(observation.get("request_id", "")),
        )
        if item
    )
    rationale = (
        "Fused candidate from LLM abductive context plus ARC-LeWM/M3.7d "
        "terminal-risk observations. These observations are grounding context "
        "only and are not counted as support or proof."
    )
    return FrontierConditionedHypothesis(
        hypothesis_id=f"m2_15_fused::{family}::{index:03d}",
        source_request_id="m2_15::llm_wm_grounding_fusion",
        game_id=str(observation.get("game_id", "")),
        frontier_context_id=f"m2_15_fused::{family}",
        frontier_reason="LLM_WM_M3_GROUNDED_CONTEXT_FUSION_CANDIDATE_ONLY",
        frontier_step=_optional_int(observation.get("source_step")),
        hypothesis_family=family,
        candidate_action=target_action,
        predicted_metric=metric,
        predicted_effect=predicted_effect,
        rationale=rationale,
        testability=HypothesisTestability(
            testable=bool(testable and not blocking_reason),
            recommended_test_type="offline_trace_context_controlled_candidate",
            target_action=target_action,
            suggested_control_actions=tuple(control for control in controls if control != target_action),
            control_policy=M2_DYNAMIC_CONTROL_POLICY,
            metric=metric,
            required_context_replay=(),
            required_action_args=((dict(target_action_args),) if target_action_args else None),
            expected_signal_type=expected_signal,
            measurable_by_existing_extractor=True,
            blocking_reason=blocking_reason,
        ),
        falsification=falsification,
        context_snapshot=ContextSnapshot(
            replay_actions=(),
            replay_action_args=None,
            frame_before_hash=source_transition_id or None,
            live_state_signature=f"m2_15_fused:{family}:{source_transition_id}",
            available_actions=available_actions,
            local_patch=None,
            terminal_state=bool(observation.get("target_terminal", True)),
        ),
        source_generation=SourceGenerationAudit(
            sources=("llm", "world_model"),
            raw_proposal_ids=source_ids,
            rationales=(rationale,),
            normalization_warnings=(
                "llm_not_counted_as_evidence",
                "world_model_score_not_counted_as_support",
                "m3_support_events_not_counted_as_support",
            ),
            priority_score=_priority_for_family(family, observation),
            priority_score_counted_as_support=False,
        ),
        status=M2_HYPOTHESIS_STATUS,
        support=0,
        controlled_test_required=True,
        truth_status=M2_TRUTH_STATUS,
        revision_performed=False,
        wrong_confirmations=0,
        trace_support_counted_as_proof=False,
        prior_counted_as_proof=False,
    )


def _m3_request_from_fused_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> M3CandidateExperimentRequest:
    target_args = _target_action_args_from_hypothesis(hypothesis)
    source_transition_id = str(hypothesis.context_snapshot.frame_before_hash or "")
    source_episode_id, source_step = _parse_source_transition_id(source_transition_id)
    context_replay = tuple(hypothesis.testability.required_context_replay)
    blocking_reason = _handoff_blocking_reason(
        hypothesis,
        target_action_args=target_args,
        source_transition_id=source_transition_id,
    )
    status = (
        M2_READY_FOR_M3_STATUS
        if hypothesis.testability.testable and blocking_reason is None
        else M2_BLOCKED_FOR_M3_STATUS
    )
    if status == M2_BLOCKED_FOR_M3_STATUS and blocking_reason is None:
        blocking_reason = hypothesis.testability.blocking_reason or "BLOCKED_NOT_TESTABLE"
    return M3CandidateExperimentRequest(
        request_id=hypothesis.hypothesis_id.replace("m2_15_fused::", "m2m3::m2_15_fused::"),
        source_hypothesis_id=hypothesis.hypothesis_id,
        game_id=hypothesis.game_id,
        context_replay=context_replay,
        context_replay_args=(
            hypothesis.testability.required_action_args if context_replay else None
        ),
        context_snapshot_hash=source_transition_id or None,
        target_action=hypothesis.testability.target_action,
        target_action_args=target_args,
        suggested_control_actions=tuple(hypothesis.testability.suggested_control_actions),
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=hypothesis.testability.metric,
        expected_signal=hypothesis.testability.expected_signal_type,
        falsification_criterion=hypothesis.falsification,
        status=status,
        source_episode_id=source_episode_id,
        source_step=source_step,
        source_transition_id=source_transition_id or None,
        context_state_origin=(
            OFFLINE_TRACE_CONTEXT_STATE_ORIGIN if source_transition_id else None
        ),
        replayability=OFFLINE_TRACE_REPLAYABILITY if source_transition_id else None,
        blocking_reason=blocking_reason,
        truth_status=M2_TRUTH_STATUS,
        support=0,
        controlled_test_required=True,
        revision_performed=False,
        wrong_confirmations=0,
    )


def _fused_hypothesis_dict(hypothesis: FrontierConditionedHypothesis) -> Dict[str, Any]:
    row = hypothesis.to_dict()
    row["revision_status"] = CANDIDATE_REVISION_STATUS
    row["ready_for_m3_candidate_experiment_request"] = bool(hypothesis.testability.testable)
    row["ready_for_a32"] = False
    row["ready_for_a33"] = False
    row["llm_output_counted_as_evidence"] = False
    row["world_model_score_counted_as_support"] = False
    row["world_model_counted_as_evidence"] = False
    row["m3_observation_counted_as_confirmation"] = False
    row["terminal_risk_support_events_counted_as_support"] = False
    row["fused_score_counted_as_support"] = False
    validate_fused_candidate_contract(row)
    return row


def _validate_candidate_only_sources(
    sources: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, payload in sources.items():
        if not payload:
            raise ValueError(f"missing required M2.15 source artifact: {name}")
        summary = dict(payload.get("summary", {}) or {})
        support = int(payload.get("support", summary.get("support", 0)) or 0)
        if support != 0:
            raise ValueError(f"{name} support must remain 0")
        for key in (
            "a32_write_performed",
            "a33_write_performed",
            "revision_performed",
            "world_model_score_counted_as_support",
            "world_model_counted_as_evidence",
            "world_model_prediction_counted_as_evidence",
        ):
            if bool(payload.get(key, summary.get(key, False))):
                raise ValueError(f"{name} must not set {key}")
        status = str(payload.get("status", summary.get("status", "")))
        if status in {"CONFIRMED", "REFUTED"}:
            raise ValueError(f"{name} must not provide a verdict status")


def _candidate_only_contract() -> Dict[str, Any]:
    return {
        "support": 0,
        "truth_status": M2_TRUTH_STATUS,
        "revision_status": CANDIDATE_REVISION_STATUS,
        "controlled_test_required": True,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _llm_hypotheses(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        dict(row)
        for row in payload.get("candidate_hypotheses", []) or []
        if isinstance(row, Mapping)
    ]


def _sample_hypotheses(
    hypotheses: Sequence[Mapping[str, Any]],
    *,
    cap: int,
) -> list[Dict[str, Any]]:
    return [
        {
            "hypothesis_id": str(row.get("hypothesis_id", "")),
            "hypothesis_family": str(row.get("hypothesis_family", "")),
            "candidate_action": str(row.get("candidate_action", "")),
            "predicted_metric": str(row.get("predicted_metric", "")),
            "status": str(row.get("status", "")),
            "support": int(row.get("support", 0) or 0),
            "truth_status": str(row.get("truth_status", "")),
        }
        for row in hypotheses[:cap]
    ]


def _terminal_risk_experiments(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in payload.get("controlled_experiments", []) or []
        if isinstance(row, Mapping)
        and str(row.get("metric", "")) == "terminal_state_after_rollout"
        and str(row.get("execution_mode", "")) == "offline_trace_context"
    ]
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("game_id", "")),
            str(row.get("target_action", "")),
            str(row.get("source_transition_id", "")),
        ),
    )


def _terminal_observation_summary(row: Mapping[str, Any]) -> Dict[str, Any]:
    baseline = dict(row.get("observed_baseline", {}) or {})
    perturbation = dict(row.get("observed_perturbation", {}) or {})
    return {
        "request_id": str(row.get("request_id", "")),
        "game_id": str(row.get("game_id", "")),
        "target_action": str(row.get("target_action", "")),
        "target_action_args": _dict_or_none(row.get("target_action_args")),
        "control_action": str(row.get("control_action", "")),
        "source_transition_id": str(row.get("source_transition_id", "")),
        "source_episode_id": str(row.get("source_episode_id", "")),
        "source_step": _optional_int(row.get("source_step")),
        "dynamic_available_actions": [
            str(action) for action in row.get("dynamic_available_actions", []) or []
        ],
        "matched_control_policy": str(row.get("matched_control_policy", "")),
        "matched_control_samples": int(row.get("matched_control_samples", 0) or 0),
        "target_trace_samples": int(row.get("target_trace_samples", 0) or 0),
        "target_terminal_rate": float(perturbation.get("terminal_rate", 0.0) or 0.0),
        "control_terminal_rate": float(baseline.get("terminal_rate", 0.0) or 0.0),
        "target_terminal": bool(perturbation.get("terminal_state_after_rollout")),
        "support_events": int(row.get("support_events", 0) or 0),
        "contradiction_events": int(row.get("contradiction_events", 0) or 0),
        "support": int(row.get("support", 0) or 0),
        "truth_status": str(row.get("truth_status", "")),
    }


def _terminal_transition_count(signal_report: Mapping[str, Any]) -> int:
    terminal = (
        (signal_report.get("signals", {}) or {}).get(
            "terminal_like_latent_neighborhoods", {}
        )
        or {}
    )
    return int(terminal.get("terminal_transition_count", 0) or 0)


def _first_observation_with_action(
    observations: Sequence[Mapping[str, Any]],
    actions: set[str],
) -> Mapping[str, Any] | None:
    for row in observations:
        if str(row.get("target_action", "")) in actions:
            return row
    for row in observations:
        if str(row.get("target_action", "")) == "ACTION3":
            return row
    return None


def _first_bp35_action6_observation(
    observations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for row in observations:
        if (
            str(row.get("game_id", "")) == "bp35-0a0ad940"
            and str(row.get("target_action", "")) == "ACTION6"
        ):
            return row
    return None


def _llm_ids(llm_items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(item.get("hypothesis_id", "")) for item in llm_items if item.get("hypothesis_id"))


def _handoff_blocking_reason(
    hypothesis: FrontierConditionedHypothesis,
    *,
    target_action_args: Mapping[str, Any] | None,
    source_transition_id: str,
) -> str | None:
    if hypothesis.testability.blocking_reason:
        return hypothesis.testability.blocking_reason
    action = hypothesis.testability.target_action
    available = set(hypothesis.context_snapshot.available_actions)
    if action == "RESET":
        return "BLOCKED_RESET_BOUNDARY"
    if not hypothesis.game_id or hypothesis.game_id == "unknown_game":
        return "BLOCKED_UNKNOWN_GAME"
    if available and action not in available:
        return "BLOCKED_ACTION_NOT_AVAILABLE"
    if action == "ACTION6" and not dict(target_action_args or {}):
        return "BLOCKED_MISSING_ACTION_ARGS"
    if not source_transition_id:
        return "BLOCKED_MISSING_OFFLINE_TRACE_LOCATOR"
    return None


def _target_action_args_from_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> Dict[str, Any] | None:
    args = hypothesis.testability.required_action_args
    if not args:
        return None
    first = args[0]
    return dict(first) if first else None


def _parse_source_transition_id(
    source_transition_id: str,
) -> tuple[str | None, int | None]:
    if not source_transition_id:
        return None, None
    parts = source_transition_id.split("::")
    if len(parts) < 4:
        return None, None
    return "::".join(parts[2:-1]) or None, _optional_int(parts[-1])


def _summary_value(payload: Mapping[str, Any], key: str, *, default: Any) -> Any:
    summary = dict(payload.get("summary", {}) or {})
    return payload.get(key, summary.get(key, default))


def _priority_for_family(family: str, observation: Mapping[str, Any]) -> float:
    base = {
        "terminal_risk_precondition": 4.0,
        "terminal_safe_alternative_action": 3.0,
        "wm_llm_disagreement_frontier": 3.5,
        "objective_completion_vs_terminal_risk_tradeoff": 3.75,
    }.get(family, 1.0)
    samples = int(observation.get("matched_control_samples", 0) or 0)
    return round(base + min(samples / 1000.0, 1.0), 4)


def _dict_or_none(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    return data if data else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M2.15 fused LLM + ARC-LeWM hypotheses.",
    )
    parser.add_argument("--llm-hypotheses", type=Path, default=DEFAULT_LLM_HYPOTHESES_PATH)
    parser.add_argument("--llm-m3-requests", type=Path, default=DEFAULT_LLM_M3_REQUESTS_PATH)
    parser.add_argument("--wm-invariant-packet", type=Path, default=DEFAULT_WM_INVARIANT_PACKET_PATH)
    parser.add_argument("--arc-lewm-signal-report", type=Path, default=DEFAULT_ARC_LEWM_SIGNAL_REPORT_PATH)
    parser.add_argument("--arc-lewm-hypotheses", type=Path, default=DEFAULT_ARC_LEWM_HYPOTHESES_PATH)
    parser.add_argument("--arc-lewm-m3-requests", type=Path, default=DEFAULT_ARC_LEWM_M3_REQUESTS_PATH)
    parser.add_argument("--m3-arc-lewm-single-result", type=Path, default=DEFAULT_M3_ARC_LEWM_SINGLE_RESULT_PATH)
    parser.add_argument("--m3-arc-lewm-replication-results", type=Path, default=DEFAULT_M3_ARC_LEWM_REPLICATION_RESULTS_PATH)
    parser.add_argument("--m3g6-results", type=Path, default=DEFAULT_M3G6_RESULTS_PATH)
    parser.add_argument("--packet-out", type=Path, default=DEFAULT_FUSED_INPUT_PACKET_OUTPUT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_FUSED_HYPOTHESES_OUTPUT_PATH)
    parser.add_argument("--m3-out", type=Path, default=DEFAULT_FUSED_M3_REQUESTS_OUTPUT_PATH)
    parser.add_argument("--handoff-out", type=Path, default=DEFAULT_FUSED_HANDOFF_VALIDATION_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_fused_hypothesis_generation(
        llm_hypotheses_path=args.llm_hypotheses,
        llm_m3_requests_path=args.llm_m3_requests,
        wm_invariant_packet_path=args.wm_invariant_packet,
        arc_lewm_signal_report_path=args.arc_lewm_signal_report,
        arc_lewm_hypotheses_path=args.arc_lewm_hypotheses,
        arc_lewm_m3_requests_path=args.arc_lewm_m3_requests,
        m3_arc_lewm_single_result_path=args.m3_arc_lewm_single_result,
        m3_arc_lewm_replication_results_path=args.m3_arc_lewm_replication_results,
        m3g6_results_path=args.m3g6_results,
        input_packet_output_path=args.packet_out,
        hypotheses_output_path=args.out,
        m3_requests_output_path=args.m3_out,
        handoff_validation_output_path=args.handoff_out,
    )
    print(
        json.dumps(
            {
                "packet_path": str(args.packet_out),
                "hypotheses_path": str(args.out),
                "m3_requests_path": str(args.m3_out),
                "handoff_validation_path": str(args.handoff_out),
                "summary": outputs["hypotheses_payload"]["summary"],
                "m3_summary": outputs["m3_payload"]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
