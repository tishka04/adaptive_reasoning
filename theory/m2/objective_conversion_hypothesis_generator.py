"""M2.G1 objective-conversion hypothesis generation from P2.G3 requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.objective_conversion_handoff_schema import (
    DEFAULT_P2_OBJECTIVE_CONVERSION_HANDOFF_REQUESTS_OUTPUT_PATH,
    HANDOFF_TARGET,
    HANDOFF_TYPE,
    TARGET_MODULES,
)
from theory.p2.objective_conversion_frontier_records import (
    OBJECTIVE_CONVERSION_AFTER_SAFE_STOP,
    OBJECTIVE_CONVERSION_FRONTIER,
    TERMINAL_SAFE_BUT_PASSIVE,
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


DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "objective_conversion_hypotheses.json"
)
OBJECTIVE_CONVERSION_SCHEMA_VERSION = "m2.objective_conversion.v1"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
DEFAULT_AVAILABLE_ACTIONS = ("ACTION3", "ACTION4", "ACTION6")


def run_objective_conversion_hypothesis_generator(
    *,
    objective_conversion_handoff_requests_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_CONVERSION_HANDOFF_REQUESTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(objective_conversion_handoff_requests_path)
    _validate_source_payload(payload)
    requests = [
        dict(row)
        for row in payload.get("objective_conversion_handoff_requests", []) or []
        if isinstance(row, Mapping)
    ]
    valid_requests = [
        request for request in requests if _is_valid_objective_conversion_request(request)
    ]
    rejected_requests = [
        {
            "request_id": str(request.get("request_id", "")),
            "reason": "invalid_objective_conversion_request_contract",
            "support": 0,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "truth_status": M2_TRUTH_STATUS,
        }
        for request in requests
        if not _is_valid_objective_conversion_request(request)
    ]
    hypotheses = [
        hypothesis
        for request in valid_requests
        for hypothesis in generate_objective_conversion_hypotheses_for_request(
            request
        )
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
        raise ValueError(f"invalid objective-conversion hypotheses generated: {invalid}")
    return build_objective_conversion_hypothesis_payload(
        source_path=str(objective_conversion_handoff_requests_path),
        source_payload=payload,
        requests=valid_requests,
        rejected_requests=rejected_requests,
        hypotheses=hypotheses,
    )


def generate_objective_conversion_hypotheses_for_request(
    request: Mapping[str, Any],
) -> tuple[FrontierConditionedHypothesis, ...]:
    request_id = str(request.get("request_id", ""))
    game_id = str(request.get("game_id", ""))
    source_frontier_id = str(request.get("source_frontier_id", ""))
    matrix = dict(request.get("suggested_initial_experiment_matrix", {}) or {})
    base_state_family = str(
        matrix.get("base_state_family", "terminal_safe_stop_or_avoidance_state")
    )
    available_actions = tuple(
        str(action)
        for action in matrix.get("single_step_actions", []) or DEFAULT_AVAILABLE_ACTIONS
    )
    return tuple(
        _make_objective_conversion_hypothesis(
            request=request,
            index=index,
            family=family,
            claim=claim,
            base_state_family=base_state_family,
            candidate_actions=candidate_actions,
            candidate_sequences=candidate_sequences,
            metric=metric,
            expected_signal=expected_signal,
            falsification_signal=falsification_signal,
            requested_experiment_style=requested_experiment_style,
            support_condition=support_condition,
            failure_condition=failure_condition,
            priority=priority,
            game_id=game_id,
            request_id=request_id,
            source_frontier_id=source_frontier_id,
            available_actions=available_actions,
        )
        for index, (
            family,
            claim,
            candidate_actions,
            candidate_sequences,
            metric,
            expected_signal,
            falsification_signal,
            requested_experiment_style,
            support_condition,
            failure_condition,
            priority,
        ) in enumerate(_hypothesis_templates(), start=1)
    )


def build_objective_conversion_hypothesis_payload(
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
                    _objective_conversion_hypothesis_dict(hypothesis, request)
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
        style
        for request in requests
        for hypothesis in by_request.get(str(request.get("request_id", "")), [])
        for style in [_requested_style_for_hypothesis(hypothesis)]
        if style
    }
    return {
        "config": {
            "source_objective_conversion_handoff_requests_path": source_path,
            "schema_version": OBJECTIVE_CONVERSION_SCHEMA_VERSION,
            "base_m2_schema_version": M2_SCHEMA_VERSION,
            "generator_mode": "objective_conversion_heuristic_only",
            "inputs_read": ["P2.G3"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["P2", "M3", "A32", "A33"],
            "execution_performed": False,
            "policy_rollout_performed": False,
        },
        "objective_conversion_hypothesis_batches": batches,
        "rejected_objective_conversion_requests": [
            dict(row) for row in rejected_requests
        ],
        "summary": {
            "objective_conversion_requests_seen": len(
                source_payload.get("objective_conversion_handoff_requests", []) or []
            ),
            "objective_conversion_requests_consumed": len(requests),
            "objective_conversion_requests_rejected": len(rejected_requests),
            "hypothesis_batches": len(batches),
            "hypotheses_generated": len(hypotheses),
            "testable_hypotheses": len(testable),
            "ready_for_m3_candidate_experiment_request": len(testable),
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
            "execution_performed": False,
            "policy_rollout_performed": False,
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
        "objective_conversion_hypotheses_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
        "execution_performed": False,
        "policy_rollout_performed": False,
        "m3_write_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _make_objective_conversion_hypothesis(
    *,
    request: Mapping[str, Any],
    index: int,
    family: str,
    claim: str,
    base_state_family: str,
    candidate_actions: Sequence[str],
    candidate_sequences: Sequence[Sequence[str]],
    metric: str,
    expected_signal: str,
    falsification_signal: str,
    requested_experiment_style: str,
    support_condition: str,
    failure_condition: str,
    priority: float,
    game_id: str,
    request_id: str,
    source_frontier_id: str,
    available_actions: Sequence[str],
) -> FrontierConditionedHypothesis:
    target_action = str(candidate_actions[0]) if candidate_actions else "ACTION3"
    controls = tuple(
        action
        for action in ("ACTION3", "ACTION4", "ACTION6")
        if action != target_action
    )
    falsification = FalsificationCriterion(
        metric=metric,
        support_condition=support_condition,
        failure_condition=failure_condition,
        minimum_effect_size=1,
    )
    if not is_metric_measurable(metric):
        falsification = default_falsification_for_metric(metric)
    testability = HypothesisTestability(
        testable=True,
        recommended_test_type=requested_experiment_style,
        target_action=target_action,
        suggested_control_actions=controls,
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=metric,
        required_context_replay=(),
        required_action_args=None,
        expected_signal_type=expected_signal,
        measurable_by_existing_extractor=is_metric_measurable(metric),
        blocking_reason=None,
    )
    context_snapshot = ContextSnapshot(
        replay_actions=(),
        replay_action_args=None,
        frame_before_hash=None,
        live_state_signature=(
            f"objective_conversion:{base_state_family}:"
            f"{request.get('candidate_problem_statement', '')}"
        ),
        available_actions=tuple(str(action) for action in available_actions),
        local_patch=None,
        terminal_state=False,
    )
    audit = SourceGenerationAudit(
        sources=("heuristic",),
        raw_proposal_ids=(f"raw::objective_conversion::{request_id}::{index:03d}",),
        rationales=(
            "P2.G3 request is a candidate-only frontier handoff, not evidence.",
            claim,
        ),
        normalization_warnings=(
            "objective_conversion_request_not_counted_as_support",
            "policy_result_not_counted_as_scientific_verdict",
        ),
        priority_score=priority,
        priority_score_counted_as_support=False,
    )
    return FrontierConditionedHypothesis(
        hypothesis_id=f"m2g1::{game_id or 'unknown_game'}::objective_conversion::H{index:03d}",
        source_request_id=request_id,
        game_id=game_id,
        frontier_context_id=source_frontier_id or "objective_conversion::safe_but_passive",
        frontier_reason=TERMINAL_SAFE_BUT_PASSIVE,
        frontier_step=None,
        hypothesis_family=family,
        candidate_action=target_action,
        predicted_metric=metric,
        predicted_effect=expected_signal,
        rationale=(
            f"{claim} Falsification signal: {falsification_signal}. "
            f"Requested experiment style: {requested_experiment_style}. "
            "This is an M2 candidate hypothesis, not a policy decision."
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


def _objective_conversion_hypothesis_dict(
    hypothesis: FrontierConditionedHypothesis,
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    row = hypothesis.to_dict()
    template = _template_by_index(_hypothesis_index(hypothesis.hypothesis_id))
    row["revision_status"] = CANDIDATE_REVISION_STATUS
    row["base_state_family"] = str(
        dict(request.get("suggested_initial_experiment_matrix", {}) or {}).get(
            "base_state_family",
            "terminal_safe_stop_or_avoidance_state",
        )
    )
    row["candidate_actions"] = list(template["candidate_actions"])
    row["candidate_sequences"] = [list(seq) for seq in template["candidate_sequences"]]
    row["claim"] = template["claim"]
    row["expected_signal"] = template["expected_signal"]
    row["falsification_signal"] = template["falsification_signal"]
    row["requested_experiment_style"] = template["requested_experiment_style"]
    row["ready_for_m3_candidate_experiment_request"] = bool(
        hypothesis.testability.testable
    )
    row["objective_conversion_hypothesis_counted_as_confirmation"] = False
    row["policy_result_counted_as_scientific_verdict"] = False
    return row


def _hypothesis_templates() -> tuple[
    tuple[
        str,
        str,
        tuple[str, ...],
        tuple[tuple[str, ...], ...],
        str,
        str,
        str,
        str,
        str,
        str,
        float,
    ],
    ...,
]:
    return (
        (
            "post_safe_stop_objective_conversion",
            "After terminal-safe stop, ACTION3 may convert relation progress into objective progress without terminal re-entry.",
            ("ACTION3",),
            (("ACTION3",),),
            "terminal_state_after_rollout",
            "terminal_adjusted_progress_after_stop improves over hold_or_stop_state",
            "no improvement over hold_or_stop_state or terminal re-entry",
            "stop_state_action_matrix",
            "ACTION3_terminal_adjusted_progress > hold_or_stop_state_terminal_adjusted_progress",
            "ACTION3_terminal_adjusted_progress <= hold_or_stop_state_terminal_adjusted_progress or ACTION3_terminal_reentry",
            5.0,
        ),
        (
            "post_safe_stop_objective_conversion",
            "After terminal-safe stop, ACTION4 may convert relation progress into objective progress without terminal re-entry.",
            ("ACTION4",),
            (("ACTION4",),),
            "terminal_state_after_rollout",
            "terminal_adjusted_progress_after_stop improves over hold_or_stop_state",
            "no improvement over hold_or_stop_state or terminal re-entry",
            "stop_state_action_matrix",
            "ACTION4_terminal_adjusted_progress > hold_or_stop_state_terminal_adjusted_progress",
            "ACTION4_terminal_adjusted_progress <= hold_or_stop_state_terminal_adjusted_progress or ACTION4_terminal_reentry",
            4.9,
        ),
        (
            "post_safe_stop_objective_conversion",
            "After terminal-safe stop, ACTION6 may reactivate objective conversion only in specific readiness contexts.",
            ("ACTION6",),
            (("ACTION6",),),
            "objective_progress_proxy",
            "ACTION6 improves objective progress only when a readiness condition is present",
            "ACTION6 gives no objective gain or re-enters terminal risk",
            "stop_state_action_matrix",
            "ACTION6_ready_context_objective_progress > hold_or_stop_state_objective_progress",
            "ACTION6_objective_progress <= hold_or_stop_state_objective_progress or ACTION6_terminal_reentry",
            4.8,
        ),
        (
            "subgoal_target_reselection",
            "After terminal-safe stop, distance_decreases toward E136/E137/E138/E139 may no longer be the right target relation.",
            ("ACTION3", "ACTION4"),
            (("ACTION3",), ("ACTION4",)),
            "objective_progress_proxy",
            "relation target ablation separates objective gain from generic distance_decreases",
            "generic distance_decreases predicts no objective gain after safe stop",
            "relation_target_ablation_after_safe_stop",
            "alternate_relation_target_objective_progress > nearest_distance_decreases_objective_progress",
            "alternate_relation_target_objective_progress <= nearest_distance_decreases_objective_progress",
            4.4,
        ),
        (
            "subgoal_target_reselection",
            "After terminal-safe stop, the policy may need to select a relation target other than the closest one.",
            ("ACTION3", "ACTION4"),
            (("ACTION3",), ("ACTION4",)),
            "levels_completed_after_rollout",
            "non-closest relation target improves level completion over closest-target relation policy",
            "non-closest relation target fails to improve completion",
            "relation_target_ablation_after_safe_stop",
            "non_closest_relation_target_levels_completed > closest_relation_target_levels_completed",
            "non_closest_relation_target_levels_completed <= closest_relation_target_levels_completed",
            4.3,
        ),
        (
            "subgoal_target_reselection",
            "After terminal-safe stop, a global configuration transition may be a better objective signal than any local relation.",
            ("ACTION3", "ACTION4", "ACTION6"),
            (("ACTION3",), ("ACTION4",), ("ACTION6",)),
            "final_game_state",
            "global configuration transition predicts objective progress better than relation_delta",
            "global transition does not outperform relation_delta or remains terminal",
            "objective_completion_vs_relation_progress_discriminator",
            "global_configuration_metric_predicts_non_terminal_objective_progress_better_than_relation_delta",
            "global_configuration_metric_does_not_predict_non_terminal_objective_progress_better_than_relation_delta",
            4.2,
        ),
        (
            "objective_readiness_condition",
            "A readiness condition may be required before any post-safe-stop conversion action is useful.",
            ("ACTION3", "ACTION4", "ACTION6"),
            (("ACTION3",), ("ACTION4",), ("ACTION6",)),
            "objective_progress_proxy",
            "conversion actions help only when an objective-readiness signature is present",
            "conversion actions do not differ between ready and non-ready states",
            "objective_completion_vs_relation_progress_discriminator",
            "ready_state_conversion_progress > non_ready_state_conversion_progress",
            "ready_state_conversion_progress <= non_ready_state_conversion_progress",
            4.0,
        ),
        (
            "objective_readiness_condition",
            "Readiness after safe stop may depend on a plateau in relation progress.",
            ("ACTION3", "ACTION4"),
            (("ACTION3",), ("ACTION4",)),
            "objective_progress_proxy",
            "plateau-conditioned conversion beats immediate conversion",
            "plateau-conditioned conversion does not beat immediate conversion",
            "objective_completion_vs_relation_progress_discriminator",
            "plateau_conditioned_conversion_progress > immediate_conversion_progress",
            "plateau_conditioned_conversion_progress <= immediate_conversion_progress",
            3.9,
        ),
        (
            "objective_readiness_condition",
            "Readiness after safe stop may depend on HUD/horizon or consumed-action budget.",
            ("ACTION3", "ACTION4", "ACTION6"),
            (("ACTION3",), ("ACTION4",), ("ACTION6",)),
            "terminal_state_after_rollout",
            "horizon-aware readiness reduces terminal re-entry while preserving objective progress",
            "horizon-aware readiness does not reduce terminal re-entry or remains passive",
            "objective_completion_vs_relation_progress_discriminator",
            "horizon_ready_conversion_terminal_rate < horizon_unaware_conversion_terminal_rate",
            "horizon_ready_conversion_terminal_rate >= horizon_unaware_conversion_terminal_rate or no_objective_gain",
            3.8,
        ),
        (
            "terminal_safe_sequence_search",
            "ACTION3,ACTION4 after safe stop may convert better than either single-step action.",
            ("ACTION3", "ACTION4"),
            (("ACTION3", "ACTION4"),),
            "levels_completed_after_rollout",
            "ACTION3_ACTION4 sequence improves completion or terminal-adjusted progress over single-step controls",
            "ACTION3_ACTION4 fails to beat single-step controls or re-enters terminal",
            "post_safe_stop_short_sequence_probe",
            "ACTION3_ACTION4_levels_completed > best_single_step_levels_completed",
            "ACTION3_ACTION4_levels_completed <= best_single_step_levels_completed or sequence_terminal_reentry",
            3.7,
        ),
        (
            "terminal_safe_sequence_search",
            "ACTION4,ACTION3 after safe stop may convert better than either single-step action.",
            ("ACTION4", "ACTION3"),
            (("ACTION4", "ACTION3"),),
            "levels_completed_after_rollout",
            "ACTION4_ACTION3 sequence improves completion or terminal-adjusted progress over single-step controls",
            "ACTION4_ACTION3 fails to beat single-step controls or re-enters terminal",
            "post_safe_stop_short_sequence_probe",
            "ACTION4_ACTION3_levels_completed > best_single_step_levels_completed",
            "ACTION4_ACTION3_levels_completed <= best_single_step_levels_completed or sequence_terminal_reentry",
            3.6,
        ),
        (
            "terminal_safe_sequence_search",
            "ACTION6,ACTION3 or ACTION6,ACTION4 after safe stop may convert only with a terminal guard.",
            ("ACTION6", "ACTION3", "ACTION4"),
            (("ACTION6", "ACTION3"), ("ACTION6", "ACTION4")),
            "terminal_state_after_rollout",
            "guarded ACTION6 sequence improves objective progress without terminal re-entry",
            "guarded ACTION6 sequence gives no objective gain or re-enters terminal",
            "post_safe_stop_short_sequence_probe",
            "guarded_ACTION6_sequence_terminal_adjusted_progress > best_non_ACTION6_sequence_terminal_adjusted_progress",
            "guarded_ACTION6_sequence_terminal_adjusted_progress <= best_non_ACTION6_sequence_terminal_adjusted_progress or terminal_reentry",
            3.5,
        ),
    )


def _template_by_index(index: int) -> Dict[str, Any]:
    template = _hypothesis_templates()[index - 1]
    return {
        "family": template[0],
        "claim": template[1],
        "candidate_actions": template[2],
        "candidate_sequences": template[3],
        "metric": template[4],
        "expected_signal": template[5],
        "falsification_signal": template[6],
        "requested_experiment_style": template[7],
    }


def _hypothesis_index(hypothesis_id: str) -> int:
    marker = "::H"
    if marker not in hypothesis_id:
        return 1
    return int(hypothesis_id.rsplit(marker, 1)[-1])


def _requested_style_for_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> str:
    return _template_by_index(_hypothesis_index(hypothesis.hypothesis_id))[
        "requested_experiment_style"
    ]


def _falsification_signal_for_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> str:
    return _template_by_index(_hypothesis_index(hypothesis.hypothesis_id))[
        "falsification_signal"
    ]


def _is_valid_objective_conversion_request(request: Mapping[str, Any]) -> bool:
    return bool(
        str(request.get("handoff_type", "")) == HANDOFF_TYPE
        and str(request.get("target", "")) == HANDOFF_TARGET
        and list(request.get("target_modules", []) or []) == TARGET_MODULES
        and str(request.get("frontier_type", "")) == OBJECTIVE_CONVERSION_FRONTIER
        and str(request.get("frontier_reason", "")) == TERMINAL_SAFE_BUT_PASSIVE
        and str(request.get("blocked_capability", ""))
        == OBJECTIVE_CONVERSION_AFTER_SAFE_STOP
        and bool(request.get("objective_conversion_review_accepted", False))
        and bool(request.get("ready_for_objective_conversion_hypothesis_generation", False))
        and bool(request.get("ready_for_m2_or_m3_objective_conversion_branch", False))
        and not bool(request.get("ready_for_direct_downstream_write", False))
        and not bool(request.get("a33_ready", False))
        and list(request.get("requested_hypothesis_families", []) or [])
        and list(request.get("requested_experiment_styles", []) or [])
        and list(request.get("scientific_questions", []) or [])
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
        raise ValueError("objective conversion source cannot be A33-ready")
    if bool(summary.get("ready_for_direct_downstream_write", False)) or bool(
        payload.get("ready_for_direct_downstream_write", False)
    ):
        raise ValueError("objective conversion source cannot request direct write")
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
        raise ValueError("objective conversion handoff cannot be confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be a scientific verdict")


def write_objective_conversion_hypotheses(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate M2.G1 objective-conversion hypotheses.",
    )
    parser.add_argument(
        "--objective-conversion-requests",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_CONVERSION_HANDOFF_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_M2_OBJECTIVE_CONVERSION_HYPOTHESES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_hypothesis_generator(
        objective_conversion_handoff_requests_path=args.objective_conversion_requests,
    )
    write_objective_conversion_hypotheses(payload, args.out)
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
