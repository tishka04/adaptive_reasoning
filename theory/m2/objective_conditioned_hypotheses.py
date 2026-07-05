"""M2.O1 objective-conditioned hypothesis generation from P2 requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.p2.objective_frontier_handoff_schema import (
    DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_REQUESTS_OUTPUT_PATH,
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


DEFAULT_M2_OBJECTIVE_CONDITIONED_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "objective_conditioned_hypotheses.json"
)
OBJECTIVE_SCHEMA_VERSION = "m2.objective_conditioned.v1"
OBJECTIVE_HANDOFF_TYPE = "OBJECTIVE_ALIGNMENT_FRONTIER_REQUEST"
OBJECTIVE_FRONTIER_TYPE = "OBJECTIVE_ALIGNMENT_FRONTIER"
OBJECTIVE_FRONTIER_REASON = (
    "LOCAL_AFFORDANCE_PRODUCTIVE_BUT_TERMINAL_OBJECTIVE_FAILED"
)
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
DEFAULT_OBJECTIVE_ACTIONS = ("ACTION6", "ACTION3", "ACTION4", "ACTION1", "ACTION2")


def run_objective_conditioned_hypotheses(
    *,
    objective_frontier_requests_path: str | Path = (
        DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_REQUESTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(objective_frontier_requests_path)
    _validate_source_payload(payload)
    requests = [
        dict(row)
        for row in payload.get("objective_frontier_requests", []) or []
        if isinstance(row, Mapping)
    ]
    valid_requests = [row for row in requests if _is_valid_objective_request(row)]
    rejected_requests = [
        {
            "request_id": str(row.get("request_id", "")),
            "reason": "invalid_objective_request_contract",
            "support": 0,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "truth_status": M2_TRUTH_STATUS,
        }
        for row in requests
        if not _is_valid_objective_request(row)
    ]
    hypotheses = [
        hypothesis
        for request in valid_requests
        for hypothesis in generate_objective_hypotheses_for_request(request)
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
        raise ValueError(f"invalid objective hypotheses generated: {invalid}")
    return build_objective_hypothesis_payload(
        source_path=str(objective_frontier_requests_path),
        source_payload=payload,
        requests=valid_requests,
        rejected_requests=rejected_requests,
        hypotheses=hypotheses,
    )


def generate_objective_hypotheses_for_request(
    request: Mapping[str, Any],
) -> tuple[FrontierConditionedHypothesis, ...]:
    request_id = str(request.get("request_id", ""))
    game_id = str(request.get("game_id", ""))
    frontier_context_id = _frontier_context_id(request)
    observed = dict(request.get("observed_pattern", {}) or {})
    terminal_runs = int(request.get("terminal_runs", 0) or 0)
    terminal_budgets = list(request.get("terminal_budgets", []) or [])
    available_actions = tuple(
        action
        for action in DEFAULT_OBJECTIVE_ACTIONS
        if action
    )
    return (
        _make_objective_hypothesis(
            index=1,
            request_id=request_id,
            game_id=game_id,
            frontier_context_id=frontier_context_id,
            family="stop_switch_criterion",
            candidate_action="ACTION6",
            metric="terminal_state_after_rollout",
            effect=(
                "Continuing repeated ACTION6 after local affordance exploitation may "
                "increase terminal outcome risk compared with stopping or switching."
            ),
            rationale=(
                f"P2 observed {terminal_runs} terminal runs where ACTION6 remained "
                "locally productive but the rollout ended in GAME_OVER with zero "
                "levels completed."
            ),
            support_condition=(
                "continue_action6_terminal_rate > stop_or_switch_terminal_rate"
            ),
            failure_condition=(
                "continue_action6_terminal_rate <= stop_or_switch_terminal_rate"
            ),
            expected_signal=(
                "continue_action6_produces_more_terminal_outcomes_than_switch_control"
            ),
            controls=("ACTION3", "ACTION4", "ACTION1", "ACTION2"),
            available_actions=available_actions,
            observed=observed,
            terminal_budgets=terminal_budgets,
            priority=5.0,
        ),
        _make_objective_hypothesis(
            index=2,
            request_id=request_id,
            game_id=game_id,
            frontier_context_id=frontier_context_id,
            family="terminal_risk_predictor",
            candidate_action="ACTION6",
            metric="final_game_state",
            effect=(
                "GAME_OVER risk may rise after a long prefix of productive repeated "
                "ACTION6 even while local metrics remain positive."
            ),
            rationale=(
                "P2.3/P2.4-terminal separated local affordance productivity from "
                "global objective failure; prefix length is a falsifiable risk signal."
            ),
            support_condition=(
                "long_action6_prefix_game_over_rate > short_action6_prefix_game_over_rate"
            ),
            failure_condition=(
                "long_action6_prefix_game_over_rate <= short_action6_prefix_game_over_rate"
            ),
            expected_signal="longer_action6_prefix_increases_game_over_rate",
            controls=("ACTION3", "ACTION4"),
            available_actions=available_actions,
            observed=observed,
            terminal_budgets=terminal_budgets,
            priority=4.5,
        ),
        _make_objective_hypothesis(
            index=3,
            request_id=request_id,
            game_id=game_id,
            frontier_context_id=frontier_context_id,
            family="subgoal_switch_after_local_affordance",
            candidate_action="ACTION3",
            metric="levels_completed_after_rollout",
            effect=(
                "Switching to a non-ACTION6 subgoal after productive ACTION6 may "
                "improve level completion compared with continuing ACTION6."
            ),
            rationale=(
                "The objective frontier asks what should replace repeated ACTION6 "
                "when local effects no longer align with the global objective."
            ),
            support_condition=(
                "switch_policy_levels_completed > continue_action6_levels_completed"
            ),
            failure_condition=(
                "switch_policy_levels_completed <= continue_action6_levels_completed"
            ),
            expected_signal=(
                "switch_action_after_productive_action6_improves_level_completion"
            ),
            controls=("ACTION6", "ACTION4", "ACTION1", "ACTION2"),
            available_actions=available_actions,
            observed=observed,
            terminal_budgets=terminal_budgets,
            priority=4.0,
        ),
        _make_objective_hypothesis(
            index=4,
            request_id=request_id,
            game_id=game_id,
            frontier_context_id=frontier_context_id,
            family="global_objective_alignment_metric",
            candidate_action="ACTION6",
            metric="objective_progress_proxy",
            effect=(
                "Local patch/object-position effects may be insufficient as a success "
                "metric; objective progress must penalize GAME_OVER and reward level "
                "completion."
            ),
            rationale=(
                "The same local ACTION6 affordance remains productive in P2 while the "
                "global terminal outcome fails, so metric alignment is itself a "
                "testable hypothesis."
            ),
            support_condition=(
                "objective_aligned_metric_selects_safer_policy_than_local_effect_metric"
            ),
            failure_condition=(
                "objective_aligned_metric_does_not_select_safer_policy_than_local_effect_metric"
            ),
            expected_signal=(
                "objective_metric_predicts_terminal_failure_better_than_local_metric"
            ),
            controls=("ACTION3", "ACTION4", "ACTION1", "ACTION2"),
            available_actions=available_actions,
            observed=observed,
            terminal_budgets=terminal_budgets,
            priority=3.5,
        ),
    )


def build_objective_hypothesis_payload(
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
                "handoff_type": str(request.get("handoff_type", "")),
                "target": str(request.get("target", "")),
                "candidate_hypotheses": [
                    _objective_hypothesis_dict(hypothesis)
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
    return {
        "config": {
            "source_objective_frontier_requests_path": source_path,
            "schema_version": OBJECTIVE_SCHEMA_VERSION,
            "base_m2_schema_version": M2_SCHEMA_VERSION,
            "generator_mode": "objective_heuristic_only",
            "inputs_read": ["P2.6"],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["P2", "M3", "A32", "A33"],
        },
        "objective_hypothesis_batches": batches,
        "rejected_objective_requests": [dict(row) for row in rejected_requests],
        "summary": {
            "objective_requests_seen": len(
                source_payload.get("objective_frontier_requests", []) or []
            ),
            "objective_requests_consumed": len(requests),
            "objective_requests_rejected": len(rejected_requests),
            "hypothesis_batches": len(batches),
            "hypotheses_generated": len(hypotheses),
            "testable_hypotheses": len(testable),
            "ready_for_m3_candidate_experiment_request": len(testable),
            "blocked_not_testable_hypotheses": len(hypotheses) - len(testable),
            "final_invalid_hypotheses": len(invalid),
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
        "objective_hypotheses_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": False,
    }


def _make_objective_hypothesis(
    *,
    index: int,
    request_id: str,
    game_id: str,
    frontier_context_id: str,
    family: str,
    candidate_action: str,
    metric: str,
    effect: str,
    rationale: str,
    support_condition: str,
    failure_condition: str,
    expected_signal: str,
    controls: Sequence[str],
    available_actions: Sequence[str],
    observed: Mapping[str, Any],
    terminal_budgets: Sequence[Any],
    priority: float,
) -> FrontierConditionedHypothesis:
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
        recommended_test_type="controlled_policy_prefix_vs_stop_or_switch",
        target_action=candidate_action,
        suggested_control_actions=tuple(
            action for action in controls if action != candidate_action
        ),
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
            "objective_frontier:"
            f"{observed.get('known_failure_state', 'GAME_OVER')}:"
            f"levels={observed.get('known_levels_completed', 0)}"
        ),
        available_actions=tuple(available_actions),
        local_patch=None,
        terminal_state=bool(observed.get("terminal_objective_failed", True)),
    )
    audit = SourceGenerationAudit(
        sources=("heuristic",),
        raw_proposal_ids=(f"raw::objective::{request_id}::{index:03d}",),
        rationales=(rationale,),
        normalization_warnings=(
            "objective_frontier_request_not_counted_as_support",
            "policy_result_not_counted_as_scientific_verdict",
        ),
        priority_score=priority,
        priority_score_counted_as_support=False,
    )
    return FrontierConditionedHypothesis(
        hypothesis_id=f"m2_o1::{request_id}::h{index:03d}",
        source_request_id=request_id,
        game_id=game_id,
        frontier_context_id=frontier_context_id,
        frontier_reason=OBJECTIVE_FRONTIER_REASON,
        frontier_step=None,
        hypothesis_family=family,
        candidate_action=candidate_action,
        predicted_metric=metric,
        predicted_effect=effect,
        rationale=(
            f"{rationale} Terminal budgets observed: "
            f"{[int(value) for value in terminal_budgets if value is not None]}."
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


def _objective_hypothesis_dict(
    hypothesis: FrontierConditionedHypothesis,
) -> Dict[str, Any]:
    row = hypothesis.to_dict()
    row["revision_status"] = CANDIDATE_REVISION_STATUS
    row["ready_for_m3_candidate_experiment_request"] = bool(
        hypothesis.testability.testable
    )
    row["objective_hypothesis_counted_as_confirmation"] = False
    row["policy_result_counted_as_scientific_verdict"] = False
    return row


def _frontier_context_id(request: Mapping[str, Any]) -> str:
    source = str(request.get("source_frontier_id", ""))
    if source:
        return source
    return "objective_alignment::local_affordance_productive_but_terminal"


def _is_valid_objective_request(request: Mapping[str, Any]) -> bool:
    return bool(
        str(request.get("handoff_type", "")) == OBJECTIVE_HANDOFF_TYPE
        and str(request.get("target", "")) == "M2_OR_M3"
        and str(request.get("frontier_type", "")) == OBJECTIVE_FRONTIER_TYPE
        and str(request.get("frontier_reason", "")) == OBJECTIVE_FRONTIER_REASON
        and bool(request.get("objective_review_accepted", False))
        and bool(request.get("ready_for_m2_or_m3_objective_branch", False))
        and not bool(request.get("ready_for_saturation_handoff", False))
        and not bool(request.get("a33_ready", False))
        and int(request.get("support", 0) or 0) == 0
        and str(request.get("revision_status", "")) == CANDIDATE_REVISION_STATUS
        and str(request.get("truth_status", "")) == "NOT_EVALUATED_BY_P2"
        and not bool(request.get("revision_performed", False))
        and int(request.get("wrong_confirmations", 0) or 0) == 0
    )


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(summary.get("revision_performed", False)):
        raise ValueError("source summary revision_performed must be false")
    if int(summary.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("source summary wrong_confirmations must remain 0")
    if bool(summary.get("ready_for_saturation_handoff", False)):
        raise ValueError("objective source cannot be ready for saturation handoff")
    if bool(summary.get("a33_ready", False)) or bool(payload.get("a33_ready", False)):
        raise ValueError("objective source cannot be A33-ready")
    if bool(payload.get("objective_frontier_request_counted_as_confirmation", False)):
        raise ValueError("objective source cannot be counted as confirmation")
    if bool(payload.get("policy_result_counted_as_scientific_verdict", False)):
        raise ValueError("policy result cannot be a scientific verdict")


def write_objective_conditioned_hypotheses(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_OBJECTIVE_CONDITIONED_HYPOTHESES_OUTPUT_PATH,
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
        description="Generate M2.O1 objective-conditioned hypotheses.",
    )
    parser.add_argument(
        "--objective-frontier-requests",
        type=Path,
        default=DEFAULT_P2_OBJECTIVE_FRONTIER_HANDOFF_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_M2_OBJECTIVE_CONDITIONED_HYPOTHESES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conditioned_hypotheses(
        objective_frontier_requests_path=args.objective_frontier_requests,
    )
    write_objective_conditioned_hypotheses(payload, args.out)
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
