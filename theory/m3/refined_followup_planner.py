"""M3.9 follow-up planner for refined M2-derived hypotheses."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .m2_observation_refinement import (
    DEFAULT_REFINED_M2_HYPOTHESES_OUTPUT_PATH,
    M3_REFINEMENT_TRUTH_STATUS,
)


DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "refined_followup_experiment_requests.json"
)
DEFAULT_REACTIVATION_METRICS = (
    "local_patch_before_after",
    "changed_pixels",
    "object_positions_before_after",
    "contact_graph_before_after",
)
GLOBAL_REPOSITIONING_MECHANIC = "global_object_repositioning_after_consumption"
READY_FOR_M3_FOLLOWUP = "READY_FOR_M3_FOLLOWUP"
BLOCKED_NO_FOLLOWUP = "BLOCKED_NO_FOLLOWUP"


@dataclass(frozen=True)
class RefinedFollowupExperimentRequest:
    """A candidate-only discriminating test request derived from M3.8."""

    request_id: str
    source_refined_hypothesis_id: str
    source_hypothesis_ids: Tuple[str, ...]
    game_id: str
    hypothesis_tested: str
    source_candidate_mechanic: str
    source_observed_effect_family: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    target_action_args: Dict[str, Any] | None
    suggested_control_actions: Tuple[str, ...]
    metrics: Tuple[str, ...]
    expected_signal: str
    falsification_criteria: Tuple[Dict[str, Any], ...]
    discriminates_between: Tuple[str, ...]
    planning_rationale: str
    control_policy: str = "m3_dynamic_available_controls"
    status: str = READY_FOR_M3_FOLLOWUP
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False
    followup_request_counted_as_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_refined_hypothesis_id": self.source_refined_hypothesis_id,
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "game_id": self.game_id,
            "hypothesis_tested": self.hypothesis_tested,
            "source_candidate_mechanic": self.source_candidate_mechanic,
            "source_observed_effect_family": self.source_observed_effect_family,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "target_action_args": (
                dict(self.target_action_args)
                if self.target_action_args is not None
                else None
            ),
            "suggested_control_actions": list(self.suggested_control_actions),
            "control_policy": self.control_policy,
            "metrics": list(self.metrics),
            "expected_signal": self.expected_signal,
            "falsification_criteria": [
                dict(item) for item in self.falsification_criteria
            ],
            "discriminates_between": list(self.discriminates_between),
            "planning_rationale": self.planning_rationale,
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
            "followup_request_counted_as_support": (
                self.followup_request_counted_as_support
            ),
        }


def run_refined_followup_planning(
    *,
    refined_hypotheses_path: str | Path = DEFAULT_REFINED_M2_HYPOTHESES_OUTPUT_PATH,
) -> Dict[str, Any]:
    payload = _load_json(refined_hypotheses_path)
    refined = [
        dict(item) for item in payload.get("refined_candidate_hypotheses", []) or []
    ]
    requests: list[RefinedFollowupExperimentRequest] = []
    skipped: list[Dict[str, Any]] = []
    for hypothesis in refined:
        request = build_followup_request_for_refined_hypothesis(hypothesis)
        if request is None:
            skipped.append(skipped_followup_reason(hypothesis))
            continue
        requests.append(request)

    return {
        "config": {
            "refined_hypotheses_path": str(refined_hypotheses_path),
            "schema_version": "m3.refined_followup_requests.v1",
            "inputs_read": ["M3.8"],
            "artifacts_not_modified": ["M2", "M3.8", "A32", "A33"],
            "execution_performed": False,
        },
        "summary": summarize_followup_requests(refined, requests, skipped),
        "followup_experiment_requests": [
            request.to_dict() for request in requests
        ],
        "skipped_refined_hypotheses": [dict(item) for item in skipped],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
        "followup_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def build_followup_request_for_refined_hypothesis(
    hypothesis: Mapping[str, Any],
) -> RefinedFollowupExperimentRequest | None:
    if str(hypothesis.get("candidate_mechanic", "")) != GLOBAL_REPOSITIONING_MECHANIC:
        return None
    context = tuple(str(item) for item in hypothesis.get("context_replay", []) or [])
    if not context:
        return None
    enabling_action = str(hypothesis.get("target_action", ""))
    blocked_skill = str(context[0])
    if not enabling_action or not blocked_skill:
        return None
    context_args = _context_args_tuple(hypothesis.get("context_replay_args"))
    followup_context = (*context, enabling_action)
    followup_args = (
        (*context_args, {})
        if context_args is not None
        else None
    )
    controls = suggested_controls_for_reactivation(
        context=context,
        enabling_action=enabling_action,
        target_action=blocked_skill,
    )
    return RefinedFollowupExperimentRequest(
        request_id=followup_request_id(hypothesis, target_action=blocked_skill),
        source_refined_hypothesis_id=str(hypothesis.get("refined_hypothesis_id", "")),
        source_hypothesis_ids=tuple(
            str(item) for item in hypothesis.get("source_hypothesis_ids", []) or []
        ),
        game_id=str(hypothesis.get("game_id", "")),
        hypothesis_tested=(
            f"{enabling_action} is a global repositioning/reset operator that "
            f"re-enables {blocked_skill} after consumption"
        ),
        source_candidate_mechanic=str(hypothesis.get("candidate_mechanic", "")),
        source_observed_effect_family=str(
            hypothesis.get("observed_effect_family", "")
        ),
        context_replay=followup_context,
        context_replay_args=followup_args,
        target_action=blocked_skill,
        target_action_args=None,
        suggested_control_actions=controls,
        metrics=DEFAULT_REACTIVATION_METRICS,
        expected_signal=(
            "reactivated_target_action_signal_exceeds_dynamic_controls"
        ),
        falsification_criteria=tuple(
            falsification_criterion(metric)
            for metric in DEFAULT_REACTIVATION_METRICS
        ),
        discriminates_between=(
            f"{enabling_action}_repositions_objects_and_reactivates_{blocked_skill}",
            f"{enabling_action}_causes_visual_motion_without_reactivation",
            f"{enabling_action}_prepares_nonlocal_state_not_captured_by_current_metrics",
        ),
        planning_rationale=(
            "M3.8 observed grounded global motion without local patch or contact "
            "graph change; the discriminating follow-up retests the consumed "
            "skill after the repositioning action."
        ),
    )


def suggested_controls_for_reactivation(
    *,
    context: Sequence[str],
    enabling_action: str,
    target_action: str,
) -> Tuple[str, ...]:
    candidates: list[str] = []
    if len(context) >= 2:
        candidates.append(str(context[-1]))
    candidates.append(str(enabling_action))
    for action in context:
        if str(action) != str(target_action):
            candidates.append(str(action))
    result: list[str] = []
    for action in candidates:
        if not action or action == target_action or action in result:
            continue
        result.append(action)
    return tuple(result)


def falsification_criterion(metric: str) -> Dict[str, Any]:
    return {
        "metric": metric,
        "support_condition": (
            "target_action_signal_after_repositioning > best_control_signal"
        ),
        "failure_condition": (
            "target_action_signal_after_repositioning <= best_control_signal"
        ),
        "minimum_effect_size": 1,
    }


def skipped_followup_reason(hypothesis: Mapping[str, Any]) -> Dict[str, Any]:
    mechanic = str(hypothesis.get("candidate_mechanic", ""))
    reason = (
        "unsupported_candidate_mechanic_for_m3_9"
        if mechanic != GLOBAL_REPOSITIONING_MECHANIC
        else "missing_context_or_target_action"
    )
    return {
        "source_refined_hypothesis_id": str(
            hypothesis.get("refined_hypothesis_id", "")
        ),
        "candidate_mechanic": mechanic,
        "reason": reason,
        "status": BLOCKED_NO_FOLLOWUP,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def summarize_followup_requests(
    refined: Sequence[Mapping[str, Any]],
    requests: Sequence[RefinedFollowupExperimentRequest],
    skipped: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "refined_hypotheses_consumed": len(refined),
        "followup_requests_generated": len(requests),
        "reactivation_tests_generated": len(
            [
                request
                for request in requests
                if "re-enables" in request.hypothesis_tested
            ]
        ),
        "multi_metric_requests": len(
            [request for request in requests if len(request.metrics) > 1]
        ),
        "skipped_refined_hypotheses": len(skipped),
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "followup_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def followup_request_id(
    hypothesis: Mapping[str, Any],
    *,
    target_action: str,
) -> str:
    source = str(hypothesis.get("refined_hypothesis_id", "")).replace("::", "_")
    return f"m3_9::{source}::retest_{target_action}"


def write_refined_followup_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _context_args_tuple(raw: Any) -> Tuple[Dict[str, Any], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build M3.9 follow-up requests from refined hypotheses.",
    )
    parser.add_argument(
        "--refined",
        type=Path,
        default=DEFAULT_REFINED_M2_HYPOTHESES_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REFINED_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_refined_followup_planning(
        refined_hypotheses_path=args.refined,
    )
    write_refined_followup_requests(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
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
