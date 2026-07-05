"""Compile M2 hypotheses into M3 candidate experiment requests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .schema import (
    M2_BLOCKED_FOR_M3_STATUS,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_READY_FOR_M3_STATUS,
    M2_TO_M3_SCHEMA_VERSION,
    M2_TRUTH_STATUS,
    FalsificationCriterion,
    FrontierConditionedHypothesis,
    M3CandidateExperimentRequest,
)
from .validators import validate_m3_request


DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "m3_candidate_experiment_requests.json"
)


def compile_m3_request(
    hypothesis: FrontierConditionedHypothesis,
) -> M3CandidateExperimentRequest:
    testability = hypothesis.testability
    ready = bool(
        testability.testable
        and testability.target_action
        and testability.metric
        and testability.expected_signal_type
        and hypothesis.falsification.metric
    )
    request = M3CandidateExperimentRequest(
        request_id=request_id_for_hypothesis(hypothesis.hypothesis_id),
        source_hypothesis_id=hypothesis.hypothesis_id,
        game_id=hypothesis.game_id,
        context_replay=tuple(testability.required_context_replay),
        context_replay_args=testability.required_action_args,
        context_snapshot_hash=hypothesis.context_snapshot.frame_before_hash,
        target_action=testability.target_action,
        target_action_args=None,
        suggested_control_actions=tuple(testability.suggested_control_actions),
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=testability.metric,
        expected_signal=testability.expected_signal_type,
        falsification_criterion=hypothesis.falsification,
        status=M2_READY_FOR_M3_STATUS if ready else M2_BLOCKED_FOR_M3_STATUS,
        truth_status=M2_TRUTH_STATUS,
        support=0,
        controlled_test_required=True,
        revision_performed=False,
        wrong_confirmations=0,
    )
    result = validate_m3_request(request)
    if not result.valid:
        raise ValueError(f"invalid M3 request {request.request_id}: {result.errors}")
    return request


def compile_m3_requests(
    hypotheses: Sequence[FrontierConditionedHypothesis],
) -> tuple[M3CandidateExperimentRequest, ...]:
    return tuple(compile_m3_request(hypothesis) for hypothesis in hypotheses)


def build_m3_requests_payload(
    hypotheses: Sequence[FrontierConditionedHypothesis],
    *,
    source_hypothesis_path: str,
) -> Dict[str, Any]:
    requests = compile_m3_requests(hypotheses)
    ready = [
        request
        for request in requests
        if request.status == M2_READY_FOR_M3_STATUS
    ]
    blocked = [
        request
        for request in requests
        if request.status == M2_BLOCKED_FOR_M3_STATUS
    ]
    return {
        "config": {
            "source_hypothesis_path": source_hypothesis_path,
            "schema_version": M2_TO_M3_SCHEMA_VERSION,
        },
        "experiment_requests": [request.to_dict() for request in requests],
        "summary": {
            "source_hypotheses": len(hypotheses),
            "experiment_requests": len(requests),
            "ready_for_m3": len(ready),
            "blocked_not_testable": len(blocked),
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
    }


def request_id_for_hypothesis(hypothesis_id: str) -> str:
    if hypothesis_id.startswith("m2::"):
        return "m2m3::" + hypothesis_id[len("m2::") :]
    return f"m2m3::{hypothesis_id}"


def write_m3_requests_payload(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_M2_M3_REQUESTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def m3_request_from_dict(data: Mapping[str, Any]) -> M3CandidateExperimentRequest:
    falsification = data.get("falsification_criterion") or {}
    return M3CandidateExperimentRequest(
        request_id=str(data.get("request_id", "")),
        source_hypothesis_id=str(data.get("source_hypothesis_id", "")),
        game_id=str(data.get("game_id", "")),
        context_replay=tuple(str(item) for item in data.get("context_replay", []) or []),
        context_replay_args=_tuple_dicts_or_none(data.get("context_replay_args")),
        context_snapshot_hash=data.get("context_snapshot_hash"),
        target_action=str(data.get("target_action", "")),
        target_action_args=(
            dict(data.get("target_action_args") or {})
            if data.get("target_action_args") is not None
            else None
        ),
        suggested_control_actions=tuple(
            str(item) for item in data.get("suggested_control_actions", []) or []
        ),
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=str(data.get("metric", "")),
        expected_signal=str(data.get("expected_signal", "")),
        falsification_criterion=FalsificationCriterion(
            metric=str(falsification.get("metric", "")),
            support_condition=str(falsification.get("support_condition", "")),
            failure_condition=str(falsification.get("failure_condition", "")),
            minimum_effect_size=falsification.get("minimum_effect_size"),
        ),
        status=str(data.get("status", M2_BLOCKED_FOR_M3_STATUS)),
        source_episode_id=_optional_str(data.get("source_episode_id")),
        source_step=_optional_int(data.get("source_step")),
        source_transition_id=_optional_str(data.get("source_transition_id")),
        context_state_origin=_optional_str(data.get("context_state_origin")),
        replayability=_optional_str(data.get("replayability")),
        blocking_reason=_optional_str(data.get("blocking_reason")),
        truth_status=str(data.get("truth_status", "")),
        support=int(data.get("support", 0) or 0),
        controlled_test_required=bool(data.get("controlled_test_required", False)),
        revision_performed=bool(data.get("revision_performed", False)),
        wrong_confirmations=int(data.get("wrong_confirmations", 0) or 0),
    )


def load_m3_requests_payload(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def requests_from_payload(
    payload: Mapping[str, Any],
) -> tuple[M3CandidateExperimentRequest, ...]:
    return tuple(
        m3_request_from_dict(row)
        for row in payload.get("experiment_requests", []) or []
        if isinstance(row, Mapping)
    )


def _tuple_dicts_or_none(value: Any) -> tuple[Dict[str, Any], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
