"""Schema-first contract for M2 hypotheses and M2 -> M3 requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Literal, Mapping, Sequence, Tuple


M2_SCHEMA_VERSION = "m2.v1"
M2_TO_M3_SCHEMA_VERSION = "m2_to_m3.v1"
M2_TRUTH_STATUS = "NOT_EVALUATED_BY_M2"
M2_HYPOTHESIS_STATUS = "UNRESOLVED"
M2_READY_FOR_M3_STATUS = "READY_FOR_M3"
M2_BLOCKED_FOR_M3_STATUS = "BLOCKED_NOT_TESTABLE"
M2_DYNAMIC_CONTROL_POLICY = "m3_dynamic_available_controls"

GeneratorSource = Literal["heuristic", "llm", "world_model"]


@dataclass(frozen=True)
class ValidationResult:
    """Validation result used by M2 schema guards."""

    valid: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.valid)


@dataclass(frozen=True)
class FalsificationCriterion:
    metric: str
    support_condition: str
    failure_condition: str
    minimum_effect_size: float | int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisTestability:
    testable: bool
    recommended_test_type: str
    target_action: str
    suggested_control_actions: Tuple[str, ...] = ()
    control_policy: Literal[
        "m3_dynamic_available_controls", "fixed_suggested_controls"
    ] = M2_DYNAMIC_CONTROL_POLICY
    metric: str = ""
    required_context_replay: Tuple[str, ...] = ()
    required_action_args: Tuple[Dict[str, Any], ...] | None = None
    expected_signal_type: str = ""
    measurable_by_existing_extractor: bool = False
    blocking_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "testable": self.testable,
            "recommended_test_type": self.recommended_test_type,
            "target_action": self.target_action,
            "suggested_control_actions": list(self.suggested_control_actions),
            "control_policy": self.control_policy,
            "metric": self.metric,
            "required_context_replay": list(self.required_context_replay),
            "required_action_args": _list_of_dicts_or_none(self.required_action_args),
            "expected_signal_type": self.expected_signal_type,
            "measurable_by_existing_extractor": self.measurable_by_existing_extractor,
            "blocking_reason": self.blocking_reason,
        }


@dataclass(frozen=True)
class ContextSnapshot:
    replay_actions: Tuple[str, ...] = ()
    replay_action_args: Tuple[Dict[str, Any], ...] | None = None
    frame_before_hash: str | None = None
    live_state_signature: str | None = None
    available_actions: Tuple[str, ...] = ()
    local_patch: Tuple[Tuple[int, ...], ...] | None = None
    terminal_state: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_actions": list(self.replay_actions),
            "replay_action_args": _list_of_dicts_or_none(self.replay_action_args),
            "frame_before_hash": self.frame_before_hash,
            "live_state_signature": self.live_state_signature,
            "available_actions": list(self.available_actions),
            "local_patch": (
                [list(row) for row in self.local_patch]
                if self.local_patch is not None
                else None
            ),
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class SourceGenerationAudit:
    sources: Tuple[GeneratorSource, ...] = ()
    raw_proposal_ids: Tuple[str, ...] = ()
    rationales: Tuple[str, ...] = ()
    normalization_warnings: Tuple[str, ...] = ()
    priority_score: float = 0.0
    priority_score_counted_as_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": list(self.sources),
            "raw_proposal_ids": list(self.raw_proposal_ids),
            "rationales": list(self.rationales),
            "normalization_warnings": list(self.normalization_warnings),
            "priority_score": round(float(self.priority_score), 4),
            "priority_score_counted_as_support": (
                self.priority_score_counted_as_support
            ),
        }


@dataclass(frozen=True)
class FrontierConditionedHypothesis:
    hypothesis_id: str
    source_request_id: str
    game_id: str
    frontier_context_id: str
    frontier_reason: str
    frontier_step: int | None
    hypothesis_family: str
    candidate_action: str
    predicted_metric: str
    predicted_effect: str
    rationale: str
    testability: HypothesisTestability
    falsification: FalsificationCriterion
    context_snapshot: ContextSnapshot
    source_generation: SourceGenerationAudit
    unlock_target_actions: Tuple[str, ...] = ()
    status: Literal["UNRESOLVED"] = M2_HYPOTHESIS_STATUS
    support: int = 0
    controlled_test_required: bool = True
    truth_status: Literal["NOT_EVALUATED_BY_M2"] = M2_TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "source_request_id": self.source_request_id,
            "game_id": self.game_id,
            "frontier_context_id": self.frontier_context_id,
            "frontier_reason": self.frontier_reason,
            "frontier_step": self.frontier_step,
            "hypothesis_family": self.hypothesis_family,
            "candidate_action": self.candidate_action,
            "predicted_metric": self.predicted_metric,
            "predicted_effect": self.predicted_effect,
            "rationale": self.rationale,
            "testability": self.testability.to_dict(),
            "falsification": self.falsification.to_dict(),
            "context_snapshot": self.context_snapshot.to_dict(),
            "source_generation": self.source_generation.to_dict(),
            "unlock_target_actions": list(self.unlock_target_actions),
            "status": self.status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


@dataclass(frozen=True)
class M3CandidateExperimentRequest:
    request_id: str
    source_hypothesis_id: str
    game_id: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    context_snapshot_hash: str | None
    target_action: str
    target_action_args: Dict[str, Any] | None
    suggested_control_actions: Tuple[str, ...]
    control_policy: Literal["m3_dynamic_available_controls"]
    metric: str
    expected_signal: str
    falsification_criterion: FalsificationCriterion
    status: Literal["READY_FOR_M3", "BLOCKED_NOT_TESTABLE"]
    source_episode_id: str | None = None
    source_step: int | None = None
    source_transition_id: str | None = None
    context_state_origin: str | None = None
    replayability: str | None = None
    blocking_reason: str | None = None
    truth_status: Literal["NOT_EVALUATED_BY_M2"] = M2_TRUTH_STATUS
    support: int = 0
    controlled_test_required: bool = True
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_hypothesis_id": self.source_hypothesis_id,
            "game_id": self.game_id,
            "context_replay": list(self.context_replay),
            "context_replay_args": _list_of_dicts_or_none(self.context_replay_args),
            "context_snapshot_hash": self.context_snapshot_hash,
            "target_action": self.target_action,
            "target_action_args": (
                dict(self.target_action_args)
                if self.target_action_args is not None
                else None
            ),
            "suggested_control_actions": list(self.suggested_control_actions),
            "control_policy": self.control_policy,
            "metric": self.metric,
            "expected_signal": self.expected_signal,
            "falsification_criterion": self.falsification_criterion.to_dict(),
            "status": self.status,
            "source_episode_id": self.source_episode_id,
            "source_step": self.source_step,
            "source_transition_id": self.source_transition_id,
            "context_state_origin": self.context_state_origin,
            "replayability": self.replayability,
            "blocking_reason": self.blocking_reason,
            "truth_status": self.truth_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


@dataclass(frozen=True)
class RawHypothesisProposal:
    proposal_id: str
    source: GeneratorSource
    source_request_id: str
    game_id: str
    frontier_context_id: str
    frontier_reason: str
    frontier_step: int | None
    hypothesis_family: str
    candidate_action: str
    predicted_metric: str
    predicted_effect: str
    rationale: str
    suggested_control_actions: Tuple[str, ...] = ()
    required_context_replay: Tuple[str, ...] = ()
    required_action_args: Tuple[Dict[str, Any], ...] | None = None
    unlock_target_actions: Tuple[str, ...] = ()
    expected_signal_type: str = (
        "target_action_changes_local_patch_more_than_control"
    )
    test_hint: str = ""
    priority_hint: float = 0.0
    raw_status: str = ""
    raw_support: int | None = None
    raw_truth_status: str = ""
    raw_revision_performed: bool | None = None
    world_model_uncertainty: float | None = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "source_request_id": self.source_request_id,
            "game_id": self.game_id,
            "frontier_context_id": self.frontier_context_id,
            "frontier_reason": self.frontier_reason,
            "frontier_step": self.frontier_step,
            "hypothesis_family": self.hypothesis_family,
            "candidate_action": self.candidate_action,
            "predicted_metric": self.predicted_metric,
            "predicted_effect": self.predicted_effect,
            "rationale": self.rationale,
            "suggested_control_actions": list(self.suggested_control_actions),
            "required_context_replay": list(self.required_context_replay),
            "required_action_args": _list_of_dicts_or_none(self.required_action_args),
            "unlock_target_actions": list(self.unlock_target_actions),
            "expected_signal_type": self.expected_signal_type,
            "test_hint": self.test_hint,
            "priority_hint": float(self.priority_hint),
            "raw_status": self.raw_status,
            "raw_support": self.raw_support,
            "raw_truth_status": self.raw_truth_status,
            "raw_revision_performed": self.raw_revision_performed,
            "world_model_uncertainty": self.world_model_uncertainty,
            "raw_payload": dict(self.raw_payload),
        }


@dataclass(frozen=True)
class RejectedProposal:
    proposal_id: str
    source: str
    source_request_id: str
    reason: str
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalContextSummary:
    selected_signal: float = 0.0
    fallback_action: str = ""
    fallback_progress: bool = False
    known_consumed_skill: str = ""
    allowed_actions: Tuple[str, ...] = ()
    available_metrics: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_signal": float(self.selected_signal),
            "fallback_action": self.fallback_action,
            "fallback_progress": self.fallback_progress,
            "known_consumed_skill": self.known_consumed_skill,
            "allowed_actions": list(self.allowed_actions),
            "available_metrics": list(self.available_metrics),
        }


def to_plain_data(value: Any) -> Any:
    """Return JSON-friendly data for dataclasses, mappings and sequences."""
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def m3_request_from_mapping(
    data: Mapping[str, Any],
) -> M3CandidateExperimentRequest:
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


def _list_of_dicts_or_none(
    values: Sequence[Mapping[str, Any]] | None,
) -> list[Dict[str, Any]] | None:
    if values is None:
        return None
    return [dict(item) for item in values]


def _tuple_dicts_or_none(value: Any) -> Tuple[Dict[str, Any], ...] | None:
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
