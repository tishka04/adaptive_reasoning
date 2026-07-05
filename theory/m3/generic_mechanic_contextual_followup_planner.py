"""M3.G0.4 planner for contextual generic mechanic follow-ups."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .generic_mechanic_evidence_consolidation import (
    DEFAULT_GENERIC_MECHANIC_EVIDENCE_CONSOLIDATION_OUTPUT_PATH,
    DUPLICATE_EVIDENCE_ONLY,
    LOW_RESET_ONLY,
    NEUTRAL_CANDIDATE_ONLY,
)
from .generic_mechanic_experiment_planner import (
    DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "generic_mechanic_contextual_followup_requests.json"
)
READY_FOR_M3_GENERIC_CONTEXTUAL_FOLLOWUP = "READY_FOR_M3_GENERIC_CONTEXTUAL_FOLLOWUP"
SCHEMA_VERSION = "m3.generic_mechanic_contextual_followups.v1"
DEFAULT_MAX_CONTEXTUAL_FOLLOWUP_REQUESTS = 16
MULTI_PREFIX_CONTEXTS = "MULTI_PREFIX_CONTEXTS"

ACTOR_PERSISTENCE_OUTSIDE_RESET = "actor_persistence_outside_reset"
ACTION_EFFECT_STABILITY = "action_effect_stability"
RELATION_CHANGE_RECURRENCE = "relation_change_recurrence"
DYNAMIC_INVARIANT_TEMPORAL = "dynamic_invariant_temporal"
EXOGENOUS_MOTION_RECURRENCE = "exogenous_motion_recurrence"

DEFAULT_PREFIXES = (("ACTION3",), ("ACTION4",), ("ACTION6",))
DEFAULT_TEMPORAL_SEQUENCES = (
    ("ACTION3", "ACTION4", "ACTION6"),
    ("ACTION6", "ACTION3", "ACTION4"),
    ("ACTION4", "ACTION4", "ACTION3"),
)


@dataclass(frozen=True)
class GenericContextualFollowupRequest:
    """Candidate-only request meant to escape the reset-only context."""

    request_id: str
    source_consolidation_id: str
    source_request_id: str
    source_hypothesis_id: str
    source_candidate_status: str
    source_mechanic_family: str
    followup_family: str
    game_id: str
    context_replay: Tuple[str, ...]
    target_action: str | None
    control_actions: Tuple[str, ...]
    temporal_action_sequences: Tuple[Tuple[str, ...], ...]
    target_entity: str | None = None
    candidate_role: str | None = None
    source_entity: str | None = None
    relation_target_entity: str | None = None
    predicted_effect_family: str | None = None
    relation_delta_type: str | None = None
    invariant_family: str | None = None
    invariant_id: str | None = None
    remaining_semantics_unknown: bool | None = None
    metrics: Tuple[str, ...] = ()
    expected_signal: str = ""
    falsification_hint: str = ""
    planning_rationale: str = ""
    context_diversity_goal: str = "increase_context_diversity"
    expected_context_diversity_gain: str = "non_reset_prefix"
    status: str = READY_FOR_M3_GENERIC_CONTEXTUAL_FOLLOWUP
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    execution_performed: bool = False
    revision_performed: bool = False
    wrong_confirmations: int = 0
    duplicate_evidence_promoted: bool = False
    followup_request_counted_as_support: bool = False
    followup_request_counted_as_confirmation: bool = False
    semantic_interpretation_counted_as_confirmation: bool = False
    a32_remains_only_verdict_location: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_consolidation_id": self.source_consolidation_id,
            "source_request_id": self.source_request_id,
            "source_hypothesis_id": self.source_hypothesis_id,
            "source_candidate_status": self.source_candidate_status,
            "source_mechanic_family": self.source_mechanic_family,
            "followup_family": self.followup_family,
            "game_id": self.game_id,
            "context_replay": list(self.context_replay),
            "context_replay_args": None,
            "target_action": self.target_action,
            "control_actions": list(self.control_actions),
            "temporal_action_sequences": [
                list(sequence) for sequence in self.temporal_action_sequences
            ],
            "target_entity": self.target_entity,
            "candidate_role": self.candidate_role,
            "source_entity": self.source_entity,
            "relation_target_entity": self.relation_target_entity,
            "predicted_effect_family": self.predicted_effect_family,
            "relation_delta_type": self.relation_delta_type,
            "invariant_family": self.invariant_family,
            "invariant_id": self.invariant_id,
            "remaining_semantics_unknown": self.remaining_semantics_unknown,
            "metrics": list(self.metrics),
            "expected_signal": self.expected_signal,
            "falsification_hint": self.falsification_hint,
            "planning_rationale": self.planning_rationale,
            "context_diversity_goal": self.context_diversity_goal,
            "expected_context_diversity_gain": self.expected_context_diversity_gain,
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "execution_performed": self.execution_performed,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "duplicate_evidence_promoted": self.duplicate_evidence_promoted,
            "followup_request_counted_as_support": (
                self.followup_request_counted_as_support
            ),
            "followup_request_counted_as_confirmation": (
                self.followup_request_counted_as_confirmation
            ),
            "semantic_interpretation_counted_as_confirmation": (
                self.semantic_interpretation_counted_as_confirmation
            ),
            "a32_remains_only_verdict_location": (
                self.a32_remains_only_verdict_location
            ),
        }


def run_generic_mechanic_contextual_followup_planning(
    *,
    evidence_consolidation_path: str | Path = (
        DEFAULT_GENERIC_MECHANIC_EVIDENCE_CONSOLIDATION_OUTPUT_PATH
    ),
    generic_requests_path: str | Path | None = None,
    max_followup_requests: int = DEFAULT_MAX_CONTEXTUAL_FOLLOWUP_REQUESTS,
) -> Dict[str, Any]:
    consolidation_payload = _load_json(evidence_consolidation_path)
    validate_evidence_consolidation_source(consolidation_payload)
    request_path = resolve_generic_requests_path(
        consolidation_payload,
        generic_requests_path=generic_requests_path,
    )
    request_payload = _load_json(request_path)
    requests_by_id = {
        str(row.get("request_id", "")): dict(row)
        for row in request_payload.get("generic_mechanic_experiment_requests", []) or []
    }
    observed_actions = observed_actions_from_requests(request_payload)
    consolidation_rows = [
        dict(row)
        for row in consolidation_payload.get("hypothesis_consolidations", []) or []
        if bool(row.get("requires_contextual_followup"))
    ]
    all_requests = build_contextual_followup_requests(
        consolidation_rows=consolidation_rows,
        requests_by_id=requests_by_id,
        observed_actions=observed_actions,
    )
    limited_requests = tuple(all_requests[: max(0, int(max_followup_requests))])
    for request in limited_requests:
        validate_contextual_followup_request(
            request.to_dict(),
            observed_actions=observed_actions,
        )
    return {
        "config": {
            "evidence_consolidation_path": str(evidence_consolidation_path),
            "generic_requests_path": str(request_path),
            "schema_version": SCHEMA_VERSION,
            "inputs_read": ["M3.G0.3", "M3.G0.1"],
            "artifacts_not_modified": ["M1", "M2", "M3.G0.2", "A32", "A33"],
            "execution_performed": False,
            "max_followup_requests": int(max_followup_requests),
            "observed_actions": list(observed_actions),
            "goal": "increase_context_diversity",
            "target_context_diversity": MULTI_PREFIX_CONTEXTS,
        },
        "summary": summarize_contextual_followup_requests(
            consolidation_payload=consolidation_payload,
            all_requests=all_requests,
            emitted_requests=limited_requests,
            observed_actions=observed_actions,
        ),
        "generic_contextual_followup_requests": [
            request.to_dict() for request in limited_requests
        ],
        "truncated_contextual_followup_requests": [
            request.to_dict() for request in all_requests[len(limited_requests) :]
        ],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "duplicate_evidence_promoted": False,
        "followup_request_counted_as_support": False,
        "followup_request_counted_as_confirmation": False,
        "semantic_interpretation_counted_as_confirmation": False,
        "a32_remains_only_verdict_location": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def build_contextual_followup_requests(
    *,
    consolidation_rows: Sequence[Mapping[str, Any]],
    requests_by_id: Mapping[str, Mapping[str, Any]],
    observed_actions: Sequence[str],
) -> Tuple[GenericContextualFollowupRequest, ...]:
    requests: list[GenericContextualFollowupRequest] = []
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in consolidation_rows:
        by_family[str(row.get("source_mechanic_family", ""))].append(row)

    requests.extend(
        actor_persistence_requests(
            by_family.get("entity_role", []),
            requests_by_id=requests_by_id,
            observed_actions=observed_actions,
            sequence_start=len(requests) + 1,
        )
    )
    requests.extend(
        action_effect_stability_requests(
            by_family.get("action_effect", []),
            requests_by_id=requests_by_id,
            observed_actions=observed_actions,
            sequence_start=len(requests) + 1,
        )
    )
    requests.extend(
        relation_recurrence_requests(
            by_family.get("relation_change", []),
            requests_by_id=requests_by_id,
            observed_actions=observed_actions,
            sequence_start=len(requests) + 1,
        )
    )
    requests.extend(
        dynamic_invariant_contextual_requests(
            by_family.get("dynamic_invariant", []),
            requests_by_id=requests_by_id,
            observed_actions=observed_actions,
            sequence_start=len(requests) + 1,
        )
    )
    return tuple(requests)


def actor_persistence_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    requests_by_id: Mapping[str, Mapping[str, Any]],
    observed_actions: Sequence[str],
    sequence_start: int,
) -> Tuple[GenericContextualFollowupRequest, ...]:
    actor_rows = [
        row
        for row in rows
        if request_for_row(row, requests_by_id).get("candidate_role") == "controllable_actor"
    ]
    if not actor_rows:
        return ()
    row = actor_rows[0]
    source = request_for_row(row, requests_by_id)
    followups: list[GenericContextualFollowupRequest] = []
    for offset, prefix in enumerate(DEFAULT_PREFIXES):
        prefix_action = prefix[0]
        target, controls = target_and_controls_after_prefix(prefix_action, observed_actions)
        if target is None:
            continue
        followups.append(
            base_followup_request(
                row,
                source,
                sequence_index=sequence_start + offset,
                followup_family=ACTOR_PERSISTENCE_OUTSIDE_RESET,
                context_replay=prefix,
                target_action=target,
                control_actions=controls,
                metrics=tuple(source.get("metrics", []) or ()),
                expected_signal=(
                    "candidate_actor_remains_traceable_and_action_affected_after_prefix"
                ),
                falsification_hint=(
                    "actor candidate weakens if it cannot be tracked or action effects "
                    "disappear outside reset"
                ),
                planning_rationale="Test controllable actor candidate after non-reset prefixes.",
            )
        )
    return tuple(followups)


def action_effect_stability_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    requests_by_id: Mapping[str, Mapping[str, Any]],
    observed_actions: Sequence[str],
    sequence_start: int,
) -> Tuple[GenericContextualFollowupRequest, ...]:
    requests: list[GenericContextualFollowupRequest] = []
    selected = [
        row
        for row in rows
        if str(row.get("candidate_status", "")) == DUPLICATE_EVIDENCE_ONLY
    ][:4]
    for index, row in enumerate(selected):
        source = request_for_row(row, requests_by_id)
        target_action = str(source.get("target_action", ""))
        prefix = alternate_prefix_for_target(target_action, observed_actions)
        if not prefix:
            continue
        controls = tuple(
            action
            for action in observed_actions
            if action != target_action and action != prefix[0]
        ) or tuple(action for action in observed_actions if action != target_action)
        requests.append(
            base_followup_request(
                row,
                source,
                sequence_index=sequence_start + len(requests),
                followup_family=ACTION_EFFECT_STABILITY,
                context_replay=prefix,
                target_action=target_action,
                control_actions=controls,
                metrics=tuple(source.get("metrics", []) or ()),
                expected_signal=(
                    f"{target_action}_effect_family_stable_after_{prefix[0]}"
                ),
                falsification_hint=(
                    "effect candidate weakens if it appears only from reset and "
                    "not after a different observed action prefix"
                ),
                planning_rationale="Test action-effect stability outside reset.",
            )
        )
    return tuple(requests)


def relation_recurrence_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    requests_by_id: Mapping[str, Mapping[str, Any]],
    observed_actions: Sequence[str],
    sequence_start: int,
) -> Tuple[GenericContextualFollowupRequest, ...]:
    requests: list[GenericContextualFollowupRequest] = []
    selected = [
        row
        for row in rows
        if str(row.get("candidate_status", "")) == DUPLICATE_EVIDENCE_ONLY
    ][:4]
    for row in selected:
        source = request_for_row(row, requests_by_id)
        target_action = str(source.get("target_action", ""))
        prefix = alternate_prefix_for_target(target_action, observed_actions)
        if not prefix:
            continue
        controls = tuple(
            action
            for action in observed_actions
            if action != target_action and action != prefix[0]
        ) or tuple(action for action in observed_actions if action != target_action)
        requests.append(
            base_followup_request(
                row,
                source,
                sequence_index=sequence_start + len(requests),
                followup_family=RELATION_CHANGE_RECURRENCE,
                context_replay=prefix,
                target_action=target_action,
                control_actions=controls,
                metrics=tuple(source.get("metrics", []) or ()),
                expected_signal=(
                    "relation_delta_recurs_for_actor_candidate_pair_outside_reset"
                ),
                falsification_hint=(
                    "relation-change candidate weakens if the same target action "
                    "does not reproduce the relation delta after prefix contexts"
                ),
                planning_rationale="Test relation-change recurrence in non-reset contexts.",
            )
        )
    return tuple(requests)


def dynamic_invariant_contextual_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    requests_by_id: Mapping[str, Mapping[str, Any]],
    observed_actions: Sequence[str],
    sequence_start: int,
) -> Tuple[GenericContextualFollowupRequest, ...]:
    requests: list[GenericContextualFollowupRequest] = []
    for row in rows:
        source = request_for_row(row, requests_by_id)
        invariant_family = str(source.get("invariant_family", ""))
        if invariant_family == "monotone_counter":
            family = DYNAMIC_INVARIANT_TEMPORAL
            expected = "invariant_changes_monotonically_across_multi_action_sequences"
        elif invariant_family == "exogenous_motion":
            family = EXOGENOUS_MOTION_RECURRENCE
            expected = "drift_candidate_recurs_across_different_action_sequences"
        else:
            continue
        requests.append(
            base_followup_request(
                row,
                source,
                sequence_index=sequence_start + len(requests),
                followup_family=family,
                context_replay=(),
                target_action=None,
                control_actions=(),
                temporal_action_sequences=temporal_sequences_for_observed_actions(
                    observed_actions
                ),
                metrics=tuple(source.get("metrics", []) or ()),
                expected_signal=expected,
                falsification_hint=(
                    "dynamic invariant weakens if the candidate variable is not "
                    "regular across multi-step action sequences"
                ),
                planning_rationale=(
                    "Test dynamic invariant candidate across multiple temporal prefixes "
                    "without assigning semantic meaning."
                ),
                expected_context_diversity_gain=MULTI_PREFIX_CONTEXTS,
            )
        )
    return tuple(requests)


def base_followup_request(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    sequence_index: int,
    followup_family: str,
    context_replay: Sequence[str],
    target_action: str | None,
    control_actions: Sequence[str],
    metrics: Sequence[str],
    expected_signal: str,
    falsification_hint: str,
    planning_rationale: str,
    temporal_action_sequences: Sequence[Sequence[str]] = (),
    expected_context_diversity_gain: str = "non_reset_prefix",
) -> GenericContextualFollowupRequest:
    return GenericContextualFollowupRequest(
        request_id=contextual_request_id(followup_family, sequence_index, source),
        source_consolidation_id=str(row.get("consolidation_id", "")),
        source_request_id=str(source.get("request_id", row.get("request_id", ""))),
        source_hypothesis_id=str(row.get("source_hypothesis_id", "")),
        source_candidate_status=str(row.get("candidate_status", "")),
        source_mechanic_family=str(row.get("source_mechanic_family", "")),
        followup_family=followup_family,
        game_id=str(source.get("game_id", "")),
        context_replay=tuple(str(action) for action in context_replay),
        target_action=target_action,
        control_actions=tuple(str(action) for action in control_actions),
        temporal_action_sequences=tuple(
            tuple(str(action) for action in sequence)
            for sequence in temporal_action_sequences
        ),
        target_entity=_optional_str(source.get("target_entity")),
        candidate_role=_optional_str(source.get("candidate_role")),
        source_entity=_optional_str(source.get("source_entity")),
        relation_target_entity=_optional_str(source.get("relation_target_entity")),
        predicted_effect_family=_optional_str(source.get("predicted_effect_family")),
        relation_delta_type=_optional_str(source.get("relation_delta_type")),
        invariant_family=_optional_str(source.get("invariant_family")),
        invariant_id=_optional_str(source.get("invariant_id")),
        remaining_semantics_unknown=(
            bool(source.get("remaining_semantics_unknown"))
            if source.get("remaining_semantics_unknown") is not None
            else None
        ),
        metrics=tuple(str(metric) for metric in metrics),
        expected_signal=expected_signal,
        falsification_hint=falsification_hint,
        planning_rationale=planning_rationale,
        expected_context_diversity_gain=expected_context_diversity_gain,
    )


def summarize_contextual_followup_requests(
    *,
    consolidation_payload: Mapping[str, Any],
    all_requests: Sequence[GenericContextualFollowupRequest],
    emitted_requests: Sequence[GenericContextualFollowupRequest],
    observed_actions: Sequence[str],
) -> Dict[str, Any]:
    source_summary = dict(consolidation_payload.get("summary", {}) or {})
    families = Counter(request.followup_family for request in emitted_requests)
    source_families = Counter(request.source_mechanic_family for request in emitted_requests)
    prefixes = {
        tuple(request.context_replay)
        for request in emitted_requests
        if request.context_replay
    }
    temporal_sequences = {
        tuple(sequence)
        for request in emitted_requests
        for sequence in request.temporal_action_sequences
    }
    return {
        "source_hypothesis_consolidations": int(
            source_summary.get("hypothesis_consolidations", 0) or 0
        ),
        "source_duplicate_evidence_hypotheses": int(
            (source_summary.get("candidate_status_counts", {}) or {}).get(
                DUPLICATE_EVIDENCE_ONLY,
                0,
            )
            or 0
        ),
        "source_context_diversity_assessment": str(
            source_summary.get("context_diversity_assessment", "")
        ),
        "contextual_followup_requests_generated": len(emitted_requests),
        "all_contextual_followup_requests_before_truncation": len(all_requests),
        "truncated_contextual_followup_requests": max(
            0,
            len(all_requests) - len(emitted_requests),
        ),
        "requests_by_followup_family": dict(sorted(families.items())),
        "requests_by_source_mechanic_family": dict(sorted(source_families.items())),
        "observed_actions": list(observed_actions),
        "non_reset_prefixes_planned": [list(prefix) for prefix in sorted(prefixes)],
        "temporal_action_sequences_planned": [
            list(sequence) for sequence in sorted(temporal_sequences)
        ],
        "expected_independent_contexts_after_execution": max(
            1,
            len(prefixes) + len(temporal_sequences),
        ),
        "target_context_diversity": MULTI_PREFIX_CONTEXTS,
        "goal": "increase_context_diversity",
        "duplicate_evidence_promoted": False,
        "execution_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def validate_contextual_followup_request(
    request: Mapping[str, Any],
    *,
    observed_actions: Sequence[str],
) -> None:
    if int(request.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.4 support must remain 0")
    if request.get("revision_status") != "CANDIDATE_ONLY":
        raise ValueError("M3.G0.4 revision status must be candidate-only")
    if request.get("truth_status") != M3_REFINEMENT_TRUTH_STATUS:
        raise ValueError("M3.G0.4 truth status must remain NOT_EVALUATED_BY_M3")
    if bool(request.get("execution_performed", False)):
        raise ValueError("M3.G0.4 cannot execute followups")
    if bool(request.get("duplicate_evidence_promoted", False)):
        raise ValueError("M3.G0.4 cannot promote duplicate evidence")
    if bool(request.get("followup_request_counted_as_support", False)):
        raise ValueError("followup request cannot count as support")
    if bool(request.get("followup_request_counted_as_confirmation", False)):
        raise ValueError("followup request cannot count as confirmation")
    actions = list(request.get("context_replay", []) or [])
    if request.get("target_action"):
        actions.append(str(request.get("target_action")))
    actions.extend(str(action) for action in request.get("control_actions", []) or [])
    for sequence in request.get("temporal_action_sequences", []) or []:
        actions.extend(str(action) for action in sequence)
    invalid = [action for action in actions if action not in observed_actions]
    if invalid:
        raise ValueError(f"unobserved action in contextual followup: {invalid[0]}")


def validate_evidence_consolidation_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G0.3 support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.3 summary support must remain 0")
    if bool(payload.get("revision_performed", False)):
        raise ValueError("M3.G0.3 must not perform revision")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("M3.G0.3 cannot write A32/A33")
    if bool(payload.get("support_events_counted_as_scientific_support", False)):
        raise ValueError("M3.G0.3 support events cannot be scientific support")
    if str(summary.get("context_diversity_assessment", "")) != LOW_RESET_ONLY:
        raise ValueError("M3.G0.4 expects LOW_RESET_ONLY source consolidation")


def resolve_generic_requests_path(
    consolidation_payload: Mapping[str, Any],
    *,
    generic_requests_path: str | Path | None,
) -> Path:
    if generic_requests_path is not None:
        return Path(generic_requests_path)
    generic_results_path = str(
        (consolidation_payload.get("config", {}) or {}).get("generic_results_path", "")
    )
    if generic_results_path and Path(generic_results_path).exists():
        results_payload = _load_json(generic_results_path)
        request_path = str(
            (results_payload.get("config", {}) or {}).get("generic_requests_path", "")
        )
        if request_path:
            return Path(request_path)
    return DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH


def observed_actions_from_requests(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    config_actions = [
        str(action)
        for action in (payload.get("config", {}) or {}).get("observed_actions", []) or []
    ]
    actions: list[str] = []
    actions.extend(config_actions)
    for request in payload.get("generic_mechanic_experiment_requests", []) or []:
        if request.get("target_action"):
            actions.append(str(request.get("target_action")))
        actions.extend(str(action) for action in request.get("control_actions", []) or [])
        actions.extend(str(action) for action in request.get("conditions", []) or [])
    return tuple(dict.fromkeys(action for action in actions if action))


def request_for_row(
    row: Mapping[str, Any],
    requests_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    return requests_by_id.get(str(row.get("request_id", "")), {})


def target_and_controls_after_prefix(
    prefix_action: str,
    observed_actions: Sequence[str],
) -> Tuple[str | None, Tuple[str, ...]]:
    candidates = [action for action in observed_actions if action != prefix_action]
    if not candidates:
        return None, ()
    target = candidates[0]
    controls = tuple(candidates[1:] or [prefix_action])
    return target, controls


def alternate_prefix_for_target(
    target_action: str,
    observed_actions: Sequence[str],
) -> Tuple[str, ...]:
    for action in observed_actions:
        if action != target_action:
            return (action,)
    return ()


def temporal_sequences_for_observed_actions(
    observed_actions: Sequence[str],
) -> Tuple[Tuple[str, ...], ...]:
    allowed = set(observed_actions)
    sequences = [
        tuple(action for action in sequence if action in allowed)
        for sequence in DEFAULT_TEMPORAL_SEQUENCES
    ]
    return tuple(sequence for sequence in sequences if len(sequence) >= 2)


def contextual_request_id(
    followup_family: str,
    sequence_index: int,
    source: Mapping[str, Any],
) -> str:
    source_id = str(source.get("request_id", "unknown")).replace("::", "_")
    return f"m3g0_4::{followup_family}::{sequence_index:03d}::{source_id}"


def write_generic_mechanic_contextual_followup_requests(
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH,
) -> None:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan generic contextual M3.G0 follow-ups from M3.G0.3."
    )
    parser.add_argument(
        "--evidence-consolidation",
        default=str(DEFAULT_GENERIC_MECHANIC_EVIDENCE_CONSOLIDATION_OUTPUT_PATH),
        help="Path to diagnostics/m3/generic_mechanic_evidence_consolidation.json.",
    )
    parser.add_argument(
        "--generic-requests",
        default=None,
        help="Optional M3.G0.1 request path.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH),
        help="Output path for contextual follow-up requests.",
    )
    parser.add_argument(
        "--max-followup-requests",
        type=int,
        default=DEFAULT_MAX_CONTEXTUAL_FOLLOWUP_REQUESTS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_generic_mechanic_contextual_followup_planning(
        evidence_consolidation_path=args.evidence_consolidation,
        generic_requests_path=args.generic_requests,
        max_followup_requests=args.max_followup_requests,
    )
    write_generic_mechanic_contextual_followup_requests(payload, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "contextual_followup_requests_generated": payload["summary"][
                    "contextual_followup_requests_generated"
                ],
                "target_context_diversity": payload["summary"][
                    "target_context_diversity"
                ],
                "execution_performed": False,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
