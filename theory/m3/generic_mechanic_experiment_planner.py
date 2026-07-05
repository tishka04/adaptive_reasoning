"""M3.G0.1 planner for generic mechanic experiments from M1.G0."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from theory.m1.general_mechanic_abstraction import (
    DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH,
)

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "generic_mechanic_experiment_requests.json"
)
READY_FOR_M3_GENERIC_EXPERIMENT = "READY_FOR_M3_GENERIC_EXPERIMENT"
SCHEMA_VERSION = "m3.generic_mechanic_experiment_requests.v1"
DEFAULT_MAX_REQUESTS = 16
DEFAULT_TARGET_QUOTAS = {
    "entity_role": 2,
    "action_effect": 4,
    "relation_change": 6,
    "dynamic_invariant": 2,
}
DEFAULT_MAX_PER_FAMILY = 6
DEFAULT_MAX_SAME_ENTITY_PAIR = 2

ENTITY_ROLE_METRICS = (
    "entity_centroid_delta",
    "entity_bbox_delta",
    "relation_graph_delta",
    "terminal_state",
)
ACTION_EFFECT_METRICS = (
    "affected_entity_count",
    "centroid_delta",
    "shape_change",
    "created_deleted_entities",
    "relation_graph_delta",
    "terminal_state",
)
RELATION_CHANGE_METRICS = (
    "distance_before_after",
    "contact_created",
    "near_relation_created",
    "relation_graph_delta",
    "terminal_state",
    "level_completed",
)
DYNAMIC_INVARIANT_METRICS = (
    "value_delta_per_action",
    "monotonicity",
    "action_correlation",
    "terminal_correlation",
)


@dataclass(frozen=True)
class GenericMechanicExperimentRequest:
    """Candidate-only request for a future generic M3.G0 controlled test."""

    request_id: str
    source_hypothesis_id: str
    source_mechanic_family: str
    test_type: str
    game_id: str
    hypothesis_tested: str
    priority_score: float
    metrics: Tuple[str, ...]
    conditions: Tuple[str, ...] = ()
    target_action: str | None = None
    control_actions: Tuple[str, ...] = ()
    target_entity: str | None = None
    source_entity: str | None = None
    relation_target_entity: str | None = None
    candidate_role: str | None = None
    predicted_effect_family: str | None = None
    relation: str | None = None
    relation_delta_type: str | None = None
    invariant_family: str | None = None
    invariant_id: str | None = None
    affected_entities: Tuple[str, ...] = ()
    latent_variables: Tuple[str, ...] = ()
    remaining_semantics_unknown: bool | None = None
    expected_signal: str = ""
    falsification_hint: str = ""
    planning_rationale: str = ""
    status: str = READY_FOR_M3_GENERIC_EXPERIMENT
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    execution_performed: bool = False
    revision_performed: bool = False
    wrong_confirmations: int = 0
    source_confidence_counted_as_support: bool = False
    priority_score_counted_as_support: bool = False
    m1_support_counted_as_m3_support: bool = False
    generic_request_counted_as_confirmation: bool = False
    a32_remains_only_verdict_location: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_hypothesis_id": self.source_hypothesis_id,
            "source_mechanic_family": self.source_mechanic_family,
            "test_type": self.test_type,
            "game_id": self.game_id,
            "hypothesis_tested": self.hypothesis_tested,
            "priority_score": round(float(self.priority_score), 6),
            "priority_score_counted_as_support": (
                self.priority_score_counted_as_support
            ),
            "conditions": list(self.conditions),
            "target_action": self.target_action,
            "control_actions": list(self.control_actions),
            "target_entity": self.target_entity,
            "source_entity": self.source_entity,
            "relation_target_entity": self.relation_target_entity,
            "candidate_role": self.candidate_role,
            "predicted_effect_family": self.predicted_effect_family,
            "relation": self.relation,
            "relation_delta_type": self.relation_delta_type,
            "invariant_family": self.invariant_family,
            "invariant_id": self.invariant_id,
            "affected_entities": list(self.affected_entities),
            "latent_variables": list(self.latent_variables),
            "remaining_semantics_unknown": self.remaining_semantics_unknown,
            "metrics": list(self.metrics),
            "expected_signal": self.expected_signal,
            "falsification_hint": self.falsification_hint,
            "planning_rationale": self.planning_rationale,
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "execution_performed": self.execution_performed,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "source_confidence_counted_as_support": (
                self.source_confidence_counted_as_support
            ),
            "m1_support_counted_as_m3_support": self.m1_support_counted_as_m3_support,
            "generic_request_counted_as_confirmation": (
                self.generic_request_counted_as_confirmation
            ),
            "a32_remains_only_verdict_location": (
                self.a32_remains_only_verdict_location
            ),
        }


def run_generic_mechanic_experiment_planning(
    *,
    general_mechanic_candidates_path: str | Path = (
        DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH
    ),
    max_requests: int = DEFAULT_MAX_REQUESTS,
    target_quotas: Mapping[str, int] | None = None,
    max_per_family: int = DEFAULT_MAX_PER_FAMILY,
    max_same_entity_pair: int = DEFAULT_MAX_SAME_ENTITY_PAIR,
) -> Dict[str, Any]:
    payload = _load_json(general_mechanic_candidates_path)
    validate_m1_general_mechanic_source(payload)
    quotas = dict(target_quotas or DEFAULT_TARGET_QUOTAS)
    observed_actions = observed_actions_from_payload(payload)
    role_index = role_index_from_payload(payload)
    dynamic_invariant_index = dynamic_invariant_index_from_payload(payload)
    candidates = candidate_rows_from_payload(payload)

    selected = select_generic_mechanic_requests(
        candidates=candidates,
        payload=payload,
        role_index=role_index,
        dynamic_invariant_index=dynamic_invariant_index,
        observed_actions=observed_actions,
        max_requests=max_requests,
        target_quotas=quotas,
        max_per_family=max_per_family,
        max_same_entity_pair=max_same_entity_pair,
    )
    for request in selected:
        validate_generic_mechanic_request(
            request.to_dict(),
            observed_actions=observed_actions,
        )

    request_rows = [request.to_dict() for request in selected]
    return {
        "config": {
            "general_mechanic_candidates_path": str(general_mechanic_candidates_path),
            "schema_version": SCHEMA_VERSION,
            "inputs_read": ["M1.G0"],
            "artifacts_not_modified": ["M1", "M2", "M3 execution", "A32", "A33"],
            "execution_performed": False,
            "max_requests": int(max_requests),
            "target_quotas": {key: int(value) for key, value in quotas.items()},
            "max_per_family": int(max_per_family),
            "max_same_entity_pair": int(max_same_entity_pair),
            "observed_actions": list(observed_actions),
            "selection_policy": "deterministic_priority_balanced_generic_mechanic_tests",
        },
        "summary": summarize_generic_mechanic_requests(
            payload=payload,
            requests=request_rows,
            observed_actions=observed_actions,
            max_requests=max_requests,
        ),
        "generic_mechanic_experiment_requests": request_rows,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "execution_performed": False,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "source_confidence_counted_as_support": False,
        "priority_score_counted_as_support": False,
        "m1_support_counted_as_m3_support": False,
        "generic_request_counted_as_confirmation": False,
        "a32_remains_only_verdict_location": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def select_generic_mechanic_requests(
    *,
    candidates: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    role_index: Mapping[str, Mapping[str, float]],
    dynamic_invariant_index: Mapping[Tuple[str | None, str], Mapping[str, Any]],
    observed_actions: Sequence[str],
    max_requests: int,
    target_quotas: Mapping[str, int],
    max_per_family: int,
    max_same_entity_pair: int,
) -> Tuple[GenericMechanicExperimentRequest, ...]:
    if int(max_requests) <= 0:
        return ()
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        family = str(row.get("mechanic_family", ""))
        if family == "action_effect" and not _row_uses_observed_action(row, observed_actions):
            continue
        if family == "relation_change" and not _row_uses_observed_action(row, observed_actions):
            continue
        by_family[family].append(row)

    selected: list[GenericMechanicExperimentRequest] = []
    seen_ids: set[str] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    per_family_counts: Counter[str] = Counter()
    selected_entity_role_entities: set[str] = set()
    selected_invariant_families: set[str] = set()
    for family in ("entity_role", "action_effect", "relation_change", "dynamic_invariant"):
        quota = min(
            int(target_quotas.get(family, 0)),
            int(max_per_family),
            int(max_requests) - len(selected),
        )
        if quota <= 0:
            continue
        rows = sorted(
            by_family.get(family, []),
            key=lambda row: _candidate_sort_key(
                row,
                family=family,
                role_index=role_index,
                observed_actions=observed_actions,
                dynamic_invariant_index=dynamic_invariant_index,
            ),
        )
        for row in rows:
            if len(selected) >= int(max_requests) or per_family_counts[family] >= quota:
                break
            hypothesis_id = str(row.get("mechanic_hypothesis_id", ""))
            if hypothesis_id in seen_ids:
                continue
            if family == "entity_role":
                entity = _first(row.get("entities"))
                if entity and entity in selected_entity_role_entities:
                    continue
            if family == "dynamic_invariant":
                invariant_family = _parse_tail(hypothesis_id, 1)
                if (
                    invariant_family
                    and invariant_family in selected_invariant_families
                ):
                    continue
            if family == "relation_change":
                pair = _entity_pair(row)
                if pair and pair_counts[pair] >= int(max_same_entity_pair):
                    continue
            request = build_generic_mechanic_request(
                row,
                game_id=str((payload.get("summary", {}) or {}).get("game_id", "")),
                role_index=role_index,
                dynamic_invariant_index=dynamic_invariant_index,
                observed_actions=observed_actions,
                sequence_index=len(selected) + 1,
            )
            if request is None:
                continue
            selected.append(request)
            seen_ids.add(hypothesis_id)
            per_family_counts[family] += 1
            if family == "entity_role" and request.target_entity:
                selected_entity_role_entities.add(request.target_entity)
            if family == "dynamic_invariant" and request.invariant_family:
                selected_invariant_families.add(request.invariant_family)
            if family == "relation_change":
                pair = _entity_pair(row)
                if pair:
                    pair_counts[pair] += 1

    return tuple(selected[: int(max_requests)])


def build_generic_mechanic_request(
    row: Mapping[str, Any],
    *,
    game_id: str,
    role_index: Mapping[str, Mapping[str, float]],
    dynamic_invariant_index: Mapping[Tuple[str | None, str], Mapping[str, Any]],
    observed_actions: Sequence[str],
    sequence_index: int,
) -> GenericMechanicExperimentRequest | None:
    family = str(row.get("mechanic_family", ""))
    if family == "entity_role":
        return build_entity_role_request(
            row,
            game_id=game_id,
            role_index=role_index,
            observed_actions=observed_actions,
            sequence_index=sequence_index,
        )
    if family == "action_effect":
        return build_action_effect_request(
            row,
            game_id=game_id,
            role_index=role_index,
            observed_actions=observed_actions,
            sequence_index=sequence_index,
        )
    if family == "relation_change":
        return build_relation_change_request(
            row,
            game_id=game_id,
            role_index=role_index,
            observed_actions=observed_actions,
            sequence_index=sequence_index,
        )
    if family == "dynamic_invariant":
        return build_dynamic_invariant_request(
            row,
            game_id=game_id,
            dynamic_invariant_index=dynamic_invariant_index,
            observed_actions=observed_actions,
            sequence_index=sequence_index,
        )
    return None


def build_entity_role_request(
    row: Mapping[str, Any],
    *,
    game_id: str,
    role_index: Mapping[str, Mapping[str, float]],
    observed_actions: Sequence[str],
    sequence_index: int,
) -> GenericMechanicExperimentRequest:
    entity = _first(row.get("entities"))
    candidate_role = _parse_tail(str(row.get("mechanic_hypothesis_id", "")), 1)
    if not candidate_role and entity:
        candidate_role = _top_role_for_entity(entity, role_index)
    test_type = (
        "actor_controllability_probe"
        if candidate_role == "controllable_actor"
        else "entity_role_stability_probe"
    )
    metrics = ENTITY_ROLE_METRICS
    if candidate_role == "timer_or_hud":
        metrics = (
            "edge_location_stability",
            "entity_size_sequence",
            "monotonicity",
            "action_correlation",
        )
    return GenericMechanicExperimentRequest(
        request_id=_request_id("entity_role", sequence_index, row),
        source_hypothesis_id=str(row.get("mechanic_hypothesis_id", "")),
        source_mechanic_family="entity_role",
        test_type=test_type,
        game_id=game_id,
        target_entity=entity,
        candidate_role=candidate_role,
        conditions=tuple(observed_actions),
        metrics=metrics,
        priority_score=priority_score_for_candidate(
            row,
            family="entity_role",
            role_index=role_index,
            observed_actions=observed_actions,
        ),
        hypothesis_tested=_predicted_effect(row),
        expected_signal="candidate_entity_behavior_separates_across_actions_or_time",
        falsification_hint=(
            "candidate role weakens if controlled actions do not differentially "
            "affect the entity or if the role-specific temporal pattern disappears"
        ),
        planning_rationale="High-priority role candidate selected for generic role probe.",
    )


def build_action_effect_request(
    row: Mapping[str, Any],
    *,
    game_id: str,
    role_index: Mapping[str, Mapping[str, float]],
    observed_actions: Sequence[str],
    sequence_index: int,
) -> GenericMechanicExperimentRequest | None:
    action = _first(row.get("actions"))
    if action not in observed_actions:
        return None
    effect_family = _parse_tail(str(row.get("mechanic_hypothesis_id", "")), 1)
    controls = tuple(action_name for action_name in observed_actions if action_name != action)
    return GenericMechanicExperimentRequest(
        request_id=_request_id("action_effect", sequence_index, row),
        source_hypothesis_id=str(row.get("mechanic_hypothesis_id", "")),
        source_mechanic_family="action_effect",
        test_type="action_effect_causality_probe",
        game_id=game_id,
        target_action=action,
        control_actions=controls,
        affected_entities=_prioritized_entities(
            row.get("entities", []),
            role_index=role_index,
            limit=8,
        ),
        predicted_effect_family=effect_family,
        latent_variables=tuple(str(value) for value in row.get("latent_variables", []) or []),
        metrics=ACTION_EFFECT_METRICS,
        priority_score=priority_score_for_candidate(
            row,
            family="action_effect",
            role_index=role_index,
            observed_actions=observed_actions,
        ),
        hypothesis_tested=_predicted_effect(row),
        expected_signal=f"{action}_effect_family_{effect_family}_exceeds_controls",
        falsification_hint=(
            "target action is not differentiated if controls produce the same "
            "entity/relation/latent deltas from the same state"
        ),
        planning_rationale="Observed action-effect candidate selected for causality probe.",
    )


def build_relation_change_request(
    row: Mapping[str, Any],
    *,
    game_id: str,
    role_index: Mapping[str, Mapping[str, float]],
    observed_actions: Sequence[str],
    sequence_index: int,
) -> GenericMechanicExperimentRequest | None:
    action = _first(row.get("actions"))
    if action not in observed_actions:
        return None
    entities = [str(value) for value in row.get("entities", []) or []]
    if len(entities) < 2:
        return None
    relation = _first(row.get("relations"))
    relation_delta_type = _parse_tail(str(row.get("mechanic_hypothesis_id", "")), 1)
    controls = tuple(action_name for action_name in observed_actions if action_name != action)
    return GenericMechanicExperimentRequest(
        request_id=_request_id("relation_change", sequence_index, row),
        source_hypothesis_id=str(row.get("mechanic_hypothesis_id", "")),
        source_mechanic_family="relation_change",
        test_type="relation_change_probe",
        game_id=game_id,
        target_action=action,
        control_actions=controls,
        source_entity=entities[0],
        relation_target_entity=entities[1],
        relation=relation,
        relation_delta_type=relation_delta_type,
        metrics=RELATION_CHANGE_METRICS,
        priority_score=priority_score_for_candidate(
            row,
            family="relation_change",
            role_index=role_index,
            observed_actions=observed_actions,
        ),
        hypothesis_tested=_predicted_effect(row),
        expected_signal=(
            f"{action}_produces_{relation_delta_type}_for_{relation}_more_than_controls"
        ),
        falsification_hint=(
            "relation-change candidate weakens if target and controls produce "
            "the same relation delta for the same entity pair"
        ),
        planning_rationale="Actor-adjacent relation delta selected for controlled relation probe.",
    )


def build_dynamic_invariant_request(
    row: Mapping[str, Any],
    *,
    game_id: str,
    dynamic_invariant_index: Mapping[Tuple[str | None, str], Mapping[str, Any]],
    observed_actions: Sequence[str],
    sequence_index: int,
) -> GenericMechanicExperimentRequest:
    entity = _first(row.get("entities"))
    invariant_family = _parse_tail(str(row.get("mechanic_hypothesis_id", "")), 1)
    invariant = dynamic_invariant_index.get((entity, invariant_family), {})
    evidence = dict(invariant.get("evidence", {}) or {})
    remaining_semantics_unknown = evidence.get("remaining_semantics_unknown")
    return GenericMechanicExperimentRequest(
        request_id=_request_id("dynamic_invariant", sequence_index, row),
        source_hypothesis_id=str(row.get("mechanic_hypothesis_id", "")),
        source_mechanic_family="dynamic_invariant",
        test_type="dynamic_invariant_probe",
        game_id=game_id,
        target_entity=entity,
        invariant_family=invariant_family,
        invariant_id=str(invariant.get("invariant_id", "")),
        affected_entities=tuple(
            str(value) for value in invariant.get("affected_entities", []) or []
        ),
        conditions=tuple(observed_actions),
        metrics=DYNAMIC_INVARIANT_METRICS,
        remaining_semantics_unknown=(
            bool(remaining_semantics_unknown)
            if remaining_semantics_unknown is not None
            else None
        ),
        priority_score=priority_score_for_candidate(
            row,
            family="dynamic_invariant",
            role_index={},
            observed_actions=observed_actions,
            dynamic_invariant_index=dynamic_invariant_index,
        ),
        hypothesis_tested=_predicted_effect(row),
        expected_signal=(
            f"{invariant_family}_changes_regularly_across_observed_actions"
        ),
        falsification_hint=(
            "invariant candidate weakens if its value is not regular across "
            "actions or does not remain stable in location/source"
        ),
        planning_rationale=(
            "Dynamic invariant candidate selected without assigning confirmed semantics."
        ),
    )


def priority_score_for_candidate(
    row: Mapping[str, Any],
    *,
    family: str,
    role_index: Mapping[str, Mapping[str, float]],
    observed_actions: Sequence[str],
    dynamic_invariant_index: Mapping[Tuple[str | None, str], Mapping[str, Any]] | None = None,
) -> float:
    score = float(row.get("confidence", 0.0) or 0.0)
    actions = [str(value) for value in row.get("actions", []) or []]
    entities = [str(value) for value in row.get("entities", []) or []]
    preconditions = {str(value) for value in row.get("preconditions", []) or []}
    hypothesis_id = str(row.get("mechanic_hypothesis_id", ""))
    candidate_role = _parse_tail(hypothesis_id, 1)
    if family == "entity_role":
        if candidate_role == "controllable_actor":
            score += 5.0
        elif candidate_role == "timer_or_hud":
            score += 4.0
        elif candidate_role == "target_candidate":
            score += 2.0
        elif candidate_role:
            score += 0.5
    if any(action in observed_actions for action in actions):
        score += 2.0
    if any(_entity_has_role(entity, "controllable_actor", role_index) for entity in entities):
        score += 3.0
    if any(_entity_has_role(entity, "timer_or_hud", role_index) for entity in entities):
        score += 2.0
    if "controllable_actor" in preconditions:
        score += 2.0
    if "target_candidate" in preconditions:
        score += 1.0
    if any(token in hypothesis_id for token in ("contact_created", "distance_decreases")):
        score += 2.0
    if any(token in hypothesis_id for token in ("near_relation_created", "containment_created")):
        score += 1.0
    if "::move_entity" in hypothesis_id or "::transform_entity" in hypothesis_id:
        score += 1.5
    if "::tick_latent" in hypothesis_id:
        score += 1.25
    if family == "dynamic_invariant":
        invariant_family = _parse_tail(hypothesis_id, 1)
        entity = _first(row.get("entities"))
        invariant = (dynamic_invariant_index or {}).get((entity, invariant_family), {})
        if invariant_family == "monotone_counter":
            score += 3.0
        elif invariant_family == "exogenous_motion":
            score += 5.0
        elif invariant_family == "irreversible_change":
            score -= 1.0
        if str(invariant.get("policy_relevance", "")):
            score += 1.0
        score += float(invariant.get("monotonicity_score", 0.0) or 0.0)
        score += float(invariant.get("action_correlation_score", 0.0) or 0.0)
    if "relation_created" in hypothesis_id or "relation_removed" in hypothesis_id:
        score -= 0.5
    return round(score, 6)


def _candidate_sort_key(
    row: Mapping[str, Any],
    *,
    family: str,
    role_index: Mapping[str, Mapping[str, float]],
    observed_actions: Sequence[str],
    dynamic_invariant_index: Mapping[Tuple[str | None, str], Mapping[str, Any]],
) -> Tuple[float, str]:
    score = priority_score_for_candidate(
        row,
        family=family,
        role_index=role_index,
        observed_actions=observed_actions,
        dynamic_invariant_index=dynamic_invariant_index,
    )
    return (-score, str(row.get("mechanic_hypothesis_id", "")))


def candidate_rows_from_payload(payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    return tuple(dict(row) for row in payload.get("mechanic_hypotheses", []) or [])


def role_index_from_payload(payload: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    index: Dict[str, Dict[str, float]] = {}
    for row in payload.get("role_hypothesis_ledger", []) or []:
        entity_id = str(row.get("entity_id", ""))
        roles: Dict[str, float] = {}
        for role_row in row.get("role_hypotheses", []) or []:
            roles[str(role_row.get("role", ""))] = float(role_row.get("score", 0.0) or 0.0)
        if entity_id:
            index[entity_id] = roles
    return index


def dynamic_invariant_index_from_payload(
    payload: Mapping[str, Any],
) -> Dict[Tuple[str | None, str], Mapping[str, Any]]:
    index: Dict[Tuple[str | None, str], Mapping[str, Any]] = {}
    for row in payload.get("dynamic_invariant_candidates", []) or []:
        entity_id = row.get("entity_id")
        family = str(row.get("invariant_family", ""))
        if family:
            index[(str(entity_id) if entity_id is not None else None, family)] = dict(row)
    return index


def observed_actions_from_payload(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    actions = [
        str(row.get("action", ""))
        for row in payload.get("action_effect_abstractions", []) or []
        if str(row.get("action", ""))
    ]
    return tuple(dict.fromkeys(actions))


def validate_m1_general_mechanic_source(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M1.G0 source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M1.G0 summary support must remain 0")
    if bool(payload.get("revision_performed", False)):
        raise ValueError("M1.G0 source must not perform revision")
    if int(payload.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("M1.G0 source contains wrong confirmations")
    if bool(payload.get("trace_support_counted_as_proof", False)):
        raise ValueError("M1.G0 source trace support cannot be proof")
    if bool(payload.get("prior_counted_as_proof", False)):
        raise ValueError("M1.G0 source prior cannot be proof")
    if summary and not bool(summary.get("ready_for_m3_g0", False)):
        raise ValueError("M1.G0 source is not ready for M3.G0")


def validate_generic_mechanic_request(
    request: Mapping[str, Any],
    *,
    observed_actions: Sequence[str],
) -> None:
    if int(request.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.1 support must remain 0")
    if request.get("revision_status") != "CANDIDATE_ONLY":
        raise ValueError("M3.G0.1 revision status must remain candidate-only")
    if request.get("truth_status") != M3_REFINEMENT_TRUTH_STATUS:
        raise ValueError("M3.G0.1 truth status must remain NOT_EVALUATED_BY_M3")
    if bool(request.get("execution_performed", False)):
        raise ValueError("M3.G0.1 planner cannot execute experiments")
    if bool(request.get("priority_score_counted_as_support", False)):
        raise ValueError("priority score cannot count as support")
    if bool(request.get("m1_support_counted_as_m3_support", False)):
        raise ValueError("M1 support cannot count as M3 support")
    if bool(request.get("generic_request_counted_as_confirmation", False)):
        raise ValueError("generic request cannot count as confirmation")
    action = request.get("target_action")
    if action is not None and str(action) not in observed_actions:
        raise ValueError(f"target action {action} was not observed")
    for control in request.get("control_actions", []) or []:
        if str(control) not in observed_actions:
            raise ValueError(f"control action {control} was not observed")
    for condition in request.get("conditions", []) or []:
        if str(condition) not in observed_actions:
            raise ValueError(f"condition action {condition} was not observed")


def summarize_generic_mechanic_requests(
    *,
    payload: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    observed_actions: Sequence[str],
    max_requests: int,
) -> Dict[str, Any]:
    source_summary = dict(payload.get("summary", {}) or {})
    family_counts = Counter(str(row.get("source_mechanic_family", "")) for row in requests)
    type_counts = Counter(str(row.get("test_type", "")) for row in requests)
    return {
        "source_mechanic_hypotheses": int(
            source_summary.get(
                "mechanic_hypotheses_generated",
                len(payload.get("mechanic_hypotheses", []) or []),
            )
            or 0
        ),
        "generic_experiment_requests_generated": len(requests),
        "requests_by_source_family": dict(sorted(family_counts.items())),
        "requests_by_test_type": dict(sorted(type_counts.items())),
        "max_requests": int(max_requests),
        "observed_actions": list(observed_actions),
        "only_observed_actions_used": True,
        "execution_performed": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "source_confidence_counted_as_support": False,
        "priority_score_counted_as_support": False,
        "m1_support_counted_as_m3_support": False,
        "generic_request_counted_as_confirmation": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def write_generic_mechanic_experiment_requests(
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH,
) -> None:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_uses_observed_action(
    row: Mapping[str, Any],
    observed_actions: Sequence[str],
) -> bool:
    actions = [str(value) for value in row.get("actions", []) or []]
    return any(action in observed_actions for action in actions)


def _entity_has_role(
    entity_id: str,
    role: str,
    role_index: Mapping[str, Mapping[str, float]],
) -> bool:
    return float(role_index.get(entity_id, {}).get(role, 0.0) or 0.0) > 0.0


def _top_role_for_entity(
    entity_id: str,
    role_index: Mapping[str, Mapping[str, float]],
) -> str | None:
    roles = role_index.get(entity_id, {})
    if not roles:
        return None
    return max(roles.items(), key=lambda item: (item[1], item[0]))[0]


def _prioritized_entities(
    entities: Iterable[Any],
    *,
    role_index: Mapping[str, Mapping[str, float]],
    limit: int,
) -> Tuple[str, ...]:
    values = [str(entity) for entity in entities]

    def sort_key(entity: str) -> Tuple[float, str]:
        roles = role_index.get(entity, {})
        score = 0.0
        score += 4.0 * float(roles.get("controllable_actor", 0.0) or 0.0)
        score += 2.0 * float(roles.get("timer_or_hud", 0.0) or 0.0)
        score += 1.0 * float(roles.get("target_candidate", 0.0) or 0.0)
        score += 0.5 * float(roles.get("moving_object", 0.0) or 0.0)
        return (-score, entity)

    return tuple(sorted(dict.fromkeys(values), key=sort_key)[: max(0, int(limit))])


def _entity_pair(row: Mapping[str, Any]) -> tuple[str, str] | None:
    entities = [str(value) for value in row.get("entities", []) or []]
    if len(entities) < 2:
        return None
    return (entities[0], entities[1])


def _first(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        for item in value:
            if item is not None:
                return str(item)
    return None


def _parse_tail(value: str, tail_index: int) -> str | None:
    parts = [part for part in str(value).split("::") if part]
    if len(parts) < tail_index:
        return None
    return parts[-tail_index]


def _predicted_effect(row: Mapping[str, Any]) -> str:
    effects = row.get("predicted_effects", []) or []
    if effects:
        return str(effects[0])
    return str(row.get("mechanic_hypothesis_id", ""))


def _request_id(prefix: str, sequence_index: int, row: Mapping[str, Any]) -> str:
    source = str(row.get("mechanic_hypothesis_id", "unknown"))
    slug = source.replace("m1g0::", "").replace("::", "_")
    return f"m3g0_1::{prefix}::{sequence_index:03d}::{slug}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan generic M3.G0 mechanic experiments from M1.G0 candidates."
    )
    parser.add_argument(
        "--m1-candidates",
        default=str(DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH),
        help="Path to diagnostics/m1/general_mechanic_candidates.json.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH),
        help="Output path for generic M3.G0.1 experiment requests.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help="Maximum number of generic experiment requests to emit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_generic_mechanic_experiment_planning(
        general_mechanic_candidates_path=args.m1_candidates,
        max_requests=args.max_requests,
    )
    write_generic_mechanic_experiment_requests(payload, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "generic_experiment_requests_generated": payload["summary"][
                    "generic_experiment_requests_generated"
                ],
                "execution_performed": False,
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
