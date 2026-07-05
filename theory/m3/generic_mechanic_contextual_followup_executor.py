"""M3.G0.5 executor for generic contextual mechanic follow-ups."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .generic_mechanic_contextual_followup_planner import (
    DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    READY_FOR_M3_GENERIC_CONTEXTUAL_FOLLOWUP,
    MULTI_PREFIX_CONTEXTS,
)
from .generic_mechanic_experiment_executor import (
    BLOCKED_ACTION_UNAVAILABLE,
    GENERIC_CELL_EXECUTED,
    entity_profiles_from_m1_payload,
    is_terminal_game_state,
    measure_entity_delta,
    measure_invariant_delta_from_entity_delta,
    measure_relation_delta,
    observed_effect_families,
    relation_pair_key,
    requested_entities_for_request,
    requested_relation_pairs_for_request,
    _select_named_action,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_RESULTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "generic_mechanic_contextual_followup_results.json"
)
CONTEXTUAL_REPLAY_POLICY = "contextual_prefix_replay"
TEMPORAL_SEQUENCE_POLICY = "contextual_temporal_sequence"
BLOCKED_CONTEXT_UNAVAILABLE = "BLOCKED_CONTEXT_UNAVAILABLE"
CONTEXT_REPRODUCED_CANDIDATE_ONLY = "CONTEXT_REPRODUCED_CANDIDATE_ONLY"
CONTEXT_FAILED_CANDIDATE_ONLY = "CONTEXT_FAILED_CANDIDATE_ONLY"
MIXED_CONTEXT_CANDIDATE_ONLY = "MIXED_CONTEXT_CANDIDATE_ONLY"
NEUTRAL_CONTEXT_CANDIDATE_ONLY = "NEUTRAL_CONTEXT_CANDIDATE_ONLY"
TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY = (
    "TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY"
)


@dataclass(frozen=True)
class GenericContextualExecutionCell:
    """One unique contextual execution shared by follow-up requests."""

    cell_signature: str
    game_id: str
    replay_policy: str
    context_replay: Tuple[str, ...]
    action: str | None
    temporal_action_sequence: Tuple[str, ...]
    tie_break_seed: int = 0
    requested_entities: Tuple[str, ...] = ()
    requested_relation_pairs: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_signature": self.cell_signature,
            "game_id": self.game_id,
            "replay_policy": self.replay_policy,
            "context_replay": list(self.context_replay),
            "context_replay_args": None,
            "action": self.action,
            "temporal_action_sequence": list(self.temporal_action_sequence),
            "tie_break_seed": int(self.tie_break_seed),
            "requested_entities": list(self.requested_entities),
            "requested_relation_pairs": [
                [left, right] for left, right in self.requested_relation_pairs
            ],
        }


def run_generic_mechanic_contextual_followup_execution(
    *,
    contextual_requests_path: str | Path = (
        DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH
    ),
    m1_candidates_path: str | Path | None = None,
    environments_dir: str | Path | None = None,
    tie_break_seed: int = 0,
    max_cells: int | None = None,
    cell_executor: Callable[[GenericContextualExecutionCell], Mapping[str, Any]]
    | None = None,
) -> Dict[str, Any]:
    request_payload = _load_json(contextual_requests_path)
    validate_contextual_request_source_payload(request_payload)
    requests = ready_contextual_requests(request_payload)
    m1_path = Path(
        m1_candidates_path
        or "diagnostics/m1/general_mechanic_candidates.json"
    )
    m1_payload = _load_json(m1_path) if m1_path.exists() else {}
    entity_profiles = entity_profiles_from_m1_payload(m1_payload)

    planned_cells, links = build_contextual_execution_cells(
        requests,
        tie_break_seed=tie_break_seed,
    )
    unique_cells = unique_contextual_execution_cells(planned_cells)
    if max_cells is not None:
        unique_cells = unique_cells[: max(0, int(max_cells))]
    executed_signatures = {cell.cell_signature for cell in unique_cells}
    executable_links = [
        link
        for link in links
        if str(link.get("cell_signature", "")) in executed_signatures
    ]

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    if cell_executor is None:
        _configure_offline_env(env_dir)

        def default_executor(cell: GenericContextualExecutionCell) -> Mapping[str, Any]:
            return execute_contextual_cell(
                cell,
                entity_profiles=entity_profiles,
                environments_dir=env_dir,
            )

        cell_executor = default_executor

    cell_results = [dict(cell_executor(cell)) for cell in unique_cells]
    result_by_signature = {
        str(row.get("cell_signature", "")): dict(row) for row in cell_results
    }
    request_results = [
        build_contextual_request_result(
            request,
            [
                result_by_signature.get(str(link.get("cell_signature", "")), {})
                for link in executable_links
                if str(link.get("request_id", "")) == str(request.get("request_id", ""))
            ],
        )
        for request in requests
    ]
    observation_links = [
        build_contextual_observation_link(
            link,
            result_by_signature.get(str(link.get("cell_signature", "")), {}),
        )
        for link in executable_links
    ]
    return build_contextual_results_payload(
        contextual_requests_path=contextual_requests_path,
        m1_candidates_path=m1_path,
        environments_dir=env_dir,
        requests=requests,
        planned_cells=planned_cells,
        unique_cells=unique_cells,
        cell_results=cell_results,
        request_results=request_results,
        observation_links=observation_links,
    )


def build_contextual_execution_cells(
    requests: Sequence[Mapping[str, Any]],
    *,
    tie_break_seed: int = 0,
) -> Tuple[Tuple[GenericContextualExecutionCell, ...], Tuple[Dict[str, Any], ...]]:
    cells: list[GenericContextualExecutionCell] = []
    links: list[Dict[str, Any]] = []
    for request in requests:
        requested_entities = requested_entities_for_request(request)
        requested_pairs = requested_relation_pairs_for_request(request)
        for action in execution_actions_for_contextual_request(request):
            cell = make_contextual_execution_cell(
                request,
                action=action,
                temporal_action_sequence=(),
                requested_entities=requested_entities,
                requested_relation_pairs=requested_pairs,
                tie_break_seed=tie_break_seed,
            )
            cells.append(cell)
            links.append(contextual_link_for_cell(request, cell, action_role_for_request(request, action)))
        for sequence in request.get("temporal_action_sequences", []) or []:
            cell = make_contextual_execution_cell(
                request,
                action=None,
                temporal_action_sequence=tuple(str(action) for action in sequence),
                requested_entities=requested_entities,
                requested_relation_pairs=requested_pairs,
                tie_break_seed=tie_break_seed,
            )
            cells.append(cell)
            links.append(contextual_link_for_cell(request, cell, "temporal_sequence"))
    return tuple(cells), tuple(links)


def make_contextual_execution_cell(
    request: Mapping[str, Any],
    *,
    action: str | None,
    temporal_action_sequence: Sequence[str],
    requested_entities: Sequence[str],
    requested_relation_pairs: Sequence[Tuple[str, str]],
    tie_break_seed: int,
) -> GenericContextualExecutionCell:
    game_id = str(request.get("game_id", ""))
    context = tuple(str(item) for item in request.get("context_replay", []) or [])
    temporal = tuple(str(item) for item in temporal_action_sequence)
    replay_policy = TEMPORAL_SEQUENCE_POLICY if temporal else CONTEXTUAL_REPLAY_POLICY
    signature = contextual_cell_signature(
        game_id=game_id,
        replay_policy=replay_policy,
        context_replay=context,
        action=action,
        temporal_action_sequence=temporal,
        tie_break_seed=tie_break_seed,
    )
    return GenericContextualExecutionCell(
        cell_signature=signature,
        game_id=game_id,
        replay_policy=replay_policy,
        context_replay=context,
        action=str(action) if action else None,
        temporal_action_sequence=temporal,
        tie_break_seed=int(tie_break_seed),
        requested_entities=tuple(dict.fromkeys(str(value) for value in requested_entities)),
        requested_relation_pairs=tuple(
            dict.fromkeys((str(left), str(right)) for left, right in requested_relation_pairs)
        ),
    )


def unique_contextual_execution_cells(
    planned_cells: Sequence[GenericContextualExecutionCell],
) -> Tuple[GenericContextualExecutionCell, ...]:
    by_signature: dict[str, GenericContextualExecutionCell] = {}
    entity_sets: dict[str, set[str]] = {}
    pair_sets: dict[str, set[Tuple[str, str]]] = {}
    for cell in planned_cells:
        by_signature.setdefault(cell.cell_signature, cell)
        entity_sets.setdefault(cell.cell_signature, set()).update(cell.requested_entities)
        pair_sets.setdefault(cell.cell_signature, set()).update(cell.requested_relation_pairs)
    return tuple(
        replace(
            cell,
            requested_entities=tuple(sorted(entity_sets.get(signature, set()))),
            requested_relation_pairs=tuple(sorted(pair_sets.get(signature, set()))),
        )
        for signature, cell in by_signature.items()
    )


def execute_contextual_cell(
    cell: GenericContextualExecutionCell,
    *,
    entity_profiles: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path,
) -> Dict[str, Any]:
    env = _make_env(cell.game_id, environments_dir)
    current_frame = _reset_env(env)
    replay_trace: list[Dict[str, Any]] = []
    for action_name in cell.context_replay:
        selected = _select_named_action(_valid_actions(env), action_name)
        if selected is None:
            return blocked_contextual_cell_result(
                cell,
                reason=f"context_action_unavailable:{action_name}",
                before_frame=snapshot_frame(current_frame),
                replay_trace=replay_trace,
            )
        current_frame = _step_env_action(env, selected)
        replay_trace.append(
            {
                "action": str(action_name),
                "action_args": dict(getattr(selected, "action_args", {}) or {}),
            }
        )
    if cell.temporal_action_sequence:
        return execute_temporal_contextual_cell(
            cell,
            env=env,
            current_frame=current_frame,
            replay_trace=replay_trace,
            entity_profiles=entity_profiles,
        )
    if not cell.action:
        return blocked_contextual_cell_result(
            cell,
            reason="missing_target_action",
            before_frame=snapshot_frame(current_frame),
            replay_trace=replay_trace,
        )
    before = snapshot_frame(current_frame)
    selected = _select_named_action(_valid_actions(env), cell.action)
    if selected is None:
        return blocked_contextual_cell_result(
            cell,
            reason=f"target_or_control_action_unavailable:{cell.action}",
            before_frame=before,
            replay_trace=replay_trace,
        )
    after_frame = _step_env_action(env, selected)
    after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
    return measured_contextual_cell_result(
        cell,
        before=before,
        after=after,
        action_args=dict(getattr(selected, "action_args", {}) or {}),
        replay_trace=replay_trace,
        entity_profiles=entity_profiles,
    )


def execute_temporal_contextual_cell(
    cell: GenericContextualExecutionCell,
    *,
    env: Any,
    current_frame: Any,
    replay_trace: Sequence[Mapping[str, Any]],
    entity_profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    before = snapshot_frame(current_frame)
    transition_rows: list[Dict[str, Any]] = []
    previous = before
    for action_name in cell.temporal_action_sequence:
        selected = _select_named_action(_valid_actions(env), action_name)
        if selected is None:
            return blocked_contextual_cell_result(
                cell,
                reason=f"temporal_action_unavailable:{action_name}",
                before_frame=previous,
                replay_trace=replay_trace,
                temporal_transition_rows=transition_rows,
            )
        next_frame = _step_env_action(env, selected)
        after_step = snapshot_frame(
            next_frame,
            fallback_available_actions=previous.available_actions,
        )
        transition_rows.append(
            measured_temporal_transition(
                action_name=str(action_name),
                before=previous,
                after=after_step,
                action_args=dict(getattr(selected, "action_args", {}) or {}),
                requested_entities=cell.requested_entities,
                requested_relation_pairs=cell.requested_relation_pairs,
                entity_profiles=entity_profiles,
            )
        )
        previous = after_step
    final = previous
    result = measured_contextual_cell_result(
        cell,
        before=before,
        after=final,
        action_args={},
        replay_trace=replay_trace,
        entity_profiles=entity_profiles,
    )
    result["temporal_transition_rows"] = transition_rows
    result["temporal_regularities"] = temporal_regularities_from_transitions(
        transition_rows,
        requested_entities=cell.requested_entities,
    )
    return result


def measured_contextual_cell_result(
    cell: GenericContextualExecutionCell,
    *,
    before: Any,
    after: Any,
    action_args: Mapping[str, Any],
    replay_trace: Sequence[Mapping[str, Any]],
    entity_profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    before_grid = np.asarray(before.grid)
    after_grid = np.asarray(after.grid)
    entity_deltas = {
        entity_id: measure_entity_delta(
            entity_id,
            before_grid=before_grid,
            after_grid=after_grid,
            entity_profiles=entity_profiles,
        )
        for entity_id in cell.requested_entities
    }
    relation_deltas = {
        relation_pair_key(left, right): measure_relation_delta(
            left,
            right,
            before_grid=before_grid,
            after_grid=after_grid,
            entity_profiles=entity_profiles,
        )
        for left, right in cell.requested_relation_pairs
    }
    invariant_deltas = {
        entity_id: measure_invariant_delta_from_entity_delta(entity_delta)
        for entity_id, entity_delta in entity_deltas.items()
    }
    changed_pixels = (
        int(np.count_nonzero(before_grid != after_grid))
        if before_grid.shape == after_grid.shape
        else int(max(before_grid.size, after_grid.size))
    )
    return {
        **cell.to_dict(),
        "status": GENERIC_CELL_EXECUTED,
        "replay_trace": [dict(row) for row in replay_trace],
        "action_args": dict(action_args),
        "entity_delta": entity_deltas,
        "relation_delta": relation_deltas,
        "invariant_delta": invariant_deltas,
        "global_delta": {
            "changed_pixels": int(changed_pixels),
            "terminal_state": is_terminal_game_state(after.game_state),
            "game_state_before": str(before.game_state),
            "game_state_after": str(after.game_state),
            "level_completed": int(getattr(after, "levels_completed", 0) or 0)
            > int(getattr(before, "levels_completed", 0) or 0),
            "levels_completed_before": int(getattr(before, "levels_completed", 0) or 0),
            "levels_completed_after": int(getattr(after, "levels_completed", 0) or 0),
        },
        "controlled_experiments_run": 1,
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 1,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "execution_performed": True,
        "semantic_interpretation": "unknown",
    }


def measured_temporal_transition(
    *,
    action_name: str,
    before: Any,
    after: Any,
    action_args: Mapping[str, Any],
    requested_entities: Sequence[str],
    requested_relation_pairs: Sequence[Tuple[str, str]],
    entity_profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    before_grid = np.asarray(before.grid)
    after_grid = np.asarray(after.grid)
    entity_deltas = {
        entity_id: measure_entity_delta(
            entity_id,
            before_grid=before_grid,
            after_grid=after_grid,
            entity_profiles=entity_profiles,
        )
        for entity_id in requested_entities
    }
    return {
        "action": action_name,
        "action_args": dict(action_args),
        "entity_delta": entity_deltas,
        "relation_delta": {
            relation_pair_key(left, right): measure_relation_delta(
                left,
                right,
                before_grid=before_grid,
                after_grid=after_grid,
                entity_profiles=entity_profiles,
            )
            for left, right in requested_relation_pairs
        },
        "invariant_delta": {
            entity_id: measure_invariant_delta_from_entity_delta(entity_delta)
            for entity_id, entity_delta in entity_deltas.items()
        },
        "global_delta": {
            "changed_pixels": int(np.count_nonzero(before_grid != after_grid))
            if before_grid.shape == after_grid.shape
            else int(max(before_grid.size, after_grid.size)),
            "terminal_state": is_terminal_game_state(after.game_state),
            "level_completed": int(getattr(after, "levels_completed", 0) or 0)
            > int(getattr(before, "levels_completed", 0) or 0),
        },
        "semantic_interpretation": "unknown",
    }


def temporal_regularities_from_transitions(
    transition_rows: Sequence[Mapping[str, Any]],
    *,
    requested_entities: Sequence[str],
) -> Dict[str, Any]:
    by_entity: Dict[str, Dict[str, Any]] = {}
    for entity_id in requested_entities:
        deltas = [
            ((row.get("invariant_delta", {}) or {}).get(entity_id, {}) or {})
            for row in transition_rows
        ]
        changed_count = sum(1 for delta in deltas if bool(delta.get("value_changed")))
        directions = [
            str(delta.get("direction", ""))
            for delta in deltas
            if delta.get("direction")
        ]
        by_entity[entity_id] = {
            "value_delta_per_action": [
                bool(delta.get("value_changed")) for delta in deltas
            ],
            "changed_count": int(changed_count),
            "steps_observed": len(deltas),
            "monotonicity": bool(deltas and changed_count == len(deltas) and len(set(directions)) <= 1),
            "direction": directions[0] if directions and len(set(directions)) == 1 else None,
            "action_correlation": bool(deltas and changed_count == len(deltas)),
            "terminal_correlation": "unknown",
            "semantic_interpretation": "unknown",
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        }
    return by_entity


def blocked_contextual_cell_result(
    cell: GenericContextualExecutionCell,
    *,
    reason: str,
    before_frame: Any,
    replay_trace: Sequence[Mapping[str, Any]],
    temporal_transition_rows: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    return {
        **cell.to_dict(),
        "status": BLOCKED_CONTEXT_UNAVAILABLE
        if reason.startswith("context_")
        else BLOCKED_ACTION_UNAVAILABLE,
        "blocked_reason": reason,
        "replay_trace": [dict(row) for row in replay_trace],
        "temporal_transition_rows": [dict(row) for row in temporal_transition_rows],
        "entity_delta": {},
        "relation_delta": {},
        "invariant_delta": {},
        "global_delta": {
            "changed_pixels": 0,
            "terminal_state": is_terminal_game_state(getattr(before_frame, "game_state", "")),
            "game_state_before": str(getattr(before_frame, "game_state", "")),
            "game_state_after": str(getattr(before_frame, "game_state", "")),
            "level_completed": False,
        },
        "controlled_experiments_run": 0,
        "support_events": 0,
        "contradiction_events": 0,
        "neutral_events": 0,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "execution_performed": False,
        "semantic_interpretation": "unknown",
    }


def build_contextual_request_result_explicit(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
    *,
    support: int,
    contradiction: int,
    neutral: int,
    status: str,
    interpretation: str,
    blocked: int,
) -> Dict[str, Any]:
    contexts = sorted(
        {
            contextual_context_key(row)
            for row in cell_results
            if str(row.get("status", "")) == GENERIC_CELL_EXECUTED
        }
    )
    return {
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "source_mechanic_family": str(request.get("source_mechanic_family", "")),
        "followup_family": str(request.get("followup_family", "")),
        "candidate_status": status,
        "observation_interpretation": interpretation,
        "cells_linked": len(cell_results),
        "executed_cells": len(cell_results) - int(blocked),
        "blocked_cells": int(blocked),
        "independent_contexts": len(contexts),
        "context_keys": contexts,
        "support_events": int(support),
        "contradiction_events": int(contradiction),
        "neutral_events": int(neutral),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "request_result_counted_as_confirmation": False,
        "contextual_signal_counted_as_scientific_support": False,
        "remaining_semantics_unknown": request.get("remaining_semantics_unknown"),
    }


def build_contextual_request_result(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    blocked = sum(1 for row in cell_results if str(row.get("status", "")) != GENERIC_CELL_EXECUTED)
    executed = [row for row in cell_results if str(row.get("status", "")) == GENERIC_CELL_EXECUTED]
    if not executed:
        return build_contextual_request_result_explicit(
            request,
            cell_results,
            support=0,
            contradiction=0,
            neutral=0,
            status="BLOCKED_CONTEXT_UNAVAILABLE",
            interpretation="no_contextual_cell_executed",
            blocked=blocked,
        )
    family = str(request.get("source_mechanic_family", ""))
    if str(request.get("followup_family", "")).endswith("_temporal") or request.get("temporal_action_sequences"):
        support, contradiction, neutral, status, interpretation = classify_temporal_request(
            request,
            executed,
        )
    elif family == "entity_role":
        support, contradiction, neutral, status, interpretation = classify_contextual_entity_role_request(
            request,
            executed,
        )
    elif family == "action_effect":
        support, contradiction, neutral, status, interpretation = classify_contextual_action_effect_request(
            request,
            executed,
        )
    elif family == "relation_change":
        support, contradiction, neutral, status, interpretation = classify_contextual_relation_change_request(
            request,
            executed,
        )
    elif family == "dynamic_invariant":
        support, contradiction, neutral, status, interpretation = classify_temporal_request(
            request,
            executed,
        )
    else:
        support, contradiction, neutral, status, interpretation = (
            0,
            0,
            1,
            NEUTRAL_CONTEXT_CANDIDATE_ONLY,
            "unknown_contextual_family",
        )
    return build_contextual_request_result_explicit(
        request,
        cell_results,
        support=support,
        contradiction=contradiction,
        neutral=neutral,
        status=status,
        interpretation=interpretation,
        blocked=blocked,
    )


def classify_contextual_entity_role_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str, str]:
    entity_id = str(request.get("target_entity", ""))
    deltas = [
        ((row.get("entity_delta", {}) or {}).get(entity_id, {}) or {})
        for row in cell_results
    ]
    trackable = sum(
        1
        for delta in deltas
        if bool(delta.get("before_present")) or bool(delta.get("after_present"))
    )
    changed = sum(1 for delta in deltas if bool(delta.get("changed")))
    if trackable and changed:
        return (
            1,
            0,
            0,
            CONTEXT_REPRODUCED_CANDIDATE_ONLY,
            "actor_candidate_trackable_and_action_affected_outside_reset",
        )
    if trackable:
        return (
            0,
            0,
            1,
            NEUTRAL_CONTEXT_CANDIDATE_ONLY,
            "actor_candidate_trackable_but_not_action_differentiated",
        )
    return (
        0,
        1,
        0,
        CONTEXT_FAILED_CANDIDATE_ONLY,
        "actor_candidate_not_trackable_outside_reset",
    )


def classify_contextual_action_effect_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str, str]:
    target_action = str(request.get("target_action", ""))
    effect = str(request.get("predicted_effect_family", ""))
    target_cells = [row for row in cell_results if str(row.get("action", "")) == target_action]
    control_cells = [row for row in cell_results if str(row.get("action", "")) != target_action]
    target_hits = sum(1 for row in target_cells if effect in observed_effect_families(row))
    control_hits = sum(1 for row in control_cells if effect in observed_effect_families(row))
    if target_hits and target_hits >= control_hits:
        return (
            1,
            0,
            0,
            CONTEXT_REPRODUCED_CANDIDATE_ONLY,
            "action_effect_reproduced_after_non_reset_prefix",
        )
    if not target_hits and control_hits:
        return (
            0,
            1,
            0,
            CONTEXT_FAILED_CANDIDATE_ONLY,
            "effect_absent_in_target_but_seen_in_contextual_control",
        )
    return (
        0,
        0,
        1,
        NEUTRAL_CONTEXT_CANDIDATE_ONLY,
        "action_effect_not_discriminated_in_contextual_probe",
    )


def classify_contextual_relation_change_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str, str]:
    target_action = str(request.get("target_action", ""))
    pair_key = relation_pair_key(
        str(request.get("source_entity", "")),
        str(request.get("relation_target_entity", "")),
    )
    expected = str(request.get("relation_delta_type", ""))
    target_cells = [row for row in cell_results if str(row.get("action", "")) == target_action]
    control_cells = [row for row in cell_results if str(row.get("action", "")) != target_action]
    target_hits = sum(
        1
        for row in target_cells
        if expected
        in set(((row.get("relation_delta", {}) or {}).get(pair_key, {}) or {}).get("delta_types", []) or [])
    )
    control_hits = sum(
        1
        for row in control_cells
        if expected
        in set(((row.get("relation_delta", {}) or {}).get(pair_key, {}) or {}).get("delta_types", []) or [])
    )
    if target_hits and target_hits >= control_hits:
        return (
            1,
            0,
            0,
            CONTEXT_REPRODUCED_CANDIDATE_ONLY,
            "relation_delta_reproduced_after_non_reset_prefix",
        )
    if not target_hits and control_hits:
        return (
            0,
            1,
            0,
            CONTEXT_FAILED_CANDIDATE_ONLY,
            "relation_delta_absent_in_target_but_seen_in_contextual_control",
        )
    return (
        0,
        0,
        1,
        NEUTRAL_CONTEXT_CANDIDATE_ONLY,
        "relation_delta_not_discriminated_in_contextual_probe",
    )


def classify_temporal_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str, str]:
    entity_id = str(request.get("target_entity", ""))
    regular = 0
    changed = 0
    for row in cell_results:
        regularity = ((row.get("temporal_regularities", {}) or {}).get(entity_id, {}) or {})
        if bool(regularity.get("monotonicity")):
            regular += 1
        if int(regularity.get("changed_count", 0) or 0) > 0:
            changed += 1
    if regular:
        return (
            1,
            0,
            0,
            TEMPORAL_REGULARITY_OBSERVED_CANDIDATE_ONLY,
            "dynamic_invariant_regular_across_contextual_temporal_sequences",
        )
    if changed:
        return (
            0,
            0,
            1,
            MIXED_CONTEXT_CANDIDATE_ONLY,
            "dynamic_invariant_changes_but_temporal_regularities_are_mixed",
        )
    return (
        0,
        1,
        0,
        CONTEXT_FAILED_CANDIDATE_ONLY,
        "dynamic_invariant_not_observed_across_contextual_temporal_sequences",
    )


def build_contextual_observation_link(
    link: Mapping[str, Any],
    cell_result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(link),
        "cell_status": str(cell_result.get("status", "")),
        "cell_executed": str(cell_result.get("status", "")) == GENERIC_CELL_EXECUTED,
        "context_key": contextual_context_key(cell_result),
        "observation_counted_as_confirmation": False,
        "duplicate_contextual_cell_counted_as_independent": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def build_contextual_results_payload(
    *,
    contextual_requests_path: str | Path,
    m1_candidates_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    planned_cells: Sequence[GenericContextualExecutionCell],
    unique_cells: Sequence[GenericContextualExecutionCell],
    cell_results: Sequence[Mapping[str, Any]],
    request_results: Sequence[Mapping[str, Any]],
    observation_links: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    support_events = sum(int(row.get("support_events", 0) or 0) for row in request_results)
    contradiction_events = sum(int(row.get("contradiction_events", 0) or 0) for row in request_results)
    neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in request_results)
    blocked_contexts = sum(1 for row in cell_results if str(row.get("status", "")) != GENERIC_CELL_EXECUTED)
    controlled = sum(int(row.get("controlled_experiments_run", 0) or 0) for row in cell_results)
    independent_contexts = count_independent_contexts(cell_results)
    context_diversity = (
        MULTI_PREFIX_CONTEXTS if independent_contexts >= 3 else "LOW_CONTEXT_DIVERSITY"
    )
    return {
        "config": {
            "contextual_requests_path": str(contextual_requests_path),
            "m1_candidates_path": str(m1_candidates_path),
            "schema_version": "m3.generic_mechanic_contextual_followup_results.v1",
            "inputs_read": ["M3.G0.4", "M1.G0 entity profiles"],
            "artifacts_not_modified": ["M1", "M2", "A32", "A33"],
            "replay_policy": CONTEXTUAL_REPLAY_POLICY,
            "temporal_sequence_policy": TEMPORAL_SEQUENCE_POLICY,
            "environments_dir": str(environments_dir),
        },
        "summary": {
            "contextual_followup_requests_consumed": len(requests),
            "planned_contextual_cells": len(planned_cells),
            "unique_contextual_execution_cells": len(unique_cells),
            "duplicate_contextual_cells": max(0, len(planned_cells) - len(unique_cells)),
            "duplicate_contextual_cells_counted_as_independent": False,
            "independent_contexts": int(independent_contexts),
            "context_diversity_assessment": context_diversity,
            "controlled_experiments_run": int(controlled),
            "blocked_contexts": int(blocked_contexts),
            "support_events": int(support_events),
            "contradiction_events": int(contradiction_events),
            "neutral_events": int(neutral_events),
            "request_status_counts": dict(
                sorted(Counter(str(row.get("candidate_status", "")) for row in request_results).items())
            ),
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "semantic_interpretation": "unknown",
            "semantic_interpretation_counted_as_confirmation": False,
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "execution_cells": [cell.to_dict() for cell in unique_cells],
        "cell_results": [dict(row) for row in cell_results],
        "request_results": [dict(row) for row in request_results],
        "hypothesis_observation_links": [dict(row) for row in observation_links],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "execution_performed": True,
        "contextual_signal_counted_as_scientific_support": False,
        "request_result_counted_as_confirmation": False,
        "cell_result_counted_as_confirmation": False,
        "semantic_interpretation_counted_as_confirmation": False,
        "a32_remains_only_verdict_location": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def execution_actions_for_contextual_request(
    request: Mapping[str, Any],
) -> Tuple[str, ...]:
    actions: list[str] = []
    target = request.get("target_action")
    if target:
        actions.append(str(target))
        actions.extend(str(action) for action in request.get("control_actions", []) or [])
    return tuple(dict.fromkeys(action for action in actions if action))


def action_role_for_request(request: Mapping[str, Any], action: str) -> str:
    if str(request.get("target_action", "")) == str(action):
        return "target_action"
    if str(action) in {str(value) for value in request.get("control_actions", []) or []}:
        return "control_action"
    return "condition_action"


def contextual_link_for_cell(
    request: Mapping[str, Any],
    cell: GenericContextualExecutionCell,
    action_role: str,
) -> Dict[str, Any]:
    return {
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "source_mechanic_family": str(request.get("source_mechanic_family", "")),
        "followup_family": str(request.get("followup_family", "")),
        "cell_signature": cell.cell_signature,
        "context_replay": list(cell.context_replay),
        "action": cell.action,
        "temporal_action_sequence": list(cell.temporal_action_sequence),
        "action_role": action_role,
        "duplicate_contextual_cell_counted_as_independent": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def count_independent_contexts(cell_results: Sequence[Mapping[str, Any]]) -> int:
    return len(
        {
            contextual_context_key(row)
            for row in cell_results
            if str(row.get("status", "")) == GENERIC_CELL_EXECUTED
        }
    )


def contextual_context_key(row: Mapping[str, Any]) -> str:
    temporal = tuple(str(action) for action in row.get("temporal_action_sequence", []) or [])
    if temporal:
        return "temporal::" + ",".join(temporal)
    context = tuple(str(action) for action in row.get("context_replay", []) or [])
    return "prefix::" + ",".join(context)


def contextual_cell_signature(
    *,
    game_id: str,
    replay_policy: str,
    context_replay: Sequence[str],
    action: str | None,
    temporal_action_sequence: Sequence[str],
    tie_break_seed: int,
) -> str:
    context = ",".join(str(item) for item in context_replay) or "reset"
    temporal = ",".join(str(item) for item in temporal_action_sequence)
    action_key = str(action) if action else f"seq:{temporal}"
    return (
        f"m3g0_5::{game_id}::{replay_policy}::"
        f"context_{context}::action_{action_key}::seed_{int(tie_break_seed)}"
    )


def validate_contextual_request_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G0.4 request support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.4 summary support must remain 0")
    if bool(payload.get("execution_performed", False)):
        raise ValueError("M3.G0.4 source must not already be executed")
    if bool(payload.get("revision_performed", False)):
        raise ValueError("M3.G0.4 source must not perform revision")
    if bool(payload.get("duplicate_evidence_promoted", False)):
        raise ValueError("M3.G0.4 source cannot promote duplicate evidence")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("M3.G0.4 source cannot write A32/A33")


def ready_contextual_requests(payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    for request in payload.get("generic_contextual_followup_requests", []) or []:
        if str(request.get("status", "")) != READY_FOR_M3_GENERIC_CONTEXTUAL_FOLLOWUP:
            continue
        if int(request.get("support", 0) or 0) != 0:
            raise ValueError("contextual followup support must remain 0")
        if bool(request.get("execution_performed", False)):
            raise ValueError("M3.G0.4 request must not already be executed")
        if bool(request.get("duplicate_evidence_promoted", False)):
            raise ValueError("M3.G0.4 request cannot promote duplicate evidence")
        rows.append(dict(request))
    return tuple(rows)


def write_generic_mechanic_contextual_followup_results(
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_RESULTS_OUTPUT_PATH,
) -> None:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute generic contextual M3.G0.5 mechanic follow-ups."
    )
    parser.add_argument(
        "--requests",
        default=str(DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_REQUESTS_OUTPUT_PATH),
        help="Path to diagnostics/m3/generic_mechanic_contextual_followup_requests.json.",
    )
    parser.add_argument(
        "--m1-candidates",
        default=None,
        help="Optional M1.G0 candidate ledger path for entity profiles.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_GENERIC_MECHANIC_CONTEXTUAL_FOLLOWUP_RESULTS_OUTPUT_PATH),
        help="Output path for generic contextual M3.G0.5 results.",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_generic_mechanic_contextual_followup_execution(
        contextual_requests_path=args.requests,
        m1_candidates_path=args.m1_candidates,
        max_cells=args.max_cells,
    )
    write_generic_mechanic_contextual_followup_results(payload, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "contextual_followup_requests_consumed": payload["summary"][
                    "contextual_followup_requests_consumed"
                ],
                "unique_contextual_execution_cells": payload["summary"][
                    "unique_contextual_execution_cells"
                ],
                "independent_contexts": payload["summary"]["independent_contexts"],
                "context_diversity_assessment": payload["summary"][
                    "context_diversity_assessment"
                ],
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
