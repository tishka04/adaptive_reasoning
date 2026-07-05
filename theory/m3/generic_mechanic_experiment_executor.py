"""M3.G0.2 executor for generic mechanic experiment requests."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from theory.m1.general_mechanic_abstraction import (
    ComponentObservation,
    DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH,
    extract_components,
)
from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame

from .generic_mechanic_experiment_planner import (
    DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    READY_FOR_M3_GENERIC_EXPERIMENT,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_GENERIC_MECHANIC_EXPERIMENT_RESULTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "generic_mechanic_experiment_results.json"
)
GENERIC_CELL_EXECUTED = "EXECUTED"
BLOCKED_ACTION_UNAVAILABLE = "BLOCKED_ACTION_UNAVAILABLE"
GENERIC_REPLAY_POLICY = "initial_reset_same_state"


@dataclass(frozen=True)
class GenericExecutionCell:
    """One unique same-state action execution shared by generic requests."""

    cell_signature: str
    game_id: str
    action: str
    replay_policy: str = GENERIC_REPLAY_POLICY
    tie_break_seed: int = 0
    requested_entities: Tuple[str, ...] = ()
    requested_relation_pairs: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_signature": self.cell_signature,
            "game_id": self.game_id,
            "action": self.action,
            "replay_policy": self.replay_policy,
            "tie_break_seed": int(self.tie_break_seed),
            "requested_entities": list(self.requested_entities),
            "requested_relation_pairs": [
                [left, right] for left, right in self.requested_relation_pairs
            ],
        }


def run_generic_mechanic_experiment_execution(
    *,
    generic_requests_path: str | Path = (
        DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH
    ),
    m1_candidates_path: str | Path | None = None,
    environments_dir: str | Path | None = None,
    tie_break_seed: int = 0,
    max_cells: int | None = None,
    cell_executor: Callable[[GenericExecutionCell], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    request_payload = _load_json(generic_requests_path)
    validate_generic_request_source_payload(request_payload)
    requests = ready_generic_requests(request_payload)
    m1_path = Path(
        m1_candidates_path
        or (request_payload.get("config", {}) or {}).get(
            "general_mechanic_candidates_path",
            DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH,
        )
    )
    m1_payload = _load_json(m1_path) if m1_path.exists() else {}
    entity_profiles = entity_profiles_from_m1_payload(m1_payload)

    planned_cells, links = build_generic_execution_cells(
        requests,
        tie_break_seed=tie_break_seed,
    )
    unique_cells = unique_generic_execution_cells(planned_cells)
    if max_cells is not None:
        unique_cells = unique_cells[: max(0, int(max_cells))]
    executed_signatures = {cell.cell_signature for cell in unique_cells}
    executable_links = [
        link for link in links if str(link.get("cell_signature", "")) in executed_signatures
    ]

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    if cell_executor is None:
        _configure_offline_env(env_dir)

        def default_executor(cell: GenericExecutionCell) -> Mapping[str, Any]:
            return execute_generic_cell(
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
        build_request_result(
            request,
            [result_by_signature.get(str(link.get("cell_signature", "")), {}) for link in executable_links if str(link.get("request_id", "")) == str(request.get("request_id", ""))],
        )
        for request in requests
    ]
    observation_links = [
        build_generic_observation_link(
            link,
            result_by_signature.get(str(link.get("cell_signature", "")), {}),
        )
        for link in executable_links
    ]
    return build_generic_results_payload(
        generic_requests_path=generic_requests_path,
        m1_candidates_path=m1_path,
        environments_dir=env_dir,
        requests=requests,
        planned_cells=planned_cells,
        unique_cells=unique_cells,
        cell_results=cell_results,
        request_results=request_results,
        observation_links=observation_links,
    )


def build_generic_execution_cells(
    requests: Sequence[Mapping[str, Any]],
    *,
    tie_break_seed: int = 0,
) -> Tuple[Tuple[GenericExecutionCell, ...], Tuple[Dict[str, Any], ...]]:
    cells: list[GenericExecutionCell] = []
    links: list[Dict[str, Any]] = []
    for request in requests:
        actions = execution_actions_for_request(request)
        requested_entities = requested_entities_for_request(request)
        requested_pairs = requested_relation_pairs_for_request(request)
        for action in actions:
            cell = make_generic_execution_cell(
                request,
                action=action,
                requested_entities=requested_entities,
                requested_relation_pairs=requested_pairs,
                tie_break_seed=tie_break_seed,
            )
            cells.append(cell)
            links.append(
                {
                    "request_id": str(request.get("request_id", "")),
                    "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
                    "source_mechanic_family": str(
                        request.get("source_mechanic_family", "")
                    ),
                    "test_type": str(request.get("test_type", "")),
                    "cell_signature": cell.cell_signature,
                    "action": str(action),
                    "action_role": action_role_for_request(request, action),
                    "duplicate_execution_cell_counted_as_independent": False,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                    "wrong_confirmations": 0,
                }
            )
    return tuple(cells), tuple(links)


def make_generic_execution_cell(
    request: Mapping[str, Any],
    *,
    action: str,
    requested_entities: Sequence[str],
    requested_relation_pairs: Sequence[Tuple[str, str]],
    tie_break_seed: int,
) -> GenericExecutionCell:
    game_id = str(request.get("game_id", ""))
    signature = generic_cell_signature(
        game_id=game_id,
        replay_policy=GENERIC_REPLAY_POLICY,
        action=str(action),
        tie_break_seed=tie_break_seed,
    )
    return GenericExecutionCell(
        cell_signature=signature,
        game_id=game_id,
        action=str(action),
        tie_break_seed=int(tie_break_seed),
        requested_entities=tuple(dict.fromkeys(str(value) for value in requested_entities)),
        requested_relation_pairs=tuple(
            dict.fromkeys((str(left), str(right)) for left, right in requested_relation_pairs)
        ),
    )


def unique_generic_execution_cells(
    planned_cells: Sequence[GenericExecutionCell],
) -> Tuple[GenericExecutionCell, ...]:
    by_signature: dict[str, GenericExecutionCell] = {}
    entity_sets: dict[str, set[str]] = {}
    pair_sets: dict[str, set[Tuple[str, str]]] = {}
    for cell in planned_cells:
        by_signature.setdefault(cell.cell_signature, cell)
        entity_sets.setdefault(cell.cell_signature, set()).update(cell.requested_entities)
        pair_sets.setdefault(cell.cell_signature, set()).update(cell.requested_relation_pairs)
    merged: list[GenericExecutionCell] = []
    for signature, cell in by_signature.items():
        merged.append(
            replace(
                cell,
                requested_entities=tuple(sorted(entity_sets.get(signature, set()))),
                requested_relation_pairs=tuple(sorted(pair_sets.get(signature, set()))),
            )
        )
    return tuple(merged)


def execute_generic_cell(
    cell: GenericExecutionCell,
    *,
    entity_profiles: Mapping[str, Mapping[str, Any]],
    environments_dir: str | Path,
) -> Dict[str, Any]:
    env = _make_env(cell.game_id, environments_dir)
    current_frame = _reset_env(env)
    before = snapshot_frame(current_frame)
    selected = _select_named_action(_valid_actions(env), cell.action)
    if selected is None:
        return blocked_generic_cell_result(
            cell,
            reason=f"no_concrete_action_available:{cell.action}",
            before_frame=before,
        )
    after_frame = _step_env_action(env, selected)
    after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
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
        "action_args": dict(getattr(selected, "action_args", {}) or {}),
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


def blocked_generic_cell_result(
    cell: GenericExecutionCell,
    *,
    reason: str,
    before_frame: Any,
) -> Dict[str, Any]:
    return {
        **cell.to_dict(),
        "status": BLOCKED_ACTION_UNAVAILABLE,
        "blocked_reason": reason,
        "global_delta": {
            "changed_pixels": 0,
            "terminal_state": is_terminal_game_state(getattr(before_frame, "game_state", "")),
            "game_state_before": str(getattr(before_frame, "game_state", "")),
            "game_state_after": str(getattr(before_frame, "game_state", "")),
            "level_completed": False,
        },
        "entity_delta": {},
        "relation_delta": {},
        "invariant_delta": {},
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


def build_request_result(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    family = str(request.get("source_mechanic_family", ""))
    if family == "entity_role":
        support, contradiction, neutral, interpretation = classify_entity_role_request(
            request, cell_results
        )
    elif family == "action_effect":
        support, contradiction, neutral, interpretation = classify_action_effect_request(
            request, cell_results
        )
    elif family == "relation_change":
        support, contradiction, neutral, interpretation = classify_relation_change_request(
            request, cell_results
        )
    elif family == "dynamic_invariant":
        support, contradiction, neutral, interpretation = classify_dynamic_invariant_request(
            request, cell_results
        )
    else:
        support, contradiction, neutral, interpretation = 0, 0, 1, "unknown_family"
    blocked = sum(1 for row in cell_results if str(row.get("status", "")) != GENERIC_CELL_EXECUTED)
    return {
        "request_id": str(request.get("request_id", "")),
        "source_hypothesis_id": str(request.get("source_hypothesis_id", "")),
        "source_mechanic_family": family,
        "test_type": str(request.get("test_type", "")),
        "observation_interpretation": interpretation,
        "cells_linked": len(cell_results),
        "blocked_cells": int(blocked),
        "support_events": int(support),
        "contradiction_events": int(contradiction),
        "neutral_events": int(neutral),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "request_result_counted_as_confirmation": False,
        "source_confidence_counted_as_support": False,
    }


def classify_entity_role_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str]:
    entity_id = str(request.get("target_entity", ""))
    role = str(request.get("candidate_role", ""))
    deltas = [
        (row.get("entity_delta", {}) or {}).get(entity_id, {})
        for row in cell_results
        if str(row.get("status", "")) == GENERIC_CELL_EXECUTED
    ]
    changed_count = sum(1 for delta in deltas if bool(delta.get("changed")))
    if role == "controllable_actor":
        return (
            (1, 0, 0, "actor_candidate_action_differentiated")
            if changed_count > 0
            else (0, 1, 0, "actor_candidate_not_moved_or_transformed")
        )
    if role == "timer_or_hud":
        return (
            (1, 0, 0, "hud_timer_candidate_changes_across_actions")
            if changed_count >= max(1, len(deltas) // 2)
            else (0, 0, 1, "hud_timer_candidate_not_decisive")
        )
    return (0, 0, 1, "role_probe_observation_neutral")


def classify_action_effect_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str]:
    target_action = str(request.get("target_action", ""))
    effect = str(request.get("predicted_effect_family", ""))
    target = next((row for row in cell_results if str(row.get("action", "")) == target_action), {})
    controls = [row for row in cell_results if str(row.get("action", "")) != target_action]
    target_has = effect in observed_effect_families(target)
    control_hits = sum(1 for row in controls if effect in observed_effect_families(row))
    if target_has and control_hits < len(controls):
        return (1, 0, 0, "target_action_effect_exceeds_at_least_one_control")
    if not target_has and control_hits > 0:
        return (0, 1, 0, "target_action_lacks_effect_seen_in_control")
    return (0, 0, 1, "action_effect_not_discriminated")


def classify_relation_change_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str]:
    target_action = str(request.get("target_action", ""))
    pair_key = relation_pair_key(
        str(request.get("source_entity", "")),
        str(request.get("relation_target_entity", "")),
    )
    expected_delta = str(request.get("relation_delta_type", ""))
    target = next((row for row in cell_results if str(row.get("action", "")) == target_action), {})
    controls = [row for row in cell_results if str(row.get("action", "")) != target_action]
    target_delta = ((target.get("relation_delta", {}) or {}).get(pair_key, {}) or {})
    target_has = expected_delta in set(target_delta.get("delta_types", []) or [])
    control_hits = 0
    for row in controls:
        delta = ((row.get("relation_delta", {}) or {}).get(pair_key, {}) or {})
        if expected_delta in set(delta.get("delta_types", []) or []):
            control_hits += 1
    if target_has and control_hits < len(controls):
        return (1, 0, 0, "target_action_relation_delta_exceeds_at_least_one_control")
    if not target_has and control_hits > 0:
        return (0, 1, 0, "target_action_lacks_relation_delta_seen_in_control")
    return (0, 0, 1, "relation_delta_not_discriminated")


def classify_dynamic_invariant_request(
    request: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
) -> Tuple[int, int, int, str]:
    entity_id = str(request.get("target_entity", ""))
    family = str(request.get("invariant_family", ""))
    deltas = [
        (row.get("invariant_delta", {}) or {}).get(entity_id, {})
        for row in cell_results
        if str(row.get("status", "")) == GENERIC_CELL_EXECUTED
    ]
    changed_count = sum(1 for delta in deltas if bool(delta.get("value_changed")))
    if family == "monotone_counter":
        same_direction = len({str(delta.get("direction", "")) for delta in deltas if delta.get("direction")}) <= 1
        if changed_count == len(deltas) and same_direction and deltas:
            return (1, 0, 0, "monotone_counter_regular_without_semantic_assignment")
    if family == "exogenous_motion" and changed_count > 0:
        return (1, 0, 0, "exogenous_motion_candidate_observed_as_entity_delta")
    return (0, 0, 1, "dynamic_invariant_not_decisive")


def observed_effect_families(cell_result: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    for delta in (cell_result.get("entity_delta", {}) or {}).values():
        families.update(str(value) for value in delta.get("effect_families", []) or [])
    if (cell_result.get("relation_delta", {}) or {}):
        if any((delta.get("delta_types", []) or []) for delta in (cell_result.get("relation_delta", {}) or {}).values()):
            families.add("change_relation")
    if any(bool(delta.get("value_changed")) for delta in (cell_result.get("invariant_delta", {}) or {}).values()):
        families.add("tick_latent")
    if int((cell_result.get("global_delta", {}) or {}).get("changed_pixels", 0) or 0) > 0:
        families.add("global_transition")
    return families


def build_generic_observation_link(
    link: Mapping[str, Any],
    cell_result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        **dict(link),
        "cell_status": str(cell_result.get("status", "")),
        "action": str(cell_result.get("action", link.get("action", ""))),
        "global_changed_pixels": int(
            (cell_result.get("global_delta", {}) or {}).get("changed_pixels", 0) or 0
        ),
        "terminal_state": bool(
            (cell_result.get("global_delta", {}) or {}).get("terminal_state", False)
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def build_generic_results_payload(
    *,
    generic_requests_path: str | Path,
    m1_candidates_path: str | Path,
    environments_dir: str | Path,
    requests: Sequence[Mapping[str, Any]],
    planned_cells: Sequence[GenericExecutionCell],
    unique_cells: Sequence[GenericExecutionCell],
    cell_results: Sequence[Mapping[str, Any]],
    request_results: Sequence[Mapping[str, Any]],
    observation_links: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    support_events = sum(int(row.get("support_events", 0) or 0) for row in request_results)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in request_results
    )
    neutral_events = sum(int(row.get("neutral_events", 0) or 0) for row in request_results)
    blocked_cells = sum(1 for row in cell_results if str(row.get("status", "")) != GENERIC_CELL_EXECUTED)
    controlled = sum(int(row.get("controlled_experiments_run", 0) or 0) for row in cell_results)
    return {
        "config": {
            "generic_requests_path": str(generic_requests_path),
            "m1_candidates_path": str(m1_candidates_path),
            "schema_version": "m3.generic_mechanic_experiment_results.v1",
            "inputs_read": ["M3.G0.1", "M1.G0 entity profiles"],
            "artifacts_not_modified": ["M1", "M2", "A32", "A33"],
            "replay_policy": GENERIC_REPLAY_POLICY,
            "environments_dir": str(environments_dir),
        },
        "summary": {
            "generic_requests_consumed": len(requests),
            "planned_execution_cells": len(planned_cells),
            "unique_execution_cells": len(unique_cells),
            "duplicate_execution_cells": max(0, len(planned_cells) - len(unique_cells)),
            "duplicate_execution_cells_counted_as_independent": False,
            "controlled_experiments_run": int(controlled),
            "blocked_controls": int(blocked_cells),
            "support_events": int(support_events),
            "contradiction_events": int(contradiction_events),
            "neutral_events": int(neutral_events),
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
        "request_result_counted_as_confirmation": False,
        "cell_result_counted_as_confirmation": False,
        "source_confidence_counted_as_support": False,
        "semantic_interpretation_counted_as_confirmation": False,
        "a32_remains_only_verdict_location": True,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def execution_actions_for_request(request: Mapping[str, Any]) -> Tuple[str, ...]:
    values: list[str] = []
    target = request.get("target_action")
    if target:
        values.append(str(target))
        values.extend(str(action) for action in request.get("control_actions", []) or [])
    else:
        values.extend(str(action) for action in request.get("conditions", []) or [])
    return tuple(dict.fromkeys(value for value in values if value))


def action_role_for_request(request: Mapping[str, Any], action: str) -> str:
    if str(request.get("target_action", "")) == str(action):
        return "target_action"
    if str(action) in {str(value) for value in request.get("control_actions", []) or []}:
        return "control_action"
    return "condition_action"


def requested_entities_for_request(request: Mapping[str, Any]) -> Tuple[str, ...]:
    values = []
    for key in ("target_entity", "source_entity", "relation_target_entity"):
        value = request.get(key)
        if value:
            values.append(str(value))
    values.extend(str(value) for value in request.get("affected_entities", []) or [])
    return tuple(dict.fromkeys(values))


def requested_relation_pairs_for_request(
    request: Mapping[str, Any],
) -> Tuple[Tuple[str, str], ...]:
    source = request.get("source_entity")
    target = request.get("relation_target_entity")
    if source and target:
        return ((str(source), str(target)),)
    return ()


def measure_entity_delta(
    entity_id: str,
    *,
    before_grid: np.ndarray,
    after_grid: np.ndarray,
    entity_profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    before_component = resolve_entity_component(entity_id, before_grid, entity_profiles)
    after_component = resolve_entity_component(entity_id, after_grid, entity_profiles)
    return compare_components(before_component, after_component)


def compare_components(
    before: ComponentObservation | None,
    after: ComponentObservation | None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "before_present": before is not None,
        "after_present": after is not None,
        "changed": False,
        "effect_families": [],
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }
    families: set[str] = set()
    if before is None and after is None:
        return row
    if before is None and after is not None:
        families.add("create_entity")
        row["after_bbox"] = list(after.bbox)
        row["after_centroid"] = [round(float(value), 4) for value in after.centroid]
        row["after_size"] = int(after.size)
    elif before is not None and after is None:
        families.add("delete_entity")
        row["before_bbox"] = list(before.bbox)
        row["before_centroid"] = [round(float(value), 4) for value in before.centroid]
        row["before_size"] = int(before.size)
    elif before is not None and after is not None:
        dy = round(float(after.centroid[0] - before.centroid[0]), 4)
        dx = round(float(after.centroid[1] - before.centroid[1]), 4)
        distance = round(math.hypot(dy, dx), 6)
        size_delta = int(after.size - before.size)
        shape_changed = before.shape_signature != after.shape_signature
        bbox_delta = [int(after.bbox[index] - before.bbox[index]) for index in range(4)]
        if distance > 0:
            families.add("move_entity")
        if size_delta != 0 or shape_changed:
            families.add("transform_entity")
        row.update(
            {
                "before_bbox": list(before.bbox),
                "after_bbox": list(after.bbox),
                "before_centroid": [round(float(value), 4) for value in before.centroid],
                "after_centroid": [round(float(value), 4) for value in after.centroid],
                "centroid_delta": [dy, dx],
                "centroid_distance": distance,
                "bbox_delta": bbox_delta,
                "size_delta": int(size_delta),
                "shape_signature_before": before.shape_signature,
                "shape_signature_after": after.shape_signature,
                "shape_changed": bool(shape_changed),
            }
        )
    row["effect_families"] = sorted(families)
    row["changed"] = bool(families)
    return row


def measure_relation_delta(
    left_id: str,
    right_id: str,
    *,
    before_grid: np.ndarray,
    after_grid: np.ndarray,
    entity_profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    before_left = resolve_entity_component(left_id, before_grid, entity_profiles)
    before_right = resolve_entity_component(right_id, before_grid, entity_profiles)
    after_left = resolve_entity_component(left_id, after_grid, entity_profiles)
    after_right = resolve_entity_component(right_id, after_grid, entity_profiles)
    before_rel = relation_features(before_left, before_right)
    after_rel = relation_features(after_left, after_right)
    delta_types: set[str] = set()
    if before_rel.get("touches") is False and after_rel.get("touches") is True:
        delta_types.add("contact_created")
    if before_rel.get("touches") is True and after_rel.get("touches") is False:
        delta_types.add("contact_removed")
    if before_rel.get("near") is False and after_rel.get("near") is True:
        delta_types.add("near_relation_created")
    if before_rel.get("near") is True and after_rel.get("near") is False:
        delta_types.add("near_relation_removed")
    if before_rel.get("contains") is False and after_rel.get("contains") is True:
        delta_types.add("containment_created")
    if before_rel.get("contains") is True and after_rel.get("contains") is False:
        delta_types.add("containment_removed")
    before_distance = before_rel.get("centroid_distance")
    after_distance = after_rel.get("centroid_distance")
    if isinstance(before_distance, (int, float)) and isinstance(after_distance, (int, float)):
        if after_distance < before_distance:
            delta_types.add("distance_decreases")
        elif after_distance > before_distance:
            delta_types.add("distance_increases")
    return {
        "before": before_rel,
        "after": after_rel,
        "delta_types": sorted(delta_types),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def relation_features(
    left: ComponentObservation | None,
    right: ComponentObservation | None,
) -> Dict[str, Any]:
    if left is None or right is None:
        return {
            "left_present": left is not None,
            "right_present": right is not None,
            "touches": False,
            "near": False,
            "contains": False,
            "centroid_distance": None,
        }
    distance = bbox_distance(left.bbox, right.bbox)
    centroid_distance = math.hypot(
        float(left.centroid[0] - right.centroid[0]),
        float(left.centroid[1] - right.centroid[1]),
    )
    return {
        "left_present": True,
        "right_present": True,
        "touches": bool(distance == 0),
        "near": bool(0 < distance <= 2.0),
        "contains": bool(bbox_contains(left.bbox, right.bbox) or bbox_contains(right.bbox, left.bbox)),
        "bbox_distance": round(float(distance), 6),
        "centroid_distance": round(float(centroid_distance), 6),
    }


def measure_invariant_delta_from_entity_delta(
    entity_delta: Mapping[str, Any],
) -> Dict[str, Any]:
    size_delta = int(entity_delta.get("size_delta", 0) or 0)
    centroid_delta = entity_delta.get("centroid_delta", []) or []
    centroid_changed = bool(
        len(centroid_delta) == 2
        and (abs(float(centroid_delta[0])) > 0 or abs(float(centroid_delta[1])) > 0)
    )
    value_changed = bool(size_delta != 0 or centroid_changed or entity_delta.get("changed"))
    direction = None
    if size_delta > 0:
        direction = "increasing"
    elif size_delta < 0:
        direction = "decreasing"
    elif centroid_changed:
        dy = float(centroid_delta[0])
        dx = float(centroid_delta[1])
        direction = "down" if abs(dy) >= abs(dx) and dy > 0 else (
            "up" if abs(dy) >= abs(dx) else ("right" if dx > 0 else "left")
        )
    return {
        "value_changed": value_changed,
        "size_delta": size_delta,
        "centroid_delta": list(centroid_delta),
        "direction": direction,
        "semantic_interpretation": "unknown",
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def resolve_entity_component(
    entity_id: str,
    grid: np.ndarray,
    entity_profiles: Mapping[str, Mapping[str, Any]],
) -> ComponentObservation | None:
    profile = entity_profiles.get(entity_id)
    if not profile:
        return None
    expected_color = int(profile.get("color", -999))
    expected_bbox = tuple(int(value) for value in profile.get("initial_bbox", []) or [])
    components = extract_components(grid, background=None, frame_index=0)
    candidates = [component for component in components if component.color == expected_color]
    if not candidates:
        return None
    if len(expected_bbox) != 4:
        return sorted(candidates, key=lambda component: (-component.size, component.bbox))[0]
    return min(
        candidates,
        key=lambda component: (
            bbox_delta_magnitude(expected_bbox, component.bbox),
            -component.size,
            component.bbox,
        ),
    )


def entity_profiles_from_m1_payload(
    payload: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("entity_tracks", []) or []:
        entity_id = str(row.get("entity_id", ""))
        bbox_sequence = row.get("bbox_sequence", []) or []
        centroid_sequence = row.get("centroid_sequence", []) or []
        shape_sequence = row.get("shape_signature_sequence", []) or []
        if not entity_id:
            continue
        profiles[entity_id] = {
            "entity_id": entity_id,
            "color": int(row.get("color", -999) or -999),
            "initial_bbox": list(bbox_sequence[0]) if bbox_sequence else [],
            "initial_centroid": list(centroid_sequence[0]) if centroid_sequence else [],
            "initial_shape_signature": str(shape_sequence[0]) if shape_sequence else "",
        }
    return profiles


def validate_generic_request_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G0.1 request support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("M3.G0.1 summary support must remain 0")
    if bool(payload.get("revision_performed", False)):
        raise ValueError("M3.G0.1 source must not perform revision")
    if bool(payload.get("a32_write_performed", False)) or bool(payload.get("a33_write_performed", False)):
        raise ValueError("M3.G0.1 source cannot write A32/A33")
    if bool(payload.get("generic_request_counted_as_confirmation", False)):
        raise ValueError("M3.G0.1 source cannot count requests as confirmation")


def ready_generic_requests(payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    for request in payload.get("generic_mechanic_experiment_requests", []) or []:
        if str(request.get("status", "")) != READY_FOR_M3_GENERIC_EXPERIMENT:
            continue
        if int(request.get("support", 0) or 0) != 0:
            raise ValueError("generic request support must remain 0")
        if bool(request.get("execution_performed", False)):
            raise ValueError("M3.G0.1 request must not already be executed")
        rows.append(dict(request))
    return tuple(rows)


def write_generic_mechanic_experiment_results(
    payload: Mapping[str, Any],
    out_path: str | Path = DEFAULT_GENERIC_MECHANIC_EXPERIMENT_RESULTS_OUTPUT_PATH,
) -> None:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def generic_cell_signature(
    *,
    game_id: str,
    replay_policy: str,
    action: str,
    tie_break_seed: int,
) -> str:
    return f"m3g0_2::{game_id}::{replay_policy}::{action}::seed_{int(tie_break_seed)}"


def relation_pair_key(left: str, right: str) -> str:
    return f"{left}::{right}"


def bbox_delta_magnitude(left: Sequence[int], right: Sequence[int]) -> int:
    return int(sum(abs(int(left[index]) - int(right[index])) for index in range(4)))


def bbox_distance(left: Sequence[int], right: Sequence[int]) -> float:
    top_a, left_a, bottom_a, right_a = [int(value) for value in left]
    top_b, left_b, bottom_b, right_b = [int(value) for value in right]
    dy = max(top_b - bottom_a - 1, top_a - bottom_b - 1, 0)
    dx = max(left_b - right_a - 1, left_a - right_b - 1, 0)
    return math.hypot(dy, dx)


def bbox_contains(left: Sequence[int], right: Sequence[int]) -> bool:
    top_a, left_a, bottom_a, right_a = [int(value) for value in left]
    top_b, left_b, bottom_b, right_b = [int(value) for value in right]
    return top_a <= top_b and left_a <= left_b and bottom_a >= bottom_b and right_a >= right_b


def is_terminal_game_state(game_state: Any) -> bool:
    return str(game_state or "").upper() == "GAME_OVER"


def _select_named_action(valid_actions: Sequence[Any], action_name: str) -> Any | None:
    matches = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == str(action_name)
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda action: json.dumps(
            dict(getattr(action, "action_args", {}) or {}),
            sort_keys=True,
        ),
    )[0]


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute generic M3.G0.2 mechanic experiment requests."
    )
    parser.add_argument(
        "--requests",
        default=str(DEFAULT_GENERIC_MECHANIC_EXPERIMENT_REQUESTS_OUTPUT_PATH),
        help="Path to diagnostics/m3/generic_mechanic_experiment_requests.json.",
    )
    parser.add_argument(
        "--m1-candidates",
        default=None,
        help="Optional M1.G0 candidate ledger path for entity profiles.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_GENERIC_MECHANIC_EXPERIMENT_RESULTS_OUTPUT_PATH),
        help="Output path for generic M3.G0.2 experiment results.",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_generic_mechanic_experiment_execution(
        generic_requests_path=args.requests,
        m1_candidates_path=args.m1_candidates,
        max_cells=args.max_cells,
    )
    write_generic_mechanic_experiment_results(payload, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "generic_requests_consumed": payload["summary"][
                    "generic_requests_consumed"
                ],
                "unique_execution_cells": payload["summary"][
                    "unique_execution_cells"
                ],
                "controlled_experiments_run": payload["summary"][
                    "controlled_experiments_run"
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
