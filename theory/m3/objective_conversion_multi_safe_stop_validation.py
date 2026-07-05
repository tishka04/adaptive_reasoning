"""M3.G2 multi-safe-stop validation for objective conversion.

This module retests the M3.G1 ACTION6-led objective-conversion signal across a
small matrix of safe-stop captures. It deliberately stays candidate-only:
``support=0``, no revision, no A32/A33 writes, and no scientific confirmation.

Design:
- read M3.G1.3 as the source signal;
- reuse M3.G1.1 execution cells and M3.G1.2 cell execution;
- replace the single safe-stop capture with several capture specs
  (condition, budget, seed);
- aggregate by unique safe-stop state before rolling up the G2 status.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_conversion_experiment_consolidation import (
    CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY,
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH,
    OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY,
    candidate_outcome_record,
    relation_progress_control_summary,
    select_best_candidate,
)
from .objective_conversion_experiment_executor import (
    CapturedSafeStop,
    ObjectiveConversionCell,
    build_execution_cells,
    execute_objective_conversion_cell,
    ready_objective_conversion_requests,
    unique_execution_cells,
)
from .objective_conversion_experiment_planner import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    HOLD_OR_STOP_STATE_CONTROL,
    RELATION_PROGRESS_POLICY_CONTROL,
    SAFE_STOP_POLICY_CONDITION,
)


DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "objective_conversion_multi_safe_stop_validation.json"
)
MULTI_SAFE_STOP_SCHEMA_VERSION = (
    "m3.objective_conversion_multi_safe_stop_validation.v1"
)

DEFAULT_SAFE_STOP_CONDITIONS = (
    SAFE_STOP_POLICY_CONDITION,
    "objective_aware_abstract_policy_lambda_1",
)
DEFAULT_SAFE_STOP_BUDGETS = (48, 64)
DEFAULT_SAFE_STOP_TIE_BREAK_SEEDS = (0, 1)

DEFAULT_TARGET_ACTION_SEQUENCES = (
    ("ACTION6",),
    ("ACTION6", "ACTION3"),
    ("ACTION6", "ACTION4"),
)

MULTI_SAFE_STOP_CONTEXTS = "MULTI_SAFE_STOP_CONTEXTS"
LOW_DUPLICATE_SAFE_STOP_CONTEXTS = "LOW_DUPLICATE_SAFE_STOP_CONTEXTS"

REPRODUCED_MULTI_SAFE_STOP_SIGNAL_CANDIDATE_ONLY = (
    "REPRODUCED_MULTI_SAFE_STOP_SIGNAL_CANDIDATE_ONLY"
)
LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY = (
    "LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY"
)
TERMINAL_RISK_REAPPEARS_CANDIDATE_ONLY = (
    "TERMINAL_RISK_REAPPEARS_CANDIDATE_ONLY"
)


@dataclass(frozen=True)
class SafeStopCaptureSpec:
    """One candidate safe-stop capture configuration for M3.G2."""

    condition: str
    budget: int
    tie_break_seed: int

    @property
    def spec_id(self) -> str:
        condition = self.condition.replace(":", "_")
        return f"m3_g2::{condition}::b{int(self.budget)}::s{int(self.tie_break_seed)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "condition": self.condition,
            "budget": int(self.budget),
            "tie_break_seed": int(self.tie_break_seed),
            "selection_counted_as_support": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        }


def run_objective_conversion_multi_safe_stop_validation(
    *,
    requests_path: str | Path = DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    source_consolidation_path: str
    | Path = DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    safe_stop_specs: Sequence[SafeStopCaptureSpec | Mapping[str, Any]] | None = None,
    conditions: Sequence[str] = DEFAULT_SAFE_STOP_CONDITIONS,
    budgets: Sequence[int] = DEFAULT_SAFE_STOP_BUDGETS,
    tie_break_seeds: Sequence[int] = DEFAULT_SAFE_STOP_TIE_BREAK_SEEDS,
    max_safe_stops: int | None = None,
    safe_stop_capturer: Callable[
        [SafeStopCaptureSpec, Sequence[Mapping[str, Any]]], CapturedSafeStop
    ]
    | None = None,
    cell_executor: Callable[
        [ObjectiveConversionCell, CapturedSafeStop], Mapping[str, Any]
    ]
    | None = None,
) -> Dict[str, Any]:
    request_payload = _load_json(requests_path)
    _validate_request_payload(request_payload)
    requests = ready_objective_conversion_requests(request_payload)

    source_payload = _load_json(source_consolidation_path)
    _validate_source_consolidation_payload(source_payload)
    target_sequences = target_action_sequences_from_source(source_payload)

    planned_cells, _links = build_execution_cells(requests)
    selected_cells = select_multi_safe_stop_cells(
        unique_execution_cells(planned_cells),
        target_sequences=target_sequences,
    )

    specs = normalize_safe_stop_specs(
        safe_stop_specs
        if safe_stop_specs is not None
        else build_default_safe_stop_specs(
            conditions=conditions,
            budgets=budgets,
            tie_break_seeds=tie_break_seeds,
        )
    )
    if max_safe_stops is not None:
        specs = specs[: max(0, int(max_safe_stops))]

    if safe_stop_capturer is None:
        safe_stop_capturer = _make_default_multi_safe_stop_capturer(
            environments_dir=environments_dir,
        )
    if cell_executor is None:
        cell_executor = _make_default_multi_safe_stop_cell_executor(
            environments_dir=environments_dir,
        )

    capture_records: List[Dict[str, Any]] = []
    unique_capture_contexts: List[Tuple[str, SafeStopCaptureSpec, CapturedSafeStop]] = []
    seen_capture_ids: Dict[str, str] = {}
    for spec in specs:
        captured = safe_stop_capturer(spec, requests)
        identity = safe_stop_identity(captured)
        duplicate_of = seen_capture_ids.get(identity)
        if duplicate_of is None:
            safe_stop_id = f"m3_g2::safe_stop::{len(unique_capture_contexts) + 1:03d}"
            seen_capture_ids[identity] = safe_stop_id
            unique_capture_contexts.append((safe_stop_id, spec, captured))
        else:
            safe_stop_id = duplicate_of
        capture_records.append(
            safe_stop_capture_record(
                safe_stop_id=safe_stop_id,
                spec=spec,
                captured=captured,
                duplicate_of=duplicate_of,
            )
        )

    execution_cells: List[Dict[str, Any]] = []
    for safe_stop_id, spec, captured in unique_capture_contexts:
        for cell in selected_cells:
            row = dict(cell_executor(cell, captured))
            row.update(
                {
                    "safe_stop_id": safe_stop_id,
                    "capture_spec_id": spec.spec_id,
                    "capture_condition": spec.condition,
                    "capture_budget": int(spec.budget),
                    "capture_tie_break_seed": int(spec.tie_break_seed),
                    "cell_result_counted_as_confirmation": False,
                    "support": 0,
                    "revision_status": "CANDIDATE_ONLY",
                    "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                    "wrong_confirmations": 0,
                }
            )
            execution_cells.append(row)

    per_safe_stop_records = [
        per_safe_stop_validation_record(
            safe_stop_id=safe_stop_id,
            spec=spec,
            captured=captured,
            cells=[
                cell
                for cell in execution_cells
                if str(cell.get("safe_stop_id", "")) == safe_stop_id
            ],
            target_sequences=target_sequences,
        )
        for safe_stop_id, spec, captured in unique_capture_contexts
    ]
    sequence_aggregates = aggregate_sequence_records(per_safe_stop_records)
    validation_status = roll_up_multi_safe_stop_status(per_safe_stop_records)
    context_diversity = context_diversity_assessment(unique_capture_contexts)

    return {
        "config": {
            "schema_version": MULTI_SAFE_STOP_SCHEMA_VERSION,
            "requests_path": str(requests_path),
            "source_consolidation_path": str(source_consolidation_path),
            "environments_dir": (
                None if environments_dir is None else str(environments_dir)
            ),
            "inputs_read": ["M3.G1.1", "M3.G1.3", "P3.G1"],
            "artifacts_not_modified": ["M2", "M3.G1", "A32", "A33"],
            "stage_produces": "multi_safe_stop_candidate_only_validation",
            "safe_stop_unit": "unique replay-exact safe_stop_state",
            "controls": [HOLD_OR_STOP_STATE_CONTROL, RELATION_PROGRESS_POLICY_CONTROL],
            "target_action_sequences": [list(seq) for seq in target_sequences],
            "support_events_counted_as_scientific_support": False,
        },
        "source_signal_summary": source_signal_summary(source_payload),
        "safe_stop_capture_specs": [spec.to_dict() for spec in specs],
        "safe_stop_captures": capture_records,
        "execution_cells": execution_cells,
        "per_safe_stop_validation_records": per_safe_stop_records,
        "sequence_aggregates": sequence_aggregates,
        "summary": summarize_multi_safe_stop_validation(
            specs=specs,
            selected_cells=selected_cells,
            execution_cells=execution_cells,
            per_safe_stop_records=per_safe_stop_records,
            validation_status=validation_status,
            context_diversity=context_diversity,
            target_sequences=target_sequences,
        ),
        "validation_outcome_status": validation_status,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "safe_stop_context_diversity": context_diversity,
        "multi_safe_stop_validation_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "validation_outcome_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def build_default_safe_stop_specs(
    *,
    conditions: Sequence[str],
    budgets: Sequence[int],
    tie_break_seeds: Sequence[int],
) -> Tuple[SafeStopCaptureSpec, ...]:
    specs: List[SafeStopCaptureSpec] = []
    for condition in conditions:
        if not str(condition):
            continue
        for budget in budgets:
            for seed in tie_break_seeds:
                specs.append(
                    SafeStopCaptureSpec(
                        condition=str(condition),
                        budget=int(budget),
                        tie_break_seed=int(seed),
                    )
                )
    return tuple(specs)


def normalize_safe_stop_specs(
    specs: Sequence[SafeStopCaptureSpec | Mapping[str, Any]],
) -> Tuple[SafeStopCaptureSpec, ...]:
    normalized: List[SafeStopCaptureSpec] = []
    for spec in specs:
        if isinstance(spec, SafeStopCaptureSpec):
            normalized.append(spec)
            continue
        normalized.append(
            SafeStopCaptureSpec(
                condition=str(spec.get("condition", SAFE_STOP_POLICY_CONDITION)),
                budget=int(spec.get("budget", 64) or 64),
                tie_break_seed=int(spec.get("tie_break_seed", 0) or 0),
            )
        )
    return tuple(normalized)


def target_action_sequences_from_source(
    payload: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], ...]:
    """Retest ACTION6-led conversion signals from M3.G1.3 plus ACTION6 ablation."""
    selected: List[Tuple[str, ...]] = []
    for record in payload.get("candidate_outcome_records", []) or []:
        if not isinstance(record, Mapping):
            continue
        actions = tuple(str(action) for action in record.get("action_or_sequence", []))
        status = str(record.get("candidate_outcome_status", ""))
        if (
            actions
            and actions[0] == "ACTION6"
            and status == CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
        ):
            selected.append(actions)

    for fallback in DEFAULT_TARGET_ACTION_SEQUENCES:
        if fallback not in selected:
            selected.append(fallback)
    return tuple(dict.fromkeys(selected))


def select_multi_safe_stop_cells(
    cells: Sequence[ObjectiveConversionCell],
    *,
    target_sequences: Sequence[Sequence[str]],
) -> Tuple[ObjectiveConversionCell, ...]:
    target_keys = {tuple(sequence) for sequence in target_sequences}
    target_horizons = {len(sequence) for sequence in target_keys}
    selected: List[ObjectiveConversionCell] = []
    seen: set[str] = set()
    for cell in cells:
        include = False
        if cell.condition_kind == "candidate":
            include = tuple(cell.action_or_sequence or ()) in target_keys
        elif cell.condition_kind == "hold":
            include = True
        elif cell.condition_kind == "relation_progress_policy":
            include = int(cell.post_stop_horizon) in target_horizons
        if include and cell.cell_signature not in seen:
            selected.append(cell)
            seen.add(cell.cell_signature)
    return tuple(selected)


def safe_stop_capture_record(
    *,
    safe_stop_id: str,
    spec: SafeStopCaptureSpec,
    captured: CapturedSafeStop,
    duplicate_of: str | None,
) -> Dict[str, Any]:
    public = captured.to_public_dict()
    return {
        **public,
        "safe_stop_id": safe_stop_id,
        "capture_spec": spec.to_dict(),
        "is_duplicate_safe_stop": duplicate_of is not None,
        "duplicate_of_safe_stop_id": duplicate_of,
        "duplicate_safe_stop_counted_as_independent": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def per_safe_stop_validation_record(
    *,
    safe_stop_id: str,
    spec: SafeStopCaptureSpec,
    captured: CapturedSafeStop,
    cells: Sequence[Mapping[str, Any]],
    target_sequences: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    executed_cells = [
        dict(cell) for cell in cells if bool(cell.get("execution_performed", False))
    ]
    relation_summary = relation_progress_control_summary(executed_cells)
    candidate_records = [
        with_safe_stop_record_context(
            candidate_outcome_record(
                cell,
                hold_baseline=float(captured.hold_baseline_terminal_adjusted_progress),
                relation_summary=relation_summary,
            ),
            safe_stop_id=safe_stop_id,
            spec=spec,
        )
        for cell in executed_cells
        if str(cell.get("condition_kind", "")) == "candidate"
    ]
    best_candidate = select_best_candidate(candidate_records)
    target_keys = {tuple(sequence) for sequence in target_sequences}
    action6_led_records = [
        record
        for record in candidate_records
        if tuple(record.get("action_or_sequence", []) or []) in target_keys
    ]
    conversion_records = [
        record for record in action6_led_records if record["beats_hold_or_stop_state"]
    ]
    terminal_records = [
        record
        for record in action6_led_records
        if bool(record.get("candidate_terminal_reentry", False))
    ]
    objective_records = [
        record
        for record in action6_led_records
        if bool(record.get("objective_completion_signal", False))
    ]
    return {
        "safe_stop_id": safe_stop_id,
        "capture_spec": spec.to_dict(),
        "safe_stop_state_hash": captured.safe_stop_state_hash,
        "captured_prefix_hash": captured.captured_prefix_hash,
        "captured_prefix_len": len(captured.prefix),
        "hold_or_stop_state_terminal_adjusted_progress": float(
            captured.hold_baseline_terminal_adjusted_progress
        ),
        "cells_executed": len(executed_cells),
        "cells_blocked": len(cells) - len(executed_cells),
        "candidate_records": candidate_records,
        "best_candidate": best_candidate,
        "relation_progress_control_summary": relation_summary,
        "action6_led_conversion_signal_observed": bool(conversion_records),
        "action6_led_terminal_risk_observed": bool(terminal_records),
        "objective_completion_observed": bool(objective_records),
        "conversion_signal_candidates": len(conversion_records),
        "terminal_risk_candidates": len(terminal_records),
        "objective_completion_candidates": len(objective_records),
        "safe_stop_outcome_status": per_safe_stop_outcome_status(
            conversion_records=conversion_records,
            terminal_records=terminal_records,
            objective_records=objective_records,
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "safe_stop_record_counted_as_confirmation": False,
    }


def with_safe_stop_record_context(
    record: Mapping[str, Any],
    *,
    safe_stop_id: str,
    spec: SafeStopCaptureSpec,
) -> Dict[str, Any]:
    return {
        **dict(record),
        "safe_stop_id": safe_stop_id,
        "capture_spec_id": spec.spec_id,
        "capture_condition": spec.condition,
        "capture_budget": int(spec.budget),
        "capture_tie_break_seed": int(spec.tie_break_seed),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def per_safe_stop_outcome_status(
    *,
    conversion_records: Sequence[Mapping[str, Any]],
    terminal_records: Sequence[Mapping[str, Any]],
    objective_records: Sequence[Mapping[str, Any]],
) -> str:
    if objective_records:
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    if terminal_records:
        return TERMINAL_RISK_REAPPEARS_CANDIDATE_ONLY
    if conversion_records:
        return CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
    return LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY


def aggregate_sequence_records(
    per_safe_stop_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_sequence: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for safe_stop in per_safe_stop_records:
        for record in safe_stop.get("candidate_records", []) or []:
            if not isinstance(record, Mapping):
                continue
            actions = tuple(str(action) for action in record.get("action_or_sequence", []))
            by_sequence.setdefault(actions, []).append(record)

    aggregates: List[Dict[str, Any]] = []
    for actions, records in sorted(by_sequence.items()):
        deltas = [
            float(record.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0)
            for record in records
        ]
        aggregates.append(
            {
                "action_or_sequence": list(actions),
                "safe_stops_tested": len(records),
                "conversion_signal_safe_stops": len(
                    [record for record in records if record["beats_hold_or_stop_state"]]
                ),
                "terminal_reentry_safe_stops": len(
                    [
                        record
                        for record in records
                        if bool(record.get("candidate_terminal_reentry", False))
                    ]
                ),
                "objective_completion_safe_stops": len(
                    [
                        record
                        for record in records
                        if bool(record.get("objective_completion_signal", False))
                    ]
                ),
                "mean_delta_vs_hold": round(
                    sum(deltas) / len(deltas) if deltas else 0.0,
                    6,
                ),
                "max_delta_vs_hold": round(max(deltas) if deltas else 0.0, 6),
                "min_delta_vs_hold": round(min(deltas) if deltas else 0.0, 6),
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "sequence_aggregate_counted_as_confirmation": False,
            }
        )
    return aggregates


def roll_up_multi_safe_stop_status(
    per_safe_stop_records: Sequence[Mapping[str, Any]],
) -> str:
    if any(bool(row.get("objective_completion_observed", False)) for row in per_safe_stop_records):
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    if any(bool(row.get("action6_led_terminal_risk_observed", False)) for row in per_safe_stop_records):
        return TERMINAL_RISK_REAPPEARS_CANDIDATE_ONLY
    reproduced = len(
        [
            row
            for row in per_safe_stop_records
            if bool(row.get("action6_led_conversion_signal_observed", False))
        ]
    )
    if len(per_safe_stop_records) >= 2 and reproduced >= 2:
        return REPRODUCED_MULTI_SAFE_STOP_SIGNAL_CANDIDATE_ONLY
    return LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY


def context_diversity_assessment(
    unique_capture_contexts: Sequence[Tuple[str, SafeStopCaptureSpec, CapturedSafeStop]],
) -> str:
    if len(unique_capture_contexts) >= 2:
        return MULTI_SAFE_STOP_CONTEXTS
    return LOW_DUPLICATE_SAFE_STOP_CONTEXTS


def summarize_multi_safe_stop_validation(
    *,
    specs: Sequence[SafeStopCaptureSpec],
    selected_cells: Sequence[ObjectiveConversionCell],
    execution_cells: Sequence[Mapping[str, Any]],
    per_safe_stop_records: Sequence[Mapping[str, Any]],
    validation_status: str,
    context_diversity: str,
    target_sequences: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    executed = [cell for cell in execution_cells if bool(cell.get("execution_performed", False))]
    blocked = [cell for cell in execution_cells if not bool(cell.get("execution_performed", False))]
    reproduced_safe_stops = [
        row
        for row in per_safe_stop_records
        if bool(row.get("action6_led_conversion_signal_observed", False))
    ]
    terminal_safe_stops = [
        row
        for row in per_safe_stop_records
        if bool(row.get("action6_led_terminal_risk_observed", False))
    ]
    objective_safe_stops = [
        row
        for row in per_safe_stop_records
        if bool(row.get("objective_completion_observed", False))
    ]
    return {
        "safe_stop_capture_specs_planned": len(specs),
        "unique_safe_stop_captures": len(per_safe_stop_records),
        "duplicate_safe_stop_captures": max(0, len(specs) - len(per_safe_stop_records)),
        "duplicate_safe_stop_captures_counted_as_independent": False,
        "target_action_sequences": [list(sequence) for sequence in target_sequences],
        "selected_cells_per_safe_stop": len(selected_cells),
        "execution_cells_total": len(execution_cells),
        "cells_executed": len(executed),
        "cells_blocked": len(blocked),
        "candidate_cells_executed": len(
            [cell for cell in executed if cell.get("condition_kind") == "candidate"]
        ),
        "hold_cells_executed": len(
            [cell for cell in executed if cell.get("condition_kind") == "hold"]
        ),
        "relation_progress_cells_executed": len(
            [
                cell
                for cell in executed
                if cell.get("condition_kind") == "relation_progress_policy"
            ]
        ),
        "reproduced_signal_safe_stops": len(reproduced_safe_stops),
        "terminal_risk_safe_stops": len(terminal_safe_stops),
        "objective_completion_safe_stops": len(objective_safe_stops),
        "validation_outcome_status": validation_status,
        "safe_stop_context_diversity": context_diversity,
        "execution_performed": True,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "multi_safe_stop_validation_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "validation_outcome_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def source_signal_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    return {
        "source_consolidation_outcome_status": str(
            summary.get("consolidation_outcome_status", "")
        ),
        "source_conversion_signal_candidates": int(
            summary.get("conversion_signal_candidates", 0) or 0
        ),
        "source_terminal_reentry_candidates": int(
            summary.get("terminal_reentry_candidates", 0) or 0
        ),
        "source_best_candidate_action_or_sequence": list(
            summary.get("best_candidate_action_or_sequence", []) or []
        ),
        "source_safe_stop_context_diversity": str(
            summary.get("safe_stop_context_diversity", "")
        ),
        "source_signal_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def safe_stop_identity(captured: CapturedSafeStop) -> str:
    return "::".join(
        [
            str(captured.captured_prefix_hash),
            str(captured.safe_stop_state_hash),
            str(captured.hold_baseline_terminal_adjusted_progress),
        ]
    )


def _make_default_multi_safe_stop_capturer(
    *,
    environments_dir: str | Path | None,
) -> Callable[[SafeStopCaptureSpec, Sequence[Mapping[str, Any]]], CapturedSafeStop]:
    def capturer(
        spec: SafeStopCaptureSpec,
        requests: Sequence[Mapping[str, Any]],
    ) -> CapturedSafeStop:
        return capture_safe_stop_for_spec(
            spec,
            requests=requests,
            environments_dir=environments_dir,
        )

    return capturer


def _make_default_multi_safe_stop_cell_executor(
    *,
    environments_dir: str | Path | None,
) -> Callable[[ObjectiveConversionCell, CapturedSafeStop], Mapping[str, Any]]:
    def executor(
        cell: ObjectiveConversionCell,
        captured: CapturedSafeStop,
    ) -> Mapping[str, Any]:
        return execute_objective_conversion_cell(
            cell,
            captured=captured,
            environments_dir=environments_dir,
        )

    return executor


def capture_safe_stop_for_spec(
    spec: SafeStopCaptureSpec,
    *,
    requests: Sequence[Mapping[str, Any]],
    environments_dir: str | Path | None,
    objective_adapter_path: str | Path | None = None,
) -> CapturedSafeStop:
    """Capture one parameterized P3.G1 safe-stop state."""
    from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir
    from theory.p3.objective_aware_abstract_policy_probe import (
        DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH,
        execute_objective_aware_condition,
        summarize_objective_aware_steps,
    )

    adapter_path = (
        Path(objective_adapter_path)
        if objective_adapter_path is not None
        else DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    adapter_payload = _load_json(adapter_path)
    adapter = dict(adapter_payload.get("objective_aware_policy_adapter", {}) or {})
    game_id = _game_id_from_requests(requests)

    steps, stop_event = execute_objective_aware_condition(
        condition=spec.condition,
        adapter=adapter,
        budget=int(spec.budget),
        tie_break_seed=int(spec.tie_break_seed),
        environments_dir=env_dir,
        game_id=game_id,
    )
    prefix = tuple(
        {
            "action": str(step.get("policy_selected_action", "")),
            "action_args": dict(step.get("action_args", {}) or {}),
        }
        for step in steps
    )
    baseline = summarize_objective_aware_steps(
        condition=spec.condition,
        steps=steps,
        budget=int(spec.budget),
        tie_break_seed=int(spec.tie_break_seed),
        stop_event=stop_event,
    )
    safe_stop_state_hash = (
        str(steps[-1].get("state_signature_after", ""))
        if steps
        else "safe_stop::initial"
    )
    return CapturedSafeStop(
        prefix=prefix,
        captured_prefix_hash=_prefix_hash(prefix),
        safe_stop_state_hash=safe_stop_state_hash,
        hold_baseline_terminal_adjusted_progress=float(
            baseline.get("terminal_adjusted_progress", 0.0) or 0.0
        ),
        hold_baseline_levels_completed=int(
            baseline.get("final_levels_completed", 0) or 0
        ),
        hold_baseline_terminal=bool(
            baseline.get("terminal_state_after_rollout", False)
        ),
        provenance={
            "safe_stop_state_source": "P3.G1",
            "safe_stop_policy_condition": spec.condition,
            "safe_stop_policy_selection_source": "m3_g2_multi_safe_stop_matrix",
            "stop_trigger_reason": str(
                stop_event.get("trigger_reason")
                or "objective_aware_terminal_risk_score_stop"
            ),
            "terminal_horizon_source": str(
                stop_event.get("terminal_horizon_source", "")
            ),
            "base_state_family": "terminal_safe_stop_or_avoidance_state",
            "budget": int(spec.budget),
            "tie_break_seed": int(spec.tie_break_seed),
        },
        capture_config={
            "selection_rule": "m3_g2_multi_safe_stop_matrix",
            "condition": spec.condition,
            "budget": int(spec.budget),
            "tie_break_seed": int(spec.tie_break_seed),
            "selection_counted_as_support": False,
        },
        adapter=adapter,
        prefix_step_dicts=tuple(dict(step) for step in steps),
    )


def _validate_request_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("request source support must remain 0")
    if bool(payload.get("execution_performed", False)) or bool(
        summary.get("execution_performed", False)
    ):
        raise ValueError("M3.G2 reads planning requests, not executed requests")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("request source must not write A32/A33")


def _validate_source_consolidation_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("source consolidation support must remain 0")
    if bool(payload.get("revision_performed", False)) or bool(
        summary.get("revision_performed", False)
    ):
        raise ValueError("source consolidation cannot be revised")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("source consolidation must not write A32/A33")
    if bool(payload.get("consolidation_status_counted_as_scientific_verdict", False)):
        raise ValueError("source consolidation status cannot be scientific verdict")
    if not payload.get("candidate_outcome_records"):
        raise ValueError("M3.G2 requires M3.G1 candidate outcome records")


def _game_id_from_requests(requests: Sequence[Mapping[str, Any]]) -> str:
    for request in requests:
        game_id = str(request.get("game_id", ""))
        if game_id:
            return game_id
    return "bp35-0a0ad940"


def _prefix_hash(prefix: Sequence[Mapping[str, Any]]) -> str:
    raw = [
        {
            "action": str(step.get("action", "")),
            "action_args": dict(step.get("action_args", {}) or {}),
        }
        for step in prefix
    ]
    return "prefix::" + json.dumps(raw, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_objective_conversion_multi_safe_stop_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M3.G2 multi-safe-stop objective-conversion validation.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--source-consolidation",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(DEFAULT_SAFE_STOP_CONDITIONS),
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=list(DEFAULT_SAFE_STOP_BUDGETS),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SAFE_STOP_TIE_BREAK_SEEDS),
    )
    parser.add_argument("--max-safe-stops", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_multi_safe_stop_validation(
        requests_path=args.requests,
        source_consolidation_path=args.source_consolidation,
        environments_dir=args.environments_dir,
        conditions=tuple(args.conditions or DEFAULT_SAFE_STOP_CONDITIONS),
        budgets=tuple(args.budgets or DEFAULT_SAFE_STOP_BUDGETS),
        tie_break_seeds=tuple(args.seeds or DEFAULT_SAFE_STOP_TIE_BREAK_SEEDS),
        max_safe_stops=args.max_safe_stops,
    )
    write_objective_conversion_multi_safe_stop_validation(payload, args.out)
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
