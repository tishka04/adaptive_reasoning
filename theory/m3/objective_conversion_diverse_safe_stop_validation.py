"""M3.G4 validation over M3.G3 diverse safe-stops.

G4 finally retests the ACTION6-led objective-conversion signal on the diverse
safe-stop substrate acquired by M3.G3. It keeps the M3 discipline:
candidate-only statuses, ``support=0``, no revision, no A32/A33 writes.

The analysis is intentionally split three ways:
- global;
- by safe-stop sampling family;
- by terminal-horizon and hold-baseline bands.

That prevents a single average from hiding a scope map such as "works on
attractor truncations but not on ACTION6 perturbations".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_conversion_experiment_consolidation import (
    CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY,
    OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY,
    TERMINAL_REENTRY_CANDIDATE_ONLY,
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
)
from .objective_conversion_multi_safe_stop_validation import (
    DEFAULT_TARGET_ACTION_SEQUENCES,
    select_multi_safe_stop_cells,
)
from .objective_conversion_safe_stop_diversity_sampler import (
    DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH,
    SUFFICIENT_FOR_M3_G4,
    prefix_hash_for_steps,
    relation_state_signature,
)


DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "objective_conversion_diverse_safe_stop_validation.json"
)
DIVERSE_SAFE_STOP_VALIDATION_SCHEMA_VERSION = (
    "m3.objective_conversion_diverse_safe_stop_validation.v1"
)

REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY = (
    "REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY"
)
LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY = (
    "LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY"
)
TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY = (
    "TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY"
)
MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY = (
    "MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY"
)
NO_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY = (
    "NO_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY"
)


def run_objective_conversion_diverse_safe_stop_validation(
    *,
    requests_path: str | Path = DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    source_g3_path: str
    | Path = DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    max_safe_stops: int | None = None,
    target_sequences: Sequence[Sequence[str]] = DEFAULT_TARGET_ACTION_SEQUENCES,
    captured_builder: Callable[[Mapping[str, Any]], CapturedSafeStop] | None = None,
    cell_executor: Callable[
        [ObjectiveConversionCell, CapturedSafeStop], Mapping[str, Any]
    ]
    | None = None,
) -> Dict[str, Any]:
    request_payload = _load_json(requests_path)
    _validate_request_payload(request_payload)
    requests = ready_objective_conversion_requests(request_payload)

    source_payload = _load_json(source_g3_path)
    _validate_source_g3_payload(source_payload)
    accepted_safe_stops = [
        dict(row)
        for row in source_payload.get("accepted_diverse_safe_stops", []) or []
        if isinstance(row, Mapping)
    ]
    if max_safe_stops is not None:
        accepted_safe_stops = accepted_safe_stops[: max(0, int(max_safe_stops))]

    planned_cells, _links = build_execution_cells(requests)
    selected_cells = select_multi_safe_stop_cells(
        unique_execution_cells(planned_cells),
        target_sequences=target_sequences,
    )

    if captured_builder is None:
        captured_builder = _make_default_captured_builder(
            environments_dir=environments_dir,
        )
    if cell_executor is None:
        cell_executor = _make_default_cell_executor(environments_dir=environments_dir)

    execution_cells: List[Dict[str, Any]] = []
    for safe_stop_record in accepted_safe_stops:
        captured = captured_builder(safe_stop_record)
        for cell in selected_cells:
            row = dict(cell_executor(cell, captured))
            row.update(safe_stop_cell_context(safe_stop_record))
            row.update(
                {
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
            safe_stop_record=record,
            cells=[
                cell
                for cell in execution_cells
                if str(cell.get("safe_stop_id", "")) == str(record.get("safe_stop_id", ""))
            ],
        )
        for record in accepted_safe_stops
    ]
    sequence_aggregates = aggregate_by_sequence(per_safe_stop_records)
    family_aggregates = aggregate_by_dimension(
        per_safe_stop_records,
        dimension_key="sampling_family",
        output_key="sampling_family",
    )
    horizon_aggregates = aggregate_by_dimension(
        per_safe_stop_records,
        dimension_key="terminal_horizon_band",
        output_key="terminal_horizon_band",
    )
    hold_baseline_aggregates = aggregate_by_dimension(
        per_safe_stop_records,
        dimension_key="hold_baseline_band",
        output_key="hold_baseline_band",
    )
    validation_status = roll_up_diverse_validation_status(
        per_safe_stop_records=per_safe_stop_records,
        family_aggregates=family_aggregates,
    )

    return {
        "config": {
            "schema_version": DIVERSE_SAFE_STOP_VALIDATION_SCHEMA_VERSION,
            "requests_path": str(requests_path),
            "source_g3_path": str(source_g3_path),
            "environments_dir": (
                None if environments_dir is None else str(environments_dir)
            ),
            "inputs_read": ["M3.G1.1", "M3.G3"],
            "artifacts_not_modified": ["M2", "M3.G3", "A32", "A33"],
            "stage_produces": "diverse_safe_stop_candidate_only_validation",
            "controls": [HOLD_OR_STOP_STATE_CONTROL, RELATION_PROGRESS_POLICY_CONTROL],
            "target_action_sequences": [list(seq) for seq in target_sequences],
            "analysis_splits": [
                "global",
                "sampling_family",
                "terminal_horizon_band",
                "hold_baseline_band",
            ],
            "support_events_counted_as_scientific_support": False,
        },
        "source_g3_diversity_summary": source_g3_diversity_summary(source_payload),
        "execution_cells": execution_cells,
        "per_safe_stop_validation_records": per_safe_stop_records,
        "sequence_aggregates": sequence_aggregates,
        "sampling_family_aggregates": family_aggregates,
        "terminal_horizon_aggregates": horizon_aggregates,
        "hold_baseline_aggregates": hold_baseline_aggregates,
        "summary": summarize_diverse_validation(
            accepted_safe_stops=accepted_safe_stops,
            selected_cells=selected_cells,
            execution_cells=execution_cells,
            per_safe_stop_records=per_safe_stop_records,
            sequence_aggregates=sequence_aggregates,
            family_aggregates=family_aggregates,
            validation_status=validation_status,
        ),
        "validation_outcome_status": validation_status,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "diverse_safe_stop_validation_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "validation_outcome_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def per_safe_stop_validation_record(
    *,
    safe_stop_record: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    executed_cells = [
        dict(cell) for cell in cells if bool(cell.get("execution_performed", False))
    ]
    relation_summary = relation_progress_control_summary(executed_cells)
    hold_baseline = float(
        safe_stop_record.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
    )
    candidate_records = [
        candidate_record_with_context(
            candidate_outcome_record(
                cell,
                hold_baseline=hold_baseline,
                relation_summary=relation_summary,
            ),
            safe_stop_record=safe_stop_record,
        )
        for cell in executed_cells
        if str(cell.get("condition_kind", "")) == "candidate"
    ]
    best_candidate = select_best_candidate(candidate_records)
    signal_records = [
        record for record in candidate_records if bool(record.get("weak_signal", False))
    ]
    medium_records = [
        record for record in candidate_records if bool(record.get("medium_signal", False))
    ]
    terminal_records = [
        record
        for record in candidate_records
        if bool(record.get("candidate_terminal_reentry", False))
    ]
    objective_records = [
        record
        for record in candidate_records
        if bool(record.get("objective_completion_signal", False))
    ]
    return {
        "safe_stop_id": str(safe_stop_record.get("safe_stop_id", "")),
        "source_plan_id": str(safe_stop_record.get("source_plan_id", "")),
        "sampling_family": str(safe_stop_record.get("sampling_family", "")),
        "terminal_horizon_remaining": terminal_horizon_remaining(safe_stop_record),
        "terminal_horizon_band": terminal_horizon_band(safe_stop_record),
        "hold_baseline_terminal_adjusted_progress": hold_baseline,
        "hold_baseline_band": hold_baseline_band(hold_baseline),
        "safe_stop_state_hash": str(safe_stop_record.get("safe_stop_state_hash", "")),
        "captured_prefix_hash": str(safe_stop_record.get("captured_prefix_hash", "")),
        "captured_prefix_len": int(safe_stop_record.get("captured_prefix_len", 0) or 0),
        "cells_executed": len(executed_cells),
        "cells_blocked": len(cells) - len(executed_cells),
        "candidate_records": candidate_records,
        "best_candidate": best_candidate,
        "relation_progress_control_summary": relation_summary,
        "weak_signal_candidates": len(signal_records),
        "medium_signal_candidates": len(medium_records),
        "terminal_risk_candidates": len(terminal_records),
        "objective_completion_candidates": len(objective_records),
        "safe_stop_outcome_status": safe_stop_outcome_status(
            signal_records=signal_records,
            terminal_records=terminal_records,
            objective_records=objective_records,
        ),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
        "safe_stop_record_counted_as_confirmation": False,
    }


def candidate_record_with_context(
    record: Mapping[str, Any],
    *,
    safe_stop_record: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate = dict(record)
    terminal = bool(candidate.get("candidate_terminal_reentry", False))
    weak = bool(
        float(candidate.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0)
        > 0.0
        and not terminal
    )
    medium = bool(weak and candidate.get("beats_relation_progress_policy", False))
    objective = bool(candidate.get("objective_completion_signal", False))
    if objective:
        strength = "VERY_STRONG_OBJECTIVE_COMPLETION_CANDIDATE_ONLY"
    elif medium:
        strength = "MEDIUM_BEATS_HOLD_AND_RELATION_CANDIDATE_ONLY"
    elif weak:
        strength = "WEAK_BEATS_HOLD_CANDIDATE_ONLY"
    elif terminal:
        strength = "TERMINAL_RISK_CANDIDATE_ONLY"
    else:
        strength = "NO_SIGNAL_CANDIDATE_ONLY"
    return {
        **candidate,
        "safe_stop_id": str(safe_stop_record.get("safe_stop_id", "")),
        "sampling_family": str(safe_stop_record.get("sampling_family", "")),
        "terminal_horizon_remaining": terminal_horizon_remaining(safe_stop_record),
        "terminal_horizon_band": terminal_horizon_band(safe_stop_record),
        "hold_baseline_band": hold_baseline_band(
            float(
                safe_stop_record.get(
                    "hold_baseline_terminal_adjusted_progress",
                    0.0,
                )
                or 0.0
            )
        ),
        "weak_signal": weak,
        "medium_signal": medium,
        "signal_strength": strength,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def aggregate_by_sequence(
    per_safe_stop_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_sequence: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for safe_stop in per_safe_stop_records:
        for record in safe_stop.get("candidate_records", []) or []:
            if isinstance(record, Mapping):
                key = tuple(str(action) for action in record.get("action_or_sequence", []))
                by_sequence.setdefault(key, []).append(record)

    aggregates: List[Dict[str, Any]] = []
    for sequence, records in sorted(by_sequence.items()):
        families = {str(record.get("sampling_family", "")) for record in records}
        deltas = [
            float(record.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0)
            for record in records
        ]
        weak = [record for record in records if bool(record.get("weak_signal", False))]
        medium = [record for record in records if bool(record.get("medium_signal", False))]
        terminal = [
            record
            for record in records
            if bool(record.get("candidate_terminal_reentry", False))
        ]
        objective = [
            record
            for record in records
            if bool(record.get("objective_completion_signal", False))
        ]
        aggregates.append(
            {
                "action_or_sequence": list(sequence),
                "safe_stops_tested": len(records),
                "sampling_families_tested": sorted(families),
                "sampling_family_count": len(families),
                "weak_signal_safe_stops": len(weak),
                "medium_signal_safe_stops": len(medium),
                "terminal_reentry_safe_stops": len(terminal),
                "objective_completion_safe_stops": len(objective),
                "mean_delta_vs_hold": round(
                    sum(deltas) / len(deltas) if deltas else 0.0,
                    6,
                ),
                "min_delta_vs_hold": round(min(deltas) if deltas else 0.0, 6),
                "max_delta_vs_hold": round(max(deltas) if deltas else 0.0, 6),
                "sequence_status": sequence_status(records),
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "sequence_aggregate_counted_as_confirmation": False,
            }
        )
    return aggregates


def aggregate_by_dimension(
    per_safe_stop_records: Sequence[Mapping[str, Any]],
    *,
    dimension_key: str,
    output_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in per_safe_stop_records:
        grouped.setdefault(str(record.get(dimension_key, "")), []).append(record)

    aggregates: List[Dict[str, Any]] = []
    for value, records in sorted(grouped.items()):
        candidate_records = [
            candidate
            for record in records
            for candidate in record.get("candidate_records", []) or []
            if isinstance(candidate, Mapping)
        ]
        signal_safe_stops = [
            record
            for record in records
            if int(record.get("weak_signal_candidates", 0) or 0) > 0
        ]
        medium_safe_stops = [
            record
            for record in records
            if int(record.get("medium_signal_candidates", 0) or 0) > 0
        ]
        terminal_safe_stops = [
            record
            for record in records
            if int(record.get("terminal_risk_candidates", 0) or 0) > 0
        ]
        objective_safe_stops = [
            record
            for record in records
            if int(record.get("objective_completion_candidates", 0) or 0) > 0
        ]
        aggregates.append(
            {
                output_key: value,
                "safe_stops": len(records),
                "candidate_records": len(candidate_records),
                "weak_signal_safe_stops": len(signal_safe_stops),
                "medium_signal_safe_stops": len(medium_safe_stops),
                "terminal_risk_safe_stops": len(terminal_safe_stops),
                "objective_completion_safe_stops": len(objective_safe_stops),
                "aggregate_status": dimension_status(records),
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": M3_REFINEMENT_TRUTH_STATUS,
                "dimension_aggregate_counted_as_confirmation": False,
            }
        )
    return aggregates


def roll_up_diverse_validation_status(
    *,
    per_safe_stop_records: Sequence[Mapping[str, Any]],
    family_aggregates: Sequence[Mapping[str, Any]],
) -> str:
    if any(
        int(row.get("objective_completion_candidates", 0) or 0) > 0
        for row in per_safe_stop_records
    ):
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    family_statuses = {
        str(row.get("aggregate_status", "")) for row in family_aggregates
    }
    signal_safe_stops = [
        row
        for row in per_safe_stop_records
        if int(row.get("weak_signal_candidates", 0) or 0) > 0
    ]
    signal_families = {
        str(row.get("sampling_family", "")) for row in signal_safe_stops
    }
    has_signal_family = bool(
        family_statuses
        & {
            REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY,
            LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY,
        }
    )
    has_non_signal_family = bool(
        family_statuses
        & {
            NO_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY,
            TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY,
        }
    )
    if has_signal_family and has_non_signal_family:
        return MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY
    if len(signal_safe_stops) >= 2 and len(signal_families) >= 2:
        return REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY
    if TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY in family_statuses:
        return TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY
    if has_signal_family:
        return LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY
    return NO_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY


def safe_stop_outcome_status(
    *,
    signal_records: Sequence[Mapping[str, Any]],
    terminal_records: Sequence[Mapping[str, Any]],
    objective_records: Sequence[Mapping[str, Any]],
) -> str:
    if objective_records:
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    if terminal_records:
        return TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY
    if signal_records:
        return CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
    return NO_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY


def sequence_status(records: Sequence[Mapping[str, Any]]) -> str:
    if any(bool(record.get("objective_completion_signal", False)) for record in records):
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    if any(bool(record.get("candidate_terminal_reentry", False)) for record in records):
        return TERMINAL_REENTRY_CANDIDATE_ONLY
    weak = [record for record in records if bool(record.get("weak_signal", False))]
    families = {str(record.get("sampling_family", "")) for record in weak}
    if len(weak) >= 2 and len(families) >= 2:
        return REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY
    if weak:
        return LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY
    return NO_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY


def dimension_status(records: Sequence[Mapping[str, Any]]) -> str:
    if any(int(row.get("objective_completion_candidates", 0) or 0) > 0 for row in records):
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    if any(int(row.get("terminal_risk_candidates", 0) or 0) > 0 for row in records):
        return TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY
    signal = [row for row in records if int(row.get("weak_signal_candidates", 0) or 0) > 0]
    if len(signal) >= 2:
        return REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY
    if signal:
        return LOCAL_ONLY_AFTER_DIVERSITY_CANDIDATE_ONLY
    return NO_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY


def summarize_diverse_validation(
    *,
    accepted_safe_stops: Sequence[Mapping[str, Any]],
    selected_cells: Sequence[ObjectiveConversionCell],
    execution_cells: Sequence[Mapping[str, Any]],
    per_safe_stop_records: Sequence[Mapping[str, Any]],
    sequence_aggregates: Sequence[Mapping[str, Any]],
    family_aggregates: Sequence[Mapping[str, Any]],
    validation_status: str,
) -> Dict[str, Any]:
    executed = [cell for cell in execution_cells if bool(cell.get("execution_performed", False))]
    blocked = [cell for cell in execution_cells if not bool(cell.get("execution_performed", False))]
    return {
        "accepted_diverse_safe_stops_consumed": len(accepted_safe_stops),
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
        "safe_stops_with_weak_signal": len(
            [
                row
                for row in per_safe_stop_records
                if int(row.get("weak_signal_candidates", 0) or 0) > 0
            ]
        ),
        "safe_stops_with_medium_signal": len(
            [
                row
                for row in per_safe_stop_records
                if int(row.get("medium_signal_candidates", 0) or 0) > 0
            ]
        ),
        "safe_stops_with_terminal_risk": len(
            [
                row
                for row in per_safe_stop_records
                if int(row.get("terminal_risk_candidates", 0) or 0) > 0
            ]
        ),
        "safe_stops_with_objective_completion": len(
            [
                row
                for row in per_safe_stop_records
                if int(row.get("objective_completion_candidates", 0) or 0) > 0
            ]
        ),
        "sequence_aggregates": len(sequence_aggregates),
        "sampling_family_aggregates": len(family_aggregates),
        "validation_outcome_status": validation_status,
        "mixed_by_safe_stop_family": (
            validation_status == MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY
        ),
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "diverse_safe_stop_validation_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "validation_outcome_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def safe_stop_cell_context(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "safe_stop_id": str(record.get("safe_stop_id", "")),
        "source_plan_id": str(record.get("source_plan_id", "")),
        "sampling_family": str(record.get("sampling_family", "")),
        "terminal_horizon_remaining": terminal_horizon_remaining(record),
        "terminal_horizon_band": terminal_horizon_band(record),
        "hold_baseline_band": hold_baseline_band(
            float(record.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0)
        ),
    }


def terminal_horizon_remaining(record: Mapping[str, Any]) -> int | None:
    estimate = dict(record.get("terminal_horizon_estimate", {}) or {})
    remaining = estimate.get("estimated_moves_remaining")
    return None if remaining is None else int(remaining)


def terminal_horizon_band(record: Mapping[str, Any]) -> str:
    remaining = terminal_horizon_remaining(record)
    if remaining is None:
        return "horizon_unknown"
    if remaining >= 55:
        return "horizon_far_ge_55"
    if remaining >= 45:
        return "horizon_mid_45_54"
    return "horizon_near_lt_45"


def hold_baseline_band(value: float) -> str:
    taps = float(value)
    if taps < 50.0:
        return "hold_low_lt_50"
    if taps < 120.0:
        return "hold_mid_50_119"
    return "hold_high_ge_120"


def source_g3_diversity_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}) or {})
    return {
        "source_diversity_status": str(summary.get("diversity_status", "")),
        "source_accepted_diverse_safe_stops": int(
            summary.get("accepted_diverse_safe_stops", 0) or 0
        ),
        "source_objective_conversion_sequences_tested": bool(
            summary.get("objective_conversion_sequences_tested", False)
        ),
        "source_g3_counted_as_confirmation": False,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
    }


def _make_default_cell_executor(
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


def _make_default_captured_builder(
    *,
    environments_dir: str | Path | None,
) -> Callable[[Mapping[str, Any]], CapturedSafeStop]:
    def builder(record: Mapping[str, Any]) -> CapturedSafeStop:
        return captured_safe_stop_from_g3_record(
            record,
            environments_dir=environments_dir,
        )

    return builder


def captured_safe_stop_from_g3_record(
    record: Mapping[str, Any],
    *,
    environments_dir: str | Path | None,
) -> CapturedSafeStop:
    from theory.p3.objective_aware_abstract_policy_probe import (
        DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH,
    )

    adapter_payload = _load_json(DEFAULT_OBJECTIVE_AWARE_ADAPTER_OUTPUT_PATH)
    adapter = dict(adapter_payload.get("objective_aware_policy_adapter", {}) or {})
    prefix = tuple(dict(step) for step in record.get("captured_prefix", []) or [])
    prefix_step_dicts = replay_prefix_step_dicts(
        prefix=prefix,
        environments_dir=environments_dir,
        game_id="bp35-0a0ad940",
        adapter=adapter,
    )
    return CapturedSafeStop(
        prefix=prefix,
        captured_prefix_hash=str(record.get("captured_prefix_hash", "")),
        safe_stop_state_hash=str(record.get("safe_stop_state_hash", "")),
        hold_baseline_terminal_adjusted_progress=float(
            record.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
        ),
        hold_baseline_levels_completed=int(
            record.get("hold_baseline_levels_completed", 0) or 0
        ),
        hold_baseline_terminal=bool(record.get("hold_baseline_terminal", False)),
        provenance={
            "safe_stop_state_source": "M3.G3",
            "safe_stop_id": str(record.get("safe_stop_id", "")),
            "sampling_family": str(record.get("sampling_family", "")),
            "source_plan_id": str(record.get("source_plan_id", "")),
        },
        capture_config={
            "selection_rule": "m3_g3_accepted_diverse_safe_stop",
            "selection_counted_as_support": False,
        },
        adapter=adapter,
        prefix_step_dicts=tuple(prefix_step_dicts),
    )


def replay_prefix_step_dicts(
    *,
    prefix: Sequence[Mapping[str, Any]],
    environments_dir: str | Path | None,
    game_id: str,
    adapter: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    from theory.m1.polymorphic_a25_adapter import _step_env_action
    from theory.m2.m3_execution_smoke import _make_env, _reset_env
    from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
    from theory.p1.bp35_sage_candidate_policy_probe import (
        measure_probe_metrics,
        state_signature,
    )
    from theory.p3.abstract_mechanic_policy_probe import (
        action_has_actor_effect,
        action_has_relation_effect,
        concrete_named_action,
        is_game_over,
    )
    from theory.real_env_option_adapter import snapshot_frame

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    env = _make_env(game_id, env_dir)
    frame = _reset_env(env)
    seen_states: set[str] = set()
    initial = snapshot_frame(frame)
    seen_states.add(state_signature(initial.grid, initial.levels_completed, initial.game_state))
    steps: List[Dict[str, Any]] = []
    for index, step in enumerate(prefix):
        before = snapshot_frame(frame)
        action_name = str(step.get("action", ""))
        action_args = dict(step.get("action_args", {}) or {})
        selected = concrete_named_action(
            list(_valid_actions(env)),
            action_name,
            action_args,
        )
        if selected is None or is_game_over(before.game_state):
            break
        after_frame = _step_env_action(env, selected)
        after = snapshot_frame(
            after_frame,
            fallback_available_actions=before.available_actions,
        )
        measurements = measure_probe_metrics(before.grid, after.grid, action_args)
        after_signature = state_signature(after.grid, after.levels_completed, after.game_state)
        changed_pixels = float(
            measurements["changed_pixels"].get("changed_pixels", 0) or 0
        )
        relation_expected = action_has_relation_effect(adapter, action_name)
        actor_expected = action_has_actor_effect(adapter, action_name)
        terminal = is_game_over(after.game_state)
        cycle = after_signature in seen_states
        useful_new_state = bool(
            changed_pixels > 0
            and not cycle
            and after.levels_completed >= before.levels_completed
            and not terminal
        )
        steps.append(
            {
                "step": index,
                "policy_selected_action": action_name,
                "action_args": action_args,
                "changed_pixels": changed_pixels,
                "actor_relation_delta_count": int(
                    relation_expected and changed_pixels > 0
                ),
                "action_effect_usefulness": int(actor_expected and changed_pixels > 0),
                "new_relation_state": int(relation_expected and useful_new_state),
                "useful_new_state": useful_new_state,
                "dead_end_or_cycle": cycle,
                "state_signature_after": after_signature,
                "levels_after": int(after.levels_completed),
                "game_state_after": str(after.game_state),
                "terminal_state_after": terminal,
                "measurements": measurements,
            }
        )
        seen_states.add(after_signature)
        frame = after_frame
        if terminal:
            break
    return tuple(steps)


def _validate_request_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("request source support must remain 0")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("request source must not write A32/A33")


def _validate_source_g3_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if str(payload.get("diversity_status", summary.get("diversity_status", ""))) != (
        SUFFICIENT_FOR_M3_G4
    ):
        raise ValueError("M3.G4 requires M3.G3 diversity_status=SUFFICIENT_FOR_M3_G4")
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("M3.G3 source support must remain 0")
    if bool(payload.get("objective_conversion_sequences_tested", False)) or bool(
        summary.get("objective_conversion_sequences_tested", False)
    ):
        raise ValueError("M3.G3 source must not already test objective sequences")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("M3.G3 source must not write A32/A33")
    if not payload.get("accepted_diverse_safe_stops"):
        raise ValueError("M3.G4 requires accepted diverse safe-stops")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_objective_conversion_diverse_safe_stop_validation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH
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
        description="Run M3.G4 diverse-safe-stop objective-conversion validation.",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--source-g3",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-safe-stops", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_diverse_safe_stop_validation(
        requests_path=args.requests,
        source_g3_path=args.source_g3,
        environments_dir=args.environments_dir,
        max_safe_stops=args.max_safe_stops,
    )
    write_objective_conversion_diverse_safe_stop_validation(payload, args.out)
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
