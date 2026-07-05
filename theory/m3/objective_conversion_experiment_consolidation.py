"""M3.G1.3 consolidation (interpretation) of objective-conversion measurements.

Reads the M3.G1.2 measurements and produces CANDIDATE-ONLY outcome /
interpretation statuses. It never produces a scientific *verdict* (the word is
intentionally absent from the artifact), never confirms/refutes/revises, and
never writes A32/A33. A32 remains the only verdict location.

Decision rule (per candidate action/sequence):
- central:   candidate beats hold_or_stop_state on
             delta_terminal_adjusted_progress_vs_hold > 0 AND terminal_reentry == false
- secondary: candidate beats the horizon-matched relation_progress_policy control

Candidate-only outcome statuses:
- CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
- NO_CONVERSION_SIGNAL_CANDIDATE_ONLY
- TERMINAL_REENTRY_CANDIDATE_ONLY
- MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY
- OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY  (even a real completion stays candidate-only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .objective_conversion_experiment_executor import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH,
    LOW_SINGLE_SAFE_STOP,
)


DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "objective_conversion_experiment_consolidation.json"
)
CONSOLIDATION_SCHEMA_VERSION = "m3.objective_conversion_experiment_consolidation.v1"

# Candidate-only outcome/interpretation statuses (never a "verdict").
CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY = "CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY"
NO_CONVERSION_SIGNAL_CANDIDATE_ONLY = "NO_CONVERSION_SIGNAL_CANDIDATE_ONLY"
TERMINAL_REENTRY_CANDIDATE_ONLY = "TERMINAL_REENTRY_CANDIDATE_ONLY"
MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY = "MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY"
OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY = (
    "OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY"
)

NEXT_STEP_MULTI_SAFE_STOP = "M3.G2 multi-safe-stop objective-conversion validation"


def run_objective_conversion_experiment_consolidation(
    *,
    results_path: str | Path = (
        DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH
    ),
) -> Dict[str, Any]:
    payload = _load_json(results_path)
    _validate_source_payload(payload)
    cells = [
        dict(row)
        for row in payload.get("execution_cells", []) or []
        if isinstance(row, Mapping)
    ]
    hold_baseline = float(
        (payload.get("safe_stop_capture", {}) or {}).get(
            "hold_baseline_terminal_adjusted_progress", 0.0
        )
        or 0.0
    )
    relation_summary = relation_progress_control_summary(cells)

    candidate_records = [
        candidate_outcome_record(
            cell,
            hold_baseline=hold_baseline,
            relation_summary=relation_summary,
        )
        for cell in cells
        if str(cell.get("condition_kind", "")) == "candidate"
        and bool(cell.get("execution_performed", False))
    ]
    best_candidate = select_best_candidate(candidate_records)
    consolidation_status = roll_up_status(candidate_records)

    return {
        "config": {
            "schema_version": CONSOLIDATION_SCHEMA_VERSION,
            "results_path": str(results_path),
            "inputs_read": ["M3.G1.2"],
            "artifacts_not_modified": ["M2", "M3.G1.1", "M3.G1.2", "A32", "A33"],
            "stage_produces": "candidate_only_outcome_interpretation",
            "central_decision_rule": (
                "candidate beats hold_or_stop_state on "
                "delta_terminal_adjusted_progress_vs_hold > 0 "
                "AND terminal_reentry == false"
            ),
            "secondary_decision_rule": "candidate beats relation_progress_policy",
        },
        "summary": summarize_consolidation(
            candidate_records=candidate_records,
            best_candidate=best_candidate,
            consolidation_status=consolidation_status,
            hold_baseline=hold_baseline,
            relation_summary=relation_summary,
        ),
        "candidate_outcome_records": candidate_records,
        "best_candidate": best_candidate,
        "relation_progress_control_summary": relation_summary,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "safe_stop_context_diversity": LOW_SINGLE_SAFE_STOP,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "consolidation_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def candidate_outcome_record(
    cell: Mapping[str, Any],
    *,
    hold_baseline: float,
    relation_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    measurements = dict(cell.get("measurements", {}) or {})
    delta_vs_hold = float(
        measurements.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0
    )
    terminal_reentry = bool(measurements.get("candidate_terminal_reentry", False))
    objective_completion = bool(measurements.get("objective_completion_signal", False))
    levels = int(measurements.get("candidate_levels_completed_after_rollout", 0) or 0)

    beats_hold = bool(delta_vs_hold > 0 and not terminal_reentry)
    relation_taps = float(relation_summary.get("best_relation_taps", 0.0) or 0.0)
    candidate_taps = float(
        measurements.get("candidate_terminal_adjusted_progress_after_stop", 0.0) or 0.0
    )
    relation_terminal = bool(
        relation_summary.get("all_relation_controls_terminal_reentry", False)
    )
    beats_relation = bool(
        candidate_taps > relation_taps or (not terminal_reentry and relation_terminal)
    )

    status = candidate_status(
        beats_hold=beats_hold,
        terminal_reentry=terminal_reentry,
        objective_completion=objective_completion,
    )
    return {
        "condition_id": str(cell.get("condition_id", "")),
        "action_or_sequence": list(cell.get("action_or_sequence") or []),
        "post_stop_horizon": int(cell.get("post_stop_horizon", 0) or 0),
        "delta_terminal_adjusted_progress_vs_hold": round(delta_vs_hold, 6),
        "candidate_terminal_adjusted_progress_after_stop": round(candidate_taps, 6),
        "hold_or_stop_state_terminal_adjusted_progress_after_stop": round(
            hold_baseline, 6
        ),
        "candidate_terminal_reentry": terminal_reentry,
        "objective_completion_signal": objective_completion,
        "candidate_levels_completed_after_rollout": levels,
        "beats_hold_or_stop_state": beats_hold,
        "beats_relation_progress_policy": beats_relation,
        "candidate_outcome_status": status,
        "safe_stop_replay_exact": bool(cell.get("safe_stop_replay_exact", False)),
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "candidate_outcome_status_counted_as_scientific_verdict": False,
    }


def candidate_status(
    *,
    beats_hold: bool,
    terminal_reentry: bool,
    objective_completion: bool,
) -> str:
    if objective_completion:
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    if terminal_reentry:
        return TERMINAL_REENTRY_CANDIDATE_ONLY
    if beats_hold:
        return CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
    return NO_CONVERSION_SIGNAL_CANDIDATE_ONLY


def relation_progress_control_summary(
    cells: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    relation_cells = [
        dict(cell)
        for cell in cells
        if str(cell.get("condition_kind", "")) == "relation_progress_policy"
        and bool(cell.get("execution_performed", False))
    ]
    taps = [
        float(
            (cell.get("measurements", {}) or {}).get(
                "candidate_terminal_adjusted_progress_after_stop", 0.0
            )
            or 0.0
        )
        for cell in relation_cells
    ]
    reentry = [
        bool(
            (cell.get("measurements", {}) or {}).get(
                "candidate_terminal_reentry", False
            )
        )
        for cell in relation_cells
    ]
    return {
        "relation_progress_controls_executed": len(relation_cells),
        "best_relation_taps": max(taps) if taps else 0.0,
        "all_relation_controls_terminal_reentry": bool(reentry) and all(reentry),
        "any_relation_control_terminal_reentry": any(reentry),
    }


def select_best_candidate(
    candidate_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not candidate_records:
        return {}
    best = max(
        candidate_records,
        key=lambda row: (
            1 if not bool(row.get("candidate_terminal_reentry")) else 0,
            float(row.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0),
            int(row.get("candidate_levels_completed_after_rollout", 0) or 0),
        ),
    )
    return dict(best)


def roll_up_status(candidate_records: Sequence[Mapping[str, Any]]) -> str:
    if not candidate_records:
        return NO_CONVERSION_SIGNAL_CANDIDATE_ONLY
    statuses = [str(row.get("candidate_outcome_status", "")) for row in candidate_records]
    if OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY in statuses:
        return OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    has_conversion = CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY in statuses
    has_terminal = TERMINAL_REENTRY_CANDIDATE_ONLY in statuses
    has_no_signal = NO_CONVERSION_SIGNAL_CANDIDATE_ONLY in statuses
    if has_conversion and (has_terminal or has_no_signal):
        return MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY
    if has_conversion:
        return CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
    if has_terminal and not has_no_signal:
        return TERMINAL_REENTRY_CANDIDATE_ONLY
    return NO_CONVERSION_SIGNAL_CANDIDATE_ONLY


def summarize_consolidation(
    *,
    candidate_records: Sequence[Mapping[str, Any]],
    best_candidate: Mapping[str, Any],
    consolidation_status: str,
    hold_baseline: float,
    relation_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    conversion_signals = [
        row
        for row in candidate_records
        if str(row.get("candidate_outcome_status"))
        == CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
    ]
    any_conversion = bool(conversion_signals) or (
        consolidation_status
        in (
            CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY,
            MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY,
            OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY,
        )
    )
    return {
        "candidate_records": len(candidate_records),
        "conversion_signal_candidates": len(conversion_signals),
        "terminal_reentry_candidates": len(
            [
                row
                for row in candidate_records
                if str(row.get("candidate_outcome_status"))
                == TERMINAL_REENTRY_CANDIDATE_ONLY
            ]
        ),
        "no_conversion_signal_candidates": len(
            [
                row
                for row in candidate_records
                if str(row.get("candidate_outcome_status"))
                == NO_CONVERSION_SIGNAL_CANDIDATE_ONLY
            ]
        ),
        "consolidation_outcome_status": consolidation_status,
        "best_candidate_action_or_sequence": list(
            best_candidate.get("action_or_sequence", []) or []
        ),
        "best_candidate_delta_vs_hold": float(
            best_candidate.get("delta_terminal_adjusted_progress_vs_hold", 0.0) or 0.0
        ),
        "hold_or_stop_state_terminal_adjusted_progress": round(hold_baseline, 6),
        "relation_progress_control_summary": dict(relation_summary),
        "safe_stop_context_diversity": LOW_SINGLE_SAFE_STOP,
        "next_step": NEXT_STEP_MULTI_SAFE_STOP if any_conversion else "",
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "m2_hypothesis_counted_as_confirmation": False,
        "experiment_result_counted_as_scientific_verdict": False,
        "consolidation_status_counted_as_scientific_verdict": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
        "a32_remains_only_verdict_location": True,
    }


def _validate_source_payload(payload: Mapping[str, Any]) -> None:
    summary = dict(payload.get("summary", {}) or {})
    if int(payload.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("source support must remain 0")
    if int(summary.get("support", 0) or 0) != 0:
        raise ValueError("source summary support must remain 0")
    if bool(payload.get("revision_performed", False)) or bool(
        summary.get("revision_performed", False)
    ):
        raise ValueError("source revision_performed must be false")
    if bool(payload.get("experiment_result_counted_as_scientific_verdict", False)):
        raise ValueError("experiment result cannot be scientific verdict")
    if bool(payload.get("a32_write_performed", False)) or bool(
        payload.get("a33_write_performed", False)
    ):
        raise ValueError("source must not write A32/A33")
    if not bool(payload.get("execution_performed", False)):
        raise ValueError("consolidation requires executed measurements")


def write_objective_conversion_experiment_consolidation(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH
    ),
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
        description="Consolidate M3.G1.3 objective-conversion measurements.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_objective_conversion_experiment_consolidation(
        results_path=args.results,
    )
    write_objective_conversion_experiment_consolidation(payload, args.out)
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
