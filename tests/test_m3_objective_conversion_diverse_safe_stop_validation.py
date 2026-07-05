"""Tests for M3.G4 diverse-safe-stop objective-conversion validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest

from theory.m3.objective_conversion_diverse_safe_stop_validation import (
    DEFAULT_OBJECTIVE_CONVERSION_DIVERSE_SAFE_STOP_VALIDATION_OUTPUT_PATH,
    MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY,
    OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY,
    REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY,
    TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY,
    captured_safe_stop_from_g3_record,
    run_objective_conversion_diverse_safe_stop_validation,
)
from theory.m3.objective_conversion_experiment_executor import (
    CapturedSafeStop,
    ObjectiveConversionCell,
    measured_cell_result,
)
from theory.m3.objective_conversion_experiment_planner import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
)
from theory.m3.objective_conversion_safe_stop_diversity_sampler import (
    DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH,
    SUFFICIENT_FOR_M3_G4,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUESTS_PATH = REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH
SOURCE_G3_PATH = (
    REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_SAFE_STOP_DIVERSITY_OUTPUT_PATH
)


def _safe_stop(
    index: int,
    *,
    family: str,
    hold: float = 100.0,
    remaining: int = 60,
) -> Mapping[str, object]:
    return {
        "safe_stop_id": f"safe_stop::{index}",
        "source_plan_id": f"plan::{index}",
        "sampling_family": family,
        "captured_prefix": [{"action": "ACTION3", "action_args": {}}],
        "captured_prefix_len": 1,
        "captured_prefix_hash": f"prefix::{index}",
        "safe_stop_state_hash": f"state::{index}",
        "relation_state_signature": f"relation::{index}",
        "terminal_horizon_estimate": {
            "estimated_moves_remaining": remaining,
            "source": "stub",
        },
        "hold_baseline_terminal_adjusted_progress": hold,
        "hold_baseline_levels_completed": 0,
        "hold_baseline_terminal": False,
        "replay_exact": True,
        "non_terminal": True,
        "terminal_safe": True,
        "hold_baseline_measurable": True,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
    }


def _source_payload(stops: list[Mapping[str, object]]) -> Mapping[str, object]:
    return {
        "summary": {
            "diversity_status": SUFFICIENT_FOR_M3_G4,
            "accepted_diverse_safe_stops": len(stops),
            "objective_conversion_sequences_tested": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
        },
        "diversity_status": SUFFICIENT_FOR_M3_G4,
        "accepted_diverse_safe_stops": stops,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "objective_conversion_sequences_tested": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _write_source(tmp_path: Path, stops: list[Mapping[str, object]]) -> Path:
    path = tmp_path / "g3_source.json"
    path.write_text(json.dumps(_source_payload(stops)), encoding="utf-8")
    return path


def _captured_builder(record: Mapping[str, object]) -> CapturedSafeStop:
    return CapturedSafeStop(
        prefix=tuple(dict(step) for step in record.get("captured_prefix", []) or []),
        captured_prefix_hash=str(record.get("captured_prefix_hash", "")),
        safe_stop_state_hash=str(record.get("safe_stop_state_hash", "")),
        hold_baseline_terminal_adjusted_progress=float(
            record.get("hold_baseline_terminal_adjusted_progress", 0.0) or 0.0
        ),
        hold_baseline_levels_completed=0,
        hold_baseline_terminal=False,
        provenance={
            "sampling_family": str(record.get("sampling_family", "")),
            "safe_stop_id": str(record.get("safe_stop_id", "")),
        },
        capture_config={"selection_counted_as_support": False},
        adapter={},
    )


def _executor_for_mode(mode_by_family: Mapping[str, str]):
    def executor(
        cell: ObjectiveConversionCell,
        captured: CapturedSafeStop,
    ) -> Mapping[str, object]:
        family = str(captured.provenance.get("sampling_family", ""))
        mode = mode_by_family.get(family, "signal")
        hold = float(captured.hold_baseline_terminal_adjusted_progress)
        actions = tuple(cell.action_or_sequence or ())
        terminal = False
        levels = 0
        objective = False
        if cell.condition_kind == "hold":
            taps = hold
        elif cell.condition_kind == "relation_progress_policy":
            taps = 0.0
            terminal = True
        elif mode == "flat":
            taps = hold
        elif mode == "terminal":
            taps = 0.0
            terminal = True
        elif mode == "objective" and actions == ("ACTION6", "ACTION3"):
            taps = hold + 100.0
            levels = 1
            objective = True
        elif actions == ("ACTION6",):
            taps = hold + 5.0
        elif actions in (("ACTION6", "ACTION3"), ("ACTION6", "ACTION4")):
            taps = hold + 15.0
        else:
            taps = hold
        return measured_cell_result(
            cell,
            captured=captured,
            candidate_terminal_adjusted_progress=taps,
            candidate_levels_completed=levels,
            candidate_terminal_reentry=terminal,
            objective_completion_signal=objective,
            diagnostics={
                "changed_pixels": 1.0,
                "relation_delta_after_stop": 1,
                "new_relation_states": 1,
                "distance_decreases_count": 1,
                "objective_readiness_signature_delta": levels,
            },
            replayed_prefix_hash=captured.captured_prefix_hash,
            safe_stop_state_hash=captured.safe_stop_state_hash,
            post_stop_steps_executed=int(cell.post_stop_horizon),
        )

    return executor


def _run(tmp_path: Path, stops: list[Mapping[str, object]], mode_by_family):
    return run_objective_conversion_diverse_safe_stop_validation(
        requests_path=REQUESTS_PATH,
        source_g3_path=_write_source(tmp_path, stops),
        captured_builder=_captured_builder,
        cell_executor=_executor_for_mode(mode_by_family),
    )


def test_reproduced_signal_across_multiple_families(tmp_path: Path) -> None:
    stops = [
        _safe_stop(1, family="base_prefix_truncation", hold=40, remaining=60),
        _safe_stop(2, family="action6_perturbation", hold=25, remaining=61),
    ]
    payload = _run(tmp_path, stops, {})
    assert payload["validation_outcome_status"] == (
        REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY
    )
    assert payload["summary"]["safe_stops_with_weak_signal"] == 2
    assert payload["summary"]["safe_stops_with_medium_signal"] == 2


def test_mixed_by_safe_stop_family_when_scope_differs(tmp_path: Path) -> None:
    stops = [
        _safe_stop(1, family="base_prefix_truncation"),
        _safe_stop(2, family="single_action_burst"),
    ]
    payload = _run(tmp_path, stops, {"single_action_burst": "flat"})
    assert payload["validation_outcome_status"] == (
        MIXED_BY_SAFE_STOP_FAMILY_CANDIDATE_ONLY
    )
    statuses = {
        row["aggregate_status"] for row in payload["sampling_family_aggregates"]
    }
    assert REPRODUCED_DIVERSE_SAFE_STOP_SIGNAL_CANDIDATE_ONLY not in statuses
    assert payload["summary"]["mixed_by_safe_stop_family"] is True


def test_terminal_risk_status_when_signal_reopens_terminal(tmp_path: Path) -> None:
    stops = [_safe_stop(1, family="action6_perturbation")]
    payload = _run(tmp_path, stops, {"action6_perturbation": "terminal"})
    assert payload["validation_outcome_status"] == (
        TERMINAL_RISK_REAPPEARS_AFTER_DIVERSITY_CANDIDATE_ONLY
    )
    assert payload["summary"]["safe_stops_with_terminal_risk"] == 1


def test_objective_completion_takes_top_candidate_status(tmp_path: Path) -> None:
    stops = [_safe_stop(1, family="base_prefix_truncation")]
    payload = _run(tmp_path, stops, {"base_prefix_truncation": "objective"})
    assert payload["validation_outcome_status"] == (
        OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    )
    assert payload["summary"]["safe_stops_with_objective_completion"] == 1


def test_outputs_family_horizon_and_hold_baseline_splits(tmp_path: Path) -> None:
    stops = [
        _safe_stop(1, family="base_prefix_truncation", hold=40, remaining=60),
        _safe_stop(2, family="action6_perturbation", hold=125, remaining=44),
    ]
    payload = _run(tmp_path, stops, {})
    assert payload["sampling_family_aggregates"]
    assert payload["terminal_horizon_aggregates"]
    assert payload["hold_baseline_aggregates"]
    assert {
        row["hold_baseline_band"] for row in payload["hold_baseline_aggregates"]
    } >= {"hold_low_lt_50", "hold_high_ge_120"}


def test_guardrails_locked(tmp_path: Path) -> None:
    payload = _run(tmp_path, [_safe_stop(1, family="base_prefix_truncation")], {})
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert payload["diverse_safe_stop_validation_counted_as_confirmation"] is False


def test_rejects_g3_source_without_sufficient_diversity(tmp_path: Path) -> None:
    path = _write_source(tmp_path, [_safe_stop(1, family="base_prefix_truncation")])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["diversity_status"] = "INSUFFICIENT_DIVERSITY"
    payload["summary"]["diversity_status"] = "INSUFFICIENT_DIVERSITY"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        run_objective_conversion_diverse_safe_stop_validation(
            requests_path=REQUESTS_PATH,
            source_g3_path=path,
            captured_builder=_captured_builder,
            cell_executor=_executor_for_mode({}),
        )


def test_real_g3_safe_stop_can_be_hydrated() -> None:
    payload = json.loads(SOURCE_G3_PATH.read_text(encoding="utf-8"))
    record = payload["accepted_diverse_safe_stops"][0]
    captured = captured_safe_stop_from_g3_record(record, environments_dir=None)
    assert captured.prefix
    assert captured.prefix_step_dicts
    assert captured.safe_stop_state_hash == record["safe_stop_state_hash"]
