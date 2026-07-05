"""Tests for M3.G2 multi-safe-stop objective-conversion validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from theory.m3.objective_conversion_experiment_consolidation import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH,
)
from theory.m3.objective_conversion_experiment_executor import (
    CapturedSafeStop,
    ObjectiveConversionCell,
    measured_cell_result,
)
from theory.m3.objective_conversion_experiment_planner import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH,
)
from theory.m3.objective_conversion_multi_safe_stop_validation import (
    DEFAULT_TARGET_ACTION_SEQUENCES,
    LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY,
    MULTI_SAFE_STOP_CONTEXTS,
    REPRODUCED_MULTI_SAFE_STOP_SIGNAL_CANDIDATE_ONLY,
    TERMINAL_RISK_REAPPEARS_CANDIDATE_ONLY,
    SafeStopCaptureSpec,
    run_objective_conversion_multi_safe_stop_validation,
    select_multi_safe_stop_cells,
    target_action_sequences_from_source,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUESTS_PATH = REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_REQUESTS_OUTPUT_PATH
SOURCE_PATH = (
    REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_CONSOLIDATION_OUTPUT_PATH
)


def _captured(spec: SafeStopCaptureSpec, *, duplicate: bool = False) -> CapturedSafeStop:
    suffix = "duplicate" if duplicate else f"{spec.condition}-{spec.budget}-{spec.tie_break_seed}"
    prefix = (
        {"action": "ACTION3", "action_args": {}},
        {"action": "ACTION4", "action_args": {}},
        {"action": "ACTION3", "action_args": {}},
    )
    return CapturedSafeStop(
        prefix=prefix,
        captured_prefix_hash=f"prefix::{suffix}",
        safe_stop_state_hash=f"safe_stop::{suffix}",
        hold_baseline_terminal_adjusted_progress=(
            100.0 if duplicate else 100.0 + float(spec.tie_break_seed)
        ),
        hold_baseline_levels_completed=0,
        hold_baseline_terminal=False,
        provenance={
            "safe_stop_state_source": "P3.G1",
            "safe_stop_policy_condition": spec.condition,
            "stop_trigger_reason": "objective_aware_terminal_risk_score_stop",
            "base_state_family": "terminal_safe_stop_or_avoidance_state",
            "budget": spec.budget,
            "tie_break_seed": spec.tie_break_seed,
        },
        capture_config={"selection_counted_as_support": False},
        adapter={},
    )


def _stub_cell_executor(
    cell: ObjectiveConversionCell,
    captured: CapturedSafeStop,
    *,
    terminal_safe_stop_hash: str = "",
) -> Mapping[str, object]:
    hold = float(captured.hold_baseline_terminal_adjusted_progress)
    actions = tuple(cell.action_or_sequence or ())
    if cell.condition_kind == "hold":
        taps = hold
        terminal = False
    elif cell.condition_kind == "relation_progress_policy":
        taps = 0.0
        terminal = True
    elif actions == ("ACTION6",):
        taps = hold + 5.0
        terminal = False
    elif actions == ("ACTION6", "ACTION3"):
        terminal = captured.safe_stop_state_hash == terminal_safe_stop_hash
        taps = 0.0 if terminal else hold + 15.0
    elif actions == ("ACTION6", "ACTION4"):
        taps = hold + 12.0
        terminal = False
    else:
        taps = hold
        terminal = False

    return measured_cell_result(
        cell,
        captured=captured,
        candidate_terminal_adjusted_progress=taps,
        candidate_levels_completed=0,
        candidate_terminal_reentry=terminal,
        objective_completion_signal=False,
        diagnostics={
            "changed_pixels": 1.0,
            "relation_delta_after_stop": 1,
            "new_relation_states": 1,
            "distance_decreases_count": 1,
            "objective_readiness_signature_delta": 0,
        },
        replayed_prefix_hash=captured.captured_prefix_hash,
        safe_stop_state_hash=captured.safe_stop_state_hash,
        post_stop_steps_executed=int(cell.post_stop_horizon),
    )


def _specs() -> tuple[SafeStopCaptureSpec, SafeStopCaptureSpec]:
    return (
        SafeStopCaptureSpec(
            condition="objective_aware_abstract_policy_lambda_0",
            budget=48,
            tie_break_seed=0,
        ),
        SafeStopCaptureSpec(
            condition="objective_aware_abstract_policy_lambda_0",
            budget=64,
            tie_break_seed=1,
        ),
    )


def _run_with_stubs(*, duplicate: bool = False, terminal_hash: str = ""):
    return run_objective_conversion_multi_safe_stop_validation(
        requests_path=REQUESTS_PATH,
        source_consolidation_path=SOURCE_PATH,
        safe_stop_specs=_specs(),
        safe_stop_capturer=lambda spec, requests: _captured(spec, duplicate=duplicate),
        cell_executor=lambda cell, captured: _stub_cell_executor(
            cell,
            captured,
            terminal_safe_stop_hash=terminal_hash,
        ),
    )


def test_retests_action6_led_signals_and_controls() -> None:
    payload = _run_with_stubs()
    assert payload["summary"]["target_action_sequences"] == [
        list(sequence) for sequence in DEFAULT_TARGET_ACTION_SEQUENCES
    ]
    assert payload["summary"]["selected_cells_per_safe_stop"] == 6
    assert payload["summary"]["candidate_cells_executed"] == 6
    assert payload["summary"]["hold_cells_executed"] == 2
    assert payload["summary"]["relation_progress_cells_executed"] == 4


def test_reproduced_signal_requires_multiple_unique_safe_stops() -> None:
    payload = _run_with_stubs()
    assert payload["validation_outcome_status"] == (
        REPRODUCED_MULTI_SAFE_STOP_SIGNAL_CANDIDATE_ONLY
    )
    assert payload["summary"]["safe_stop_context_diversity"] == MULTI_SAFE_STOP_CONTEXTS
    assert payload["summary"]["unique_safe_stop_captures"] == 2
    assert payload["summary"]["reproduced_signal_safe_stops"] == 2


def test_duplicate_safe_stops_are_not_counted_as_independent() -> None:
    payload = _run_with_stubs(duplicate=True)
    assert payload["validation_outcome_status"] == (
        LOCAL_SAFE_STOP_ONLY_SIGNAL_CANDIDATE_ONLY
    )
    assert payload["summary"]["unique_safe_stop_captures"] == 1
    assert payload["summary"]["duplicate_safe_stop_captures"] == 1
    assert payload["summary"]["duplicate_safe_stop_captures_counted_as_independent"] is False


def test_terminal_reentry_takes_terminal_risk_status() -> None:
    terminal_hash = "safe_stop::objective_aware_abstract_policy_lambda_0-64-1"
    payload = _run_with_stubs(terminal_hash=terminal_hash)
    assert payload["validation_outcome_status"] == TERMINAL_RISK_REAPPEARS_CANDIDATE_ONLY
    assert payload["summary"]["terminal_risk_safe_stops"] == 1


def test_target_selection_uses_source_action6_led_signal_plus_ablation() -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    targets = target_action_sequences_from_source(source)
    assert ("ACTION6",) in targets
    assert ("ACTION6", "ACTION3") in targets
    assert ("ACTION6", "ACTION4") in targets


def test_cell_selection_keeps_hold_and_horizon_matched_relation_controls() -> None:
    from theory.m3.objective_conversion_experiment_executor import (
        build_execution_cells,
        ready_objective_conversion_requests,
        unique_execution_cells,
    )

    requests = ready_objective_conversion_requests(
        json.loads(REQUESTS_PATH.read_text(encoding="utf-8"))
    )
    planned, _links = build_execution_cells(requests)
    selected = select_multi_safe_stop_cells(
        unique_execution_cells(planned),
        target_sequences=DEFAULT_TARGET_ACTION_SEQUENCES,
    )
    assert [cell.condition_kind for cell in selected].count("hold") == 1
    assert [cell.condition_kind for cell in selected].count("relation_progress_policy") == 2
    assert [cell.condition_kind for cell in selected].count("candidate") == 3


def test_guardrails_locked() -> None:
    payload = _run_with_stubs()
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["revision_performed"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert payload["multi_safe_stop_validation_counted_as_confirmation"] is False
    for safe_stop in payload["per_safe_stop_validation_records"]:
        assert safe_stop["support"] == 0
        for record in safe_stop["candidate_records"]:
            assert record["support"] == 0
            assert record["truth_status"] == "NOT_EVALUATED_BY_M3"


def test_rejects_source_consolidation_with_support(tmp_path: Path) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    source["support"] = 1
    bad_path = tmp_path / "bad_source.json"
    bad_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError):
        run_objective_conversion_multi_safe_stop_validation(
            requests_path=REQUESTS_PATH,
            source_consolidation_path=bad_path,
            safe_stop_specs=_specs(),
            safe_stop_capturer=lambda spec, requests: _captured(spec),
            cell_executor=_stub_cell_executor,
        )
