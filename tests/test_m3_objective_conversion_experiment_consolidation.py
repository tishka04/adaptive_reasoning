"""Tests for M3.G1.3 objective-conversion experiment consolidation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.m3.objective_conversion_experiment_executor import (
    DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH,
)
from theory.m3.objective_conversion_experiment_consolidation import (
    CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY,
    MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY,
    NO_CONVERSION_SIGNAL_CANDIDATE_ONLY,
    OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY,
    TERMINAL_REENTRY_CANDIDATE_ONLY,
    candidate_status,
    roll_up_status,
    run_objective_conversion_experiment_consolidation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_EXPERIMENT_RESULTS_OUTPUT_PATH


def _run():
    return run_objective_conversion_experiment_consolidation(results_path=RESULTS_PATH)


def test_consolidation_produces_candidate_records() -> None:
    payload = _run()
    assert payload["summary"]["candidate_records"] >= 1
    for record in payload["candidate_outcome_records"]:
        assert record["candidate_outcome_status"].endswith("_CANDIDATE_ONLY")


def test_central_rule_marks_safe_positive_delta_as_conversion_signal() -> None:
    payload = _run()
    for record in payload["candidate_outcome_records"]:
        delta = record["delta_terminal_adjusted_progress_vs_hold"]
        reentry = record["candidate_terminal_reentry"]
        status = record["candidate_outcome_status"]
        if not record["objective_completion_signal"]:
            if reentry:
                assert status == TERMINAL_REENTRY_CANDIDATE_ONLY
            elif delta > 0:
                assert status == CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
            else:
                assert status == NO_CONVERSION_SIGNAL_CANDIDATE_ONLY


def test_best_candidate_prefers_non_terminal_higher_delta() -> None:
    payload = _run()
    best = payload["best_candidate"]
    if best:
        assert best["candidate_terminal_reentry"] is False
        for record in payload["candidate_outcome_records"]:
            if not record["candidate_terminal_reentry"]:
                assert (
                    best["delta_terminal_adjusted_progress_vs_hold"]
                    >= record["delta_terminal_adjusted_progress_vs_hold"]
                )


def test_status_helpers_pure_logic() -> None:
    assert (
        candidate_status(beats_hold=True, terminal_reentry=False, objective_completion=False)
        == CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY
    )
    assert (
        candidate_status(beats_hold=False, terminal_reentry=True, objective_completion=False)
        == TERMINAL_REENTRY_CANDIDATE_ONLY
    )
    assert (
        candidate_status(beats_hold=False, terminal_reentry=False, objective_completion=False)
        == NO_CONVERSION_SIGNAL_CANDIDATE_ONLY
    )
    assert (
        candidate_status(beats_hold=True, terminal_reentry=False, objective_completion=True)
        == OBJECTIVE_COMPLETION_OBSERVED_CANDIDATE_ONLY
    )


def test_roll_up_mixed_when_conversion_and_terminal_present() -> None:
    records = [
        {"candidate_outcome_status": CONVERSION_SIGNAL_OBSERVED_CANDIDATE_ONLY},
        {"candidate_outcome_status": TERMINAL_REENTRY_CANDIDATE_ONLY},
    ]
    assert roll_up_status(records) == MIXED_CONVERSION_SIGNAL_CANDIDATE_ONLY


def test_no_verdict_word_in_artifact() -> None:
    payload = _run()
    assert "verdict" not in json.dumps(payload).lower().replace(
        "counted_as_scientific_verdict", ""
    ).replace("only_verdict_location", "")


def test_guardrails_locked() -> None:
    payload = _run()
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["revision_performed"] is False
    assert payload["experiment_result_counted_as_scientific_verdict"] is False
    assert payload["consolidation_status_counted_as_scientific_verdict"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    assert payload["a32_remains_only_verdict_location"] is True


def test_rejects_source_with_support() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    payload["support"] = 1
    from theory.m3.objective_conversion_experiment_consolidation import (
        _validate_source_payload,
    )

    with pytest.raises(ValueError):
        _validate_source_payload(payload)
