"""A13 active non-ar25 correspondence micro-run tests."""

from __future__ import annotations

from pathlib import Path

from theory.epistemic_metrics import HypothesisStatus
from theory.non_ar25_active_micro_run import run_non_ar25_active_micro_run


def test_ft09_active_micro_run_revises_unresolved_hypothesis_from_live_transition():
    result = run_non_ar25_active_micro_run(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
    )

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.active_transition_non_ar25
    assert result.transition_count == 1
    assert result.env_actions == 1
    assert result.candidate_key == (
        "correspondence::ACTION6::modifies::colors9_8"
    )

    assert result.status_before == HypothesisStatus.UNRESOLVED
    assert result.status_after == HypothesisStatus.CONFIRMED
    assert result.hypothesis_status_changed
    assert result.evidence_permits_status_change
    assert result.status_reason == "observed_source_target_color_transform"

    assert result.pair_change_pixels >= 4
    assert "source_target_color_transform" in result.observed_predicates
    assert result.wrong_confirmations == 0
    assert result.score is not None
    assert result.score.wrong_confirmations == 0

    assert result.selected_action is not None
    assert result.selected_action.name == "ACTION6"
    assert result.selected_action.action_args["x"] is not None
    assert result.selected_action.action_args["y"] is not None

    assert result.transition_update is not None
    record = result.transition_update.record
    assert record.action.name == "ACTION6"
    assert record.action.x is not None
    assert record.action.y is not None
    assert record.diff.num_changed > 0

    ledger_record = result.record()
    assert ledger_record is not None
    assert ledger_record.status == HypothesisStatus.CONFIRMED
    assert ledger_record.experiments_spent == 1
