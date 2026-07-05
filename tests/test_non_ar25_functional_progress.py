"""A26 non-ar25 functional progress tests."""

from __future__ import annotations

from pathlib import Path

from theory.non_ar25_functional_progress import run_non_ar25_functional_progress


def test_non_ar25_functional_progress_after_relation_agenda():
    result = run_non_ar25_functional_progress(
        trace_path=Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl"),
        max_candidates=20,
        min_pixel_support=1,
    )

    assert result.error == ""
    assert result.non_ar25_multi_relation_agenda
    assert result.active_transition_non_ar25
    assert result.active_test_called_only_when_local_agenda_observed
    assert result.relation_observation_active
    assert result.functional_progress_non_ar25
    assert result.transition_count == 1
    assert result.env_actions == 1
    assert result.wrong_confirmations == 0
    assert result.trace_support_counted_as_proof is False

    assert result.progress is not None
    assert result.progress.source_color == 9
    assert result.progress.target_color == 8
    assert result.progress.source_to_target_pixels > 0
    assert result.progress.source_reduction > 0
    assert result.progress.target_gain > 0
    assert result.progress.source_target_gap_reduced
    assert result.progress.useful_new_state
    assert "source_target_gap_reduced" in result.progress.progress_signals

    assert result.agenda_result is not None
    assert result.agenda_result.hypothesis_remains_unresolved_until_observation
    assert result.agenda_result.validation_test_called_only_when_local_agenda_observed
    assert result.agenda_result.trace_support_counted_as_proof is False
