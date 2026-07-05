"""A23 option composition tests."""

from __future__ import annotations

from theory.option_composition import run_option_composition


def test_relation_option_composes_into_correspondence_validation():
    result = run_option_composition(max_actions=50, max_option_attempts=1)

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.relation_option_success
    assert result.relation_observed
    assert result.correspondence_option_called
    assert result.correspondence_option_called_only_if_relation_observed
    assert result.full_composed_chain_attempted
    assert result.wrong_confirmations == 0
    assert result.confirmation_precision >= 0.95

    assert result.relation_event is not None
    assert result.relation_event.kind == "prepare_option"
    assert result.relation_event.termination == "success"
    assert "source_target_relation_satisfied" in result.relation_event.predicates_after

    assert result.validation_event is not None
    assert result.validation_event.kind == "option_attempt"
    assert result.validation_event.action == "ACTION2"
    assert result.validation_event.termination == "success"
    assert "source_target_relation_satisfied" in result.validation_event.predicates_present
    assert (
        "strong_ready_to_validate_correspondence"
        in result.validation_event.predicates_present
    )
