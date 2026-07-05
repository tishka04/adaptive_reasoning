"""A24 multi-relation option composition tests."""

from __future__ import annotations

from theory.multi_relation_option_composition import (
    default_ar25_relation_preconditions,
    missing_relation_preconditions,
    run_multi_relation_option_composition,
)


def test_missing_relation_preconditions_choose_order_without_score():
    preconditions = default_ar25_relation_preconditions()

    missing = missing_relation_preconditions(
        preconditions,
        {
            "active_color_pair_10_11",
            "selected_source_matches_target_shape",
        },
    )

    assert [item.option.name for item in missing] == [
        "prepare_relation_aligned_with",
        "avoid_relation_failed_validation_context_absent",
    ]
    assert [item.option.mode for item in missing] == ["prepare", "avoid"]


def test_multi_relation_option_composition_gates_validation_on_all_relations():
    result = run_multi_relation_option_composition(
        max_actions=50,
        max_option_attempts=1,
    )

    assert result.error == ""
    assert result.trace_dependent is False
    assert result.multiple_relations_candidate
    assert result.includes_prepare_and_avoid_options
    assert result.selection_reason == "missing_preconditions"
    assert result.order_chosen_by_missing_preconditions
    assert result.selected_option_order == [
        "prepare_relation_same_shape",
        "prepare_relation_aligned_with",
        "avoid_relation_failed_validation_context_absent",
    ]
    assert result.missing_preconditions == [
        "selected_source_matches_target_shape",
        "source_target_aligned",
        "no_recent_failed_validation_same_context",
    ]

    assert result.validation_called
    assert result.all_required_relation_preconditions_observed
    assert result.validation_called_only_when_all_required_relations_observed
    assert result.full_composed_chain_attempted
    assert result.relation_options_success
    assert result.wrong_confirmations == 0
    assert result.confirmation_precision >= 0.95

    assert result.relation_event is not None
    assert result.relation_event.kind == "prepare_option"
    assert result.relation_event.termination == "success"

    assert result.validation_event is not None
    assert result.validation_event.kind == "option_attempt"
    assert result.validation_event.action == "ACTION2"
    assert result.validation_event.termination == "success"
    for predicate in result.missing_preconditions:
        assert predicate in result.validation_event.predicates_present
