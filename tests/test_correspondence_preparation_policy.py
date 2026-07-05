"""A8 correspondence preparation policy tests."""

from __future__ import annotations

from theory.correspondence_hypothesis import correspondence_key
from theory.correspondence_preparation_policy import PrepareCorrespondencePolicy
from theory.theory_option import TheoryOption


def test_prepare_policy_selects_actions_from_missing_predicates():
    option = TheoryOption(
        name="validate_correspondence_colors10_11",
        target_rule=correspondence_key("ACTION2", "validates", (10, 11)),
        initiation_predicate="ready_to_validate_correspondence",
        precondition_key="precondition::ready",
        policy_action="ACTION2",
        pair_colors=(10, 11),
    )
    policy = PrepareCorrespondencePolicy()

    move = policy.choose(
        option=option,
        predicates_present={"selected_pair_exists"},
        available_actions=["ACTION1", "ACTION2", "ACTION4", "ACTION5"],
    )
    assert move is not None
    assert move.action == "ACTION4"
    assert move.role == "move"
    assert move.target_predicate == "controller_on_source"
    assert move.action != option.policy_action

    switch = policy.choose(
        option=option,
        predicates_present={"selected_pair_exists", "controller_on_source"},
        available_actions=["ACTION2", "ACTION5"],
    )
    assert switch is not None
    assert switch.action == "ACTION5"
    assert switch.role == "control_switch"
    assert switch.target_predicate == "recent_control_switch"

    ready = policy.choose(
        option=option,
        predicates_present={
            "selected_pair_exists",
            "controller_on_source",
            "recent_control_switch",
            "ready_to_validate_correspondence",
        },
        available_actions=["ACTION2", "ACTION5"],
    )
    assert ready is None
