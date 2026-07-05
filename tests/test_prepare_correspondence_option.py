"""A11 PrepareCorrespondenceOption tests."""

from __future__ import annotations

from theory.ar25_live_option_micro_run import _bootstrap_ar25_option_theory
from theory.live_transition_loop import LiveTransitionBeliefLoop
from theory.prepare_correspondence_option import PrepareCorrespondenceOption
from theory.theory_option import build_options_from_theory


def test_prepare_option_initiates_only_before_strong_ready():
    theory = _bootstrap_ar25_option_theory("ar25-e3c63847")
    loop = LiveTransitionBeliefLoop(
        "ar25-e3c63847",
        theory=theory,
        available_actions=[f"ACTION{idx}" for idx in range(1, 8)],
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    validation_option = build_options_from_theory(loop.theory)[0]
    prepare_option = PrepareCorrespondenceOption.from_validation_option(
        validation_option,
        max_steps=14,
    )

    assert prepare_option.name == "prepare_correspondence_colors10_11"
    assert prepare_option.target_rule == validation_option.target_rule
    assert prepare_option.can_initiate(
        loop,
        {
            "active_color_pair_10_11",
            "selected_pair_exists",
            "selected_source_matches_target_color",
        },
        ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    )
    assert not prepare_option.can_initiate(
        loop,
        {
            "active_color_pair_10_11",
            "selected_pair_exists",
            "selected_source_matches_target_shape",
            "source_target_relation_satisfied",
            "strong_ready_to_validate_correspondence",
        },
        ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    )
