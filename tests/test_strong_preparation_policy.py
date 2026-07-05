"""A10 discriminating readiness tests."""

from __future__ import annotations

import logging
import os

from theory.contextual_readiness import ContextualReadinessDiscriminator
from theory.precondition_hypothesis import extract_precondition_predicates
from theory.real_env_option_adapter import snapshot_frame
from theory.strong_preparation_policy import (
    StrongPrepareCorrespondencePolicy,
    discriminating_readiness_gaps,
)
from theory.theory_option import TheoryOption


TARGET_RULE = "correspondence::ACTION2::validates::colors10_11"


def _option() -> TheoryOption:
    return TheoryOption(
        name="validate_correspondence_colors10_11",
        target_rule=TARGET_RULE,
        initiation_predicate="ready_to_validate_correspondence",
        precondition_key=(
            "precondition::correspondence::ACTION2::validates::"
            "colors10_11::ready_to_validate_correspondence"
        ),
        policy_action="ACTION2",
        pair_colors=(10, 11),
    )


def test_strong_policy_selects_actions_from_missing_predicates():
    policy = StrongPrepareCorrespondencePolicy()
    option = _option()

    align_plan = policy.choose(
        option=option,
        predicates_present={
            "active_color_pair_10_11",
            "selected_pair_exists",
            "selected_source_matches_target_color",
        },
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    )

    assert align_plan is not None
    assert align_plan.action == "ACTION3"
    assert align_plan.role == "move"
    assert align_plan.target_predicate == "selected_source_matches_target_shape"

    relation_plan = policy.choose(
        option=option,
        predicates_present={
            "active_color_pair_10_11",
            "selected_pair_exists",
            "selected_source_matches_target_color",
            "selected_source_matches_target_shape",
            "source_target_projected_aligned",
        },
        available_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    )

    assert relation_plan is not None
    assert relation_plan.action == "ACTION2"
    assert relation_plan.target_predicate == "source_target_relation_satisfied"

    assert policy.choose(
        option=option,
        predicates_present={
            "active_color_pair_10_11",
            "selected_source_matches_target_shape",
            "source_target_relation_satisfied",
        },
        available_actions=["ACTION1", "ACTION2"],
    ) is None


def test_projection_predicates_split_weak_and_strong_ar25():
    logging.disable(logging.CRITICAL)
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = "environment_files"
    os.environ.setdefault("ARC_API_KEY", "test")

    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir="environment_files",
    )
    discriminator = ContextualReadinessDiscriminator()

    weak_env = arc.make("ar25-e3c63847")
    weak_frame = weak_env.step(GameAction.RESET)
    for action in ("ACTION4", "ACTION4", "ACTION5"):
        weak_frame = weak_env.step(getattr(GameAction, action))
    weak_predicates = extract_precondition_predicates(
        snapshot_frame(weak_frame).grid,
        target_rule=TARGET_RULE,
        pair_colors=(10, 11),
        previous_action="ACTION5",
        recent_actions=("ACTION4", "ACTION4", "ACTION5"),
        recent_correspondence_successes=(True,),
    )
    weak_assessment = discriminator.assess(
        weak_predicates,
        target_rule=TARGET_RULE,
    )

    assert "ready_to_validate_correspondence" in weak_predicates
    assert "source_target_relation_satisfied" not in weak_predicates
    assert weak_assessment.weak_ready
    assert not weak_assessment.strong_ready

    strong_env = arc.make("ar25-e3c63847")
    strong_frame = strong_env.step(GameAction.RESET)
    strong_actions = ("ACTION3",) * 5 + ("ACTION2",) * 9
    for action in strong_actions:
        strong_frame = strong_env.step(getattr(GameAction, action))
    strong_predicates = extract_precondition_predicates(
        snapshot_frame(strong_frame).grid,
        target_rule=TARGET_RULE,
        pair_colors=(10, 11),
        previous_action="ACTION2",
        recent_actions=("ACTION2", "ACTION2", "ACTION2", "ACTION2"),
    )
    strong_assessment = discriminator.assess(
        strong_predicates,
        target_rule=TARGET_RULE,
    )

    assert discriminating_readiness_gaps(strong_predicates) == []
    assert "selected_source_matches_target_shape" in strong_predicates
    assert "source_target_relation_satisfied" in strong_predicates
    assert strong_assessment.strong_ready
