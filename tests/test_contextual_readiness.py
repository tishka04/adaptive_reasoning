"""A9 contextual readiness discrimination tests."""

from __future__ import annotations

from theory.contextual_readiness import (
    NO_FAILED_SAME_CONTEXT,
    STRONG_READY,
    WEAK_READY,
    ContextualReadinessDiscriminator,
)
from theory.epistemic_metrics import HypothesisStatus
from theory.theory_option import TheoryOptionInvocation


def test_contextual_readiness_refutes_failed_weak_context_only():
    discriminator = ContextualReadinessDiscriminator()
    target_rule = "correspondence::ACTION2::validates::colors10_11"
    weak_predicates = {
        "selected_pair_exists",
        "controller_on_source",
        "recent_control_switch",
        "active_color_pair_10_11",
        "ready_to_validate_correspondence",
    }

    before = discriminator.assess(weak_predicates, target_rule=target_rule)

    assert before.weak_ready
    assert WEAK_READY in before.predicates
    assert NO_FAILED_SAME_CONTEXT in before.predicates
    assert STRONG_READY not in before.predicates
    assert discriminator.can_attempt_validation(before)

    hypothesis = discriminator.observe_validation_result(
        before,
        TheoryOptionInvocation(
            option_name="validate_correspondence_colors10_11",
            target_rule=target_rule,
            precondition_key="precondition::ready",
            policy_action="ACTION2",
            step=3,
            actual_action="ACTION2",
            predicates_present=list(before.predicates),
            termination="contradiction",
        ),
    )
    after = discriminator.assess(weak_predicates, target_rule=target_rule)

    assert hypothesis.status == HypothesisStatus.REFUTED
    assert after.context.key == before.context.key
    assert after.weak_ready
    assert NO_FAILED_SAME_CONTEXT not in after.predicates
    assert not discriminator.can_attempt_validation(after)


def test_contextual_readiness_marks_strong_when_relation_is_satisfied():
    discriminator = ContextualReadinessDiscriminator()
    target_rule = "correspondence::ACTION2::validates::colors10_11"

    assessment = discriminator.assess(
        {
            "selected_pair_exists",
            "controller_on_source",
            "recent_control_switch",
            "source_target_aligned",
            "active_color_pair_10_11",
        },
        target_rule=target_rule,
    )

    assert assessment.weak_ready
    assert assessment.strong_ready
    assert STRONG_READY in assessment.predicates
    assert discriminator.can_attempt_validation(assessment)
