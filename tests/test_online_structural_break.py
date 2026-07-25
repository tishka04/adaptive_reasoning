from __future__ import annotations

from theory.online_structural_break import OnlineStructuralBreakDetector
from theory.online_terminal_relational_stencil import (
    StencilTheoryAssessment,
)


BASE_RULES = {"void": True, "filled": False}


def _assessment(
    signature: str,
    violations: int,
) -> StencilTheoryAssessment:
    return StencilTheoryAssessment(
        structural_signature=signature,
        applicable=True,
        total_constraints=8,
        total_violations=violations,
        improving_actions=max(0, violations),
        stencil_count=1,
        click_count=8,
    )


def test_requires_repeated_residuals_and_preserves_intermittent_success():
    detector = OnlineStructuralBreakDetector(
        minimum_consecutive_residuals=3,
    )
    before = _assessment("novel", 4)
    failed_after = _assessment("novel", 4)
    successful_after = _assessment("novel", 3)

    first = detector.observe_base_prediction(
        before=before,
        after=failed_after,
        predicted_reduction=1,
        action_no_effect=False,
        terminal_success=False,
        base_rules=BASE_RULES,
    )
    assert first.prediction_failed is True
    assert first.break_detected is False

    success = detector.observe_base_prediction(
        before=before,
        after=successful_after,
        predicted_reduction=1,
        action_no_effect=False,
        terminal_success=False,
        base_rules=BASE_RULES,
    )
    assert success.prediction_failed is False
    assert success.consecutive_residuals == 0
    assert detector.summary()["breaks_detected"] == 0


def test_novel_regime_break_suspends_only_that_context_and_generates_rules():
    detector = OnlineStructuralBreakDetector(
        minimum_consecutive_residuals=3,
    )
    before = _assessment("broken-context", 4)
    after = _assessment("broken-context", 4)

    outcome = None
    for _ in range(3):
        outcome = detector.observe_base_prediction(
            before=before,
            after=after,
            predicted_reduction=1,
            action_no_effect=False,
            terminal_success=False,
            base_rules=BASE_RULES,
        )
    assert outcome is not None
    assert outcome.break_detected is True
    assert detector.is_suspended("broken-context") is True
    assert detector.is_suspended("other-context") is False
    summary = detector.summary()
    assert summary["hypotheses_generated"] == 3
    assert summary["old_theory_suspensions"] == 1


def test_unconditioned_memory_detects_break_but_keeps_old_theory_active():
    detector = OnlineStructuralBreakDetector(
        condition_memory_by_regime=False,
        minimum_consecutive_residuals=3,
    )
    before = _assessment("blind-memory", 3)
    for _ in range(3):
        detector.observe_base_prediction(
            before=before,
            after=before,
            predicted_reduction=1,
            action_no_effect=False,
            terminal_success=False,
            base_rules=BASE_RULES,
        )
    assert detector.summary()["breaks_detected"] == 1
    assert detector.is_suspended("blind-memory") is False
    assert detector.revision_hypothesis("blind-memory") is None


def test_revision_needs_two_terminal_confirmations_before_promotion():
    detector = OnlineStructuralBreakDetector(
        minimum_consecutive_residuals=3,
        minimum_terminal_confirmations=2,
    )
    before = _assessment("revision-context", 3)
    for _ in range(3):
        detector.observe_base_prediction(
            before=before,
            after=before,
            predicted_reduction=1,
            action_no_effect=False,
            terminal_success=False,
            base_rules=BASE_RULES,
        )
    hypothesis = detector.revision_hypothesis("revision-context")
    assert hypothesis is not None
    detector.note_revision_action(hypothesis.hypothesis_id)

    nominated = detector.observe_revision_outcome(
        hypothesis_id=hypothesis.hypothesis_id,
        after=before,
        terminal_success=True,
        game_over=False,
    )
    assert nominated == "nominated"
    assert detector.confirmed_revision_rules("revision-context") == {}

    confirmed = detector.observe_revision_outcome(
        hypothesis_id=hypothesis.hypothesis_id,
        after=before,
        terminal_success=True,
        game_over=False,
    )
    assert confirmed == "confirmed"
    assert detector.confirmed_revision_rules("revision-context")
    assert detector.is_suspended("revision-context") is False


def test_compatible_regime_requires_effect_failure_as_second_signal():
    detector = OnlineStructuralBreakDetector(
        minimum_consecutive_residuals=3,
    )
    before = _assessment("known-context", 3)
    detector.observe_base_prediction(
        before=before,
        after=_assessment("next", 0),
        predicted_reduction=1,
        action_no_effect=False,
        terminal_success=True,
        base_rules=BASE_RULES,
    )
    for _ in range(3):
        outcome = detector.observe_base_prediction(
            before=before,
            after=before,
            predicted_reduction=1,
            action_no_effect=False,
            terminal_success=False,
            base_rules=BASE_RULES,
        )
    assert outcome.break_detected is False

    for _ in range(3):
        outcome = detector.observe_base_prediction(
            before=before,
            after=before,
            predicted_reduction=1,
            action_no_effect=True,
            terminal_success=False,
            base_rules=BASE_RULES,
        )
    assert outcome.break_detected is True


def test_repeated_local_success_makes_novel_regime_resistant_to_noise():
    detector = OnlineStructuralBreakDetector(
        minimum_consecutive_residuals=3,
    )
    before = _assessment("locally-validated-context", 3)
    reduced = _assessment("locally-validated-context", 2)

    for _ in range(2):
        detector.observe_base_prediction(
            before=before,
            after=reduced,
            predicted_reduction=1,
            action_no_effect=False,
            terminal_success=False,
            base_rules=BASE_RULES,
        )
    for _ in range(3):
        outcome = detector.observe_base_prediction(
            before=before,
            after=before,
            predicted_reduction=1,
            action_no_effect=False,
            terminal_success=False,
            base_rules=BASE_RULES,
        )

    assert outcome.break_detected is False
    assert detector.is_suspended("locally-validated-context") is False
    assert (
        detector.summary()["regime_memory"][0]["status"]
        == "compatible"
    )


def test_repeated_nonterminal_theory_completion_detects_goal_break():
    detector = OnlineStructuralBreakDetector(
        minimum_consecutive_residuals=3,
    )
    before = _assessment("changed-terminal-condition", 1)
    completed = _assessment("changed-terminal-condition", 0)

    for _ in range(3):
        outcome = detector.observe_base_prediction(
            before=before,
            after=completed,
            predicted_reduction=1,
            action_no_effect=False,
            terminal_success=False,
            base_rules=BASE_RULES,
        )

    assert outcome.prediction_failed is False
    assert outcome.terminal_condition_failed is True
    assert outcome.break_detected is True
    assert detector.is_suspended("changed-terminal-condition") is True
    assert detector.summary()["terminal_condition_residuals"] == 3
