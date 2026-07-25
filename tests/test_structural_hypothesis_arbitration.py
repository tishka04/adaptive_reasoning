"""SAGE.9x generic structural-hypothesis arbitration tests."""

from theory.structural_hypothesis_arbitration import (
    StructuralExperimentOption,
    select_discriminating_structural_experiment,
    sequential_structural_experiment,
    surviving_hypotheses,
)


def _options():
    return (
        StructuralExperimentOption(
            action_key="historical-test",
            predicted_reductions=(
                ("h0", 1),
                ("h1", -1),
                ("h2", -1),
            ),
        ),
        StructuralExperimentOption(
            action_key="discriminating-test",
            predicted_reductions=(
                ("h0", 0),
                ("h1", 2),
                ("h2", -1),
            ),
        ),
        StructuralExperimentOption(
            action_key="weak-test",
            predicted_reductions=(
                ("h0", 0),
                ("h1", 0),
                ("h2", 1),
            ),
        ),
    )


def test_active_arbiter_maximizes_disagreement_not_generation_order():
    active = select_discriminating_structural_experiment(
        options=_options(),
        hypothesis_priority=("h0", "h1", "h2"),
    )
    sequential = sequential_structural_experiment(
        options=_options(),
        hypothesis_priority=("h0", "h1", "h2"),
    )

    assert active is not None
    assert sequential is not None
    assert active.action_key == "discriminating-test"
    assert active.hypothesis_id == "h1"
    assert sequential.action_key == "historical-test"
    assert surviving_hypotheses(
        choice=active,
        observed_reduction=2,
    ) == ("h1",)


def test_active_choice_is_stable_when_hypothesis_priority_is_permuted():
    forward = select_discriminating_structural_experiment(
        options=_options(),
        hypothesis_priority=("h0", "h1", "h2"),
    )
    reverse = select_discriminating_structural_experiment(
        options=_options(),
        hypothesis_priority=("h2", "h1", "h0"),
    )

    assert forward is not None
    assert reverse is not None
    assert forward.action_key == reverse.action_key
