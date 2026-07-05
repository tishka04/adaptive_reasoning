"""A15 generic discriminating experiment designer tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.epistemic_metrics import HypothesisStatus
from theory.generic_discriminating_experiment_designer import (
    DiscriminatingPrediction,
    GenericDiscriminatingExperimentDesigner,
)


@dataclass(frozen=True)
class _FakeAction:
    name: str
    raw_action: str
    action_args: dict[str, int]


def test_designer_selects_coordinate_with_max_observable_prediction_divergence():
    grid = np.zeros((6, 6), dtype=np.int32)
    grid[1, 1] = 4
    grid[2, 2] = 9
    actions = [
        _FakeAction("ACTION6", "low_divergence", {"x": 1, "y": 1}),
        _FakeAction("ACTION6", "high_divergence", {"x": 2, "y": 2}),
    ]
    hypotheses = [
        DiscriminatingPrediction("h::4::5", "ACTION6", 4, 5, support=3),
        DiscriminatingPrediction("h::4::6", "ACTION6", 4, 6, support=2),
        DiscriminatingPrediction("h::9::8", "ACTION6", 9, 8, support=10),
        DiscriminatingPrediction("h::9::12", "ACTION6", 9, 12, support=5),
        DiscriminatingPrediction("h::9::14", "ACTION6", 9, 14, support=1),
    ]

    choice = GenericDiscriminatingExperimentDesigner().design(
        hypotheses=hypotheses,
        live_grid=grid,
        available_actions=actions,
    )

    assert choice is not None
    assert choice.action.raw_action == "high_divergence"
    assert choice.action.action_args == {"x": 2, "y": 2}
    assert choice.observed_source_color == 9
    assert choice.expected_divergence == 3.0
    assert choice.candidate_pool_size == 3
    assert choice.competing_keys == ("h::9::8", "h::9::12")
    assert choice.predicted_pairs == ((9, 8), (9, 12))
    assert choice.prediction_families == ("color_transform", "color_transform")
    assert choice.predicted_outcomes == ("9->8", "9->12")
    assert choice.has_divergent_predictions
    assert choice.trace_support_counted_as_proof is False


def test_designer_supports_non_color_effect_scope_divergence():
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[1, 1] = 9
    actions = [_FakeAction("ACTION6", "click_source", {"x": 1, "y": 1})]
    hypotheses = [
        DiscriminatingPrediction(
            "effect_scope::ACTION6::source9::local",
            "ACTION6",
            9,
            family="effect_scope",
            predicate="effect_scope",
            predicted_outcome="local",
        ),
        DiscriminatingPrediction(
            "effect_scope::ACTION6::source9::global",
            "ACTION6",
            9,
            family="effect_scope",
            predicate="effect_scope",
            predicted_outcome="global",
        ),
    ]

    choice = GenericDiscriminatingExperimentDesigner().design(
        hypotheses=hypotheses,
        live_grid=grid,
        available_actions=actions,
        preferred_family="effect_scope",
    )

    assert choice is not None
    assert choice.action.raw_action == "click_source"
    assert choice.predicted_pairs == ()
    assert choice.prediction_families == ("effect_scope", "effect_scope")
    assert choice.predicted_outcomes == ("local", "global")
    assert choice.divergence_reason == "same_source_different_effect_scope"
    assert choice.has_divergent_predictions


def test_designer_supports_relation_predicate_divergence():
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[1, 1] = 9
    actions = [_FakeAction("ACTION6", "click_source", {"x": 1, "y": 1})]
    hypotheses = [
        DiscriminatingPrediction(
            "relation::ACTION6::same_shape::colors9_8::preserved",
            "ACTION6",
            9,
            8,
            family="relation",
            predicate="same_shape",
            predicted_outcome="preserved",
        ),
        DiscriminatingPrediction(
            "relation::ACTION6::same_shape::colors9_8::broken",
            "ACTION6",
            9,
            8,
            family="relation",
            predicate="same_shape",
            predicted_outcome="broken",
        ),
    ]

    choice = GenericDiscriminatingExperimentDesigner().design(
        hypotheses=hypotheses,
        live_grid=grid,
        available_actions=actions,
        preferred_family="relation",
    )

    assert choice is not None
    assert choice.action.raw_action == "click_source"
    assert choice.prediction_families == ("relation", "relation")
    assert choice.predicted_outcomes == ("preserved", "broken")
    assert choice.predicted_pairs == ((9, 8), (9, 8))
    assert choice.divergence_reason == "same_source_different_same_shape"
    assert choice.has_divergent_predictions


def test_designer_requires_unresolved_divergent_predictions():
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[1, 1] = 9
    actions = [_FakeAction("ACTION6", "only_action", {"x": 1, "y": 1})]
    hypotheses = [
        DiscriminatingPrediction("h::9::8", "ACTION6", 9, 8),
        DiscriminatingPrediction(
            "h::9::12",
            "ACTION6",
            9,
            12,
            status=HypothesisStatus.CONFIRMED,
        ),
    ]

    choice = GenericDiscriminatingExperimentDesigner().design(
        hypotheses=hypotheses,
        live_grid=grid,
        available_actions=actions,
    )

    assert choice is None
