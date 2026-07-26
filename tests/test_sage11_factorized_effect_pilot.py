"""Tests for the frozen SAGE.11 factorized effect pilot v2."""

from __future__ import annotations

from theory.sage11.factorized_effect_pilot_runner import (
    CORE_HEADS,
    _factor_labels,
    evaluate_factorized_gate,
    make_factor_classifier,
)
from theory.sage11.splits import SOURCE_VALIDATION


def test_factor_labels_preserve_component_partial_credit():
    labels = _factor_labels((
        "effect:changed_cells(few)",
        "effect:player_moved(True)",
        "effect:value_multiset_delta(few,few)",
        "progress:level_complete(False)",
        "risk:game_over(True)",
    ))
    assert labels == {
        "changed_cells": 2,
        "player_moved": 1,
        "level_complete": 0,
        "game_over": 1,
    }


def test_factor_classifier_configuration_is_frozen_and_balanced():
    classifier = make_factor_classifier()
    assert classifier.learning_rate == 0.08
    assert classifier.max_depth == 4
    assert classifier.max_iter == 100
    assert classifier.early_stopping is False
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 11


def test_factorized_gate_requires_gain_heads_and_every_validation_game():
    heads = {
        head: {"full_minus_action_only": 0.10}
        for head in CORE_HEADS
    }
    games = {
        game: {"composite": {"full_minus_action_only": 0.0}}
        for game in SOURCE_VALIDATION
    }
    passing = evaluate_factorized_gate(
        heads,
        {"full_minus_action_only": 0.10},
        games,
    )
    assert passing["go"]

    games["ls20"]["composite"]["full_minus_action_only"] = -0.001
    failing = evaluate_factorized_gate(
        heads,
        {"full_minus_action_only": 0.10},
        games,
    )
    assert not failing["go"]
