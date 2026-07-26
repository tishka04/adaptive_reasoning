"""Tests for the frozen source-train-only anti-shortcut audit."""

from __future__ import annotations

import numpy as np

from theory.sage11.anti_shortcut_audit import (
    _conditional_action_shuffle,
    _signature_identity_metrics,
    evaluate_anti_shortcut_gate,
    make_audit_classifier,
)
from theory.sage11.splits import SOURCE_TRAIN


def _passing_gate_inputs():
    heads = {
        "changed_cells": {
            "full_minus_best_baseline": 0.10,
            "signature_ablation_drop": 0.01,
        },
        "player_moved": {
            "full_minus_best_baseline": 0.20,
            "signature_ablation_drop": 0.01,
        },
    }
    composite = {
        "conditional_action_shuffle_degradation": 0.10,
        "signature_ablation_drop": 0.01,
    }
    folds = {game: 0.0 for game in SOURCE_TRAIN}
    signature = {"row_weighted_majority_game_accuracy": 0.95}
    return heads, composite, folds, signature


def test_anti_shortcut_gate_requires_every_frozen_condition():
    heads, composite, folds, signature = _passing_gate_inputs()
    assert evaluate_anti_shortcut_gate(
        heads,
        composite,
        folds,
        signature,
    )["passed"]

    heads["changed_cells"]["full_minus_best_baseline"] = 0.099
    assert not evaluate_anti_shortcut_gate(
        heads,
        composite,
        folds,
        signature,
    )["passed"]


def test_signature_purity_fails_only_when_effect_model_relies_on_it():
    heads, composite, folds, signature = _passing_gate_inputs()
    heads["changed_cells"]["signature_ablation_drop"] = 0.021
    report = evaluate_anti_shortcut_gate(
        heads,
        composite,
        folds,
        signature,
    )
    assert not report["no_fixed_signature_shortcut_reliance"]


def test_conditional_shuffle_preserves_signature_and_state_columns():
    features = np.asarray([
        [1, 1, 10],
        [1, 2, 20],
        [1, 3, 30],
        [0, 4, 40],
    ], dtype=np.float32)
    shuffled = _conditional_action_shuffle(
        features,
        signature_columns=(0,),
        action_dependent_columns=(1,),
        random_state=11,
    )
    assert np.array_equal(shuffled[:, 0], features[:, 0])
    assert np.array_equal(shuffled[:, 2], features[:, 2])
    assert sorted(shuffled[:3, 1]) == [1, 2, 3]
    assert shuffled[3, 1] == 4


def test_fixed_signatures_report_game_identification_purity():
    features = np.asarray([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ], dtype=np.float32)
    metrics = _signature_identity_metrics(
        features,
        np.asarray(["a", "a", "b", "b"]),
        (0, 1),
    )
    assert metrics["row_weighted_majority_game_accuracy"] == 1.0
    assert metrics["game_exclusive_row_rate"] == 1.0


def test_audit_classifier_is_frozen_and_balanced():
    classifier = make_audit_classifier()
    assert classifier.learning_rate == 0.08
    assert classifier.max_depth == 4
    assert classifier.max_iter == 100
    assert classifier.early_stopping is False
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 11
