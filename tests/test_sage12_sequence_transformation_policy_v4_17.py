from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.sage12.mt.clustering import TransformationMatch
from theory.sage12.mt.graph import MorphoTopologicalGraph
from theory.sage12.sequence_transformation_policy_v4_17 import (
    TEMPORAL_EBM_COEFFICIENT,
    TRANSFORMATION_COEFFICIENT,
    _argmax,
    _compose_scores,
    _deterministic_permutation,
    _mt_value,
)


@dataclass
class _Memory:
    matches: tuple[TransformationMatch, ...]

    def retrieve(self, *_args, **_kwargs):
        return self.matches


def _graph() -> MorphoTopologicalGraph:
    return MorphoTopologicalGraph(
        nodes=(),
        relations=(),
        invariants={},
        action_name="ACTION1",
        action_family="move",
        signature="synthetic",
    )


def test_composition_uses_frozen_within_panel_coefficients() -> None:
    policy = (0.0, 1.0, 2.0)
    transformation = (2.0, 0.0, 1.0)
    energy = (1.0, 2.0, 0.0)

    actual = _compose_scores(policy, transformation, energy)
    expected = (
        (np.asarray(policy) - np.mean(policy)) / np.std(policy)
        + TRANSFORMATION_COEFFICIENT
        * (np.asarray(transformation) - np.mean(transformation))
        / np.std(transformation)
        - TEMPORAL_EBM_COEFFICIENT
        * (np.asarray(energy) - np.mean(energy))
        / np.std(energy)
    )

    assert np.allclose(actual, expected)


def test_composition_is_stable_when_a_component_is_constant() -> None:
    actual = _compose_scores(
        (1.0, 2.0, 3.0),
        (0.5, 0.5, 0.5),
        (4.0, 4.0, 4.0),
    )

    assert np.allclose(actual, (-1.22474487, 0.0, 1.22474487))


def test_transform_permutation_is_deterministic_and_nonidentity() -> None:
    values = np.asarray((0.1, 0.2, 0.3, 0.4))

    left = _deterministic_permutation(values, key="panel")
    right = _deterministic_permutation(values, key="panel")

    assert np.array_equal(left, right)
    assert not np.array_equal(left, values)
    assert sorted(left.tolist()) == sorted(values.tolist())


def test_unknown_transform_receives_only_negative_uncertainty() -> None:
    value, matches = _mt_value(
        (1.0, 0.0),
        0.35,
        _graph(),
        _Memory(()),
    )

    assert value == -0.35
    assert matches == 0


def test_transform_value_uses_observed_productive_and_risk_evidence() -> None:
    memory = _Memory(
        (
            TransformationMatch(
                prototype_id="productive",
                similarity=0.9,
                productive_probability=0.9,
                risk_probability=0.1,
                support=30,
                game_diversity=4,
                candidate_alias="",
            ),
            TransformationMatch(
                prototype_id="risky",
                similarity=0.5,
                productive_probability=0.2,
                risk_probability=0.8,
                support=25,
                game_diversity=3,
                candidate_alias="",
            ),
        )
    )

    value, matches = _mt_value(
        (1.0, 0.0),
        0.2,
        _graph(),
        memory,
    )

    assert value > 0.7
    assert matches == 2


def test_argmax_breaks_ties_toward_lowest_arm_index() -> None:
    assert _argmax((1.0, 1.0, 0.0), (8, 3, 1)) == 1
