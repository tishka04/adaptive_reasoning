from __future__ import annotations

import numpy as np

from theory.sage12.mt.transition import compile_mt_transition
from theory.sage12.topological_causal_control_v4_19 import (
    _compose,
    _fit_model,
    _predict,
)
from theory.sage12.topological_invariants_v4_19 import (
    FACTOR_NAMES,
    compile_topological_transition,
    feature_vector,
    forbidden_field_hits,
    permutation_invariant,
    topological_invariants,
)


def _transition():
    before = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    after = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 2, 0],
            [0, 0, 1, 2, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    return compile_mt_transition(
        before,
        "ACTION4",
        after,
        player_position_before=(1, 1),
        player_position_after=(1, 2),
        productive=True,
        risk=False,
    )


def test_topological_features_are_exactly_permutation_invariant() -> None:
    graph = _transition().graph_before
    assert permutation_invariant(graph)
    reversed_graph = graph.permuted(tuple(reversed(range(len(graph.nodes)))))
    assert np.array_equal(feature_vector(graph), feature_vector(reversed_graph))


def test_topological_compiler_factorizes_observed_delta() -> None:
    compiled = compile_topological_transition(
        _transition(),
        terminal_progress=False,
        risk=False,
    )
    assert set(compiled.factors) == set(FACTOR_NAMES)
    assert compiled.factors["contact_added"]
    assert compiled.factors["bridge_removed"]
    assert not compiled.factors["risk"]
    assert -1.0 <= compiled.local_value <= 1.0
    assert compiled.correspondence["confident_fraction"] >= 0.0


def test_relational_controls_change_only_authorized_view() -> None:
    graph = _transition().graph_before
    full = feature_vector(graph)
    removed = feature_vector(graph, remove_relations=True)
    swapped = feature_vector(graph, swap_binding=True)
    static = feature_vector(graph, static_only=True)
    assert full.shape == removed.shape == swapped.shape == static.shape
    assert not np.array_equal(full, removed)
    assert np.isfinite(full).all()


def test_student_payload_has_no_identity_or_coordinate_fields() -> None:
    graph = _transition().graph_before
    payload = {
        "features": feature_vector(graph).tolist(),
        "invariants": topological_invariants(graph),
    }
    assert forbidden_field_hits(payload) == []


def test_compact_predictor_has_factor_value_and_uncertainty_heads() -> None:
    generator = np.random.default_rng(19)
    arrays = {
        "full": generator.normal(size=(12, 512)).astype(np.float32),
        "factors": generator.integers(0, 2, size=(12, len(FACTOR_NAMES))).astype(
            np.float32
        ),
        "values": generator.uniform(-1, 1, size=(12, 4)).astype(np.float32),
    }
    model, history = _fit_model(
        arrays,
        np.arange(12, dtype=np.int64),
        device="cpu",
        epochs=1,
        seed=19,
    )
    factors, values, uncertainty, latent = _predict(
        model,
        arrays["full"],
        device="cpu",
    )
    assert factors.shape == (12, len(FACTOR_NAMES))
    assert values.shape == (12, 4)
    assert uncertainty.shape == (12,)
    assert latent.shape == (12, 64)
    assert history[-1]["loss"] >= 0.0


def test_deployed_score_penalizes_uncertainty() -> None:
    policy = np.asarray([0.0, 1.0, 2.0])
    value = np.asarray([0.0, 1.0, 2.0])
    energy = np.asarray([0.0, 0.0, 0.0])
    confident = _compose(policy, value, energy, np.asarray([0.0, 0.0, 1.0]))
    unpenalized = _compose(policy, value, energy)
    assert confident[-1] < unpenalized[-1]
