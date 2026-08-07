from __future__ import annotations

from theory.sage_t.contracts import ActionCandidate
from theory.sage_t.live_shadow_pilot_v4 import (
    BaselineInclusiveController,
    BaselineInclusiveDecisionEngine,
    find_baseline_candidate,
    load_frozen_manifest,
)


def test_manifest_registers_only_baseline_sequence_inclusion() -> None:
    manifest = load_frozen_manifest()

    assert manifest["inference_changes"] == [
        "reserve one counterfactual sequence for the exact baseline action"
    ]
    assert manifest["authority"]["mode"] == "shadow"
    assert manifest["challenger"]["can_satisfy_t8_0_gate"] is False


def test_parameterized_baseline_action_is_resolved_exactly() -> None:
    legal = tuple(
        ActionCandidate("ACTION6", {"x": x, "y": 7}) for x in range(12)
    )

    selected = find_baseline_candidate(
        symbolic_action_name="ACTION6",
        symbolic_action_data={"game_id": "ignored-local-metadata", "x": 11, "y": 7},
        legal_actions=legal,
    )

    assert selected == legal[11]


def test_baseline_action_survives_a_saturated_sequence_budget() -> None:
    legal = tuple(
        ActionCandidate("ACTION6", {"x": x, "y": 3}) for x in range(12)
    )
    engine = BaselineInclusiveDecisionEngine(
        maximum_sequences=8,
        maximum_particles=4,
        ordinary_horizon=1,
        preferred_action=legal[11],
    )

    sequences = engine.generate_sequences(
        legal,
        memory_macros=((legal[10],),),
    )

    assert len(sequences) == 8
    assert sequences[0].actions == (legal[11],)
    assert any(sequence.actions == (legal[11],) for sequence in sequences)


def test_instrumented_controller_uses_baseline_inclusive_engine() -> None:
    controller = BaselineInclusiveController()

    assert isinstance(
        controller.decision_engine,
        BaselineInclusiveDecisionEngine,
    )
