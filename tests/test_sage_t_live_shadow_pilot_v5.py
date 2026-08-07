from __future__ import annotations

from pathlib import Path
from typing import Any

from theory.sage_t import live_shadow_pilot as base
from theory.sage_t.contracts import ActionCandidate
from theory.sage_t.live_shadow_pilot_v5 import (
    MaterializedActionController,
    load_frozen_manifest,
    materialized_baseline_candidate,
)


def test_manifest_freezes_materialized_action_injection_in_shadow() -> None:
    manifest = load_frozen_manifest()

    assert manifest["inference_changes"] == [
        "add the materializable baseline action to the local candidate set",
        "reserve one counterfactual sequence for that exact action",
    ]
    assert manifest["authority"]["mode"] == "shadow"
    assert manifest["challenger"]["can_satisfy_t8_0_gate"] is False


def test_long_manifest_freezes_twenty_five_actions_per_game() -> None:
    path = (
        Path(__file__).parents[1]
        / "theory"
        / "sage_t"
        / "sage_t8_5_long_frozen_manifest.json"
    )
    manifest = load_frozen_manifest(path)

    assert manifest["action_budget_per_reset"] == 25
    assert manifest["gate"]["minimum_actions"] == 50
    assert manifest["source_train_games"] == [
        "lp85-305b61c3",
        "su15-4c352900",
    ]


def test_missing_parameterization_becomes_a_local_candidate() -> None:
    legal = (
        ActionCandidate("ACTION6", {"x": 4, "y": 29}),
        ActionCandidate("ACTION6", {"x": 56, "y": 29}),
    )

    candidate = materialized_baseline_candidate(
        symbolic_action_name="ACTION6",
        symbolic_action_data={"x": 12, "y": 1},
        legal_actions=legal,
    )

    assert candidate == ActionCandidate("ACTION6", {"x": 12, "y": 1})
    assert candidate not in legal


def test_controller_injects_and_prioritizes_materialized_action(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_decide(self, **kwargs):  # type: ignore[no-untyped-def]
        captured["legal_actions"] = kwargs["legal_actions"]
        captured["preferred"] = self.decision_engine.preferred_action
        return "shadow-result"

    monkeypatch.setattr(
        base.InstrumentedSageTController,
        "decide",
        fake_decide,
    )
    controller = MaterializedActionController()
    result = controller.decide(
        symbolic_action_name="ACTION6",
        symbolic_action_data={"x": 12, "y": 1},
        legal_actions=(
            ActionCandidate("ACTION6", {"x": 4, "y": 29}),
            ActionCandidate("ACTION6", {"x": 56, "y": 29}),
        ),
    )

    expected = ActionCandidate("ACTION6", {"x": 12, "y": 1})
    assert result == "shadow-result"
    assert captured["legal_actions"][0] == expected
    assert captured["preferred"] == expected
    assert controller.decision_engine.preferred_action is None
