from __future__ import annotations

from pathlib import Path

from theory.sage_t.live_shadow_pilot_v10 import (
    MAXIMUM_ACTIONS_PER_GAME,
    MAXIMUM_TOTAL_ACTIONS,
    MINIMUM_ACTIONS_PER_GAME,
    MINIMUM_TOTAL_ACTIONS,
    RESETS,
    ExtendedLiveController,
    freeze_action_manifest,
    freeze_confirmation_manifest,
    load_confirmation_manifest,
)
from theory.sage_t.posterior_v11 import BudgetedRepairProgramPosterior


def test_extended_action_manifest_has_seven_resets(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"

    manifest = freeze_action_manifest(output_path=path)

    assert manifest["resets"] == RESETS == 7
    assert MAXIMUM_ACTIONS_PER_GAME == 350
    assert MAXIMUM_TOTAL_ACTIONS == 700
    assert manifest["gate"]["minimum_actions"] == MINIMUM_TOTAL_ACTIONS == 400
    assert MINIMUM_ACTIONS_PER_GAME == 200
    assert manifest["authority"]["mode"] == "shadow"


def test_confirmation_binds_r3_gate_and_extended_actions(
    tmp_path: Path,
) -> None:
    actions = tmp_path / "actions.json"
    confirmation = tmp_path / "confirmation.json"
    freeze_action_manifest(output_path=actions)
    frozen = freeze_confirmation_manifest(
        output_path=confirmation,
        action_manifest_path=actions,
    )

    loaded, gate = load_confirmation_manifest(
        confirmation,
        action_manifest_path=actions,
    )

    assert loaded == frozen
    assert gate["status"] == "READY_FOR_T8_6J_R3_LONG_LIVE"
    assert loaded["minimum_actions"] == 400
    assert loaded["maximum_actions"] == 700
    assert loaded["maximum_environment_interactions"] == 1400
    assert loaded["authority"] == "shadow"
    assert loaded["source_validation_authorized"] is False


def test_live_controller_uses_globally_budgeted_repair() -> None:
    controller = ExtendedLiveController(
        caps={
            "maximum_programs": 8,
            "maximum_sequences": 8,
            "maximum_particles_per_decision": 4,
            "ordinary_horizon": 1,
        }
    )

    assert isinstance(controller.posterior, BudgetedRepairProgramPosterior)
    assert controller.posterior.maximum_repair_contexts == 16
    assert controller.effective_mode.value == "shadow"
