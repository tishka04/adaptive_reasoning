from __future__ import annotations

from pathlib import Path

from theory.sage_t.live_shadow_pilot_v8 import (
    ACTIONS_PER_GAME,
    ACTIONS_PER_RESET,
    RESETS,
    TOTAL_ACTIONS,
    IncrementalLiveController,
    freeze_confirmation_manifest,
    freeze_long_action_manifest,
    load_confirmation_manifest,
)
from theory.sage_t.posterior_v9 import IncrementalMinimumKLProgramPosterior


def test_long_action_manifest_has_four_hundred_shadow_actions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.json"

    manifest = freeze_long_action_manifest(output_path=path)

    assert manifest["action_budget_per_reset"] == ACTIONS_PER_RESET == 50
    assert manifest["resets"] == RESETS == 4
    assert ACTIONS_PER_GAME == 200
    assert TOTAL_ACTIONS == 400
    assert manifest["gate"]["minimum_actions"] == 400
    assert manifest["authority"]["mode"] == "shadow"


def test_confirmation_binds_equivalence_and_long_action_protocol(
    tmp_path: Path,
) -> None:
    actions = tmp_path / "actions.json"
    confirmation = tmp_path / "confirmation.json"
    freeze_long_action_manifest(output_path=actions)
    frozen = freeze_confirmation_manifest(
        output_path=confirmation,
        action_manifest_path=actions,
    )

    loaded, equivalence = load_confirmation_manifest(
        confirmation,
        action_manifest_path=actions,
    )

    assert loaded == frozen
    assert equivalence["status"] == "READY_FOR_T8_6J_LONG_LIVE"
    assert loaded["actions"] == 400
    assert loaded["actions_per_game"] == 200
    assert loaded["authority"] == "shadow"
    assert loaded["source_validation_authorized"] is False


def test_live_controller_uses_incremental_posterior() -> None:
    controller = IncrementalLiveController(
        caps={
            "maximum_programs": 8,
            "maximum_sequences": 8,
            "maximum_particles_per_decision": 4,
            "ordinary_horizon": 1,
        }
    )

    assert isinstance(controller.posterior, IncrementalMinimumKLProgramPosterior)
    assert controller.effective_mode.value == "shadow"
