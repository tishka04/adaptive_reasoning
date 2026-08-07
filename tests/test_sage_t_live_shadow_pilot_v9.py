from __future__ import annotations

from pathlib import Path

from theory.sage_t.live_shadow_pilot_v9 import (
    ACTIONS_PER_GAME,
    ACTIONS_PER_RESET,
    RESETS,
    TOTAL_ACTIONS,
    RepairMemoLiveController,
    _repair_metrics,
    freeze_confirmation_manifest,
    load_confirmation_manifest,
)
from theory.sage_t.posterior_v10 import ContextMemoizedRepairProgramPosterior


def test_confirmation_binds_r2_gate_and_long_action_protocol(
    tmp_path: Path,
) -> None:
    confirmation = tmp_path / "confirmation.json"

    frozen = freeze_confirmation_manifest(output_path=confirmation)
    loaded, repair_report = load_confirmation_manifest(confirmation)

    assert loaded == frozen
    assert repair_report["status"] == "READY_FOR_T8_6J_R2_LONG_LIVE"
    assert loaded["actions"] == TOTAL_ACTIONS == 400
    assert loaded["environment_interactions"] == 800
    assert loaded["actions_per_reset"] == ACTIONS_PER_RESET == 50
    assert loaded["resets"] == RESETS == 4
    assert ACTIONS_PER_GAME == 200
    assert loaded["authority"] == "shadow"
    assert loaded["source_validation_authorized"] is False


def test_live_controller_uses_context_memoized_repair() -> None:
    controller = RepairMemoLiveController(
        caps={
            "maximum_programs": 8,
            "maximum_sequences": 8,
            "maximum_particles_per_decision": 4,
            "ordinary_horizon": 1,
        }
    )

    assert isinstance(
        controller.posterior,
        ContextMemoizedRepairProgramPosterior,
    )
    assert controller.effective_mode.value == "shadow"


def test_repair_metrics_use_observed_snapshot_deltas() -> None:
    class FakePosterior:
        def performance_snapshot(self):  # type: ignore[no-untyped-def]
            return {
                "unique_repair_contexts": 2,
                "repair_context_skips": 7,
                "semantic_cache_hits": 11,
            }

    class FakeController:
        posterior = FakePosterior()

    metrics = _repair_metrics(
        [
            {"repairs_attempted_delta": 1, "repairs_admitted_delta": 3},
            {"repairs_attempted_delta": 0, "repairs_admitted_delta": 0},
        ],
        {"lp85": FakeController()},  # type: ignore[arg-type]
    )

    assert metrics["attempted"] == 1
    assert metrics["admitted"] == 3
    assert metrics["maximum_attempted_per_observation"] == 1
    assert metrics["maximum_admitted_per_observation"] == 3
    assert metrics["unique_contexts"] == 2
    assert metrics["context_skips"] == 7
    assert metrics["semantic_cache_hits"] == 11
