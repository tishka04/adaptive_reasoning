from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from theory.sage_t import t10_2_5_protocol as protocol
from theory.sage_t import t10_2_5_runtime as runtime


def _receipt() -> dict[str, object]:
    seeds = protocol._derive_recovery_seeds(
        {"checkpoint_checksum": "a" * 64, "orphan_work_id": "work"}
    )
    return {
        "recovery_seeds": seeds,
        "recovery_lanes": [
            protocol._recovery_lane("su15-4c352900", seed) for seed in seeds
        ],
    }


def test_worker_terminator_refuses_collector_pid() -> None:
    with pytest.raises(runtime.WorkerProtocolError, match="collector pid"):
        runtime._terminate_worker_tree(os.getpid())


def test_worker_watchdog_targets_only_registered_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(runtime, "_terminate_worker_tree", calls.append)

    runtime._worker_hard_watchdog_entry(
        4242,
        runtime.time.monotonic() - 1.0,
        runtime.threading.Event(),
        runtime.threading.Event(),
    )

    assert calls == [4242]
    assert calls[0] != os.getpid()


def test_worker_only_binding_is_scoped() -> None:
    original = runtime._parent_runtime.ActionBudgetSourceFactory.run_reset

    with runtime.worker_only_watchdog_binding():
        assert (
            runtime._parent_runtime.ActionBudgetSourceFactory.run_reset
            is runtime._worker_scoped_run_reset
        )

    assert runtime._parent_runtime.ActionBudgetSourceFactory.run_reset is original


def test_recovery_journal_binding_registers_three_odd_lanes() -> None:
    original_seeds = runtime._kernel_runtime.CONFIRMATION_SEEDS
    original_registry = runtime._kernel_runtime.source_lane_registry
    original_cap = runtime._kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS

    with runtime.recovery_journal_bindings(_receipt()) as lanes:
        assert len(lanes) == 3
        assert all(lane.seed % 2 == 1 for lane in lanes)
        assert runtime._kernel_runtime.source_lane_registry() == lanes
        assert (
            runtime._kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS
            == protocol.RECOVERY_MAXIMUM_ACTIONS
        )
        assert all(
            [work.controller for work in runtime._kernel_runtime.reset_work_specs(lane)]
            == [
                "capacity_matched_independent",
                "learned",
                "capacity_matched_independent",
                "learned",
            ]
            for lane in lanes
        )

    assert runtime._kernel_runtime.CONFIRMATION_SEEDS == original_seeds
    assert runtime._kernel_runtime.source_lane_registry is original_registry
    assert runtime._kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS == original_cap


def test_recovery_factory_uses_frozen_kernel_for_cache_binding() -> None:
    factory = object.__new__(runtime.RecoveryDualCachedSourceFactory)
    factory.cache_root = "new"
    factory.predecessor_cache_root = "old"
    factory.continuation_manifest_checksum = "c" * 64
    factory.predecessor_manifest_checksum = "p" * 64
    factory.cache_kernel_manifest_checksum = "k" * 64
    factory.manifest_checksum = "r" * 64

    coordinator = factory._coordinator()

    assert coordinator.parent_kernel_manifest_checksum == "k" * 64
    assert coordinator.parent_kernel_manifest_checksum != factory.manifest_checksum


def test_accepted_audit_preserves_logical_seed_and_controller_order() -> None:
    orphan_lane = {
        "split": "leave_one_game_out_confirmation",
        "game_id": "su15-4c352900",
        "seed": 111,
        "lane_id": "orphan",
    }
    reset_controllers = [
        "capacity_matched_independent",
        "learned",
        "capacity_matched_independent",
        "learned",
    ]
    recovery_lane = SimpleNamespace(
        game_id="su15-4c352900",
        seed=1000001,
        to_dict=lambda: {"game_id": "su15-4c352900", "seed": 1000001},
    )
    accepted = SimpleNamespace(
        lane=recovery_lane,
        resets=[SimpleNamespace(work=SimpleNamespace(controller=item)) for item in reset_controllers],
        cross_fit_unit={
            "held_out_game": "su15-4c352900",
            "seed": 1000001,
            "training_games": ["bp35-0a0ad940", "lp85-305b61c3"],
            "held_out_prefit_events_used": 0,
        },
    )
    parent_units = []
    for game in runtime._kernel_protocol.SOURCE_GAMES:
        if game == "su15-4c352900":
            continue
        for seed in runtime._kernel_runtime.CONFIRMATION_SEEDS:
            parent_units.append(
                SimpleNamespace(
                    cross_fit_unit={
                        "held_out_game": game,
                        "seed": seed,
                        "training_games": [
                            item for item in runtime._kernel_protocol.SOURCE_GAMES if item != game
                        ],
                        "held_out_prefit_events_used": 0,
                    }
                )
            )
    parent_units.extend(
        SimpleNamespace(
            cross_fit_unit={
                "held_out_game": "su15-4c352900",
                "seed": seed,
                "training_games": ["bp35-0a0ad940", "lp85-305b61c3"],
                "held_out_prefit_events_used": 0,
            }
        )
        for seed in (112, 113)
    )
    audit = runtime._accepted_cross_fit_audit(
        manifest={
            "manifest_checksum": "m" * 64,
            "migration_receipt": {"orphan_lane": orphan_lane},
        },
        parent_state={"complete_lanes": parent_units},
        recovery_state={"accepted": accepted},
        accepted_events=[{"event_id": "one"}],
    )

    assert audit["passed"] is True
    replacement = next(
        unit for unit in audit["units"] if unit.get("logical_replacement_for_lane_id")
    )
    assert replacement["seed"] == 111
    assert replacement["physical_recovery_seed"] == 1000001

