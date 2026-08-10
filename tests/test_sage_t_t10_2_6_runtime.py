from __future__ import annotations

import multiprocessing
from types import SimpleNamespace
from typing import Any

import pytest

from theory.sage_t import t10_2_6_protocol as protocol
from theory.sage_t import t10_2_6_runtime as runtime


class _ProductionSpawnProbeFactory(
    runtime._predecessor_runtime.RecoveryDualCachedSourceFactory
):
    """Fail after work decode, before runtime or environment construction."""

    def __init__(self, receipt: dict[str, Any]) -> None:
        self.manifest = {"migration_receipt": receipt}
        self.watchdog = None
        self._runtime = None

    def restore_completed_discovery(self, events: Any) -> None:
        del events
        raise RuntimeError("probe_reached_after_work_decode")


def _receipt() -> dict[str, Any]:
    anchor = {
        "parent_checkpoint_checksum": "a" * 64,
        "predecessor_failure_report_checksum": "b" * 64,
        "predecessor_recovery_seeds": [1650445, 1412747, 1438133],
    }
    seeds = protocol._derive_recovery_seeds(anchor)
    return {
        "recovery_seeds": seeds,
        "recovery_lanes": [
            protocol._recovery_lane("su15-4c352900", seed) for seed in seeds
        ],
    }


def _first_work(receipt: dict[str, Any]) -> dict[str, Any]:
    with runtime.recovery_journal_bindings(receipt) as lanes:
        return runtime._kernel_runtime.reset_work_specs(lanes[0])[0].to_dict()


def _spawn_parse_probe(
    receipt: dict[str, Any], work_payload: dict[str, Any], queue: Any
) -> None:
    try:
        with runtime.recovery_journal_bindings(receipt):
            work = runtime._kernel_runtime.ResetWorkSpec.from_dict(work_payload)
        queue.put({"accepted": True, "seed": work.lane.seed})
    except BaseException as exc:  # pragma: no cover - asserted through child output
        queue.put({"accepted": False, "error": f"{type(exc).__name__}:{exc}"})


def test_parent_recovery_binding_registers_and_restores_fresh_lanes() -> None:
    receipt = _receipt()
    original_seeds = runtime._kernel_runtime.CONFIRMATION_SEEDS
    original_registry = runtime._kernel_runtime.source_lane_registry
    original_cap = runtime._kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS

    with runtime.recovery_journal_bindings(receipt) as lanes:
        assert tuple(lane.seed for lane in lanes) == tuple(receipt["recovery_seeds"])
        assert runtime._kernel_runtime.source_lane_registry() == lanes
        assert (
            runtime._kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS
            == protocol.RECOVERY_MAXIMUM_ACTIONS
        )
        assert runtime._kernel_runtime.ResetWorkSpec.from_dict(
            runtime._kernel_runtime.reset_work_specs(lanes[0])[0].to_dict()
        ).lane.seed == lanes[0].seed

    assert runtime._kernel_runtime.CONFIRMATION_SEEDS == original_seeds
    assert runtime._kernel_runtime.source_lane_registry is original_registry
    assert runtime._kernel_runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS == original_cap


def test_spawned_process_can_decode_fresh_seed_before_any_action() -> None:
    receipt = _receipt()
    work = _first_work(receipt)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_parse_probe, args=(receipt, work, queue))

    process.start()
    process.join(timeout=20.0)

    assert process.exitcode == 0
    observed = queue.get(timeout=5.0)
    assert observed == {"accepted": True, "seed": work["lane"]["seed"]}


def test_production_spawn_target_reaches_worker_after_seed_decode() -> None:
    receipt = _receipt()
    work = _first_work(receipt)
    context = multiprocessing.get_context("spawn")
    outbound = context.Queue()
    inbound = context.Queue()
    cancel = context.Event()
    factory = _ProductionSpawnProbeFactory(receipt)
    process = context.Process(
        target=runtime._recovery_spawn_worker_entry,
        args=(factory, work, (), {}, cancel, outbound, inbound),
    )

    process.start()
    process.join(timeout=20.0)

    assert process.exitcode == 0
    message = outbound.get(timeout=5.0)
    assert message["kind"] == "worker_failed"
    assert message["error_kind"] == "RuntimeError"
    assert message["work_id"] == work["work_id"]


def test_production_child_entry_installs_registry_before_frozen_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()
    work = _first_work(receipt)
    factory = object.__new__(
        runtime._predecessor_runtime.RecoveryDualCachedSourceFactory
    )
    factory.manifest = {"migration_receipt": receipt}
    observed: list[int] = []

    def probe(*args: Any) -> None:
        decoded = runtime._kernel_runtime.ResetWorkSpec.from_dict(args[1])
        observed.append(decoded.lane.seed)

    monkeypatch.setattr(
        runtime._predecessor_runtime._predecessor_runtime,
        "_dual_cache_reset_worker_entry",
        probe,
    )

    runtime._recovery_spawn_worker_entry(factory, work)

    assert observed == [work["lane"]["seed"]]


def test_spawn_target_binding_is_scoped_after_dual_cache_binding() -> None:
    parent_runtime = runtime._predecessor_runtime._parent_runtime
    original = parent_runtime._action_budget_reset_worker_entry

    with runtime.recovery_spawn_worker_binding():
        assert (
            parent_runtime._action_budget_reset_worker_entry
            is runtime._recovery_spawn_worker_entry
        )

    assert parent_runtime._action_budget_reset_worker_entry is original


def test_collect_cli_returns_nonzero_on_scientific_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runtime,
        "collect_phase",
        lambda **_: {"status": "FAIL_T10_2_6_RECOVERY"},
    )

    code = runtime.main(["collect"])

    assert code == 3
    assert "FAIL_T10_2_6_RECOVERY" in capsys.readouterr().out


def test_accepted_audit_preserves_logical_seed_and_controller_order() -> None:
    orphan_lane = {
        "split": "leave_one_game_out_confirmation",
        "game_id": "su15-4c352900",
        "seed": 111,
        "lane_id": "orphan",
    }
    controllers = [
        "capacity_matched_independent",
        "learned",
        "capacity_matched_independent",
        "learned",
    ]
    recovery_lane = SimpleNamespace(
        game_id="su15-4c352900",
        seed=2_000_001,
        to_dict=lambda: {"game_id": "su15-4c352900", "seed": 2_000_001},
    )
    accepted = SimpleNamespace(
        lane=recovery_lane,
        resets=[SimpleNamespace(work=SimpleNamespace(controller=item)) for item in controllers],
        cross_fit_unit={
            "held_out_game": "su15-4c352900",
            "seed": 2_000_001,
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
                            item
                            for item in runtime._kernel_protocol.SOURCE_GAMES
                            if item != game
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
    assert replacement["physical_recovery_seed"] == 2_000_001
