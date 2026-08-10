from __future__ import annotations

import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from theory.sage_t import t10_2_7_protocol as protocol
from theory.sage_t import t10_2_7_runtime as runtime


def _receipt() -> dict[str, Any]:
    anchor = {
        "partial_journal_checksum": "a" * 64,
        "partial_intent_checksum": "b" * 64,
        "t10_2_5_recovery_seeds": [1650445, 1412747, 1438133],
        "t10_2_6_recovery_seeds": [2320669, 2468453, 2325737],
    }
    seeds = protocol._derive_recovery_seeds(anchor)
    return {
        "receipt_checksum": "c" * 64,
        "recovery_seeds": seeds,
        "recovery_lanes": [
            protocol._recovery_lane("su15-4c352900", seed) for seed in seeds
        ],
        "orphan_lane": {
            "split": "leave_one_game_out_confirmation",
            "game_id": "su15-4c352900",
            "seed": 111,
            "lane_id": "orphan",
        },
    }


def _kernel() -> dict[str, Any]:
    return {
        "manifest_checksum": protocol.PARENT_KERNEL_MANIFEST_CHECKSUM,
        "environment_sha256": "d" * 64,
        "source_plan": {"frozen": True},
        "firewall": {"production_authority": False},
    }


def _manifest() -> dict[str, Any]:
    kernel = _kernel()
    return {
        "manifest_checksum": "e" * 64,
        "migration_receipt": _receipt(),
        "execution_manifest_contract": protocol._execution_contract(kernel),
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


def test_execution_manifest_preserves_every_kernel_field_and_adds_identity() -> None:
    kernel = _kernel()
    manifest = _manifest()

    execution = runtime.build_execution_manifest(
        protocol_manifest=manifest,
        kernel=kernel,
    )

    assert execution["environment_sha256"] == kernel["environment_sha256"]
    assert execution["source_plan"] == kernel["source_plan"]
    assert execution["firewall"] == kernel["firewall"]
    assert execution["manifest_checksum"] == manifest["manifest_checksum"]
    assert execution["migration_receipt"] == manifest["migration_receipt"]


def test_execution_manifest_rejects_a_contract_from_another_kernel() -> None:
    manifest = _manifest()
    drifted = _kernel()
    drifted["environment_sha256"] = "f" * 64

    with pytest.raises(runtime.ManifestDriftError, match="execution contract"):
        runtime.build_execution_manifest(
            protocol_manifest=manifest,
            kernel=drifted,
        )


def test_first_event_seal_preflight_uses_kernel_environment_provenance() -> None:
    result = runtime._first_event_seal_preflight(
        protocol_manifest=_manifest(),
        kernel=_kernel(),
    )

    assert result["passed"] is True
    assert result["environment_sha256"] == "d" * 64
    assert len(result["event_checksum"]) == 64


def test_runner_exception_is_durably_closed_as_unattestable_lane(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    receipt = manifest["migration_receipt"]
    with runtime.recovery_journal_bindings(receipt) as lanes:
        lane = lanes[0]
        work = runtime._kernel_runtime.reset_work_specs(lane)[0]
        journal = runtime._kernel_runtime.DurableCollectionJournal(
            tmp_path / "journal",
            manifest_checksum=manifest["manifest_checksum"],
        )

        class FailingFactory:
            def run_reset(self, **kwargs: Any) -> Any:
                intent = runtime._kernel_runtime.ActionIntent(
                    lane=lane,
                    reset_index=0,
                    step_index=0,
                    action={
                        "name": "ACTION6",
                        "parameter_arity": 2,
                        "grounding_sha256": "a" * 64,
                    },
                    manifest_checksum=manifest["manifest_checksum"],
                )
                kwargs["journal"].record_intent(intent)
                raise KeyError("synthetic_parent_seal_failure")

        report = runtime._run_reset_and_finalize(
            factory=FailingFactory(),
            journal=journal,
            work=work,
            discovery_events=(),
            prior={},
            process_context=None,
            lane_remaining_seconds=10.0,
            cooperative_collection_remaining_seconds=10.0,
            absolute_collection_remaining_seconds=10.0,
            clock=lambda: 1.0,
        )
        lane_report = runtime._predecessor_runtime._predecessor_runtime._finalize_recovery_lane(
            journal,
            lane,
            discovery_events=(),
            elapsed_seconds=0.0,
        )

        assert report.status == "UNATTESTABLE"
        assert report.stop_reason == "parent_runner_exception:KeyError"
        assert report.issued_intents == 1
        assert report.sealed_events == 0
        assert report.unresolved_intents == 1
        assert journal.read_reset_report(work) == report
        assert journal.reset_accounting(work).equation_holds
        assert lane_report.status == "ABORTED"
        assert journal.read_lane_report(lane) == lane_report


def test_spawned_process_can_decode_t10_2_7_seed_before_any_action() -> None:
    receipt = _receipt()
    work = _first_work(receipt)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_parse_probe, args=(receipt, work, queue))

    process.start()
    process.join(timeout=20.0)

    assert process.exitcode == 0
    assert queue.get(timeout=5.0) == {"accepted": True, "seed": work["lane"]["seed"]}


def test_child_entry_requires_environment_provenance_before_frozen_worker() -> None:
    receipt = _receipt()
    work = _first_work(receipt)
    factory_type = runtime._predecessor_runtime._predecessor_runtime.RecoveryDualCachedSourceFactory
    factory = object.__new__(factory_type)
    factory.manifest = {"migration_receipt": receipt}

    with pytest.raises(runtime.WorkerProtocolError, match="environment provenance"):
        runtime._recovery_spawn_worker_entry(factory, work)


def test_spawn_target_binding_is_scoped() -> None:
    parent_runtime = runtime._predecessor_runtime._predecessor_runtime._parent_runtime
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
        lambda **_: {"status": "FAIL_T10_2_7_RECOVERY"},
    )

    code = runtime.main(["collect"])

    assert code == 3
    assert "FAIL_T10_2_7_RECOVERY" in capsys.readouterr().out


def test_cross_fit_audit_marks_both_predecessor_failures_excluded() -> None:
    controllers = [
        "capacity_matched_independent",
        "learned",
        "capacity_matched_independent",
        "learned",
    ]
    recovery_lane = SimpleNamespace(
        game_id="su15-4c352900",
        seed=3_000_001,
        to_dict=lambda: {"game_id": "su15-4c352900", "seed": 3_000_001},
    )
    accepted = SimpleNamespace(
        lane=recovery_lane,
        resets=[SimpleNamespace(work=SimpleNamespace(controller=item)) for item in controllers],
        cross_fit_unit={
            "held_out_game": "su15-4c352900",
            "seed": 3_000_001,
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
            "manifest_checksum": "e" * 64,
            "migration_receipt": {"orphan_lane": _receipt()["orphan_lane"]},
        },
        parent_state={"complete_lanes": parent_units},
        recovery_state={"accepted": accepted},
        accepted_events=[{"event_id": "one"}],
    )

    assert audit["passed"] is True
    assert audit["checks"]["t10_2_5_zero_action_failures_excluded"] is True
    assert audit["checks"]["t10_2_6_partial_lane_excluded"] is True
