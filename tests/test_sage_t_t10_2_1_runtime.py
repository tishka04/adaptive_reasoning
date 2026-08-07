from __future__ import annotations

import copy
import inspect
import json
import queue
from pathlib import Path
from typing import Any

import pytest

from theory.sage_t import t10_2_1_protocol as protocol
from theory.sage_t import t10_2_1_runtime as runtime


def _manifest() -> dict[str, Any]:
    return {
        "manifest_checksum": "a" * 64,
        "environment_sha256": "b" * 64,
        "artifact_contract": {
            "artifact_root": protocol.DEFAULT_OUTPUT_DIR.as_posix(),
        },
    }


def _lane(
    *,
    split: runtime.LaneSplit = "discovery",
    game_index: int = 0,
    seed: int | None = None,
) -> runtime.SourceLaneKey:
    selected_seed = (
        runtime.DISCOVERY_SEEDS[0]
        if split == "discovery"
        else runtime.CONFIRMATION_SEEDS[0]
    )
    return runtime.SourceLaneKey(
        split=split,
        game_id=runtime.SOURCE_GAMES[game_index],
        seed=selected_seed if seed is None else seed,
    )


def _intent(
    lane: runtime.SourceLaneKey,
    *,
    reset_index: int = 0,
    step_index: int = 0,
    action_name: str = "ACTION1",
) -> runtime.ActionIntent:
    return runtime.ActionIntent(
        lane=lane,
        reset_index=reset_index,
        step_index=step_index,
        action={
            "name": action_name,
            "parameter_arity": 0,
            "grounding_sha256": "c" * 64,
        },
        manifest_checksum=_manifest()["manifest_checksum"],
    )


def test_runtime_namespace_and_registered_schedule_match_protocol() -> None:
    assert runtime.COLLECTION_FORMAT_VERSION == protocol.FORMAT_VERSION
    assert runtime.JOURNAL_FORMAT_VERSION == protocol.JOURNAL_FORMAT_VERSION
    assert runtime.CHECKPOINT_FORMAT_VERSION == protocol.CHECKPOINT_FORMAT_VERSION
    assert runtime.DEFAULT_OUTPUT_DIR == protocol.DEFAULT_OUTPUT_DIR
    assert runtime.SOURCE_MAXIMUM_AUTHORIZED_INTENTS == 4_608
    assert len(runtime.source_lane_registry()) == 18
    assert len({lane.lane_id for lane in runtime.source_lane_registry()}) == 18


def test_confirmation_cross_fit_is_true_counterbalanced_alternation() -> None:
    observed: dict[int, tuple[str, ...]] = {}
    for seed in runtime.CONFIRMATION_SEEDS:
        order = runtime.confirmation_controller_order(seed)
        observed[seed] = order
        assert len(order) == runtime.SOURCE_RESETS_PER_LANE
        assert order.count("learned") == 2
        assert order.count("capacity_matched_independent") == 2
        assert all(left != right for left, right in zip(order, order[1:]))

        lane = _lane(split="leave_one_game_out_confirmation", seed=seed)
        assert tuple(
            work.controller for work in runtime.reset_work_specs(lane)
        ) == order

    assert observed[111] == (
        "capacity_matched_independent",
        "learned",
        "capacity_matched_independent",
        "learned",
    )
    assert observed[112] == tuple(reversed(observed[111]))
    assert observed[113] == observed[111]


def _exact_cross_fit_material() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    manifest = _manifest()
    manifest["portable_code_sha256"] = {
        "theory/sage_t/t10_2_1_runtime.py": protocol.canonical_file_sha256(
            Path(runtime.__file__)
        )
    }
    manifest["code_sha256"] = {
        "theory/sage_t/t10_2_runtime.py": "d" * 64,
    }
    source_events: list[dict[str, Any]] = [
        {
            "event_id": f"discovery-{game_index}",
            "game_id": game_id,
            "split": "discovery",
        }
        for game_index, game_id in enumerate(runtime.SOURCE_GAMES)
    ]
    units: list[dict[str, Any]] = []
    for held_out_game in runtime.SOURCE_GAMES:
        training_games = tuple(
            game for game in runtime.SOURCE_GAMES if game != held_out_game
        )
        donor_ids = [
            str(event["event_id"])
            for event in source_events
            if event["split"] == "discovery"
            and event["game_id"] in training_games
        ]
        for seed in runtime.CONFIRMATION_SEEDS:
            resets: list[dict[str, Any]] = []
            for reset_index, controller in enumerate(
                runtime.confirmation_controller_order(seed)
            ):
                source_events.append(
                    {
                        "event_id": (
                            f"confirmation-{held_out_game}-{seed}-{reset_index}"
                        ),
                        "game_id": held_out_game,
                        "seed": seed,
                        "split": "leave_one_game_out_confirmation",
                        "selection": {
                            "reset_index": reset_index,
                            "controller": controller,
                        },
                    }
                )
                resets.append(
                    {
                        "reset_index": reset_index,
                        "controller": controller,
                        "action_count": 1,
                        "online_observations": 1,
                        "error_count": 0,
                        "initial_particle_count": 8,
                        "initial_class_count": 4,
                        "final_particle_count": 8,
                        "final_class_count": 4,
                        "stop_reason": "budget_exhausted",
                    }
                )
            units.append(
                {
                    "held_out_game": held_out_game,
                    "seed": seed,
                    "training_games": list(training_games),
                    "donor_event_count": len(donor_ids),
                    "donor_event_ids_sha256": runtime.canonical_sha256(donor_ids),
                    "held_out_prefit_events_used": 0,
                    "resets": resets,
                }
            )
    factory = runtime.T10_2_1SourceFactory(manifest=manifest)
    binding = protocol._factory_binding(factory, manifest)
    return manifest, source_events, units, binding


def _cross_fit_checks(
    manifest: dict[str, Any],
    source_events: list[dict[str, Any]],
    units: list[dict[str, Any]],
    factory: dict[str, Any],
) -> dict[str, bool]:
    with protocol._legacy_bindings():
        return protocol._legacy_cross_fit_checks(
            manifest=manifest,
            source_events=source_events,
            factory=factory,
            units=units,
        )


def test_cross_fit_audit_binds_exact_nine_units_factory_and_donors() -> None:
    manifest, source_events, units, factory = _exact_cross_fit_material()
    checks = _cross_fit_checks(manifest, source_events, units, factory)

    assert len(units) == 9
    assert {
        (unit["held_out_game"], unit["seed"]) for unit in units
    } == {
        (game, seed)
        for game in runtime.SOURCE_GAMES
        for seed in runtime.CONFIRMATION_SEEDS
    }
    assert factory["code_bound"] is True
    assert all(checks.values())


@pytest.mark.parametrize(
    ("tamper", "failed_check"),
    (
        ("missing_unit", "exact_nine_units"),
        ("factory", "factory_code_bound"),
        ("donor", "donor_events_bound"),
    ),
)
def test_cross_fit_semantic_tamper_is_detected(
    tamper: str,
    failed_check: str,
) -> None:
    manifest, source_events, units, factory = _exact_cross_fit_material()
    units = copy.deepcopy(units)
    factory = dict(factory)
    if tamper == "missing_unit":
        units.pop()
    elif tamper == "factory":
        factory["source_sha256"] = "f" * 64
    else:
        units[0]["donor_event_ids_sha256"] = "f" * 64

    checks = _cross_fit_checks(manifest, source_events, units, factory)
    assert checks[failed_check] is False
    assert not all(checks.values())


def test_journal_is_write_once_and_orders_intent_event_then_posterior(
    tmp_path: Path,
) -> None:
    lane = _lane()
    work = runtime.reset_work_specs(lane)[0]
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    intent = _intent(lane)
    sealed_event = runtime._t10_2_protocol.seal_event({"event_id": "event-0"})

    with pytest.raises(runtime.JournalIntegrityError, match="durable action intent"):
        journal.record_physical_event(intent=intent, event=sealed_event)

    assert journal.record_intent(intent) is True
    assert journal.record_intent(intent) is False
    conflicting = _intent(lane, action_name="ACTION2")
    assert conflicting.intent_id == intent.intent_id
    with pytest.raises(runtime.JournalConflictError, match="immutable durable record"):
        journal.record_intent(conflicting)

    premature_update = runtime.PosteriorUpdateReceipt(
        intent_id=intent.intent_id,
        event_checksum="d" * 64,
        status="APPLIED",
        posterior_state_sha256="e" * 64,
    )
    with pytest.raises(runtime.JournalIntegrityError, match="durable event"):
        journal.record_posterior_update(intent=intent, receipt=premature_update)

    physical = journal.record_physical_event(intent=intent, event=sealed_event)
    update = runtime.PosteriorUpdateReceipt(
        intent_id=intent.intent_id,
        event_checksum=physical.event_checksum,
        status="APPLIED",
        posterior_state_sha256="e" * 64,
    )
    assert journal.record_posterior_update(intent=intent, receipt=update) is True
    assert journal.record_posterior_update(intent=intent, receipt=update) is False

    accounting = journal.reset_accounting(work)
    assert accounting.authorized_intent_count == 1
    assert accounting.sealed_event_count == 1
    assert accounting.explicitly_unresolved_intent_count == 0
    assert accounting.unknown_intent_count == 0
    assert accounting.posterior_update_count == 1
    assert accounting.equation_holds


def test_parent_handler_persists_intent_before_outcome_and_outcome_before_update(
    tmp_path: Path,
) -> None:
    lane = _lane()
    work = runtime.reset_work_specs(lane)[0]
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    intent = _intent(lane)
    handler = runtime._ParentJournalMessageHandler(
        journal=journal,
        work=work,
        manifest=_manifest(),
        clock=lambda: 1.0,
        intent_deadline=10.0,
    )

    with pytest.raises(runtime.WorkerProtocolError, match="acknowledged intent"):
        handler(
            {
                "kind": "physical_event",
                "intent_id": intent.intent_id,
                "payload": {"event": {"event_id": "event-before-intent"}},
            }
        )

    intent_ack = handler({"kind": "action_intent", "payload": intent.to_dict()})
    assert intent_ack == {
        "kind": "intent_ack",
        "record_id": intent.intent_id,
        "accepted": True,
    }
    event_ack = handler(
        {
            "kind": "physical_event",
            "intent_id": intent.intent_id,
            "payload": {"event": {"event_id": "event-after-intent"}},
        }
    )
    assert event_ack is not None
    update = runtime.PosteriorUpdateReceipt(
        intent_id=intent.intent_id,
        event_checksum=str(event_ack["event_checksum"]),
        status="APPLIED",
        posterior_state_sha256="f" * 64,
    )
    posterior_ack = handler(
        {
            "kind": "posterior_update",
            "intent_id": intent.intent_id,
            "payload": update.to_dict(),
        }
    )
    assert posterior_ack == {
        "kind": "posterior_ack",
        "record_id": intent.intent_id,
        "accepted": True,
    }
    assert journal.reset_accounting(work).equation_holds


def test_parent_refuses_new_intent_at_cooperative_deadline(tmp_path: Path) -> None:
    lane = _lane()
    work = runtime.reset_work_specs(lane)[0]
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    handler = runtime._ParentJournalMessageHandler(
        journal=journal,
        work=work,
        manifest=_manifest(),
        clock=lambda: 10.0,
        intent_deadline=10.0,
    )

    with pytest.raises(runtime.WorkerTimeoutError, match="before intent authorization"):
        handler({"kind": "action_intent", "payload": _intent(lane).to_dict()})
    assert journal.reset_accounting(work).authorized_intent_count == 0


def test_deadline_crossing_during_intent_fsync_closes_without_ack_or_action(
    tmp_path: Path,
) -> None:
    lane = _lane()
    work = runtime.reset_work_specs(lane)[0]
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    intent = _intent(lane)
    handler = runtime._ParentJournalMessageHandler(
        journal=journal,
        work=work,
        manifest=_manifest(),
        clock=iter((9.0, 10.0)).__next__,
        intent_deadline=10.0,
    )

    with pytest.raises(
        runtime.WorkerTimeoutError,
        match="crossed while persisting intent",
    ):
        handler({"kind": "action_intent", "payload": intent.to_dict()})

    accounting = journal.reset_accounting(work)
    assert accounting.authorized_intent_count == 1
    assert accounting.sealed_event_count == 0
    assert accounting.explicitly_unresolved_intent_count == 1
    assert accounting.equation_holds
    with pytest.raises(runtime.WorkerProtocolError, match="acknowledged intent"):
        handler(
            {
                "kind": "physical_event",
                "intent_id": intent.intent_id,
                "payload": {"event": {"event_id": "must-not-execute"}},
            }
        )


def test_unresolved_intent_is_explicit_charged_and_excluded_from_posterior(
    tmp_path: Path,
) -> None:
    lane = _lane()
    work = runtime.reset_work_specs(lane)[0]
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    intent = _intent(lane)
    journal.record_intent(intent)
    unresolved = journal.record_unresolved_intent(
        intent=intent,
        reason="worker_failed",
    )

    assert unresolved.to_dict()["charged_action"] is True
    accounting = journal.reset_accounting(work)
    assert accounting.authorized_intent_count == 1
    assert accounting.sealed_event_count == 0
    assert accounting.explicitly_unresolved_intent_count == 1
    assert accounting.posterior_update_count == 0
    assert accounting.equation_holds

    with pytest.raises(runtime.JournalIntegrityError, match="durable event"):
        journal.record_posterior_update(
            intent=intent,
            receipt=runtime.PosteriorUpdateReceipt(
                intent_id=intent.intent_id,
                event_checksum="d" * 64,
                status="SKIPPED",
            ),
        )

    report = runtime.ResetReport(
        work=work,
        status="ABORTED",
        issued_intents=1,
        sealed_events=0,
        unresolved_intents=1,
        posterior_updates=0,
        elapsed_seconds=1.0,
        stop_reason="worker_failed",
        event_ids_sha256=runtime.canonical_sha256([]),
        continuation={},
    )
    journal.write_reset_report(report)
    assert journal.all_events(complete_resets_only=False) == ()


def test_resume_never_replays_partial_reset_and_refuses_unknown_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    destination = tmp_path / runtime.DEFAULT_OUTPUT_DIR
    journal = runtime.DurableCollectionJournal(
        destination / runtime.JOURNAL_DIRECTORY_NAME,
        manifest_checksum=manifest["manifest_checksum"],
    )
    lane = _lane()
    work = runtime.reset_work_specs(lane)[0]
    journal.record_intent(_intent(lane))

    with pytest.raises(runtime.ResumeRefusalError, match="replay is forbidden"):
        journal.assert_safe_resume_boundary(work)
    with pytest.raises(runtime.JournalIntegrityError, match="physical replay"):
        runtime.CollectionCheckpoint(
            manifest_checksum=manifest["manifest_checksum"],
            lane_registry_sha256=runtime.canonical_sha256(
                [item.to_dict() for item in runtime.source_lane_registry()]
            ),
            lane_reports=(),
            cumulative_active_seconds=0.0,
            journal_reconstructed=True,
            checkpoint_reconstructed=True,
            physical_steps_replayed_on_resume=1,
            revision=0,
        )

    (journal.root / "unknown-topology.json").write_text("{}\n", encoding="utf-8")
    assert journal.accounting().unknown_intent_count > 0
    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)
    report = runtime._collect_phase_impl(
        manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        output_dir=runtime.DEFAULT_OUTPUT_DIR,
        repo_root=tmp_path,
        env_factory=None,
        clock=iter((0.0, 1.0, 1.0)).__next__,
        _started_override=0.0,
    )
    assert report["status"] == "DATA_OR_PROVENANCE_INVALID"
    assert report["checks"]["no_unknown_intents"] is False
    assert report["action_accounting"]["unknown_intent_count"] == 1


def test_resume_accepts_reset_boundary_but_refuses_partial_physical_reset(
    tmp_path: Path,
) -> None:
    lane = _lane()
    first, second, *_ = runtime.reset_work_specs(lane)
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )

    journal.assert_safe_resume_boundary(first)
    complete = runtime.ResetReport(
        work=first,
        status="COMPLETE",
        issued_intents=0,
        sealed_events=0,
        unresolved_intents=0,
        posterior_updates=0,
        elapsed_seconds=1.0,
        stop_reason="terminal",
        event_ids_sha256=runtime.canonical_sha256([]),
        continuation={},
    )
    journal.write_reset_report(complete)
    journal.assert_safe_resume_boundary(first)
    journal.assert_safe_resume_boundary(second)

    journal.record_intent(_intent(lane, reset_index=second.reset_index))
    with pytest.raises(runtime.ResumeRefusalError, match="replay is forbidden"):
        journal.assert_safe_resume_boundary(second)


def test_factory_restarts_only_from_safe_reset_boundary(tmp_path: Path) -> None:
    class SpawnBoundaryReached(RuntimeError):
        pass

    class NoProcessContext:
        queue_calls = 0

        def Queue(self) -> Any:
            self.queue_calls += 1
            raise SpawnBoundaryReached

    lane = _lane()
    work = runtime.reset_work_specs(lane)[0]
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    factory = runtime.T10_2_1SourceFactory(manifest=_manifest())
    context = NoProcessContext()
    arguments = {
        "work": work,
        "journal": journal,
        "discovery_events": (),
        "continuation": {},
        "process_context": context,
        "lane_remaining_seconds": runtime.LANE_HARD_SECONDS,
        "cooperative_collection_remaining_seconds": (
            runtime.COLLECTION_COOPERATIVE_SECONDS
        ),
        "absolute_collection_remaining_seconds": (
            runtime.COLLECTION_ABSOLUTE_SECONDS
        ),
        "clock": lambda: 1.0,
    }

    with pytest.raises(SpawnBoundaryReached):
        factory.run_reset(**arguments)
    assert context.queue_calls == 1

    journal.record_intent(_intent(lane))
    context.queue_calls = 0
    with pytest.raises(runtime.ResumeRefusalError, match="replay is forbidden"):
        factory.run_reset(**arguments)
    assert context.queue_calls == 0


def test_local_reset_timeout_closes_without_replay_and_keeps_later_work(
    tmp_path: Path,
) -> None:
    first_lane, second_lane, *_ = runtime.source_lane_registry()
    first_reset, next_reset, *_ = runtime.reset_work_specs(first_lane)
    next_lane_reset = runtime.reset_work_specs(second_lane)[0]
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    journal.record_intent(_intent(first_lane))
    report = runtime._reset_report_from_outcome(
        journal=journal,
        work=first_reset,
        outcome=runtime.WorkerOutcome(
            status="HARD_TIMEOUT",
            elapsed_seconds=runtime.RESET_HARD_SECONDS,
            payload={
                "completed": False,
                "stop_reason": "hard_reset_timeout",
            },
            error_kind="WorkerTimeoutError",
        ),
        prior_continuation={},
    )

    assert report.status == "UNATTESTABLE"
    assert report.unresolved_intents == 1
    assert journal.reset_accounting(first_reset).equation_holds
    with pytest.raises(runtime.ResumeRefusalError, match="already ended"):
        journal.assert_safe_resume_boundary(first_reset)
    assert runtime._continuation_before(journal, next_reset) == {}
    journal.assert_safe_resume_boundary(next_reset)
    journal.assert_safe_resume_boundary(next_lane_reset)


@pytest.mark.parametrize(
    "reason",
    (
        "hard_reset_timeout",
        "cooperative_reset_deadline",
        "interrupted_before_reset_commit",
        "parent_interrupted",
    ),
)
def test_local_reset_stop_reasons_do_not_cancel_collection(reason: str) -> None:
    assert runtime._collection_stop_scope(reason) == "local_reset"


@pytest.mark.parametrize(
    "reason",
    (
        "registered_collection_deadline",
        "resource_gate",
        "data_or_provenance_error",
        "environment_call_unattestable",
    ),
)
def test_global_stop_reasons_cancel_remaining_collection(reason: str) -> None:
    assert runtime._collection_stop_scope(reason) == "global"


def test_minimal_data_report_replays_without_reading_corrupt_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    destination = tmp_path / runtime.DEFAULT_OUTPUT_DIR
    report = runtime._publish_minimal_data_report(
        destination=destination,
        manifest_checksum=manifest["manifest_checksum"],
        started=0.0,
        clock=lambda: 1.0,
        error=runtime.JournalIntegrityError("fixture corruption"),
        limits=protocol.DEFAULT_RESOURCE_LIMITS,
    )
    report_bytes = (
        destination / runtime.COLLECTION_REPORT_FILENAME
    ).read_bytes()
    journal_root = destination / runtime.JOURNAL_DIRECTORY_NAME
    journal_root.mkdir(parents=True)
    corrupt_journal = journal_root / "journal.json"
    corrupt_checkpoint = destination / runtime.CHECKPOINT_FILENAME
    corrupt_journal.write_bytes(b"{not-canonical-json\n")
    corrupt_checkpoint.write_bytes(b"{not-canonical-json\n")

    monkeypatch.setattr(protocol, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        protocol,
        "enforce_output_artifacts",
        lambda *args, **kwargs: None,
        raising=False,
    )
    replayed = runtime._collect_phase_impl(
        manifest_path=tmp_path / protocol.DEFAULT_MANIFEST_RELATIVE_PATH,
        output_dir=runtime.DEFAULT_OUTPUT_DIR,
        repo_root=tmp_path,
        env_factory=None,
        clock=lambda: pytest.fail("terminal DATA replay consulted the clock"),
        _started_override=0.0,
    )

    assert replayed == report
    assert (
        destination / runtime.COLLECTION_REPORT_FILENAME
    ).read_bytes() == report_bytes
    assert corrupt_journal.read_bytes() == b"{not-canonical-json\n"
    assert corrupt_checkpoint.read_bytes() == b"{not-canonical-json\n"


def test_checkpoint_reconstruction_is_idempotent_monotone_and_tamper_evident(
    tmp_path: Path,
) -> None:
    journal = runtime.DurableCollectionJournal(
        tmp_path / "journal",
        manifest_checksum=_manifest()["manifest_checksum"],
    )
    lane = _lane()
    first = journal.reconstruct_checkpoint(
        cumulative_active_seconds=12.0,
        open_lane=lane,
        open_lane_elapsed_seconds=4.0,
    )
    first_bytes = journal.checkpoint_path.read_bytes()
    second = journal.reconstruct_checkpoint(
        cumulative_active_seconds=5.0,
        open_lane=lane,
        open_lane_elapsed_seconds=1.0,
    )

    assert second == first
    assert second.cumulative_active_seconds == 12.0
    assert second.open_lane_elapsed_seconds == 4.0
    assert second.physical_steps_replayed_on_resume == 0
    assert journal.checkpoint_path.read_bytes() == first_bytes

    closed = journal.reconstruct_checkpoint(
        cumulative_active_seconds=12.0,
        close_open_lane=True,
    )
    assert closed.cumulative_active_seconds >= first.cumulative_active_seconds
    assert closed.open_lane_id is None
    assert closed.open_lane_elapsed_seconds == 0.0
    assert journal.reconstruct_checkpoint(close_open_lane=True) == closed

    tampered = json.loads(journal.checkpoint_path.read_text(encoding="utf-8"))
    tampered["revision"] += 1
    journal.checkpoint_path.write_text(
        runtime.canonical_json(tampered) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runtime.JournalIntegrityError, match="checkpoint checksum"):
        journal.load_checkpoint()


def test_watchdog_hard_timeout_terminates_then_kills_stuck_worker() -> None:
    class StuckProcess:
        def __init__(self) -> None:
            self.alive = True
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self.alive = False

        def join(self, timeout: float) -> None:
            self.join_calls += 1

    process = StuckProcess()
    watchdog = runtime.ProcessResetWatchdog(
        clock=lambda: runtime.RESET_HARD_SECONDS + 1.0,
        poll_seconds=0.01,
        termination_grace_seconds=0.0,
    )
    outcome = watchdog.supervise(
        process,
        work=runtime.reset_work_specs(_lane())[0],
        cancel_event=object(),
        outbound_queue=queue.Queue(),
        inbound_queue=queue.Queue(),
        message_handler=lambda message: None,
        cooperative_seconds=runtime.RESET_COOPERATIVE_SECONDS,
        hard_seconds=runtime.RESET_HARD_SECONDS,
        started_at=0.0,
    )

    assert outcome.status == "HARD_TIMEOUT"
    assert outcome.elapsed_seconds >= runtime.RESET_HARD_SECONDS
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == 2


def _open_invocation(path: Path, *, invocation_id: str) -> dict[str, Any]:
    return runtime._write_invocation_state(
        path,
        {
            "format_version": runtime.INVOCATION_STATE_FORMAT_VERSION,
            "manifest_checksum": _manifest()["manifest_checksum"],
            "invocation_id": invocation_id,
            "status": "OPEN",
            "base_cumulative_active_seconds": 0.0,
            "cumulative_active_seconds": 0.0,
            "monotonic_started": 1.0,
            "wall_monotonic_started": 1.0,
        },
    )


def test_invocation_terminal_first_writer_wins_closed_timeout_race(
    tmp_path: Path,
) -> None:
    closed_root = tmp_path / "closed-wins"
    closed_open_path = closed_root / runtime.INVOCATION_STATE_FILENAME
    closed_terminal_path = closed_root / runtime.INVOCATION_TERMINAL_FILENAME
    closed_open = _open_invocation(
        closed_open_path,
        invocation_id=runtime.canonical_sha256("closed-wins"),
    )
    closed = runtime._claim_invocation_terminal(
        closed_terminal_path,
        opened=closed_open,
        status="CLOSED",
        cumulative_active_seconds=10.0,
        terminal_monotonic=2.0,
    )
    closed_bytes = closed_terminal_path.read_bytes()
    late_timeout = runtime._record_global_hard_timeout(
        closed_open_path,
        closed_terminal_path,
    )

    assert late_timeout == closed
    assert late_timeout["status"] == "CLOSED"
    assert closed_terminal_path.read_bytes() == closed_bytes

    timeout_root = tmp_path / "timeout-wins"
    timeout_open_path = timeout_root / runtime.INVOCATION_STATE_FILENAME
    timeout_terminal_path = timeout_root / runtime.INVOCATION_TERMINAL_FILENAME
    timeout_open = _open_invocation(
        timeout_open_path,
        invocation_id=runtime.canonical_sha256("timeout-wins"),
    )
    timeout = runtime._record_global_hard_timeout(
        timeout_open_path,
        timeout_terminal_path,
    )
    assert timeout is not None
    timeout_bytes = timeout_terminal_path.read_bytes()
    late_close = runtime._claim_invocation_terminal(
        timeout_terminal_path,
        opened=timeout_open,
        status="CLOSED",
        cumulative_active_seconds=10.0,
        terminal_monotonic=2.0,
    )

    assert late_close == timeout
    assert late_close["status"] == "HARD_TIMEOUT"
    assert timeout_terminal_path.read_bytes() == timeout_bytes


def test_collection_report_is_bound_to_open_and_terminal_invocation_receipts(
    tmp_path: Path,
) -> None:
    open_path = tmp_path / runtime.INVOCATION_STATE_FILENAME
    terminal_path = tmp_path / runtime.INVOCATION_TERMINAL_FILENAME
    opened = _open_invocation(
        open_path,
        invocation_id=runtime.canonical_sha256("report-binding"),
    )
    report_core = {
        "status": "T10_2_1_SOURCE_COLLECTION_COMPLETE",
    }
    terminal = runtime._claim_invocation_terminal(
        terminal_path,
        opened=opened,
        status="CLOSED",
        cumulative_active_seconds=10.0,
        terminal_monotonic=2.0,
        report_core_checksum=runtime.canonical_sha256(report_core),
    )
    report = {
        **report_core,
        "invocation": runtime._invocation_report_binding(opened, terminal),
    }

    runtime._validate_report_invocation_binding(
        report,
        opened=opened,
        terminal=terminal,
        required=True,
    )
    assert report["invocation"]["open_state_checksum"] == (
        opened["state_checksum"]
    )
    assert report["invocation"]["terminal_checksum"] == (
        terminal["terminal_checksum"]
    )

    tampered = copy.deepcopy(report)
    tampered["invocation"]["terminal_checksum"] = "f" * 64
    with pytest.raises(runtime.JournalIntegrityError, match="binding drifted"):
        runtime._validate_report_invocation_binding(
            tampered,
            opened=opened,
            terminal=terminal,
            required=True,
        )

    timeout_root = tmp_path / "hard-timeout"
    timeout_open_path = timeout_root / runtime.INVOCATION_STATE_FILENAME
    timeout_terminal_path = timeout_root / runtime.INVOCATION_TERMINAL_FILENAME
    timeout_open = _open_invocation(
        timeout_open_path,
        invocation_id=runtime.canonical_sha256("timed-out-report"),
    )
    timeout_terminal = runtime._record_global_hard_timeout(
        timeout_open_path,
        timeout_terminal_path,
    )
    assert timeout_terminal is not None
    timed_out_success = {
        "status": "T10_2_1_SOURCE_COLLECTION_COMPLETE",
        "invocation": runtime._invocation_report_binding(
            timeout_open,
            timeout_terminal,
        ),
    }
    with pytest.raises(runtime.JournalIntegrityError, match="hard timeout"):
        runtime._validate_report_invocation_binding(
            timed_out_success,
            opened=timeout_open,
            terminal=timeout_terminal,
            required=True,
        )


def _otherwise_complete_collection(
    invocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "T10_2_1_SOURCE_COLLECTION_COMPLETE",
        "event_count": 1,
        "action_accounting": {
            "authorized_intent_count": 1,
            "sealed_event_count": 1,
            "explicitly_unresolved_intent_count": 0,
            "unknown_intent_count": 0,
            "maximum_authorized_intents": (
                protocol.SOURCE_MAXIMUM_ACTIONS
            ),
            "equation_holds": True,
        },
        "durability": {
            "lane_report_count": protocol.SOURCE_LANE_COUNT,
            "reset_report_count": protocol.SOURCE_RESET_REPORT_COUNT,
            "journal_reconstructed": True,
            "checkpoint_reconstructed": True,
            "physical_steps_replayed_on_resume": 0,
        },
        "checks": {"registered": True},
        "cross_fit_checks": {"registered": True},
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }
    if invocation is not None:
        payload["invocation"] = invocation
    return payload


def test_protocol_rejects_complete_collection_bound_to_hard_timeout(
    tmp_path: Path,
) -> None:
    closed_root = tmp_path / "closed"
    closed_open = _open_invocation(
        closed_root / runtime.INVOCATION_STATE_FILENAME,
        invocation_id=runtime.canonical_sha256("protocol-closed"),
    )
    closed_report_core = _otherwise_complete_collection()
    closed_terminal = runtime._claim_invocation_terminal(
        closed_root / runtime.INVOCATION_TERMINAL_FILENAME,
        opened=closed_open,
        status="CLOSED",
        cumulative_active_seconds=10.0,
        terminal_monotonic=2.0,
        report_core_checksum=runtime.canonical_sha256(closed_report_core),
    )
    closed_report = {
        **closed_report_core,
        "invocation": runtime._invocation_report_binding(
            closed_open,
            closed_terminal,
        ),
    }
    assert protocol._collection_acquisition_passed(closed_report)

    timeout_root = tmp_path / "timeout"
    timeout_open_path = timeout_root / runtime.INVOCATION_STATE_FILENAME
    timeout_open = _open_invocation(
        timeout_open_path,
        invocation_id=runtime.canonical_sha256("protocol-timeout"),
    )
    timeout_terminal = runtime._record_global_hard_timeout(timeout_open_path)
    assert timeout_terminal is not None
    timeout_report = _otherwise_complete_collection(
        runtime._invocation_report_binding(timeout_open, timeout_terminal)
    )
    assert not protocol._collection_acquisition_passed(timeout_report)


def test_collection_watchdog_requests_windows_process_tree_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCancelEvent:
        @staticmethod
        def wait(timeout: float) -> bool:
            return False

        @staticmethod
        def is_set() -> bool:
            return False

    class FakeOS:
        name = "nt"

    captured: dict[str, Any] = {}

    class FakeCompleted:
        returncode = 0

    def fake_run(command: list[str], **kwargs: Any) -> FakeCompleted:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeCompleted()

    invocation_state_path = tmp_path / runtime.INVOCATION_STATE_FILENAME
    invocation_terminal_path = tmp_path / runtime.INVOCATION_TERMINAL_FILENAME
    opened = _open_invocation(
        invocation_state_path,
        invocation_id=runtime.canonical_sha256("watchdog-fixture"),
    )
    monkeypatch.setattr(runtime, "os", FakeOS())
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    runtime._collection_hard_watchdog_entry(
        parent_pid=1234,
        deadline_monotonic=99.0,
        cancel_event=FakeCancelEvent(),
        invocation_state_path=str(invocation_state_path),
        invocation_terminal_path=str(invocation_terminal_path),
    )

    assert captured["command"] == [
        "taskkill.exe",
        "/PID",
        "1234",
        "/T",
        "/F",
    ]
    assert captured["kwargs"]["check"] is False
    timeout_terminal = runtime._read_invocation_terminal(
        invocation_terminal_path,
        opened=opened,
    )
    assert timeout_terminal is not None
    assert timeout_terminal["status"] == "HARD_TIMEOUT"
    assert timeout_terminal["cumulative_active_seconds"] == (
        runtime.COLLECTION_ABSOLUTE_SECONDS
    )


def test_closed_without_bound_report_keeps_deadline_kill_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCancelEvent:
        @staticmethod
        def wait(timeout: float) -> bool:
            return False

        @staticmethod
        def is_set() -> bool:
            return False

    class FakeOS:
        name = "nt"

    class FakeCompleted:
        returncode = 0

    class AdvancingClock:
        current = 100.0

        def __call__(self) -> float:
            self.current += runtime.LANE_FINALIZATION_SECONDS
            return self.current

    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> FakeCompleted:
        captured["command"] = command
        return FakeCompleted()

    open_path = tmp_path / runtime.INVOCATION_STATE_FILENAME
    terminal_path = tmp_path / runtime.INVOCATION_TERMINAL_FILENAME
    opened = _open_invocation(
        open_path,
        invocation_id=runtime.canonical_sha256("closed-without-report"),
    )
    runtime._claim_invocation_terminal(
        terminal_path,
        opened=opened,
        status="CLOSED",
        cumulative_active_seconds=10.0,
        terminal_monotonic=2.0,
        report_core_checksum=runtime.canonical_sha256({"projected": True}),
    )
    assert runtime._remaining_collection_watchdog_seconds(tmp_path) == (
        runtime.LANE_FINALIZATION_SECONDS
    )
    monkeypatch.setattr(runtime, "os", FakeOS())
    monkeypatch.setattr(runtime.time, "monotonic", AdvancingClock())
    def fail_post_deadline_sleep(seconds: float) -> None:
        raise AssertionError(f"post-deadline grace attempted: {seconds}")

    monkeypatch.setattr(runtime.time, "sleep", fail_post_deadline_sleep)
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    runtime._collection_hard_watchdog_entry(
        parent_pid=1234,
        deadline_monotonic=99.0,
        cancel_event=FakeCancelEvent(),
        invocation_state_path=str(open_path),
        invocation_terminal_path=str(terminal_path),
    )

    assert captured["command"] == [
        "taskkill.exe",
        "/PID",
        "1234",
        "/T",
        "/F",
    ]


def test_reset_budget_starts_before_process_spawn() -> None:
    class Clock:
        current = 10.0

        def __call__(self) -> float:
            return self.current

    class Process:
        def __init__(self, clock: Clock) -> None:
            self.clock = clock
            self.pid = 4321

        def start(self) -> None:
            self.clock.current = 15.0

    class Context:
        def __init__(self, clock: Clock) -> None:
            self.clock = clock

        @staticmethod
        def Queue() -> queue.Queue[Any]:
            return queue.Queue()

        @staticmethod
        def Event() -> object:
            return object()

        def Process(self, **kwargs: Any) -> Process:
            return Process(self.clock)

    class Journal:
        @staticmethod
        def assert_safe_resume_boundary(work: runtime.ResetWorkSpec) -> None:
            return None

        @staticmethod
        def intents_for_reset(
            work: runtime.ResetWorkSpec,
        ) -> tuple[runtime.ActionIntent, ...]:
            return ()

    class Watchdog:
        started_at: float | None = None

        def supervise(self, process: Any, **kwargs: Any) -> runtime.WorkerOutcome:
            self.started_at = float(kwargs["started_at"])
            return runtime.WorkerOutcome(
                status="COMPLETE",
                elapsed_seconds=5.0,
                payload={"completed": True},
            )

    clock = Clock()
    watchdog = Watchdog()
    factory = object.__new__(runtime.T10_2_1SourceFactory)
    factory.manifest = _manifest()
    factory.watchdog = watchdog
    factory.clone_for_worker = lambda: object()

    outcome = factory.run_reset(
        work=runtime.reset_work_specs(_lane())[0],
        journal=Journal(),
        discovery_events=(),
        continuation={},
        process_context=Context(clock),
        lane_remaining_seconds=runtime.LANE_HARD_SECONDS,
        cooperative_collection_remaining_seconds=(
            runtime.COLLECTION_COOPERATIVE_SECONDS
        ),
        absolute_collection_remaining_seconds=runtime.COLLECTION_ABSOLUTE_SECONDS,
        clock=clock,
    )

    assert outcome.status == "COMPLETE"
    assert watchdog.started_at == 10.0
    assert clock.current == 15.0


def test_registered_wall_budgets_and_terminal_report_order() -> None:
    assert runtime.RESET_COOPERATIVE_SECONDS < runtime.RESET_HARD_SECONDS
    assert (
        runtime.SOURCE_RESETS_PER_LANE * runtime.RESET_HARD_SECONDS
        + runtime.LANE_FINALIZATION_SECONDS
        == runtime.LANE_HARD_SECONDS
    )
    assert (
        len(runtime.source_lane_registry()) * runtime.LANE_HARD_SECONDS
        < runtime.COLLECTION_COOPERATIVE_SECONDS
        < runtime.COLLECTION_ABSOLUTE_SECONDS
    )

    source = inspect.getsource(runtime._collect_phase_impl)
    artifact_guard = source.index("protocol.enforce_output_artifacts(destination")
    terminal_write = source.rindex("_write_once(report_path, report)")
    assert artifact_guard < terminal_write
    reset_scope = source.index(
        "_collection_stop_scope(reset_report.stop_reason)"
    )
    assert source.index("continue", reset_scope) < source.index(
        "break", reset_scope
    )
    lane_scope = source.index("_collection_stop_scope(prior_reason)")
    assert source.index("continue", lane_scope) < source.index(
        "break", lane_scope
    )
    resource_stop = source.index("except _t10_2_protocol.ResourceGateError")
    data_stop = source.index("except Exception as exc")
    assert source.index('terminal_reason = "resource_gate"', resource_stop) > (
        resource_stop
    )
    assert source.index("data_error = exc", data_stop) > data_stop
