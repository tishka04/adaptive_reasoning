from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from theory.sage_t import t10_3_2_protocol as protocol
from theory.sage_t import t10_3_2_runtime as runtime
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry


@dataclass(frozen=True)
class _Action:
    name: str
    action_args: dict


def _manifest() -> dict:
    return {
        "manifest_checksum": "m" * 64,
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
        },
    }


def _write_inflight(destination: Path, work: protocol.WorkSpec) -> None:
    decision = SimpleNamespace(source="unified_exploration", reason="test")
    intent = runtime._intent_payload(
        _manifest(),
        work,
        step_index=0,
        selected=_Action("ACTION1", {}),
        decision=decision,
        registry_checksum=None,
    )
    protocol.write_json_once(
        runtime._work_path(destination, "intents", work, "0000.json"), intent
    )


def test_status_accepts_exactly_one_inflight_intent_under_live_lock(
    tmp_path: Path,
) -> None:
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.work_specs("discover-core")[0]
    _write_inflight(destination, work)
    destination.mkdir(parents=True, exist_ok=True)
    lock = {
        "format_version": "sage-t10.3.2-collector-lock-v1",
        "pid": 123,
        "process_start": time.time(),
        "nonce": "n",
        "phase": "discover-core",
        "heartbeat": time.time(),
    }
    (destination / runtime.LOCK_FILENAME).write_text(
        json.dumps(lock), encoding="utf-8"
    )
    payload = runtime.status(tmp_path, _manifest())
    assert payload["status"] == "RUNNING"
    assert payload["accounting"]["inflight_intents"] == 1
    assert payload["accounting"]["inflight_valid"] is True


def test_interrupted_branch_is_closed_without_physical_replay(tmp_path: Path) -> None:
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.work_specs("discover-core")[0]
    _write_inflight(destination, work)
    assert runtime.status(tmp_path, _manifest())["status"] == "INTERRUPTED"

    assert runtime._recover_orphans(destination, _manifest()) == 1
    receipt = runtime._read_signed(
        runtime._receipt_for_work(destination, work), "receipt_checksum"
    )
    assert receipt["status"] == "ABORTED_PROCESS_INTERRUPTION"
    assert receipt["issued_intents"] == 1
    assert receipt["sealed_events"] == 0
    assert receipt["unresolved_intents"] == 1
    assert receipt["physical_actions_replayed"] == 0
    accounting = runtime._journal_accounting(destination)
    assert accounting["equation_holds"] is True
    assert accounting["incomplete_work_ids"] == []


def test_collector_lock_rejects_concurrent_owner(tmp_path: Path) -> None:
    path = tmp_path / runtime.LOCK_FILENAME
    first = runtime._CollectorLock(path, "discover-core")
    second = runtime._CollectorLock(path, "discover-core")
    first.acquire()
    try:
        before = json.loads(path.read_text(encoding="utf-8"))["heartbeat"]
        first.heartbeat()
        after = json.loads(path.read_text(encoding="utf-8"))["heartbeat"]
        assert after >= before
        with pytest.raises(protocol.IntegrityError, match="collector is active"):
            second.acquire()
    finally:
        first.release()


def test_preflight_runs_all_three_scenarios_through_unified_controller(
    tmp_path: Path,
) -> None:
    payload = runtime.preflight(tmp_path, _manifest())
    assert payload["status"] == "PASS_T10_3_2_PREFLIGHT"
    assert payload["physical_actions"] == 0
    assert payload["checks"]["same_controller_closed_loop"] is True
    assert payload["checks"]["posterior_updated_each_action"] is True
    assert payload["checks"]["path_length_10"] is True
    assert payload["checks"]["mixed_beyond_16"] is True


def test_checkpoint_is_signed_and_write_once_artifacts_are_immutable(
    tmp_path: Path,
) -> None:
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.work_specs("discover-core")[0]
    checkpoint = runtime._write_checkpoint(
        destination,
        _manifest(),
        phase="discover-core",
        work=work,
        state="READY",
        registry=ProgressProgramRegistry(),
    )
    protocol.verify_signed(checkpoint, "checkpoint_checksum")
    assert checkpoint["current_work_id"] == work.work_id
    assert checkpoint["physical_actions_replayed"] == 0

    artifact = tmp_path / "write-once.json"
    protocol.write_json_once(artifact, {"value": 1})
    protocol.write_json_once(artifact, {"value": 1})
    with pytest.raises(protocol.IntegrityError, match="write-once"):
        protocol.write_json_once(artifact, {"value": 2})
