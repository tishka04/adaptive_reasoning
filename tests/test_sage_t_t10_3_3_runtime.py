from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from theory.sage_t import t10_3_2_runtime as parent_runtime
from theory.sage_t import t10_3_3_protocol as protocol
from theory.sage_t import t10_3_3_runtime as runtime
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry


def _manifest() -> dict:
    return {
        "manifest_checksum": "m" * 64,
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_2_events_training_authorized": False,
            "t10_3_2_physical_replay_authorized": False,
        },
        "superseded_t10_3_2": dict(protocol.SUPERSEDED_T10_3_2),
        "binding_recovery": {"persistent_coordinates": False},
        "matrix": {"total_resets": 30, "total_maximum_actions": 6144},
    }


def test_preflight_recovers_ambiguous_target_without_persisting_anchor(
    tmp_path: Path,
) -> None:
    payload = runtime.preflight(tmp_path, _manifest())
    assert payload["status"] == "PASS_T10_3_3_PREFLIGHT"
    assert payload["physical_actions"] == 0
    assert payload["checks"]["ambiguous_target_progress"] is True
    assert payload["checks"]["same_target_reacquired"] is True
    assert payload["checks"]["structural_collision_observed"] is True
    assert payload["checks"]["persistent_coordinates_absent"] is True
    assert payload["checks"]["path_length_10"] is True
    assert payload["checks"]["mixed_beyond_16"] is True


def test_parent_runtime_contract_patch_is_scoped() -> None:
    original_protocol = parent_runtime.protocol
    original_controller = parent_runtime.GoalDirectedSageTController
    with runtime._t10_3_3_contracts():
        assert parent_runtime.protocol is protocol
        assert (
            parent_runtime.GoalDirectedSageTController
            is runtime.RelationalGoalDirectedSageTController
        )
    assert parent_runtime.protocol is original_protocol
    assert parent_runtime.GoalDirectedSageTController is original_controller


def test_interrupted_intent_is_closed_without_replay(tmp_path: Path) -> None:
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.work_specs("discover-core")[0]
    selected = SimpleNamespace(name="ACTION6", action_args={"x": 1, "y": 1})
    decision = SimpleNamespace(source="legacy_fallback", reason="test")
    with runtime._t10_3_3_contracts():
        intent = parent_runtime._intent_payload(
            _manifest(),
            work,
            step_index=0,
            selected=selected,
            decision=decision,
            registry_checksum=None,
        )
        protocol.write_json_once(
            parent_runtime._work_path(
                destination, "intents", work, "0000.json"
            ),
            intent,
        )
        assert parent_runtime._recover_orphans(destination, _manifest()) == 1
        receipt = parent_runtime._read_signed(
            parent_runtime._receipt_for_work(destination, work),
            "receipt_checksum",
        )
    assert receipt["status"] == "ABORTED_PROCESS_INTERRUPTION"
    assert receipt["issued_intents"] == 1
    assert receipt["unresolved_intents"] == 1
    assert receipt["physical_actions_replayed"] == 0


def test_offline_audit_attests_parent_exclusion(tmp_path: Path) -> None:
    payload = runtime.audit(tmp_path, _manifest())
    assert payload["status"] == "PASS_T10_3_3_OFFLINE_AUDIT"
    assert payload["parent_events_used_for_training"] == 0
    encoded = json.dumps(payload, sort_keys=True)
    assert "PASS_T10_3_3_OFFLINE_AUDIT" in encoded


def test_runtime_loop_seals_binding_telemetry_and_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.steps = 0

        def close(self) -> None:
            return None

    environment = FakeEnvironment()

    def frame(step: int):
        grid = np.zeros((4, 4), dtype=np.int16)
        grid[1, 1] = step % 2
        return SimpleNamespace(
            grid=grid,
            game_state="NOT_FINISHED",
            levels_completed=0,
            available_actions=("ACTION1",),
        )

    action = SimpleNamespace(name="ACTION1", action_args={})
    monkeypatch.setattr(runtime.live, "_make_real_env", lambda *_args: environment)
    monkeypatch.setattr(runtime.live, "_reset_env", lambda _env: frame(0))
    monkeypatch.setattr(runtime.live, "snapshot_frame", lambda value, **_kwargs: value)
    monkeypatch.setattr(runtime.live, "_valid_actions", lambda _env: (action,))

    def step(_env, _action):
        environment.steps += 1
        return frame(environment.steps)

    monkeypatch.setattr(runtime.live, "_step_env_action", step)
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.WorkSpec(
        "discover-core", "lp85-305b61c3", 3161, "goal_directed_sage_t", 0, 2
    )
    lock = SimpleNamespace(heartbeat=lambda: None)
    with runtime._t10_3_3_contracts():
        receipt = runtime._run_work(
            tmp_path,
            destination,
            _manifest(),
            work,
            ProgressProgramRegistry(),
            lock,
            registry_checksum=None,
        )
    assert receipt["complete"] is True
    assert receipt["issued_intents"] == 2
    assert receipt["sealed_events"] == 2
    assert receipt["ephemeral_action_data_persisted"] is False
    assert isinstance(receipt["binding_rejections"], dict)
    assert isinstance(receipt["binding_method_uses"], dict)
    checkpoint = json.loads(
        (destination / parent_runtime.CHECKPOINT_FILENAME).read_text(encoding="utf-8")
    )
    protocol.verify_signed(checkpoint, "checkpoint_checksum")
