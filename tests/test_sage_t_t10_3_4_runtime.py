from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from theory.sage_t import t10_3_2_runtime as durable
from theory.sage_t import t10_3_4_protocol as protocol
from theory.sage_t import t10_3_4_runtime as runtime
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
            "t10_3_3_events_training_authorized": False,
            "t10_3_3_positive_witness_prior_authorized": False,
            "t10_3_3_physical_replay_authorized": False,
        },
        "superseded_t10_3_3": dict(protocol.SUPERSEDED_T10_3_3),
        "bounded_compute": {
            "stop_after_first_sealed_level": True,
            "same_profile_for_active_and_baseline": True,
            "transition_history_limit": 32,
            "operator_planning_enabled": False,
        },
        "gates": {"maximum_decision_p95_ms": 2500.0},
        "matrix": {"total_resets": 30, "total_maximum_actions": 6144},
    }


def test_preflight_covers_fast_path_history_and_early_stop(tmp_path: Path) -> None:
    payload = runtime.preflight(tmp_path, _manifest())
    assert payload["status"] == "PASS_T10_3_4_PREFLIGHT"
    assert payload["physical_actions"] == 0
    assert payload["checks"]["active_option_fast_path"] is True
    assert payload["checks"]["transition_history_bounded"] is True
    assert payload["checks"]["operator_planning_disabled"] is True
    assert payload["checks"]["early_success_policy"] is True
    assert payload["checks"]["stage_timing_schema_complete"] is True


def test_durability_contract_patch_is_scoped() -> None:
    original_protocol = durable.protocol
    original_controller = durable.GoalDirectedSageTController
    original_pair = durable._controller_pair
    with runtime._t10_3_4_contracts():
        assert durable.protocol is protocol
        assert (
            durable.GoalDirectedSageTController
            is runtime.BoundedGoalDirectedSageTController
        )
        assert durable._controller_pair is runtime._controller_pair
    assert durable.protocol is original_protocol
    assert durable.GoalDirectedSageTController is original_controller
    assert durable._controller_pair is original_pair


def test_offline_audit_attests_positive_parent_without_reuse(tmp_path: Path) -> None:
    payload = runtime.audit(tmp_path, _manifest())
    assert payload["status"] == "PASS_T10_3_4_OFFLINE_AUDIT"
    assert payload["parent_events_used_for_training"] == 0
    assert payload["parent_positive_witness_imported_as_prior"] is False


def test_runtime_stops_immediately_after_sealed_level_and_records_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.steps = 0

        def close(self) -> None:
            return None

    class FakeController:
        def on_reset(self) -> None:
            return None

        def select_action(self, **_kwargs):
            return SimpleNamespace(
                action_name="ACTION1",
                action_data={},
                source="bounded_legacy_fallback",
                reason="test",
            )

        def observe_transition(self, **_kwargs) -> None:
            return None

        def summary(self):
            return {
                "bounded_fast_path_decisions": 0,
                "transition_history_limit": 32,
                "maximum_retained_transitions": 2,
                "operator_induction_interval": 8,
                "operator_planning_enabled": False,
            }

    environment = FakeEnvironment()

    def frame(step: int):
        grid = np.zeros((4, 4), dtype=np.int16)
        grid[1, 1] = step % 2
        return SimpleNamespace(
            grid=grid,
            game_state="NOT_FINISHED",
            levels_completed=int(step >= 2),
            available_actions=("ACTION1",),
        )

    action = SimpleNamespace(name="ACTION1", action_args={})
    monkeypatch.setattr(runtime.live, "_make_real_env", lambda *_args: environment)
    monkeypatch.setattr(runtime.live, "_reset_env", lambda _env: frame(0))
    monkeypatch.setattr(runtime.live, "snapshot_frame", lambda value, **_kwargs: value)
    monkeypatch.setattr(runtime.live, "_valid_actions", lambda _env: (action,))
    monkeypatch.setattr(
        runtime,
        "_controller_pair",
        lambda *_args, **_kwargs: (FakeController(), None),
    )

    def step(_env, _action):
        environment.steps += 1
        return frame(environment.steps)

    monkeypatch.setattr(runtime.live, "_step_env_action", step)
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.WorkSpec(
        "discover-core",
        "lp85-305b61c3",
        3181,
        "unified_sage_t_off",
        0,
        10,
    )
    receipt = runtime._run_work(
        tmp_path,
        destination,
        _manifest(),
        work,
        ProgressProgramRegistry(),
        SimpleNamespace(heartbeat=lambda: None),
        registry_checksum=None,
    )
    assert receipt["complete"] is True
    assert receipt["issued_intents"] == 2
    assert receipt["sealed_events"] == 2
    assert receipt["level_delta"] == 1
    assert receipt["stop_reason"] == "LEVEL_PROGRESS_SEALED"
    assert receipt["planned_early_stop"] is True
    assert receipt["actions_saved_by_early_stop"] == 8
    assert len(receipt["stage_timings_ms"]) == 2
    assert set(receipt["stage_p95_ms"]) == set(runtime.STAGE_TIMING_KEYS)
    checkpoint = json.loads(
        (destination / durable.CHECKPOINT_FILENAME).read_text(encoding="utf-8")
    )
    protocol.verify_signed(checkpoint, "checkpoint_checksum")


def test_interrupted_intent_is_closed_without_physical_replay(tmp_path: Path) -> None:
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.work_specs("discover-core")[0]
    selected = SimpleNamespace(name="ACTION6", action_args={"x": 1, "y": 1})
    decision = SimpleNamespace(source="legacy_fallback", reason="test")
    with runtime._t10_3_4_contracts():
        intent = durable._intent_payload(
            _manifest(),
            work,
            step_index=0,
            selected=selected,
            decision=decision,
            registry_checksum=None,
        )
        protocol.write_json_once(
            durable._work_path(destination, "intents", work, "0000.json"),
            intent,
        )
        assert durable._recover_orphans(destination, _manifest()) == 1
        receipt = durable._read_signed(
            durable._receipt_for_work(destination, work),
            "receipt_checksum",
        )
    assert receipt["status"] == "ABORTED_PROCESS_INTERRUPTION"
    assert receipt["unresolved_intents"] == 1
    assert receipt["physical_actions_replayed"] == 0
