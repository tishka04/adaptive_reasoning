from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from theory.sage_t import t10_3_2_runtime as durable
from theory.sage_t import t10_3_5_protocol as protocol
from theory.sage_t import t10_3_5_runtime as runtime
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
            "t10_3_4_events_training_authorized": False,
            "t10_3_4_positive_witness_prior_authorized": False,
            "t10_3_4_physical_replay_authorized": False,
        },
        "superseded_t10_3_4": dict(protocol.SUPERSEDED_T10_3_4),
        "scheduled_control": {
            "shared_legacy_proposal_each_action": True,
            "full_unified_decision_path_enabled": False,
            "full_unified_observation_path_enabled": False,
            "lightweight_effect_model_each_transition": True,
            "sage_t_posterior_each_transition": True,
            "active_option_fast_path": True,
            "productive_option_extension": True,
            "maximum_option_horizon": 32,
            "same_schedule_for_active_and_baseline": True,
        },
        "gates": {
            "maximum_decision_p95_ms": 2500.0,
            "maximum_controller_cycle_p95_ms": 2500.0,
            "structural_collision_policy": "fail_closed_if_observed_not_required_to_occur",
        },
        "matrix": {"total_resets": 30, "total_maximum_actions": 6144},
    }


def test_preflight_covers_schedule_extension_and_cycle_latency(tmp_path: Path) -> None:
    payload = runtime.preflight(tmp_path, _manifest())
    assert payload["status"] == "PASS_T10_3_5_PREFLIGHT"
    assert payload["physical_actions"] == 0
    assert payload["checks"]["full_unified_paths_never_entered"] is True
    assert payload["checks"]["productive_option_extended"] is True
    assert payload["checks"]["posterior_updated_each_action"] is True
    assert payload["checks"]["controller_cycle_p95"] is True
    assert payload["checks"]["transition_history_bounded"] is True


def test_durability_contract_patch_is_scoped() -> None:
    original_protocol = durable.protocol
    original_controller = durable.GoalDirectedSageTController
    with runtime._t10_3_5_contracts():
        assert durable.protocol is protocol
        assert durable.GoalDirectedSageTController is runtime.ScheduledGoalDirectedSageTController
        assert durable._controller_pair is runtime._controller_pair
        assert durable._run_work is runtime._run_work
    assert durable.protocol is original_protocol
    assert durable.GoalDirectedSageTController is original_controller


def test_offline_audit_attests_negative_parent_without_reuse(tmp_path: Path) -> None:
    payload = runtime.audit(tmp_path, _manifest())
    assert payload["status"] == "PASS_T10_3_5_OFFLINE_AUDIT"
    assert payload["parent_events_used_for_training"] == 0
    assert payload["parent_positive_witness_imported_as_prior"] is False


def test_runtime_stops_after_level_and_records_controller_cycle(
    tmp_path: Path, monkeypatch
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
                action_name="ACTION1", action_data={},
                source="scheduled_legacy_proposal", reason="test",
            )

        def observe_transition(self, **_kwargs) -> None:
            return None

        def summary(self):
            return {
                "bounded_fast_path_decisions": 0,
                "scheduled_sage_decisions": 0,
                "scheduled_legacy_decisions": 2,
                "lightweight_observations": 2,
                "full_unified_decisions": 0,
                "full_unified_observations": 0,
                "maximum_retained_transitions": 2,
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
        runtime, "_controller_pair", lambda *_args, **_kwargs: (FakeController(), None)
    )

    def step(_env, _action):
        environment.steps += 1
        return frame(environment.steps)

    monkeypatch.setattr(runtime.live, "_step_env_action", step)
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.WorkSpec(
        "discover-core", "lp85-305b61c3", 3201,
        "unified_sage_t_off", 0, 10,
    )
    receipt = runtime._run_work(
        tmp_path, destination, _manifest(), work,
        ProgressProgramRegistry(), SimpleNamespace(heartbeat=lambda: None),
        registry_checksum=None,
    )
    assert receipt["issued_intents"] == 2
    assert receipt["sealed_events"] == 2
    assert receipt["level_delta"] == 1
    assert receipt["stop_reason"] == "LEVEL_PROGRESS_SEALED"
    assert receipt["actions_saved_by_early_stop"] == 8
    assert len(receipt["controller_cycle_latencies_ms"]) == 2
    assert set(receipt["stage_p95_ms"]) == set(runtime.STAGE_TIMING_KEYS)
    checkpoint = json.loads(
        (destination / durable.CHECKPOINT_FILENAME).read_text(encoding="utf-8")
    )
    protocol.verify_signed(checkpoint, "checkpoint_checksum")


def test_interrupted_intent_is_closed_without_physical_replay(tmp_path: Path) -> None:
    destination = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    work = protocol.work_specs("discover-core")[0]
    selected = SimpleNamespace(name="ACTION6", action_args={"x": 1, "y": 1})
    decision = SimpleNamespace(source="scheduled_legacy_proposal", reason="test")
    with runtime._t10_3_5_contracts():
        intent = durable._intent_payload(
            _manifest(), work, step_index=0, selected=selected,
            decision=decision, registry_checksum=None,
        )
        protocol.write_json_once(
            durable._work_path(destination, "intents", work, "0000.json"), intent
        )
        assert durable._recover_orphans(destination, _manifest()) == 1
        receipt = durable._read_signed(
            durable._receipt_for_work(destination, work), "receipt_checksum"
        )
    assert receipt["status"] == "ABORTED_PROCESS_INTERRUPTION"
    assert receipt["unresolved_intents"] == 1
    assert receipt["physical_actions_replayed"] == 0
