from __future__ import annotations

from theory.sage_t import t10_3_6_protocol as protocol
from theory.sage_t.goal_directed_v10_3_6 import FunctionalGoalDirectedSageTController
from theory.sage_t.t10_3_6_runtime import (
    _basic_checks,
    _controller_pair,
    _synthetic_binding_cycle,
    preflight,
)
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry


def test_witness_controller_loads_descriptor_but_not_grounded_actions() -> None:
    work = next(
        item
        for item in protocol.work_specs("witness-core")
        if item.game_id == "lp85-305b61c3"
    )
    _, goal = _controller_pair(work, ProgressProgramRegistry(), registry_checksum=None)

    assert isinstance(goal, FunctionalGoalDirectedSageTController)
    assert goal.witness_schema == "repeat_target"
    assert goal.witness_horizon == 5
    assert goal.exploration_offset == work.reset_index


def test_preflight_has_no_latency_gate(tmp_path) -> None:
    manifest = {
        "manifest_checksum": "synthetic-manifest",
        "functional_contract": {"latency_is_telemetry_only": True},
    }
    result = preflight(tmp_path, manifest)

    assert result["status"] == "PASS_T10_3_6_PREFLIGHT"
    assert result["checks"]["latency_telemetry_only"] is True
    assert not any("p95" in key for key in result["checks"])


def test_basic_scientific_checks_ignore_latency_values() -> None:
    receipt = {
        "issued_intents": 1,
        "sealed_events": 1,
        "unresolved_intents": 0,
        "errors": [],
        "illegal_actions": 0,
        "physical_actions_replayed": 0,
        "lightweight_observations": 1,
        "decision_latencies_ms": [999_999.0],
        "controller_cycle_latencies_ms": [999_999.0],
    }
    checks = _basic_checks([receipt], 1)

    assert all(checks.values())
    assert checks["latency_not_a_gate"] is True


def test_synthetic_functional_loop_updates_same_posterior() -> None:
    result = _synthetic_binding_cycle(offset=0)

    assert result["posterior_events"] == 5
    assert result["option_successes"] == 1
