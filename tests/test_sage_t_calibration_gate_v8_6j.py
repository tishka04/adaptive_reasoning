from __future__ import annotations

from pathlib import Path

from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6j import (
    BENCHMARK_ACTIONS,
    _incremental_runner,
    _new_incremental_posterior,
    freeze_manifest,
    load_manifest,
)
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor


def test_manifest_binds_latency_only_failure_and_keeps_firewall(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    assert loaded["parent_failure"] == "latency_tail_ratio_only"
    assert loaded["benchmark_actions"] == BENCHMARK_ACTIONS
    assert loaded["gates"]["exact_long_final_posterior"] is True
    assert loaded["firewall"]["source_validation_opened"] is False
    assert loaded["firewall"]["authority"] == "shadow"


def test_incremental_runner_restores_frozen_components() -> None:
    original_posterior = v86._new_posterior
    original_generator = v86._programs_for
    original_executor = v86.ProgramExecutor

    with _incremental_runner():
        assert v86._new_posterior is _new_incremental_posterior
        assert v86._programs_for is programs_for_with_structural_goal_guard
        assert v86.ProgramExecutor is StructuralRoleProgramExecutor

    assert v86._new_posterior is original_posterior
    assert v86._programs_for is original_generator
    assert v86.ProgramExecutor is original_executor
