from __future__ import annotations

from pathlib import Path

from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6j_r2 import (
    _new_posterior,
    _runner,
    freeze_manifest,
    load_manifest,
)
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor


def test_manifest_binds_assimilation_failure_and_stays_shadow(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    assert loaded["repair_budget"] == "one_attempt_per_context_across_resets"
    assert loaded["firewall"]["authority"] == "shadow"
    assert loaded["firewall"]["source_validation_opened"] is False


def test_runner_restores_components() -> None:
    original_posterior = v86._new_posterior
    original_generator = v86._programs_for
    original_executor = v86.ProgramExecutor

    with _runner():
        assert v86._new_posterior is _new_posterior
        assert v86._programs_for is programs_for_with_structural_goal_guard
        assert v86.ProgramExecutor is StructuralRoleProgramExecutor

    assert v86._new_posterior is original_posterior
    assert v86._programs_for is original_generator
    assert v86.ProgramExecutor is original_executor
