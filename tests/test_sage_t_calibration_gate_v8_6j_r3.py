from __future__ import annotations

from pathlib import Path

from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6j_r3 import (
    MAXIMUM_REPAIR_CONTEXTS,
    _new_posterior,
    _runner,
    freeze_manifest,
    load_manifest,
)
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.posterior_v8 import T8_6G_POLICIES
from theory.sage_t.posterior_v11 import BudgetedRepairProgramPosterior
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor


def test_manifest_binds_long_horizon_failures_and_stays_shadow(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    assert loaded["maximum_repair_contexts_per_game"] == 16
    assert loaded["firewall"]["authority"] == "shadow"
    assert loaded["firewall"]["source_validation_opened"] is False


def test_runner_uses_budgeted_posterior_and_restores_components() -> None:
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


def test_new_posterior_has_frozen_global_budget() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    posterior = _new_posterior(
        T8_6G_POLICIES[
            "terminal_tempered_20_family_floor_0501_minimum_kl"
        ].with_repair_v2(),
        executor=StructuralRoleProgramExecutor(),
        manifest=manifest,
    )

    assert isinstance(posterior, BudgetedRepairProgramPosterior)
    assert posterior.maximum_repair_contexts == MAXIMUM_REPAIR_CONTEXTS == 16
