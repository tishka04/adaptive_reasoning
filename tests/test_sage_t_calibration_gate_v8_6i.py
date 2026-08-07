from __future__ import annotations

import json
from pathlib import Path

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.calibration_gate_v8_6h import _new_minimum_kl_posterior
from theory.sage_t.calibration_gate_v8_6i import (
    DEFAULT_ACTION_SCHEDULES,
    SELECTED_POLICY,
    T8_6I_POLICIES,
    _structural_goal_runner,
    freeze_manifest,
    load_manifest,
)
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor


def test_manifest_binds_t8_6h_and_changes_only_structural_conditioning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    frozen = freeze_manifest(output_path=path)

    loaded = load_manifest(path)

    assert loaded == frozen
    invariants = loaded["frozen_invariants"]
    assert invariants["selected_posterior"] == SELECTED_POLICY
    assert invariants["posterior_implementation"] == "unchanged_from_t8_6g"
    assert invariants["goal_bridge"] == "unchanged_from_t8_6h"
    assert invariants["goal_guard_support"] == 0
    assert invariants["goal_guard_prior_delta"] == -0.10
    assert "absolute_coordinates" in invariants["forbidden_program_inputs"]
    assert loaded["firewall"]["source_validation_opened"] is False
    assert loaded["firewall"]["authority"] == "shadow"


def test_t8_6i_reuses_exact_parent_schedule() -> None:
    schedules = json.loads(
        Path(DEFAULT_ACTION_SCHEDULES).read_text(encoding="utf-8")
    )

    assert len(schedules) == 64
    assert all(1 <= len(actions) <= 5 for actions in schedules.values())
    assert "legacy" in T8_6I_POLICIES
    assert SELECTED_POLICY in T8_6I_POLICIES


def test_runner_patches_and_restores_all_three_frozen_components() -> None:
    original_posterior = v86._new_posterior
    original_generator = v86._programs_for
    original_executor = v86.ProgramExecutor

    with _structural_goal_runner():
        assert v86._new_posterior is _new_minimum_kl_posterior
        assert v86._programs_for is programs_for_with_structural_goal_guard
        assert v86.ProgramExecutor is StructuralRoleProgramExecutor

    assert v86._new_posterior is original_posterior
    assert v86._programs_for is original_generator
    assert v86.ProgramExecutor is original_executor


def test_all_goal_signals_survive_in_top8_after_full_history_replay() -> None:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    manifest = v86.load_t7_manifest(verify_code=True)
    policy = T8_6I_POLICIES[SELECTED_POLICY].with_repair_v2()

    with _structural_goal_runner():
        rows = v86._run_teacher_shocks(
            pairs,
            manifest=manifest,
            policies={policy.name: policy},
        )

    goals = [row for row in rows if row["positive_kind"] == "goal"]
    assert len(goals) == 3
    assert all(int(row["compatible_after_assembly"]) > 0 for row in goals)
    assert all(int(row["compatible_top8"]) > 0 for row in goals)
    assert all(row["diagnosis"] == "NONE" for row in goals)
