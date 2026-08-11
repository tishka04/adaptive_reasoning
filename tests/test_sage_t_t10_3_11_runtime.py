from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_11_protocol as protocol
from theory.sage_t import t10_3_11_runtime as runtime
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry
from theory.sage_t.goal_directed_v10_3_11 import (
    GoalConditionedSageTController,
    GoalConditionedUnifiedCognitiveController,
)


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-manifest",
        "functional_contract": {"level_increment_is_only_success_credit": True},
        "firewall": {},
    }


def test_preflight_is_delayed_goal_conditioned_and_bounded(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))

    result = runtime.preflight(tmp_path, _manifest())

    assert result["status"] == "PASS_T10_3_11_PREFLIGHT"
    assert all(result["checks"].values())
    assert result["physical_actions"] == 0
    assert all(row["actions"] >= 18 for row in result["scenarios"].values())


def test_controller_pair_uses_conditioning_and_true_ablation() -> None:
    active_work = next(
        row
        for row in protocol.work_specs("discover-sequence")
        if row.arm == "goal_conditioned_sage_t"
    )
    active, goal = runtime._controller_pair(
        active_work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )
    assert isinstance(active, GoalConditionedUnifiedCognitiveController)
    assert isinstance(goal, GoalConditionedSageTController)
    assert goal.goal_conditioning_enabled is True

    ablation_work = next(
        row
        for row in protocol.work_specs("discover-sequence")
        if row.arm == "goal_ablation_sage_t"
    )
    ablation, ablation_goal = runtime._controller_pair(
        ablation_work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )
    assert isinstance(ablation, GoalConditionedUnifiedCognitiveController)
    assert isinstance(ablation_goal, GoalConditionedSageTController)
    assert ablation_goal.goal_conditioning_enabled is False


def test_terminal_report_fails_closed_before_offline_gates(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))
    manifest = {"manifest_checksum": "synthetic-manifest", "firewall": {}}

    report = runtime.terminal_report(tmp_path, manifest)

    assert report["verdict"] == "INVALID_PROVENANCE"
    assert report["physical_actions_replayed"] == 0

