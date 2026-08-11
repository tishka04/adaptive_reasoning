from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_10_protocol as protocol
from theory.sage_t import t10_3_10_runtime as runtime
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry
from theory.sage_t.goal_directed_v10_3_10 import (
    DirectionalProgressSageTController,
    DirectionalProgressUnifiedCognitiveController,
)


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-manifest",
        "functional_contract": {"latency_is_telemetry_only": True},
        "firewall": {},
    }


def test_preflight_wins_trap_and_bounds_posterior(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))

    result = runtime.preflight(tmp_path, _manifest())

    assert result["status"] == "PASS_T10_3_10_PREFLIGHT"
    assert all(result["checks"].values())
    assert result["physical_actions"] == 0


def test_controller_pair_is_directional_and_symmetric() -> None:
    active_work = protocol.work_specs("discover-sequence")[0]
    active, goal = runtime._controller_pair(
        active_work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )
    assert isinstance(active, DirectionalProgressUnifiedCognitiveController)
    assert isinstance(goal, DirectionalProgressSageTController)
    assert goal.phase == "discovery"

    off_work = next(
        row for row in protocol.work_specs("confirm") if row.arm == "unified_sage_t_off"
    )
    off, off_goal = runtime._controller_pair(
        off_work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )
    assert isinstance(off, DirectionalProgressUnifiedCognitiveController)
    assert off_goal is None


def test_trajectory_diversity_requires_two_fingerprints_per_game() -> None:
    assert runtime._diversified_trajectories(
        {"a": ("one", "two"), "b": ("three", "four")}
    )
    assert not runtime._diversified_trajectories(
        {"a": ("one", "one"), "b": ("three", "four")}
    )


def test_terminal_report_fails_closed_before_offline_gates(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))
    manifest = {"manifest_checksum": "synthetic-manifest", "firewall": {}}

    report = runtime.terminal_report(tmp_path, manifest)

    assert report["verdict"] == "INVALID_PROVENANCE"
    assert report["physical_actions_replayed"] == 0

