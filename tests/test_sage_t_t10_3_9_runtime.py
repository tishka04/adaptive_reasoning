from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_9_protocol as protocol
from theory.sage_t import t10_3_9_runtime as runtime
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry
from theory.sage_t.goal_directed_v10_3_9 import CausalSubgoalSageTController


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-manifest",
        "functional_contract": {"latency_is_telemetry_only": True},
        "firewall": {},
    }


def test_preflight_covers_total_observation_and_seed_diversity(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))

    result = runtime.preflight(tmp_path, _manifest())

    assert result["status"] == "PASS_T10_3_9_PREFLIGHT"
    assert all(result["checks"].values())
    assert result["physical_actions"] == 0


def test_controller_pair_uses_causal_controller_and_fresh_seed() -> None:
    work = protocol.work_specs("discover-sequence")[0]

    _, goal = runtime._controller_pair(
        work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )

    assert isinstance(goal, CausalSubgoalSageTController)
    assert goal.phase == "discovery"
    assert goal.summary()["exploration_seed_persisted"] is False


def test_reproduction_pair_prioritizes_mixed_registry_candidates() -> None:
    work = protocol.work_specs("reproduce-sequence")[0]

    _, goal = runtime._controller_pair(
        work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )

    assert isinstance(goal, CausalSubgoalSageTController)
    assert goal.reproduce_mixed_registry is True


def test_trajectory_diversity_requires_two_distinct_fingerprints_per_game() -> None:
    assert runtime._diversified_trajectories(
        {"a": ("one", "two"), "b": ("three", "four")}
    )
    assert not runtime._diversified_trajectories(
        {"a": ("one", "one"), "b": ("three", "four")}
    )


def test_terminal_report_fails_closed_before_discovery(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))
    manifest = {
        "manifest_checksum": "synthetic-manifest",
        "firewall": {},
    }

    report = runtime.terminal_report(tmp_path, manifest)

    assert report["verdict"] == "INVALID_PROVENANCE"
    assert report["physical_actions_replayed"] == 0
