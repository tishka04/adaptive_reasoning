"""SAGE.10e authority-repair regression harness tests."""

from __future__ import annotations

from theory import benchmark_score_runner
from theory.ft09_regression_benchmark import (
    run_ft09_authority_repair_benchmark,
)


def test_authority_repair_harness_enforces_all_four_gates():
    def fake_arm_runner(layer_name, _controller_factory):
        return {
            "max_level_reached": 6,
            "levels_completed_delta": 23,
            "wins": 1,
            "actions_executed": 1146,
            "protected_route_preemptions": 0,
            "frontier_experiments": int(layer_name != "post_9u_baseline"),
            "frontier_per_level_rearms": int(layer_name == "full"),
            "terminal_multiform_selections": int(
                layer_name in {"plus_9w", "plus_10a", "full"}
            ),
            "terminal_multiform_demotions": int(layer_name == "full"),
            "controller_errors": [],
        }

    payload = run_ft09_authority_repair_benchmark(
        arm_runner=fake_arm_runner,
        write_path=None,
    )

    assert payload["all_gates_passed"] is True
    assert all(payload["gates"].values())
    assert all(payload["layer_monotonicity"].values())
    assert payload["layers"]["full"][
        "protected_route_preemptions"
    ] == 0
    assert payload["procedural_liveness"]["multiform"][
        "selections"
    ] == 3
    assert payload["procedural_liveness"]["multiform"][
        "demotions"
    ] == 1
    assert payload["procedural_liveness"]["multiform"][
        "reactivations"
    ] == 1


def test_performance_protocol_cli_defaults_to_fourteen_resets(monkeypatch):
    captured = {}

    def fake_run_benchmark_score(**kwargs):
        captured.update(kwargs)
        return {
            "total_levels_completed": 0,
            "total_wins": 0,
            "maximum_level_reached": 0,
            "normalized_score_proxy": 0.0,
            "wall_clock_seconds": 0.0,
        }

    monkeypatch.setattr(
        benchmark_score_runner,
        "run_benchmark_score",
        fake_run_benchmark_score,
    )

    result = benchmark_score_runner.main(
        [
            "--games",
            "ft09",
            "--seeds",
            "0",
            "--budgets",
            "1",
        ]
    )

    assert result == 0
    assert captured["resets"] == 14
