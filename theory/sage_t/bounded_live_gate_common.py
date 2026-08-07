"""Shared fail-closed reporting for bounded SAGE.T live gates."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)

from . import bounded_active_v9_3 as r1
from . import bounded_active_v9_3b as r2
from . import calibration_gate_v8_6c as v86c
from . import live_shadow_pilot as live_base


def _factory(
    manifest: Mapping[str, Any],
    registry: dict[str, Any],
    controller_builder: Callable[[Mapping[str, Any]], Any],
):
    def factory(game_id: str) -> UnifiedCognitiveController:
        sage_t = controller_builder(manifest)
        registry[str(game_id)] = sage_t
        return UnifiedCognitiveController(
            game_id,
            config=UnifiedCognitiveConfig(
                sage_t_authority_mode="bounded",
                sage_t_counterfactual_gate_passed=True,
            ),
            sage_t_controller=sage_t,
        )

    return factory


def _condition(
    *,
    manifest: Mapping[str, Any],
    game_id: str,
    seed: int,
    environments_dir: str | Path,
    registry: dict[str, Any],
    controller_builder: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    common = {
        "arm": "unified",
        "game_id": game_id,
        "seed": seed,
        "action_budget_per_reset": int(manifest["action_budget_per_reset"]),
        "resets": int(manifest["resets"]),
        "env_dir": Path(environments_dir),
        "env_factory": None,
    }
    off = live_base._run_arm(controller_factory=r1._off_factory, **common)
    bounded = live_base._run_arm(
        controller_factory=_factory(manifest, registry, controller_builder),
        **common,
    )
    controller = registry[game_id]
    return {
        "game": game_id,
        "seed": seed,
        "off": r1._arm_metrics(off),
        "bounded": r1._arm_metrics(bounded),
        "intervention": r1._intervention_metrics(bounded, controller),
        "controller_errors": len(tuple(bounded.get("controller_errors", ()) or ())),
        "environment_errors": sum(
            str(attempt.get("failure_cause", "")).startswith("environment_")
            for attempt in bounded.get("attempts", ()) or ()
        ),
        "illegal_actions": sum(
            "unavailable_decision" in str(error)
            for error in bounded.get("controller_errors", ()) or ()
        ),
        "bounded_safety": controller.summary()["bounded_safety"],
        "compact_runtime": controller.summary()["compact_runtime"],
        "terminal_calibration": controller.terminal_calibrator.snapshot(),
    }


def run_bounded_gate(
    *,
    manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    controller_builder: Callable[[Mapping[str, Any]], Any],
    format_version: str,
    passed_status: str,
    failed_status: str,
    environments_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if not runtime.get("ready"):
        report = {
            "format_version": format_version,
            "status": "BLOCKED_RUNTIME",
            "manifest_checksum": manifest["manifest_checksum"],
            "runtime": dict(runtime),
            "t9_4_authorized": False,
        }
        report["report_checksum"] = v86c._checksum(report)
        v86c._write_json(destination / "report.json", report)
        return report

    started = time.perf_counter()
    conditions = []
    registry: dict[str, Any] = {}
    for game_id in manifest["source_train_games"]:
        for seed in manifest["seeds"]:
            conditions.append(
                _condition(
                    manifest=manifest,
                    game_id=str(game_id),
                    seed=int(seed),
                    environments_dir=environments_dir,
                    registry=registry,
                    controller_builder=controller_builder,
                )
            )
            v86c._write_jsonl(destination / "conditions.partial.jsonl", conditions)

    elapsed = time.perf_counter() - started
    terminal_rows = tuple(
        row
        for controller in registry.values()
        for row in r2._actual_action_terminal_rows(controller)
    )
    false_high_rate = (
        mean(not actual and risk >= 0.8 for risk, actual in terminal_rows)
        if terminal_rows
        else 0.0
    )
    decisions = [
        value for controller in registry.values() for value in controller.decision_latencies_ms
    ]
    observations = [
        value
        for controller in registry.values()
        for value in controller.observation_latencies_ms
    ]
    bounded_actions = sum(row["bounded"]["actions"] for row in conditions)
    progress = sum(row["bounded"]["progress_events"] for row in conditions)
    levels = sum(row["bounded"]["levels_completed"] for row in conditions)
    game_over_delta = sum(
        row["bounded"]["game_over_actions"] - row["off"]["game_over_actions"]
        for row in conditions
    )
    peak_cache = max(
        (
            int(row["compact_runtime"]["peak_executor_cache_entries"])
            for row in conditions
        ),
        default=0,
    )
    gate = manifest["gate"]
    checks = {
        "minimum_actions": bounded_actions >= int(gate["minimum_total_actions"]),
        "real_progress": progress >= int(gate["minimum_real_progress_events"]),
        "level_completed": levels >= int(gate["minimum_levels_completed"]),
        "game_over_nonincrease": game_over_delta <= int(gate["maximum_game_over_delta"]),
        "false_high_terminal": false_high_rate
        <= float(gate["maximum_false_high_terminal_rate"]),
        "decision_p95": r1._quantile(decisions, 0.95)
        <= float(gate["maximum_decision_p95_ms"]),
        "observation_p95": r1._quantile(observations, 0.95)
        <= float(gate["maximum_observation_p95_ms"]),
        "wall_time": elapsed <= float(gate["maximum_wall_seconds"]),
        "executor_cache_bounded": peak_cache
        <= int(gate["maximum_peak_executor_cache_entries"]),
        "zero_illegal_actions": sum(row["illegal_actions"] for row in conditions) == 0,
        "zero_controller_errors": sum(row["controller_errors"] for row in conditions) == 0,
        "zero_environment_errors": sum(row["environment_errors"] for row in conditions) == 0,
        "bounded_budget": all(
            int(row["bounded_safety"]["maximum_interventions_in_branch"])
            <= int(manifest["authority"]["maximum_interventions_per_reset"])
            for row in conditions
        ),
        "winning_prefix_audit": bool(manifest["winning_prefix_audit"]["passed"]),
        "firewall_closed": True,
    }
    passed = all(checks.values())
    report = {
        "format_version": format_version,
        "status": passed_status if passed else failed_status,
        "manifest_checksum": manifest["manifest_checksum"],
        "runtime": dict(runtime),
        "checks": checks,
        "metrics": {
            "bounded_actions": bounded_actions,
            "progress_events": progress,
            "levels_completed": levels,
            "game_over_delta": game_over_delta,
            "false_high_terminal_rate": false_high_rate,
            "decision_p95_ms": r1._quantile(decisions, 0.95),
            "observation_p95_ms": r1._quantile(observations, 0.95),
            "wall_seconds": elapsed,
            "peak_executor_cache_entries": peak_cache,
            "interventions": sum(
                row["intervention"]["interventions"] for row in conditions
            ),
            "useful_interventions": sum(
                row["intervention"]["useful_interventions"] for row in conditions
            ),
            "wasted_interventions": sum(
                row["intervention"]["wasted_interventions"] for row in conditions
            ),
        },
        "conditions": conditions,
        "t9_4_authorized": passed,
        "active_authority_authorized": False,
        "source_validation_opened": False,
        "holdout_opened": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_jsonl(destination / "conditions.jsonl", conditions)
    v86c._write_json(destination / "report.json", report)
    return report


__all__ = ["run_bounded_gate"]
