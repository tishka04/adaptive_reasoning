"""SAGE.T9.5 frozen paired active gate on source-validation games.

This runner reuses the exact T9.4 controller policy.  It is resumable at the
game/seed boundary and keeps the SAGE11 holdout and ar25 firewall closed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import bounded_active_v9_3 as r1
from . import bounded_active_v9_3b as r2
from . import calibration_gate_v8_6c as v86c
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v11 as validation_v8_7
from . import paired_active_gate_v9_4 as active

FORMAT_VERSION = "sage-t9.5-source-validation-active-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_5_source_validation_active_manifest.json"
)
DEFAULT_PARENT_REPORT = active.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_VALIDATION_REPORT = validation_v8_7.DEFAULT_OUTPUT_DIR / (
    "t8_7_source_validation_report.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "active_v9_5"
SEEDS = (1061, 1062, 1063, 1064, 1065)
RESETS = 14
ACTIONS_PER_RESET = 72
CONFIGURED_ACTIONS_PER_PAIR = RESETS * ACTIONS_PER_RESET


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "paired_active_gate_v9_4.py": v86c._file_sha256(
            directory / "paired_active_gate_v9_4.py"
        ),
        "source_validation_active_v9_5.py": v86c._file_sha256(
            directory / "source_validation_active_v9_5.py"
        ),
    }


def _checked_report(
    path: str | Path,
    *,
    expected_status: str,
) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError(f"report checksum mismatch: {path}")
    if report.get("status") != expected_status:
        raise ValueError(f"unexpected report status: {path}")
    return report


def freeze_manifest(
    *, output_path: str | Path = DEFAULT_MANIFEST_PATH
) -> dict[str, Any]:
    parent_manifest = active.load_manifest()
    parent_report = _checked_report(
        DEFAULT_PARENT_REPORT,
        expected_status="T9_4_PASSED",
    )
    validation_manifest = validation_v8_7.load_confirmation_manifest()
    validation_report = _checked_report(
        DEFAULT_VALIDATION_REPORT,
        expected_status="T8_7_PASSED",
    )
    if parent_report.get("source_validation_active_gate_authorized") is not True:
        raise ValueError("T9.4 did not authorize active source-validation")
    games = list(validation_manifest["source_validation_games"])
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_5_ACTIVE_SOURCE_VALIDATION",
        "frozen_at": "2026-08-06",
        "parent_t9_4_manifest_checksum": parent_manifest["manifest_checksum"],
        "parent_t9_4_report_checksum": parent_report["report_checksum"],
        "parent_t8_7_manifest_checksum": validation_manifest["manifest_checksum"],
        "parent_t8_7_report_checksum": validation_report["report_checksum"],
        "code_sha256": _code_hashes(),
        "policy": {
            "identity": "exact_t9_4_safe_active_no_repair",
            "controller_caps": dict(parent_manifest["controller_caps"]),
            "experimental_authority": dict(
                parent_manifest["experimental_authority"]
            ),
            "selected_terminal_policy": parent_manifest[
                "selected_terminal_policy"
            ],
            "winning_prefix_audit": dict(
                parent_manifest["winning_prefix_audit"]
            ),
            "retouched_after_t9_4": False,
        },
        "source_validation_games": games,
        "seeds": list(SEEDS),
        "resets_per_pair": RESETS,
        "actions_per_reset": ACTIONS_PER_RESET,
        "configured_actions_per_pair": CONFIGURED_ACTIONS_PER_PAIR,
        "configured_pairs": len(games) * len(SEEDS),
        "configured_active_action_budget": (
            len(games) * len(SEEDS) * CONFIGURED_ACTIONS_PER_PAIR
        ),
        "pairing": {
            "same_game_seed_reset_and_action_budget": True,
            "bootstrap_unit": "game_seed",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 9505,
        },
        "gate": {
            "minimum_pairs": 15,
            "minimum_configured_actions_per_pair": 1_000,
            "minimum_total_level_advantage": 1,
            "strictly_positive_level_rate_delta_ci_lower": True,
            "minimum_nonnegative_games": 2,
            "maximum_game_over_rate_delta": 0.0,
            "maximum_false_high_terminal_rate": 0.05,
            "minimum_progress_action_top8_rate": 1.0,
            "maximum_decision_p95_ms": 2_500.0,
            "maximum_observation_p95_ms": 3_000.0,
            "maximum_wall_seconds": 21_600.0,
            "maximum_illegal_actions": 0,
            "maximum_controller_errors": 0,
            "maximum_environment_errors": 0,
        },
        "firewall": {
            "source_train_modified": False,
            "source_validation_opened": True,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_active_authority": False,
        },
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.5 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.5 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_5_ACTIVE_SOURCE_VALIDATION":
        raise ValueError("T9.5 manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.5 code drifted")
    parent_manifest = active.load_manifest()
    parent_report = _checked_report(
        DEFAULT_PARENT_REPORT,
        expected_status="T9_4_PASSED",
    )
    validation_manifest = validation_v8_7.load_confirmation_manifest()
    validation_report = _checked_report(
        DEFAULT_VALIDATION_REPORT,
        expected_status="T8_7_PASSED",
    )
    bindings = {
        "parent_t9_4_manifest_checksum": parent_manifest["manifest_checksum"],
        "parent_t9_4_report_checksum": parent_report["report_checksum"],
        "parent_t8_7_manifest_checksum": validation_manifest["manifest_checksum"],
        "parent_t8_7_report_checksum": validation_report["report_checksum"],
    }
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"T9.5 binding drifted: {key}")
    if payload.get("source_validation_games") != list(
        validation_manifest["source_validation_games"]
    ):
        raise ValueError("T9.5 source-validation games drifted")
    if payload.get("seeds") != list(SEEDS):
        raise ValueError("T9.5 seeds drifted")
    if int(payload.get("resets_per_pair", 0)) != RESETS:
        raise ValueError("T9.5 reset count drifted")
    if int(payload.get("configured_actions_per_pair", 0)) < 1_000:
        raise ValueError("T9.5 action budget is below 1,000 per pair")
    policy = payload.get("policy", {})
    if policy.get("retouched_after_t9_4") is not False:
        raise ValueError("T9.5 policy was retouched")
    if policy.get("controller_caps") != parent_manifest.get("controller_caps"):
        raise ValueError("T9.5 controller caps differ from T9.4")
    if policy.get("experimental_authority") != parent_manifest.get(
        "experimental_authority"
    ):
        raise ValueError("T9.5 authority differs from T9.4")
    firewall = payload.get("firewall", {})
    if firewall.get("source_validation_opened") is not True or any(
        bool(firewall.get(key))
        for key in (
            "source_train_modified",
            "ar25_opened",
            "holdout_opened",
            "production_active_authority",
        )
    ):
        raise ValueError("T9.5 firewall is invalid")
    return payload


def _controller_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    policy = manifest["policy"]
    return {
        "controller_caps": dict(policy["controller_caps"]),
        "experimental_authority": dict(policy["experimental_authority"]),
        "selected_terminal_policy": policy["selected_terminal_policy"],
    }


def _factory(
    manifest: Mapping[str, Any],
    registry: dict[str, active.SafeActiveController],
):
    controller_manifest = _controller_manifest(manifest)

    def factory(game_id: str):  # type: ignore[no-untyped-def]
        controller = active.build_controller(controller_manifest)
        registry[str(game_id)] = controller
        return active.UnifiedCognitiveController(
            game_id,
            config=active.UnifiedCognitiveConfig(
                sage_t_authority_mode="active",
                sage_t_counterfactual_gate_passed=True,
                sage_t_active_gate_passed=True,
            ),
            sage_t_controller=controller,
        )

    return factory


def _progress_action_top8(
    arm: Mapping[str, Any],
    controller: active.SafeActiveController,
) -> tuple[int, int]:
    steps = r1._flat_steps(arm)
    decisions = [
        record
        for record in controller.compact_records
        if record.get("kind") == "decision"
    ]
    positives = 0
    top8 = 0
    for step, decision in zip(steps, decisions):
        progress = int(step.get("levels_after", 0)) > int(
            step.get("levels_before", 0)
        )
        if not progress:
            continue
        positives += 1
        action = r2._action_key(decision)
        rank = next(
            (
                index
                for index, item in enumerate(
                    decision.get("sequences", ()) or (),
                    start=1,
                )
                if item.get("sequence") and item["sequence"][0] == action
            ),
            None,
        )
        top8 += int(rank is not None and rank <= 8)
    return positives, top8


def _condition_path(
    output_dir: str | Path,
    *,
    game_id: str,
    seed: int,
) -> Path:
    safe_game = game_id.replace("/", "_").replace("\\", "_")
    return Path(output_dir) / "conditions" / f"{safe_game}__seed_{seed}.json"


def _load_condition(
    path: Path,
    *,
    manifest_checksum: str,
) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    checksum = str(row.pop("condition_checksum", ""))
    if checksum != v86c._checksum(row):
        raise ValueError(f"T9.5 condition checksum mismatch: {path}")
    if row.get("manifest_checksum") != manifest_checksum:
        raise ValueError(f"T9.5 condition manifest mismatch: {path}")
    row["condition_checksum"] = checksum
    return row


def _run_condition(
    *,
    manifest: Mapping[str, Any],
    game_id: str,
    seed: int,
    environments_dir: str | Path,
) -> dict[str, Any]:
    common = {
        "arm": "unified",
        "game_id": game_id,
        "seed": seed,
        "action_budget_per_reset": int(manifest["actions_per_reset"]),
        "resets": int(manifest["resets_per_pair"]),
        "env_dir": Path(environments_dir),
        "env_factory": None,
    }
    off = live_base._run_arm(controller_factory=r1._off_factory, **common)
    registry: dict[str, active.SafeActiveController] = {}
    started = time.perf_counter()
    selected = live_base._run_arm(
        controller_factory=_factory(manifest, registry),
        **common,
    )
    active_wall = time.perf_counter() - started
    controller = registry[game_id]
    off_metrics = r1._arm_metrics(off)
    active_metrics = r1._arm_metrics(selected)
    progress_positives, progress_top8 = _progress_action_top8(
        selected,
        controller,
    )
    terminal_rows = r2._actual_action_terminal_rows(controller)
    false_high = sum(
        not actual and risk >= 0.8 for risk, actual in terminal_rows
    )
    active_actions = max(1, int(active_metrics["actions"]))
    off_actions = max(1, int(off_metrics["actions"]))
    row: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "manifest_checksum": manifest["manifest_checksum"],
        "game": game_id,
        "seed": seed,
        "configured_actions": int(manifest["configured_actions_per_pair"]),
        "off": off_metrics,
        "active": active_metrics,
        "level_rate_delta": (
            float(active_metrics["levels_completed"]) / active_actions
            - float(off_metrics["levels_completed"]) / off_actions
        ),
        "game_over_rate_delta": (
            float(active_metrics["game_over_actions"]) / active_actions
            - float(off_metrics["game_over_actions"]) / off_actions
        ),
        "intervention": r1._intervention_metrics(selected, controller),
        "progress_action_top8": {
            "positive_actions": progress_positives,
            "top8": progress_top8,
        },
        "false_high_terminal": {
            "false_high": false_high,
            "observations": len(terminal_rows),
        },
        "decision_latencies_ms": list(controller.decision_latencies_ms),
        "observation_latencies_ms": list(controller.observation_latencies_ms),
        "controller_errors": len(tuple(selected.get("controller_errors", ()) or ())),
        "environment_errors": sum(
            str(attempt.get("failure_cause", "")).startswith("environment_")
            for attempt in selected.get("attempts", ()) or ()
        ),
        "illegal_actions": sum(
            "unavailable_decision" in str(error)
            for error in selected.get("controller_errors", ()) or ()
        ),
        "effective_mode": controller.summary()["effective_mode"],
        "active_safety": controller.summary()["active_safety"],
        "compact_runtime": controller.summary()["compact_runtime"],
        "posterior_performance": controller.posterior.performance_snapshot(),
        "active_wall_seconds": active_wall,
    }
    row["condition_checksum"] = v86c._checksum(row)
    return row


def _quantile(values: Sequence[float], probability: float) -> float:
    return r1._quantile(values, probability)


def build_report(
    conditions: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    level_interval = active._paired_interval(
        [float(row["level_rate_delta"]) for row in conditions],
        samples=int(manifest["pairing"]["bootstrap_samples"]),
        seed=int(manifest["pairing"]["bootstrap_seed"]),
    )
    game_over_interval = active._paired_interval(
        [float(row["game_over_rate_delta"]) for row in conditions],
        samples=int(manifest["pairing"]["bootstrap_samples"]),
        seed=int(manifest["pairing"]["bootstrap_seed"]) + 1,
    )
    active_actions = sum(int(row["active"]["actions"]) for row in conditions)
    active_levels = sum(
        int(row["active"]["levels_completed"]) for row in conditions
    )
    off_levels = sum(int(row["off"]["levels_completed"]) for row in conditions)
    by_game = {
        game: sum(
            int(row["active"]["levels_completed"])
            - int(row["off"]["levels_completed"])
            for row in conditions
            if row["game"] == game
        )
        for game in manifest["source_validation_games"]
    }
    decision_latencies = [
        float(value)
        for row in conditions
        for value in row["decision_latencies_ms"]
    ]
    observation_latencies = [
        float(value)
        for row in conditions
        for value in row["observation_latencies_ms"]
    ]
    progress_positive = sum(
        int(row["progress_action_top8"]["positive_actions"])
        for row in conditions
    )
    progress_top8 = sum(
        int(row["progress_action_top8"]["top8"]) for row in conditions
    )
    false_high = sum(
        int(row["false_high_terminal"]["false_high"]) for row in conditions
    )
    terminal_observations = sum(
        int(row["false_high_terminal"]["observations"])
        for row in conditions
    )
    gate = manifest["gate"]
    checks = {
        "all_pairs": len(conditions) >= int(gate["minimum_pairs"]),
        "configured_budget": all(
            int(row["configured_actions"])
            >= int(gate["minimum_configured_actions_per_pair"])
            for row in conditions
        ),
        "level_advantage": active_levels - off_levels
        >= int(gate["minimum_total_level_advantage"]),
        "paired_level_rate_ci_positive": float(level_interval["lower_95"]) > 0.0,
        "nonnegative_games": sum(value >= 0 for value in by_game.values())
        >= int(gate["minimum_nonnegative_games"]),
        "game_over_rate_nonincrease": float(game_over_interval["upper_95"])
        <= float(gate["maximum_game_over_rate_delta"]),
        "false_high_terminal": (
            false_high / max(1, terminal_observations)
            <= float(gate["maximum_false_high_terminal_rate"])
        ),
        "progress_action_top8": (
            progress_positive > 0
            and progress_top8 / progress_positive
            >= float(gate["minimum_progress_action_top8_rate"])
        ),
        "decision_p95": _quantile(decision_latencies, 0.95)
        <= float(gate["maximum_decision_p95_ms"]),
        "observation_p95": _quantile(observation_latencies, 0.95)
        <= float(gate["maximum_observation_p95_ms"]),
        "wall_time": wall_seconds <= float(gate["maximum_wall_seconds"]),
        "zero_illegal_actions": sum(int(row["illegal_actions"]) for row in conditions)
        <= int(gate["maximum_illegal_actions"]),
        "zero_controller_errors": sum(
            int(row["controller_errors"]) for row in conditions
        )
        <= int(gate["maximum_controller_errors"]),
        "zero_environment_errors": sum(
            int(row["environment_errors"]) for row in conditions
        )
        <= int(gate["maximum_environment_errors"]),
        "effective_active_mode": all(
            row["effective_mode"] == "active" for row in conditions
        ),
        "policy_frozen": manifest["policy"]["retouched_after_t9_4"] is False,
        "holdout_closed": manifest["firewall"]["holdout_opened"] is False,
        "ar25_closed": manifest["firewall"]["ar25_opened"] is False,
    }
    passed = all(checks.values())
    interventions = sum(
        int(row["intervention"]["interventions"]) for row in conditions
    )
    useful = sum(
        int(row["intervention"]["useful_interventions"]) for row in conditions
    )
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "T9_5_PASSED" if passed else "T9_5_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "runtime": dict(runtime),
        "checks": checks,
        "metrics": {
            "pairs": len(conditions),
            "configured_active_action_budget": manifest[
                "configured_active_action_budget"
            ],
            "actual_active_actions": active_actions,
            "active_levels_completed": active_levels,
            "baseline_levels_completed": off_levels,
            "total_level_advantage": active_levels - off_levels,
            "level_rate_delta_interval": level_interval,
            "game_over_rate_delta_interval": game_over_interval,
            "per_game_level_advantage": by_game,
            "progress_actions": progress_positive,
            "progress_action_top8_rate": (
                progress_top8 / progress_positive if progress_positive else None
            ),
            "false_high_terminal_rate": false_high
            / max(1, terminal_observations),
            "decision_p95_ms": _quantile(decision_latencies, 0.95),
            "observation_p95_ms": _quantile(observation_latencies, 0.95),
            "wall_seconds": wall_seconds,
            "interventions": interventions,
            "useful_interventions": useful,
            "wasted_interventions": interventions - useful,
            "useful_intervention_rate": useful / max(1, interventions),
        },
        "conditions": [
            {
                key: value
                for key, value in dict(row).items()
                if key
                not in {
                    "decision_latencies_ms",
                    "observation_latencies_ms",
                }
            }
            for row in conditions
        ],
        "active_authority_authorized": passed,
        "holdout_authorized": passed,
        "holdout_opened": False,
        "ar25_opened": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    return report


def run_gate(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    maximum_new_conditions: int | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    runtime = live_base.runtime_capabilities()
    destination = Path(output_dir)
    if not runtime.get("ready"):
        report = {
            "format_version": FORMAT_VERSION,
            "status": "BLOCKED_RUNTIME",
            "manifest_checksum": manifest["manifest_checksum"],
            "runtime": runtime,
            "active_authority_authorized": False,
            "holdout_authorized": False,
        }
        report["report_checksum"] = v86c._checksum(report)
        v86c._write_json(destination / "report.json", report)
        return report

    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for game_id in manifest["source_validation_games"]:
        for seed in manifest["seeds"]:
            path = _condition_path(output_dir, game_id=str(game_id), seed=int(seed))
            if path.exists():
                row = _load_condition(
                    path,
                    manifest_checksum=str(manifest["manifest_checksum"]),
                )
                completed[(str(game_id), int(seed))] = row
    new_conditions = 0
    for game_id in manifest["source_validation_games"]:
        for seed in manifest["seeds"]:
            key = (str(game_id), int(seed))
            if key in completed:
                continue
            if (
                maximum_new_conditions is not None
                and new_conditions >= max(0, int(maximum_new_conditions))
            ):
                break
            row = _run_condition(
                manifest=manifest,
                game_id=key[0],
                seed=key[1],
                environments_dir=environments_dir,
            )
            v86c._write_json(_condition_path(output_dir, game_id=key[0], seed=key[1]), row)
            completed[key] = row
            new_conditions += 1
        if (
            maximum_new_conditions is not None
            and new_conditions >= max(0, int(maximum_new_conditions))
        ):
            break

    ordered = [
        completed[(str(game_id), int(seed))]
        for game_id in manifest["source_validation_games"]
        for seed in manifest["seeds"]
        if (str(game_id), int(seed)) in completed
    ]
    if len(ordered) < int(manifest["configured_pairs"]):
        report = {
            "format_version": FORMAT_VERSION,
            "status": "T9_5_IN_PROGRESS",
            "manifest_checksum": manifest["manifest_checksum"],
            "runtime": runtime,
            "completed_pairs": len(ordered),
            "configured_pairs": manifest["configured_pairs"],
            "new_pairs_this_run": new_conditions,
            "active_authority_authorized": False,
            "holdout_authorized": False,
            "holdout_opened": False,
        }
        report["report_checksum"] = v86c._checksum(report)
        v86c._write_json(destination / "progress.json", report)
        return report

    report = build_report(
        ordered,
        manifest=manifest,
        runtime=runtime,
        wall_seconds=sum(float(row["active_wall_seconds"]) for row in ordered),
    )
    v86c._write_json(destination / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maximum-new-conditions", type=int)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        result = freeze_manifest(output_path=args.manifest)
    else:
        result = run_gate(
            manifest_path=args.manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
            maximum_new_conditions=args.maximum_new_conditions,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return (
        0
        if args.freeze
        or result.get("status") in {"T9_5_PASSED", "T9_5_IN_PROGRESS"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIONS_PER_RESET",
    "CONFIGURED_ACTIONS_PER_PAIR",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "RESETS",
    "SEEDS",
    "build_report",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_gate",
]
