"""T8.7 frozen paired live shadow validation with the unchanged T8.6j model."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from . import calibration_gate_v8_6c as v86c
from . import live_shadow_pilot as live_base
from . import live_shadow_pilot_v5 as t8_5
from . import live_shadow_pilot_v6 as t8_6_live
from . import live_shadow_pilot_v9 as live_r2
from . import live_shadow_pilot_v10 as live_r3

FORMAT_VERSION = "sage-t8.7-source-validation-live-v1"
DEFAULT_ACTION_MANIFEST = Path(__file__).with_name(
    "sage_t8_7_source_validation_action_manifest.json"
)
DEFAULT_CONFIRMATION_MANIFEST = Path(__file__).with_name(
    "sage_t8_7_source_validation_manifest.json"
)
DEFAULT_PARENT_REPORT = (
    live_r3.DEFAULT_OUTPUT_DIR / "t8_6j_r3_extended_live_report.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "source_validation_v8_7"
VALIDATION_GAMES = (
    "re86-4e57566e",
    "ls20-9607627b",
    "sc25-f9b21a2f",
)
VALIDATION_SEED = 1061
RESETS = 2
ACTIONS_PER_RESET = 25
MAXIMUM_ACTIONS = len(VALIDATION_GAMES) * RESETS * ACTIONS_PER_RESET


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "live_shadow_pilot_v11.py": v86c._file_sha256(
            directory / "live_shadow_pilot_v11.py"
        )
    }


def _load_parent_report(path: str | Path = DEFAULT_PARENT_REPORT) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.6j-r3 extended-live report checksum mismatch")
    if report.get("status") != "READY_TO_PREPARE_T8_7_SOURCE_VALIDATION":
        raise ValueError("T8.6j did not authorize T8.7")
    if report.get("source_validation_authorized") is not True:
        raise ValueError("T8.7 source-validation is not authorized")
    if report.get("bounded_authority_authorized") is not False:
        raise ValueError("bounded authority opened before T8.7")
    return report


def freeze_action_manifest(
    *,
    output_path: str | Path = DEFAULT_ACTION_MANIFEST,
    parent_path: str | Path = live_r3.DEFAULT_ACTION_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
) -> dict[str, Any]:
    parent = t8_5.load_frozen_manifest(parent_path)
    report = _load_parent_report(parent_report_path)
    t7 = live_base.load_t7_1_manifest()
    payload = json.loads(json.dumps(parent))
    payload.pop("manifest_checksum", None)
    payload["status"] = "FROZEN_BEFORE_SOURCE_VALIDATION_LIVE"
    payload["frozen_at"] = "2026-08-06"
    payload["split"] = "source_validation"
    payload["source_train_games"] = list(VALIDATION_GAMES)
    payload["forbidden_games"] = [
        *list(t7["source_train_games"]),
        "ar25-e3c63847",
    ]
    payload["action_budget_per_reset"] = ACTIONS_PER_RESET
    payload["resets"] = RESETS
    payload["seeds"] = [VALIDATION_SEED]
    payload["parent_t8_6j_report_checksum"] = report["report_checksum"]
    payload["challenger"] = {
        "label": "T8_7_UNCHANGED_T8_6J_SOURCE_VALIDATION",
        "purpose": "validate calibration transfer without model changes",
        "model_changes": [],
        "can_authorize_control": False,
    }
    payload["gate"] = {
        **dict(parent["gate"]),
        "minimum_actions": 60,
        "minimum_finite_surprise_samples": 60,
        "minimum_prediction_coverage": 1.0,
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_action_manifest(
    path: str | Path = DEFAULT_ACTION_MANIFEST,
    *,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.7 action manifest checksum mismatch")
    if payload.get("status") != "FROZEN_BEFORE_SOURCE_VALIDATION_LIVE":
        raise ValueError("T8.7 action manifest is not frozen")
    if payload.get("split") != "source_validation":
        raise ValueError("T8.7 is not source-validation")
    configured = tuple(str(game) for game in payload["source_train_games"])
    if configured != VALIDATION_GAMES:
        raise ValueError("T8.7 validation games drifted")
    t7 = live_base.load_t7_1_manifest()
    source_train = {str(game) for game in t7["source_train_games"]}
    if source_train & {game.split("-", 1)[0] for game in configured}:
        raise ValueError("T8.7 opened a source-train game")
    parent = _load_parent_report(parent_report_path)
    if payload.get("parent_t8_6j_report_checksum") != parent.get(
        "report_checksum"
    ):
        raise ValueError("T8.7 parent report drifted")
    if payload.get("authority", {}).get("mode") != "shadow":
        raise ValueError("T8.7 must remain shadow-only")
    if payload.get("challenger", {}).get("model_changes") != []:
        raise ValueError("T8.7 contains a model change")
    if payload.get("inference_changes") != [
        "add the materializable baseline action to the local candidate set",
        "reserve one counterfactual sequence for that exact action",
    ]:
        raise ValueError("T8.7 inference protocol drifted")
    directory = Path(__file__).resolve().parent
    expected = payload.get("code_sha256", {})
    for name in ("live_shadow_pilot.py", "live_shadow_pilot_v5.py"):
        if expected.get(name) != v86c._file_sha256(directory / name):
            raise ValueError(f"T8.7 dependency drifted: {name}")
    return payload


def freeze_confirmation_manifest(
    *,
    output_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    action_manifest_path: str | Path = DEFAULT_ACTION_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
) -> dict[str, Any]:
    action = load_action_manifest(
        action_manifest_path,
        parent_report_path=parent_report_path,
    )
    parent = _load_parent_report(parent_report_path)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T8_7_SOURCE_VALIDATION",
        "frozen_at": "2026-08-06",
        "parent_t8_6j_report_checksum": parent["report_checksum"],
        "action_manifest_checksum": action["manifest_checksum"],
        "code_sha256": _code_hashes(),
        "model": "unchanged_t8_6j_r3",
        "source_validation_games": list(VALIDATION_GAMES),
        "seed": VALIDATION_SEED,
        "resets": RESETS,
        "actions_per_reset": ACTIONS_PER_RESET,
        "maximum_actions": MAXIMUM_ACTIONS,
        "authority": "shadow",
        "gate": {
            "minimum_actions": 60,
            "minimum_actions_per_game": 15,
            "minimum_prediction_coverage": 1.0,
            "maximum_false_high_terminal_rate": 0.05,
            "high_terminal_threshold": 0.8,
            "maximum_decision_p95_ms": 2500.0,
            "maximum_observation_p95_ms": 3000.0,
            "maximum_wall_seconds": 300.0,
            "maximum_repair_contexts_per_game": 16,
        },
        "firewall": {
            "source_validation_opened": True,
            "source_train_modified": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "bounded_authority": False,
            "active_authority": False,
        },
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_confirmation_manifest(
    path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    *,
    action_manifest_path: str | Path = DEFAULT_ACTION_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T8.7 confirmation manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T8.7 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T8_7_SOURCE_VALIDATION":
        raise ValueError("T8.7 confirmation manifest is not frozen")
    action = load_action_manifest(
        action_manifest_path,
        parent_report_path=parent_report_path,
    )
    parent = _load_parent_report(parent_report_path)
    if payload.get("action_manifest_checksum") != action.get(
        "manifest_checksum"
    ):
        raise ValueError("T8.7 action manifest drifted")
    if payload.get("parent_t8_6j_report_checksum") != parent.get(
        "report_checksum"
    ):
        raise ValueError("T8.7 parent report drifted")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T8.7 code drifted")
    firewall = payload.get("firewall", {})
    if payload.get("authority") != "shadow" or any(
        bool(firewall.get(key))
        for key in ("ar25_opened", "holdout_opened", "bounded_authority", "active_authority")
    ):
        raise ValueError("T8.7 firewall is open")
    return payload


def _validation_checks(
    policy_rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
    *,
    selected_condition: str,
    base_report: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    registry: Mapping[str, live_r3.ExtendedLiveController],
) -> dict[str, Any]:
    selected = [row for row in policy_rows if row["condition"] == selected_condition]
    gate = confirmation["gate"]
    by_game: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_game.setdefault(str(row["game"]), []).append(row)
    threshold = float(gate["high_terminal_threshold"])
    false_high_rates = {}
    for game, rows in by_game.items():
        false_high_rates[game] = mean(
            float(
                row.get("terminal_probability") is not None
                and float(row["terminal_probability"]) >= threshold
                and not bool(row["actual_terminal"])
            )
            for row in rows
        )
    decisions = [
        float(row["decision_latency_ms"])
        for row in selected
        if row.get("decision_latency_ms") is not None
    ]
    observations = [
        float(row["observation_latency_ms"])
        for row in selected
        if row.get("observation_latency_ms") is not None
    ]
    coverage = (
        mean(float(bool(row["prediction_available"])) for row in selected)
        if selected
        else 0.0
    )
    repairs = live_r2._repair_metrics(base_rows, registry)  # type: ignore[arg-type]
    safety = dict(base_report.get("safety", {}))
    checks = {
        "minimum_actions": len(selected) >= int(gate["minimum_actions"]),
        "minimum_actions_per_game": len(by_game) == len(VALIDATION_GAMES)
        and min((len(rows) for rows in by_game.values()), default=0)
        >= int(gate["minimum_actions_per_game"]),
        "prediction_coverage": coverage >= float(gate["minimum_prediction_coverage"]),
        "false_high_terminal_rate": bool(false_high_rates)
        and max(false_high_rates.values())
        <= float(gate["maximum_false_high_terminal_rate"]),
        "decision_p95": bool(decisions)
        and t8_6_live._quantile(decisions, 0.95)
        <= float(gate["maximum_decision_p95_ms"]),
        "observation_p95": bool(observations)
        and t8_6_live._quantile(observations, 0.95)
        <= float(gate["maximum_observation_p95_ms"]),
        "wall_time": float(base_report.get("wall_clock_seconds", math.inf))
        <= float(gate["maximum_wall_seconds"]),
        "global_repair_budget": all(
            int(item.get("unique_repair_contexts", 0))
            <= int(gate["maximum_repair_contexts_per_game"])
            for item in repairs["performance"].values()
        ),
        "same_actions": bool(safety.get("actions_match")),
        "same_resets": bool(safety.get("same_reset_states")),
        "zero_interventions": int(safety.get("interventions", 0)) == 0,
        "zero_illegal_actions": int(safety.get("illegal_actions", 0)) == 0,
        "zero_controller_errors": int(safety.get("controller_errors", 0)) == 0,
        "zero_environment_errors": int(safety.get("environment_errors", 0)) == 0,
        "no_semantic_collapse": not any(bool(row["semantic_collapse"]) for row in selected),
        "holdout_closed": True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "actions": len(selected),
            "actions_by_game": {game: len(rows) for game, rows in by_game.items()},
            "prediction_coverage": coverage,
            "false_high_terminal_rate_by_game": false_high_rates,
            "decision_p95_ms": t8_6_live._quantile(decisions, 0.95),
            "observation_p95_ms": t8_6_live._quantile(observations, 0.95),
            "wall_clock_seconds": base_report.get("wall_clock_seconds"),
            "repairs": repairs,
            "semantic_collapses": sum(bool(row["semantic_collapse"]) for row in selected),
        },
    }


def run_source_validation(
    *,
    confirmation_manifest_path: str | Path = DEFAULT_CONFIRMATION_MANIFEST,
    action_manifest_path: str | Path = DEFAULT_ACTION_MANIFEST,
    parent_report_path: str | Path = DEFAULT_PARENT_REPORT,
    environments_dir: str | Path = "environment_files",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    confirmation = load_confirmation_manifest(
        confirmation_manifest_path,
        action_manifest_path=action_manifest_path,
        parent_report_path=parent_report_path,
    )
    registry: dict[str, live_r3.ExtendedLiveController] = {}
    previous_factory = t8_5._controller_factory
    previous_t8_5_loader = t8_5.load_frozen_manifest
    previous_base_loader = live_base.load_frozen_manifest
    validation_loader = lambda path=action_manifest_path: load_action_manifest(
        path,
        parent_report_path=parent_report_path,
    )
    t8_5._controller_factory = live_r3._factory_builder(registry=registry)
    t8_5.load_frozen_manifest = validation_loader
    live_base.load_frozen_manifest = validation_loader
    started = time.perf_counter()
    try:
        base_report = t8_5.run_live_shadow_pilot(
            manifest_path=action_manifest_path,
            environments_dir=environments_dir,
            output_dir=output_dir,
        )
    finally:
        t8_5._controller_factory = previous_factory
        t8_5.load_frozen_manifest = previous_t8_5_loader
        live_base.load_frozen_manifest = previous_base_loader
    destination = Path(output_dir)
    base_rows = [
        json.loads(line)
        for line in (destination / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    policy_rows = t8_6_live._policy_live_rows(registry, base_rows)
    selected_condition = next(iter(registry.values())).selected_name
    validation = _validation_checks(
        policy_rows,
        base_rows,
        selected_condition=selected_condition,
        base_report=base_report,
        confirmation=confirmation,
        registry=registry,
    )
    passed = bool(validation["passed"])
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "T8_7_PASSED" if passed else "T8_7_FAILED_CLOSED",
        "manifest_checksum": confirmation["manifest_checksum"],
        "parent_t8_6j_report_checksum": confirmation["parent_t8_6j_report_checksum"],
        "validation": validation,
        "base_live_report_checksum": base_report.get("report_checksum"),
        "elapsed_seconds": time.perf_counter() - started,
        "source_validation_executed": True,
        "t9_audit_authorized": True,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
        "holdout_opened": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    v86c._write_jsonl(destination / "policy_rows.jsonl", policy_rows)
    v86c._write_json(destination / "t8_7_source_validation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-manifest", default=str(DEFAULT_ACTION_MANIFEST))
    parser.add_argument(
        "--confirmation-manifest", default=str(DEFAULT_CONFIRMATION_MANIFEST)
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environments-dir", default="environment_files")
    parser.add_argument("--freeze-actions", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze_actions:
        result = freeze_action_manifest(output_path=args.action_manifest)
    elif args.freeze:
        result = freeze_confirmation_manifest(
            output_path=args.confirmation_manifest,
            action_manifest_path=args.action_manifest,
        )
    else:
        result = run_source_validation(
            confirmation_manifest_path=args.confirmation_manifest,
            action_manifest_path=args.action_manifest,
            environments_dir=args.environments_dir,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze_actions or args.freeze or result.get("status") == "T8_7_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIONS_PER_RESET",
    "DEFAULT_ACTION_MANIFEST",
    "DEFAULT_CONFIRMATION_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "MAXIMUM_ACTIONS",
    "RESETS",
    "VALIDATION_GAMES",
    "VALIDATION_SEED",
    "freeze_action_manifest",
    "freeze_confirmation_manifest",
    "load_action_manifest",
    "load_confirmation_manifest",
    "main",
    "run_source_validation",
]
