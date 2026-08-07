"""SAGE.T9.1 source-train terminal recalibration gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from . import calibration_gate_v8_6c as v86c
from . import live_shadow_pilot_v10 as live_r3
from . import reachability_audit_v9 as t9_0
from .terminal_calibration_v9 import (
    T9_1_POLICIES,
    ObservedSafetyCalibrator,
    TerminalCalibrationPolicy,
)

FORMAT_VERSION = "sage-t9.1-terminal-calibration-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t9_1_terminal_manifest.json"
)
DEFAULT_INPUT_ROWS = live_r3.DEFAULT_OUTPUT_DIR / "rows.jsonl"
DEFAULT_PARENT_REPORT = t9_0.DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "terminal_v9_1"
SOURCE_GAMES = ("lp85", "su15")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 9101


def _code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: v86c._file_sha256(directory / name)
        for name in ("terminal_calibration_v9.py", "terminal_gate_v9_1.py")
    }


def _load_parent_report(path: str | Path = DEFAULT_PARENT_REPORT) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(report)
    checksum = str(unsigned.pop("report_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.0 report checksum mismatch")
    if report.get("status") != "T9_0_COMPLETE" or report.get("t9_1_authorized") is not True:
        raise ValueError("T9.0 did not authorize T9.1")
    return report


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    input_rows_path: str | Path = DEFAULT_INPUT_ROWS,
) -> dict[str, Any]:
    parent = _load_parent_report()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "FROZEN_BEFORE_T9_1_SOURCE_TRAIN_GATE",
        "frozen_at": "2026-08-06",
        "parent_t9_0_report_checksum": parent["report_checksum"],
        "input_rows_sha256": v86c._file_sha256(Path(input_rows_path)),
        "code_sha256": _code_hashes(),
        "source_train_games": list(SOURCE_GAMES),
        "candidate_policies": {
            name: v86c._json_safe(policy.__dict__)
            for name, policy in T9_1_POLICIES.items()
        },
        "selection": {
            "maximum_false_high_terminal_rate_per_game": 0.05,
            "require_lp85_log_loss_interval_lower_positive": True,
            "require_su15_log_loss_interval_lower_nonnegative": True,
            "require_observed_danger_preservation": True,
            "tie_break": ["lp85_log_loss_delta", "lp85_brier_delta", "minimum_safe_observations"],
        },
        "bootstrap": {
            "unit": "reset",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "firewall": {
            "authority": "shadow",
            "source_train_only": True,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "bounded_authority": False,
            "active_authority": False,
        },
    }
    payload["manifest_checksum"] = v86c._checksum(payload)
    v86c._write_json(Path(output_path), payload)
    return payload


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    input_rows_path: str | Path = DEFAULT_INPUT_ROWS,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != v86c._checksum(unsigned):
        raise ValueError("T9.1 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T9.1 manifest")
    if payload.get("status") != "FROZEN_BEFORE_T9_1_SOURCE_TRAIN_GATE":
        raise ValueError("T9.1 manifest is not frozen")
    if payload.get("code_sha256") != _code_hashes():
        raise ValueError("T9.1 code drifted")
    if payload.get("input_rows_sha256") != v86c._file_sha256(Path(input_rows_path)):
        raise ValueError("T9.1 live rows drifted")
    if payload.get("parent_t9_0_report_checksum") != _load_parent_report()["report_checksum"]:
        raise ValueError("T9.1 parent report drifted")
    expected = {
        name: v86c._json_safe(policy.__dict__)
        for name, policy in T9_1_POLICIES.items()
    }
    if payload.get("candidate_policies") != expected:
        raise ValueError("T9.1 candidate policies drifted")
    firewall = payload.get("firewall", {})
    if firewall.get("authority") != "shadow" or any(
        bool(firewall.get(key))
        for key in (
            "source_validation_opened",
            "ar25_opened",
            "holdout_opened",
            "bounded_authority",
            "active_authority",
        )
    ):
        raise ValueError("T9.1 firewall is open")
    return payload


def _game(value: str) -> str:
    return next((game for game in SOURCE_GAMES if str(value).startswith(game)), "")


def _binary_metrics(probability: float, actual: int) -> tuple[float, float]:
    bounded = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return (
        (bounded - int(actual)) ** 2,
        -(actual * math.log(bounded) + (1 - actual) * math.log(1 - bounded)),
    )


def evaluate_policy(
    rows: Sequence[Mapping[str, Any]],
    policy: TerminalCalibrationPolicy,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    calibrators = {
        game: ObservedSafetyCalibrator(policy) for game in SOURCE_GAMES
    }
    output = []
    danger_predictions = 0
    preserved_danger_predictions = 0
    for row in rows:
        game = _game(str(row["game_id"]))
        if not game:
            continue
        calibrator = calibrators[game]
        action = calibrator.action_from_key(str(row["action_key"]))
        raw = float(row["predicted_terminal"])
        was_danger = calibrator.is_observed_danger(action)
        calibrated = calibrator.calibrate(action, raw)
        if was_danger and raw >= policy.high_probability_threshold:
            danger_predictions += 1
            preserved_danger_predictions += int(calibrated == raw)
        actual = int(bool(row["actual_terminal"]))
        raw_brier, raw_loss = _binary_metrics(raw, actual)
        calibrated_brier, calibrated_loss = _binary_metrics(calibrated, actual)
        output.append(
            {
                "game": game,
                "reset": int(row["reset_index"]),
                "step": int(row["step"]),
                "action_key": action.key,
                "actual_terminal": actual,
                "raw_probability": raw,
                "calibrated_probability": calibrated,
                "raw_brier": raw_brier,
                "calibrated_brier": calibrated_brier,
                "raw_log_loss": raw_loss,
                "calibrated_log_loss": calibrated_loss,
            }
        )
        calibrator.observe_outcome(action, bool(actual))
    per_game = {}
    for game in SOURCE_GAMES:
        selected = [row for row in output if row["game"] == game]
        per_game[game] = {
            "rows": len(selected),
            "terminal_positives": sum(row["actual_terminal"] for row in selected),
            "raw_brier": mean(row["raw_brier"] for row in selected),
            "calibrated_brier": mean(row["calibrated_brier"] for row in selected),
            "brier_delta": mean(
                row["raw_brier"] - row["calibrated_brier"] for row in selected
            ),
            "raw_log_loss": mean(row["raw_log_loss"] for row in selected),
            "calibrated_log_loss": mean(
                row["calibrated_log_loss"] for row in selected
            ),
            "log_loss_delta": mean(
                row["raw_log_loss"] - row["calibrated_log_loss"] for row in selected
            ),
            "false_high_terminal_rate": mean(
                float(
                    not row["actual_terminal"]
                    and row["calibrated_probability"]
                    >= policy.high_probability_threshold
                )
                for row in selected
            ),
            "calibrated_predictions": sum(
                row["calibrated_probability"] < row["raw_probability"]
                for row in selected
            ),
        }
    return (
        {
            "policy": policy.name,
            "per_game": per_game,
            "danger_predictions": danger_predictions,
            "preserved_danger_predictions": preserved_danger_predictions,
            "danger_preservation_rate": (
                1.0
                if danger_predictions == 0
                else preserved_danger_predictions / danger_predictions
            ),
            "calibrators": {
                game: calibrator.snapshot()
                for game, calibrator in calibrators.items()
            },
        },
        tuple(output),
    )


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    game: str,
    metric: str,
    manifest: Mapping[str, Any],
) -> dict[str, float]:
    selected = [row for row in rows if row["game"] == game]
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[int(row["reset"])].append(row)
    clusters = tuple(grouped.values())
    deltas = [
        mean(float(row[f"raw_{metric}"]) - float(row[f"calibrated_{metric}"]) for row in cluster)
        for cluster in clusters
    ]
    rng = random.Random(int(manifest["bootstrap"]["seed"]))
    samples = []
    for _ in range(int(manifest["bootstrap"]["samples"])):
        samples.append(mean(rng.choice(deltas) for _ in deltas))
    samples.sort()
    tail = (1.0 - float(manifest["bootstrap"]["confidence"])) / 2.0

    def quantile(value: float) -> float:
        index = min(len(samples) - 1, max(0, int(value * (len(samples) - 1))))
        return samples[index]

    return {
        "mean": mean(deltas),
        "lower_95": quantile(tail),
        "upper_95": quantile(1.0 - tail),
        "clusters": len(clusters),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    conditions = {}
    condition_rows = {}
    survivors = []
    for name, policy in T9_1_POLICIES.items():
        metrics, evaluated = evaluate_policy(rows, policy)
        intervals = {
            game: {
                metric: _cluster_interval(
                    evaluated,
                    game=game,
                    metric=metric,
                    manifest=manifest,
                )
                for metric in ("brier", "log_loss")
            }
            for game in SOURCE_GAMES
        }
        metrics["paired_intervals"] = intervals
        checks = {
            "false_high_terminal_rate": all(
                metrics["per_game"][game]["false_high_terminal_rate"]
                <= float(
                    manifest["selection"]["maximum_false_high_terminal_rate_per_game"]
                )
                for game in SOURCE_GAMES
            ),
            "lp85_brier_improved": intervals["lp85"]["brier"]["lower_95"] > 0.0,
            "lp85_log_loss_improved": intervals["lp85"]["log_loss"]["lower_95"] > 0.0,
            "su15_brier_nonnegative": intervals["su15"]["brier"]["lower_95"] >= 0.0,
            "su15_log_loss_nonnegative": intervals["su15"]["log_loss"]["lower_95"] >= 0.0,
            "observed_danger_preserved": metrics["danger_preservation_rate"] == 1.0,
        }
        metrics["checks"] = checks
        metrics["passed"] = all(checks.values())
        conditions[name] = metrics
        condition_rows[name] = evaluated
        if metrics["passed"]:
            survivors.append(name)
    selected = (
        max(
            survivors,
            key=lambda name: (
                conditions[name]["per_game"]["lp85"]["log_loss_delta"],
                conditions[name]["per_game"]["lp85"]["brier_delta"],
                -T9_1_POLICIES[name].minimum_safe_observations,
            ),
        )
        if survivors
        else ""
    )
    passed = bool(selected)
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "T9_1_PASSED" if passed else "T9_1_FAILED_CLOSED",
        "manifest_checksum": manifest["manifest_checksum"],
        "selected_policy": selected or None,
        "conditions": conditions,
        "rows": len(rows),
        "t9_2_authorized": passed,
        "bounded_authority_authorized": False,
        "active_authority_authorized": False,
        "source_validation_opened": False,
        "holdout_opened": False,
    }
    report["report_checksum"] = v86c._checksum(report)
    return report, (() if not selected else condition_rows[selected])


def run_gate(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    input_rows_path: str | Path = DEFAULT_INPUT_ROWS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, input_rows_path=input_rows_path)
    rows = tuple(
        json.loads(line)
        for line in Path(input_rows_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    report, selected_rows = build_report(rows, manifest=manifest)
    destination = Path(output_dir)
    v86c._write_jsonl(destination / "selected_rows.jsonl", selected_rows)
    v86c._write_json(destination / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--input-rows", default=str(DEFAULT_INPUT_ROWS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.freeze:
        result = freeze_manifest(
            output_path=args.manifest,
            input_rows_path=args.input_rows,
        )
    else:
        result = run_gate(
            manifest_path=args.manifest,
            input_rows_path=args.input_rows,
            output_dir=args.output_dir,
        )
    print(json.dumps(v86c._json_safe(result), indent=2, sort_keys=True))
    return 0 if args.freeze or result.get("status") == "T9_1_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "build_report",
    "evaluate_policy",
    "freeze_manifest",
    "load_manifest",
    "main",
    "run_gate",
]
