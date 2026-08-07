"""Frozen SAGE.T10.1 source-validation progress-witness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .progress_witness_v10 import (
    CONTROL_FAMILY,
    SearchConfig,
    SearchOutcome,
    search_progress_witness,
)

FORMAT_VERSION = "sage-t10.1-source-validation-progress-v1"
SOURCE_VALIDATION_GAMES = (
    "re86-4e57566e",
    "ls20-9607627b",
    "sc25-f9b21a2f",
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t10_1_source_validation_manifest.json"
)
DEFAULT_PARENT_REPORT = (
    Path("training") / "sage_t" / "progress_witness_v10_0b" / "report.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "progress_witness_v10_1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _checked_payload(path: str | Path, *, checksum_key: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop(checksum_key, ""))
    if checksum != _checksum(unsigned):
        raise ValueError(f"checksum mismatch: {path}")
    return payload


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    verify_code: bool = True,
) -> dict[str, Any]:
    manifest = _checked_payload(path, checksum_key="manifest_checksum")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported T10.1 manifest")
    if manifest.get("status") != "FROZEN_BEFORE_T10_1_SOURCE_VALIDATION":
        raise ValueError("T10.1 manifest is not frozen")
    if manifest.get("source_validation_games") != list(SOURCE_VALIDATION_GAMES):
        raise ValueError("T10.1 source-validation games drifted")
    parent = _checked_payload(DEFAULT_PARENT_REPORT, checksum_key="report_checksum")
    if parent.get("status") != "PASS_T10_0_AUTHORIZE_T10_1":
        raise ValueError("T10.0b did not authorize T10.1")
    if parent.get("report_checksum") != manifest.get("parent_t10_0b_report_checksum"):
        raise ValueError("T10.1 parent report binding drifted")
    if verify_code:
        directory = Path(__file__).resolve().parent
        paths = {
            "progress_witness_v10.py": directory / "progress_witness_v10.py",
            "source_validation_progress_v10_1.py": Path(__file__).resolve(),
        }
        current = {
            name: hashlib.sha256(file.read_bytes()).hexdigest()
            for name, file in paths.items()
        }
        if current != manifest.get("code_sha256"):
            raise ValueError("T10.1 code drifted")
    firewall = manifest.get("firewall", {})
    if firewall.get("source_validation_opened") is not True or any(
        bool(firewall.get(key))
        for key in ("ar25_opened", "holdout_opened", "production_authority")
    ):
        raise ValueError("T10.1 firewall is invalid")
    return manifest


def build_report(
    *,
    manifest: Mapping[str, Any],
    outcomes: Sequence[SearchOutcome],
    wall_seconds: float,
    include_scan_rows: bool = True,
) -> dict[str, Any]:
    gate = dict(manifest["gate"])
    positive = [outcome for outcome in outcomes if outcome.passed]
    total_levels = sum(
        int(outcome.witness.level_delta)
        for outcome in positive
        if outcome.witness is not None
    )
    game_over_events = sum(outcome.terminal_events for outcome in outcomes)
    checks = {
        "all_games_executed": len(outcomes) == len(SOURCE_VALIDATION_GAMES)
        and {outcome.game for outcome in outcomes} == set(SOURCE_VALIDATION_GAMES),
        "minimum_progress_games": len(positive)
        >= int(gate["minimum_progress_games"]),
        "minimum_total_levels": total_levels >= int(gate["minimum_total_levels"]),
        "posterior_top8": all(
            outcome.witness is not None
            and outcome.witness.posterior_rank is not None
            and outcome.witness.posterior_rank <= int(gate["maximum_posterior_rank"])
            for outcome in positive
        ),
        "coordinate_free": all(
            outcome.witness is not None
            and bool(outcome.witness.transferable_payload)
            for outcome in positive
        ),
        "zero_illegal_actions": sum(outcome.illegal_actions for outcome in outcomes)
        <= int(gate["maximum_illegal_actions"]),
        "zero_errors": sum(len(outcome.errors) for outcome in outcomes)
        <= int(gate["maximum_errors"]),
        "game_over_bound": game_over_events <= int(gate["maximum_game_over_events"]),
        "wall_time": float(wall_seconds) <= float(gate["maximum_wall_seconds"]),
        "holdout_closed": True,
        "ar25_closed": True,
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "T10.1_SOURCE_VALIDATION",
        "status": (
            "PASS_T10_1_AUTHORIZE_INTEGRATION_PILOT" if passed else "FAIL_CLOSED"
        ),
        "manifest_checksum": manifest["manifest_checksum"],
        "source_validation_games": list(SOURCE_VALIDATION_GAMES),
        "config": dict(manifest["search_config"]),
        "outcomes": [
            outcome.to_dict(include_scan_rows=include_scan_rows)
            for outcome in outcomes
        ],
        "diagnosis_counts": dict(Counter(outcome.diagnosis for outcome in outcomes)),
        "metrics": {
            "progress_games": len(positive),
            "total_levels": total_levels,
            "actions_executed": sum(outcome.actions_executed for outcome in outcomes),
            "illegal_actions": sum(outcome.illegal_actions for outcome in outcomes),
            "errors": sum(len(outcome.errors) for outcome in outcomes),
            "game_over_events": game_over_events,
        },
        "checks": checks,
        "passed": passed,
        "firewall": {
            "source_validation_opened": True,
            "holdout_opened": False,
            "ar25_opened": False,
            "production_authority": False,
            "integration_pilot_authorized": passed,
        },
        "wall_seconds": round(float(wall_seconds), 3),
        "scientific_limit": (
            "T10.1 evaluates active grounding of a transferred control skeleton. "
            "It is not a paired controller-level advantage test and cannot open "
            "the final holdout."
        ),
    }
    unsigned = dict(report)
    report["report_checksum"] = _checksum(unsigned)
    return report


def run_source_validation(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    include_scan_rows: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    config = SearchConfig(**dict(manifest["search_config"]))
    started = time.perf_counter()
    outcomes = [
        search_progress_witness(
            game,
            config=config,
            enabled_control_families=(CONTROL_FAMILY,),
        )
        for game in SOURCE_VALIDATION_GAMES
    ]
    report = build_report(
        manifest=manifest,
        outcomes=outcomes,
        wall_seconds=time.perf_counter() - started,
        include_scan_rows=include_scan_rows,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen SAGE.T10.1 gate.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    report = run_source_validation(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        include_scan_rows=not args.compact,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "metrics": report["metrics"],
                "report_checksum": report["report_checksum"],
                "output": str(Path(args.output_dir) / "report.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "SOURCE_VALIDATION_GAMES",
    "build_report",
    "load_manifest",
    "run_source_validation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
