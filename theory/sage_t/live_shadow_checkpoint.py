"""Checkpointed T8.0.1 executor for the immutable T8.0 live protocol.

T8.0.1 changes only execution durability: each pre-registered condition runs
in an isolated subprocess and writes an atomic checkpoint.  Games, seeds,
budgets, controller caps and scientific gates are inherited byte-for-byte from
T8.0.  A timeout becomes an explicit failed condition instead of erasing the
conditions that already completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from theory.unified_cognition_ab_benchmark import _run_arm

from .live_shadow_pilot import (
    DEFAULT_MANIFEST_PATH as T8_MANIFEST_PATH,
)
from .live_shadow_pilot import (
    LiveShadowRow,
    _condition_summary,
    _controller_factory,
    build_report,
    rows_from_paired_arms,
    runtime_capabilities,
)
from .live_shadow_pilot import (
    load_frozen_manifest as load_t8_manifest,
)

FORMAT_VERSION = "sage-t8.0.1-live-shadow-checkpoint-v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t8_0_1_frozen_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "live_shadow_pilot_v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _checksum(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(_json_safe(value)).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(_json_safe(row)) + "\n")
    os.replace(temporary, path)


def load_frozen_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("manifest_checksum", ""))
    if checksum != _checksum(unsigned):
        raise ValueError("SAGE.T8.0.1 manifest checksum mismatch")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported SAGE.T8.0.1 manifest")
    if payload.get("status") != "FROZEN_BEFORE_CHECKPOINTED_RETRY":
        raise ValueError("SAGE.T8.0.1 manifest is not frozen")
    base = load_t8_manifest(T8_MANIFEST_PATH)
    if payload.get("base_t8_manifest_checksum") != base["manifest_checksum"]:
        raise ValueError("SAGE.T8.0.1 base T8 manifest drifted")
    expected_hash = payload.get("code_sha256", {}).get(
        "live_shadow_checkpoint.py"
    )
    if not expected_hash:
        raise ValueError("SAGE.T8.0.1 code hash is missing")
    if _file_sha256(Path(__file__)) != expected_hash:
        raise ValueError("SAGE.T8.0.1 checkpoint runner drifted")
    if payload.get("scientific_protocol_changes") != []:
        raise ValueError("SAGE.T8.0.1 cannot change the scientific protocol")
    return payload


def _checkpoint_name(game_id: str, seed: int) -> str:
    short = str(game_id).split("-", 1)[0]
    return f"{short}_seed{int(seed)}.json"


def _validated_runtime(base: dict[str, Any]) -> dict[str, Any]:
    runtime = runtime_capabilities()
    expected = {
        "arc-agi": str(base["runtime"]["arc_agi"]),
        "arcengine": str(base["runtime"]["arcengine"]),
    }
    observed = dict(runtime.get("versions", {}) or {})
    versions_match = all(
        observed.get(package) == version for package, version in expected.items()
    )
    runtime["expected_versions"] = expected
    runtime["versions_match"] = versions_match
    runtime["ready"] = bool(runtime.get("ready")) and versions_match
    if not versions_match and not runtime.get("reason"):
        runtime["reason"] = "sdk_version_mismatch"
    return runtime


def run_condition(
    *,
    game_id: str,
    seed: int,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    retry_manifest = load_frozen_manifest()
    base = load_t8_manifest(T8_MANIFEST_PATH)
    if str(game_id) not in base["source_train_games"]:
        raise ValueError("condition game is outside the frozen T8 panel")
    if int(seed) not in {int(value) for value in base["seeds"]}:
        raise ValueError("condition seed is outside the frozen T8 panel")
    runtime = _validated_runtime(base)
    if not runtime.get("ready"):
        raise RuntimeError(str(runtime.get("reason", "runtime unavailable")))
    common = {
        "arm": "unified",
        "game_id": str(game_id),
        "seed": int(seed),
        "action_budget_per_reset": int(base["action_budget_per_reset"]),
        "resets": int(base["resets"]),
        "env_dir": Path("environment_files"),
        "env_factory": None,
    }
    started = time.perf_counter()
    off = _run_arm(
        controller_factory=_controller_factory(mode="off", manifest=base),
        **common,
    )
    shadow = _run_arm(
        controller_factory=_controller_factory(mode="shadow", manifest=base),
        **common,
    )
    elapsed = time.perf_counter() - started
    rows = rows_from_paired_arms(
        game_id=str(game_id),
        seed=int(seed),
        off=off,
        shadow=shadow,
    )
    condition = _condition_summary(
        game_id=str(game_id),
        seed=int(seed),
        off=off,
        shadow=shadow,
        wall_clock_seconds=elapsed,
    )
    condition["checkpoint_status"] = "COMPLETE"
    payload = {
        "format_version": FORMAT_VERSION,
        "retry_manifest_checksum": retry_manifest["manifest_checksum"],
        "base_t8_manifest_checksum": base["manifest_checksum"],
        "checkpoint_status": "COMPLETE",
        "game_id": str(game_id),
        "seed": int(seed),
        "wall_clock_seconds": elapsed,
        "condition": condition,
        "rows": [asdict(row) for row in rows],
    }
    payload["checkpoint_checksum"] = _checksum(payload)
    destination = Path(output_dir) / "checkpoints" / _checkpoint_name(
        game_id,
        seed,
    )
    _write_json(destination, payload)
    return payload


def _write_timeout_checkpoint(
    *,
    game_id: str,
    seed: int,
    timeout_seconds: float,
    output_dir: str | Path,
    stderr: str = "",
) -> dict[str, Any]:
    retry_manifest = load_frozen_manifest()
    base = load_t8_manifest(T8_MANIFEST_PATH)
    condition = {
        "game_id": str(game_id),
        "seed": int(seed),
        "off_actions": 0,
        "shadow_actions": 0,
        "same_action_trace": False,
        "same_reset_states": False,
        "controller_errors": 0,
        "illegal_actions": 0,
        "environment_errors": 1,
        "interventions": 0,
        "trace_errors": 0,
        "effective_mode": "shadow",
        "wall_clock_seconds": float(timeout_seconds),
        "checkpoint_status": "TIMEOUT",
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "retry_manifest_checksum": retry_manifest["manifest_checksum"],
        "base_t8_manifest_checksum": base["manifest_checksum"],
        "checkpoint_status": "TIMEOUT",
        "game_id": str(game_id),
        "seed": int(seed),
        "wall_clock_seconds": float(timeout_seconds),
        "stderr_tail": str(stderr)[-2000:],
        "condition": condition,
        "rows": [],
    }
    payload["checkpoint_checksum"] = _checksum(payload)
    destination = Path(output_dir) / "checkpoints" / _checkpoint_name(
        game_id,
        seed,
    )
    _write_json(destination, payload)
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    checksum = str(unsigned.pop("checkpoint_checksum", ""))
    if checksum != _checksum(unsigned):
        raise ValueError(f"checkpoint checksum mismatch: {path}")
    return payload


def aggregate_checkpoints(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    retry_manifest = load_frozen_manifest()
    base = load_t8_manifest(T8_MANIFEST_PATH)
    destination = Path(output_dir)
    rows: list[LiveShadowRow] = []
    conditions = []
    checkpoint_status = {}
    total_wall = 0.0
    for game_id in base["source_train_games"]:
        for seed in base["seeds"]:
            name = _checkpoint_name(str(game_id), int(seed))
            path = destination / "checkpoints" / name
            if not path.exists():
                payload = _write_timeout_checkpoint(
                    game_id=str(game_id),
                    seed=int(seed),
                    timeout_seconds=0.0,
                    output_dir=destination,
                    stderr="missing checkpoint",
                )
                payload["checkpoint_status"] = "MISSING"
                payload["condition"]["checkpoint_status"] = "MISSING"
            else:
                payload = _load_checkpoint(path)
            if payload.get("retry_manifest_checksum") != retry_manifest[
                "manifest_checksum"
            ]:
                raise ValueError(f"retry manifest mismatch: {path}")
            if payload.get("base_t8_manifest_checksum") != base[
                "manifest_checksum"
            ]:
                raise ValueError(f"base T8 manifest mismatch: {path}")
            checkpoint_status[name] = str(payload.get("checkpoint_status", ""))
            conditions.append(dict(payload["condition"]))
            total_wall += float(payload.get("wall_clock_seconds", 0.0) or 0.0)
            rows.extend(LiveShadowRow(**row) for row in payload.get("rows", ()))
    report_manifest = dict(base)
    report_manifest["manifest_checksum"] = retry_manifest["manifest_checksum"]
    report = build_report(
        rows,
        manifest=report_manifest,
        conditions=conditions,
        runtime=_validated_runtime(base),
        wall_clock_seconds=total_wall,
    )
    report["format_version"] = FORMAT_VERSION
    report["base_t8_manifest_checksum"] = base["manifest_checksum"]
    report["checkpoint_status"] = checkpoint_status
    report["completed_conditions"] = sum(
        status == "COMPLETE" for status in checkpoint_status.values()
    )
    report["timed_out_conditions"] = sum(
        status == "TIMEOUT" for status in checkpoint_status.values()
    )
    unsigned = dict(report)
    unsigned.pop("report_checksum", None)
    report["report_checksum"] = _checksum(unsigned)
    _write_jsonl(destination / "rows.jsonl", [asdict(row) for row in rows])
    _write_json(destination / "report.json", report)
    return report


def run_all(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    retry_manifest = load_frozen_manifest()
    base = load_t8_manifest(T8_MANIFEST_PATH)
    timeout = float(retry_manifest["condition_timeout_seconds"])
    for game_id in base["source_train_games"]:
        for seed in base["seeds"]:
            command = (
                sys.executable,
                "-m",
                "theory.sage_t.live_shadow_checkpoint",
                "condition",
                "--game",
                str(game_id),
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as error:
                _write_timeout_checkpoint(
                    game_id=str(game_id),
                    seed=int(seed),
                    timeout_seconds=timeout,
                    output_dir=output_dir,
                    stderr=str(error.stderr or ""),
                )
                continue
            if completed.returncode != 0:
                _write_timeout_checkpoint(
                    game_id=str(game_id),
                    seed=int(seed),
                    timeout_seconds=0.0,
                    output_dir=output_dir,
                    stderr=completed.stderr,
                )
    return aggregate_checkpoints(output_dir=output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    condition = subparsers.add_parser("condition")
    condition.add_argument("--game", required=True)
    condition.add_argument("--seed", type=int, required=True)
    condition.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    if args.command == "condition":
        result = run_condition(
            game_id=args.game,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.command == "aggregate":
        result = aggregate_checkpoints(output_dir=args.output_dir)
    else:
        result = run_all(output_dir=args.output_dir)
    summary = {
        "status": result.get("status", result.get("checkpoint_status")),
        "diagnosis": result.get("diagnosis"),
        "rows": result.get("rows", []),
        "completed_conditions": result.get("completed_conditions"),
        "timed_out_conditions": result.get("timed_out_conditions"),
    }
    if isinstance(summary["rows"], list):
        summary["rows"] = len(summary["rows"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "aggregate_checkpoints",
    "load_frozen_manifest",
    "main",
    "run_all",
    "run_condition",
]
