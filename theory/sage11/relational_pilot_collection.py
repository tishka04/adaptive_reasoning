"""Collect the pre-registered 10,027-row source-train relational pilot."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from theory.non_ar25_active_micro_run import _env_dir

from .relational_dataset import (
    RELATIONAL_PILOT_GAME_QUOTAS,
    RELATIONAL_PILOT_TARGET_TRANSITIONS,
    RelationalJsonlCapture,
    build_relational_manifest,
    make_relational_controller,
    relational_shard_metadata,
    verify_relational_manifest,
)
from .source_dataset_runner import (
    DEFAULT_DUPLICATE_SATURATION_PATIENCE,
    _collect_game,
)
from .splits import SOURCE_TRAIN


DEFAULT_RELATIONAL_OUTPUT_DIR = (
    Path("training") / "sage11" / "relational_pilot_v1"
)
DEFAULT_RELATIONAL_MANIFEST_PATH = (
    DEFAULT_RELATIONAL_OUTPUT_DIR / "manifest.json"
)
RELATIONAL_COLLECTION_FORMAT_VERSION = (
    "sage11-relational-pilot-collection-v1"
)


def run_relational_pilot_collection(
    *,
    output_dir: str | Path = DEFAULT_RELATIONAL_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    action_budget_per_reset: int = 400,
    max_raw_multiplier: int = 50,
    checkpoint_every_resets: int = 10,
    duplicate_saturation_patience: int = (
        DEFAULT_DUPLICATE_SATURATION_PATIENCE
    ),
    workers: int = 1,
) -> Dict[str, Any]:
    """Collect every frozen source-train quota without opening other games."""
    destination = Path(output_dir)
    shard_dir = destination / "shards"
    base_work_dir = destination / "base_work"
    shard_dir.mkdir(parents=True, exist_ok=True)
    base_work_dir.mkdir(parents=True, exist_ok=True)
    environment_root = (
        Path(environments_dir)
        if environments_dir is not None
        else _env_dir()
    )
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("relational collection seeds must be unique")

    reports: Dict[str, Dict[str, Any]] = {}
    pending: Dict[str, Dict[str, Any]] = {}
    for game in SOURCE_TRAIN:
        quota = int(RELATIONAL_PILOT_GAME_QUOTAS[game])
        sidecar_path = shard_dir / f"{game}.jsonl"
        base_path = base_work_dir / f"{game}.jsonl"
        checkpoint_path = base_work_dir / f"{game}.checkpoint.json"
        if _completed_relational_game(
            sidecar_path,
            checkpoint_path,
            game=game,
            quota=quota,
        ):
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            reports[game] = {
                **checkpoint,
                "relational_shard": relational_shard_metadata(
                    sidecar_path,
                    expected_game=game,
                    expected_rows=quota,
                ),
                "resumed_completed_game": True,
            }
            continue
        pending[game] = {
            "game_id": game,
            "quota": quota,
            "sidecar_path": sidecar_path,
            "base_path": base_path,
            "checkpoint_path": checkpoint_path,
            "environment_root": environment_root,
            "seeds": seed_values,
            "action_budget_per_reset": int(action_budget_per_reset),
            "max_raw_multiplier": int(max_raw_multiplier),
            "checkpoint_every_resets": int(checkpoint_every_resets),
            "duplicate_saturation_patience": int(
                duplicate_saturation_patience
            ),
        }

    worker_count = max(1, min(int(workers), len(pending) or 1))
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _collect_relational_game_job,
                    arguments,
                ): game
                for game, arguments in pending.items()
            }
            for future in as_completed(futures):
                game = futures[future]
                reports[game] = future.result()
    else:
        for game, arguments in pending.items():
            reports[game] = _collect_relational_game_job(arguments)

    shard_metadata = [
        relational_shard_metadata(
            shard_dir / f"{game}.jsonl",
            expected_game=game,
            expected_rows=int(RELATIONAL_PILOT_GAME_QUOTAS[game]),
        )
        for game in SOURCE_TRAIN
    ]
    manifest = build_relational_manifest(shard_metadata)
    manifest_path = destination / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    verify_relational_manifest(manifest_path)
    report = {
        "format_version": RELATIONAL_COLLECTION_FORMAT_VERSION,
        "status": "COMPLETE",
        "target_transitions": RELATIONAL_PILOT_TARGET_TRANSITIONS,
        "total_transitions": sum(
            int(item["transitions"])
            for item in shard_metadata
        ),
        "source_train_games": list(SOURCE_TRAIN),
        "game_quotas": dict(RELATIONAL_PILOT_GAME_QUOTAS),
        "seeds": list(seed_values),
        "workers": worker_count,
        "game_reports": {
            game: reports[game]
            for game in SOURCE_TRAIN
        },
        "manifest_path": _repo_path(manifest_path),
        "manifest_checksum": manifest["manifest_checksum"],
        "source_validation_shards_opened": False,
        "historical_shards_opened": False,
        "holdout_shards_opened": False,
    }
    _write_json_atomic(destination / "collection_report.json", report)
    return {"manifest": manifest, "report": report}


def _collect_relational_game_job(
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    values = dict(arguments)
    game = str(values["game_id"])
    quota = int(values["quota"])
    sidecar_path = Path(values["sidecar_path"])
    capture_holder: Dict[str, RelationalJsonlCapture] = {}

    def controller_factory(game_id: str, delegate: Any) -> Any:
        capture = capture_holder.get("capture")
        if capture is None:
            capture = RelationalJsonlCapture(
                sidecar_path,
                expected_existing_rows=len(delegate.builder.records),
            )
            capture_holder["capture"] = capture
        elif capture.count != len(delegate.builder.records):
            raise ValueError(
                "relational capture and base builder lost lockstep"
            )
        return make_relational_controller(game_id, delegate, capture)

    base_report = _collect_game(
        game_id=game,
        quota=quota,
        shard_path=Path(values["base_path"]),
        metadata_path=Path(values["checkpoint_path"]),
        schema_path=None,
        environment_root=Path(values["environment_root"]),
        seeds=tuple(int(seed) for seed in values["seeds"]),
        action_budget_per_reset=int(values["action_budget_per_reset"]),
        max_raw_multiplier=int(values["max_raw_multiplier"]),
        checkpoint_every_resets=int(
            values["checkpoint_every_resets"]
        ),
        duplicate_saturation_patience=int(
            values["duplicate_saturation_patience"]
        ),
        env_factory=values.get("env_factory"),
        controller_factory=controller_factory,
    )
    capture = capture_holder.get("capture")
    if capture is None or capture.count != quota:
        raise ValueError(
            f"relational capture incomplete for {game}: "
            f"{0 if capture is None else capture.count}/{quota}"
        )
    metadata = relational_shard_metadata(
        sidecar_path,
        expected_game=game,
        expected_rows=quota,
    )
    return {
        **base_report,
        "relational_shard": metadata,
        "resumed_completed_game": False,
    }


def _completed_relational_game(
    sidecar_path: Path,
    checkpoint_path: Path,
    *,
    game: str,
    quota: int,
) -> bool:
    if not sidecar_path.exists() or not checkpoint_path.exists():
        return False
    try:
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if (
            str(checkpoint.get("game_id")) != str(game)
            or int(checkpoint.get("quota", -1)) != int(quota)
            or int(checkpoint.get("accepted_transitions", -1))
            != int(quota)
            or str(checkpoint.get("status")) != "COMPLETE"
        ):
            return False
        relational_shard_metadata(
            sidecar_path,
            expected_game=game,
            expected_rows=quota,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the 10,027-row source-train SAGE.11 relational pilot."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RELATIONAL_OUTPUT_DIR,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--action-budget", type=int, default=400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_relational_pilot_collection(
        output_dir=args.out_dir,
        environments_dir=args.environments_dir,
        seeds=tuple(
            int(item)
            for item in str(args.seeds).split(",")
            if item.strip()
        ),
        action_budget_per_reset=int(args.action_budget),
        workers=int(args.workers),
    )
    print(json.dumps({
        "status": result["report"]["status"],
        "total_transitions": result["manifest"]["total_transitions"],
        "manifest_checksum": result["manifest"]["manifest_checksum"],
        "schema_checksum": result["manifest"][
            "relational_schema"
        ]["checksum"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RELATIONAL_MANIFEST_PATH",
    "DEFAULT_RELATIONAL_OUTPUT_DIR",
    "run_relational_pilot_collection",
]
