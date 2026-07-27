"""Prospective chronological collector for SAGE12 V4.2.1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .mechanic_replication_collection import run_collection as _run_collection
from .target_mechanic_recovery import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    TARGET_EFFECT_LABELS,
    load_frozen_manifest,
)

COLLECTION_FORMAT_VERSION = "sage12-target-mechanic-collection-v4.2.1"


def run_collection(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    rehearsal_path = destination / "source_rehearsal.json"
    preflight_path = destination / "source_train_preflight.json"
    if not rehearsal_path.exists() or not preflight_path.exists():
        raise RuntimeError("V4.2.1 source gates are missing")
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        rehearsal.get("status") != "PASS_SOURCE_REHEARSAL"
        or not all(dict(rehearsal.get("checks", {})).values())
    ):
        raise RuntimeError("V4.2.1 rehearsal did not authorize collection")
    if (
        preflight.get("status") != "PASS_SOURCE_TRAIN_PREFLIGHT"
        or not all(dict(preflight.get("gates", {})).values())
    ):
        raise RuntimeError("V4.2.1 preflight did not authorize collection")
    if (
        rehearsal.get("frozen_manifest_checksum") != frozen["manifest_checksum"]
        or preflight.get("frozen_manifest_checksum")
        != frozen["manifest_checksum"]
    ):
        raise RuntimeError("V4.2.1 source gate/manifest mismatch")
    return _run_collection(
        frozen_manifest_path=frozen_manifest_path,
        output_dir=destination,
        environments_dir=environments_dir,
        manifest_loader=load_frozen_manifest,
        collection_format_version=COLLECTION_FORMAT_VERSION,
        collection_phase="v4_2_1_target_prospective_fixed",
        salt_prefix="v4.2.1-target",
        effect_labels=TARGET_EFFECT_LABELS,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-manifest",
        default=str(DEFAULT_FROZEN_MANIFEST_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environments-dir")
    args = parser.parse_args(argv)
    result = run_collection(
        frozen_manifest_path=args.frozen_manifest,
        output_dir=args.output_dir,
        environments_dir=args.environments_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["COLLECTION_FORMAT_VERSION", "run_collection"]
