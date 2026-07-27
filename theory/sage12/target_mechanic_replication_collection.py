"""Prospective chronological collector for the SAGE12 V4.2 target pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .mechanic_replication_collection import run_collection as _run_collection
from .target_mechanic_replication import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    TARGET_EFFECT_LABELS,
    load_frozen_manifest,
)

COLLECTION_FORMAT_VERSION = "sage12-target-mechanic-collection-v4.2"


def run_collection(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Collect only after the immutable V4.2 source preflight passed."""
    destination = Path(output_dir)
    preflight_path = destination / "source_train_preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("V4.2 source preflight is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "PASS_SOURCE_TRAIN_PREFLIGHT"
        or not all(dict(preflight.get("gates", {})).values())
    ):
        raise RuntimeError("V4.2 source preflight did not authorize collection")
    frozen = load_frozen_manifest(frozen_manifest_path)
    if preflight.get("frozen_manifest_checksum") != frozen["manifest_checksum"]:
        raise RuntimeError("V4.2 preflight/manifest checksum mismatch")
    return _run_collection(
        frozen_manifest_path=frozen_manifest_path,
        output_dir=destination,
        environments_dir=environments_dir,
        manifest_loader=load_frozen_manifest,
        collection_format_version=COLLECTION_FORMAT_VERSION,
        collection_phase="v4_2_target_prospective_fixed",
        salt_prefix="v4.2-target",
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
