"""CLI for the frozen SAGE.T12.5b.3 prospective contrast experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .progress_contrast_experiment import (
    progress_contrast_status,
    run_progress_contrast_collection,
)
from .progress_contrast_protocol import freeze_progress_contrast

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "training" / "sage_t" / "progress_contrast_t12_5b_3_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="freeze the prospective collection before any ARC call"
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    collect = subparsers.add_parser(
        "collect", help="run the fixed source-train detour contrast collection"
    )
    collect.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    collect.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "collection"))
    collect.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="inspect the signed firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument(
        "--receipt", default=str(DEFAULT_OUTPUT / "collection" / "contrast_receipt.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        manifest = freeze_progress_contrast(
            output_path=args.manifest,
            parent_manifest_path=args.parent_manifest,
            parent_receipt_path=args.parent_receipt,
            root=DEFAULT_ROOT,
            allow_dirty=bool(args.allow_dirty),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["scientific_claims_authorized"] else 3
    if args.phase == "collect":
        receipt = run_progress_contrast_collection(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            environments_dir=args.environments_dir,
            root=DEFAULT_ROOT,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = progress_contrast_status(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        root=DEFAULT_ROOT,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
