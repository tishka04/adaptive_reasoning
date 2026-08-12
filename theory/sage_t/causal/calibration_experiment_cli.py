"""CLI for the preregistered SAGE.T12.4a.1 calibration transport test."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .calibration_experiment import (
    calibration_experiment_status,
    collect_calibration_experiment,
    train_calibration_experiment,
)
from .calibration_protocol import freeze_calibration_experiment

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "calibration_t12_4a_1_bp35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="seal the calibration-only T12.4a failure and prospective split",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    collect = subparsers.add_parser(
        "collect",
        help="collect unopened train, calibration and confirmation seeds",
    )
    collect.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    collect.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "collection"))
    collect.add_argument("--environments-dir", default="environment_files")

    train = subparsers.add_parser(
        "train",
        help="fit the representation, calibrate separately, then confirm once",
    )
    train.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    train.add_argument(
        "--collection-receipt",
        default=str(DEFAULT_OUTPUT / "collection" / "collection_receipt.json"),
    )
    train.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "training"))

    status = subparsers.add_parser("status", help="verify T12.4a.1 artifacts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_calibration_experiment(
                output_path=args.manifest,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "collect":
            result = collect_calibration_experiment(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "train":
            result = train_calibration_experiment(
                manifest_path=args.manifest,
                collection_receipt_path=args.collection_receipt,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = calibration_experiment_status(
                manifest_path=args.manifest,
                receipt_path=args.receipt,
            )
            exit_code = 0
    except (
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        result = {
            "format_version": "sage-t12.4a.1-calibration-cli-error-v1",
            "phase": args.phase,
            "status": "FAILED_CLOSED",
            "reason": f"{type(exc).__name__}:{exc}",
        }
        exit_code = 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "build_parser", "main"]
