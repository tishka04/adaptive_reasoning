"""CLI for the frozen SAGE.T12.5b.5 goal-viability experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .goal_viability_experiment import (
    goal_viability_status,
    run_goal_viability_calibration,
    run_goal_viability_evaluation,
)
from .goal_viability_protocol import freeze_goal_viability

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "training" / "sage_t" / "goal_viability_t12_5b_5_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="freeze the paired viability assay before any T12.5b.5 ARC call",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    calibrate = subparsers.add_parser(
        "calibrate",
        help="run the fixed 8701 first-action viability branches",
    )
    calibrate.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    calibrate.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT / "calibration"),
    )
    calibrate.add_argument("--environments-dir", default="environment_files")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate only a passed sealed viability contrast on 8705",
    )
    evaluate.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    evaluate.add_argument(
        "--calibration-receipt",
        default=str(DEFAULT_OUTPUT / "calibration" / "calibration_receipt.json"),
    )
    evaluate.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT / "evaluation"),
    )
    evaluate.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="inspect the signed firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument(
        "--calibration-receipt",
        default=str(DEFAULT_OUTPUT / "calibration" / "calibration_receipt.json"),
    )
    status.add_argument(
        "--evaluation-receipt",
        default=str(DEFAULT_OUTPUT / "evaluation" / "evaluation_receipt.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        manifest = freeze_goal_viability(
            output_path=args.manifest,
            parent_manifest_path=args.parent_manifest,
            parent_receipt_path=args.parent_receipt,
            root=DEFAULT_ROOT,
            allow_dirty=bool(args.allow_dirty),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["scientific_claims_authorized"] else 3
    if args.phase == "calibrate":
        receipt = run_goal_viability_calibration(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            environments_dir=args.environments_dir,
            root=DEFAULT_ROOT,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    if args.phase == "evaluate":
        receipt = run_goal_viability_evaluation(
            manifest_path=args.manifest,
            calibration_receipt_path=args.calibration_receipt,
            output_dir=args.output_dir,
            environments_dir=args.environments_dir,
            root=DEFAULT_ROOT,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = goal_viability_status(
        manifest_path=args.manifest,
        calibration_receipt_path=args.calibration_receipt,
        evaluation_receipt_path=args.evaluation_receipt,
        root=DEFAULT_ROOT,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
