"""CLI for the frozen SAGE.T12.5c paired goal-cursor control."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .goal_cursor_control_experiment import (
    goal_cursor_control_status,
    run_goal_cursor_control,
)
from .goal_cursor_control_protocol import freeze_goal_cursor_control

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "training" / "sage_t" / "goal_cursor_control_t12_5c_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="freeze the equal-capacity pair before any T12.5c ARC call",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="run the fixed eight-trial paired control once",
    )
    run.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "control"))
    run.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="inspect the signed firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument(
        "--control-receipt",
        default=str(DEFAULT_OUTPUT / "control" / "control_receipt.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        manifest = freeze_goal_cursor_control(
            output_path=args.manifest,
            parent_manifest_path=args.parent_manifest,
            parent_receipt_path=args.parent_receipt,
            root=DEFAULT_ROOT,
            allow_dirty=bool(args.allow_dirty),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["scientific_claims_authorized"] else 3
    if args.phase == "run":
        receipt = run_goal_cursor_control(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            environments_dir=args.environments_dir,
            root=DEFAULT_ROOT,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = goal_cursor_control_status(
        manifest_path=args.manifest,
        control_receipt_path=args.control_receipt,
        root=DEFAULT_ROOT,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
