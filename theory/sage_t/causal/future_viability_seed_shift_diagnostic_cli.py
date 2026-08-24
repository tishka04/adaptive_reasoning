"""CLI for the offline SAGE.T12.6.1b seed-shift diagnostic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .future_viability_seed_shift_diagnostic_experiment import (
    future_viability_seed_shift_diagnostic_status,
    run_future_viability_seed_shift_diagnostic,
)
from .future_viability_seed_shift_diagnostic_protocol import (
    freeze_future_viability_seed_shift_diagnostic,
)

DEFAULT_OUTPUT = (
    Path("training")
    / "sage_t"
    / "future_viability_seed_shift_diagnostic_t12_6_1b_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="freeze 9202 attribution axes and parent bindings"
    )
    freeze.add_argument("--hierarchy-manifest", required=True)
    freeze.add_argument("--hierarchy-evaluation-receipt", required=True)
    freeze.add_argument("--conflict-manifest", required=True)
    freeze.add_argument("--conflict-diagnostic-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))

    diagnose = subparsers.add_parser(
        "diagnose", help="attribute the frozen 9202 ranking collapse"
    )
    diagnose.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT / "manifest.json")
    )
    diagnose.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "diagnostic")
    )

    status = subparsers.add_parser("status", help="inspect the signed firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument(
        "--diagnostic-receipt",
        default=str(DEFAULT_OUTPUT / "diagnostic" / "diagnostic_receipt.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        manifest = freeze_future_viability_seed_shift_diagnostic(
            output_path=args.manifest,
            hierarchy_manifest_path=args.hierarchy_manifest,
            hierarchy_evaluation_receipt_path=args.hierarchy_evaluation_receipt,
            conflict_manifest_path=args.conflict_manifest,
            conflict_diagnostic_receipt_path=args.conflict_diagnostic_receipt,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["firewall"]["diagnostic_authorized"] else 3
    if args.phase == "diagnose":
        receipt = run_future_viability_seed_shift_diagnostic(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = future_viability_seed_shift_diagnostic_status(
        manifest_path=args.manifest,
        diagnostic_receipt_path=args.diagnostic_receipt,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "build_parser", "main"]
