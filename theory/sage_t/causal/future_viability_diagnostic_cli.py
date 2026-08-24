"""Command line interface for the offline SAGE.T12.6a diagnostic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .future_viability_diagnostic_experiment import (
    future_viability_diagnostic_status,
    run_future_viability_diagnostic,
)
from .future_viability_diagnostic_protocol import (
    freeze_future_viability_diagnostic,
)

DEFAULT_OUTPUT = (
    Path("training") / "sage_t" / "future_viability_diagnostic_t12_6a_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="freeze post-hoc axes before reading individual errors"
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-compile-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))

    diagnose = subparsers.add_parser(
        "diagnose", help="inspect only the frozen T12.6 training fold miss"
    )
    diagnose.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    diagnose.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "diagnostic")
    )

    status = subparsers.add_parser("status", help="inspect the diagnostic firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument(
        "--diagnostic-receipt",
        default=str(DEFAULT_OUTPUT / "diagnostic" / "diagnostic_receipt.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        manifest = freeze_future_viability_diagnostic(
            output_path=args.manifest,
            parent_manifest_path=args.parent_manifest,
            parent_compile_receipt_path=args.parent_compile_receipt,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["firewall"]["diagnostic_authorized"] else 3
    if args.phase == "diagnose":
        receipt = run_future_viability_diagnostic(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = future_viability_diagnostic_status(
        manifest_path=args.manifest,
        diagnostic_receipt_path=args.diagnostic_receipt,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "build_parser", "main"]
