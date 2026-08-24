"""CLI for the fail-closed SAGE.T12.6.1c source-train experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .future_viability_reliability_hierarchy_experiment import (
    compile_future_viability_reliability_hierarchy,
    future_viability_reliability_status,
)
from .future_viability_reliability_hierarchy_protocol import (
    freeze_future_viability_reliability_hierarchy,
)

DEFAULT_OUTPUT = (
    Path("training")
    / "sage_t"
    / "future_viability_reliability_t12_6_1c_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze source-train candidates")
    freeze.add_argument("--hierarchy-manifest", required=True)
    freeze.add_argument("--hierarchy-compile-receipt", required=True)
    freeze.add_argument("--seed-shift-diagnostic-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))

    compile_parser = subparsers.add_parser(
        "compile", help="select and compile from training archives only"
    )
    compile_parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT / "manifest.json")
    )
    compile_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "compile")
    )

    status = subparsers.add_parser("status", help="inspect the signed firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument(
        "--compile-receipt",
        default=str(DEFAULT_OUTPUT / "compile" / "compile_receipt.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        manifest = freeze_future_viability_reliability_hierarchy(
            output_path=args.manifest,
            hierarchy_manifest_path=args.hierarchy_manifest,
            hierarchy_compile_receipt_path=args.hierarchy_compile_receipt,
            seed_shift_diagnostic_receipt_path=(
                args.seed_shift_diagnostic_receipt
            ),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["firewall"]["compile_authorized"] else 3
    if args.phase == "compile":
        receipt = compile_future_viability_reliability_hierarchy(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = future_viability_reliability_status(
        manifest_path=args.manifest,
        compile_receipt_path=args.compile_receipt,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "build_parser", "main"]
