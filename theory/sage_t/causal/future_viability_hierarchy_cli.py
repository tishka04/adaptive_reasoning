"""CLI for the fail-closed SAGE.T12.6.1 hierarchy experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .future_viability_hierarchy_experiment import (
    compile_future_viability_hierarchy,
    evaluate_future_viability_hierarchy,
    future_viability_hierarchy_status,
)
from .future_viability_hierarchy_protocol import (
    freeze_future_viability_hierarchy,
)

DEFAULT_OUTPUT = (
    Path("training") / "sage_t" / "future_viability_hierarchy_t12_6_1_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze hierarchy and gates")
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-compile-receipt", required=True)
    freeze.add_argument("--diagnostic-manifest", required=True)
    freeze.add_argument("--diagnostic-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))

    compile_parser = subparsers.add_parser(
        "compile", help="cross-fit hierarchy on development archives"
    )
    compile_parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT / "manifest.json")
    )
    compile_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "compile")
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="open sealed evaluation only after compile passes"
    )
    evaluate.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    evaluate.add_argument(
        "--compile-receipt",
        default=str(DEFAULT_OUTPUT / "compile" / "compile_receipt.json"),
    )
    evaluate.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "evaluation")
    )

    status = subparsers.add_parser("status", help="inspect signed firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument(
        "--compile-receipt",
        default=str(DEFAULT_OUTPUT / "compile" / "compile_receipt.json"),
    )
    status.add_argument(
        "--evaluation-receipt",
        default=str(DEFAULT_OUTPUT / "evaluation" / "evaluation_receipt.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        manifest = freeze_future_viability_hierarchy(
            output_path=args.manifest,
            parent_manifest_path=args.parent_manifest,
            parent_compile_receipt_path=args.parent_compile_receipt,
            diagnostic_manifest_path=args.diagnostic_manifest,
            diagnostic_receipt_path=args.diagnostic_receipt,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["firewall"]["compile_authorized"] else 3
    if args.phase == "compile":
        receipt = compile_future_viability_hierarchy(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    if args.phase == "evaluate":
        receipt = evaluate_future_viability_hierarchy(
            manifest_path=args.manifest,
            compile_receipt_path=args.compile_receipt,
            output_dir=args.output_dir,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = future_viability_hierarchy_status(
        manifest_path=args.manifest,
        compile_receipt_path=args.compile_receipt,
        evaluation_receipt_path=args.evaluation_receipt,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "build_parser", "main"]
