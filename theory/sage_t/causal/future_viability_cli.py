"""CLI for the zero-SDK SAGE.T12.6 future-viability audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .future_viability_experiment import (
    compile_future_viability,
    evaluate_future_viability,
    future_viability_status,
)
from .future_viability_protocol import freeze_future_viability

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "future_viability_t12_6_bp35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze the chronological audit")
    freeze.add_argument("--authority-manifest", required=True)
    freeze.add_argument("--authority-receipt", required=True)
    freeze.add_argument("--training-manifest", required=True)
    freeze.add_argument("--training-receipt", required=True)
    freeze.add_argument("--evaluation-manifest", required=True)
    freeze.add_argument("--evaluation-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    compile_parser = subparsers.add_parser(
        "compile", help="cross-fit and seal models on archives 9101-9103"
    )
    compile_parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT / "manifest.json")
    )
    compile_parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "compile")
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate frozen models on later archives 9201-9203"
    )
    evaluate.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    evaluate.add_argument(
        "--compile-receipt",
        default=str(DEFAULT_OUTPUT / "compile" / "compile_receipt.json"),
    )
    evaluate.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "evaluation")
    )

    status = subparsers.add_parser("status", help="inspect the signed firewall")
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
        manifest = freeze_future_viability(
            output_path=args.manifest,
            authority_manifest_path=args.authority_manifest,
            authority_receipt_path=args.authority_receipt,
            training_manifest_path=args.training_manifest,
            training_receipt_path=args.training_receipt,
            evaluation_manifest_path=args.evaluation_manifest,
            evaluation_receipt_path=args.evaluation_receipt,
            root=DEFAULT_ROOT,
            allow_dirty=bool(args.allow_dirty),
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["scientific_claims_authorized"] else 3
    if args.phase == "compile":
        receipt = compile_future_viability(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            root=DEFAULT_ROOT,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    if args.phase == "evaluate":
        receipt = evaluate_future_viability(
            manifest_path=args.manifest,
            compile_receipt_path=args.compile_receipt,
            output_dir=args.output_dir,
            root=DEFAULT_ROOT,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["passed"] else 3
    status = future_viability_status(
        manifest_path=args.manifest,
        compile_receipt_path=args.compile_receipt,
        evaluation_receipt_path=args.evaluation_receipt,
        root=DEFAULT_ROOT,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
