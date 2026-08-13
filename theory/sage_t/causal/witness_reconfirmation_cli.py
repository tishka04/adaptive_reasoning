"""CLI for preregistered SAGE.T12.4a.2 progress-witness reconfirmation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .witness_reconfirmation_experiment import (
    run_witness_reconfirmation,
    witness_reconfirmation_status,
)
from .witness_reconfirmation_protocol import freeze_witness_reconfirmation

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "training" / "sage_t" / "witness_reconfirmation_t12_4a_2_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="seal the two T12.4a.1 progress routes and paired replay protocol",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--collection-receipt", required=True)
    freeze.add_argument(
        "--witness-registry",
        default=str(DEFAULT_OUTPUT / "witnesses.sealed.json"),
    )
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="replay routes and paired suffix/deletion branches three times",
    )
    run.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "confirmation"))
    run.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="verify T12.4a.2 artifacts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_witness_reconfirmation(
                output_path=args.manifest,
                witness_registry_path=args.witness_registry,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                collection_receipt_path=args.collection_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "run":
            result = run_witness_reconfirmation(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = witness_reconfirmation_status(
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
            "format_version": "sage-t12.4a.2-witness-cli-error-v1",
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
