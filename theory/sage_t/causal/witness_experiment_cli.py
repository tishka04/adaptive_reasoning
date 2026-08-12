"""CLI for the preregistered SAGE.T12.3a progress-witness experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .witness_experiment import (
    run_witness_confirmation,
    witness_experiment_status,
)
from .witness_protocol import freeze_witness_experiment

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "progress_witness_t12_3a_bp35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="seal the two T12.2 progress witnesses and T12.3a protocol"
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument(
        "--witness-registry",
        default=str(DEFAULT_OUTPUT / "witnesses.sealed.json"),
    )
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    run = subparsers.add_parser(
        "run", help="replay each witness and its suffix/deletion control three times"
    )
    run.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "confirmation"))
    run.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="verify T12.3a manifest and receipt")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_witness_experiment(
                output_path=args.manifest,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                witness_registry_path=args.witness_registry,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "run":
            result = run_witness_confirmation(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = witness_experiment_status(
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
            "format_version": "sage-t12.3a-witness-cli-error-v1",
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
