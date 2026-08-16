"""CLI for the offline SAGE.T12.5b.2 progress-discrimination audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .progress_discrimination_experiment import (
    progress_discrimination_status,
    run_progress_discrimination_audit,
)
from .progress_discrimination_protocol import freeze_progress_discrimination

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "training"
    / "sage_t"
    / "progress_discrimination_t12_5b_2_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="seal the negative T12.5b-r1 result and offline audit"
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    audit = subparsers.add_parser(
        "audit", help="derive local affordances and observed hard contrasts offline"
    )
    audit.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    audit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "audit"))

    status = subparsers.add_parser(
        "status", help="verify T12.5b.2 artifacts and diagnostic route"
    )
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_progress_discrimination(
                output_path=args.manifest,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "audit":
            result = run_progress_discrimination_audit(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = progress_discrimination_status(
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
            "format_version": "sage-t12.5b.2-discrimination-cli-error-v1",
            "phase": args.phase,
            "reason": f"{type(exc).__name__}:{exc}",
            "status": "FAILED_CLOSED",
        }
        exit_code = 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_OUTPUT", "build_parser", "main"]
