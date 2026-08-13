"""CLI for T12.4a.4d.1 hazard abstraction and diverse paired search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .hazard_diversity_experiment import (
    compile_hazard_diversity,
    hazard_diversity_status,
    run_hazard_diversity_experiment,
)
from .hazard_diversity_protocol import freeze_hazard_diversity

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "training"
    / "sage_t"
    / "hazard_diversity_t12_4a_4d_1_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)

    freeze = phases.add_parser(
        "freeze",
        help="seal the failed parent evidence, cross-fit and prospective design",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    compile_phase = phases.add_parser(
        "compile",
        help="cross-fit and seal the identity-free terminal-hazard model",
    )
    compile_phase.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT / "manifest.json")
    )
    compile_phase.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "compile")
    )

    run = phases.add_parser(
        "run",
        help="execute prospective three-arm paired search on fresh seeds",
    )
    run.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    run.add_argument(
        "--compile-receipt",
        default=str(DEFAULT_OUTPUT / "compile" / "compile_receipt.json"),
    )
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "paired"))
    run.add_argument("--environments-dir", default="environment_files")

    status = phases.add_parser("status", help="verify the frozen lineage and receipts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--compile-receipt")
    status.add_argument("--active-receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_hazard_diversity(
                output_path=args.manifest,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "compile":
            result = compile_hazard_diversity(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "run":
            result = run_hazard_diversity_experiment(
                manifest_path=args.manifest,
                compile_receipt_path=args.compile_receipt,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = hazard_diversity_status(
                manifest_path=args.manifest,
                compile_receipt_path=args.compile_receipt,
                active_receipt_path=args.active_receipt,
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
            "format_version": "sage-t12.4a.4d.1-hazard-diversity-cli-error-v1",
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
