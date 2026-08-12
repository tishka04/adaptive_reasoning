"""CLI for the preregistered SAGE.T12.3e lineage-shield retest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .lineage_shield_experiment import (
    lineage_shield_experiment_status,
    run_lineage_shield_experiment,
)
from .lineage_shield_protocol import freeze_lineage_shield_experiment

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "lineage_shield_t12_3e_bp35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="seal the passed T12.3d parent and confirmed T12.3b shield evidence",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument(
        "--source-registry",
        default=str(DEFAULT_OUTPUT / "shield_inputs.sealed.json"),
    )
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="compare lineage archives without and with the terminal shield",
    )
    run.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "paired"))
    run.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="verify T12.3e artifacts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_lineage_shield_experiment(
                output_path=args.manifest,
                source_registry_path=args.source_registry,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "run":
            result = run_lineage_shield_experiment(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = lineage_shield_experiment_status(
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
            "format_version": "sage-t12.3e-lineage-shield-cli-error-v1",
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
