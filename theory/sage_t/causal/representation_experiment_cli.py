"""CLI for the preregistered SAGE.T12.4a representation repair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .representation_experiment import (
    collect_representation_experiment,
    representation_experiment_status,
    train_representation_experiment,
)
from .representation_protocol import freeze_representation_experiment

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "representation_t12_4a_bp35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="seal the representation-only T12.4 failure and prospective split",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    collect = subparsers.add_parser(
        "collect",
        help="collect shielded control archives on unopened source-train seeds",
    )
    collect.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    collect.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "collection"))
    collect.add_argument("--environments-dir", default="environment_files")

    train = subparsers.add_parser(
        "train",
        help="compare relational/context features with legacy and action-only controls",
    )
    train.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    train.add_argument(
        "--collection-receipt",
        default=str(DEFAULT_OUTPUT / "collection" / "collection_receipt.json"),
    )
    train.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "training"))

    status = subparsers.add_parser("status", help="verify T12.4a artifacts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_representation_experiment(
                output_path=args.manifest,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "collect":
            result = collect_representation_experiment(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "train":
            result = train_representation_experiment(
                manifest_path=args.manifest,
                collection_receipt_path=args.collection_receipt,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = representation_experiment_status(
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
            "format_version": "sage-t12.4a-representation-cli-error-v1",
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
