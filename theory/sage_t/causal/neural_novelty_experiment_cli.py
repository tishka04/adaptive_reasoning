"""CLI for the preregistered SAGE.T12.4 neural novelty experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .neural_novelty_experiment import (
    evaluate_neural_novelty_experiment,
    neural_novelty_experiment_status,
    train_neural_novelty_experiment,
)
from .neural_novelty_protocol import freeze_neural_novelty_experiment

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "neural_novelty_t12_4_bp35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="seal T12.3e ancestry and compile audited neural labels",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument(
        "--dataset",
        default=str(DEFAULT_OUTPUT / "dataset.sealed.json"),
    )
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    train = subparsers.add_parser(
        "train",
        help="fit and validate the small frozen symbolic state/action MLP",
    )
    train.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    train.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "training"))

    evaluate = subparsers.add_parser(
        "evaluate",
        help="compare the frozen shield with and without neural action scoring",
    )
    evaluate.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    evaluate.add_argument(
        "--training-receipt",
        default=str(DEFAULT_OUTPUT / "training" / "training_receipt.json"),
    )
    evaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "paired"))
    evaluate.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="verify T12.4 artifacts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_neural_novelty_experiment(
                output_path=args.manifest,
                dataset_path=args.dataset,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "train":
            result = train_neural_novelty_experiment(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "evaluate":
            result = evaluate_neural_novelty_experiment(
                manifest_path=args.manifest,
                training_receipt_path=args.training_receipt,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = neural_novelty_experiment_status(
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
            "format_version": "sage-t12.4-neural-novelty-cli-error-v1",
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
