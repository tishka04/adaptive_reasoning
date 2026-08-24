"""Fail-closed CLI for SAGE.T12.6.1d prospective confirmation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .future_viability_prospective_experiment import (
    adjudicate_future_viability_confirmation,
    collect_future_viability_batch,
    commit_future_viability_predictions,
    future_viability_prospective_status,
    preflight_future_viability_confirmation,
    seal_future_viability_collection,
)
from .future_viability_prospective_protocol import (
    freeze_future_viability_prospective_confirmation,
)

DEFAULT_OUTPUT = (
    Path("training") / "sage_t" / "future_viability_confirmation_t12_6_1d_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    freeze = phases.add_parser("freeze", help="freeze the fresh-seed protocol")
    freeze.add_argument("--reliability-manifest", required=True)
    freeze.add_argument("--reliability-compile-receipt", required=True)
    freeze.add_argument("--hazard-manifest", required=True)
    freeze.add_argument("--hazard-compile-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))

    preflight = phases.add_parser("preflight", help="offline collection preflight")
    preflight.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    preflight.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "preflight"))
    preflight.add_argument("--environments-dir", default="environment_files")

    collect = phases.add_parser(
        "collect-batch", help="physically collect one preregistered batch"
    )
    collect.add_argument("--batch", choices=("pilot", "completion"), required=True)
    collect.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    collect.add_argument(
        "--preflight-receipt",
        default=str(DEFAULT_OUTPUT / "preflight" / "preflight_receipt.json"),
    )
    collect.add_argument("--pilot-receipt")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--environments-dir", default="environment_files")

    seal = phases.add_parser("seal-collection", help="seal both label-blind batches")
    seal.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    seal.add_argument("--pilot-receipt", required=True)
    seal.add_argument("--completion-receipt", required=True)
    seal.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "collection" / "sealed")
    )

    predict = phases.add_parser("predict", help="commit scores before opening labels")
    predict.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    predict.add_argument("--collection-seal-receipt", required=True)
    predict.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "prediction"))

    adjudicate = phases.add_parser(
        "adjudicate", help="open exact-state labels and apply frozen gates"
    )
    adjudicate.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    adjudicate.add_argument("--collection-seal-receipt", required=True)
    adjudicate.add_argument("--prediction-receipt", required=True)
    adjudicate.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "adjudication")
    )

    status = phases.add_parser("status", help="inspect receipts and phase firewall")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--preflight-receipt")
    status.add_argument("--pilot-receipt")
    status.add_argument("--completion-receipt")
    status.add_argument("--collection-seal-receipt")
    status.add_argument("--prediction-receipt")
    status.add_argument("--adjudication-receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "freeze":
            result = freeze_future_viability_prospective_confirmation(
                output_path=args.manifest,
                reliability_manifest_path=args.reliability_manifest,
                reliability_compile_receipt_path=args.reliability_compile_receipt,
                hazard_manifest_path=args.hazard_manifest,
                hazard_compile_receipt_path=args.hazard_compile_receipt,
            )
            exit_code = 0 if result["firewall"]["preflight_authorized"] else 3
        elif args.phase == "preflight":
            result = preflight_future_viability_confirmation(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "collect-batch":
            result = collect_future_viability_batch(
                manifest_path=args.manifest,
                preflight_receipt_path=args.preflight_receipt,
                output_dir=args.output_dir,
                batch=args.batch,
                pilot_receipt_path=args.pilot_receipt,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "seal-collection":
            result = seal_future_viability_collection(
                manifest_path=args.manifest,
                pilot_receipt_path=args.pilot_receipt,
                completion_receipt_path=args.completion_receipt,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "predict":
            result = commit_future_viability_predictions(
                manifest_path=args.manifest,
                collection_seal_receipt_path=args.collection_seal_receipt,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "adjudicate":
            result = adjudicate_future_viability_confirmation(
                manifest_path=args.manifest,
                collection_seal_receipt_path=args.collection_seal_receipt,
                prediction_receipt_path=args.prediction_receipt,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = future_viability_prospective_status(
                manifest_path=args.manifest,
                preflight_receipt_path=args.preflight_receipt,
                pilot_receipt_path=args.pilot_receipt,
                completion_receipt_path=args.completion_receipt,
                collection_seal_receipt_path=args.collection_seal_receipt,
                prediction_receipt_path=args.prediction_receipt,
                adjudication_receipt_path=args.adjudication_receipt,
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
            "format_version": "sage-t12.6.1d-prospective-cli-error-v1",
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
