"""CLI for the preregistered offline SAGE.T12.5 causal-progress gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .progress_experiment import causal_progress_status, compile_causal_progress
from .progress_protocol import freeze_causal_progress

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "causal_progress_t12_5_bp35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="seal T12.4a.4c typed effects and T12.4a.3 order evidence"
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--ablation-receipt", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    compile_phase = subparsers.add_parser(
        "compile", help="fit rival progress automata and replicate them offline"
    )
    compile_phase.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT / "manifest.json")
    )
    compile_phase.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "compiled")
    )

    status = subparsers.add_parser("status", help="verify T12.5 artifacts and gates")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_causal_progress(
                output_path=args.manifest,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                ablation_receipt_path=args.ablation_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "compile":
            result = compile_causal_progress(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = causal_progress_status(
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
            "format_version": "sage-t12.5-causal-progress-cli-error-v1",
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
