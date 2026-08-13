"""CLI for T12.4a.3 exhaustive option minimization and shadow compilation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .option_minimization_experiment import (
    compile_option_shadow,
    option_minimization_status,
    run_option_ablation,
)
from .option_minimization_protocol import freeze_option_minimization

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "training" / "sage_t" / "option_minimization_t12_4a_3_bp35"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="seal the two confirmed contexts and exhaustive 64-subsequence protocol",
    )
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-receipt", required=True)
    freeze.add_argument("--program-registry", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument("--allow-dirty", action="store_true")

    ablate = subparsers.add_parser(
        "ablate",
        help="run all exact-prefix subsequences and reversed controls",
    )
    ablate.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    ablate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "ablation"))
    ablate.add_argument("--environments-dir", default="environment_files")

    compile_shadow = subparsers.add_parser(
        "compile-shadow",
        help="compile a passed minimal option into the common posterior without control",
    )
    compile_shadow.add_argument(
        "--manifest",
        default=str(DEFAULT_OUTPUT / "manifest.json"),
    )
    compile_shadow.add_argument(
        "--ablation-receipt",
        default=str(DEFAULT_OUTPUT / "ablation" / "option_ablation_receipt.json"),
    )
    compile_shadow.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT / "shadow_compile"),
    )

    status = subparsers.add_parser("status", help="verify T12.4a.3 artifacts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipt")
    status.add_argument("--compile-receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_option_minimization(
                output_path=args.manifest,
                parent_manifest_path=args.parent_manifest,
                parent_receipt_path=args.parent_receipt,
                program_registry_path=args.program_registry,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "ablate":
            result = run_option_ablation(
                Path(args.manifest),
                output_dir=Path(args.output_dir),
                environments_dir=Path(args.environments_dir),
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "compile-shadow":
            result = compile_option_shadow(
                Path(args.manifest),
                ablation_receipt_path=Path(args.ablation_receipt),
                output_dir=Path(args.output_dir),
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = option_minimization_status(
                Path(args.manifest),
                receipt_path=(None if args.receipt is None else Path(args.receipt)),
                compile_receipt_path=(
                    None
                    if args.compile_receipt is None
                    else Path(args.compile_receipt)
                ),
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
            "format_version": "sage-t12.4a.3-option-minimization-cli-error-v1",
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
