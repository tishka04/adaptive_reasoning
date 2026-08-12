"""CLI for the preregistered SAGE.T12.1 graph-exploration experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .graph_experiment import (
    compile_option_phase,
    extract_option_phase,
    graph_experiment_status,
    run_archive_phase,
    run_neural_phase,
    run_shield_phase,
    run_transfer_phase,
    train_novelty_phase,
)
from .graph_protocol import freeze_graph_experiment

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "graph_explore_t12_1"


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="freeze games, gates, code hashes and the 3 GiB cap"
    )
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument(
        "--stage", choices=("source_train", "regression", "historical"), required=True
    )
    freeze.add_argument("--games", required=True, help="comma-separated game ids")
    freeze.add_argument("--program-registry")
    freeze.add_argument("--allow-dirty", action="store_true")

    archive = subparsers.add_parser(
        "archive", help="run paired historical baseline and pure symbolic Go-Explore"
    )
    archive.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    archive.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "archive"))
    archive.add_argument("--environments-dir", default="environment_files")

    shield = subparsers.add_parser(
        "shield", help="test the frozen multi-step terminal shield"
    )
    shield.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    shield.add_argument(
        "--archive-receipt",
        default=str(DEFAULT_OUTPUT / "archive" / "archive_receipt.json"),
    )
    shield.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "shield"))
    shield.add_argument("--environments-dir", default="environment_files")

    novelty = subparsers.add_parser(
        "train-novelty", help="fit the small symbolic action change/novelty MLP"
    )
    novelty.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    novelty.add_argument(
        "--shield-receipt",
        default=str(DEFAULT_OUTPUT / "shield" / "shield_receipt.json"),
    )
    novelty.add_argument(
        "--archive-receipt",
        default=str(DEFAULT_OUTPUT / "archive" / "archive_receipt.json"),
    )
    novelty.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "novelty"))

    neural = subparsers.add_parser(
        "neural", help="test neural ordering against the shielded symbolic archive"
    )
    neural.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    neural.add_argument(
        "--novelty-receipt",
        default=str(DEFAULT_OUTPUT / "novelty" / "novelty_receipt.json"),
    )
    neural.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "neural"))
    neural.add_argument("--environments-dir", default="environment_files")

    option = subparsers.add_parser(
        "extract-option", help="minimize the first archived progression by exact replay"
    )
    option.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    option.add_argument(
        "--archive-receipt",
        default=str(DEFAULT_OUTPUT / "archive" / "archive_receipt.json"),
    )
    option.add_argument(
        "--parent-receipt",
        default=str(DEFAULT_OUTPUT / "neural" / "neural_ordering_receipt.json"),
    )
    option.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "option"))
    option.add_argument("--environments-dir", default="environment_files")

    compile_option = subparsers.add_parser(
        "compile-option", help="compile the option into complete causal particles"
    )
    compile_option.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT / "manifest.json")
    )
    compile_option.add_argument(
        "--option-receipt",
        default=str(DEFAULT_OUTPUT / "option" / "option_receipt.json"),
    )
    compile_option.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT / "compiled_option")
    )

    transfer = subparsers.add_parser(
        "transfer", help="run exact-entry paired transfer to the next three levels"
    )
    transfer.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    transfer.add_argument(
        "--compilation-receipt",
        default=str(
            DEFAULT_OUTPUT / "compiled_option" / "option_compilation_receipt.json"
        ),
    )
    transfer.add_argument(
        "--archive-receipt",
        default=str(DEFAULT_OUTPUT / "archive" / "archive_receipt.json"),
    )
    transfer.add_argument(
        "--novelty-receipt",
        default=str(DEFAULT_OUTPUT / "novelty" / "novelty_receipt.json"),
    )
    transfer.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "transfer"))
    transfer.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="verify the manifest and receipt chain")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--receipts", nargs="*", default=())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "freeze":
            result = freeze_graph_experiment(
                output_path=args.manifest,
                stage=args.stage,
                game_ids=_csv(args.games),
                program_registry_path=args.program_registry,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "archive":
            result = run_archive_phase(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "shield":
            result = run_shield_phase(
                manifest_path=args.manifest,
                archive_receipt_path=args.archive_receipt,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "train-novelty":
            result = train_novelty_phase(
                manifest_path=args.manifest,
                shield_receipt_path=args.shield_receipt,
                archive_receipt_path=args.archive_receipt,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "neural":
            result = run_neural_phase(
                manifest_path=args.manifest,
                novelty_receipt_path=args.novelty_receipt,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "extract-option":
            result = extract_option_phase(
                manifest_path=args.manifest,
                archive_receipt_path=args.archive_receipt,
                parent_receipt_path=args.parent_receipt,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "compile-option":
            result = compile_option_phase(
                manifest_path=args.manifest,
                option_receipt_path=args.option_receipt,
                output_dir=args.output_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "transfer":
            result = run_transfer_phase(
                manifest_path=args.manifest,
                compilation_receipt_path=args.compilation_receipt,
                archive_receipt_path=args.archive_receipt,
                novelty_receipt_path=args.novelty_receipt,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = graph_experiment_status(
                manifest_path=args.manifest,
                receipt_paths=args.receipts,
            )
            exit_code = 0
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        result = {
            "format_version": "sage-t12.1-graph-cli-error-v1",
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
