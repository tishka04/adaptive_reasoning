"""Command-line entry point for preregistered SAGE.T causal experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .experiment import (
    DEFAULT_ARMS,
    experiment_status,
    freeze_experiment,
    run_experiment,
    run_replay,
    seal_bundle_plan,
    seal_program_registry,
)

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_ROOT / "training" / "sage_t" / "causal_experiment_v1"


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    programs = subparsers.add_parser(
        "seal-programs",
        help="validate and sign complete rival programs",
    )
    programs.add_argument("--input", required=True)
    programs.add_argument("--output", required=True)

    bundles = subparsers.add_parser(
        "seal-bundles",
        help="validate and sign exact-prefix intervention plans",
    )
    bundles.add_argument("--input", required=True)
    bundles.add_argument("--program-registry", required=True)
    bundles.add_argument("--output", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="freeze paired games, seeds, arms, budgets and checksums",
    )
    freeze.add_argument("--program-registry", required=True)
    freeze.add_argument("--bundle-plan", required=True)
    freeze.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    freeze.add_argument(
        "--stage",
        choices=("source_train", "source_validation", "historical", "regression"),
        required=True,
    )
    freeze.add_argument("--games", required=True, help="comma-separated game ids")
    freeze.add_argument("--seeds", default="0,1,2")
    freeze.add_argument("--resets", type=int, default=3)
    freeze.add_argument("--action-budget", type=int, default=64)
    freeze.add_argument("--authority", choices=("shadow", "bounded"), default="shadow")
    freeze.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    freeze.add_argument("--parent-receipt")
    freeze.add_argument(
        "--allow-dirty",
        action="store_true",
        help="create a smoke-only manifest that cannot pass a scientific gate",
    )

    replay = subparsers.add_parser(
        "replay",
        help="execute preregistered exact-prefix intervention bundles",
    )
    replay.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    replay.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "replay"))
    replay.add_argument("--environments-dir", default="environment_files")

    run = subparsers.add_parser(
        "run",
        help="execute strictly paired baseline, posterior and ablation arms",
    )
    run.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    run.add_argument(
        "--replay-receipt",
        default=str(DEFAULT_OUTPUT / "replay" / "replay_receipt.json"),
    )
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT / "paired"))
    run.add_argument("--environments-dir", default="environment_files")

    status = subparsers.add_parser("status", help="verify manifest and receipts")
    status.add_argument("--manifest", default=str(DEFAULT_OUTPUT / "manifest.json"))
    status.add_argument("--replay-receipt")
    status.add_argument("--gate-receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.phase == "seal-programs":
            result = seal_program_registry(args.input, args.output)
            exit_code = 0
        elif args.phase == "seal-bundles":
            result = seal_bundle_plan(
                args.input,
                args.program_registry,
                args.output,
            )
            exit_code = 0
        elif args.phase == "freeze":
            result = freeze_experiment(
                program_registry_path=args.program_registry,
                bundle_plan_path=args.bundle_plan,
                output_path=args.manifest,
                stage=args.stage,
                game_ids=_csv(args.games),
                seeds=_ints(args.seeds),
                resets=args.resets,
                action_budget_per_reset=args.action_budget,
                authority=args.authority,
                arms=_csv(args.arms),
                parent_receipt_path=args.parent_receipt,
                allow_dirty=args.allow_dirty,
            )
            exit_code = 0 if result["scientific_claims_authorized"] else 3
        elif args.phase == "replay":
            result = run_replay(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        elif args.phase == "run":
            result = run_experiment(
                manifest_path=args.manifest,
                replay_receipt_path=args.replay_receipt,
                output_dir=args.output_dir,
                environments_dir=args.environments_dir,
            )
            exit_code = 0 if result["passed"] else 3
        else:
            result = experiment_status(
                manifest_path=args.manifest,
                replay_receipt_path=args.replay_receipt,
                gate_receipt_path=args.gate_receipt,
            )
            exit_code = 0
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        result = {
            "format_version": "sage-t11-causal-cli-error-v1",
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
