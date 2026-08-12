"""Deterministic source-train inputs for the SAGE.T11.2 bp35 iteration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .bp35_iteration_v2 import ACTION_CATALOG, programs as v2_programs
from .contracts import CausalProgram


def programs() -> tuple[CausalProgram, ...]:
    return tuple(
        replace(
            program,
            provenance=(
                "source:bp35-source-train",
                "iteration:sage-t11.2-replay-prior-progress-witness",
                f"hypothesis:{program.program_id}",
            ),
        )
        for program in v2_programs()
    )


def raw_program_registry() -> dict[str, Any]:
    return {
        "design_metadata": {
            "created_at": "2026-08-12",
            "source_split": "source_train",
            "game_build": "bp35-0a0ad940",
            "branch_outcomes_observed": False,
            "hypothesis_scope": (
                "action-conditioned player dynamics with program-local progress witnesses"
            ),
            "mechanism_change": (
                "same structural rivals as T11.1; corrected local witness semantics "
                "and checksum-bound replay-to-control evidence"
            ),
            "provenance": [
                "reports/SAGE_T11_2_REPLAY_PRIOR_PROGRESS_WITNESS_PROTOCOL.md",
                "training/sage_t/causal_bp35_v2/paired/paired_report.json",
                "theory/sage_t/causal/bp35_iteration_v3.py",
            ],
        },
        "games": {
            "bp35": {
                "action_catalog": list(ACTION_CATALOG),
                "catalog_basis": "Grounded candidates exposed by _valid_actions.",
                "programs": [program.to_dict() for program in programs()],
            }
        },
    }


def raw_bundle_plan(previous_bundle_path: str | Path) -> dict[str, Any]:
    previous = json.loads(Path(previous_bundle_path).read_text(encoding="utf-8"))
    return {
        "design_metadata": {
            "created_at": "2026-08-12",
            "source_split": "source_train",
            "game_build": "bp35-0a0ad940",
            "branch_outcomes_observed": False,
            "branch_policy": "unchanged exact T11.1 ACTION3/ACTION4/ACTION6 branches",
            "prefixes_reused_from": str(Path(previous_bundle_path).as_posix()),
            "prefixes_exact_on_previous_replay": True,
            "provenance": [
                "reports/SAGE_T11_2_REPLAY_PRIOR_PROGRESS_WITNESS_PROTOCOL.md",
                "training/sage_t/causal_inputs_v2/bundles.raw.json",
            ],
        },
        "bundles": list(previous["bundles"]),
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite causal input: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--previous-bundles",
        default="training/sage_t/causal_inputs_v2/bundles.raw.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.output_dir)
    _write(output / "programs.raw.json", raw_program_registry())
    _write(output / "bundles.raw.json", raw_bundle_plan(args.previous_bundles))
    print(
        json.dumps(
            {
                "programs": str(output / "programs.raw.json"),
                "bundles": str(output / "bundles.raw.json"),
                "program_count": len(programs()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["programs", "raw_bundle_plan", "raw_program_registry"]
