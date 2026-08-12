"""Deterministic source-train inputs for the SAGE.T11.1 bp35 iteration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    ActionInterventionSpec,
    BindingSpec,
    CausalProgram,
    CausalVariableSpec,
    GoalSpec,
    MechanismSpec,
    ObservationModelSpec,
    ParentRef,
)

ACTION_CATALOG = ("ACTION3", "ACTION4", "ACTION6")
PLAYER_CENTER = "summary.role.player.center"
LEVELS_COMPLETED = "counter.levels_completed"
GAME_OVER = "fact.game_over"


def _program(
    program_id: str,
    *,
    position_parameters: Mapping[str, Any],
    progress_predicate: str,
    description_length: float,
) -> CausalProgram:
    return CausalProgram(
        program_id=program_id,
        bindings=BindingSpec(
            {
                "player": "role:player",
                "level_progress": LEVELS_COMPLETED,
            }
        ),
        variables=(
            CausalVariableSpec(PLAYER_CENTER, "position"),
            CausalVariableSpec(LEVELS_COMPLETED, "counter"),
            CausalVariableSpec(GAME_OVER, "terminal", (False, True)),
        ),
        mechanisms=(
            MechanismSpec(
                f"{program_id}_player_transition",
                PLAYER_CENTER,
                (ParentRef(PLAYER_CENTER),),
                "action_position",
                dict(position_parameters),
            ),
            MechanismSpec(
                f"{program_id}_level_persistence",
                LEVELS_COMPLETED,
                (ParentRef(LEVELS_COMPLETED),),
                "identity",
            ),
            MechanismSpec(
                f"{program_id}_failure_persistence",
                GAME_OVER,
                (ParentRef(GAME_OVER),),
                "identity",
            ),
        ),
        action_model=tuple(
            ActionInterventionSpec(action_name) for action_name in ACTION_CATALOG
        ),
        goal=GoalSpec(
            f"{LEVELS_COMPLETED} >= 1",
            (progress_predicate,),
            f"{GAME_OVER} == true",
        ),
        observation_model=ObservationModelSpec(
            channels=("variables", "terminal", "goal", "progress"),
            channel_weights={
                "variables": 3.0,
                "terminal": 4.0,
                "goal": 2.0,
                "progress": 2.0,
            },
            noise_floor=0.05,
        ),
        description_length=description_length,
        provenance=(
            "source:bp35-source-train",
            "iteration:sage-t11.1-action-local",
            f"hypothesis:{program_id}",
        ),
    )


def programs() -> tuple[CausalProgram, ...]:
    left_goal = f"{PLAYER_CENTER} == [38.5,14]"
    right_goal = f"{PLAYER_CENTER} == [38.5,28]"
    return (
        _program(
            "bp35_player_persistence_left_goal",
            position_parameters={},
            progress_predicate=left_goal,
            description_length=5.0,
        ),
        _program(
            "bp35_source_columns_left_goal",
            position_parameters={
                "columns_by_action": {"ACTION3": 14, "ACTION4": 28}
            },
            progress_predicate=left_goal,
            description_length=6.0,
        ),
        _program(
            "bp35_source_columns_right_goal",
            position_parameters={
                "columns_by_action": {"ACTION3": 14, "ACTION4": 28}
            },
            progress_predicate=right_goal,
            description_length=6.0,
        ),
        _program(
            "bp35_reversed_columns_right_goal",
            position_parameters={
                "columns_by_action": {"ACTION3": 28, "ACTION4": 14}
            },
            progress_predicate=right_goal,
            description_length=7.0,
        ),
        _program(
            "bp35_source_displacement_left_goal",
            position_parameters={
                "deltas_by_action": {"ACTION3": [0, -8], "ACTION4": [0, 6]}
            },
            progress_predicate=left_goal,
            description_length=7.0,
        ),
        _program(
            "bp35_reversed_displacement_right_goal",
            position_parameters={
                "deltas_by_action": {"ACTION3": [0, 8], "ACTION4": [0, -6]}
            },
            progress_predicate=right_goal,
            description_length=8.0,
        ),
        _program(
            "bp35_click_grounding_left_goal",
            position_parameters={
                "ground_action": "ACTION6",
                "row_key": "y",
                "column_key": "x",
            },
            progress_predicate=left_goal,
            description_length=8.0,
        ),
        _program(
            "bp35_source_columns_click_grounding_right_goal",
            position_parameters={
                "columns_by_action": {"ACTION3": 14, "ACTION4": 28},
                "ground_action": "ACTION6",
                "row_key": "y",
                "column_key": "x",
            },
            progress_predicate=right_goal,
            description_length=9.0,
        ),
    )


def raw_program_registry() -> dict[str, Any]:
    return {
        "design_metadata": {
            "created_at": "2026-08-12",
            "source_split": "source_train",
            "game_build": "bp35-0a0ad940",
            "branch_outcomes_observed": False,
            "hypothesis_scope": "action-conditioned player-role dynamics with joint subgoals",
            "source_train_prior": {
                "reset_player_center": [38.5, 22.0],
                "action3_player_center": [38.5, 14.0],
                "action4_player_center": [38.5, 28.0],
                "action6_30_12_player_center": [38.5, 22.0],
            },
            "notes": [
                "Source-train one-step observations are priors, not validation evidence.",
                "Every particle jointly encodes dynamics, progress, failure, and final goal.",
                "ACTION7 remains excluded because no grounded candidate is exposed by _valid_actions.",
            ],
            "provenance": [
                "reports/SAGE_T11_1_ACTION_LOCAL_3GIB_PROTOCOL.md",
                "training/sage_t/causal_bp35_v1/paired/paired_report.json",
                "theory/sage_t/causal/bp35_iteration_v2.py",
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
            "branch_policy": "ACTION3/ACTION4 controls versus grounded ACTION6 click",
            "prefixes_reused_from": str(Path(previous_bundle_path).as_posix()),
            "prefix_replay_repetitions": 3,
            "prefixes_exact_on_all_previous_repetitions": True,
            "provenance": [
                "reports/SAGE_T11_1_ACTION_LOCAL_3GIB_PROTOCOL.md",
                "training/sage_t/causal_inputs/bundles.raw.json",
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
        default="training/sage_t/causal_inputs/bundles.raw.json",
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


__all__ = ["ACTION_CATALOG", "main", "programs", "raw_bundle_plan", "raw_program_registry"]
