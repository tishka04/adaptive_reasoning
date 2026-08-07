from __future__ import annotations

import json
from pathlib import Path

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t import calibration_gate_v8_6i as v86i
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.live_shadow_pilot_v7 import (
    SELECTED_POLICY,
    StructuralGoalFragmentProposer,
    StructuralLiveController,
    freeze_confirmation_manifest,
    load_confirmation_manifest,
)
from theory.sage_t.posterior_v8 import (
    MinimumKLFamilyFloorProgramPosterior,
)
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor
from theory.sage_t.synthesis import ProgramAssembler


def _goal_sequences() -> list[dict[str, object]]:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    return [
        sequence
        for sequence in v86._signal_sequences(pairs)
        if sequence["positive_kind"] == "goal"
    ]


def test_live_proposer_matches_frozen_offline_generator() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    generator = manifest["generator"]
    proposer = StructuralGoalFragmentProposer()
    assembler = ProgramAssembler(
        maximum_programs=int(generator["maximum_programs"]),
        maximum_dynamics_beam=int(generator["maximum_dynamics_beam"]),
    )
    for sequence in _goal_sequences():
        revealed = [
            next(arm for arm in panel.arms if arm.action.key == key)
            for panel, key in zip(sequence["panels"], sequence["keys"])
        ]
        actions = tuple(
            sorted(
                {
                    arm.action.action_name
                    for panel in sequence["panels"]
                    for arm in panel.arms
                }
            )
        )
        proposal = proposer.propose(
            available_actions=actions,
            transitions=revealed,
        )
        live = assembler.assemble(
            proposal.fragments,
            available_actions=actions,
        )
        offline = programs_for_with_structural_goal_guard(
            actions,
            revealed,
            manifest,
        )

        assert live == offline


def test_confirmation_manifest_binds_passing_selection_and_stays_shadow(
    tmp_path: Path,
) -> None:
    path = tmp_path / "confirmation.json"

    frozen = freeze_confirmation_manifest(output_path=path)
    loaded, report = load_confirmation_manifest(path)

    assert loaded == frozen
    assert report["status"] == "READY_FOR_T8_6I_LIVE_CONFIRMATION"
    assert loaded["selected_challenger"] == SELECTED_POLICY
    assert loaded["actions"] == 50
    assert loaded["actions_per_game"] == 25
    assert loaded["authority"] == "shadow"
    assert loaded["source_validation_authorized"] is False
    assert loaded["bounded_authority_authorized"] is False
    assert loaded["active_authority_authorized"] is False


def test_confirmation_manifest_rejects_open_authority(tmp_path: Path) -> None:
    path = tmp_path / "confirmation.json"
    payload = freeze_confirmation_manifest(output_path=path)
    payload["source_validation_authorized"] = True
    unsigned = dict(payload)
    unsigned.pop("manifest_checksum", None)
    payload["manifest_checksum"] = v86i.v86c._checksum(unsigned)
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_confirmation_manifest(path)
    except ValueError as error:
        assert "firewall" in str(error)
    else:
        raise AssertionError("open authority was accepted")


def test_live_controller_materializes_only_selected_structural_posterior() -> None:
    controller = StructuralLiveController(
        caps={
            "maximum_programs": 8,
            "maximum_sequences": 8,
            "maximum_particles_per_decision": 4,
            "ordinary_horizon": 1,
        }
    )

    assert len(controller.controllers) == 1
    assert controller.effective_mode.value == "shadow"
    assert isinstance(controller.selected.executor, StructuralRoleProgramExecutor)
    assert isinstance(
        controller.posterior, MinimumKLFamilyFloorProgramPosterior
    )
    assert controller.selected_name.endswith("_repair_v2")
