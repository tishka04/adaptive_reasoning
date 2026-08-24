from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from theory.sage_t.causal import goal_viability_experiment as experiment
from theory.sage_t.causal import goal_viability_protocol as protocol_module
from theory.sage_t.causal.goal_viability import (
    audit_calibration_trials,
    audit_evaluation_trials,
    evaluation_registry_payload,
)
from theory.sage_t.causal.goal_viability_cli import build_parser
from theory.sage_t.causal.goal_viability_experiment import LocalProgramTrial
from theory.sage_t.causal.goal_viability_protocol import (
    GoalViabilityProtocol,
    freeze_goal_viability,
    load_goal_viability_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "local_program_utility_t12_5b_4_bp35"


def _synthetic_trials(
    *,
    phase: str,
    branches=None,
    lineage_seed: int = 8_701,
    progress: bool = True,
) -> tuple[LocalProgramTrial, ...]:
    protocol = GoalViabilityProtocol()
    selected = tuple(branches or protocol.calibration_branches)
    prefix = tuple(
        {
            "action_name": "ACTION4",
            "available": True,
            "delta": {"exact_changed": True, "mechanism": {"index": index}},
            "position": index,
        }
        for index in range(protocol.target_stage)
    )
    output = []
    for branch in selected:
        for repetition in range(protocol.repetitions_per_branch):
            is_progress = bool(branch.goal_cursor_advance and progress)
            terminal = not branch.goal_cursor_advance
            candidate = tuple(
                {
                    "action_name": action,
                    "available": True,
                    "delta": {
                        "exact_changed": True,
                        "mechanism": {"step": index, "branch": branch.first_action},
                    },
                    "position": len(prefix) + index,
                }
                for index, action in enumerate(branch.program_actions)
            )
            output.append(
                LocalProgramTrial(
                    trial_id=(
                        f"{phase}_{lineage_seed}_{branch.branch_id}_{repetition}"
                    ),
                    phase=phase,
                    lineage_seed=lineage_seed,
                    context_id=protocol.context_id,
                    program_id=branch.branch_id,
                    program_actions=branch.program_actions,
                    repetition=repetition,
                    original_prefix_exact=True,
                    detour_available=True,
                    detour_neutral=True,
                    detour_terminal=False,
                    detour_context_hash=f"stage3-{lineage_seed}",
                    prefix_exact=True,
                    prefix_steps=prefix,
                    candidate_steps=candidate,
                    executed_action_count=len(branch.program_actions),
                    program_complete=True,
                    level_delta=1 if is_progress else 0,
                    progressed=is_progress,
                    terminal=terminal,
                    terminal_failure=terminal,
                    terminal_state="GAME_OVER" if terminal else "NOT_FINISHED",
                    sdk_calls_after=len(output) + 1,
                )
            )
    return tuple(output)


def _calibration_audit(*, progress: bool = True):
    protocol = GoalViabilityProtocol()
    return audit_calibration_trials(
        trials=[
            item.to_dict()
            for item in _synthetic_trials(
                phase="calibration",
                branches=protocol.calibration_branches,
                lineage_seed=protocol.calibration_lineage_seed,
                progress=progress,
            )
        ],
        expected_branches=protocol.calibration_branches,
        repetitions_per_branch=protocol.repetitions_per_branch,
    )


def test_protocol_is_bounded_two_phase_and_has_no_control_command() -> None:
    protocol = GoalViabilityProtocol()
    assert [branch.program_actions for branch in protocol.calibration_branches] == [
        ("ACTION3", "ACTION3"),
        ("ACTION4", "ACTION3", "ACTION3"),
        ("ACTION6", "ACTION3", "ACTION3"),
    ]
    assert len(protocol.transport_branches) == 2
    assert protocol.expected_calibration_trials == 6
    assert protocol.expected_evaluation_trials == 4
    assert protocol.maximum_total_sdk_calls == 1_750
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["calibrate"]).phase == "calibrate"
    assert parser.parse_args(["evaluate"]).phase == "evaluate"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["control"])
    with pytest.raises(SystemExit):
        parser.parse_args(["holdout"])


def test_protocol_rejects_post_hoc_schedule_change() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        GoalViabilityProtocol(candidate_first_actions=("ACTION3", "ACTION4"))


def test_freeze_binds_integrity_clean_negative_parent_and_goal_witness(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_goal_viability(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "calibration" / "calibration_receipt.json",
        root=_repo(),
    )
    loaded = load_goal_viability_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "FAIL_T12_5B_4_NO_LOCAL_PROGRESS_PROGRAM"
    )
    assert loaded["parent"]["receipt"]["next_phase_authorized"] is False
    assert loaded["selection"]["goal_continuation"] == ["ACTION3", "ACTION3"]
    assert loaded["selection"]["fatal_detour_action"] == "ACTION4"
    assert len(loaded["selection"]["fatal_detour_evidence_ids"]) == 2
    assert loaded["design"]["immediate_milestone_neutrality_is_not_viability"]
    assert loaded["firewall"]["calibration_collection_authorized"]
    assert loaded["firewall"]["evaluation_collection_authorized"] is False
    assert loaded["firewall"]["t12_5c_control_freeze_authorized"] is False


def test_calibration_registers_score_independent_goal_viability_contrast() -> None:
    audit = _calibration_audit()
    metrics = audit["metrics"]
    assert metrics["fixed_branch_schedule_completed"]
    assert metrics["context_replay_is_exact"]
    assert metrics["cursor_advance_safe_progress_count"] == 1
    assert metrics["cursor_mismatch_rejected_count"] == 1
    assert metrics["viability_contrast_count"] == 1
    assert audit["selection"]["score_used_for_branch_selection"] is False
    assert audit["selection"]["progress"]["branch_id"] == "ACTION3>ACTION3"
    assert audit["selection"]["control"]["branch_id"] == (
        "ACTION4>ACTION3>ACTION3"
    )


def test_no_progress_is_scientific_miss_not_schedule_failure() -> None:
    audit = _calibration_audit(progress=False)
    assert audit["metrics"]["fixed_branch_schedule_completed"]
    assert audit["metrics"]["outcomes_are_deterministic"]
    assert audit["metrics"]["cursor_advance_safe_progress_count"] == 0
    assert audit["metrics"]["viability_contrast_count"] == 0


def test_nontransport_missing_action_is_not_converted_to_zero_effect() -> None:
    protocol = GoalViabilityProtocol()
    trials = list(_synthetic_trials(phase="calibration"))
    for index, trial in enumerate(trials):
        if trial.program_actions[0] != "ACTION6":
            continue
        unavailable = (
            {
                "action_name": "ACTION6",
                "available": False,
                "delta": {"exact_changed": False, "mechanism": {}},
                "position": protocol.target_stage,
            },
        )
        trials[index] = replace(
            trial,
            candidate_steps=unavailable,
            executed_action_count=0,
            program_complete=False,
            terminal=False,
            terminal_failure=False,
            terminal_state="NOT_FINISHED",
        )
    audit = audit_calibration_trials(
        trials=[item.to_dict() for item in trials],
        expected_branches=protocol.calibration_branches,
        repetitions_per_branch=protocol.repetitions_per_branch,
    )
    assert audit["metrics"]["missing_branch_count"] == 1
    assert audit["metrics"]["transport_first_actions_available"]
    assert audit["metrics"]["viability_contrast_count"] == 1


def test_registered_viability_pair_transfers_on_second_lineage() -> None:
    protocol = GoalViabilityProtocol()
    calibration = _calibration_audit()
    registry = evaluation_registry_payload(
        manifest_checksum="m" * 64,
        protocol_checksum=protocol.checksum,
        calibration_evidence_checksum="e" * 64,
        selection=calibration["selection"],
    )
    branches = tuple(
        branch
        for name in ("progress", "control")
        for branch in protocol.calibration_branches
        if branch.branch_id == registry["branches"][name]["branch_id"]
    )
    audit = audit_evaluation_trials(
        trials=[
            item.to_dict()
            for item in _synthetic_trials(
                phase="evaluation",
                branches=branches,
                lineage_seed=protocol.evaluation_lineage_seed,
            )
        ],
        evaluation_registry=registry,
        repetitions_per_branch=protocol.repetitions_per_branch,
    )
    assert audit["metrics"]["progress_branch_transferred"]
    assert audit["metrics"]["control_branch_rejected"]
    assert audit["metrics"]["goal_viability_contrast_transferred"]


def test_synthetic_phases_pass_only_to_t12_5c_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = GoalViabilityProtocol()
    manifest = {
        "claim_boundary": {
            "authorized": "source-train goal viability calibration",
            "not_authorized": ["environment control", "holdout performance"],
        },
        "firewall": {"calibration_collection_authorized": True},
        "game_id": "bp35",
        "inputs": {"successful_prefix_lengths": {"8701": 0, "8705": 0}},
        "manifest_checksum": "m" * 64,
        "parent": {
            "manifest": {"path": "unused-parent.json"},
            "receipt": {
                "receipt_checksum": "r" * 64,
                "status": "FAIL_T12_5B_4_NO_LOCAL_PROGRESS_PROGRAM",
            },
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    monkeypatch.setattr(
        experiment,
        "load_goal_viability_manifest",
        lambda *args, **kwargs: manifest,
    )

    def fake_collect(**kwargs):
        return _synthetic_trials(
            phase=kwargs["phase"],
            branches=tuple(kwargs["branches"]),
            lineage_seed=int(kwargs["lineage_seed"]),
        )

    monkeypatch.setattr(experiment, "_collect_branches", fake_collect)
    calibration_dir = tmp_path / "calibration"
    calibration_receipt = experiment.run_goal_viability_calibration(
        manifest_path="unused.json",
        output_dir=calibration_dir,
        environments_dir="unused",
        env_factory=lambda game_id: None,
        root=_repo(),
    )
    assert calibration_receipt["passed"] is True
    assert calibration_receipt["status"] == "PASS_T12_5B_5_CALIBRATION_GATE"
    assert (calibration_dir / "evaluation_registry.json").is_file()
    pre_evaluation = experiment.goal_viability_status(
        manifest_path="unused.json",
        calibration_receipt_path=calibration_dir / "calibration_receipt.json",
        evaluation_receipt_path=tmp_path / "missing.json",
        root=_repo(),
    )
    assert pre_evaluation["firewall"]["evaluation_collection_authorized"]
    assert pre_evaluation["firewall"]["t12_5c_control_freeze_authorized"] is False

    evaluation_dir = tmp_path / "evaluation"
    evaluation_receipt = experiment.run_goal_viability_evaluation(
        manifest_path="unused.json",
        calibration_receipt_path=calibration_dir / "calibration_receipt.json",
        output_dir=evaluation_dir,
        environments_dir="unused",
        env_factory=lambda game_id: None,
        root=_repo(),
    )
    assert evaluation_receipt["passed"] is True
    assert evaluation_receipt["status"] == "PASS_T12_5B_5_GOAL_VIABILITY_GATE"
    status = experiment.goal_viability_status(
        manifest_path="unused.json",
        calibration_receipt_path=calibration_dir / "calibration_receipt.json",
        evaluation_receipt_path=evaluation_dir / "evaluation_receipt.json",
        root=_repo(),
    )
    assert status["next_phase_authorized"]
    assert status["firewall"]["t12_5c_control_freeze_authorized"]
    assert status["firewall"]["environment_collection_authorized"] is False
    assert status["firewall"]["causal_progress_control_authorized"] is False
