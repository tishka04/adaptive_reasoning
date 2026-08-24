from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from theory.sage_t.causal import goal_cursor_control_experiment as experiment
from theory.sage_t.causal import goal_cursor_control_protocol as protocol_module
from theory.sage_t.causal.goal_cursor_control import audit_goal_cursor_control
from theory.sage_t.causal.goal_cursor_control_cli import build_parser
from theory.sage_t.causal.goal_cursor_control_experiment import LocalProgramTrial
from theory.sage_t.causal.goal_cursor_control_protocol import (
    GoalCursorControlProtocol,
    freeze_goal_cursor_control,
    load_goal_cursor_control_manifest,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "goal_viability_t12_5b_5_bp35"


def _synthetic_trials(
    *,
    control_progress: bool = False,
    treatment_progress: bool = True,
) -> tuple[LocalProgramTrial, ...]:
    protocol = GoalCursorControlProtocol()
    arm_by_name = {arm.name: arm for arm in protocol.arms}
    output: list[LocalProgramTrial] = []
    for entry in protocol.schedule:
        arm = arm_by_name[entry.arm_name]
        is_treatment = arm.name == "goal_cursor"
        progressed = treatment_progress if is_treatment else control_progress
        terminal_failure = bool(not is_treatment and entry.lineage_seed == 8_701)
        prefix = tuple(
            {
                "action_name": "ACTION4",
                "available": True,
                "delta": {
                    "exact_changed": True,
                    "mechanism": {
                        "lineage": entry.lineage_seed,
                        "stage": index,
                    },
                },
                "position": index,
            }
            for index in range(protocol.target_stage)
        )
        candidate = tuple(
            {
                "action_name": action,
                "available": True,
                "delta": {
                    "exact_changed": True,
                    "mechanism": {
                        "arm": arm.name,
                        "lineage": entry.lineage_seed,
                        "slot": index,
                    },
                },
                "position": len(prefix) + index,
            }
            for index, action in enumerate(arm.program_actions)
        )
        output.append(
            LocalProgramTrial(
                trial_id=(
                    f"paired_{entry.order_index:02d}_lineage_{entry.lineage_seed}_"
                    f"{arm.name}_rep_{entry.repetition}"
                ),
                phase="paired_control",
                lineage_seed=entry.lineage_seed,
                context_id=protocol.context_id,
                program_id=arm.program_id,
                program_actions=arm.program_actions,
                repetition=entry.repetition,
                original_prefix_exact=True,
                detour_available=True,
                detour_neutral=True,
                detour_terminal=False,
                detour_context_hash=f"stage3-{entry.lineage_seed}",
                prefix_exact=True,
                prefix_steps=prefix,
                candidate_steps=candidate,
                executed_action_count=len(arm.program_actions),
                program_complete=True,
                level_delta=1 if progressed else 0,
                progressed=progressed,
                terminal=terminal_failure,
                terminal_failure=terminal_failure,
                terminal_state="GAME_OVER" if terminal_failure else "NOT_FINISHED",
                sdk_calls_after=len(output) + 1,
            )
        )
    return tuple(output)


def _audit(trials=None):
    protocol = GoalCursorControlProtocol()
    selected = tuple(trials or _synthetic_trials())
    return audit_goal_cursor_control(
        trials=[item.to_dict() for item in selected],
        arms=protocol.arms,
        schedule=protocol.schedule,
    )


def test_protocol_is_equal_capacity_bounded_and_has_one_run_phase() -> None:
    protocol = GoalCursorControlProtocol()
    assert [arm.program_actions for arm in protocol.arms] == [
        ("ACTION3", "ACTION3"),
        ("ACTION4", "ACTION3"),
    ]
    assert len({len(arm.program_actions) for arm in protocol.arms}) == 1
    assert protocol.expected_trials == 8
    assert [entry.arm_name for entry in protocol.schedule] == [
        "goal_cursor",
        "binding_swap",
        "binding_swap",
        "goal_cursor",
        "binding_swap",
        "goal_cursor",
        "goal_cursor",
        "binding_swap",
    ]
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
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["holdout"])


def test_protocol_rejects_post_hoc_arm_or_budget_change() -> None:
    with pytest.raises(ValueError, match="preregistered value changed"):
        GoalCursorControlProtocol(binding_swap_program=("ACTION4", "ACTION3", "ACTION3"))
    with pytest.raises(ValueError, match="preregistered value changed"):
        GoalCursorControlProtocol(maximum_sdk_calls=1_001)


def test_freeze_binds_passed_t12_5b_5_and_all_parent_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_goal_cursor_control(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "evaluation" / "evaluation_receipt.json",
        root=_repo(),
    )
    loaded = load_goal_cursor_control_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "PASS_T12_5B_5_GOAL_VIABILITY_GATE"
    )
    assert len(loaded["parent"]["artifacts"]) == 5
    assert loaded["selection"]["capacity_match"]["maximum_action_slots_per_arm"] == 2
    assert loaded["selection"]["score_used_for_arm_assignment_or_labels"] is False
    assert loaded["firewall"]["paired_control_collection_authorized"]
    assert loaded["firewall"]["t12_6_freeze_authorized"] is False


def test_paired_control_passes_with_terminal_heterogeneity_retained() -> None:
    audit = _audit()
    metrics = audit["metrics"]
    assert metrics["fixed_counterbalanced_schedule_completed"]
    assert metrics["context_replay_is_exact"]
    assert metrics["equal_capacity_horizon"]
    assert metrics["goal_cursor_safe_progress"]
    assert metrics["binding_swap_control_rejected"]
    assert metrics["paired_advantage_all_lineages"]
    assert metrics["terminal_failure_heterogeneous_across_lineages"]


def test_control_progress_is_scientific_miss_not_integrity_failure() -> None:
    audit = _audit(_synthetic_trials(control_progress=True))
    metrics = audit["metrics"]
    assert metrics["fixed_counterbalanced_schedule_completed"]
    assert metrics["context_replay_is_exact"]
    assert metrics["outcomes_are_deterministic"]
    assert metrics["binding_swap_control_rejected"] is False
    assert metrics["paired_advantage_all_lineages"] is False


def test_missing_live_action_is_not_converted_to_zero_effect() -> None:
    trials = list(_synthetic_trials())
    index = next(
        position
        for position, trial in enumerate(trials)
        if trial.lineage_seed == 8_705
        and trial.program_id == "ACTION4>ACTION3"
        and trial.repetition == 0
    )
    trials[index] = replace(
        trials[index],
        candidate_steps=(
            {
                "action_name": "ACTION4",
                "available": False,
                "delta": {"exact_changed": False, "mechanism": {}},
                "position": 3,
            },
        ),
        executed_action_count=0,
        program_complete=False,
        terminal=False,
        terminal_failure=False,
        terminal_state="NOT_FINISHED",
    )
    audit = _audit(trials)
    assert audit["metrics"]["actions_available_or_terminal"] is False
    assert audit["metrics"]["availability_is_deterministic"] is False


def test_synthetic_run_passes_only_to_t12_6_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = GoalCursorControlProtocol()
    parent_receipt = _parent() / "evaluation" / "evaluation_receipt.json"
    manifest = {
        "claim_boundary": {
            "authorized": "source-train paired causal control",
            "not_authorized": ["environment control", "holdout performance"],
        },
        "firewall": {"paired_control_collection_authorized": True},
        "game_id": "bp35",
        "inputs": {},
        "manifest_checksum": "m" * 64,
        "parent": {
            "receipt": {
                "path": str(parent_receipt),
                "receipt_checksum": "r" * 64,
                "status": "PASS_T12_5B_5_GOAL_VIABILITY_GATE",
            }
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    monkeypatch.setattr(
        experiment,
        "load_goal_cursor_control_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_collect_control_trials",
        lambda **kwargs: _synthetic_trials(),
    )
    output = tmp_path / "control"
    receipt = experiment.run_goal_cursor_control(
        manifest_path="unused.json",
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: None,
        root=_repo(),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE"
    assert receipt["metrics"]["t12_6_freeze_authorized"]
    status = experiment.goal_cursor_control_status(
        manifest_path="unused.json",
        control_receipt_path=output / "control_receipt.json",
        root=_repo(),
    )
    assert status["next_phase_authorized"]
    assert status["firewall"]["t12_6_freeze_authorized"]
    assert status["firewall"]["environment_collection_authorized"] is False
    assert status["firewall"]["controller_authority"] is False
    assert status["firewall"]["source_validation_opened"] is False
