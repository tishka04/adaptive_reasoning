from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from theory.sage_t.causal import local_program_utility_experiment as experiment
from theory.sage_t.causal import local_program_utility_protocol as protocol_module
from theory.sage_t.causal.local_program_utility import (
    audit_calibration_trials,
    audit_evaluation_trials,
    evaluation_registry_payload,
    program_id,
)
from theory.sage_t.causal.local_program_utility_cli import build_parser
from theory.sage_t.causal.local_program_utility_experiment import LocalProgramTrial
from theory.sage_t.causal.local_program_utility_protocol import (
    LocalProgramUtilityProtocol,
    freeze_local_program_utility,
    load_local_program_utility_manifest,
)
from theory.sage_t.causal.option_minimization_experiment import (
    _load_contextual_option,
)
from theory.sage_t.causal.options import MinimalCausalOption
from theory.sage_t.causal.progress import CausalProgressProgram
from theory.sage_t.causal.progress_shadow import (
    posterior_from_snapshot,
    step_from_projection,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return _repo() / "training" / "sage_t" / "progress_contrast_t12_5b_3_bp35"


def _source_inputs():
    shadow = _repo() / "training" / "sage_t" / "progress_shadow_t12_5b_r1_bp35"
    manifest = json.loads((shadow / "manifest.json").read_text())
    posterior_payload = json.loads(
        (_repo() / manifest["inputs"]["posterior"]["path"]).read_text()
    )
    registry = json.loads(
        (_repo() / manifest["inputs"]["program_registry"]["path"]).read_text()
    )
    programs = tuple(
        CausalProgressProgram.from_dict(dict(item)) for item in registry["programs"]
    )
    milestones = next(
        item.milestones for item in programs if item.progress_kind == "ordered_effects"
    )
    option_payload = _load_contextual_option(
        _repo()
        / "training"
        / "sage_t"
        / "option_minimization_t12_4a_3r1_bp35"
        / "ablation"
        / "minimal_option.json"
    )
    option = MinimalCausalOption.from_dict(dict(option_payload["option"]))
    return posterior_from_snapshot(posterior_payload), milestones, option


def _synthetic_trials(
    *,
    phase: str,
    programs: tuple[tuple[str, ...], ...],
    lineage_seed: int,
) -> tuple[LocalProgramTrial, ...]:
    protocol = LocalProgramUtilityProtocol()
    _, milestones, _ = _source_inputs()
    prefix = [
        milestone.as_step(position=index, action_name="PREFIX")
        for index, milestone in enumerate(milestones[: protocol.target_stage])
    ]
    prefix.append(
        step_from_projection(
            action_name=protocol.detour_action,
            vector=(0,) * len(protocol.allowed_effect_features),
            features=protocol.allowed_effect_features,
            position=protocol.target_stage,
        )
    )
    output = []
    for actions in programs:
        identifier = program_id(actions)
        if tuple(actions) == ("ACTION3", "ACTION3"):
            candidate = (
                milestones[3].as_step(position=len(prefix), action_name="ACTION3"),
                milestones[4].as_step(
                    position=len(prefix) + 1,
                    action_name="ACTION3",
                ),
            )
            level_delta = 1
            terminal = False
        elif tuple(actions) == ("ACTION4", "ACTION4"):
            first = [0] * len(protocol.allowed_effect_features)
            first[0] = 1_000
            candidate = (
                step_from_projection(
                    action_name="ACTION4",
                    vector=first,
                    features=protocol.allowed_effect_features,
                    position=len(prefix),
                ),
                step_from_projection(
                    action_name="ACTION4",
                    vector=(0,) * len(protocol.allowed_effect_features),
                    features=protocol.allowed_effect_features,
                    position=len(prefix) + 1,
                ),
            )
            level_delta = 0
            terminal = False
        else:
            candidate = tuple(
                step_from_projection(
                    action_name=action,
                    vector=(0,) * len(protocol.allowed_effect_features),
                    features=protocol.allowed_effect_features,
                    position=len(prefix) + index,
                )
                for index, action in enumerate(actions)
            )
            level_delta = 0
            terminal = tuple(actions) == ("ACTION6", "ACTION6")
        for repetition in range(protocol.repetitions_per_program):
            output.append(
                LocalProgramTrial(
                    trial_id=f"{phase}_{lineage_seed}_{identifier}_{repetition}",
                    phase=phase,
                    lineage_seed=lineage_seed,
                    context_id=protocol.context_id,
                    program_id=identifier,
                    program_actions=tuple(actions),
                    repetition=repetition,
                    original_prefix_exact=True,
                    detour_available=True,
                    detour_neutral=True,
                    detour_terminal=False,
                    detour_context_hash=f"context-{lineage_seed}",
                    prefix_exact=True,
                    prefix_steps=tuple(prefix),
                    candidate_steps=candidate,
                    executed_action_count=len(actions),
                    program_complete=True,
                    level_delta=level_delta,
                    progressed=level_delta > 0,
                    terminal=terminal,
                    terminal_failure=terminal and level_delta <= 0,
                    terminal_state="GAME_OVER" if terminal else "NOT_FINISHED",
                    sdk_calls_after=len(output) + 1,
                )
            )
    return tuple(output)


def _calibration_audit():
    protocol = LocalProgramUtilityProtocol()
    posterior, _, _ = _source_inputs()
    return audit_calibration_trials(
        trials=[
            item.to_dict()
            for item in _synthetic_trials(
                phase="calibration",
                programs=protocol.calibration_programs,
                lineage_seed=protocol.calibration_lineage_seed,
            )
        ],
        expected_programs=protocol.calibration_programs,
        repetitions_per_program=protocol.repetitions_per_program,
        transport_actions=protocol.transport_actions,
        features=protocol.allowed_effect_features,
        posterior=posterior,
        minimum_distractor_magnitude_gap=(
            protocol.minimum_distractor_magnitude_gap
        ),
    )


def test_protocol_is_bounded_two_phase_and_has_no_control_command() -> None:
    protocol = LocalProgramUtilityProtocol()
    assert len(protocol.calibration_programs) == 36
    assert len(protocol.transport_programs) == 12
    assert protocol.expected_calibration_trials == 72
    assert protocol.expected_evaluation_trials == 4
    assert protocol.maximum_calibration_sdk_calls == 6_500
    assert protocol.maximum_evaluation_sdk_calls == 1_000
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


def test_freeze_binds_negative_parent_and_shallow_safe_context(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_local_program_utility(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "collection" / "contrast_receipt.json",
        root=_repo(),
    )
    loaded = load_local_program_utility_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "FAIL_T12_5B_3_COLLECTION_INTEGRITY_GATE"
    )
    assert loaded["parent"]["receipt"]["next_phase_authorized"] is False
    assert loaded["selection"]["context_id"] == "stage_3_action4_depth_1"
    assert loaded["selection"]["shared_transport_actions"] == [
        "ACTION3",
        "ACTION4",
    ]
    assert loaded["design"][
        "candidate_terminal_is_scientific_risk_not_integrity_failure"
    ]
    assert loaded["firewall"]["calibration_collection_authorized"]
    assert loaded["firewall"]["evaluation_collection_authorized"] is False
    assert loaded["firewall"]["t12_5c_control_freeze_authorized"] is False


def test_calibration_registers_score_independent_hard_program_contrast() -> None:
    audit = _calibration_audit()
    metrics = audit["metrics"]
    assert metrics["fixed_program_schedule_completed"]
    assert metrics["context_replay_is_exact"]
    assert metrics["transport_safe_progress_program_count"] == 1
    assert metrics["hard_utility_contrast_count"] == 1
    assert metrics["causal_contrast_correct"]
    assert metrics["unsafe_program_count"] == 1
    assert audit["selection"]["causal_score_used_for_program_selection"] is False
    assert audit["selection"]["progress"]["program_id"] == "ACTION3>ACTION3"
    assert audit["selection"]["distractor"]["program_id"] == "ACTION4>ACTION4"


def test_candidate_terminal_is_risk_evidence_not_integrity_failure() -> None:
    audit = _calibration_audit()
    metrics = audit["metrics"]
    assert metrics["unsafe_program_count"] == 1
    assert metrics["availability_is_deterministic"]
    assert metrics["effects_are_deterministic_when_complete"]
    assert metrics["outcomes_are_deterministic"]


def test_registered_pair_transfers_on_fresh_lineage() -> None:
    protocol = LocalProgramUtilityProtocol()
    posterior, _, _ = _source_inputs()
    calibration = _calibration_audit()
    registry = evaluation_registry_payload(
        manifest_checksum="m" * 64,
        protocol_checksum=protocol.checksum,
        calibration_evidence_checksum="e" * 64,
        selection=calibration["selection"],
    )
    programs = (
        tuple(registry["programs"]["progress"]["program_actions"]),
        tuple(registry["programs"]["distractor"]["program_actions"]),
    )
    audit = audit_evaluation_trials(
        trials=[
            item.to_dict()
            for item in _synthetic_trials(
                phase="evaluation",
                programs=programs,
                lineage_seed=protocol.evaluation_lineage_seed,
            )
        ],
        evaluation_registry=registry,
        repetitions_per_program=protocol.repetitions_per_program,
        transport_actions=protocol.transport_actions,
        features=protocol.allowed_effect_features,
        posterior=posterior,
    )
    assert audit["metrics"]["progress_program_transferred"]
    assert audit["metrics"]["distractor_stable_safe_nonprogress"]
    assert audit["metrics"]["causal_utility_transferred"]


def test_synthetic_phases_pass_only_to_t12_5c_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    protocol = LocalProgramUtilityProtocol()
    posterior, _, _ = _source_inputs()
    manifest = {
        "claim_boundary": {
            "authorized": "source-train local program calibration",
            "not_authorized": ["environment control", "holdout performance"],
        },
        "firewall": {"calibration_collection_authorized": True},
        "game_id": "bp35",
        "inputs": {
            "successful_prefix_lengths": {"8701": 0, "8705": 0},
        },
        "manifest_checksum": "m" * 64,
        "parent": {
            "receipt": {
                "receipt_checksum": "r" * 64,
                "status": "FAIL_T12_5B_3_COLLECTION_INTEGRITY_GATE",
            }
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }
    calibration_trials = _synthetic_trials(
        phase="calibration",
        programs=protocol.calibration_programs,
        lineage_seed=protocol.calibration_lineage_seed,
    )

    monkeypatch.setattr(
        experiment,
        "load_local_program_utility_manifest",
        lambda *args, **kwargs: manifest,
    )

    def fake_collect(**kwargs):
        if kwargs["phase"] == "calibration":
            return calibration_trials, posterior
        return (
            _synthetic_trials(
                phase="evaluation",
                programs=tuple(tuple(item) for item in kwargs["programs"]),
                lineage_seed=protocol.evaluation_lineage_seed,
            ),
            posterior,
        )

    monkeypatch.setattr(experiment, "_collect_programs", fake_collect)
    calibration_dir = tmp_path / "calibration"
    calibration_receipt = experiment.run_local_program_calibration(
        manifest_path="unused.json",
        output_dir=calibration_dir,
        environments_dir="unused",
        env_factory=lambda game_id: None,
        root=_repo(),
    )
    assert calibration_receipt["passed"] is True
    assert calibration_receipt["status"] == "PASS_T12_5B_4_CALIBRATION_GATE"
    assert (calibration_dir / "evaluation_registry.json").is_file()
    pre_evaluation = experiment.local_program_utility_status(
        manifest_path="unused.json",
        calibration_receipt_path=calibration_dir / "calibration_receipt.json",
        evaluation_receipt_path=tmp_path / "missing.json",
        root=_repo(),
    )
    assert pre_evaluation["firewall"]["evaluation_collection_authorized"]
    assert pre_evaluation["firewall"]["t12_5c_control_freeze_authorized"] is False

    evaluation_dir = tmp_path / "evaluation"
    evaluation_receipt = experiment.run_local_program_evaluation(
        manifest_path="unused.json",
        calibration_receipt_path=calibration_dir / "calibration_receipt.json",
        output_dir=evaluation_dir,
        environments_dir="unused",
        env_factory=lambda game_id: None,
        root=_repo(),
    )
    assert evaluation_receipt["passed"] is True
    assert evaluation_receipt["status"] == (
        "PASS_T12_5B_4_LOCAL_PROGRAM_UTILITY_GATE"
    )
    status = experiment.local_program_utility_status(
        manifest_path="unused.json",
        calibration_receipt_path=calibration_dir / "calibration_receipt.json",
        evaluation_receipt_path=evaluation_dir / "evaluation_receipt.json",
        root=_repo(),
    )
    assert status["next_phase_authorized"]
    assert status["firewall"]["t12_5c_control_freeze_authorized"]
    assert status["firewall"]["environment_collection_authorized"] is False
    assert status["firewall"]["causal_progress_control_authorized"] is False
