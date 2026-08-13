from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal import option_applicability_experiment as experiment
from theory.sage_t.causal import option_applicability_protocol as protocol_module
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.option_applicability_cli import build_parser
from theory.sage_t.causal.option_applicability_protocol import (
    OptionApplicabilityProtocol,
    freeze_option_applicability,
    load_option_applicability_manifest,
)
from theory.sage_t.causal.options import MinimalCausalOption, MinimalOptionStep
from theory.sage_t.causal.witness_protocol import ProgressWitness, WitnessStep


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class FakeAction:
    id: int
    data: dict | None = None


@dataclass
class FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1, 2, 3, 4)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(index) for index in (1, 2, 3, 4)]


class ApplicabilityEnv:
    option = (4, 4, 4, 3, 3)

    def __init__(self) -> None:
        self._game = FakeGame()
        self.level = 0
        self.history: list[int] = []
        self.route_tag = 0

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        if value == 0:
            self.level = 0
            self.history = []
            self.route_tag = 0
        elif self.level == 0:
            if not self.history and value in {1, 2}:
                self.route_tag = value
            else:
                self.history.append(value)
            if tuple(self.history[-5:]) == self.option:
                self.level = 1
                self.history = []
                self.route_tag = 0
        else:
            self.history.append(value)
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0, 0] = self.level + 1
        grid[0, 1] = self.route_tag
        for index, item in enumerate(self.history[-8:]):
            grid[1 + index // 4, index % 4] = item
        return FakeFrame(grid, levels_completed=self.level)


def _option() -> MinimalCausalOption:
    return MinimalCausalOption(
        initiation_signature="synthetic-entry",
        initiation_exact_hash="synthetic-exact-entry",
        steps=tuple(
            MinimalOptionStep(f"ACTION{action}", {})
            for action in ApplicabilityEnv.option
        ),
        source_evidence_ids=("witness-8701", "witness-8705"),
        reproduction_count=8,
        minimization_evaluations=384,
        source="t12_4a_4b_test",
    )


def _witness(first_action: int, seed: int) -> tuple[ProgressWitness, str]:
    env = ApplicabilityEnv()
    frame = env.step(0)
    initial_hash = state_signature_from_frame(frame)
    steps = []
    actions = (first_action, 3, 4, 4, 4, 3, 3)
    successful_anchor = ""
    for position, action in enumerate(actions):
        source_hash = state_signature_from_frame(frame)
        frame = env.step(action)
        target_hash = state_signature_from_frame(frame)
        if position == 0:
            successful_anchor = target_hash
        steps.append(
            WitnessStep(
                expected_source_hash=source_hash,
                action=GroundedAction(f"ACTION{action}"),
                expected_target_hash=target_hash,
                level_delta=1 if position == len(actions) - 1 else 0,
                success=position == len(actions) - 1,
            )
        )
    return (
        ProgressWitness(
            witness_id=f"witness-{seed}",
            game_id="bp35",
            source_seed=seed,
            source_arm="lineage_control",
            source_archive_sha256=str(seed) * 8,
            source_progress_edge_id=f"edge-{seed}",
            initial_exact_hash=initial_hash,
            initial_level=0,
            target_exact_hash=state_signature_from_frame(frame),
            target_level=1,
            steps=tuple(steps),
        ),
        successful_anchor,
    )


def _synthetic_manifest() -> tuple[dict, tuple[ProgressWitness, ...]]:
    protocol = OptionApplicabilityProtocol()
    first, first_anchor = _witness(1, 8701)
    second, second_anchor = _witness(2, 8705)
    witnesses = (first, second)
    return (
        {
            "firewall": {"option_applicability_audit_authorized": True},
            "game_id": "bp35",
            "inputs": {
                "failed_anchor_hash": first.target_exact_hash,
                "failed_anchor_level": 1,
                "option_checksum": _option().checksum,
                "successful_anchor_hashes": {
                    "8701": first_anchor,
                    "8705": second_anchor,
                },
                "successful_prefix_lengths": {"8701": 1, "8705": 1},
            },
            "manifest_checksum": "m" * 64,
            "parent": {
                "negative_receipt": {
                    "receipt_checksum": "n" * 64,
                    "status": "FAIL_T12_4A_4_OPTION_TRANSFER_GATE",
                }
            },
            "protocol": asdict(protocol),
            "protocol_checksum": protocol.checksum,
        },
        witnesses,
    )


def test_protocol_is_bounded_diagnostic_and_has_no_activation_cli() -> None:
    protocol = OptionApplicabilityProtocol()
    assert protocol.expected_trial_count == 16
    assert protocol.maximum_sdk_calls == 1_200
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    assert protocol.persist_raw_frames is False
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "manifest.json",
            "--parent-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])


def test_freeze_binds_real_negative_transfer_without_opening_authority(
    monkeypatch,
    tmp_path,
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "option_transfer_t12_4a_4_bp35"
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_option_applicability(
        output_path=manifest_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "transfer" / "transfer_receipt.json",
        root=repo,
    )
    loaded = load_option_applicability_manifest(manifest_path, root=repo)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["negative_receipt"]["status"] == (
        "FAIL_T12_4A_4_OPTION_TRANSFER_GATE"
    )
    assert loaded["inputs"]["successful_prefix_lengths"] == {
        "8701": 58,
        "8705": 55,
    }
    assert loaded["firewall"]["option_applicability_audit_authorized"] is True
    assert loaded["firewall"]["option_control_authorized"] is False
    assert loaded["firewall"]["neural_training_authorized"] is False


def test_audit_reproduces_contrast_and_emits_one_diagnosis(
    monkeypatch,
    tmp_path,
) -> None:
    manifest, witnesses = _synthetic_manifest()
    monkeypatch.setattr(
        experiment,
        "load_option_applicability_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_load_inputs",
        lambda *args, **kwargs: (witnesses, _option()),
    )
    output = tmp_path / "audit"
    receipt = experiment.run_option_applicability(
        manifest_path="unused.json",
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: ApplicabilityEnv(),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_4A_4B_APPLICABILITY_AUDIT_GATE"
    assert receipt["metrics"]["trial_count"] == 16
    assert receipt["metrics"]["prefix_exact_trials"] == 16
    assert receipt["metrics"]["successful_context_progressions"] == 4
    assert receipt["metrics"]["failed_context_progressions"] == 0
    assert receipt["metrics"]["classification"] in (
        protocol_module.AUTHORIZED_DIAGNOSES
    )
    assert receipt["metrics"]["terminal_failures"] == 0
    assert receipt["metrics"]["sdk_calls"]["used_sdk_calls"] <= 1_200
    assert (output / "applicability_trials.json").is_file()
    assert (output / "applicability_diagnosis.json").is_file()
    assert (output / "applicability_receipt.json").is_file()

