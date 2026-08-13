from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal import witness_experiment
from theory.sage_t.causal import witness_reconfirmation_experiment as experiment
from theory.sage_t.causal import witness_reconfirmation_protocol as protocol_module
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.witness_protocol import ProgressWitness, WitnessStep
from theory.sage_t.causal.witness_reconfirmation_cli import build_parser
from theory.sage_t.causal.witness_reconfirmation_protocol import (
    WitnessReconfirmationProtocol,
    _archive_artifacts,
    _parent_is_global_calibration_failure,
    extract_reconfirmation_witnesses,
    freeze_witness_reconfirmation,
    load_reconfirmation_manifest,
    load_reconfirmation_registry,
)


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
    available_actions: tuple[int, ...] = (1, 2, 3)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(1), FakeAction(2), FakeAction(3)]


class TwoStepSuffixEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.flavour = 0
        self.suffix_count = 0

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        if value == 0:
            self.flavour = 0
            self.suffix_count = 0
            return self._frame(0)
        if value in {1, 2} and self.suffix_count == 0:
            self.flavour = value
        elif value == 3 and self.flavour in {1, 2}:
            self.suffix_count += 1
        marker = 99 if self.suffix_count >= 2 else self.flavour * 10 + self.suffix_count
        return self._frame(marker)

    def _frame(self, marker: int) -> FakeFrame:
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = marker
        progress = self.suffix_count >= 2
        return FakeFrame(
            grid,
            state="WIN" if progress else "NOT_FINISHED",
            levels_completed=int(progress),
        )


def _witness(first_action: int, seed: int) -> ProgressWitness:
    env = TwoStepSuffixEnv()
    frame = env.step(0)
    steps = []
    for action_id in (first_action, 3, 3):
        source_hash = state_signature_from_frame(frame)
        before_level = frame.levels_completed
        frame = env.step(action_id)
        steps.append(
            WitnessStep(
                expected_source_hash=source_hash,
                action=GroundedAction(f"ACTION{action_id}"),
                expected_target_hash=state_signature_from_frame(frame),
                level_delta=frame.levels_completed - before_level,
                success=frame.levels_completed > before_level,
            )
        )
    return ProgressWitness(
        witness_id=f"witness-{seed}",
        game_id="bp35",
        source_seed=seed,
        source_arm="lineage_shield_control",
        source_archive_sha256=str(seed) * 8,
        source_progress_edge_id=f"edge-{seed}",
        initial_exact_hash=steps[0].expected_source_hash,
        initial_level=0,
        target_exact_hash=steps[-1].expected_target_hash,
        target_level=1,
        steps=tuple(steps),
    )


def test_real_parent_failure_and_progress_provenance_are_exact() -> None:
    repo = _repo()
    root = repo / "training" / "sage_t" / "calibration_t12_4a_1_bp35"
    manifest = protocol_module.load_calibration_manifest(root / "manifest.json", root=repo)
    receipt = protocol_module.load_calibration_receipt(
        root / "training" / "calibration_receipt.json",
        manifest=manifest,
        root=repo,
    )
    collection = protocol_module.load_calibration_receipt(
        root / "collection" / "collection_receipt.json",
        manifest=manifest,
        root=repo,
        require_passed=True,
    )
    assert _parent_is_global_calibration_failure(manifest, receipt)
    witnesses = extract_reconfirmation_witnesses(_archive_artifacts(collection))
    assert [item.source_seed for item in witnesses] == [8701, 8705]
    assert [len(item.steps) for item in witnesses] == [64, 61]
    assert len({item.initial_exact_hash for item in witnesses}) == 1
    assert len({item.target_exact_hash for item in witnesses}) == 1
    assert [step.action.action_name for step in witnesses[0].steps[-6:]] == [
        "ACTION3",
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    ]


def test_freeze_seals_real_routes_and_keeps_option_extraction_closed(
    monkeypatch,
    tmp_path,
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "calibration_t12_4a_1_bp35"
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "witnesses.sealed.json"
    manifest = freeze_witness_reconfirmation(
        output_path=manifest_path,
        witness_registry_path=registry_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "training" / "calibration_receipt.json",
        collection_receipt_path=parent / "collection" / "collection_receipt.json",
        root=repo,
    )
    loaded = load_reconfirmation_manifest(
        manifest_path,
        root=repo,
        verify_code=False,
    )
    _, witnesses = load_reconfirmation_registry(registry_path)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert [len(item.steps) for item in witnesses] == [64, 61]
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert loaded["firewall"]["option_extraction_authorized"] is False
    assert loaded["firewall"]["t12_4a_3_option_freeze_authorized"] is False
    assert loaded["firewall"]["neural_training_authorized"] is False


def test_reconfirmation_run_is_paired_exact_and_authorizes_only_option_freeze(
    monkeypatch,
    tmp_path,
) -> None:
    witnesses = (_witness(1, 8701), _witness(2, 8705))
    selected = SimpleNamespace(
        source_seeds=(8701, 8705),
        expected_route_lengths=(3, 3),
        expected_common_suffix=("ACTION3", "ACTION3"),
        repetitions_per_route=3,
        repetitions_per_suffix_branch=3,
        minimum_successful_route_replays=3,
        minimum_successful_suffix_replays=3,
        minimum_paired_contrasts=3,
        minimum_step_exact_rate=1.0,
        maximum_sdk_calls=2_048,
        maximum_artifact_bytes_per_run=3 * 1024**3,
    )
    manifest = {
        "manifest_checksum": "m" * 64,
        "protocol_checksum": "p" * 64,
        "protocol": {},
        "scientific_claims_authorized": True,
        "witness_registry": {
            "path": "unused.json",
            "sha256": "w" * 64,
            "registry_checksum": "g" * 64,
        },
        "parent": {
            "receipt": {
                "receipt_checksum": "r" * 64,
                "status": "FAIL_T12_4A_1_CALIBRATION_GATE",
                "failure_class": "GLOBAL_CALIBRATOR_TRANSPORT_FAILURE",
            },
            "collection_receipt": {"receipt_checksum": "c" * 64},
        },
        "firewall": {"witness_reconfirmation_authorized": True},
    }
    monkeypatch.setattr(experiment, "load_reconfirmation_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(
        experiment,
        "load_reconfirmation_registry",
        lambda *a, **k: ({}, witnesses),
    )
    monkeypatch.setattr(
        experiment,
        "WitnessReconfirmationProtocol",
        lambda **kwargs: selected,
    )
    monkeypatch.setattr(witness_experiment, "_reset_env", lambda env: env.step(0))
    output = tmp_path / "confirmation"
    report = experiment.run_witness_reconfirmation(
        manifest_path="unused.json",
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: TwoStepSuffixEnv(),
    )
    assert report["passed"] is True
    assert report["status"] == "PASS_T12_4A_2_WITNESS_GATE"
    assert report["metrics"]["step_exact_rate"] == 1.0
    for item in report["metrics"]["per_witness"]:
        assert item["route_confirmations"] == 3
        assert item["suffix_confirmations"] == 3
        assert item["deletion_control_exact_no_progress"] == 3
        assert item["paired_contrast_confirmations"] == 3
    assert (output / "witness_receipt.json").is_file()
    assert (output / "intervention_bundles.json").is_file()


def test_protocol_is_strictly_bounded_and_cli_has_no_extraction_phase() -> None:
    protocol = WitnessReconfirmationProtocol()
    assert protocol.minimum_successful_route_replays == 3
    assert protocol.minimum_successful_suffix_replays == 3
    assert protocol.minimum_paired_contrasts == 3
    assert protocol.minimum_step_exact_rate == 1.0
    assert protocol.maximum_sdk_calls == 2_048
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "parent.json",
            "--parent-receipt",
            "receipt.json",
            "--collection-receipt",
            "collection.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["extract-option"])
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
