from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from theory.sage_t.causal import calibration_experiment, calibration_protocol
from theory.sage_t.causal.archive import abstract_state_to_payload
from theory.sage_t.causal.calibration_experiment import (
    PlattCalibrator,
    _fit_calibrator,
    _probabilities,
    train_calibration_experiment,
)
from theory.sage_t.causal.calibration_experiment_cli import build_parser
from theory.sage_t.causal.calibration_protocol import (
    CalibrationProtocol,
    _parent_failure_is_calibration_only,
    load_calibration_manifest,
)
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.relational_novelty import ArchiveContext
from theory.sage_t.contracts import AbstractEntity, AbstractState, GroundFact


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _state(index: int) -> AbstractState:
    return AbstractState(
        entities=(
            AbstractEntity(
                "player",
                ("object", "player"),
                (("area", "one"),),
                (float(index % 8), float(index % 8)),
            ),
            AbstractEntity(
                "target",
                ("object", "target"),
                (("area", "small"),),
                (5.0, 7.0),
            ),
        ),
        true_facts=frozenset({GroundFact("changed", (f"e{index % 3}",))}),
    )


def test_protocol_has_disjoint_prospective_three_way_split() -> None:
    protocol = CalibrationProtocol()
    train = set(protocol.training_seeds)
    calibration = {protocol.calibration_seed}
    validation = set(protocol.validation_seeds)
    assert len(train) == 3
    assert len(validation) == 2
    assert not train & calibration
    assert not train & validation
    assert not calibration & validation
    assert train | calibration | validation == set(protocol.collection_seeds)
    assert not set(protocol.collection_seeds) & set(protocol.opened_seeds_excluded)
    assert len(protocol.collection_seeds) * protocol.sdk_calls_per_seed <= 26_000
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3


def test_real_t12_4a_failure_is_calibration_only() -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "representation_t12_4a_bp35"
    manifest = calibration_protocol.load_representation_manifest(
        parent / "manifest.json",
        root=repo,
    )
    receipt = calibration_protocol.load_representation_receipt(
        parent / "training" / "representation_receipt.json",
        manifest=manifest,
        root=repo,
    )
    assert _parent_failure_is_calibration_only(manifest, receipt)
    assert receipt["metrics"]["maximum_ece"] > manifest["protocol"]["maximum_ece"]


def test_freeze_binds_negative_parent_and_keeps_authority_closed(
    monkeypatch,
    tmp_path,
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "representation_t12_4a_bp35"
    monkeypatch.setattr(
        calibration_protocol,
        "_git_state",
        lambda root: {"commit": "b" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = calibration_protocol.freeze_calibration_experiment(
        output_path=manifest_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "training" / "representation_receipt.json",
        root=repo,
    )
    loaded = load_calibration_manifest(
        manifest_path,
        root=repo,
        verify_code=False,
    )
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["failure_class"] == (
        "CALIBRATION_TRANSPORT_ONLY"
    )
    assert loaded["firewall"]["calibration_collection_authorized"] is True
    assert loaded["firewall"]["calibration_training_authorized"] is False
    assert loaded["firewall"]["neural_active_evaluation_authorized"] is False
    assert loaded["firewall"]["option_extraction_authorized"] is False
    assert loaded["firewall"]["t12_4b_freeze_authorized"] is False


def test_platt_calibrator_is_small_monotone_and_reduces_biased_ece() -> None:
    logits = torch.tensor(
        [[value, value] for value in (-4.0, -2.0, 0.0, 2.0, 4.0) for _ in range(20)],
        dtype=torch.float32,
    )
    rates = (0.20, 0.30, 0.50, 0.70, 0.80)
    targets = []
    for rate in rates:
        positives = round(rate * 20)
        targets.extend([[float(index < positives)] * 2 for index in range(20)])
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    protocol = SimpleNamespace(
        torch_seed=7,
        calibration_learning_rate=0.03,
        calibration_steps=500,
    )
    calibrator = _fit_calibrator(
        logits,
        target_tensor,
        protocol=protocol,
        seed_offset=0,
    )
    assert isinstance(calibrator, PlattCalibrator)
    assert sum(parameter.numel() for parameter in calibrator.parameters()) == 4
    assert all(value > 0.0 for value in calibrator.scale.tolist())
    before = _probabilities(logits)
    after = _probabilities(logits, calibrator)
    flat_targets = [(float(item[0]), float(item[1])) for item in targets]
    before_ece = calibration_experiment._maximum_ece(before, flat_targets)[2]
    after_ece = calibration_experiment._maximum_ece(after, flat_targets)[2]
    assert after_ece < before_ece


def test_training_never_fits_confirmation_and_emits_gated_receipt(
    monkeypatch,
    tmp_path,
) -> None:
    states = {f"s{index}": abstract_state_to_payload(_state(index)) for index in range(8)}
    examples = []
    split_seed = (
        [("train", 8701)] * 40
        + [("train", 8702)] * 40
        + [("train", 8703)] * 40
        + [("calibration", 8704)] * 40
        + [("validation", 8705)] * 40
        + [("validation", 8706)] * 40
    )
    for index, (split, seed) in enumerate(split_seed):
        positive = bool(index % 2)
        action = GroundedAction(
            f"ACTION{1 + index % 2}",
            {"x": 7 if positive else 40, "y": 5 if positive else 40},
        )
        examples.append(
            {
                "seed": seed,
                "split": split,
                "source_state_id": f"s{index % 8}",
                "action": {
                    "action_name": action.action_name,
                    "action_data": dict(action.action_data),
                },
                "action_key": action.key,
                "archive_context": ArchiveContext(
                    cell_visits=1 + index % 8,
                    action_attempts=index % 4,
                    cell_expansions=index % 9,
                    unique_tried_actions=index % 2,
                    legal_actions=2,
                    archive_cells=1 + index,
                    global_edges=index,
                    global_action_trials=index % 12,
                    global_action_changed=index % 6,
                    global_action_novel=index % 4,
                    cell_action_trials=index % 4,
                    cell_action_changed=index % 3,
                    cell_action_novel=index % 2,
                ).to_dict(),
                "semantic_changed": positive,
                "novel": positive,
                "example_id": f"example-{index}",
            }
        )
    dataset = {
        "states": states,
        "examples": examples,
        "dataset_checksum": "d" * 64,
    }
    manifest = {
        "manifest_checksum": "m" * 64,
        "protocol_checksum": "p" * 64,
        "protocol": {},
        "parent": {"receipt": {"receipt_checksum": "r" * 64}},
    }
    collection_receipt = {
        "receipt_checksum": "c" * 64,
        "status": "PASS_T12_4A_1_COLLECTION_GATE",
        "artifacts": {
            "dataset": {
                "path": "unused.json",
                "sha256": "d" * 64,
                "dataset_checksum": "d" * 64,
            }
        },
    }
    protocol = SimpleNamespace(
        training_seeds=(8701, 8702, 8703),
        calibration_seed=8704,
        validation_seeds=(8705, 8706),
        hidden_dim=8,
        batch_size=16,
        training_epochs=1,
        learning_rate=1e-3,
        torch_seed=1,
        calibration_steps=20,
        calibration_learning_rate=0.03,
        minimum_training_examples=1,
        minimum_calibration_examples=1,
        minimum_validation_examples=1,
        maximum_parameters=15_000,
        maximum_calibration_parameters=4,
        minimum_change_brier_gain=-1.0,
        minimum_novelty_brier_gain=-1.0,
        minimum_legacy_mean_brier_improvement=-1.0,
        minimum_state_shuffle_change_degradation=-1.0,
        minimum_context_shuffle_novelty_degradation=-1.0,
        minimum_relation_ablation_degradation=-1.0,
        minimum_calibration_ece_improvement=-1.0,
        maximum_calibrated_brier_regression=1.0,
        maximum_pooled_ece=1.0,
        maximum_per_seed_ece=1.0,
        maximum_artifact_bytes_per_run=3 * 1024**3,
    )
    monkeypatch.setattr(
        calibration_experiment,
        "load_calibration_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        calibration_experiment,
        "load_calibration_receipt",
        lambda *args, **kwargs: collection_receipt,
    )
    monkeypatch.setattr(
        calibration_experiment,
        "load_calibration_dataset",
        lambda path, protocol: dataset,
    )
    monkeypatch.setattr(
        calibration_experiment,
        "CalibrationProtocol",
        lambda **kwargs: protocol,
    )
    report = train_calibration_experiment(
        manifest_path="unused.json",
        collection_receipt_path="unused-receipt.json",
        output_dir=tmp_path / "training",
    )
    assert report["passed"] is True
    assert report["metrics"]["training_seed_count"] == 3
    assert report["metrics"]["calibration_seed_count"] == 1
    assert report["metrics"]["validation_seed_count"] == 2
    assert report["metrics"]["calibration_parameter_count"] == 4
    assert set(report["metrics"]["per_seed_ece"]) == {"8705", "8706"}
    assert (tmp_path / "training" / "calibration_receipt.json").is_file()


def test_cli_exposes_no_active_or_option_phase() -> None:
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
    assert parser.parse_args(["collect"]).phase == "collect"
    assert parser.parse_args(["train"]).phase == "train"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["extract-option"])
