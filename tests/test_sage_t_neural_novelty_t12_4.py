from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage_t.causal import neural_novelty_experiment, neural_novelty_protocol
from theory.sage_t.causal.archive import abstract_state_to_payload
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.neural_novelty_experiment import (
    _aggregate_active_gate,
    run_neural_novelty_arm,
    train_neural_novelty_experiment,
)
from theory.sage_t.causal.neural_novelty_experiment_cli import build_parser
from theory.sage_t.causal.neural_novelty_protocol import (
    NeuralNoveltyProtocol,
    compile_neural_novelty_dataset,
    load_neural_novelty_dataset,
    load_neural_novelty_manifest,
)
from theory.sage_t.causal.shield_model import ProgressProtectedTerminalShield
from theory.sage_t.contracts import AbstractState, GroundFact


@dataclass
class FakeAction:
    id: int
    data: dict | None = None


@dataclass
class FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1, 2)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(1), FakeAction(2)]


class TwoActionEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.count = 0

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        if value == 0:
            self.count = 0
        else:
            self.count += value
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = self.count
        return FakeFrame(
            grid,
            state="GAME_OVER" if self.count >= 5 else "NOT_FINISHED",
        )


class ActionTwoScorer:
    def score(self, state, action):
        del state
        return (1.0, 1.0) if action.action_name == "ACTION2" else (0.0, 0.0)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _paired_evaluation() -> dict:
    path = (
        _repo()
        / "training"
        / "sage_t"
        / "lineage_shield_t12_3e_bp35"
        / "paired"
        / "paired_evaluation.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_dataset_rejects_universal_pixel_change_and_uses_semantic_change() -> None:
    protocol = NeuralNoveltyProtocol()
    dataset = compile_neural_novelty_dataset(
        paired_evaluation=_paired_evaluation(),
        root=_repo(),
        protocol=protocol,
    )
    assert dataset["qa"]["all"]["raw_changed_prevalence"] == 1.0
    assert dataset["qa"]["raw_label_is_universal"] is True
    assert 0.05 < dataset["qa"]["train"]["semantic_changed_prevalence"] < 0.95
    assert 0.05 < dataset["qa"]["train"]["novelty_prevalence"] < 0.95
    assert dataset["qa"]["train"]["examples"] == 1_232
    assert dataset["qa"]["validation"]["examples"] == 236
    assert dataset["qa"]["train_validation_state_overlap"] == 5
    assert dataset["qa"]["validation_state_overlap_fraction"] < 0.05
    assert dataset["qa"]["validation_state_action_overlap_fraction"] < 0.02
    assert dataset["label_amendment"]["rejection_reason"] == (
        "universal_derived_label"
    )


def test_freeze_binds_passed_t12_3e_and_audited_dataset(
    monkeypatch, tmp_path
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "lineage_shield_t12_3e_bp35"
    monkeypatch.setattr(
        neural_novelty_protocol,
        "_git_state",
        lambda root: {"commit": "f" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    dataset_path = tmp_path / "dataset.sealed.json"
    manifest = neural_novelty_protocol.freeze_neural_novelty_experiment(
        output_path=manifest_path,
        dataset_path=dataset_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "paired" / "lineage_shield_receipt.json",
        root=repo,
    )
    loaded = load_neural_novelty_manifest(
        manifest_path, root=repo, verify_code=False
    )
    dataset = load_neural_novelty_dataset(dataset_path)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "PASS_T12_3E_LINEAGE_SHIELD_GATE"
    )
    assert loaded["firewall"]["neural_novelty_training_authorized"] is True
    assert loaded["firewall"]["neural_active_evaluation_authorized"] is False
    assert loaded["firewall"]["option_extraction_authorized"] is False
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert dataset["qa"]["raw_label_is_universal"] is True


def test_active_runner_uses_lineage_shield_and_fixed_neural_scorer() -> None:
    shield_payload = ProgressProtectedTerminalShield().to_dict()
    run, _, neural_metrics = run_neural_novelty_arm(
        game_id="bp35",
        seed=8101,
        sdk_call_budget=24,
        burst_schedule=(4, 8, 16),
        environments_dir="unused",
        shield_payload=shield_payload,
        env_factory=lambda game_id: TwoActionEnv(),
        novelty_scorer=ActionTwoScorer(),  # type: ignore[arg-type]
    )
    metrics = run.metrics()
    assert metrics["lineage_attached_transitions"] >= 1
    assert metrics["lineage_rebased_transitions"] == 0
    assert neural_metrics["scored_decisions"] >= 1
    assert neural_metrics["neural_action_changes"] >= 1
    assert neural_metrics["p95_decision_latency_ms"] >= 0.0


def test_training_phase_fits_writes_checkpoint_and_emits_receipt(
    monkeypatch, tmp_path
) -> None:
    states = {
        "s0": abstract_state_to_payload(
            AbstractState(true_facts=frozenset({GroundFact("changed", ("e0",))}))
        ),
        "s1": abstract_state_to_payload(
            AbstractState(true_facts=frozenset({GroundFact("changed", ("e1",))}))
        ),
    }
    examples = []
    for index in range(96):
        action = GroundedAction(f"ACTION{1 + index % 2}")
        examples.append(
            {
                "split": "train" if index < 64 else "validation",
                "source_state_id": f"s{index % 2}",
                "action": {
                    "action_name": action.action_name,
                    "action_data": {},
                },
                "semantic_changed": bool(index % 2),
                "novel": bool((index // 2) % 2),
                "example_id": f"example-{index}",
            }
        )
    dataset = {
        "dataset_checksum": "d" * 64,
        "states": states,
        "examples": examples,
        "qa": {
            "train": {
                "semantic_changed_prevalence": 0.5,
                "novelty_prevalence": 0.5,
            },
            "validation": {
                "semantic_changed_prevalence": 0.5,
                "novelty_prevalence": 0.5,
            },
        },
    }
    manifest = {
        "manifest_checksum": "m" * 64,
        "protocol_checksum": "p" * 64,
        "protocol": {},
        "dataset": {"path": "unused.json"},
        "parent": {"receipt": {"receipt_checksum": "r" * 64}},
        "scientific_claims_authorized": True,
        "firewall": {"neural_novelty_training_authorized": True},
    }
    protocol = SimpleNamespace(
        torch_seed=1,
        hidden_dim=8,
        batch_size=16,
        learning_rate=1e-3,
        training_epochs=1,
        minimum_training_examples=1,
        minimum_validation_examples=1,
        maximum_parameters=15_000,
        minimum_brier_gain=-1.0,
        minimum_state_shuffle_degradation=-1.0,
        maximum_ece=1.0,
        maximum_artifact_bytes_per_run=3 * 1024**3,
    )
    monkeypatch.setattr(
        neural_novelty_experiment,
        "load_neural_novelty_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        neural_novelty_experiment,
        "load_neural_novelty_dataset",
        lambda path, protocol: dataset,
    )
    monkeypatch.setattr(
        neural_novelty_experiment,
        "NeuralNoveltyProtocol",
        lambda **kwargs: protocol,
    )
    report = train_neural_novelty_experiment(
        manifest_path="unused.json",
        output_dir=tmp_path / "training",
    )
    assert report["passed"] is True
    assert report["metrics"]["training_examples"] == 64
    assert report["metrics"]["validation_examples"] == 32
    assert (tmp_path / "training" / "neural_novelty_predictor.pt").is_file()
    assert (tmp_path / "training" / "training_receipt.json").is_file()


def _arm(
    *,
    cells: int,
    terminal: int,
    progress: int,
    neural_changes: int,
) -> dict:
    return {
        "metrics": {
            "symbolic_cells": cells,
            "sdk_calls": 100,
            "symbolic_cells_per_1000_sdk_calls": cells * 10.0,
            "terminal_edges": terminal,
            "exploration_actions": 100,
            "progress_edges": progress,
            "replay_exact_rate": 1.0,
        },
        "shield_metrics": {"vetoes": 2},
        "neural_metrics": {
            "scored_decisions": 100 if neural_changes else 0,
            "neural_action_changes": neural_changes,
            "mean_decision_latency_ms": 1.0 if neural_changes else 0.0,
            "p95_decision_latency_ms": 2.0 if neural_changes else 0.0,
        },
    }


def test_active_gate_requires_utility_not_only_offline_prediction() -> None:
    protocol = NeuralNoveltyProtocol()
    conditions = tuple(
        {
            "seed": seed,
            "arms": {
                "lineage_shield_control": _arm(
                    cells=100,
                    terminal=10,
                    progress=1,
                    neural_changes=0,
                ),
                "lineage_shield_neural": _arm(
                    cells=115,
                    terminal=9,
                    progress=1,
                    neural_changes=3,
                ),
            },
        }
        for seed in protocol.evaluation_seeds
    )
    passed, metrics = _aggregate_active_gate(
        protocol=protocol,
        conditions=conditions,
        sdk_calls=1_000,
    )
    assert passed
    assert metrics["relative_coverage_gain"] == pytest.approx(0.15)
    assert metrics["terminal_rate_ratio"] == pytest.approx(0.9)

    no_utility = tuple(
        {
            **condition,
            "arms": {
                **condition["arms"],
                "lineage_shield_neural": _arm(
                    cells=100,
                    terminal=9,
                    progress=1,
                    neural_changes=3,
                ),
            },
        }
        for condition in conditions
    )
    assert not _aggregate_active_gate(
        protocol=protocol,
        conditions=no_utility,
        sdk_calls=1_000,
    )[0]


def test_cli_exposes_only_frozen_t12_4_phases() -> None:
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
    assert parser.parse_args(["train"]).phase == "train"
    assert parser.parse_args(["evaluate"]).phase == "evaluate"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["extract-option"])
