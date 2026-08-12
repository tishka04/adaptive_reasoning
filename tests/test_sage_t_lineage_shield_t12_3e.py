from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage_t.causal import lineage_shield_protocol
from theory.sage_t.causal.lineage_protocol import load_lineage_manifest
from theory.sage_t.causal.lineage_shield_experiment import (
    _aggregate_gate,
    run_lineage_shield_arm,
)
from theory.sage_t.causal.lineage_shield_experiment_cli import build_parser
from theory.sage_t.causal.lineage_shield_protocol import (
    LineageShieldProtocol,
    _source_shield_evidence,
    load_lineage_shield_manifest,
    load_lineage_shield_registry,
)
from theory.sage_t.causal.shield_experiment import WitnessShieldTrial
from theory.sage_t.causal.shield_model import ProgressProtectedTerminalShield
from theory.sage_t.causal.shield_protocol import (
    load_shield_manifest,
    load_shield_receipt,
)


@dataclass
class FakeAction:
    id: int
    data: dict | None = None


@dataclass
class FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1,)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(1)]


class ThreeStepTerminalEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.count = 0

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        self.count = 0 if value == 0 else self.count + 1
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = self.count
        return FakeFrame(
            grid,
            state="GAME_OVER" if self.count >= 3 else "NOT_FINISHED",
        )


class DenyAllShield:
    def __init__(self) -> None:
        self.vetoes = 0

    def allows(self, cell_id, action):
        del cell_id, action
        self.vetoes += 1
        return False


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_evidence() -> tuple[dict, ProgressProtectedTerminalShield]:
    repo = _repo()
    lineage_root = repo / "training" / "sage_t" / "replay_lineage_t12_3c_bp35"
    lineage_manifest = load_lineage_manifest(
        lineage_root / "manifest.json", root=repo, verify_code=False
    )
    shield_manifest = load_shield_manifest(
        repo / lineage_manifest["parent"]["manifest"]["path"],
        root=repo,
        verify_code=False,
    )
    shield_receipt = load_shield_receipt(
        repo / lineage_manifest["parent"]["receipt"]["path"],
        manifest=shield_manifest,
        root=repo,
    )
    evidence = _source_shield_evidence(
        shield_manifest=shield_manifest,
        shield_receipt=shield_receipt,
        root=repo,
        protocol=LineageShieldProtocol(),
    )
    payload_path = repo / evidence["terminal_shield"]["path"]
    shield = ProgressProtectedTerminalShield.from_dict(
        lineage_shield_protocol._read_json(payload_path)
    )
    return evidence, shield


def test_real_t12_3b_shield_inputs_are_complete_and_confirmed() -> None:
    evidence, shield = _source_evidence()
    assert len(evidence["terminal_candidate_ids"]) == 12
    assert evidence["protected_action_pairs"] == 99
    assert len(evidence["witness_ids"]) == 2
    assert shield.metrics()["confirmed_terminal_traces"] == 12
    assert shield.metrics()["confirmed_unsafe_actions"] == 177
    assert shield.metrics()["multi_step_hazard_observed"] is True


def test_lineage_runner_applies_shield_before_the_real_action() -> None:
    control, _ = run_lineage_shield_arm(
        game_id="bp35",
        seed=7701,
        sdk_call_budget=20,
        burst_schedule=(4, 8, 16),
        environments_dir="unused",
        env_factory=lambda game_id: ThreeStepTerminalEnv(),
    )
    deny = DenyAllShield()
    treatment, used = run_lineage_shield_arm(
        game_id="bp35",
        seed=7701,
        sdk_call_budget=20,
        burst_schedule=(4, 8, 16),
        environments_dir="unused",
        env_factory=lambda game_id: ThreeStepTerminalEnv(),
        shield=deny,  # type: ignore[arg-type]
    )
    control_metrics = control.metrics()
    treatment_metrics = treatment.metrics()
    assert control_metrics["terminal_edges"] >= 1
    assert control_metrics["lineage_attached_transitions"] >= 1
    assert control_metrics["lineage_rebased_transitions"] == 0
    assert treatment_metrics["terminal_edges"] == 0
    assert treatment_metrics["edges"] == 0
    assert used is deny
    assert deny.vetoes >= 1


def _arm(*, terminal: int, cells: int, progress: int, vetoes: int) -> dict:
    return {
        "metrics": {
            "terminal_edges": terminal,
            "exploration_actions": 100,
            "symbolic_cells": cells,
            "sdk_calls": 100,
            "progress_edges": progress,
            "replay_exact_rate": 1.0,
            "symbolic_cells_per_1000_sdk_calls": cells * 10.0,
            "lineage_attached_transitions": 100,
            "shortest_prefix_rebases_avoided": 1,
            "lineage_rebased_transitions": 0,
        },
        "shield_metrics": {"vetoes": vetoes},
    }


def _trial(witness_id: str, repetition: int) -> WitnessShieldTrial:
    return WitnessShieldTrial(
        witness_id=witness_id,
        repetition=repetition,
        exact=True,
        progressed=True,
        all_actions_protected=True,
        vetoed_actions=0,
        calls=2,
        final_exact_hash="exact",
        first_divergence="",
        events=(),
    )


def test_gate_requires_shield_effect_coverage_lineage_and_witness_safety() -> None:
    protocol = LineageShieldProtocol()
    evidence, shield = _source_evidence()
    trials = tuple(
        _trial(witness_id, repetition)
        for witness_id in evidence["witness_ids"]
        for repetition in range(protocol.witness_repetitions)
    )
    conditions = tuple(
        {
            "seed": seed,
            "arms": {
                "lineage_control": _arm(
                    terminal=10, cells=100, progress=1, vetoes=0
                ),
                "lineage_terminal_shield": _arm(
                    terminal=5, cells=90, progress=1, vetoes=2
                ),
            },
        }
        for seed in protocol.evaluation_seeds
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        source_registry=evidence,
        source_shield=shield,
        witness_trials=trials,
        conditions=conditions,
        sdk_calls=1_000,
    )
    assert passed
    assert metrics["terminal_rate_ratio"] == 0.5
    assert metrics["coverage_ratio"] == 0.9
    assert metrics["lineage_rebased_transitions"] == 0

    replay_regression = list(conditions)
    replay_regression[0] = {
        **replay_regression[0],
        "arms": {
            **replay_regression[0]["arms"],
            "lineage_terminal_shield": {
                **replay_regression[0]["arms"]["lineage_terminal_shield"],
                "metrics": {
                    **replay_regression[0]["arms"]["lineage_terminal_shield"][
                        "metrics"
                    ],
                    "replay_exact_rate": 0.94,
                },
            },
        },
    }
    assert not _aggregate_gate(
        protocol=protocol,
        source_registry=evidence,
        source_shield=shield,
        witness_trials=trials,
        conditions=replay_regression,
        sdk_calls=1_000,
    )[0]


def test_freeze_binds_passed_t12_3d_and_failed_replay_only_t12_3b(
    monkeypatch, tmp_path
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "confirmed_control_t12_3d_bp35"
    monkeypatch.setattr(
        lineage_shield_protocol,
        "_git_state",
        lambda root: {"commit": "e" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "shield_inputs.sealed.json"
    manifest = lineage_shield_protocol.freeze_lineage_shield_experiment(
        output_path=manifest_path,
        source_registry_path=registry_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "paired" / "provenance_receipt.json",
        root=repo,
    )
    loaded = load_lineage_shield_manifest(
        manifest_path, root=repo, verify_code=False
    )
    registry = load_lineage_shield_registry(registry_path)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "PASS_T12_3D_CONFIRMED_CONTROL_GATE"
    )
    assert loaded["source_t12_3b"]["receipt"]["failure_class"] == (
        "REPLAY_EXACT_ONLY"
    )
    assert len(registry["terminal_candidate_ids"]) == 12
    assert registry["protected_action_pairs"] == 99
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert loaded["firewall"]["terminal_shield_production_authority"] is False
    assert loaded["firewall"]["neural_training_authorized"] is False


def test_cli_exposes_only_freeze_run_and_status() -> None:
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
        parser.parse_args(["train"])
