from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal import option_transfer_experiment as experiment
from theory.sage_t.causal import option_transfer_protocol as protocol_module
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.option_transfer_cli import build_parser
from theory.sage_t.causal.option_transfer_protocol import (
    OptionTransferProtocol,
    freeze_option_transfer,
    load_option_transfer_manifest,
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


class TransferEnv:
    sequence = (4, 4, 4, 3, 3)

    def __init__(self, *, maximum_level: int = 4) -> None:
        self._game = FakeGame()
        self.maximum_level = maximum_level
        self.level = 0
        self.history: list[int] = []

    def step(self, action, data=None):
        del data
        value = int(getattr(action, "value", action))
        if value == 0:
            self.level = 0
            self.history = []
        elif self.level == 0 and value in {1, 2}:
            self.level = 1
            self.history = []
        else:
            self.history.append(value)
            if tuple(self.history) == self.sequence and self.level < self.maximum_level:
                self.level += 1
                self.history = []
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[0, 0] = self.level
        for index, item in enumerate(self.history[:5]):
            grid[1 + index // 3, index % 3] = item
        return FakeFrame(grid, levels_completed=self.level)


def _witness(first_action: int, seed: int) -> ProgressWitness:
    env = TransferEnv()
    frame = env.step(0)
    source_hash = state_signature_from_frame(frame)
    frame = env.step(first_action)
    step = WitnessStep(
        expected_source_hash=source_hash,
        action=GroundedAction(f"ACTION{first_action}"),
        expected_target_hash=state_signature_from_frame(frame),
        level_delta=1,
        success=True,
    )
    return ProgressWitness(
        witness_id=f"witness-{seed}",
        game_id="bp35",
        source_seed=seed,
        source_arm="lineage_control",
        source_archive_sha256=str(seed) * 8,
        source_progress_edge_id=f"edge-{seed}",
        initial_exact_hash=step.expected_source_hash,
        initial_level=0,
        target_exact_hash=step.expected_target_hash,
        target_level=1,
        steps=(step,),
    )


def _option() -> MinimalCausalOption:
    return MinimalCausalOption(
        initiation_signature="synthetic-entry",
        initiation_exact_hash="synthetic-exact-entry",
        steps=tuple(
            MinimalOptionStep(f"ACTION{action}", {})
            for action in TransferEnv.sequence
        ),
        source_evidence_ids=("witness-8701", "witness-8705"),
        reproduction_count=8,
        minimization_evaluations=384,
        source="t12_4a_4_test",
    )


def _synthetic_manifest() -> dict:
    protocol = OptionTransferProtocol()
    witnesses = (_witness(1, 8701), _witness(2, 8705))
    return {
        "firewall": {"option_transfer_experiment_authorized": True},
        "game_id": "bp35",
        "inputs": {
            "entry_exact_hash": witnesses[0].target_exact_hash,
            "entry_level": 1,
            "option_checksum": _option().checksum,
            "posterior_owner_mass": 1.0,
        },
        "manifest_checksum": "m" * 64,
        "parent": {
            "ablation_receipt": {"receipt_checksum": "a" * 64},
            "compile_receipt": {
                "receipt_checksum": "c" * 64,
                "status": "PASS_T12_4A_3_SHADOW_COMPILE_GATE",
            },
        },
        "protocol": asdict(protocol),
        "protocol_checksum": protocol.checksum,
    }


def test_protocol_is_bounded_paired_and_cli_has_no_activation() -> None:
    protocol = OptionTransferProtocol()
    assert protocol.minimum_transferred_levels == 2
    assert protocol.maximum_transfer_levels == 3
    assert protocol.repetitions_per_branch == 4
    assert protocol.maximum_sdk_calls == 4_500
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "manifest.json",
            "--ablation-receipt",
            "ablation.json",
            "--compile-receipt",
            "compile.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["run"]).phase == "run"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])


def test_freeze_binds_real_t12_4a_3_chain_and_keeps_authority_closed(
    monkeypatch,
    tmp_path,
) -> None:
    repo = _repo()
    parent = repo / "training" / "sage_t" / "option_minimization_t12_4a_3r1_bp35"
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_option_transfer(
        output_path=manifest_path,
        parent_manifest_path=parent / "manifest.json",
        ablation_receipt_path=parent / "ablation" / "option_ablation_receipt.json",
        compile_receipt_path=parent / "shadow_compile" / "shadow_compile_receipt.json",
        root=repo,
    )
    loaded = load_option_transfer_manifest(manifest_path, root=repo)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["inputs"]["option_checksum"] == (
        "9e03be5f720532883a424cd58f6e02b0d8eed69b63498a4c170220f863653851"
    )
    assert loaded["inputs"]["route_lengths"] == [64, 61]
    assert loaded["firewall"]["option_transfer_experiment_authorized"] is True
    assert loaded["firewall"]["option_control_authorized"] is False
    assert loaded["firewall"]["t12_4a_5_option_control_freeze_authorized"] is False


def test_transfer_confirms_three_levels_with_strict_controls(
    monkeypatch,
    tmp_path,
) -> None:
    witnesses = (_witness(1, 8701), _witness(2, 8705))
    manifest = _synthetic_manifest()
    monkeypatch.setattr(
        experiment,
        "load_option_transfer_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_load_inputs",
        lambda *args, **kwargs: (witnesses, _option()),
    )
    output = tmp_path / "transfer"
    receipt = experiment.run_option_transfer(
        manifest_path="unused.json",
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: TransferEnv(maximum_level=4),
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_4A_4_OPTION_TRANSFER_GATE"
    assert receipt["metrics"]["confirmed_transfer_levels"] == 3
    assert receipt["metrics"]["trial_count"] == 60
    assert receipt["metrics"]["sdk_calls"]["used_sdk_calls"] == 636
    assert receipt["metrics"]["terminal_failures"] == 0
    assert all(
        stage["passed"] for stage in receipt["metrics"]["stage_results"]
    )
    assert (output / "transfer_receipt.json").is_file()
    assert (output / "intervention_bundles.json").is_file()


def test_transfer_gate_passes_at_two_levels_and_records_third_level_limit(
    monkeypatch,
    tmp_path,
) -> None:
    witnesses = (_witness(1, 8701), _witness(2, 8705))
    manifest = _synthetic_manifest()
    monkeypatch.setattr(
        experiment,
        "load_option_transfer_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        experiment,
        "_load_inputs",
        lambda *args, **kwargs: (witnesses, _option()),
    )
    receipt = experiment.run_option_transfer(
        manifest_path="unused.json",
        output_dir=tmp_path / "transfer-limit",
        environments_dir="unused",
        env_factory=lambda game_id: TransferEnv(maximum_level=3),
    )
    assert receipt["passed"] is True
    assert receipt["metrics"]["confirmed_transfer_levels"] == 2
    assert receipt["metrics"]["attempted_transfer_levels"] == 3
    assert receipt["metrics"]["stage_results"][-1]["passed"] is False
    assert receipt["metrics"]["final_confirmed_level"] == 3
