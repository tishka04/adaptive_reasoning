from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage.live_prefix_counterfactual_collector import state_signature_from_frame
from theory.sage_t.causal import provenance_protocol
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.provenance_experiment import (
    ConfirmedControlTrial,
    _aggregate_gate,
    replay_confirmed_control,
)
from theory.sage_t.causal.provenance_experiment_cli import build_parser
from theory.sage_t.causal.provenance_protocol import (
    ConfirmedControlProtocol,
    ConfirmedReplayControl,
    _confirmed_controls,
    _parent_failure_is_control_provenance_only,
)
from theory.sage_t.causal.lineage_protocol import (
    load_lineage_manifest,
    load_lineage_receipt,
)
from theory.sage_t.causal.witness_protocol import (
    load_witness_manifest,
    load_witness_receipt,
    load_witness_registry,
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


class DeterministicEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.count = 0

    def step(self, action, data=None):
        del data
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.count = 0
        else:
            self.count += 1
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = self.count
        return FakeFrame(grid)


def _synthetic_control(control_id: str, witness_id: str) -> ConfirmedReplayControl:
    env = DeterministicEnv()
    frame = env.step(0)
    hashes = [state_signature_from_frame(frame)]
    actions = (GroundedAction("ACTION1"), GroundedAction("ACTION1"))
    for _ in actions:
        frame = env.step(1)
        hashes.append(state_signature_from_frame(frame))
    return ConfirmedReplayControl(
        control_id=control_id,
        witness_id=witness_id,
        game_id="bp35",
        route_checksum=(control_id[-1] * 64),
        source_seed=6501,
        source_arm="burst_archive",
        prior_route_confirmations=3,
        actions=actions,
        expected_hashes=tuple(hashes),
    )


def test_real_t12_3a_witnesses_are_two_unique_confirmed_controls() -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "training" / "sage_t" / "progress_witness_t12_3a_bp35"
    manifest = load_witness_manifest(source / "manifest.json", root=repo)
    receipt = load_witness_receipt(
        source / "confirmation" / "witness_receipt.json",
        manifest=manifest,
        root=repo,
    )
    _, witnesses = load_witness_registry(source / "witnesses.sealed.json")
    controls = _confirmed_controls(
        witnesses, receipt, protocol=ConfirmedControlProtocol()
    )
    assert len(controls) == 2
    assert {item.depth for item in controls} == {36, 63}
    assert len({item.route_checksum for item in controls}) == 2
    assert len({item.control_checksum for item in controls}) == 2
    assert all(item.prior_route_confirmations == 3 for item in controls)


def test_t12_3c_parent_failure_is_control_provenance_only() -> None:
    repo = Path(__file__).resolve().parents[1]
    parent = repo / "training" / "sage_t" / "replay_lineage_t12_3c_bp35"
    manifest = load_lineage_manifest(parent / "manifest.json", root=repo)
    receipt = load_lineage_receipt(
        parent / "paired" / "lineage_receipt.json",
        manifest=manifest,
        root=repo,
    )
    assert _parent_failure_is_control_provenance_only(manifest, receipt)


def test_confirmed_control_is_checked_after_every_action(monkeypatch) -> None:
    control = _synthetic_control("control-a", "witness-a")
    monkeypatch.setattr(
        "theory.sage_t.causal.provenance_experiment._reset_env",
        lambda env: env.step(0),
    )
    trial = replay_confirmed_control(
        control=control,
        repetition=0,
        environments_dir="unused",
        env_factory=lambda game_id: DeterministicEnv(),
    )
    assert trial.exact
    assert trial.calls == 3
    assert len(trial.events) == 3


def _arm(*, exact: float, cells: float, progress: int, treatment: bool):
    metrics = {
        "replay_exact_rate": exact,
        "symbolic_cells_per_1000_sdk_calls": cells,
        "progress_edges": progress,
        "sdk_calls": 100,
    }
    if treatment:
        metrics.update(
            {
                "lineage_attached_transitions": 40,
                "shortest_prefix_rebases_avoided": 2,
                "lineage_rebased_transitions": 0,
            }
        )
    return {"metrics": metrics}


def _trial(control: ConfirmedReplayControl, repetition: int, *, exact: bool = True):
    return ConfirmedControlTrial(
        control_id=control.control_id,
        witness_id=control.witness_id,
        repetition=repetition,
        exact=exact,
        calls=3,
        first_divergence_step=None if exact else 1,
        first_divergence_kind="" if exact else "state_hash",
        expected_hash="expected",
        observed_hash="expected" if exact else "observed",
        events=(),
    )


def test_gate_requires_confirmed_controls_and_fresh_seed_non_regression() -> None:
    protocol = ConfirmedControlProtocol()
    controls = (
        _synthetic_control("control-a", "witness-a"),
        _synthetic_control("control-b", "witness-b"),
    )
    trials = tuple(
        _trial(control, repetition)
        for control in controls
        for repetition in range(protocol.control_repetitions)
    )
    conditions = tuple(
        {
            "seed": seed,
            "arms": {
                "shortest_prefix_control": _arm(
                    exact=0.90,
                    cells=100.0,
                    progress=0,
                    treatment=False,
                ),
                "lineage_preserving": _arm(
                    exact=1.0,
                    cells=90.0,
                    progress=0,
                    treatment=True,
                ),
            },
        }
        for seed in protocol.evaluation_seeds
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        controls=controls,
        control_trials=trials,
        conditions=conditions,
        sdk_calls=1_000,
    )
    assert passed
    assert metrics["confirmed_control_exact_rate"] == 1.0
    assert metrics["unique_route_checksums"] == 2
    assert metrics["replay_regression_seeds"] == 0

    failed_control_trials = (*trials[:-1], _trial(controls[1], 2, exact=False))
    assert not _aggregate_gate(
        protocol=protocol,
        controls=controls,
        control_trials=failed_control_trials,
        conditions=conditions,
        sdk_calls=1_000,
    )[0]

    regressed = list(conditions)
    regressed[0] = {
        **regressed[0],
        "arms": {
            **regressed[0]["arms"],
            "shortest_prefix_control": _arm(
                exact=1.0,
                cells=100.0,
                progress=1,
                treatment=False,
            ),
            "lineage_preserving": _arm(
                exact=0.99,
                cells=90.0,
                progress=0,
                treatment=True,
            ),
        },
    }
    assert not _aggregate_gate(
        protocol=protocol,
        controls=controls,
        control_trials=trials,
        conditions=regressed,
        sdk_calls=1_000,
    )[0]


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
        parser.parse_args(["neural"])


def test_freeze_binds_failed_t12_3c_and_passed_t12_3a(
    monkeypatch, tmp_path
) -> None:
    repo = Path(__file__).resolve().parents[1]
    parent = repo / "training" / "sage_t" / "replay_lineage_t12_3c_bp35"
    monkeypatch.setattr(
        provenance_protocol,
        "_git_state",
        lambda root: {"commit": "d" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "confirmed_controls.sealed.json"
    manifest = provenance_protocol.freeze_provenance_experiment(
        output_path=manifest_path,
        control_registry_path=registry_path,
        parent_manifest_path=parent / "manifest.json",
        parent_receipt_path=parent / "paired" / "lineage_receipt.json",
        root=repo,
    )
    loaded = provenance_protocol.load_provenance_manifest(
        manifest_path, root=repo, verify_code=False
    )
    _, controls = provenance_protocol.load_provenance_registry(registry_path)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["failure_class"] == (
        "CONFIRMED_CONTROL_PROVENANCE_ONLY"
    )
    assert loaded["control_source_t12_3a"]["receipt"]["passed"] is True
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert loaded["firewall"]["neural_training_authorized"] is False
    assert len(controls) == 2
