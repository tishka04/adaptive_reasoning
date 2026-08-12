from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import theory.sage_t.causal.graph_protocol as graph_protocol
from theory.sage_t.causal.graph_experiment import (
    _intervention_bundles,
    run_go_explore_arm,
)
from theory.sage_t.causal.graph_experiment_cli import build_parser
from theory.sage_t.causal.graph_protocol import (
    GraphExploreProtocol,
    freeze_graph_experiment,
    load_graph_manifest,
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
    available_actions: tuple[int, ...] = (1, 2)


class FakeGame:
    def _get_valid_actions(self):
        return [FakeAction(1), FakeAction(2)]


class FakeEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.grid = np.zeros((5, 5), dtype=np.int32)
        self.level = 0

    def step(self, action, data=None):
        del data
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.grid.fill(0)
            self.level = 0
            return FakeFrame(self.grid.copy())
        if value == 1:
            self.grid[2, 2] = 1
            self.level = 1
            return FakeFrame(self.grid.copy(), "WIN", self.level)
        self.grid[2, 3] = 2
        return FakeFrame(self.grid.copy(), levels_completed=self.level)


def test_cli_exposes_the_preregistered_phase_chain() -> None:
    parser = build_parser()
    for phase in (
        "freeze",
        "archive",
        "shield",
        "train-novelty",
        "neural",
        "extract-option",
        "compile-option",
        "transfer",
        "status",
    ):
        suffix = (
            ["--stage", "source_train", "--games", "bp35"]
            if phase == "freeze"
            else []
        )
        assert parser.parse_args([phase, *suffix]).phase == phase


def test_freeze_binds_three_gibibyte_cap_and_closed_firewall(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        graph_protocol,
        "_git_state",
        lambda root: {"commit": "a" * 40, "dirty": False, "dirty_entries": 0},
    )
    registry = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "sage_t"
        / "causal_inputs"
        / "programs.sealed.json"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_graph_experiment(
        output_path=manifest_path,
        stage="source_train",
        game_ids=("bp35",),
        program_registry_path=registry,
    )
    loaded = load_graph_manifest(manifest_path, verify_code=False)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3
    assert not any(loaded["firewall"].values())
    assert GraphExploreProtocol().terminal_horizon == 64


def test_symbolic_archive_runner_uses_exact_replay_with_injected_env() -> None:
    archive, _ = run_go_explore_arm(
        game_id="bp35",
        sdk_call_budget=16,
        environments_dir="unused",
        env_factory=lambda game_id: FakeEnv(),
        maximum_cells=16,
    )
    assert archive.sdk_calls <= 16
    assert archive.metrics()["progress_edges"] >= 1
    assert archive.metrics()["replay_exact_rate"] == 1.0
    bundles = _intervention_bundles(archive, game_id="bp35", seed=0)
    assert bundles
    assert all(len(bundle["branches"]) >= 2 for bundle in bundles)
    assert all(bundle["prefix_hash"] for bundle in bundles)
