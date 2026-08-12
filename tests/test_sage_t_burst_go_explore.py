from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import theory.sage_t.causal.burst_protocol as burst_protocol
from theory.sage_t.causal.burst_experiment import (
    _paired_gate,
    run_burst_experiment,
    run_burst_go_explore_arm,
)
from theory.sage_t.causal.burst_experiment_cli import build_parser
from theory.sage_t.causal.burst_protocol import (
    BurstExploreProtocol,
    freeze_burst_experiment,
    load_burst_manifest,
    load_burst_receipt,
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


class FourStepProgressEnv:
    def __init__(self) -> None:
        self._game = FakeGame()
        self.count = 0

    def step(self, action, data=None):
        del data
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.count = 0
            return FakeFrame(np.zeros((3, 3), dtype=np.int32))
        self.count += 1
        level = int(self.count >= 4)
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[1, 1] = self.count
        return FakeFrame(
            grid,
            state="WIN" if level else "NOT_FINISHED",
            levels_completed=level,
        )


def test_burst_runner_reuses_one_restore_for_a_four_action_progression() -> None:
    run = run_burst_go_explore_arm(
        game_id="bp35",
        seed=6501,
        sdk_call_budget=12,
        environments_dir="unused",
        env_factory=lambda game_id: FourStepProgressEnv(),
        maximum_cells=32,
    )
    assert run.archive.metrics()["progress_edges"] >= 1
    assert run.archive.metrics()["replay_exact_rate"] == 1.0
    assert run.excursions[0].requested_horizon == 4
    assert run.excursions[0].executed_actions == 4
    assert run.excursions[0].restoration_calls == 1
    assert run.excursions[0].stopped_reason == "progress"


def _arm(*, cells: int, edges: int, progress: int = 0, terminal: int = 0):
    return {
        "metrics": {
            "symbolic_cells_per_1000_sdk_calls": cells / 8.192,
            "exploration_action_fraction": edges / 8192,
            "progress_edges": progress,
            "terminal_edges": terminal,
            "exploration_actions": edges,
            "replay_exact_rate": 1.0,
        }
    }


def test_preregistered_gate_needs_efficiency_coverage_and_progress() -> None:
    conditions = [
        {
            "seed": seed,
            "arms": {
                "one_step_archive": _arm(cells=100, edges=500),
                "burst_archive": _arm(
                    cells=140,
                    edges=1500,
                    progress=int(seed == 6501),
                    terminal=10,
                ),
            },
        }
        for seed in (6501, 6502, 6503)
    ]
    passed, metrics = _paired_gate(conditions, BurstExploreProtocol())
    assert passed
    assert metrics["aggregate_action_efficiency_ratio"] == 3.0
    assert metrics["aggregate_relative_coverage_gain"] >= 0.25
    assert metrics["burst_progress_edges"] == 1

    no_progress = [
        {
            **condition,
            "arms": {
                **condition["arms"],
                "burst_archive": _arm(cells=140, edges=1500),
            },
        }
        for condition in conditions
    ]
    assert not _paired_gate(no_progress, BurstExploreProtocol())[0]


def test_freeze_is_bound_to_failed_t12_1_parent(monkeypatch, tmp_path) -> None:
    repo = Path(__file__).resolve().parents[1]
    parent_root = repo / "training" / "sage_t" / "graph_explore_t12_1_bp35"
    monkeypatch.setattr(
        burst_protocol,
        "_git_state",
        lambda root: {"commit": "b" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_burst_experiment(
        output_path=manifest_path,
        game_id="bp35",
        parent_manifest_path=parent_root / "manifest.json",
        parent_receipt_path=parent_root / "archive" / "archive_receipt.json",
        root=repo,
    )
    loaded = load_burst_manifest(manifest_path, root=repo, verify_code=False)
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["passed"] is False
    assert loaded["parent"]["receipt"]["status"] == "FAIL_ARCHIVE_GATE"
    assert loaded["protocol"]["burst_schedule"] == [4, 8, 16]
    assert loaded["storage"]["maximum_artifact_bytes_per_run"] == 3 * 1024**3


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


def test_small_paired_run_emits_a_signed_bounded_receipt(
    monkeypatch, tmp_path
) -> None:
    repo = Path(__file__).resolve().parents[1]
    parent_root = repo / "training" / "sage_t" / "graph_explore_t12_1_bp35"
    monkeypatch.setattr(
        burst_protocol,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    freeze_burst_experiment(
        output_path=manifest_path,
        game_id="bp35",
        parent_manifest_path=parent_root / "manifest.json",
        parent_receipt_path=parent_root / "archive" / "archive_receipt.json",
        root=repo,
        protocol=BurstExploreProtocol(sdk_call_budget_per_seed_arm=32),
    )
    output = tmp_path / "paired"
    report = run_burst_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        environments_dir="unused",
        env_factory=lambda game_id: FourStepProgressEnv(),
    )
    receipt = load_burst_receipt(
        output / "burst_receipt.json",
        manifest=load_burst_manifest(manifest_path, root=repo, verify_code=False),
    )
    assert report["storage"]["within_budget"]
    assert receipt["phase"] == "burst_archive"
    assert receipt["parent_t12_1_receipt_checksum"]
    assert (output / "intervention_bundles.json").is_file()
