"""Executable/resumable SAGE.11 source-corpus runner tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from theory.sage11.source_dataset_runner import (
    _completed_game_checkpoint,
    _overflow_allocation,
    _partial_game_checkpoint,
    run_source_dataset_collection,
    verify_source_dataset,
    write_capacity_report,
)
from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION


@dataclass
class _FakeAction:
    id: int
    data: dict | None = None


@dataclass
class _FakeFrame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1, 2, 3)


class _FakeGame:
    def _get_valid_actions(self):
        return [_FakeAction(1), _FakeAction(2), _FakeAction(3)]


class _UniqueStateEnv:
    def __init__(self) -> None:
        self._game = _FakeGame()
        self.step_index = 0
        self.grid = np.zeros((8, 8), dtype=np.int32)

    def step(self, action, data=None):
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.step_index = 0
            self.grid.fill(0)
            return _FakeFrame(self.grid.copy())
        self.step_index += 1
        self.grid[0, 0] = self.step_index
        self.grid[1, 1] = value
        return _FakeFrame(self.grid.copy())


def test_source_runner_collects_exact_mixture_and_resumes_completed_game(
    tmp_path: Path,
):
    created: list[str] = []

    def factory(game_id: str):
        created.append(game_id)
        return _UniqueStateEnv()

    output = tmp_path / "dataset"
    first = run_source_dataset_collection(
        output_dir=output,
        curriculum_path=tmp_path / "curriculum.json",
        game_ids=["bp35"],
        source_train_quota=20,
        source_validation_quota=10,
        action_budget_per_reset=25,
        max_raw_multiplier=4,
        checkpoint_every_resets=1,
        env_factory=factory,
        require_full_curriculum=False,
    )

    manifest = first["manifest"]
    assert manifest["total_transitions"] == 20
    assert manifest["split_counts"] == {"source_train": 20}
    assert manifest["policy_counts"] == {
        "active_controller": 14,
        "frontier_stall_probe": 2,
        "uniform_legal": 4,
    }
    verify_source_dataset(output / "manifest.json")
    created_after_first = len(created)

    second = run_source_dataset_collection(
        output_dir=output,
        curriculum_path=tmp_path / "curriculum.json",
        game_ids=["bp35"],
        source_train_quota=20,
        source_validation_quota=10,
        action_budget_per_reset=25,
        max_raw_multiplier=4,
        checkpoint_every_resets=1,
        env_factory=factory,
        require_full_curriculum=False,
    )
    assert len(created) == created_after_first
    assert (
        second["report"]["game_reports"]["bp35"][
            "resumed_completed_game"
        ]
        is True
    )
    assert (
        second["report"]["game_reports"]["bp35"][
            "seed_window_resets"
        ]
        == 200
    )


def test_capacity_report_proves_optimistic_shortfall(tmp_path: Path):
    checkpoint_dir = tmp_path / "work_shards"
    checkpoint_dir.mkdir()
    games = (*SOURCE_TRAIN, *SOURCE_VALIDATION)
    for game in games:
        accepted = 27 if game == "lp85" else 2681 if game == "sp80" else 8000
        status = "SATURATED" if game in {"lp85", "sp80"} else "COMPLETE"
        payload = {
            "game_id": game,
            "status": status,
            "quota": 8000,
            "accepted_transitions": accepted,
            "raw_transitions": accepted + 4000,
            "seeds_attempted": [0, 1, 2, 3, 4],
            "shard_sha256": game * 8,
        }
        (checkpoint_dir / f"{game}.checkpoint.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    report = write_capacity_report(tmp_path)

    assert report["status"] == "BLOCKED_CAPACITY"
    assert report["optimistic_upper_bound"] == 98_708
    assert report["minimum_shortfall"] == 1_292
    assert len(report["report_checksum"]) == 64


def test_approved_overflow_pool_is_global_and_deterministic():
    allocation = _overflow_allocation(
        1_292,
        games=("cd82", "dc22", "g50t", "ka59", "tr87"),
        selected=(*SOURCE_TRAIN, *SOURCE_VALIDATION),
    )

    assert allocation == {
        "cd82": 259,
        "dc22": 259,
        "g50t": 258,
        "ka59": 258,
        "tr87": 258,
    }
    assert sum(allocation.values()) == 1_292

    with pytest.raises(ValueError, match="must be unique"):
        _overflow_allocation(
            1_292,
            games=("cd82", "cd82"),
            selected=(*SOURCE_TRAIN, *SOURCE_VALIDATION),
        )


def test_partial_checkpoint_resumes_only_a_checksummed_in_progress_shard(
    tmp_path: Path,
):
    shard = tmp_path / "bp35.jsonl"
    shard.write_text('{"row":1}\n', encoding="utf-8")
    checkpoint = tmp_path / "bp35.checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "format_version": "sage11-source-collection-v2",
                "status": "IN_PROGRESS",
                "quota": 20,
                "accepted_transitions": 1,
                "raw_transitions": 3,
                "resets": 1,
                "shard_sha256": hashlib.sha256(
                    shard.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    resumed = _partial_game_checkpoint(
        checkpoint,
        shard_path=shard,
        expected_transitions=20,
    )

    assert resumed is not None
    assert resumed["accepted_transitions"] == 1
    assert not _completed_game_checkpoint(
        checkpoint,
        shard_path=shard,
        expected_transitions=20,
        expected_seeds=(0, 1),
    )
    assert _completed_game_checkpoint(
        checkpoint,
        shard_path=shard,
        expected_transitions=20,
        expected_seeds=(0, 1),
        allow_in_progress=True,
    )
    shard.write_text('{"row":2}\n', encoding="utf-8")
    assert (
        _partial_game_checkpoint(
            checkpoint,
            shard_path=shard,
            expected_transitions=20,
        )
        is None
    )
