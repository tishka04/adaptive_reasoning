"""Compact no-ablation performance-track tests."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from theory.benchmark_score_runner import run_benchmark_score


@dataclass
class _Action:
    id: int
    data: dict | None = None


@dataclass
class _Frame:
    frame: np.ndarray
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1,)


class _Game:
    def _get_valid_actions(self):
        return [_Action(1)]


class _TwoLevelEnv:
    def __init__(self) -> None:
        self._game = _Game()
        self.levels = 0
        self.grid = np.zeros((5, 5), dtype=np.int32)
        self.grid[2, 2] = 2

    def step(self, action, data=None):
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.levels = 0
            self.grid.fill(0)
            self.grid[2, 2] = 2
            return _Frame(self.grid.copy())
        self.levels += 1
        self.grid[2, 2] += 1
        return _Frame(
            self.grid.copy(),
            state="WIN" if self.levels >= 2 else "NOT_FINISHED",
            levels_completed=self.levels,
        )


def test_score_runner_skips_ablation_arm_and_writes_compact_history(
    tmp_path,
):
    output = tmp_path / "score.json"
    history = tmp_path / "history.json"
    payload = run_benchmark_score(
        game_ids=["synthetic"],
        seeds=[0],
        action_budgets=[4],
        resets=2,
        env_factory=lambda _game_id: _TwoLevelEnv(),
        label="test",
        write_path=output,
        history_path=history,
    )

    assert payload["schema_version"] == "sage.benchmark_score.v1"
    assert payload["total_levels_completed"] == 4
    assert payload["total_wins"] == 2
    assert payload["maximum_level_reached"] == 2
    assert payload["normalized_score_proxy"] == 2.0
    row = payload["rows"][0]
    assert row["levels_completed"] == 4
    assert len(row["actions_to_each_level"]) == 4
    assert "trace" not in row
    assert "controller_summary" not in row
    assert output.exists()
    history_payload = json.loads(history.read_text(encoding="utf-8"))
    assert len(history_payload["runs"]) == 1
    assert history_payload["runs"][0]["label"] == "test"
