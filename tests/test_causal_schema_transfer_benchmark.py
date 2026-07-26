"""Protocol tests for frozen source-to-target causal-schema evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.causal_schema_transfer_benchmark import (
    run_causal_schema_transfer_benchmark,
    run_frozen_causal_schema_target_benchmark,
)
from theory.online_transferable_causal_schema import (
    FrozenCausalSchemaLibrary,
)


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


class _SourceTerminalEnv:
    def __init__(self) -> None:
        self._game = _Game()
        self.grid = np.zeros((5, 5), dtype=np.int32)
        self.grid[2, 2] = 2

    def step(self, action, data=None):
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.grid.fill(0)
            self.grid[2, 2] = 2
            return _Frame(self.grid.copy())
        self.grid[2, 2] = 3
        return _Frame(
            self.grid.copy(),
            state="WIN",
            levels_completed=1,
        )


class _TargetStallEnv:
    def __init__(self) -> None:
        self._game = _Game()
        self.grid = np.zeros((5, 5), dtype=np.int32)
        self.grid[2, 2] = 7

    def step(self, action, data=None):
        name = str(getattr(action, "name", ""))
        value = int(getattr(action, "value", action))
        if name == "RESET" or value == 0:
            self.grid.fill(0)
            self.grid[2, 2] = 7
            return _Frame(self.grid.copy())
        self.grid[2, 2] = 8 if self.grid[2, 2] == 7 else 7
        return _Frame(self.grid.copy())


def test_transfer_benchmark_freezes_source_and_activates_target_probe():
    def factory(game_id):
        if game_id == "source":
            return _SourceTerminalEnv()
        return _TargetStallEnv()

    payload = run_causal_schema_transfer_benchmark(
        source_game_id="source",
        target_game_ids=("target",),
        seed=0,
        source_action_budget_per_reset=2,
        source_resets=2,
        target_action_budget_per_reset=10,
        target_resets=4,
        env_factory=factory,
    )

    assert payload["schema_version"] == "sage.causal_schema_transfer.v1"
    assert payload["protocol"]["source_controller_reused_on_targets"] is False
    assert payload["frozen_library"]["frozen"] is True
    assert payload["aggregate"]["source_schemas"] >= 1
    assert payload["gates"]["G1_frozen_abstract_library"] is True
    assert payload["gates"]["G2_source_evidence_is_probe_only"] is True
    assert payload["gates"]["G3_target_mechanism_activated"] is True
    assert payload["targets"][0]["same_fresh_reset_states"] is True

    imported = FrozenCausalSchemaLibrary.from_dict(
        payload["frozen_library"]
    )
    target_only = run_frozen_causal_schema_target_benchmark(
        library=imported,
        target_game_ids=("target",),
        seed=0,
        action_budget_per_reset=10,
        resets=4,
        env_factory=factory,
    )
    assert (
        target_only["schema_version"]
        == "sage.causal_schema_target_only.v1"
    )
    assert target_only["gates"]["G1_frozen_library_imported"] is True
    assert target_only["gates"]["G3_target_mechanism_activated"] is True
