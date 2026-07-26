from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage11.dataset import NeuroTransition
from theory.sage11.relational_dataset import (
    RelationalJsonlCapture,
    RelationalPilotTransition,
    iter_relational_shard,
    relational_shard_metadata,
)
from theory.sage11.relational_features import (
    RELATIONAL_FEATURE_SCHEMA,
    encode_relational_features,
)
from theory.sage11.relational_effect_pilot import (
    evaluate_relational_pilot_gate,
)
from theory.sage11.relational_pilot_collection import (
    _collect_relational_game_job,
)


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
        del data
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


def _observation() -> SimpleNamespace:
    player_object = SimpleNamespace(
        bbox=(2, 2, 2, 2),
        center=(2.0, 2.0),
        area=1,
    )
    first = SimpleNamespace(
        bbox=(2, 3, 2, 3),
        center=(2.0, 3.0),
        area=1,
    )
    second = SimpleNamespace(
        bbox=(4, 3, 5, 3),
        center=(4.5, 3.0),
        area=2,
    )
    player = SimpleNamespace(position=(2, 2))
    return SimpleNamespace(
        objects=[player_object, first, second],
        best_player=player,
    )


def _base_transition() -> NeuroTransition:
    return NeuroTransition(
        game_id="bp35",
        seed=0,
        reset_index=0,
        step_index=0,
        policy_arm="uniform_legal",
        action_name="ACTION6",
        action_data={"x": 3, "y": 4},
        atoms_before=("state:game_state(not_finished)",),
        atoms_after=("state:game_state(not_finished)",),
        effect_atoms=(
            "effect:changed_cells(one)",
            "effect:player_moved(False)",
            "progress:level_complete(False)",
            "risk:game_over(False)",
        ),
        changed=True,
        noop=False,
        unsafe=False,
    )


def test_relational_encoder_preserves_geometry_without_coordinates() -> None:
    schema = RELATIONAL_FEATURE_SCHEMA
    vector = encode_relational_features(
        _observation(),
        action_name="ACTION6",
        action_data={"x": 3, "y": 4},
    )
    feature = {
        name: vector[index]
        for index, name in enumerate(schema.feature_names)
    }
    assert schema.feature_count == 52
    assert len(schema.state_feature_indices) == 22
    assert len(schema.action_dependent_feature_indices) == 30
    assert feature["state_object_count:few"] == 1
    assert feature["state_player:present"] == 1
    assert feature["state_object_pair:any_column_aligned"] == 1
    assert feature["state_player_object:any_contact"] == 1
    assert feature["action_target:has_xy"] == 1
    assert feature["action_target:inside_object"] == 1
    assert feature["action_target:contacts_object"] == 1
    assert feature["action_target:object_distance_zero"] == 1
    assert feature["action_target:player_distance_two_to_four"] == 1
    assert not any(
        name.endswith(":x") or name.endswith(":y")
        for name in schema.feature_names
    )


def test_relational_encoder_marks_coordinate_free_action_unavailable() -> None:
    vector = encode_relational_features(
        _observation(),
        action_name="ACTION1",
        action_data={},
    )
    mapping = RELATIONAL_FEATURE_SCHEMA.feature_to_index
    assert vector[mapping["action_target:has_xy"]] == 0
    assert vector[
        mapping["action_target:object_distance_unavailable"]
    ] == 1
    assert vector[
        mapping["action_target:player_distance_unavailable"]
    ] == 1
    assert np.count_nonzero(
        vector[
            list(RELATIONAL_FEATURE_SCHEMA.action_dependent_feature_indices)
        ]
    ) == 2


def test_relational_transition_round_trip_binds_schema() -> None:
    vector = encode_relational_features(
        _observation(),
        action_name="ACTION6",
        action_data={"x": 3, "y": 4},
    )
    record = RelationalPilotTransition(
        base_transition=_base_transition(),
        relational_features_before=tuple(float(value) for value in vector),
    )
    restored = RelationalPilotTransition.from_dict(record.to_dict())
    assert restored == record
    assert (
        restored.relational_schema_checksum
        == RELATIONAL_FEATURE_SCHEMA.checksum
    )
    with pytest.raises(ValueError, match="feature width"):
        RelationalPilotTransition(
            base_transition=_base_transition(),
            relational_features_before=(0.0,),
        )


def test_relational_pilot_gate_requires_changed_action_and_relation_value() -> None:
    heads = {
        "changed_cells": {
            "full_minus_best_baseline": 0.11,
            "full_minus_without_relations": 0.06,
        },
        "player_moved": {},
    }
    composite = {
        "conditional_action_shuffle_degradation": 0.12,
    }
    folds = {
        f"game{index}": (0.01 if index < 9 else -0.04)
        for index in range(11)
    }
    passed = evaluate_relational_pilot_gate(
        heads,
        composite,
        folds,
    )
    assert passed["passed"] is True

    failed_heads = {
        **heads,
        "changed_cells": {
            **heads["changed_cells"],
            "full_minus_without_relations": 0.049,
        },
    }
    failed = evaluate_relational_pilot_gate(
        failed_heads,
        composite,
        folds,
    )
    assert failed["passed"] is False
    assert (
        failed["relational_changed_contribution_at_least_0_05"]
        is False
    )


def test_relational_sidecar_round_trip_is_resumable(tmp_path) -> None:
    vector = encode_relational_features(
        _observation(),
        action_name="ACTION6",
        action_data={"x": 3, "y": 4},
    )
    record = RelationalPilotTransition(
        base_transition=_base_transition(),
        relational_features_before=tuple(float(value) for value in vector),
    )
    path = tmp_path / "bp35.jsonl"
    capture = RelationalJsonlCapture(path, expected_existing_rows=0)
    capture.append(record)
    resumed = RelationalJsonlCapture(path, expected_existing_rows=1)
    assert resumed.records == (record,)
    assert tuple(iter_relational_shard(path)) == (record,)


def test_relational_game_collection_captures_live_geometry(tmp_path) -> None:
    sidecar = tmp_path / "shards" / "bp35.jsonl"
    result = _collect_relational_game_job({
        "game_id": "bp35",
        "quota": 10,
        "sidecar_path": sidecar,
        "base_path": tmp_path / "base" / "bp35.jsonl",
        "checkpoint_path": (
            tmp_path / "base" / "bp35.checkpoint.json"
        ),
        "environment_root": tmp_path,
        "seeds": (0,),
        "action_budget_per_reset": 20,
        "max_raw_multiplier": 4,
        "checkpoint_every_resets": 1,
        "duplicate_saturation_patience": 100,
        "env_factory": lambda _game: _UniqueStateEnv(),
    })
    assert result["status"] == "COMPLETE"
    metadata = relational_shard_metadata(
        sidecar,
        expected_game="bp35",
        expected_rows=10,
    )
    assert metadata["transitions"] == 10
    rows = tuple(iter_relational_shard(sidecar))
    assert all(
        len(row.relational_features_before)
        == RELATIONAL_FEATURE_SCHEMA.feature_count
        for row in rows
    )
