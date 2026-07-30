from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from theory.sage12.goal_conditioned_trajectory_value_v4_18 import (
    GOALS,
    HORIZONS,
    _selected_index,
    _sequence_stream,
    _trajectory_targets,
    graph_feature_vector,
)


def _graph(*, relation: str = "adjacent", x: int = 3) -> dict:
    return {
        "root": {
            "action_name": "ACTION6",
            "action_family": "click",
            "game_id": "source_game",
            "x": x,
            "actor_relation": relation,
            "root_kind": "occupied_object",
        },
        "neighbors": [
            {
                "shape": "compact",
                "proximity": "near",
                "relative_size": "smaller",
                "x": x + 1,
            },
            {
                "shape": "wide",
                "proximity": "far",
                "relative_size": "larger",
                "x": x + 2,
            },
        ],
    }


def _row(
    *,
    effects: dict[str, bool] | None = None,
    productive: bool = False,
) -> dict:
    return {
        "teacher": {
            "observed_effects": effects or {},
            "productive": productive,
        }
    }


def test_graph_features_exclude_identity_coordinates_and_neighbor_order() -> None:
    first = _graph(x=3)
    second = _graph(x=55)
    second["root"]["game_id"] = "different_game"
    second["root"]["action_name"] = "ACTION1"
    second["neighbors"] = list(reversed(second["neighbors"]))
    assert np.array_equal(
        graph_feature_vector(first),
        graph_feature_vector(second),
    )


def test_relation_removed_view_erases_relation_changes() -> None:
    adjacent = _graph(relation="adjacent")
    distant = _graph(relation="distant")
    distant["neighbors"][0]["proximity"] = "far"
    assert not np.array_equal(
        graph_feature_vector(adjacent),
        graph_feature_vector(distant),
    )
    assert np.array_equal(
        graph_feature_vector(adjacent, remove_relations=True),
        graph_feature_vector(distant, remove_relations=True),
    )


def test_multi_horizon_credit_reaches_preparatory_actions() -> None:
    sequence = [
        _row(productive=True),
        _row(effects={"path_opened": True}),
        _row(effects={"level_complete": True}),
        _row(effects={"game_over": True}),
    ]
    targets = _trajectory_targets(sequence, 0)
    assert targets.shape == (len(GOALS), len(HORIZONS))
    assert targets[GOALS.index("access"), HORIZONS.index(8)] == pytest.approx(0.97)
    assert targets[
        GOALS.index("terminal_progress"),
        HORIZONS.index(8),
    ] == pytest.approx(0.97**2)
    assert targets[GOALS.index("risk"), HORIZONS.index(8)] == pytest.approx(
        -(0.97**3)
    )


def test_sequence_stream_rejects_reappearing_episode(tmp_path: Path) -> None:
    path = tmp_path / "choices.jsonl"
    rows = [
        {"audit": {"source_file": "a", "episode_id": "one"}},
        {"audit": {"source_file": "a", "episode_id": "two"}},
        {"audit": {"source_file": "a", "episode_id": "one"}},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reappeared"):
        list(_sequence_stream(path))


def test_selected_index_is_deterministic_on_ties() -> None:
    assert _selected_index([1.0, 1.0, 0.0], [9, 3, 1]) == 1
