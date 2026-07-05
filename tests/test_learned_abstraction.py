"""Smoke tests for the learned-abstraction pivot pipeline.

These are env-free and fast: they validate the feature schema stability, the
top-k color-pair inference, label logic, dataset I/O shapes, and an end-to-end
model train/predict roundtrip on a tiny synthetic fixture.

Run with the bundled interpreter (it has scikit-learn):
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe -m pytest tests/test_learned_abstraction.py -q
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import abstraction_labels as labels
import build_abstraction_dataset as dataset_builder
import game_splits
from build_break_event_dataset import break_class
from abstraction_dataset_io import (
    ACTION_EFFECT_TARGETS,
    HISTORY_FEATURE_SCHEMA,
    action_effect_xy,
    load_dataset,
    state_matrix,
)
from extract_state_abstractions import (
    FEATURE_DIM,
    FEATURE_SCHEMA,
    LARGEST_COMPONENT_FEATURE_SCHEMA,
    delta_features,
    extract_state_features,
    infer_active_color_pairs,
    largest_component_local_features,
)


def _random_grid(rng: random.Random, *, colors=(0, 10, 11, 3), h=12, w=12):
    return [[rng.choice(colors) for _ in range(w)] for _ in range(h)]


# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #
def test_feature_schema_is_stable_and_complete():
    feats = extract_state_features([[0, 0], [0, 0]])
    assert list(feats.keys()) == FEATURE_SCHEMA
    assert len(FEATURE_SCHEMA) == FEATURE_DIM
    # No duplicate column names.
    assert len(set(FEATURE_SCHEMA)) == len(FEATURE_SCHEMA)


def test_feature_extraction_is_deterministic():
    grid = [[0, 10, 10, 0], [0, 10, 0, 11], [0, 0, 11, 11]]
    a = extract_state_features(grid)
    b = extract_state_features(grid)
    assert a == b


def test_largest_component_local_features_are_stable():
    grid = [[0, 0, 0, 0], [0, 7, 7, 0], [0, 7, 0, 3], [0, 0, 0, 3]]
    feats = largest_component_local_features(grid, cursor=(1, 1))
    assert list(feats.keys()) == LARGEST_COMPONENT_FEATURE_SCHEMA
    assert feats["largest_component_width"] == pytest.approx(0.5)
    assert feats["largest_component_height"] == pytest.approx(0.5)
    assert feats["cursor_present"] == 1.0
    assert feats["cursor_distance_to_largest_bbox"] == pytest.approx(0.0)


def test_delta_features_keys_prefixed():
    g1 = [[0, 10], [0, 0]]
    g2 = [[10, 10], [0, 11]]
    delta = delta_features(extract_state_features(g1), extract_state_features(g2))
    assert all(k.startswith("delta_") for k in delta)
    assert len(delta) == FEATURE_DIM


# --------------------------------------------------------------------------- #
# Top-k pair inference recovers the ar25 {10, 11} families
# --------------------------------------------------------------------------- #
def test_top_pair_inference_recovers_ar25_families():
    grid = [[0] * 8 for _ in range(8)]
    # Two dominant families: color 10 (left blob) and color 11 (right blob).
    for y in range(1, 6):
        for x in range(1, 3):
            grid[y][x] = 10
    for y in range(1, 6):
        for x in range(5, 7):
            grid[y][x] = 11
    # A tiny distractor of another color.
    grid[7][7] = 3
    pairs = infer_active_color_pairs(grid, k=1)
    assert pairs, "expected at least one inferred pair"
    assert set(pairs[0]) == {10, 11}


# --------------------------------------------------------------------------- #
# Label logic
# --------------------------------------------------------------------------- #
def test_macro_label_avoid_on_game_over():
    assert labels.macro_label(delta={}, changed_cells=5, total_cells=100, game_over=True) == "AVOID"


def test_macro_label_explore_on_no_change():
    assert labels.macro_label(delta={}, changed_cells=0, total_cells=100, game_over=False) == "EXPLORE_ACTION"


def test_macro_label_break_dominates():
    delta = {"delta_largest_component_size": -20.0}
    assert labels.macro_label(delta=delta, changed_cells=10, total_cells=100, game_over=False) == "BREAK_LARGEST_COMPONENT"


def test_no_op_score_monotonic():
    assert labels.no_op_score(0, 1000) == 1.0
    assert labels.no_op_score(5, 1000) < labels.no_op_score(1, 1000)


def test_human_replay_continues_across_terminal_episode_boundaries(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.raw = SimpleNamespace(
                grid=[[0, 0], [0, 0]],
                levels_completed=0,
                state="NOT_FINISHED",
            )

        def transition(self, action):
            transitions = {
                "RESET": ([[2, 0], [0, 0]], 1, "NOT_FINISHED"),
                "ACTION1": ([[1, 0], [0, 0]], 1, "GAME_OVER"),
                "ACTION2": ([[2, 2], [0, 0]], 2, "NOT_FINISHED"),
                "ACTION3": ([[2, 2], [3, 0]], 2, "GAME_OVER"),
            }
            grid, level, state = transitions[action]
            self.raw = SimpleNamespace(
                grid=grid,
                levels_completed=level,
                state=state,
            )
            return self.raw

    steps = [
        {"episode_id": "ep-a", "action": "RESET"},
        {"episode_id": "ep-a", "action": "ACTION1"},
        {"episode_id": "ep-b", "action": "RESET"},
        {"episode_id": "ep-b", "action": "ACTION2"},
        {"episode_id": "ep-b", "action": "ACTION3"},
    ]
    env = FakeEnv()
    monkeypatch.setattr(dataset_builder, "_human_trace_steps", lambda _short_id: steps)
    monkeypatch.setattr(dataset_builder, "_make_env", lambda _arc, _game_id: env)
    monkeypatch.setattr(dataset_builder, "_obs", lambda _env: _env.raw)
    monkeypatch.setattr(dataset_builder, "_primary_grid", lambda raw: raw.grid)
    monkeypatch.setattr(dataset_builder, "_state_name", str)
    monkeypatch.setattr(
        dataset_builder,
        "_step",
        lambda _env, _game_id, action, _data: _env.transition(action),
    )
    monkeypatch.setattr(
        dataset_builder,
        "_step_with_data",
        lambda _env, action, _data: _env.transition(action),
    )

    rows = dataset_builder._run_human_replay(
        arc=object(),
        full_game_id="fake-game",
        short_id="fake",
        episode_index=10,
        max_steps=20,
        with_debug=False,
    )

    assert [row["action"] for row in rows] == ["ACTION1", "ACTION2", "ACTION3"]
    assert [row["episode_index"] for row in rows] == [10, 11, 11]
    assert rows[0]["game_over"] is True
    assert rows[1]["history_features"]["last_action"] is None
    assert rows[1]["level"] == 1


def test_break_event_class_thresholds():
    assert break_class(-50.0) == "BIG_BREAK"
    assert break_class(-49.9) == "OTHER_CHANGE"
    assert break_class(0.0) == "SMALL_CHANGE"
    assert break_class(4.9) == "SMALL_CHANGE"
    assert break_class(5.0) == "OTHER_CHANGE"


# --------------------------------------------------------------------------- #
# Game splits
# --------------------------------------------------------------------------- #
def test_ar25_isolated_from_train_and_unseen():
    assert "ar25" in game_splits.AR25_EVAL
    assert "ar25" not in game_splits.PUBLIC_SEEN
    assert "ar25" not in game_splits.PUBLIC_UNSEEN
    assert set(game_splits.PUBLIC_UNSEEN).isdisjoint(game_splits.PUBLIC_SEEN)


def test_split_for_game():
    assert game_splits.split_for_game("ar25-e3c63847") == "ar25"
    assert game_splits.split_for_game("wa30") == "unseen"


# --------------------------------------------------------------------------- #
# Dataset I/O + model train/predict roundtrip
# --------------------------------------------------------------------------- #
def _synthetic_rows(n=80, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        g1 = _random_grid(rng)
        g2 = _random_grid(rng)
        f1 = extract_state_features(g1)
        f2 = extract_state_features(g2)
        delta = delta_features(f1, f2)
        action = f"ACTION{rng.randint(1, 7)}"
        level_up = rng.random() < 0.1
        rows.append(
            {
                "game_id": "tn36-test",
                "level": 0,
                "episode_source": "random",
                "episode_index": i,
                "step": 0,
                "action": action,
                "action_data": None,
                "state_features": f1,
                "next_state_features": f2,
                "delta_features": delta,
                "changed_cells": 5,
                "level_up": level_up,
                "game_over": False,
                "future_level_up": level_up,
                "steps_to_level_up": 0 if level_up else -1,
                "future_game_over_soon": False,
                "progress_score": 1.0 if level_up else 0.25,
                "macro_label": labels.macro_label(
                    delta=delta, changed_cells=5, total_cells=144, game_over=False
                ),
            }
        )
    return rows


def test_dataset_io_shapes():
    rows = _synthetic_rows(40)
    x = state_matrix(rows)
    assert x.shape == (40, FEATURE_DIM)
    xe, ye, names = action_effect_xy(rows)
    assert xe.shape == (40, FEATURE_DIM + len(HISTORY_FEATURE_SCHEMA) + 7)
    assert ye.shape == (40, len(ACTION_EFFECT_TARGETS))
    assert names == ACTION_EFFECT_TARGETS


def test_model_a_train_predict_roundtrip(tmp_path: Path):
    train_action_effect_model = pytest.importorskip("train_action_effect_model")
    dataset = tmp_path / "tiny.jsonl"
    with dataset.open("w", encoding="utf-8") as handle:
        for row in _synthetic_rows(80):
            handle.write(json.dumps(row) + "\n")
    model_out = tmp_path / "action_effect.joblib"
    metrics = train_action_effect_model.train(
        dataset_path=str(dataset),
        model_out=str(model_out),
        train_games="all",
        sources=None,
        test_size=0.25,
        n_estimators=20,
        seed=0,
        quiet=True,
    )
    assert model_out.exists()
    assert "mean_r2" in metrics

    from learned_scoring import LearnedScorer  # value model needed too
    train_value_model = pytest.importorskip("train_value_model")
    value_out = tmp_path / "value.joblib"
    macro_out = tmp_path / "macro.joblib"
    data = load_dataset(str(dataset))
    value = train_value_model.train_value_model(data.rows, test_size=0.25, seed=0)
    macro = train_value_model.train_macro_model(data.rows, test_size=0.25, seed=0)
    from joblib import dump

    dump(value["bundle"], value_out)
    dump(macro["bundle"], macro_out)

    scorer = LearnedScorer(str(model_out), str(value_out))
    feats = extract_state_features(_random_grid(random.Random(1)))
    scores = scorer.score_actions(feats, [f"ACTION{i}" for i in range(1, 8)])
    assert len(scores) == 7
    assert all(np.isfinite(s.learned_score) for s in scores)
