"""Shared loading utilities for the abstraction transition dataset.

Kept tiny and dependency-light (numpy only) so every training/eval script
reads the JSONL the same way and shares the action encoding + feature order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from extract_state_abstractions import (
    FEATURE_SCHEMA,
    FEATURE_DIM,
    LARGEST_COMPONENT_FEATURE_SCHEMA,
)

ACTIONS = [f"ACTION{i}" for i in range(1, 8)]
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
HISTORY_FEATURE_SCHEMA = (
    [f"last_action_is_{action}" for action in ACTIONS]
    + [f"prev_action_is_{action}" for action in ACTIONS]
    + ["action_repeat_count", "steps_since_state_change"]
)

# Action-effect (Model A) regression targets, drawn from delta_features.
ACTION_EFFECT_TARGETS = [
    "delta_largest_component_size",
    "delta_component_count",
    "delta_top_pair_0_global_correspondence",
    "delta_fragmentation_ratio",
]


@dataclass
class Dataset:
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def filter_games(self, game_ids: List[str]) -> "Dataset":
        wanted = set(game_ids) | {g.split("-", 1)[0] for g in game_ids}
        kept = [
            r
            for r in self.rows
            if r.get("game_id") in wanted or r.get("game_id", "").split("-", 1)[0] in wanted
        ]
        return Dataset(kept)

    def filter_sources(self, sources: Optional[List[str]]) -> "Dataset":
        if not sources:
            return self
        keep = set(sources)
        return Dataset([r for r in self.rows if r.get("episode_source") in keep])


def load_dataset(path: str | Path) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset(rows)


def feature_vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([float(feats.get(name, 0.0)) for name in FEATURE_SCHEMA], dtype=np.float32)


def state_matrix(rows: List[Dict[str, Any]], *, key: str = "state_features") -> np.ndarray:
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([feature_vector(r[key]) for r in rows])


def action_onehot(rows: List[Dict[str, Any]]) -> np.ndarray:
    mat = np.zeros((len(rows), len(ACTIONS)), dtype=np.float32)
    for i, row in enumerate(rows):
        idx = ACTION_INDEX.get(row.get("action", ""))
        if idx is not None:
            mat[i, idx] = 1.0
    return mat


def make_history_features(
    action_history: Sequence[str],
    action_repeat_count: int,
    steps_since_state_change: int,
) -> Dict[str, Any]:
    """Compact, JSON-friendly history context before the current action."""

    last_two = [str(action) for action in action_history[-2:]]
    return {
        "last_action": last_two[-1] if last_two else None,
        "last_two_actions": last_two,
        "action_repeat_count": int(action_repeat_count),
        "steps_since_state_change": int(steps_since_state_change),
    }


def _history_action_pair(history: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    last_action = history.get("last_action")
    last_two = history.get("last_two_actions")
    prev_action = None
    if isinstance(last_two, list):
        if len(last_two) >= 2:
            prev_action = str(last_two[-2])
        if len(last_two) >= 1 and last_action is None:
            last_action = str(last_two[-1])
    return (str(last_action) if last_action else None, prev_action)


def history_feature_vector(history: Mapping[str, Any] | None) -> np.ndarray:
    vec = np.zeros(len(HISTORY_FEATURE_SCHEMA), dtype=np.float32)
    if not history:
        return vec
    last_action, prev_action = _history_action_pair(history)
    if last_action in ACTION_INDEX:
        vec[ACTION_INDEX[last_action]] = 1.0
    if prev_action in ACTION_INDEX:
        vec[len(ACTIONS) + ACTION_INDEX[prev_action]] = 1.0
    vec[-2] = float(history.get("action_repeat_count", 0) or 0)
    vec[-1] = float(history.get("steps_since_state_change", 0) or 0)
    return vec


def history_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(HISTORY_FEATURE_SCHEMA)), dtype=np.float32)
    return np.stack([history_feature_vector(r.get("history_features")) for r in rows])


def largest_component_feature_vector(feats: Mapping[str, float] | None) -> np.ndarray:
    if not feats:
        return np.zeros(len(LARGEST_COMPONENT_FEATURE_SCHEMA), dtype=np.float32)
    return np.array(
        [float(feats.get(name, 0.0)) for name in LARGEST_COMPONENT_FEATURE_SCHEMA],
        dtype=np.float32,
    )


def largest_component_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(LARGEST_COMPONENT_FEATURE_SCHEMA)), dtype=np.float32)
    return np.stack(
        [largest_component_feature_vector(r.get("largest_component_features")) for r in rows]
    )


def largest_component_input_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Specialized inputs: largest-component local features + history + action."""

    return np.concatenate([largest_component_matrix(rows), history_matrix(rows), action_onehot(rows)], axis=1)


def largest_component_input_feature_names() -> List[str]:
    return (
        list(LARGEST_COMPONENT_FEATURE_SCHEMA)
        + list(HISTORY_FEATURE_SCHEMA)
        + [f"is_{action}" for action in ACTIONS]
    )


def transition_input_matrix(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Inputs for transition-conditioned models: state + history + action."""

    return np.concatenate([state_matrix(rows), history_matrix(rows), action_onehot(rows)], axis=1)


def action_effect_xy(rows: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Model A inputs (state ⊕ action one-hot) and multi-output delta targets."""

    x = transition_input_matrix(rows)
    y = np.stack(
        [
            np.array(
                [float(r["delta_features"].get(t, 0.0)) for t in ACTION_EFFECT_TARGETS],
                dtype=np.float32,
            )
            for r in rows
        ]
    ) if rows else np.zeros((0, len(ACTION_EFFECT_TARGETS)), dtype=np.float32)
    return x, y, list(ACTION_EFFECT_TARGETS)


def input_feature_names() -> List[str]:
    return list(FEATURE_SCHEMA) + list(HISTORY_FEATURE_SCHEMA) + [f"is_{a}" for a in ACTIONS]
