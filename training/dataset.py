"""
ARC-AGI-3 Transition Dataset for training JEPA world model + EBM scorer.

Loads the JSON transitions collected by collect_transitions.py,
splits into train/test per-game (80/20), and provides PyTorch batches.

Each sample contains:
  - grid_before   (H, W)   int grid
  - grid_after    (H, W)   int grid
  - action_idx    int       (0=RESET, 1-7=ACTION1-7)
  - action_bag    (8,)      bag-of-actions encoding
  - context_before (20,)    context features
  - context_after  (20,)    context features
  - level_changed  bool
  - game_over      bool
  - anything_changed bool
  - states_discovered float (proxy: num_changes / grid_size)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

ACTION_NAMES = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
                "ACTION5", "ACTION6", "ACTION7"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}

MAX_GRID = 64
NUM_CHANNELS = 16


def grid_to_onehot(grid: np.ndarray, max_size: int = MAX_GRID) -> torch.Tensor:
    """Convert int grid to one-hot tensor (C, H, W), padded to max_size."""
    h, w = grid.shape
    padded = np.zeros((max_size, max_size), dtype=np.int64)
    ph, pw = min(h, max_size), min(w, max_size)
    padded[:ph, :pw] = grid[:ph, :pw]
    t = torch.tensor(padded, dtype=torch.long).clamp(0, NUM_CHANNELS - 1)
    return torch.nn.functional.one_hot(t, NUM_CHANNELS).permute(2, 0, 1).float()


def build_context(t: dict) -> torch.Tensor:
    """Build a 20-dim context vector from a transition dict."""
    vec = torch.zeros(20)

    # Player position (normalized)
    if t.get("player_pos_before"):
        shape = t.get("grid_shape", [64, 64])
        h, w = shape[0], shape[1]
        vec[0] = t["player_pos_before"][0] / max(h, 1)
        vec[1] = t["player_pos_before"][1] / max(w, 1)
        vec[2] = 1.0  # player identified

    # Level
    vec[3] = t.get("level_before", 0) / 10.0
    vec[4] = t.get("step", 0) / 200.0

    # Memory stats
    ms = t.get("memory_summary", {})
    vec[5] = ms.get("total_actions", 0) / 200.0
    vec[6] = ms.get("states_visited", 0) / 200.0
    vec[7] = ms.get("max_level", 0) / 10.0
    vec[8] = ms.get("total_game_overs", 0) / 10.0
    vec[9] = ms.get("total_resets", 0) / 10.0

    # Grid stats
    shape = t.get("grid_shape", [64, 64])
    vec[10] = shape[0] / 64.0
    vec[11] = shape[1] / 64.0
    vec[12] = t.get("n_objects_before", 0) / 20.0

    # Action semantics count
    move_acts = ms.get("movement_actions", [])
    vec[13] = len(move_acts) / 7.0

    # Game state
    state_map = {"NOT_PLAYED": 0, "NOT_FINISHED": 0.5, "WIN": 1.0, "GAME_OVER": -1.0}
    vec[14] = state_map.get(t.get("state_before", "NOT_FINISHED"), 0)

    return vec


def build_context_after(t: dict) -> torch.Tensor:
    """Build a 20-dim context vector for the after-state."""
    vec = torch.zeros(20)

    if t.get("player_pos_after"):
        shape = t.get("grid_shape", [64, 64])
        h, w = shape[0], shape[1]
        vec[0] = t["player_pos_after"][0] / max(h, 1)
        vec[1] = t["player_pos_after"][1] / max(w, 1)
        vec[2] = 1.0

    vec[3] = t.get("level_after", 0) / 10.0
    vec[4] = (t.get("step", 0) + 1) / 200.0

    ms = t.get("memory_summary", {})
    vec[5] = ms.get("total_actions", 0) / 200.0
    vec[6] = ms.get("states_visited", 0) / 200.0
    vec[7] = ms.get("max_level", 0) / 10.0
    vec[8] = ms.get("total_game_overs", 0) / 10.0
    vec[9] = ms.get("total_resets", 0) / 10.0

    shape = t.get("grid_shape", [64, 64])
    vec[10] = shape[0] / 64.0
    vec[11] = shape[1] / 64.0
    vec[12] = t.get("n_objects_after", 0) / 20.0

    move_acts = ms.get("movement_actions", [])
    vec[13] = len(move_acts) / 7.0

    state_map = {"NOT_PLAYED": 0, "NOT_FINISHED": 0.5, "WIN": 1.0, "GAME_OVER": -1.0}
    vec[14] = state_map.get(t.get("state_after", "NOT_FINISHED"), 0)

    return vec


def action_to_bag(action_name: str) -> torch.Tensor:
    """Convert action name to bag-of-actions encoding (8,)."""
    bag = torch.zeros(8)
    idx = ACTION_TO_IDX.get(action_name, 0)
    bag[idx] = 1.0
    return bag


class TransitionDataset(Dataset):
    """PyTorch dataset of game transitions."""

    def __init__(self, transitions: List[dict]):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict:
        t = self.transitions[idx]

        grid_before = np.array(t["grid_before"], dtype=np.int32)
        grid_after = np.array(t["grid_after"], dtype=np.int32)

        return {
            "grid_before_oh": grid_to_onehot(grid_before),
            "grid_after_oh": grid_to_onehot(grid_after),
            "ctx_before": build_context(t),
            "ctx_after": build_context_after(t),
            "action_idx": ACTION_TO_IDX.get(t["action"], 0),
            "action_bag": action_to_bag(t["action"]),
            "level_changed": float(t.get("level_changed", False)),
            "game_over": float(t.get("game_over", False)),
            "anything_changed": float(t.get("anything_changed", False)),
            "num_changes": t.get("num_changes", 0),
            "grid_shape": t.get("grid_shape", [64, 64]),
        }


def load_transitions(data_dir: str | Path) -> Tuple[List[dict], List[dict]]:
    """
    Load all transition JSONs from data_dir and split into train/test.

    Split: 80% train / 20% test, stratified by game.
    Games with too few transitions are entirely in train.
    """
    data_dir = Path(data_dir)
    all_train = []
    all_test = []

    json_files = sorted(data_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "collection_summary.json"]

    if not json_files:
        raise FileNotFoundError(f"No transition files found in {data_dir}")

    for jf in json_files:
        with open(jf) as f:
            transitions = json.load(f)

        if len(transitions) < 10:
            # Too few — all goes to train
            all_train.extend(transitions)
            continue

        # Shuffle within this game
        random.shuffle(transitions)
        split_idx = int(len(transitions) * 0.8)
        all_train.extend(transitions[:split_idx])
        all_test.extend(transitions[split_idx:])

    logger.info(f"Loaded {len(all_train)} train + {len(all_test)} test transitions "
                f"from {len(json_files)} game files")

    return all_train, all_test


def make_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders."""
    random.seed(seed)
    train_data, test_data = load_transitions(data_dir)

    train_ds = TransitionDataset(train_data)
    test_ds = TransitionDataset(test_data)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    return train_loader, test_loader
