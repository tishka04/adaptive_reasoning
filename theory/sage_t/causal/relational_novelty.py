"""Relational and archive-context representation for SAGE.T12.4a."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from theory.sage_t.contracts import AbstractState

from .contracts import GroundedAction
from .novelty import FEATURE_DIM as LEGACY_FEATURE_DIM
from .novelty import NoveltyPrediction, encode_state_action

RELATIONAL_NOVELTY_FORMAT = "sage-t12.4a-relational-novelty-v1"
RELATIONAL_DIM = 24
ARCHIVE_CONTEXT_DIM = 14
RELATIONAL_FEATURE_DIM = LEGACY_FEATURE_DIM + RELATIONAL_DIM + ARCHIVE_CONTEXT_DIM


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class ArchiveContext:
    """Only information available before executing the candidate action."""

    cell_visits: int = 0
    action_attempts: int = 0
    cell_expansions: int = 0
    unique_tried_actions: int = 0
    legal_actions: int = 0
    archive_cells: int = 0
    global_edges: int = 0
    global_action_trials: int = 0
    global_action_changed: int = 0
    global_action_novel: int = 0
    cell_action_trials: int = 0
    cell_action_changed: int = 0
    cell_action_novel: int = 0

    def __post_init__(self) -> None:
        for name in asdict(self):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"archive context cannot be negative: {name}")

    @property
    def untried(self) -> bool:
        return self.action_attempts == 0

    def encode(self) -> tuple[float, ...]:
        global_trials = self.global_action_trials
        cell_trials = self.cell_action_trials
        values = (
            min(1.0, self.cell_visits / 32.0),
            min(1.0, self.action_attempts / 16.0),
            min(1.0, self.cell_expansions / 64.0),
            float(self.untried),
            self.unique_tried_actions / max(1, self.legal_actions),
            min(1.0, self.legal_actions / 64.0),
            min(1.0, self.archive_cells / 512.0),
            min(1.0, self.global_edges / 4_096.0),
            min(1.0, global_trials / 128.0),
            (self.global_action_changed + 1) / (global_trials + 2),
            (self.global_action_novel + 1) / (global_trials + 2),
            min(1.0, cell_trials / 16.0),
            (self.cell_action_changed + 1) / (cell_trials + 2),
            (self.cell_action_novel + 1) / (cell_trials + 2),
        )
        if len(values) != ARCHIVE_CONTEXT_DIM:
            raise RuntimeError("T12.4a archive-context feature contract changed")
        return tuple(float(value) for value in values)

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchiveContext:
        return cls(
            **{
                name: int(payload.get(name, 0))
                for name in cls.__dataclass_fields__
            }
        )


def _numeric(action: GroundedAction, key: str) -> float | None:
    raw = action.action_data.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _action_point(
    state: AbstractState,
    action: GroundedAction,
) -> tuple[float, float] | None:
    x_value = _numeric(action, "x")
    y_value = _numeric(action, "y")
    if x_value is not None and y_value is not None:
        return y_value, x_value
    player = next(
        (
            entity
            for entity in state.entities
            if entity.has_role("player") and entity.center is not None
        ),
        None,
    )
    if player is None or player.center is None:
        return None
    dx_value = _numeric(action, "dx")
    dy_value = _numeric(action, "dy")
    if dx_value is None and dy_value is None:
        return None
    return (
        player.center[0] + (0.0 if dy_value is None else dy_value),
        player.center[1] + (0.0 if dx_value is None else dx_value),
    )


def encode_action_entity_relations(
    state: AbstractState,
    action: GroundedAction,
) -> tuple[float, ...]:
    point = _action_point(state, action)
    located = tuple(entity for entity in state.entities if entity.center is not None)
    if point is None:
        return (0.0,) * RELATIONAL_DIM
    row, column = point
    scale = math.hypot(64.0, 64.0)

    def distance(entity: Any) -> float:
        assert entity.center is not None
        return math.hypot(row - entity.center[0], column - entity.center[1])

    def role_distance(role: str) -> float:
        members = [distance(entity) for entity in located if entity.has_role(role)]
        return 1.0 if not members else min(1.0, min(members) / scale)

    nearest = min(located, key=distance) if located else None
    nearest_distance = scale if nearest is None else distance(nearest)
    player = next((entity for entity in located if entity.has_role("player")), None)
    player_row = row if player is None else player.center[0]  # type: ignore[index]
    player_column = column if player is None else player.center[1]  # type: ignore[index]
    values = (
        1.0,
        _clamp(row / 64.0, 0.0, 1.0),
        _clamp(column / 64.0, 0.0, 1.0),
        _clamp((row - player_row) / 64.0),
        _clamp((column - player_column) / 64.0),
        min(1.0, math.hypot(row - player_row, column - player_column) / scale),
        min(1.0, nearest_distance / scale),
        role_distance("player"),
        role_distance("target"),
        role_distance("hazardous"),
        role_distance("collectible"),
        float(nearest_distance <= 1.5),
        float(nearest_distance <= 4.0),
        float(nearest_distance <= 8.0),
        float(nearest is not None and nearest.has_role("player")),
        float(nearest is not None and nearest.has_role("target")),
        float(nearest is not None and nearest.has_role("hazardous")),
        float(nearest is not None and nearest.has_role("collectible")),
        float(abs(row - player_row) <= 0.5),
        float(abs(column - player_column) <= 0.5),
        float(
            any(entity.has_role("target") and distance(entity) <= 1.5 for entity in located)
        ),
        float(
            any(
                entity.has_role("hazardous") and distance(entity) <= 1.5
                for entity in located
            )
        ),
        float(
            any(
                entity.has_role("collectible") and distance(entity) <= 1.5
                for entity in located
            )
        ),
        float(nearest_distance <= 1.5),
    )
    if len(values) != RELATIONAL_DIM:
        raise RuntimeError("T12.4a relational feature contract changed")
    return tuple(float(value) for value in values)


def encode_relational_state_action(
    state: AbstractState,
    action: GroundedAction,
    context: ArchiveContext,
    *,
    include_relations: bool = True,
    include_context: bool = True,
) -> tuple[float, ...]:
    relations = (
        encode_action_entity_relations(state, action)
        if include_relations
        else (0.0,) * RELATIONAL_DIM
    )
    archive_context = (
        context.encode() if include_context else (0.0,) * ARCHIVE_CONTEXT_DIM
    )
    encoded = (*encode_state_action(state, action), *relations, *archive_context)
    if len(encoded) != RELATIONAL_FEATURE_DIM:
        raise RuntimeError("T12.4a feature contract changed")
    return tuple(float(value) for value in encoded)


class RelationalChangeNoveltyMLP(nn.Module):
    """Two small independent heads avoid negative transfer between targets."""

    def __init__(self, *, hidden_dim: int = 32) -> None:
        super().__init__()
        hidden = max(8, min(64, int(hidden_dim)))

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(RELATIONAL_FEATURE_DIM, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        self.change_head = head()
        self.novelty_head = head()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != RELATIONAL_FEATURE_DIM:
            raise ValueError("T12.4a features have the wrong dimension")
        return torch.cat(
            (self.change_head(features), self.novelty_head(features)),
            dim=-1,
        )


class RelationalNoveltyPredictor:
    def __init__(self, *, seed: int = 0, hidden_dim: int = 32) -> None:
        torch.manual_seed(int(seed))
        self.seed = int(seed)
        self.hidden_dim = max(8, min(64, int(hidden_dim)))
        self.model = RelationalChangeNoveltyMLP(hidden_dim=self.hidden_dim).cpu()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def predict(
        self,
        state: AbstractState,
        action: GroundedAction,
        context: ArchiveContext,
        *,
        include_relations: bool = True,
        include_context: bool = True,
    ) -> NoveltyPrediction:
        features = torch.tensor(
            [
                encode_relational_state_action(
                    state,
                    action,
                    context,
                    include_relations=include_relations,
                    include_context=include_context,
                )
            ],
            dtype=torch.float32,
        )
        self.model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(features))[0].tolist()
        return NoveltyPrediction(float(probabilities[0]), float(probabilities[1]))

    def save(self, path: str | Path, *, metadata: Mapping[str, Any]) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": RELATIONAL_NOVELTY_FORMAT,
                "seed": self.seed,
                "hidden_dim": self.hidden_dim,
                "state_dict": self.model.state_dict(),
                "metadata": dict(metadata),
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> RelationalNoveltyPredictor:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format_version") != RELATIONAL_NOVELTY_FORMAT:
            raise ValueError("unsupported T12.4a relational checkpoint")
        predictor = cls(
            seed=int(payload.get("seed", 0)),
            hidden_dim=int(payload.get("hidden_dim", 32)),
        )
        predictor.model.load_state_dict(payload["state_dict"])
        return predictor


__all__ = [
    "ARCHIVE_CONTEXT_DIM",
    "RELATIONAL_DIM",
    "RELATIONAL_FEATURE_DIM",
    "ArchiveContext",
    "RelationalChangeNoveltyMLP",
    "RelationalNoveltyPredictor",
    "encode_action_entity_relations",
    "encode_relational_state_action",
]
