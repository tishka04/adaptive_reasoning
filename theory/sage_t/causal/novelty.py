"""Small online action-conditioned change/novelty predictor.

Only symbolic state features and grounded action metadata are consumed.  The
model has no terminal head and no visual encoder; safety remains the authority
of :mod:`terminal_shield`.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from theory.sage_t.contracts import AbstractState

from .contracts import GroundedAction

NOVELTY_FORMAT = "sage-t12.1-online-novelty-v1"
FACT_HASH_DIM = 32
ACTION_DIM = 8
SCALAR_DIM = 16
FEATURE_DIM = FACT_HASH_DIM + ACTION_DIM + SCALAR_DIM


def _stable_bucket(value: str, modulo: int) -> tuple[int, float]:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % modulo
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return bucket, sign


def _numeric_arg(action: GroundedAction, key: str) -> tuple[float, float]:
    raw = action.action_data.get(key)
    if raw is None:
        return 0.0, 0.0
    try:
        return max(-1.0, min(1.0, float(raw) / 64.0)), 1.0
    except (TypeError, ValueError):
        return 0.0, 1.0


def encode_state_action(state: AbstractState, action: GroundedAction) -> tuple[float, ...]:
    fact_features = [0.0] * FACT_HASH_DIM
    public_facts = tuple(sorted(fact.key for fact in state.true_facts))
    denominator = math.sqrt(max(1, len(public_facts)))
    for fact in public_facts:
        bucket, sign = _stable_bucket(fact, FACT_HASH_DIM)
        fact_features[bucket] += sign / denominator

    action_features = [0.0] * ACTION_DIM
    action_name = action.action_name.upper()
    if action_name.startswith("ACTION"):
        try:
            index = int(action_name[6:]) - 1
        except ValueError:
            index = -1
        if 0 <= index < ACTION_DIM:
            action_features[index] = 1.0

    role_counts = {
        role: sum(entity.has_role(role) for entity in state.entities)
        for role in ("player", "target", "hazardous", "collectible")
    }
    relation_count = sum(len(fact.terms) >= 2 for fact in state.true_facts)
    player = next(
        (entity for entity in state.entities if entity.has_role("player")),
        None,
    )
    player_row = 0.0 if player is None or player.center is None else player.center[0] / 64.0
    player_column = 0.0 if player is None or player.center is None else player.center[1] / 64.0
    x_value, x_present = _numeric_arg(action, "x")
    y_value, y_present = _numeric_arg(action, "y")
    dx_value, dx_present = _numeric_arg(action, "dx")
    dy_value, dy_present = _numeric_arg(action, "dy")
    scalar_features = [
        min(1.0, len(state.entities) / 64.0),
        min(1.0, role_counts["player"] / 4.0),
        min(1.0, role_counts["target"] / 32.0),
        min(1.0, role_counts["hazardous"] / 16.0),
        min(1.0, role_counts["collectible"] / 16.0),
        min(1.0, relation_count / 128.0),
        min(1.0, len(state.true_facts) / 256.0),
        min(1.0, state.counter("levels_completed") / 16.0),
        min(1.0, state.regime_index / 16.0),
        max(-1.0, min(1.0, player_row)),
        max(-1.0, min(1.0, player_column)),
        x_value,
        y_value,
        dx_value,
        dy_value,
        max(x_present, y_present, dx_present, dy_present),
    ]
    encoded = (*fact_features, *action_features, *scalar_features)
    if len(encoded) != FEATURE_DIM:
        raise RuntimeError("novelty feature contract changed unexpectedly")
    return tuple(float(value) for value in encoded)


class ChangeNoveltyMLP(nn.Module):
    def __init__(self, *, hidden_dim: int = 32) -> None:
        super().__init__()
        hidden = max(8, min(64, int(hidden_dim)))
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != FEATURE_DIM:
            raise ValueError("change/novelty features have the wrong dimension")
        return self.network(features)


@dataclass(frozen=True)
class NoveltyExample:
    features: tuple[float, ...]
    changed: bool
    novel: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "changed": self.changed,
            "novel": self.novel,
        }


@dataclass(frozen=True)
class NoveltyPrediction:
    change_probability: float
    novelty_probability: float


class OnlineNoveltyPredictor:
    """Prequential CPU learner: predict first, update only afterwards."""

    def __init__(
        self,
        *,
        seed: int = 0,
        hidden_dim: int = 32,
        maximum_examples: int = 4_096,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> None:
        torch.manual_seed(int(seed))
        self.seed = int(seed)
        self.hidden_dim = max(8, min(64, int(hidden_dim)))
        self.maximum_examples = max(32, min(4_096, int(maximum_examples)))
        self.batch_size = max(8, min(128, int(batch_size)))
        self.learning_rate = float(learning_rate)
        self.model = ChangeNoveltyMLP(hidden_dim=self.hidden_dim).cpu()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        self.examples: deque[NoveltyExample] = deque(maxlen=self.maximum_examples)
        self.prequential: list[tuple[NoveltyPrediction, bool, bool]] = []
        self.updates = 0

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def predict(self, state: AbstractState, action: GroundedAction) -> NoveltyPrediction:
        features = torch.tensor(
            [encode_state_action(state, action)], dtype=torch.float32
        )
        self.model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(features))[0].tolist()
        return NoveltyPrediction(float(probabilities[0]), float(probabilities[1]))

    def score(self, state: AbstractState, action: GroundedAction) -> tuple[float, float]:
        prediction = self.predict(state, action)
        return prediction.change_probability, prediction.novelty_probability

    def observe(
        self,
        state: AbstractState,
        action: GroundedAction,
        *,
        changed: bool,
        novel: bool,
        update: bool = True,
    ) -> NoveltyPrediction:
        prediction = self.predict(state, action)
        self.prequential.append((prediction, bool(changed), bool(novel)))
        self.examples.append(
            NoveltyExample(
                features=encode_state_action(state, action),
                changed=bool(changed),
                novel=bool(novel),
            )
        )
        if update and len(self.examples) >= self.batch_size:
            self._train_latest_batch()
        return prediction

    def _train_latest_batch(self) -> float:
        batch = tuple(self.examples)[-self.batch_size :]
        features = torch.tensor([item.features for item in batch], dtype=torch.float32)
        targets = torch.tensor(
            [[float(item.changed), float(item.novel)] for item in batch],
            dtype=torch.float32,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(features)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach().cpu().item())

    def metrics(self) -> dict[str, Any]:
        if not self.prequential:
            return {
                "examples": 0,
                "parameter_count": self.parameter_count,
                "updates": self.updates,
            }
        change_probabilities = [item[0].change_probability for item in self.prequential]
        novelty_probabilities = [item[0].novelty_probability for item in self.prequential]
        change_targets = [float(item[1]) for item in self.prequential]
        novelty_targets = [float(item[2]) for item in self.prequential]
        return {
            "examples": len(self.prequential),
            "buffer_examples": len(self.examples),
            "parameter_count": self.parameter_count,
            "updates": self.updates,
            "change_prevalence": sum(change_targets) / len(change_targets),
            "novelty_prevalence": sum(novelty_targets) / len(novelty_targets),
            "change_brier": brier_score(change_probabilities, change_targets),
            "novelty_brier": brier_score(novelty_probabilities, novelty_targets),
            "mean_brier": 0.5
            * (
                brier_score(change_probabilities, change_targets)
                + brier_score(novelty_probabilities, novelty_targets)
            ),
            "change_ece": expected_calibration_error(
                change_probabilities, change_targets
            ),
            "novelty_ece": expected_calibration_error(
                novelty_probabilities, novelty_targets
            ),
        }

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite novelty checkpoint: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": NOVELTY_FORMAT,
                "seed": self.seed,
                "hidden_dim": self.hidden_dim,
                "maximum_examples": self.maximum_examples,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "state_dict": self.model.state_dict(),
                "examples": [item.to_dict() for item in self.examples],
                "updates": self.updates,
                "metadata": dict(metadata or {}),
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> OnlineNoveltyPredictor:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format_version") != NOVELTY_FORMAT:
            raise ValueError("unsupported novelty checkpoint")
        predictor = cls(
            seed=int(payload.get("seed", 0)),
            hidden_dim=int(payload.get("hidden_dim", 32)),
            maximum_examples=int(payload.get("maximum_examples", 4_096)),
            batch_size=int(payload.get("batch_size", 32)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
        )
        predictor.model.load_state_dict(payload["state_dict"])
        predictor.examples.extend(
            NoveltyExample(
                features=tuple(float(value) for value in row["features"]),
                changed=bool(row["changed"]),
                novel=bool(row["novel"]),
            )
            for row in payload.get("examples", ())
        )
        predictor.updates = int(payload.get("updates", 0))
        return predictor


def brier_score(probabilities: Sequence[float], targets: Sequence[float]) -> float:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("Brier score needs equal non-empty vectors")
    return sum(
        (float(probability) - float(target)) ** 2
        for probability, target in zip(probabilities, targets)
    ) / len(probabilities)


def expected_calibration_error(
    probabilities: Sequence[float],
    targets: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("ECE needs equal non-empty vectors")
    total = len(probabilities)
    error = 0.0
    for index in range(max(1, int(bins))):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            pair
            for pair in zip(probabilities, targets)
            if lower <= float(pair[0]) < upper or index == bins - 1 and float(pair[0]) == 1.0
        ]
        if not members:
            continue
        confidence = sum(float(item[0]) for item in members) / len(members)
        accuracy = sum(float(item[1]) for item in members) / len(members)
        error += len(members) / total * abs(confidence - accuracy)
    return error


__all__ = [
    "FEATURE_DIM",
    "NOVELTY_FORMAT",
    "ChangeNoveltyMLP",
    "NoveltyExample",
    "NoveltyPrediction",
    "OnlineNoveltyPredictor",
    "brier_score",
    "encode_state_action",
    "expected_calibration_error",
]
