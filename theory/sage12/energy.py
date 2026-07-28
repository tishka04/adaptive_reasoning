"""Explicit and optionally learned energy functions for semantic trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence, Tuple

from .world_model import SemanticTrajectory


@dataclass(frozen=True)
class EnergyWeights:
    goal_distance: float = 3.0
    risk: float = 8.0
    uncertainty: float = 1.5
    unlikely: float = 1.0
    cost: float = 0.25
    contradiction: float = 2.0


@dataclass(frozen=True)
class EnergyBreakdown:
    goal_distance: float
    risk: float
    uncertainty: float
    unlikely: float
    cost: float
    contradiction: float
    total: float

    def features(self) -> Tuple[float, ...]:
        return (
            self.goal_distance,
            self.risk,
            self.uncertainty,
            self.unlikely,
            self.cost,
            self.contradiction,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class HeuristicTrajectoryEnergy:
    """Auditable lower-is-better baseline used before any learned EBM."""

    def __init__(self, weights: EnergyWeights | None = None) -> None:
        self.weights = weights or EnergyWeights()

    def score(
        self,
        trajectory: SemanticTrajectory,
        *,
        goal_predicate: str,
    ) -> EnergyBreakdown:
        goal_distance = (
            0.0
            if _goal_present(goal_predicate, trajectory.final_state)
            else 1.0
        )
        risk = 1.0 if any(
            _predicate_name(predicate) == "game_over"
            for predicate in trajectory.final_state
        ) else 0.0
        unlikely = -math.log(max(1e-6, trajectory.probability))
        contradiction = sum(
            1.0
            for step in trajectory.steps
            if set(step.option.asserted_effects)
            & set(step.option.retracted_effects)
        )
        cost = float(trajectory.length)
        uncertainty = max(0.0, float(trajectory.uncertainty))
        total = (
            self.weights.goal_distance * goal_distance
            + self.weights.risk * risk
            + self.weights.uncertainty * uncertainty
            + self.weights.unlikely * unlikely
            + self.weights.cost * cost
            + self.weights.contradiction * contradiction
        )
        return EnergyBreakdown(
            goal_distance=goal_distance,
            risk=risk,
            uncertainty=uncertainty,
            unlikely=unlikely,
            cost=cost,
            contradiction=contradiction,
            total=total,
        )


class PairwiseTrajectoryEBM:
    """Tiny optional pairwise ranker; never granted authority by construction."""

    def __init__(
        self,
        *,
        input_width: int = 6,
        hidden_width: int = 16,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for the learned EBM") from exc
        torch.manual_seed(int(seed))
        self._torch = torch
        self.input_width = max(1, int(input_width))
        self.model = nn.Sequential(
            nn.Linear(self.input_width, max(2, int(hidden_width))),
            nn.Tanh(),
            nn.Linear(max(2, int(hidden_width)), 1),
        )
        self.trained_pairs = 0

    @property
    def device(self) -> str:
        return str(next(self.model.parameters()).device)

    def to(self, device: str) -> "PairwiseTrajectoryEBM":
        self.model.to(device)
        return self

    def energies(
        self,
        features: Sequence[Sequence[float]],
    ) -> Tuple[float, ...]:
        if not features:
            return ()
        if any(len(row) != self.input_width for row in features):
            raise ValueError("trajectory feature width does not match EBM input")
        torch = self._torch
        with torch.no_grad():
            tensor = torch.tensor(
                features,
                dtype=torch.float32,
                device=next(self.model.parameters()).device,
            )
            values = self.model(tensor).squeeze(-1)
        return tuple(float(value) for value in values.cpu().tolist())

    def fit_pairs(
        self,
        preferred: Sequence[Sequence[float]],
        rejected: Sequence[Sequence[float]],
        *,
        epochs: int = 50,
        learning_rate: float = 1e-3,
    ) -> Tuple[float, ...]:
        """Minimize softplus(E(preferred)-E(rejected))."""
        if len(preferred) != len(rejected) or not preferred:
            raise ValueError("preferred and rejected pairs must be non-empty")
        if any(
            len(row) != self.input_width
            for row in tuple(preferred) + tuple(rejected)
        ):
            raise ValueError("trajectory feature width does not match EBM input")
        torch = self._torch
        device = next(self.model.parameters()).device
        positive = torch.tensor(
            preferred, dtype=torch.float32, device=device
        )
        negative = torch.tensor(
            rejected, dtype=torch.float32, device=device
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate),
        )
        losses = []
        self.model.train()
        for _ in range(max(1, int(epochs))):
            optimizer.zero_grad()
            preferred_energy = self.model(positive).squeeze(-1)
            rejected_energy = self.model(negative).squeeze(-1)
            loss = torch.nn.functional.softplus(
                preferred_energy - rejected_energy
            ).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.model.eval()
        self.trained_pairs += len(preferred)
        return tuple(losses)


def _goal_present(goal_predicate: str, state: Iterable[str]) -> bool:
    target = str(goal_predicate)
    return any(
        predicate == target
        or _predicate_name(predicate) == target
        for predicate in state
    )


def _predicate_name(predicate: str) -> str:
    return str(predicate).split("|", 1)[0]


__all__ = [
    "EnergyBreakdown",
    "EnergyWeights",
    "HeuristicTrajectoryEnergy",
    "PairwiseTrajectoryEBM",
]
