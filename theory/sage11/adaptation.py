"""Bounded online adaptation for frozen SAGE.11 encoders."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OnlineAdaptationConfig:
    replay_capacity: int = 2048
    update_interval: int = 32
    maximum_gradient_steps: int = 4
    context_size: int = 32
    learning_rate: float = 1e-3


class OnlineDynamicsAdapter(nn.Module):
    """Small game context/dynamics head; the base encoder stays frozen."""

    def __init__(
        self,
        latent_size: int,
        action_size: int,
        *,
        config: OnlineAdaptationConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or OnlineAdaptationConfig()
        self.context = nn.Parameter(torch.zeros(self.config.context_size))
        width = max(32, int(latent_size))
        self.dynamics = nn.Sequential(
            nn.Linear(
                latent_size + action_size + self.config.context_size,
                width,
            ),
            nn.SiLU(),
            nn.Linear(width, latent_size),
        )
        self._replay: Deque[Tuple[Tensor, Tensor, Tensor]] = deque(
            maxlen=self.config.replay_capacity
        )
        self._optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
        )
        self._observations = 0
        self._updates = 0
        self._gradient_steps = 0

    def reset_game_seed(self) -> None:
        """Reset all target-local state at each game/seed boundary."""
        self._replay.clear()
        self._optimizer.state.clear()
        with torch.no_grad():
            self.context.zero_()
        self._observations = 0
        self._updates = 0
        self._gradient_steps = 0

    def observe(
        self,
        latent: Tensor,
        action: Tensor,
        next_latent: Tensor,
    ) -> Dict[str, int | float]:
        self._replay.append((
            latent.detach().cpu(),
            action.detach().cpu(),
            next_latent.detach().cpu(),
        ))
        self._observations += 1
        loss_value = 0.0
        steps = 0
        if (
            self._observations % self.config.update_interval == 0
            and self._replay
        ):
            steps = min(
                self.config.maximum_gradient_steps,
                len(self._replay),
            )
            for batch in list(self._replay)[-steps:]:
                loss_value += self._update_one(*batch)
            self._updates += 1
            self._gradient_steps += steps
        return {
            "observations": self._observations,
            "updates": self._updates,
            "gradient_steps": self._gradient_steps,
            "last_update_steps": steps,
            "loss": loss_value / max(1, steps),
        }

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        context = self.context.expand(latent.shape[0], -1)
        return F.normalize(
            self.dynamics(torch.cat((latent, action, context), dim=-1)),
            dim=-1,
        )

    def _update_one(
        self,
        latent: Tensor,
        action: Tensor,
        target: Tensor,
    ) -> float:
        device = self.context.device
        latent = latent.to(device)
        action = action.to(device)
        target = target.to(device)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
            action = action.unsqueeze(0)
            target = target.unsqueeze(0)
        self._optimizer.zero_grad(set_to_none=True)
        prediction = self(latent, action)
        loss = 1.0 - F.cosine_similarity(
            prediction,
            target,
            dim=-1,
        ).mean()
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().cpu().item())

    def summary(self) -> Dict[str, int]:
        return {
            "replay_capacity": self.config.replay_capacity,
            "replay_size": len(self._replay),
            "observations": self._observations,
            "updates": self._updates,
            "gradient_steps": self._gradient_steps,
            "update_interval": self.config.update_interval,
            "maximum_gradient_steps": (
                self.config.maximum_gradient_steps
            ),
        }


__all__ = ["OnlineAdaptationConfig", "OnlineDynamicsAdapter"]
