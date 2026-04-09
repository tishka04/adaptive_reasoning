"""
Auxiliary prediction heads for the JEPA-style world model.

From the predicted latent next-state ẑ_{t+1}, these heads predict
actionable quantities that the router uses for decision-making:

  - probability of validity (will the solution be feasible?)
  - expected score improvement
  - expected compute cost
  - expected need for repair
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class AuxPredictions:
    """Predictions from auxiliary heads, used by the router."""
    validity_prob: torch.Tensor       # (batch,) probability solution will be valid
    score_improvement: torch.Tensor   # (batch,) expected delta in objective
    compute_cost: torch.Tensor        # (batch,) expected normalised compute cost 0-1
    repair_prob: torch.Tensor         # (batch,) probability repair will be needed


class AuxiliaryHeads(nn.Module):
    """
    Four lightweight heads on top of the predicted latent state.
    Each is a 2-layer MLP producing a scalar.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        def _make_head(out_activation: Optional[nn.Module] = None) -> nn.Sequential:
            layers = [
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            ]
            if out_activation is not None:
                layers.append(out_activation)
            return nn.Sequential(*layers)

        self.validity_head = _make_head(nn.Sigmoid())
        self.score_head = _make_head()  # unbounded regression
        self.cost_head = _make_head(nn.Sigmoid())
        self.repair_head = _make_head(nn.Sigmoid())

    def forward(self, z_hat: torch.Tensor) -> AuxPredictions:
        """
        Args:
            z_hat: (batch, latent_dim) — predicted next latent state

        Returns:
            AuxPredictions with all scalar predictions squeezed to (batch,)
        """
        return AuxPredictions(
            validity_prob=self.validity_head(z_hat).squeeze(-1),
            score_improvement=self.score_head(z_hat).squeeze(-1),
            compute_cost=self.cost_head(z_hat).squeeze(-1),
            repair_prob=self.repair_head(z_hat).squeeze(-1),
        )

    def loss(
        self,
        preds: AuxPredictions,
        targets: dict,
        weights: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compute combined auxiliary loss.

        targets should contain keys: 'valid', 'score_delta', 'cost', 'needed_repair'
        all as float tensors of shape (batch,).
        """
        w = weights or {}
        bce = nn.functional.binary_cross_entropy
        mse = nn.functional.mse_loss

        loss_val = bce(preds.validity_prob, targets["valid"])
        loss_score = mse(preds.score_improvement, targets["score_delta"])
        loss_cost = mse(preds.compute_cost, targets["cost"])
        loss_repair = bce(preds.repair_prob, targets["needed_repair"])

        total = (
            w.get("validity", 1.0) * loss_val
            + w.get("score", 1.0) * loss_score
            + w.get("cost", 0.5) * loss_cost
            + w.get("repair", 1.0) * loss_repair
        )
        return total
