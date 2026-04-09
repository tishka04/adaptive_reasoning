"""
Energy-Based Model router — scores candidate reasoning actions.

For each candidate r_i, computes:
    E(z_t, r_i, ẑ_{t+1}^{(i)})

Low energy = promising reasoning action.

The router can:
  - select the single lowest-energy candidate
  - return top-k candidates for parallel verification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..world_model.aux_heads import AuxPredictions


@dataclass
class RoutingDecision:
    """Result of the EBM routing step."""
    selected_idx: int
    selected_energy: float
    all_energies: List[float]
    top_k_indices: List[int]
    top_k_energies: List[float]
    aux_predictions: Optional[AuxPredictions] = None


class EBMRouter(nn.Module):
    """
    Energy-Based Model that scores (state, action, predicted_next_state) tuples.

    Architecture: 2-4 layer MLP, ~10-50M parameters.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        aux_dim: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Input: z_t ⊕ r_i ⊕ ẑ_{t+1} ⊕ aux_predictions
        input_dim = latent_dim + action_dim + latent_dim + aux_dim

        layers = []
        in_d = input_dim
        for i in range(num_layers):
            out_d = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = out_d

        # Scalar energy output
        layers.append(nn.Linear(in_d, 1))
        self.energy_net = nn.Sequential(*layers)

    def forward(
        self,
        z_t: torch.Tensor,
        action_emb: torch.Tensor,
        z_hat: torch.Tensor,
        aux: Optional[AuxPredictions] = None,
    ) -> torch.Tensor:
        """
        Compute energy for a single (state, action, predicted_state) tuple.

        Args:
            z_t: (batch, latent_dim)
            action_emb: (batch, action_dim)
            z_hat: (batch, latent_dim) — predicted next state
            aux: optional auxiliary predictions

        Returns:
            energy: (batch, 1) — lower is better
        """
        if aux is not None:
            aux_vec = torch.stack([
                aux.validity_prob,
                aux.score_improvement,
                aux.compute_cost,
                aux.repair_prob,
            ], dim=-1)
        else:
            aux_vec = torch.zeros(z_t.shape[0], 4, device=z_t.device)

        x = torch.cat([z_t, action_emb, z_hat, aux_vec], dim=-1)
        return self.energy_net(x)

    def score_candidates(
        self,
        z_t: torch.Tensor,
        action_embs: torch.Tensor,
        z_hats: torch.Tensor,
        aux_list: Optional[List[AuxPredictions]] = None,
        top_k: int = 1,
    ) -> RoutingDecision:
        """
        Score multiple candidates and return routing decision.

        Args:
            z_t: (1, latent_dim)
            action_embs: (1, K, action_dim)
            z_hats: (1, K, latent_dim)
            aux_list: optional list of K AuxPredictions
            top_k: number of top candidates to return

        Returns:
            RoutingDecision
        """
        K = action_embs.shape[1]
        energies = []

        for i in range(K):
            a_i = action_embs[:, i, :]
            z_hat_i = z_hats[:, i, :]
            aux_i = aux_list[i] if aux_list else None
            e_i = self.forward(z_t, a_i, z_hat_i, aux_i)
            energies.append(e_i.item())

        energy_tensor = torch.tensor(energies)
        sorted_idx = torch.argsort(energy_tensor).tolist()
        top_k_idx = sorted_idx[:top_k]

        return RoutingDecision(
            selected_idx=sorted_idx[0],
            selected_energy=energies[sorted_idx[0]],
            all_energies=energies,
            top_k_indices=top_k_idx,
            top_k_energies=[energies[i] for i in top_k_idx],
        )

    def ranking_loss(
        self,
        energy_good: torch.Tensor,
        energy_bad: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """
        Margin-based ranking loss: E(good) should be lower than E(bad) by margin.

        L = max(0, E(good) - E(bad) + margin)
        """
        return torch.clamp(energy_good - energy_bad + margin, min=0.0).mean()


class RuleBasedRouter:
    """
    Fallback rule-based router for Phase 2 (before EBM is trained).

    Uses simple heuristics to select among candidates.
    """

    # Priority by domain
    _DOMAIN_MODE_PRIORITY = {
        "planning": ["hierarchical", "llm_codegen", "repair", "global_opt"],
        "scheduling": ["global_opt", "hierarchical", "repair", "llm_codegen"],
        "optimization": ["global_opt", "llm_codegen", "hierarchical", "repair"],
        "coding": ["llm_codegen", "repair", "hierarchical", "global_opt"],
        "unknown": ["hierarchical", "global_opt", "llm_codegen", "repair"],
    }

    def select(
        self,
        candidates: list,
        domain: str = "unknown",
        feasible: bool = True,
        iteration: int = 0,
    ) -> int:
        """Return index of best candidate based on rules."""
        if not feasible and iteration > 0:
            # Prefer repair if infeasible (but not on first iteration — nothing to repair)
            for i, c in enumerate(candidates):
                mode = c.mode if hasattr(c, "mode") else c.get("mode", "")
                if mode == "repair":
                    return i

        priority = self._DOMAIN_MODE_PRIORITY.get(domain, self._DOMAIN_MODE_PRIORITY["unknown"])
        best_idx = 0
        best_rank = len(priority)
        for i, c in enumerate(candidates):
            mode = c.mode if hasattr(c, "mode") else c.get("mode", "")
            rank = priority.index(mode) if mode in priority else len(priority)
            if rank < best_rank:
                best_rank = rank
                best_idx = i
        return best_idx
