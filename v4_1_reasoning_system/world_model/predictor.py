"""
JEPA-style transition predictor — predicts the latent future state
of the reasoning process after a candidate reasoning action.

    ẑ_{t+1} = P(z_t, r_t)

Does NOT reconstruct raw tokens or full observations. Predicts only
the latent consequences that matter for routing decisions.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TransitionPredictor(nn.Module):
    """
    Given current latent state z_t and a reasoning action embedding r,
    predict the next latent state ẑ_{t+1}.

    Architecture: MLP with residual connection, ~20-50M params.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Action embedding projection
        self.action_proj = nn.Linear(action_dim, latent_dim)

        # Transition MLP
        layers = []
        in_dim = latent_dim * 2  # concat(z_t, proj(r))
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.transition = nn.Sequential(*layers)

        # Residual gate: learnable interpolation between current state and prediction
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (batch, latent_dim) — current latent reasoning state
            action_embedding: (batch, action_dim) — reasoning action vector

        Returns:
            z_hat: (batch, latent_dim) — predicted next latent state
        """
        r_proj = self.action_proj(action_embedding)
        combined = torch.cat([z_t, r_proj], dim=-1)
        delta = self.transition(combined)

        # Gated residual: z_hat = gate * z_t + (1 - gate) * delta
        gate_input = torch.cat([z_t, delta], dim=-1)
        g = self.gate(gate_input)
        z_hat = g * z_t + (1.0 - g) * delta
        return z_hat

    def predict_batch(
        self,
        z_t: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state for multiple candidate actions at once.

        Args:
            z_t: (batch, latent_dim)
            action_embeddings: (batch, num_candidates, action_dim)

        Returns:
            z_hats: (batch, num_candidates, latent_dim)
        """
        B, K, A = action_embeddings.shape
        z_t_expanded = z_t.unsqueeze(1).expand(B, K, -1)  # (B, K, latent_dim)

        z_flat = z_t_expanded.reshape(B * K, -1)
        a_flat = action_embeddings.reshape(B * K, A)

        z_hat_flat = self.forward(z_flat, a_flat)
        return z_hat_flat.reshape(B, K, self.latent_dim)


class ActionEncoder(nn.Module):
    """
    Encodes a reasoning candidate description into a dense vector.

    A reasoning candidate is: (mode, budget, strictness, tool_hint)
    """

    MODE_VOCAB = ["hierarchical", "global_opt", "repair", "llm_codegen"]
    STRICTNESS_VOCAB = ["fast", "verified", "strict"]

    def __init__(self, action_dim: int = 32):
        super().__init__()
        self.action_dim = action_dim

        self.mode_emb = nn.Embedding(len(self.MODE_VOCAB), 8)
        self.strictness_emb = nn.Embedding(len(self.STRICTNESS_VOCAB), 4)
        # budget is a scalar 0-1
        # tool_hint encoded as small hash
        self.proj = nn.Sequential(
            nn.Linear(8 + 4 + 1 + 8, action_dim),
            nn.LayerNorm(action_dim),
            nn.GELU(),
        )

        self._mode_to_idx = {m: i for i, m in enumerate(self.MODE_VOCAB)}
        self._strict_to_idx = {s: i for i, s in enumerate(self.STRICTNESS_VOCAB)}

    def forward(
        self,
        mode_idx: torch.Tensor,
        budget: torch.Tensor,
        strictness_idx: torch.Tensor,
        tool_hint_hash: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mode_idx: (batch,) int
            budget: (batch, 1) float
            strictness_idx: (batch,) int
            tool_hint_hash: (batch, 8) float — simple hash of tool hint string

        Returns:
            (batch, action_dim)
        """
        m = self.mode_emb(mode_idx)
        s = self.strictness_emb(strictness_idx)
        combined = torch.cat([m, s, budget, tool_hint_hash], dim=-1)
        return self.proj(combined)

    def encode_candidate(self, candidate: dict, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Encode a single candidate dict into an action embedding."""
        mode = candidate.get("mode", "hierarchical")
        mode_idx = torch.tensor([self._mode_to_idx.get(mode, 0)], device=device)

        budget_val = {"low": 0.25, "medium": 0.5, "high": 0.75}.get(
            candidate.get("budget", "medium"), 0.5
        )
        budget = torch.tensor([[budget_val]], device=device)

        strict = candidate.get("strictness", "verified")
        strict_idx = torch.tensor([self._strict_to_idx.get(strict, 1)], device=device)

        hint = candidate.get("tool_hint", "")
        hint_hash = torch.zeros(1, 8, device=device)
        for i, ch in enumerate(hint[:8]):
            hint_hash[0, i] = float(ord(ch) % 100) / 100.0

        return self.forward(mode_idx, budget, strict_idx, hint_hash)

    def encode_candidates(self, candidates: list, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Encode a list of candidate dicts → (1, K, action_dim)."""
        embeddings = [self.encode_candidate(c, device) for c in candidates]
        return torch.stack(embeddings, dim=1)  # (1, K, action_dim)
