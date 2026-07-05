"""
Game World Model — JEPA-style latent prediction for ARC-AGI-3.

Adapts the v4_1 architecture:
  - StateEncoder   → GameStateEncoder:  grid + memory → latent z_t
  - TransitionPredictor → GamePredictor: (z_t, strategy) → ẑ_{t+1}
  - AuxiliaryHeads → GameAuxHeads: ẑ_{t+1} → (progress_prob, risk, novelty)

The world model predicts in LATENT space (not pixel space), following
the JEPA principle: predict abstract consequences, not raw observations.

Learns online from game experience — no pre-training required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .state_describer import GameObservation
from .strategy_generator import GameStrategy, StrategyType

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class WorldModelConfig:
    """Configuration for the game world model."""
    grid_channels: int = 16         # max distinct cell values (0-15)
    max_grid_size: int = 64         # max grid dimension
    latent_dim: int = 64            # latent space dimensionality
    strategy_dim: int = 32          # strategy embedding dim
    hidden_dim: int = 128           # MLP hidden dim
    num_layers: int = 2             # MLP depth
    dropout: float = 0.1
    learning_rate: float = 1e-3
    device: str = "cpu"


# ------------------------------------------------------------------
# Game State Encoder: grid + context → latent z_t
# ------------------------------------------------------------------
class GameStateEncoder(nn.Module):
    """
    Encodes an ARC-AGI-3 game observation into a fixed-size latent vector z_t.

    Architecture:
      - Grid → one-hot → small CNN → flatten → MLP
      - Context features (player pos, action counts, level, etc.) → MLP
      - Concatenate → fusion MLP → z_t
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg

        # Grid encoder: one-hot(16) → Conv → pool → flatten
        self.grid_conv = nn.Sequential(
            nn.Conv2d(cfg.grid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),  # → (32, 8, 8)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # → (16, 4, 4) = 256
            nn.Flatten(),
        )
        grid_flat_dim = 16 * 4 * 4  # 256

        # Context encoder: scalar features → MLP
        context_dim = 20  # player pos(2) + level(1) + actions(1) + states(1) + ...
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
        )

        # Fusion: grid_flat + context → z_t
        fusion_in = grid_flat_dim + cfg.hidden_dim // 2
        layers: list[nn.Module] = []
        in_d = fusion_in
        for _ in range(cfg.num_layers):
            layers.extend([
                nn.Linear(in_d, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
            in_d = cfg.hidden_dim
        layers.append(nn.Linear(cfg.hidden_dim, cfg.latent_dim))
        self.fusion = nn.Sequential(*layers)

    def forward(
        self,
        grid_onehot: torch.Tensor,     # (B, 16, H, W)
        context_vec: torch.Tensor,      # (B, context_dim)
    ) -> torch.Tensor:
        """Encode game state → latent z_t (B, latent_dim)."""
        g = self.grid_conv(grid_onehot)
        c = self.context_proj(context_vec)
        combined = torch.cat([g, c], dim=-1)
        return self.fusion(combined)


# ------------------------------------------------------------------
# Strategy Encoder: GameStrategy → dense embedding
# ------------------------------------------------------------------
class StrategyEncoder(nn.Module):
    """Encodes a GameStrategy into a dense vector for the predictor."""

    _TYPE_VOCAB = list(StrategyType)

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        n_types = len(self._TYPE_VOCAB)

        self.type_emb = nn.Embedding(n_types, 8)
        self.confidence_proj = nn.Linear(1, 4)
        # Action plan: bag-of-actions encoding (8 possible actions)
        self.action_proj = nn.Linear(8, 12)
        # Fusion
        self.proj = nn.Sequential(
            nn.Linear(8 + 4 + 12, cfg.strategy_dim),
            nn.LayerNorm(cfg.strategy_dim),
            nn.GELU(),
        )
        self._type_to_idx = {t: i for i, t in enumerate(self._TYPE_VOCAB)}

    def encode_strategy(
        self, strategy: GameStrategy, device: torch.device
    ) -> torch.Tensor:
        """Encode a single strategy → (1, strategy_dim)."""
        # Type embedding
        type_idx = self._type_to_idx.get(strategy.strategy_type, 0)
        t_emb = self.type_emb(torch.tensor([type_idx], device=device))

        # Confidence
        conf = self.confidence_proj(
            torch.tensor([[strategy.confidence]], device=device)
        )

        # Bag-of-actions: count occurrences of each ACTION
        action_bag = torch.zeros(1, 8, device=device)
        for a in strategy.action_plan:
            if a == "RESET":
                action_bag[0, 0] += 1
            else:
                try:
                    idx = int(a.replace("ACTION", ""))
                    if 1 <= idx <= 7:
                        action_bag[0, idx] += 1
                except ValueError:
                    pass
        a_emb = self.action_proj(action_bag)

        combined = torch.cat([t_emb, conf, a_emb], dim=-1)
        return self.proj(combined)

    def encode_batch(
        self, strategies: List[GameStrategy], device: torch.device
    ) -> torch.Tensor:
        """Encode multiple strategies → (1, K, strategy_dim)."""
        embs = [self.encode_strategy(s, device) for s in strategies]
        return torch.stack(embs, dim=1)


# ------------------------------------------------------------------
# Transition Predictor: (z_t, strategy_emb) → ẑ_{t+1}
# ------------------------------------------------------------------
class GamePredictor(nn.Module):
    """
    JEPA-style predictor: predicts latent next state from
    current state and strategy embedding.

    Uses gated residual (same as v4_1 TransitionPredictor).
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg

        self.strategy_proj = nn.Linear(cfg.strategy_dim, cfg.latent_dim)

        layers: list[nn.Module] = []
        in_d = cfg.latent_dim * 2
        for i in range(cfg.num_layers):
            out_d = cfg.hidden_dim if i < cfg.num_layers - 1 else cfg.latent_dim
            layers.extend([
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
            in_d = out_d
        self.transition = nn.Sequential(*layers)

        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_t: torch.Tensor,            # (B, latent_dim)
        strategy_emb: torch.Tensor,    # (B, strategy_dim)
    ) -> torch.Tensor:
        """Predict latent next state: ẑ_{t+1} = P(z_t, s)."""
        s_proj = self.strategy_proj(strategy_emb)
        combined = torch.cat([z_t, s_proj], dim=-1)
        delta = self.transition(combined)
        g = self.gate(torch.cat([z_t, delta], dim=-1))
        return g * z_t + (1.0 - g) * delta

    def predict_batch(
        self,
        z_t: torch.Tensor,            # (1, latent_dim)
        strategy_embs: torch.Tensor,   # (1, K, strategy_dim)
    ) -> torch.Tensor:
        """Predict for K strategies → (1, K, latent_dim)."""
        B, K, S = strategy_embs.shape
        z_exp = z_t.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        s_flat = strategy_embs.reshape(B * K, S)
        z_hat = self.forward(z_exp, s_flat)
        return z_hat.reshape(B, K, self.cfg.latent_dim)


# ------------------------------------------------------------------
# Auxiliary prediction heads
# ------------------------------------------------------------------
@dataclass
class GameAuxPredictions:
    """Predicted consequences of a strategy."""
    progress_prob: torch.Tensor     # probability of level progress
    risk_prob: torch.Tensor         # probability of game over
    novelty_score: torch.Tensor     # expected state novelty (new states)


class GameAuxHeads(nn.Module):
    """Auxiliary heads on predicted latent: progress, risk, novelty."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        dim = cfg.latent_dim

        def _head(act: Optional[nn.Module] = None) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)]
            if act:
                layers.append(act)
            return nn.Sequential(*layers)

        self.progress_head = _head(nn.Sigmoid())
        self.risk_head = _head(nn.Sigmoid())
        self.novelty_head = _head(nn.Sigmoid())

    def forward(self, z_hat: torch.Tensor) -> GameAuxPredictions:
        return GameAuxPredictions(
            progress_prob=self.progress_head(z_hat).squeeze(-1),
            risk_prob=self.risk_head(z_hat).squeeze(-1),
            novelty_score=self.novelty_head(z_hat).squeeze(-1),
        )


# ------------------------------------------------------------------
# Full Game World Model
# ------------------------------------------------------------------
class GameWorldModel:
    """
    Complete JEPA-style world model for ARC-AGI-3 games.

    Composes encoder, strategy encoder, predictor, and aux heads.
    Supports online learning from game transitions.
    """

    def __init__(self, cfg: Optional[WorldModelConfig] = None):
        self.cfg = cfg or WorldModelConfig()
        self.device = torch.device(self.cfg.device)

        # Build modules
        self.state_encoder = GameStateEncoder(self.cfg).to(self.device)
        self.strategy_encoder = StrategyEncoder(self.cfg).to(self.device)
        self.predictor = GamePredictor(self.cfg).to(self.device)
        self.aux_heads = GameAuxHeads(self.cfg).to(self.device)

        # Online learning buffer
        self._transition_buffer: List[Dict] = []  # observed transitions
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._trained_steps: int = 0

    def encode_observation(self, obs: GameObservation) -> torch.Tensor:
        """Encode a game observation into latent z_t."""
        grid_oh = self._grid_to_onehot(obs.raw_grid)
        ctx = self._observation_to_context(obs)
        with torch.no_grad():
            z = self.state_encoder(grid_oh, ctx)
        return z

    def predict_strategy_outcomes(
        self,
        z_t: torch.Tensor,
        strategies: List[GameStrategy],
    ) -> Tuple[torch.Tensor, List[GameAuxPredictions]]:
        """
        Predict latent outcomes for each candidate strategy.

        Returns:
            z_hats: (1, K, latent_dim) — predicted next states
            aux_list: list of K GameAuxPredictions
        """
        s_embs = self.strategy_encoder.encode_batch(strategies, self.device)
        with torch.no_grad():
            z_hats = self.predictor.predict_batch(z_t, s_embs)
            aux_list = []
            for k in range(z_hats.shape[1]):
                aux = self.aux_heads(z_hats[:, k, :])
                aux_list.append(aux)
        return z_hats, aux_list

    def record_transition(
        self,
        obs_before: GameObservation,
        strategy: GameStrategy,
        obs_after: GameObservation,
        level_changed: bool,
        game_over: bool,
        states_discovered: int,
    ) -> None:
        """Record an observed transition for online learning."""
        self._transition_buffer.append({
            "grid_before": obs_before.raw_grid.copy(),
            "ctx_before": self._observation_to_context(obs_before).detach(),
            "strategy": strategy,
            "grid_after": obs_after.raw_grid.copy(),
            "ctx_after": self._observation_to_context(obs_after).detach(),
            "level_changed": level_changed,
            "game_over": game_over,
            "states_discovered": states_discovered,
        })

    def update(self, train_steps: int = 5) -> float:
        """
        Online update: train on recent transitions.

        Returns average loss.
        """
        if len(self._transition_buffer) < 2:
            return 0.0

        if self._optimizer is None:
            params = (
                list(self.state_encoder.parameters())
                + list(self.predictor.parameters())
                + list(self.aux_heads.parameters())
            )
            self._optimizer = torch.optim.Adam(
                params, lr=self.cfg.learning_rate
            )

        self.state_encoder.train()
        self.predictor.train()
        self.aux_heads.train()

        total_loss = 0.0
        for _ in range(train_steps):
            # Sample a mini-batch from buffer
            import random
            batch = random.sample(
                self._transition_buffer,
                min(len(self._transition_buffer), 8),
            )

            loss = self._compute_loss(batch)
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.state_encoder.parameters(), 1.0
            )
            self._optimizer.step()
            total_loss += loss.item()
            self._trained_steps += 1

        self.state_encoder.eval()
        self.predictor.eval()
        self.aux_heads.eval()

        return total_loss / train_steps

    def _compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        """Compute JEPA-style loss on a batch of transitions."""
        loss = torch.tensor(0.0, device=self.device)

        for item in batch:
            # Encode before and after states
            g_before = self._grid_to_onehot(item["grid_before"])
            g_after = self._grid_to_onehot(item["grid_after"])
            ctx_before = item["ctx_before"].to(self.device)
            ctx_after = item["ctx_after"].to(self.device)

            z_t = self.state_encoder(g_before, ctx_before)
            z_next = self.state_encoder(g_after, ctx_after)

            # Predict next state from strategy
            s_emb = self.strategy_encoder.encode_strategy(
                item["strategy"], self.device
            )
            z_hat = self.predictor(z_t, s_emb)

            # JEPA loss: predicted latent should match actual latent
            prediction_loss = nn.functional.mse_loss(z_hat, z_next.detach())

            # Auxiliary losses
            aux = self.aux_heads(z_hat)
            progress_target = torch.tensor(
                [1.0 if item["level_changed"] else 0.0], device=self.device
            )
            risk_target = torch.tensor(
                [1.0 if item["game_over"] else 0.0], device=self.device
            )
            novelty_target = torch.tensor(
                [min(item["states_discovered"] / 10.0, 1.0)], device=self.device
            )

            aux_loss = (
                nn.functional.binary_cross_entropy(aux.progress_prob, progress_target)
                + nn.functional.binary_cross_entropy(aux.risk_prob, risk_target)
                + nn.functional.mse_loss(aux.novelty_score, novelty_target)
            )

            loss = loss + prediction_loss + 0.5 * aux_loss

        return loss / len(batch)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save model weights to a checkpoint file."""
        torch.save({
            "cfg": self.cfg,
            "encoder": self.state_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "aux_heads": self.aux_heads.state_dict(),
            "strategy_encoder": self.strategy_encoder.state_dict(),
            "trained_steps": self._trained_steps,
        }, path)
        logger.info(f"World model saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load pre-trained weights from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.state_encoder.load_state_dict(ckpt["encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.aux_heads.load_state_dict(ckpt["aux_heads"])
        if "strategy_encoder" in ckpt:
            self.strategy_encoder.load_state_dict(ckpt["strategy_encoder"])
        self._trained_steps = ckpt.get("trained_steps", 0)
        logger.info(f"World model loaded from {path} (trained_steps={self._trained_steps})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _grid_to_onehot(self, grid: np.ndarray) -> torch.Tensor:
        """Convert grid to one-hot tensor (1, C, H, W)."""
        h, w = grid.shape
        # Pad or crop to max_grid_size
        ms = self.cfg.max_grid_size
        padded = np.zeros((ms, ms), dtype=np.int64)
        ph, pw = min(h, ms), min(w, ms)
        padded[:ph, :pw] = grid[:ph, :pw]

        t = torch.tensor(padded, dtype=torch.long, device=self.device)
        t = t.clamp(0, self.cfg.grid_channels - 1)
        onehot = nn.functional.one_hot(t, self.cfg.grid_channels)
        # (H, W, C) → (C, H, W) → (1, C, H, W)
        return onehot.permute(2, 0, 1).float().unsqueeze(0)

    def _observation_to_context(self, obs: GameObservation) -> torch.Tensor:
        """Extract context features from observation → (1, context_dim)."""
        vec = torch.zeros(1, 20, device=self.device)

        # Player position (normalized)
        if obs.player_info:
            h, w = obs.raw_grid.shape
            vec[0, 0] = obs.player_info["y"] / max(h, 1)
            vec[0, 1] = obs.player_info["x"] / max(w, 1)
            vec[0, 2] = 1.0  # player identified

        # Level and counters
        vec[0, 3] = obs.level / 10.0
        vec[0, 4] = obs.action_counter / 200.0

        # Memory stats
        ms = obs.memory_summary
        vec[0, 5] = ms.get("total_actions", 0) / 200.0
        vec[0, 6] = ms.get("states_visited", 0) / 200.0
        vec[0, 7] = ms.get("max_level", 0) / 10.0
        vec[0, 8] = ms.get("total_game_overs", 0) / 10.0
        vec[0, 9] = ms.get("total_resets", 0) / 10.0

        # Grid stats
        h, w = obs.raw_grid.shape
        vec[0, 10] = h / 64.0
        vec[0, 11] = w / 64.0
        vec[0, 12] = len(obs.objects) / 20.0

        # Action semantics
        vec[0, 13] = len(obs.action_semantics) / 7.0

        # Game state encoding
        state_map = {"NOT_PLAYED": 0, "NOT_FINISHED": 0.5, "WIN": 1.0, "GAME_OVER": -1.0}
        vec[0, 14] = state_map.get(obs.game_state, 0)

        return vec
