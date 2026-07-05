"""
Energy-Based Model Scorer — scores candidate game strategies.

Adapts the v4_1 EBMRouter for the game domain:
  E(z_t, s_i, ẑ_{t+1}^{(i)}) → scalar energy

Low energy = promising strategy.

Combines:
  - Learned energy from neural network (trained online)
  - Heuristic energy from auxiliary predictions (progress, risk, novelty)
  - Prior energy from template confidence scores

The EBM learns which (state, strategy, predicted_outcome) combinations
lead to progress, and assigns low energy to promising ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn as nn

from .game_world_model import GameAuxPredictions, WorldModelConfig
from .strategy_generator import GameStrategy
from .goal_pursuit import GameObjective

if TYPE_CHECKING:
    from .trajectory_sampler import SampledTrajectory

logger = logging.getLogger(__name__)


@dataclass
class ScoringDecision:
    """Result of EBM scoring over candidate strategies."""
    selected_idx: int
    selected_energy: float
    all_energies: List[float]
    selected_strategy: GameStrategy
    top_k_indices: List[int]


@dataclass
class TrajectoryScoringDecision:
    """Result of scoring sampled trajectories."""

    selected_idx: int
    selected_energy: float
    selected_score: float
    all_energies: List[float]
    all_scores: List[float]
    selected_trajectory: "SampledTrajectory"
    top_k_indices: List[int]


class GameEBM(nn.Module):
    """
    Energy-Based Model for scoring game strategies.

    E(z_t, s_i, ẑ_{t+1}) = neural_energy + heuristic_energy

    Lower energy = better strategy.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg

        # Neural energy: (z_t ⊕ s_i ⊕ ẑ_{t+1} ⊕ aux) → scalar
        input_dim = cfg.latent_dim + cfg.strategy_dim + cfg.latent_dim + 3  # 3 = aux
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LayerNorm(cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

        # Heuristic weights (learned online)
        self.heuristic_weights = nn.Parameter(torch.tensor([
            -2.0,   # progress_prob: negative = low energy when high progress
            +2.0,   # risk_prob: positive = high energy when high risk
            -1.0,   # novelty_score: negative = low energy when high novelty
            -0.5,   # confidence: negative = low energy when high confidence
        ]))

    def forward(
        self,
        z_t: torch.Tensor,           # (B, latent_dim)
        strategy_emb: torch.Tensor,   # (B, strategy_dim)
        z_hat: torch.Tensor,          # (B, latent_dim)
        aux: GameAuxPredictions,
    ) -> torch.Tensor:
        """Compute energy for a (state, strategy, predicted_outcome) tuple."""
        aux_vec = torch.stack([
            aux.progress_prob,
            aux.risk_prob,
            aux.novelty_score,
        ], dim=-1)

        x = torch.cat([z_t, strategy_emb, z_hat, aux_vec], dim=-1)
        neural_energy = self.energy_net(x).squeeze(-1)

        return neural_energy

    def score_strategies(
        self,
        z_t: torch.Tensor,                        # (1, latent_dim)
        strategies: List[GameStrategy],
        strategy_embs: torch.Tensor,               # (1, K, strategy_dim)
        z_hats: torch.Tensor,                      # (1, K, latent_dim)
        aux_list: List[GameAuxPredictions],
        top_k: int = 1,
    ) -> ScoringDecision:
        """
        Score K candidate strategies and select the best one.

        Returns:
            ScoringDecision with selected strategy and all energies.
        """
        K = len(strategies)
        energies: List[float] = []

        with torch.no_grad():
            for i in range(K):
                s_emb = strategy_embs[:, i, :]
                z_hat_i = z_hats[:, i, :]
                aux_i = aux_list[i]

                # Neural energy
                neural_e = self.forward(z_t, s_emb, z_hat_i, aux_i)

                # Heuristic energy from aux predictions + confidence prior
                heuristic_features = torch.tensor([
                    aux_i.progress_prob.item(),
                    aux_i.risk_prob.item(),
                    aux_i.novelty_score.item(),
                    strategies[i].confidence,
                ], device=z_t.device)
                heuristic_e = (self.heuristic_weights * heuristic_features).sum()

                total_energy = neural_e.item() + heuristic_e.item()
                energies.append(total_energy)

        # Select lowest energy
        sorted_indices = sorted(range(K), key=lambda i: energies[i])
        top_k_idx = sorted_indices[:top_k]

        return ScoringDecision(
            selected_idx=sorted_indices[0],
            selected_energy=energies[sorted_indices[0]],
            all_energies=energies,
            selected_strategy=strategies[sorted_indices[0]],
            top_k_indices=top_k_idx,
        )


class GameEnergyScorer:
    """
    High-level interface for energy-based strategy scoring.

    Wraps the GameEBM and manages online learning from feedback.
    """

    def __init__(self, cfg: Optional[WorldModelConfig] = None):
        self.cfg = cfg or WorldModelConfig()
        self.device = torch.device(self.cfg.device)
        self.ebm = GameEBM(self.cfg).to(self.device)
        self.ebm.eval()

        # Training state
        self._feedback_buffer: List[dict] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._trained: bool = False

    def score(
        self,
        z_t: torch.Tensor,
        strategies: List[GameStrategy],
        strategy_embs: torch.Tensor,
        z_hats: torch.Tensor,
        aux_list: List[GameAuxPredictions],
    ) -> ScoringDecision:
        """Score strategies and return the best one."""
        return self.ebm.score_strategies(
            z_t, strategies, strategy_embs, z_hats, aux_list
        )

    def score_for_goal(
        self,
        z_t: torch.Tensor,
        strategies: List[GameStrategy],
        strategy_embs: torch.Tensor,
        z_hats: torch.Tensor,
        aux_list: List[GameAuxPredictions],
        goal: GameObjective,
    ) -> ScoringDecision:
        """Score goal-strategy pairs, not bare strategies.

        Adds a goal-alignment bias: strategies tagged with the matching
        goal_id get an energy reduction proportional to goal confidence.
        This makes the EBM answer "how promising is this strategy FOR
        advancing objective G?" instead of just "how promising is this
        strategy?".
        """
        base_decision = self.ebm.score_strategies(
            z_t, strategies, strategy_embs, z_hats, aux_list
        )

        # Apply goal-alignment bias
        adjusted_energies = list(base_decision.all_energies)
        for i, s in enumerate(strategies):
            goal_id = s.metadata.get("goal_id", "")
            if goal_id == goal.id:
                # Matching goal → reduce energy (more promising)
                adjusted_energies[i] -= goal.confidence * 1.5
            else:
                # Mismatched goal → increase energy
                adjusted_energies[i] += 0.5

        # Re-rank with adjusted energies
        K = len(strategies)
        sorted_indices = sorted(range(K), key=lambda i: adjusted_energies[i])
        top_k_idx = sorted_indices[:1]

        return ScoringDecision(
            selected_idx=sorted_indices[0],
            selected_energy=adjusted_energies[sorted_indices[0]],
            all_energies=adjusted_energies,
            selected_strategy=strategies[sorted_indices[0]],
            top_k_indices=top_k_idx,
        )

    def score_trajectories(
        self,
        z_t: torch.Tensor,
        trajectories: List["SampledTrajectory"],
        strategy_embs: torch.Tensor,
        z_hats: torch.Tensor,
        aux_list: List[GameAuxPredictions],
        *,
        human_cap: float = 0.25,
    ) -> TrajectoryScoringDecision:
        """Score sampled trajectories using energy plus explicit planning terms."""
        if not trajectories:
            raise ValueError("score_trajectories requires at least one trajectory")

        energies: List[float] = []
        scores: List[float] = []
        with torch.no_grad():
            for i, traj in enumerate(trajectories):
                s_emb = strategy_embs[:, i, :]
                z_hat_i = z_hats[:, i, :]
                aux_i = aux_list[i]

                neural_e = self.ebm(z_t, s_emb, z_hat_i, aux_i)
                heuristic_features = torch.tensor([
                    aux_i.progress_prob.item(),
                    aux_i.risk_prob.item(),
                    aux_i.novelty_score.item(),
                    traj.metadata.get("generator_confidence", 0.5),
                ], device=z_t.device)
                heuristic_e = (self.ebm.heuristic_weights * heuristic_features).sum()
                total_energy = float(neural_e.item() + heuristic_e.item())
                energies.append(total_energy)

                goal_progress = float(
                    traj.metadata.get("goal_progress", aux_i.progress_prob.item())
                )
                novelty = float(traj.metadata.get("novelty", aux_i.novelty_score.item()))
                risk = float(traj.metadata.get("risk", aux_i.risk_prob.item()))
                human_compat = min(
                    human_cap,
                    float(traj.metadata.get("human_compatibility", 0.0)),
                )
                score = (
                    -total_energy
                    + goal_progress
                    + novelty
                    - risk
                    + human_compat
                )
                traj.energy = total_energy
                traj.score = score
                traj.novelty = novelty
                traj.risk = risk
                scores.append(score)

        sorted_indices = sorted(range(len(trajectories)), key=lambda i: scores[i], reverse=True)
        top_k_idx = sorted_indices[: min(3, len(sorted_indices))]
        return TrajectoryScoringDecision(
            selected_idx=sorted_indices[0],
            selected_energy=energies[sorted_indices[0]],
            selected_score=scores[sorted_indices[0]],
            all_energies=energies,
            all_scores=scores,
            selected_trajectory=trajectories[sorted_indices[0]],
            top_k_indices=top_k_idx,
        )

    def record_feedback(
        self,
        z_t: torch.Tensor,
        strategy_emb: torch.Tensor,
        z_hat: torch.Tensor,
        aux: GameAuxPredictions,
        was_good: bool,
    ) -> None:
        """Record whether a strategy choice was good or bad."""
        self._feedback_buffer.append({
            "z_t": z_t.detach().cpu(),
            "strategy_emb": strategy_emb.detach().cpu(),
            "z_hat": z_hat.detach().cpu(),
            "aux_progress": aux.progress_prob.detach().cpu(),
            "aux_risk": aux.risk_prob.detach().cpu(),
            "aux_novelty": aux.novelty_score.detach().cpu(),
            "was_good": was_good,
        })

    def record_goal_feedback(
        self,
        z_t: torch.Tensor,
        strategy_emb: torch.Tensor,
        z_hat: torch.Tensor,
        aux: GameAuxPredictions,
        progress_score: float,
    ) -> None:
        """Record goal-conditioned feedback using continuous progress score.

        Progress > 0.15 counts as good, otherwise bad. The score is
        stored for future weighted training.
        """
        from .goal_pursuit import PARTIAL_THRESHOLD
        self._feedback_buffer.append({
            "z_t": z_t.detach().cpu(),
            "strategy_emb": strategy_emb.detach().cpu(),
            "z_hat": z_hat.detach().cpu(),
            "aux_progress": aux.progress_prob.detach().cpu(),
            "aux_risk": aux.risk_prob.detach().cpu(),
            "aux_novelty": aux.novelty_score.detach().cpu(),
            "was_good": progress_score >= PARTIAL_THRESHOLD,
            "progress_score": progress_score,
        })

    def update(self, train_steps: int = 3) -> float:
        """Train EBM on feedback: good strategies → low energy."""
        if len(self._feedback_buffer) < 4:
            return 0.0

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.ebm.parameters(), lr=self.cfg.learning_rate
            )

        self.ebm.train()
        total_loss = 0.0

        import random
        for _ in range(train_steps):
            # Sample good and bad examples
            good = [f for f in self._feedback_buffer if f["was_good"]]
            bad = [f for f in self._feedback_buffer if not f["was_good"]]

            if not good or not bad:
                break

            g = random.choice(good)
            b = random.choice(bad)

            # Compute energies
            aux_g = GameAuxPredictions(
                g["aux_progress"].to(self.device),
                g["aux_risk"].to(self.device),
                g["aux_novelty"].to(self.device),
            )
            aux_b = GameAuxPredictions(
                b["aux_progress"].to(self.device),
                b["aux_risk"].to(self.device),
                b["aux_novelty"].to(self.device),
            )

            e_good = self.ebm(
                g["z_t"].to(self.device),
                g["strategy_emb"].to(self.device),
                g["z_hat"].to(self.device),
                aux_g,
            )
            e_bad = self.ebm(
                b["z_t"].to(self.device),
                b["strategy_emb"].to(self.device),
                b["z_hat"].to(self.device),
                aux_b,
            )

            # Margin ranking loss: E(good) < E(bad) - margin
            loss = torch.clamp(e_good - e_bad + 1.0, min=0.0).mean()

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item()

        self.ebm.eval()
        self._trained = True
        return total_loss / max(train_steps, 1)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save EBM weights to a checkpoint file."""
        torch.save({
            "cfg": self.cfg,
            "ebm": self.ebm.state_dict(),
            "trained": self._trained,
        }, path)
        logger.info(f"EBM scorer saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load pre-trained EBM weights from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.ebm.load_state_dict(ckpt["ebm"])
        self._trained = ckpt.get("trained", True)
        self.ebm.eval()
        logger.info(f"EBM scorer loaded from {path}")
