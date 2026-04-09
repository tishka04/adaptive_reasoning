"""
World model trainer — trains the JEPA-style transition predictor
and auxiliary heads on collected reasoning trajectories.

Training objectives:
  1. Latent prediction loss: ||ẑ_{t+1} - z_{t+1}||^2
  2. Auxiliary losses: validity, score improvement, compute cost, repair need

Phase 3 in the training order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..memory.replay import ReplayBuffer
from ..world_model.aux_heads import AuxiliaryHeads, AuxPredictions
from ..world_model.encoder import StateEncoder
from ..world_model.predictor import ActionEncoder, TransitionPredictor


class WorldModelDataset(Dataset):
    """
    Dataset of (z_t, action, z_{t+1}_actual, aux_targets) tuples
    extracted from reasoning trajectories.
    """

    def __init__(self, records: List[Dict[str, torch.Tensor]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.records[idx]


def build_world_model_dataset(
    replay_buffer: ReplayBuffer,
    latent_dim: int = 128,
    action_dim: int = 32,
) -> List[Dict[str, torch.Tensor]]:
    """
    Extract training data from the replay buffer.

    For each consecutive pair of steps in a trajectory, create a record:
      - z_t: latent state before action
      - action_emb: action embedding
      - z_next: actual latent state after action
      - targets: {valid, score_delta, cost, needed_repair}
    """
    records = []

    for traj in replay_buffer._buffer:
        tensors = replay_buffer.get_tensors(traj.trajectory_id)

        for i, step in enumerate(traj.steps):
            z_t = tensors.get(f"step_{step.step_idx}_z_t")
            action = tensors.get(f"step_{step.step_idx}_action_emb")
            z_actual = tensors.get(f"step_{step.step_idx}_z_actual")

            # If no actual next state, try the next step's z_t
            if z_actual is None and i + 1 < len(traj.steps):
                next_step = traj.steps[i + 1]
                z_actual = tensors.get(f"step_{next_step.step_idx}_z_t")

            if z_t is None or action is None or z_actual is None:
                continue

            # Flatten dimensions
            z_t_flat = z_t.squeeze(0) if z_t.dim() > 1 else z_t
            action_flat = action.squeeze(0) if action.dim() > 1 else action
            z_actual_flat = z_actual.squeeze(0) if z_actual.dim() > 1 else z_actual

            # Ensure correct dimensions
            if z_t_flat.shape[0] != latent_dim or z_actual_flat.shape[0] != latent_dim:
                continue
            if action_flat.shape[0] != action_dim:
                continue

            # Build auxiliary targets from step metadata
            verifier = step.verifier_result or {}
            solver = step.solver_result or {}

            records.append({
                "z_t": z_t_flat,
                "action_emb": action_flat,
                "z_next": z_actual_flat,
                "valid": torch.tensor(float(step.success), dtype=torch.float32),
                "score_delta": torch.tensor(step.score_delta, dtype=torch.float32),
                "cost": torch.tensor(
                    min(step.elapsed_seconds / 30.0, 1.0), dtype=torch.float32
                ),
                "needed_repair": torch.tensor(
                    1.0 if not step.success else 0.0, dtype=torch.float32
                ),
            })

    return records


class WorldModelTrainer:
    """
    Trains the transition predictor and auxiliary heads.

    Losses:
      - Latent MSE: ||P(z_t, r) - z_{t+1}||^2
      - Aux losses: BCE for validity/repair, MSE for score/cost
    """

    def __init__(
        self,
        predictor: TransitionPredictor,
        aux_heads: AuxiliaryHeads,
        lr: float = 3e-4,
        latent_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.5,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.predictor = predictor
        self.aux_heads = aux_heads
        self.device = device
        self.latent_loss_weight = latent_loss_weight
        self.aux_loss_weight = aux_loss_weight

        self.optimizer = optim.AdamW(
            list(predictor.parameters()) + list(aux_heads.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch. Returns metrics dict."""
        self.predictor.train()
        self.aux_heads.train()

        total_latent_loss = 0.0
        total_aux_loss = 0.0
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            z_t = batch["z_t"]
            action = batch["action_emb"]
            z_next = batch["z_next"]

            # Predict next state
            z_hat = self.predictor(z_t, action)

            # Latent prediction loss
            latent_loss = nn.functional.mse_loss(z_hat, z_next)

            # Auxiliary predictions and loss
            aux_preds = self.aux_heads(z_hat)
            aux_targets = {
                "valid": batch["valid"],
                "score_delta": batch["score_delta"],
                "cost": batch["cost"],
                "needed_repair": batch["needed_repair"],
            }
            aux_loss = self.aux_heads.loss(aux_preds, aux_targets)

            # Combined loss
            loss = self.latent_loss_weight * latent_loss + self.aux_loss_weight * aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.predictor.parameters()) + list(self.aux_heads.parameters()),
                1.0,
            )
            self.optimizer.step()

            total_latent_loss += latent_loss.item()
            total_aux_loss += aux_loss.item()
            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()

        return {
            "loss": total_loss / max(n_batches, 1),
            "latent_loss": total_latent_loss / max(n_batches, 1),
            "aux_loss": total_aux_loss / max(n_batches, 1),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.predictor.eval()
        self.aux_heads.eval()

        total_latent_loss = 0.0
        total_aux_loss = 0.0
        validity_correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                z_hat = self.predictor(batch["z_t"], batch["action_emb"])
                latent_loss = nn.functional.mse_loss(z_hat, batch["z_next"])

                aux_preds = self.aux_heads(z_hat)
                aux_targets = {
                    "valid": batch["valid"],
                    "score_delta": batch["score_delta"],
                    "cost": batch["cost"],
                    "needed_repair": batch["needed_repair"],
                }
                aux_loss = self.aux_heads.loss(aux_preds, aux_targets)

                total_latent_loss += latent_loss.item()
                total_aux_loss += aux_loss.item()

                # Accuracy for validity prediction
                preds = (aux_preds.validity_prob > 0.5).float()
                validity_correct += (preds == batch["valid"]).sum().item()
                total += batch["valid"].shape[0]

        n = max(len(dataloader), 1)
        return {
            "val_latent_loss": total_latent_loss / n,
            "val_aux_loss": total_aux_loss / n,
            "val_validity_acc": validity_correct / max(total, 1),
        }
