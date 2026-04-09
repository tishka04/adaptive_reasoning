"""
Training utilities for the EBM router.

Uses ranking loss on good vs bad reasoning actions collected from
reasoning trajectories stored in memory/replay.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .ebm_router import EBMRouter


class RouterTrajectoryDataset(Dataset):
    """
    Dataset of (z_t, action_good, z_hat_good, action_bad, z_hat_bad, aux_good, aux_bad)
    tuples extracted from reasoning rollouts.

    A "good" action is one that led to verified improvement.
    A "bad" action is one that led to failure or regression.
    """

    def __init__(self, records: List[Dict[str, torch.Tensor]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.records[idx]


def collate_router_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for RouterTrajectoryDataset."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class RouterTrainer:
    """
    Trains the EBM router on paired (good, bad) reasoning actions.

    Training objective: margin ranking loss
        L = max(0, E(good) - E(bad) + margin)
    """

    def __init__(
        self,
        router: EBMRouter,
        lr: float = 1e-4,
        margin: float = 1.0,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.router = router
        self.margin = margin
        self.device = device
        self.optimizer = optim.AdamW(
            router.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 50,
    ) -> Dict[str, float]:
        """Train one epoch. Returns metrics dict."""
        self.router.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            z_t = batch["z_t"]
            a_good = batch["action_good"]
            z_hat_good = batch["z_hat_good"]
            a_bad = batch["action_bad"]
            z_hat_bad = batch["z_hat_bad"]

            # Compute energies
            e_good = self.router(z_t, a_good, z_hat_good)
            e_bad = self.router(z_t, a_bad, z_hat_bad)

            # Ranking loss
            loss = self.router.ranking_loss(e_good, e_bad, self.margin)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            correct += (e_good < e_bad).sum().item()
            total += z_t.shape[0]

        self.scheduler.step()

        return {
            "loss": total_loss / max(len(dataloader), 1),
            "accuracy": correct / max(total, 1),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a validation set."""
        self.router.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                z_t = batch["z_t"]
                a_good = batch["action_good"]
                z_hat_good = batch["z_hat_good"]
                a_bad = batch["action_bad"]
                z_hat_bad = batch["z_hat_bad"]

                e_good = self.router(z_t, a_good, z_hat_good)
                e_bad = self.router(z_t, a_bad, z_hat_bad)

                loss = self.router.ranking_loss(e_good, e_bad, self.margin)
                total_loss += loss.item()
                correct += (e_good < e_bad).sum().item()
                total += z_t.shape[0]

        return {
            "val_loss": total_loss / max(len(dataloader), 1),
            "val_accuracy": correct / max(total, 1),
        }

    @staticmethod
    def build_pairs_from_rollout(
        rollout: List[Dict],
        latent_dim: int = 128,
        action_dim: int = 32,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert a reasoning rollout into training pairs.

        Each rollout step should have:
            - z_t: latent state
            - candidates: list of (action_emb, z_hat, outcome)
            - selected_idx: which candidate was chosen
            - success: bool

        Returns list of paired records for RouterTrajectoryDataset.
        """
        pairs = []
        for step in rollout:
            candidates = step.get("candidates", [])
            if len(candidates) < 2:
                continue

            z_t = step["z_t"]
            good_candidates = [c for c in candidates if c.get("outcome", {}).get("success", False)]
            bad_candidates = [c for c in candidates if not c.get("outcome", {}).get("success", True)]

            for g in good_candidates:
                for b in bad_candidates:
                    pairs.append({
                        "z_t": z_t,
                        "action_good": g["action_emb"],
                        "z_hat_good": g["z_hat"],
                        "action_bad": b["action_emb"],
                        "z_hat_bad": b["z_hat"],
                    })

        return pairs
