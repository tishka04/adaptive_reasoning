"""
Router training — trains the EBM router on reasoning trajectory data.

Uses ranking loss on good vs bad reasoning actions.
Phase 4 in the training order.

Provides a high-level function that:
  1. Extracts training pairs from the replay buffer
  2. Creates train/val splits
  3. Trains the EBM router
  4. Returns training metrics
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, random_split

from ..memory.replay import ReplayBuffer
from ..router.ebm_router import EBMRouter
from ..router.routing_train import (
    RouterTrainer,
    RouterTrajectoryDataset,
    collate_router_batch,
)


def train_router_from_buffer(
    router: EBMRouter,
    replay_buffer: ReplayBuffer,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    margin: float = 1.0,
    val_split: float = 0.2,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end router training from replay buffer data.

    Args:
        router: EBMRouter to train
        replay_buffer: buffer containing reasoning trajectories
        num_epochs: training epochs
        batch_size: batch size
        lr: learning rate
        margin: ranking loss margin
        val_split: fraction of data for validation
        device: compute device
        verbose: print progress

    Returns:
        Dict with training history and final metrics
    """
    # Extract training pairs
    pairs = replay_buffer.get_training_pairs()

    if len(pairs) < 4:
        return {
            "status": "insufficient_data",
            "num_pairs": len(pairs),
            "message": f"Need at least 4 training pairs, got {len(pairs)}. Collect more rollouts first.",
        }

    if verbose:
        print(f"Router training: {len(pairs)} pairs extracted from {len(replay_buffer)} trajectories")

    # Create dataset
    dataset = RouterTrajectoryDataset(pairs)

    # Train/val split
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_router_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_router_batch,
    )

    # Train
    trainer = RouterTrainer(
        router=router,
        lr=lr,
        margin=margin,
        device=device,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_accuracy"])

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            best_state = {k: v.clone() for k, v in router.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{num_epochs}: "
                f"loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} "
                f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_accuracy']:.3f}"
            )

    # Restore best model
    if best_state is not None:
        router.load_state_dict(best_state)

    return {
        "status": "trained",
        "num_pairs": len(pairs),
        "num_epochs": num_epochs,
        "best_val_accuracy": best_val_acc,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
        "history": history,
    }
