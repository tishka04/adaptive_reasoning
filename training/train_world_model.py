"""
Train the JEPA World Model on collected ARC-AGI-3 transitions.

The world model learns to:
  1. Encode game grids → latent z_t  (GameStateEncoder)
  2. Predict z_{t+1} from (z_t, action)  (GamePredictor)
  3. Predict auxiliary signals: progress, risk, novelty  (GameAuxHeads)

Loss = JEPA prediction loss + auxiliary BCE/MSE losses

Usage:
    python training/train_world_model.py --data training/data --epochs 50 --batch 32

Output:
    training/checkpoints/world_model_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from v4_1_reasoning_system.arc_agi.game_world_model import (
    GameStateEncoder, StrategyEncoder, GamePredictor, GameAuxHeads,
    WorldModelConfig,
)
from training.dataset import make_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class ActionProjector(nn.Module):
    """
    Projects action bag-of-words (8,) into strategy_dim for the predictor.

    During inference the StrategyEncoder produces strategy_dim embeddings
    from GameStrategy objects. For training we don't have GameStrategy
    objects — instead we have the raw action taken. This small module
    bridges that gap so the predictor can be trained end-to-end.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(8, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.strategy_dim),
            nn.LayerNorm(cfg.strategy_dim),
        )

    def forward(self, action_bag: torch.Tensor) -> torch.Tensor:
        return self.proj(action_bag)


def train_one_epoch(
    encoder: GameStateEncoder,
    predictor: GamePredictor,
    aux_heads: GameAuxHeads,
    action_proj: ActionProjector,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    aux_weight: float = 0.5,
) -> dict:
    encoder.train()
    predictor.train()
    aux_heads.train()
    action_proj.train()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0

    for batch in loader:
        grid_before = batch["grid_before_oh"].to(device)
        grid_after = batch["grid_after_oh"].to(device)
        ctx_before = batch["ctx_before"].to(device)
        ctx_after = batch["ctx_after"].to(device)
        action_bag = batch["action_bag"].to(device)
        level_changed = batch["level_changed"].float().to(device)
        game_over = batch["game_over"].float().to(device)
        anything_changed = batch["anything_changed"].float().to(device)
        num_changes = batch["num_changes"].float().to(device)

        # Encode before and after states
        z_t = encoder(grid_before, ctx_before)
        z_next = encoder(grid_after, ctx_after)

        # Project action → strategy-like embedding
        s_emb = action_proj(action_bag)

        # Predict next latent
        z_hat = predictor(z_t, s_emb)

        # JEPA loss: predicted latent should match actual next latent
        pred_loss = nn.functional.mse_loss(z_hat, z_next.detach())

        # Auxiliary losses
        aux = aux_heads(z_hat)
        progress_loss = nn.functional.binary_cross_entropy(
            aux.progress_prob, level_changed
        )
        risk_loss = nn.functional.binary_cross_entropy(
            aux.risk_prob, game_over
        )
        # Novelty: normalize num_changes by grid area
        grid_shape = batch["grid_shape"]
        grid_area = (grid_shape[0] * grid_shape[1]).float().to(device)
        novelty_target = (num_changes / grid_area.clamp(min=1)).clamp(0, 1).float()
        novelty_loss = nn.functional.mse_loss(aux.novelty_score, novelty_target)

        aux_loss = progress_loss + risk_loss + novelty_loss
        loss = pred_loss + aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_aux_loss += aux_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "pred_loss": total_pred_loss / max(n_batches, 1),
        "aux_loss": total_aux_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    encoder: GameStateEncoder,
    predictor: GamePredictor,
    aux_heads: GameAuxHeads,
    action_proj: ActionProjector,
    loader,
    device: torch.device,
    aux_weight: float = 0.5,
) -> dict:
    encoder.eval()
    predictor.eval()
    aux_heads.eval()
    action_proj.eval()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0
    # Track aux accuracy
    correct_progress = 0
    correct_risk = 0
    total_samples = 0

    for batch in loader:
        grid_before = batch["grid_before_oh"].to(device)
        grid_after = batch["grid_after_oh"].to(device)
        ctx_before = batch["ctx_before"].to(device)
        ctx_after = batch["ctx_after"].to(device)
        action_bag = batch["action_bag"].to(device)
        level_changed = batch["level_changed"].float().to(device)
        game_over = batch["game_over"].float().to(device)
        num_changes = batch["num_changes"].float().to(device)

        z_t = encoder(grid_before, ctx_before)
        z_next = encoder(grid_after, ctx_after)
        s_emb = action_proj(action_bag)
        z_hat = predictor(z_t, s_emb)

        pred_loss = nn.functional.mse_loss(z_hat, z_next)

        aux = aux_heads(z_hat)
        progress_loss = nn.functional.binary_cross_entropy(
            aux.progress_prob, level_changed
        )
        risk_loss = nn.functional.binary_cross_entropy(
            aux.risk_prob, game_over
        )
        grid_shape = batch["grid_shape"]
        grid_area = (grid_shape[0] * grid_shape[1]).float().to(device)
        novelty_target = (num_changes / grid_area.clamp(min=1)).clamp(0, 1).float()
        novelty_loss = nn.functional.mse_loss(aux.novelty_score, novelty_target)

        aux_loss = progress_loss + risk_loss + novelty_loss
        loss = pred_loss + aux_weight * aux_loss

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_aux_loss += aux_loss.item()
        n_batches += 1

        # Accuracy
        bs = level_changed.shape[0]
        correct_progress += ((aux.progress_prob > 0.5).float() == level_changed).sum().item()
        correct_risk += ((aux.risk_prob > 0.5).float() == game_over).sum().item()
        total_samples += bs

    return {
        "loss": total_loss / max(n_batches, 1),
        "pred_loss": total_pred_loss / max(n_batches, 1),
        "aux_loss": total_aux_loss / max(n_batches, 1),
        "progress_acc": correct_progress / max(total_samples, 1),
        "risk_acc": correct_risk / max(total_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train JEPA World Model")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "training" / "data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "training" / "checkpoints"))
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading transitions from {args.data}")
    train_loader, test_loader = make_dataloaders(
        args.data, batch_size=args.batch, seed=42
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")

    # Build model
    cfg = WorldModelConfig(device=args.device)
    encoder = GameStateEncoder(cfg).to(device)
    predictor = GamePredictor(cfg).to(device)
    aux_heads = GameAuxHeads(cfg).to(device)
    action_proj = ActionProjector(cfg).to(device)

    # Count parameters
    n_params = sum(
        p.numel() for m in [encoder, predictor, aux_heads, action_proj]
        for p in m.parameters()
    )
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    all_params = (
        list(encoder.parameters()) +
        list(predictor.parameters()) +
        list(aux_heads.parameters()) +
        list(action_proj.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Training loop
    best_test_loss = float("inf")

    logger.info(f"Training for {args.epochs} epochs on {device}")
    logger.info("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            encoder, predictor, aux_heads, action_proj,
            optimizer, train_loader, device,
        )
        test_metrics = evaluate(
            encoder, predictor, aux_heads, action_proj,
            test_loader, device,
        )
        scheduler.step()

        elapsed = time.time() - t0

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train: loss={train_metrics['loss']:.4f} pred={train_metrics['pred_loss']:.4f} aux={train_metrics['aux_loss']:.4f} | "
            f"Test: loss={test_metrics['loss']:.4f} pred={test_metrics['pred_loss']:.4f} "
            f"progress_acc={test_metrics['progress_acc']:.2%} risk_acc={test_metrics['risk_acc']:.2%}"
        )

        # Save best
        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            ckpt = {
                "epoch": epoch,
                "cfg": cfg,
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "aux_heads": aux_heads.state_dict(),
                "action_proj": action_proj.state_dict(),
                "test_loss": test_metrics["loss"],
                "test_metrics": test_metrics,
            }
            torch.save(ckpt, out_dir / "world_model_best.pt")
            logger.info(f"  → Saved best model (test_loss={best_test_loss:.4f})")

    # Save final
    ckpt = {
        "epoch": args.epochs,
        "cfg": cfg,
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "aux_heads": aux_heads.state_dict(),
        "action_proj": action_proj.state_dict(),
        "test_loss": test_metrics["loss"],
    }
    torch.save(ckpt, out_dir / "world_model_final.pt")

    logger.info(f"\nTraining complete. Best test loss: {best_test_loss:.4f}")
    logger.info(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
