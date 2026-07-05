"""
Train the EBM (Energy-Based Model) Scorer on collected ARC-AGI-3 transitions.

The EBM learns to assign low energy to (state, action, outcome) tuples
that lead to progress (level change, new states) and high energy to
those that don't (no change, game over).

Training uses contrastive margin-ranking loss:
  - Positive: transitions where something changed / level progressed
  - Negative: transitions with no effect / game over

Requires a pre-trained world model checkpoint (for the encoder).

Usage:
    python training/train_ebm.py --data training/data --wm training/checkpoints/world_model_best.pt --epochs 30

Output:
    training/checkpoints/ebm_best.pt
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from v4_1_reasoning_system.arc_agi.game_world_model import (
    GameStateEncoder, GamePredictor, GameAuxHeads,
    WorldModelConfig, GameAuxPredictions,
)
from v4_1_reasoning_system.arc_agi.energy_scorer import GameEBM
from training.dataset import make_dataloaders
from training.train_world_model import ActionProjector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_world_model(
    ckpt_path: str, device: torch.device
) -> tuple:
    """Load pre-trained world model components (frozen)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    encoder = GameStateEncoder(cfg).to(device)
    predictor = GamePredictor(cfg).to(device)
    aux_heads = GameAuxHeads(cfg).to(device)
    action_proj = ActionProjector(cfg).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    aux_heads.load_state_dict(ckpt["aux_heads"])
    action_proj.load_state_dict(ckpt["action_proj"])

    # Freeze all world model components
    for m in [encoder, predictor, aux_heads, action_proj]:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    logger.info(f"Loaded world model from {ckpt_path} (epoch {ckpt['epoch']})")
    return cfg, encoder, predictor, aux_heads, action_proj


def train_one_epoch(
    ebm: GameEBM,
    encoder: GameStateEncoder,
    predictor: GamePredictor,
    aux_heads: GameAuxHeads,
    action_proj: ActionProjector,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    margin: float = 1.0,
) -> dict:
    """
    Train EBM with contrastive loss.

    For each batch, split into positive (progress) and negative (no progress)
    samples. Form pairs and train with margin ranking loss.
    """
    ebm.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        grid_before = batch["grid_before_oh"].to(device)
        ctx_before = batch["ctx_before"].to(device)
        action_bag = batch["action_bag"].to(device)
        level_changed = batch["level_changed"].float().to(device)
        game_over = batch["game_over"].float().to(device)
        anything_changed = batch["anything_changed"].float().to(device)

        # Encode states and predict outcomes (frozen WM)
        with torch.no_grad():
            z_t = encoder(grid_before, ctx_before)
            s_emb = action_proj(action_bag)
            z_hat = predictor(z_t, s_emb)
            aux = aux_heads(z_hat)

        # Compute energy for all samples
        energy = ebm(z_t, s_emb, z_hat, aux)

        # Define "good" = level changed OR anything changed (and not game over)
        is_good = ((level_changed > 0.5) | ((anything_changed > 0.5) & (game_over < 0.5)))
        is_bad = ~is_good

        good_idx = torch.where(is_good)[0]
        bad_idx = torch.where(is_bad)[0]

        if len(good_idx) == 0 or len(bad_idx) == 0:
            continue

        # Sample pairs: good energy should be lower than bad energy
        n_pairs = min(len(good_idx), len(bad_idx), 16)
        g_sample = good_idx[torch.randperm(len(good_idx))[:n_pairs]]
        b_sample = bad_idx[torch.randperm(len(bad_idx))[:n_pairs]]

        e_good = energy[g_sample]
        e_bad = energy[b_sample]

        # Margin ranking loss: E(good) < E(bad) - margin
        loss = torch.clamp(e_good - e_bad + margin, min=0.0).mean()

        # Regularize: push good energies down, bad energies up
        reg_loss = 0.01 * (e_good.mean() ** 2 + torch.clamp(-e_bad, min=0).mean())
        loss = loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(
    ebm: GameEBM,
    encoder: GameStateEncoder,
    predictor: GamePredictor,
    aux_heads: GameAuxHeads,
    action_proj: ActionProjector,
    loader,
    device: torch.device,
) -> dict:
    """Evaluate EBM: check if good transitions get lower energy than bad."""
    ebm.eval()

    good_energies = []
    bad_energies = []

    for batch in loader:
        grid_before = batch["grid_before_oh"].to(device)
        ctx_before = batch["ctx_before"].to(device)
        action_bag = batch["action_bag"].to(device)
        level_changed = batch["level_changed"].to(device)
        game_over = batch["game_over"].to(device)
        anything_changed = batch["anything_changed"].to(device)

        z_t = encoder(grid_before, ctx_before)
        s_emb = action_proj(action_bag)
        z_hat = predictor(z_t, s_emb)
        aux = aux_heads(z_hat)

        energy = ebm(z_t, s_emb, z_hat, aux)

        is_good = ((level_changed > 0.5) | ((anything_changed > 0.5) & (game_over < 0.5)))

        for i in range(energy.shape[0]):
            e = energy[i].item()
            if is_good[i]:
                good_energies.append(e)
            else:
                bad_energies.append(e)

    avg_good = sum(good_energies) / max(len(good_energies), 1)
    avg_bad = sum(bad_energies) / max(len(bad_energies), 1)

    # Ranking accuracy: how often E(good) < E(bad) for random pairs
    n_test = min(len(good_energies), len(bad_energies), 1000)
    correct = 0
    for _ in range(n_test):
        g = random.choice(good_energies) if good_energies else 0
        b = random.choice(bad_energies) if bad_energies else 0
        if g < b:
            correct += 1

    rank_acc = correct / max(n_test, 1)

    return {
        "avg_good_energy": avg_good,
        "avg_bad_energy": avg_bad,
        "energy_gap": avg_bad - avg_good,
        "rank_accuracy": rank_acc,
        "n_good": len(good_energies),
        "n_bad": len(bad_energies),
    }


def main():
    parser = argparse.ArgumentParser(description="Train EBM Scorer")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "training" / "data"))
    parser.add_argument("--wm", type=str, default=str(PROJECT_ROOT / "training" / "checkpoints" / "world_model_best.pt"),
                        help="Path to pre-trained world model checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "training" / "checkpoints"))
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen world model
    cfg, encoder, predictor, aux_heads, action_proj = load_world_model(args.wm, device)

    # Build EBM
    ebm = GameEBM(cfg).to(device)
    n_params = sum(p.numel() for p in ebm.parameters())
    logger.info(f"EBM parameters: {n_params:,}")

    # Load data
    train_loader, test_loader = make_dataloaders(
        args.data, batch_size=args.batch, seed=42
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Training loop
    best_rank_acc = 0.0

    logger.info(f"Training EBM for {args.epochs} epochs on {device}")
    logger.info("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            ebm, encoder, predictor, aux_heads, action_proj,
            optimizer, train_loader, device, margin=args.margin,
        )
        test_metrics = evaluate(
            ebm, encoder, predictor, aux_heads, action_proj,
            test_loader, device,
        )
        scheduler.step()

        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Test: E_good={test_metrics['avg_good_energy']:.3f} "
            f"E_bad={test_metrics['avg_bad_energy']:.3f} "
            f"gap={test_metrics['energy_gap']:.3f} "
            f"rank_acc={test_metrics['rank_accuracy']:.2%}"
        )

        # Save best by ranking accuracy
        if test_metrics["rank_accuracy"] > best_rank_acc:
            best_rank_acc = test_metrics["rank_accuracy"]
            ckpt = {
                "epoch": epoch,
                "cfg": cfg,
                "ebm": ebm.state_dict(),
                "test_metrics": test_metrics,
            }
            torch.save(ckpt, out_dir / "ebm_best.pt")
            logger.info(f"  -> Saved best EBM (rank_acc={best_rank_acc:.2%})")

    # Save final
    ckpt = {
        "epoch": args.epochs,
        "cfg": cfg,
        "ebm": ebm.state_dict(),
        "test_metrics": test_metrics,
    }
    torch.save(ckpt, out_dir / "ebm_final.pt")

    logger.info(f"\nTraining complete. Best rank accuracy: {best_rank_acc:.2%}")
    logger.info(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
