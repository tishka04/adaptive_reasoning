"""
Visual Cortex -- CNN-based frame predictor for ARC-AGI-3 games.

Predicts the next game grid given the current grid and an action.
Trains online from observed (grid, action, next_grid) transitions
collected during Phase 1 fast exploration.  Used in Phase 2 to:

  1. Predict consequences of candidate actions (mental simulation)
  2. Describe action effects in natural language for LLM strategy generation
  3. Enable multi-step look-ahead planning without taking real actions

Architecture
------------
Compact U-Net with FiLM (Feature-wise Linear Modulation) action
conditioning.

  Input : one-hot grid  (B, 16, 64, 64)  +  action features  (B, 10)
  Output: per-pixel logits (B, 16, 64, 64) -> argmax -> predicted grid

  Encoder   3 x (Conv3x3 + BN + ReLU) with MaxPool    ~97 K params
  FiLM      action -> per-channel scale & shift          ~8 K params
  Bottleneck Conv3x3 + BN + ReLU                       ~148 K params
  Decoder   3 x (Upsample + skip-cat + Conv3x3)        ~147 K params
  Head      1x1 conv                                      256 params
                                                 Total  ~450 K params

The residual shortcut from input one-hot to output logits biases the
model toward "no change", so it only needs to learn the delta.

Loss: per-pixel cross-entropy with extra weight on changed cells.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class VisualCortexConfig:
    """Configuration for the Visual Cortex."""
    max_grid_size: int = 64       # pad all grids to this
    num_classes: int = 16         # cell values 0-15
    num_actions: int = 8          # RESET(0) + ACTION1-7
    enc_channels: Tuple[int, ...] = (32, 64, 128)
    action_emb_dim: int = 32     # dense action embedding size
    buffer_capacity: int = 10_000
    batch_size: int = 32
    learning_rate: float = 1e-3
    change_weight: float = 5.0   # extra loss weight on changed pixels
    device: str = "cpu"


# ======================================================================
# Building blocks
# ======================================================================
class ConvBlock(nn.Module):
    """Conv 3x3 + BatchNorm + ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Conditions convolutional feature maps on a conditioning vector
    by computing per-channel scale (gamma) and shift (beta):

        features' = gamma * features + beta

    Initialised close to identity (gamma~1, beta~0) so the
    untrained model passes inputs through unchanged.
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, num_channels)
        self.beta_proj = nn.Linear(cond_dim, num_channels)
        # Identity init
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(
        self, features: torch.Tensor, cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) conv feature maps
            cond:     (B, cond_dim) conditioning vector
        Returns:
            Modulated features (B, C, H, W)
        """
        gamma = self.gamma_proj(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta


# ======================================================================
# Frame Predictor (U-Net with FiLM)
# ======================================================================
class FramePredictor(nn.Module):
    """
    Compact U-Net with FiLM action conditioning.

    Given a one-hot encoded grid + raw action features, predicts
    per-pixel logits for the next grid.

    Skip connections keep most static cells accurate.
    A residual shortcut from input one-hot to output biases the
    model toward "no change" (learns only the delta).
    """

    def __init__(self, cfg: VisualCortexConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.num_classes
        ch = cfg.enc_channels  # (32, 64, 128)
        raw_action_dim = cfg.num_actions + 2  # one-hot(8) + norm x,y (2)

        # Raw action features -> dense embedding
        self.action_proj = nn.Sequential(
            nn.Linear(raw_action_dim, cfg.action_emb_dim),
            nn.ReLU(),
            nn.Linear(cfg.action_emb_dim, cfg.action_emb_dim),
            nn.ReLU(),
        )

        # Encoder
        self.enc1 = ConvBlock(C, ch[0])       # 16 -> 32
        self.enc2 = ConvBlock(ch[0], ch[1])   # 32 -> 64
        self.enc3 = ConvBlock(ch[1], ch[2])   # 64 -> 128
        self.pool = nn.MaxPool2d(2)

        # Bottleneck with FiLM conditioning
        self.film = FiLMLayer(cfg.action_emb_dim, ch[2])
        self.bottleneck = ConvBlock(ch[2], ch[2])  # 128 -> 128

        # Decoder with skip connections
        self.up3 = ConvBlock(ch[2] + ch[2], ch[1])   # 256 -> 64
        self.up2 = ConvBlock(ch[1] + ch[1], ch[0])   # 128 -> 32
        self.up1 = ConvBlock(ch[0] + ch[0], C)       # 64  -> 16

        # Output head
        self.head = nn.Conv2d(C, C, 1)

    def forward(
        self,
        grid_onehot: torch.Tensor,     # (B, 16, 64, 64)
        action_raw: torch.Tensor,      # (B, 10) raw action features
    ) -> torch.Tensor:
        """Returns per-pixel logits (B, 16, 64, 64)."""
        action_emb = self.action_proj(action_raw)  # (B, action_emb_dim)

        # ── Encoder ──
        e1 = self.enc1(grid_onehot)            # (B, 32, 64, 64)
        e2 = self.enc2(self.pool(e1))          # (B, 64, 32, 32)
        e3 = self.enc3(self.pool(e2))          # (B, 128, 16, 16)

        # ── Bottleneck with action conditioning ──
        b = self.pool(e3)                      # (B, 128, 8, 8)
        b = self.film(b, action_emb)
        b = self.bottleneck(b)                 # (B, 128, 8, 8)

        # ── Decoder ──
        d3 = F.interpolate(b, size=e3.shape[2:], mode="nearest")
        d3 = self.up3(torch.cat([d3, e3], dim=1))  # (B, 64, 16, 16)

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="nearest")
        d2 = self.up2(torch.cat([d2, e2], dim=1))  # (B, 32, 32, 32)

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="nearest")
        d1 = self.up1(torch.cat([d1, e1], dim=1))  # (B, 16, 64, 64)

        logits = self.head(d1)

        # Residual shortcut: bias toward "no change"
        logits = logits + grid_onehot

        return logits


# ======================================================================
# Transition buffer item
# ======================================================================
@dataclass
class _Transition:
    grid_before: np.ndarray          # (H, W) uint8
    action: int                      # 0-7
    action_data: Optional[Dict]      # {"x": .., "y": ..} for ACTION6
    grid_after: np.ndarray           # (H, W) uint8


# ======================================================================
# Visual Cortex (high-level manager)
# ======================================================================
class VisualCortex:
    """
    Complete visual-cortex system for ARC-AGI-3 games.

    Lifecycle
    ---------
    Phase 1 (exploration):
        for each action taken:
            cortex.record_transition(before, act, data, after)
        cortex.train(steps=50)   # end of Phase 1

    Phase 2 (strategy):
        descs = cortex.describe_action_effects(grid, avail)
        # -> inject `descs` into LLM strategy-generation prompt
        preds = cortex.predict_all_actions(grid, avail)
        # -> mental simulation for planning

    Public API
    ----------
    record_transition(...)   add observed transition to buffer
    train(steps)             online training, returns avg loss
    predict(grid, act)       single action prediction
    predict_all_actions(...) batched prediction for all available actions
    imagine_sequence(...)    multi-step mental simulation
    describe_action_effects(...) NL descriptions per action
    get_action_summary(...)  formatted summary for LLM context
    """

    def __init__(self, cfg: Optional[VisualCortexConfig] = None):
        self.cfg = cfg or VisualCortexConfig()
        self.device = torch.device(self.cfg.device)
        self.model = FramePredictor(self.cfg).to(self.device)
        self.model.eval()

        self._buffer: deque[_Transition] = deque(maxlen=self.cfg.buffer_capacity)
        self._optimizer: Optional[torch.optim.Adam] = None
        self._trained_steps: int = 0
        self._grid_size: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def trained_steps(self) -> int:
        return self._trained_steps

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def record_transition(
        self,
        grid_before: np.ndarray,
        action: int,
        action_data: Optional[Dict] = None,
        grid_after: Optional[np.ndarray] = None,
    ) -> None:
        """Record an observed (state, action, next_state) transition."""
        if grid_before is None or grid_after is None:
            return
        if self._grid_size is None:
            self._grid_size = grid_before.shape[:2]
        self._buffer.append(_Transition(
            grid_before=grid_before.copy(),
            action=action,
            action_data=dict(action_data) if action_data else None,
            grid_after=grid_after.copy(),
        ))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, steps: int = 10) -> float:
        """Online training on buffered transitions. Returns avg loss."""
        if len(self._buffer) < 4:
            return 0.0

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.cfg.learning_rate,
            )

        self.model.train()
        total_loss = 0.0

        for _ in range(steps):
            batch = random.sample(
                list(self._buffer),
                min(len(self._buffer), self.cfg.batch_size),
            )

            grids_oh, act_raws, targets, change_masks = [], [], [], []
            for t in batch:
                gb = self._pad_grid(t.grid_before)
                ga = self._pad_grid(t.grid_after)
                grids_oh.append(self._grid_to_onehot(gb))
                act_raws.append(self._raw_action_vec(t.action, t.action_data, gb.shape))
                targets.append(torch.tensor(ga, dtype=torch.long, device=self.device))
                change_masks.append(torch.tensor(
                    (gb != ga).astype(np.float32), device=self.device,
                ))

            grid_oh = torch.stack(grids_oh)          # (B, 16, 64, 64)
            act_raw = torch.stack(act_raws)           # (B, 10)
            target = torch.stack(targets)             # (B, 64, 64) long
            mask = torch.stack(change_masks)          # (B, 64, 64) float

            logits = self.model(grid_oh, act_raw)     # (B, 16, 64, 64)

            # Weighted cross-entropy: upweight changed cells
            ce = F.cross_entropy(logits, target, reduction="none")  # (B, 64, 64)
            weight = 1.0 + self.cfg.change_weight * mask
            loss = (ce * weight).mean()

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._optimizer.step()

            total_loss += loss.item()
            self._trained_steps += 1

        self.model.eval()
        return total_loss / max(steps, 1)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        grid: np.ndarray,
        action: int,
        action_data: Optional[Dict] = None,
    ) -> np.ndarray:
        """Predict next grid for a single action.

        Returns:
            Predicted grid (H, W) uint8, same shape as input.
        """
        self.model.eval()
        h, w = grid.shape[:2]
        padded = self._pad_grid(grid)

        grid_oh = self._grid_to_onehot(padded).unsqueeze(0)
        act_raw = self._raw_action_vec(action, action_data, padded.shape).unsqueeze(0)

        logits = self.model(grid_oh, act_raw)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred[:h, :w]

    @torch.no_grad()
    def predict_all_actions(
        self,
        grid: np.ndarray,
        available_actions: List[int],
    ) -> Dict[int, np.ndarray]:
        """Predict next grid for all available actions (batched).

        Returns:
            Dict mapping action_int -> predicted grid (H, W) uint8.
        """
        self.model.eval()
        if not available_actions:
            return {}

        h, w = grid.shape[:2]
        padded = self._pad_grid(grid)
        grid_oh = self._grid_to_onehot(padded).unsqueeze(0)  # (1, 16, 64, 64)

        K = len(available_actions)
        grid_batch = grid_oh.expand(K, -1, -1, -1)  # (K, 16, 64, 64)
        act_batch = torch.stack([
            self._raw_action_vec(a, None, padded.shape) for a in available_actions
        ])  # (K, 10)

        logits = self.model(grid_batch, act_batch)  # (K, 16, 64, 64)
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (K, 64, 64)

        return {act: preds[i, :h, :w] for i, act in enumerate(available_actions)}

    @torch.no_grad()
    def imagine_sequence(
        self,
        grid: np.ndarray,
        actions: List[Tuple[int, Optional[Dict]]],
    ) -> List[np.ndarray]:
        """Simulate a multi-step action sequence (auto-regressive).

        Returns:
            List of predicted grids, one per action.
        """
        self.model.eval()
        frames: List[np.ndarray] = []
        current = grid.copy()
        for act_int, act_data in actions:
            pred = self.predict(current, act_int, act_data)
            frames.append(pred)
            current = pred
        return frames

    # ------------------------------------------------------------------
    # Action-effect description (for LLM strategy generation)
    # ------------------------------------------------------------------
    def describe_action_effects(
        self,
        grid: np.ndarray,
        available_actions: List[int],
    ) -> Dict[int, str]:
        """Describe what each action does in natural language.

        Returns:
            Dict mapping action_int -> human-readable description.
        """
        if self._trained_steps < 10:
            return {a: "unknown (visual cortex not yet trained)" for a in available_actions}

        predictions = self.predict_all_actions(grid, available_actions)
        return {
            act: self._describe_diff(grid, pred, act)
            for act, pred in predictions.items()
        }

    def get_action_summary(
        self,
        grid: np.ndarray,
        available_actions: List[int],
    ) -> str:
        """Formatted summary of all action effects for LLM context injection."""
        descs = self.describe_action_effects(grid, available_actions)
        lines = ["Predicted action effects (visual cortex):"]
        for act in sorted(descs.keys()):
            lines.append(f"  ACTION{act}: {descs[act]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Structured analysis (for associative memory integration)
    # ------------------------------------------------------------------
    def analyze_action_effects(
        self,
        grid: np.ndarray,
        available_actions: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Produce structured per-action analysis for memory integration.

        Returns dict mapping action_int -> {
            "change_rate": float,      # 0-1 fraction of cells predicted to change
            "direction": (dy, dx),     # predicted displacement or None
            "danger_score": float,     # 0-1 estimated risk (drastic change proxy)
            "n_changed": int,          # number of cells predicted to change
        }
        """
        if self._trained_steps < 10:
            return {}

        predictions = self.predict_all_actions(grid, available_actions)
        results: Dict[int, Dict[str, Any]] = {}

        for act, pred in predictions.items():
            diff_mask = grid != pred
            n_changed = int(diff_mask.sum())
            total = max(grid.size, 1)
            change_rate = n_changed / total

            # Movement detection
            direction: Optional[Tuple[float, float]] = None
            appeared = (grid == 0) & (pred != 0)
            disappeared = (grid != 0) & (pred == 0)
            if appeared.any() and disappeared.any():
                app_ys, app_xs = np.where(appeared)
                dis_ys, dis_xs = np.where(disappeared)
                if len(app_ys) > 0 and len(dis_ys) > 0:
                    dy = float(app_ys.mean() - dis_ys.mean())
                    dx = float(app_xs.mean() - dis_xs.mean())
                    if abs(dy) > 0.1 or abs(dx) > 0.1:
                        direction = (dy, dx)

            # Danger: large changes or many disappeared non-zero cells
            danger = 0.0
            if change_rate > 0.3:
                danger = min(1.0, change_rate)
            if disappeared.sum() > appeared.sum() * 2:
                danger = max(danger, 0.5)

            results[act] = {
                "change_rate": change_rate,
                "direction": direction,
                "danger_score": danger,
                "n_changed": n_changed,
            }

        return results

    @torch.no_grad()
    def compute_action_similarity(
        self,
        grid: np.ndarray,
        available_actions: List[int],
    ) -> Dict[Tuple[int, int], float]:
        """Compute pairwise similarity between action effects.

        Returns dict mapping (act_i, act_j) -> cosine_similarity in [0, 1].
        Actions with similar predicted grids get high similarity.
        """
        if self._trained_steps < 10 or len(available_actions) < 2:
            return {}

        predictions = self.predict_all_actions(grid, available_actions)

        # Flatten predicted grids into vectors
        flat: Dict[int, np.ndarray] = {}
        for act, pred in predictions.items():
            flat[act] = pred.flatten().astype(np.float32)

        similarity: Dict[Tuple[int, int], float] = {}
        acts = list(flat.keys())
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                a, b = flat[acts[i]], flat[acts[j]]
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a > 0 and norm_b > 0:
                    cos = float(np.dot(a, b) / (norm_a * norm_b))
                else:
                    cos = 1.0
                similarity[(acts[i], acts[j])] = cos
                similarity[(acts[j], acts[i])] = cos

        return similarity

    # ------------------------------------------------------------------
    # Diff analysis helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _describe_diff(
        before: np.ndarray, after: np.ndarray, action: int,
    ) -> str:
        """Analyse grid difference and produce a natural-language description."""
        diff_mask = before != after
        n_changed = int(diff_mask.sum())
        total = before.size

        if n_changed == 0:
            return "no visible effect"

        if n_changed > total * 0.5:
            return f"major change ({n_changed}/{total} cells)"

        changed_ys, changed_xs = np.where(diff_mask)
        parts: List[str] = []

        # ── Movement detection ──────────────────────────────────
        appeared = (before == 0) & (after != 0)
        disappeared = (before != 0) & (after == 0)

        if appeared.any() and disappeared.any():
            app_ys, app_xs = np.where(appeared)
            dis_ys, dis_xs = np.where(disappeared)
            if len(app_ys) > 0 and len(dis_ys) > 0:
                dy = float(app_ys.mean() - dis_ys.mean())
                dx = float(app_xs.mean() - dis_xs.mean())
                if abs(dy) > abs(dx):
                    direction = "down" if dy > 0 else "up"
                elif abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "diagonal"
                parts.append(f"moves {direction} (~{abs(dy):.0f}dy, ~{abs(dx):.0f}dx)")

        # ── Scope ───────────────────────────────────────────────
        if n_changed <= 10:
            parts.append(f"{n_changed} cells changed")
        else:
            yr = (int(changed_ys.min()), int(changed_ys.max()))
            xr = (int(changed_xs.min()), int(changed_xs.max()))
            parts.append(
                f"{n_changed} cells in y=[{yr[0]},{yr[1]}] x=[{xr[0]},{xr[1]}]"
            )

        # ── New value colours ───────────────────────────────────
        old_vals = set(before[diff_mask].tolist())
        new_vals = set(after[diff_mask].tolist())
        novel = sorted(new_vals - old_vals)
        if novel:
            parts.append(f"new values {novel}")

        return "; ".join(parts) if parts else f"{n_changed} cells changed"

    # ------------------------------------------------------------------
    # Tensor / grid helpers
    # ------------------------------------------------------------------
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad grid to (max_grid_size, max_grid_size)."""
        ms = self.cfg.max_grid_size
        h, w = grid.shape[:2]
        if h == ms and w == ms:
            return grid
        padded = np.zeros((ms, ms), dtype=np.uint8)
        ph, pw = min(h, ms), min(w, ms)
        padded[:ph, :pw] = grid[:ph, :pw]
        return padded

    def _grid_to_onehot(self, grid: np.ndarray) -> torch.Tensor:
        """Padded grid -> one-hot tensor (C, H, W)."""
        t = torch.tensor(grid, dtype=torch.long, device=self.device)
        t = t.clamp(0, self.cfg.num_classes - 1)
        oh = F.one_hot(t, self.cfg.num_classes)  # (H, W, C)
        return oh.permute(2, 0, 1).float()       # (C, H, W)

    def _raw_action_vec(
        self,
        action: int,
        action_data: Optional[Dict],
        grid_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Build the raw action feature vector (num_actions + 2)."""
        vec = torch.zeros(self.cfg.num_actions + 2, device=self.device)
        vec[min(action, self.cfg.num_actions - 1)] = 1.0  # one-hot
        h, w = grid_shape[:2]
        if action_data and isinstance(action_data, dict):
            vec[-2] = action_data.get("x", 0) / max(w, 1)
            vec[-1] = action_data.get("y", 0) / max(h, 1)
        return vec

    # ------------------------------------------------------------------
    # Stats / serialisation
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return visual cortex statistics."""
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "buffer_size": len(self._buffer),
            "trained_steps": self._trained_steps,
            "parameters": n_params,
            "grid_size": self._grid_size,
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "trained_steps": self._trained_steps,
            "cfg": self.cfg,
        }, path)
        logger.info(f"Visual cortex saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self._trained_steps = ckpt.get("trained_steps", 0)
        self.model.eval()
        logger.info(f"Visual cortex loaded from {path} (steps={self._trained_steps})")
