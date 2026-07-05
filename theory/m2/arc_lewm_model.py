"""Tiny ARC-LeWM latent transition model for M2.14c."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn


ACTION_VOCAB = (
    "PAD",
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
    "ACTION8",
    "UNKNOWN",
)
ACTION_TO_ID = {action: index for index, action in enumerate(ACTION_VOCAB)}
DEFAULT_CANVAS_SIZE = (64, 64)


def action_to_id(action: str) -> int:
    return ACTION_TO_ID.get(str(action).upper(), ACTION_TO_ID["UNKNOWN"])


def encode_action_args(
    action_args: Mapping[str, Any] | None,
    *,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> torch.Tensor:
    args = dict(action_args or {})
    height, width = canvas_size
    x_present = "x" in args
    y_present = "y" in args
    x = _safe_float(args.get("x", 0.0))
    y = _safe_float(args.get("y", 0.0))
    x_den = max(float(width - 1), 1.0)
    y_den = max(float(height - 1), 1.0)
    return torch.tensor(
        [
            max(0.0, min(x / x_den, 1.0)),
            max(0.0, min(y / y_den, 1.0)),
            1.0 if x_present else 0.0,
            1.0 if y_present else 0.0,
        ],
        dtype=torch.float32,
    )


def pad_crop_grid(
    grid: Sequence[Sequence[int]],
    *,
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = canvas_size
    canvas = torch.zeros((height, width), dtype=torch.long)
    mask = torch.zeros((height, width), dtype=torch.float32)
    for y, row in enumerate(grid[:height]):
        for x, value in enumerate(row[:width]):
            canvas[y, x] = max(0, min(int(value), 15))
            mask[y, x] = 1.0
    return canvas, mask


class GridEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 128,
        num_colors: int = 16,
        color_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.color_embed = nn.Embedding(num_colors, color_embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(color_embed_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.project = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, grid: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if grid.dim() != 3:
            raise ValueError("grid must have shape [batch, height, width]")
        grid = grid.clamp(min=0, max=self.color_embed.num_embeddings - 1)
        embedded = self.color_embed(grid).permute(0, 3, 1, 2).contiguous()
        features = self.conv(embedded)
        if mask is None:
            pooled = features.mean(dim=(2, 3))
        else:
            if mask.dim() != 3:
                raise ValueError("mask must have shape [batch, height, width]")
            mask_4d = mask.unsqueeze(1).to(dtype=features.dtype, device=features.device)
            masked = features * mask_4d
            denom = mask_4d.sum(dim=(2, 3)).clamp_min(1.0)
            pooled = masked.sum(dim=(2, 3)) / denom
        return self.project(pooled)


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 128,
        action_embed_dim: int = 32,
        args_embed_dim: int = 32,
        use_hud: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_hud = use_hud
        self.action_embed = nn.Embedding(len(ACTION_VOCAB), action_embed_dim)
        self.args_embed = nn.Sequential(
            nn.Linear(4, args_embed_dim),
            nn.ReLU(),
        )
        hud_dim = 8 if use_hud else 0
        if use_hud:
            self.hud_embed = nn.Sequential(nn.Linear(1, hud_dim), nn.ReLU())
        else:
            self.hud_embed = None
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim + args_embed_dim + hud_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        action_ids: torch.Tensor,
        action_args: torch.Tensor | None = None,
        hud: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_ids.dim() != 1:
            action_ids = action_ids.view(-1)
        action_emb = self.action_embed(action_ids.to(device=z_t.device))
        if action_args is None:
            action_args = torch.zeros((z_t.shape[0], 4), dtype=z_t.dtype, device=z_t.device)
        else:
            action_args = action_args.to(dtype=z_t.dtype, device=z_t.device)
        args_emb = self.args_embed(action_args)
        features = [z_t, action_emb, args_emb]
        if self.use_hud:
            if hud is None:
                hud = torch.zeros((z_t.shape[0], 1), dtype=z_t.dtype, device=z_t.device)
            hud = hud.to(dtype=z_t.dtype, device=z_t.device)
            features.append(self.hud_embed(hud))
        return self.net(torch.cat(features, dim=1))


class ARCLeWMModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 128,
        use_hud: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_hud = use_hud
        self.encoder = GridEncoder(latent_dim=latent_dim)
        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            use_hud=use_hud,
        )

    def forward(
        self,
        grid_t: torch.Tensor,
        mask_t: torch.Tensor,
        action_ids: torch.Tensor,
        action_args: torch.Tensor | None = None,
        hud: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_t = self.encoder(grid_t, mask_t)
        z_hat_t1 = self.predictor(z_t, action_ids, action_args, hud)
        return z_hat_t1, z_t


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
