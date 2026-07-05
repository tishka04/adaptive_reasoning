"""Losses and collapse diagnostics for M2.14c ARC-LeWM."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def prediction_mse(z_hat_t1: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
    if z_hat_t1.shape != z_t1.shape:
        raise ValueError(
            f"latent shape mismatch: predicted {tuple(z_hat_t1.shape)} vs target {tuple(z_t1.shape)}"
        )
    return F.mse_loss(z_hat_t1, z_t1)


def sigreg_loss(z_batch: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if z_batch.dim() != 2:
        raise ValueError("SIGReg expects latent batch with shape [batch, dim]")
    if z_batch.shape[0] < 2:
        return z_batch.pow(2).mean()
    centered = z_batch - z_batch.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(eps)
    std_loss = (std - 1.0).pow(2).mean()
    covariance = centered.T @ centered / max(z_batch.shape[0] - 1, 1)
    covariance = covariance / (std[:, None] * std[None, :]).clamp_min(eps)
    off_diag = covariance - torch.diag(torch.diagonal(covariance))
    cov_loss = off_diag.pow(2).mean()
    mean_loss = z_batch.mean(dim=0).pow(2).mean()
    return std_loss + cov_loss + 0.1 * mean_loss


def collapse_diagnostics(
    z_batch: torch.Tensor,
    *,
    min_std: float = 1e-4,
    min_rank: int = 2,
) -> Dict[str, Any]:
    with torch.no_grad():
        finite = torch.isfinite(z_batch).all().item()
        if z_batch.numel() == 0:
            return {
                "latent_std_mean": 0.0,
                "latent_std_min": 0.0,
                "latent_cov_rank_proxy": 0,
                "latent_norm_mean": 0.0,
                "collapse_detected": True,
                "latent_variance_above_min": False,
                "nan_or_inf_detected": True,
            }
        if not finite:
            finite_values = z_batch[torch.isfinite(z_batch)]
            norm_mean = (
                float(finite_values.abs().mean().item())
                if finite_values.numel()
                else 0.0
            )
            return {
                "latent_std_mean": 0.0,
                "latent_std_min": 0.0,
                "latent_cov_rank_proxy": 0,
                "latent_norm_mean": norm_mean,
                "collapse_detected": True,
                "latent_variance_above_min": False,
                "nan_or_inf_detected": True,
            }
        std = z_batch.std(dim=0, unbiased=False)
        centered = z_batch - z_batch.mean(dim=0, keepdim=True)
        if z_batch.shape[0] >= 2:
            singular_values = torch.linalg.svdvals(centered.float())
            rank_proxy = int((singular_values > min_std).sum().item())
        else:
            rank_proxy = 1 if float(std.max().item()) > min_std else 0
        std_mean = float(std.mean().item())
        std_min = float(std.min().item())
        latent_variance_above_min = bool(std_mean > min_std and std_min >= 0.0)
        collapse_detected = (
            not finite
            or std_mean <= min_std
            or rank_proxy < min(min_rank, z_batch.shape[-1])
        )
        return {
            "latent_std_mean": std_mean,
            "latent_std_min": std_min,
            "latent_cov_rank_proxy": rank_proxy,
            "latent_norm_mean": float(z_batch.norm(dim=1).mean().item()),
            "collapse_detected": bool(collapse_detected),
            "latent_variance_above_min": latent_variance_above_min,
            "nan_or_inf_detected": not bool(finite),
        }
