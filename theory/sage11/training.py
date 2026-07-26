"""SAGE.11 losses, calibration, and pre-registered promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor
from torch.nn import functional as F

from .model import Sage11GraphWorldModel
from .splits import SAGE11_SPLITS


@dataclass(frozen=True)
class LossWeights:
    jepa: float = 1.0
    changed_cells: float = 1.0
    player_moved: float = 1.0
    action_contrast: float = 0.25
    consistency: float = 0.25
    progress: float = 0.5
    terminal: float = 0.5
    risk: float = 0.5
    noop: float = 0.5


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    bootstrap_loss_weight: float = 0.25
    device: str = "cuda"


def world_model_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, Tensor],
    *,
    weights: LossWeights | None = None,
    terminal_head_enabled: bool,
) -> Dict[str, Tensor]:
    """Compute JEPA/effect/contrast/consistency/calibration objectives."""
    weight = weights or LossWeights()
    jepa = 1.0 - F.cosine_similarity(
        output["next_latent"],
        targets["next_latent"].detach(),
        dim=-1,
    ).mean()
    changed_cells = F.cross_entropy(
        output["changed_cells_logits"],
        targets["changed_cells"].long(),
    )
    player_moved = F.binary_cross_entropy_with_logits(
        output["player_moved_logit"],
        targets["player_moved"].float(),
    )
    progress_weights = torch.where(
        targets["progress_is_weak"].bool(),
        torch.full_like(targets["progress"].float(), 0.25),
        torch.ones_like(targets["progress"].float()),
    )
    progress_raw = F.binary_cross_entropy_with_logits(
        output["progress_logit"],
        targets["progress"].float(),
        reduction="none",
    )
    progress = (progress_raw * progress_weights).mean()
    risk = F.binary_cross_entropy_with_logits(
        output["risk_logit"],
        targets["risk"].float(),
    )
    noop = F.binary_cross_entropy_with_logits(
        output["noop_logit"],
        targets["noop"].float(),
    )
    terminal = torch.zeros_like(jepa)
    if terminal_head_enabled:
        terminal = F.binary_cross_entropy_with_logits(
            output["terminal_logit"],
            targets["terminal"].float(),
        )

    members = output["members"]
    member_latents = torch.stack(
        [member["next_latent"] for member in members],
        dim=0,
    )
    consistency = member_latents.var(dim=0, unbiased=False).mean()
    if "shuffled_next_latent" in targets:
        positive = F.cosine_similarity(
            output["next_latent"],
            targets["next_latent"].detach(),
            dim=-1,
        )
        negative = F.cosine_similarity(
            output["next_latent"],
            targets["shuffled_next_latent"].detach(),
            dim=-1,
        )
        action_contrast = F.relu(0.2 - positive + negative).mean()
    else:
        action_contrast = torch.zeros_like(jepa)
    total = (
        weight.jepa * jepa
        + weight.changed_cells * changed_cells
        + weight.player_moved * player_moved
        + weight.action_contrast * action_contrast
        + weight.consistency * consistency
        + weight.progress * progress
        + weight.terminal * terminal
        + weight.risk * risk
        + weight.noop * noop
    )
    return {
        "total": total,
        "jepa": jepa,
        "changed_cells": changed_cells,
        "player_moved": player_moved,
        "action_contrast": action_contrast,
        "consistency": consistency,
        "progress": progress,
        "terminal": terminal,
        "risk": risk,
        "noop": noop,
    }


class Sage11WorldModelTrainer:
    """Concrete source-only optimizer with bootstrap-resampled heads."""

    def __init__(
        self,
        model: Sage11GraphWorldModel,
        *,
        config: TrainerConfig | None = None,
        loss_weights: LossWeights | None = None,
        dataset_manifest_checksum: str = "",
        streaming_feature_schema_checksum: str = "",
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self.loss_weights = loss_weights or LossWeights()
        self.dataset_manifest_checksum = str(dataset_manifest_checksum)
        self.streaming_feature_schema_checksum = str(
            streaming_feature_schema_checksum
        )
        configured_schema = (
            self.model.config.streaming_feature_schema_checksum
        )
        if (
            configured_schema
            and self.streaming_feature_schema_checksum
            and configured_schema
            != self.streaming_feature_schema_checksum
        ):
            raise ValueError(
                "trainer and model streaming schemas do not match"
            )
        requested = torch.device(self.config.device)
        self.device = (
            requested
            if requested.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.steps = 0

    def train_step(
        self,
        batch: Mapping[str, Tensor],
    ) -> Dict[str, float]:
        """Run one update; callers must supply firewall-approved source data."""
        self.model.train()
        tensors = {
            key: value.to(self.device)
            for key, value in batch.items()
        }
        with torch.no_grad():
            next_latent = self.model.encoder(
                tensors["next_atom_ids"],
                tensors["next_atom_mask"],
            )
            shuffled = torch.roll(next_latent, shifts=1, dims=0)
        output = self.model(
            tensors["atom_ids"],
            tensors["atom_mask"],
            tensors["streaming_features"],
        )
        targets = {
            "next_latent": next_latent,
            "shuffled_next_latent": shuffled,
            "changed_cells": tensors["changed_cells"],
            "player_moved": tensors["player_moved"],
            "progress": tensors["progress"],
            "progress_is_weak": tensors["progress_is_weak"],
            "terminal": tensors["terminal"],
            "risk": tensors["risk"],
            "noop": tensors["noop"],
        }
        losses = world_model_loss(
            output,
            targets,
            weights=self.loss_weights,
            terminal_head_enabled=self.model.terminal_head_enabled,
        )
        bootstrap = self._bootstrap_loss(output["members"], targets)
        total = (
            losses["total"]
            + self.config.bootstrap_loss_weight * bootstrap
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        self.steps += 1
        return {
            **{
                name: float(value.detach().cpu().item())
                for name, value in losses.items()
            },
            "bootstrap": float(bootstrap.detach().cpu().item()),
            "optimized_total": float(total.detach().cpu().item()),
            "step": float(self.steps),
        }

    def checkpoint(self) -> Dict[str, Any]:
        return {
            "metadata": {
                **self.model.checkpoint_metadata(),
                "trainer": asdict(self.config),
                "steps": self.steps,
                "split_registry_checksum": SAGE11_SPLITS.checksum,
                "dataset_manifest_checksum": (
                    self.dataset_manifest_checksum
                ),
                "streaming_feature_schema_checksum": (
                    self.streaming_feature_schema_checksum
                    or self.model.config.streaming_feature_schema_checksum
                ),
            },
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), target)

    def _bootstrap_loss(
        self,
        members: Sequence[Mapping[str, Tensor]],
        targets: Mapping[str, Tensor],
    ) -> Tensor:
        batch_size = int(targets["changed_cells"].shape[0])
        losses = []
        for index, member in enumerate(members):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(11_000 + self.steps * 17 + index)
            mask = torch.rand(
                batch_size,
                generator=generator,
                device=self.device,
            ) < 0.632
            if not mask.any():
                mask[index % batch_size] = True
            member_loss = (
                F.cross_entropy(
                    member["changed_cells_logits"][mask],
                    targets["changed_cells"][mask].long(),
                )
                + F.binary_cross_entropy_with_logits(
                    member["player_moved_logit"][mask],
                    targets["player_moved"][mask].float(),
                )
                + F.binary_cross_entropy_with_logits(
                    member["risk_logit"][mask],
                    targets["risk"][mask].float(),
                )
                + F.binary_cross_entropy_with_logits(
                    member["noop_logit"][mask],
                    targets["noop"][mask].float(),
                )
            )
            losses.append(member_loss)
        return torch.stack(losses).mean()


@dataclass(frozen=True)
class WorldModelGateMetrics:
    """All source-only metrics required before shadow-mode integration."""

    changed_transition_accuracy: float
    persistence_changed_accuracy: float
    action_shuffle_degradation: float
    changed_cells_macro_f1: float
    changed_cells_majority_macro_f1: float
    player_moved_macro_f1: float
    player_moved_majority_macro_f1: float
    risk_ece: float
    noop_ece: float
    latent_feature_std: float
    validation_games: tuple[str, ...]


@dataclass(frozen=True)
class WorldModelGateReport:
    passed: bool
    gates: Mapping[str, bool]
    metrics: WorldModelGateMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": dict(self.gates),
            "metrics": asdict(self.metrics),
        }


def evaluate_world_model_gates(
    metrics: WorldModelGateMetrics,
) -> WorldModelGateReport:
    """Apply the amended change-weighted, causal, and calibration gates."""
    changed_accuracy_gain = (
        metrics.changed_transition_accuracy
        - metrics.persistence_changed_accuracy
    )
    gates = {
        "change_weighted_persistence_beat": (
            metrics.changed_transition_accuracy
            > metrics.persistence_changed_accuracy
        ),
        "changed_transition_gain_at_least_15pct": (
            changed_accuracy_gain >= 0.15
        ),
        "action_shuffle_degradation_at_least_10pct": (
            metrics.action_shuffle_degradation >= 0.10
        ),
        "changed_cells_macro_f1_majority_plus_0_10": (
            metrics.changed_cells_macro_f1
            >= metrics.changed_cells_majority_macro_f1 + 0.10
        ),
        "player_moved_macro_f1_majority_plus_0_10": (
            metrics.player_moved_macro_f1
            >= metrics.player_moved_majority_macro_f1 + 0.10
        ),
        "risk_ece_at_most_0_10": metrics.risk_ece <= 0.10,
        "noop_ece_at_most_0_10": metrics.noop_ece <= 0.10,
        "no_latent_collapse": metrics.latent_feature_std >= 0.01,
        "source_validation_only": set(metrics.validation_games).issubset({
            "re86",
            "ls20",
            "sc25",
        }),
    }
    return WorldModelGateReport(
        passed=all(gates.values()),
        gates=gates,
        metrics=metrics,
    )


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    bins: int = 10,
) -> float:
    probabilities_array = np.asarray(probabilities, dtype=float)
    labels_array = np.asarray(labels, dtype=float)
    if probabilities_array.shape != labels_array.shape:
        raise ValueError("probabilities and labels must have equal shape")
    if probabilities_array.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    total = float(probabilities_array.size)
    error = 0.0
    for index in range(len(edges) - 1):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (probabilities_array >= lower)
            & (
                probabilities_array <= upper
                if index == len(edges) - 2
                else probabilities_array < upper
            )
        )
        if not mask.any():
            continue
        confidence = probabilities_array[mask].mean()
        accuracy = labels_array[mask].mean()
        error += mask.sum() / total * abs(confidence - accuracy)
    return float(error)


def effect_macro_f1(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> float:
    return float(
        f1_score(labels, predictions, average="macro", zero_division=0)
    )


__all__ = [
    "LossWeights",
    "Sage11WorldModelTrainer",
    "TrainerConfig",
    "WorldModelGateMetrics",
    "WorldModelGateReport",
    "effect_macro_f1",
    "evaluate_world_model_gates",
    "expected_calibration_error",
    "world_model_loss",
]
