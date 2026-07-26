"""Compact bootstrap graph world model for SAGE.11."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dataset import MINIMUM_STRONG_TERMINAL_EVENTS
from .streaming_features import (
    CHANGED_BUCKETS,
    STREAMING_FEATURE_FORMAT_VERSION,
)


@dataclass(frozen=True)
class WorldModelConfig:
    vocab_size: int = 4096
    streaming_features: int = 77
    changed_cell_classes: int = len(CHANGED_BUCKETS)
    hidden_size: int = 192
    latent_size: int = 128
    message_layers: int = 3
    bootstrap_heads: int = 5
    max_parameters: int = 5_000_000
    streaming_feature_version: str = STREAMING_FEATURE_FORMAT_VERSION
    streaming_feature_schema_checksum: str = ""
    format_version: str = "sage11-world-model-v2"


class GraphAtomEncoder(nn.Module):
    """Permutation-invariant message encoder over typed symbolic atoms."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
        )
        self.messages = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.SiLU(),
                nn.LayerNorm(config.hidden_size),
            )
            for _ in range(config.message_layers)
        ])
        self.projection = nn.Linear(config.hidden_size, config.latent_size)

    def forward(self, atom_ids: Tensor, atom_mask: Tensor) -> Tensor:
        nodes = self.embedding(atom_ids)
        mask = atom_mask.to(nodes.dtype).unsqueeze(-1)
        for message in self.messages:
            pooled = (nodes * mask).sum(dim=1, keepdim=True)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            nodes = nodes + message(torch.cat(
                (nodes, pooled.expand_as(nodes)),
                dim=-1,
            ))
        pooled = (nodes * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.projection(pooled), dim=-1)


class BootstrapPredictionHead(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        joint = config.latent_size + config.streaming_features
        self.trunk = nn.Sequential(
            nn.Linear(joint, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
        )
        self.next_latent = nn.Linear(config.hidden_size, config.latent_size)
        self.changed_cells = nn.Linear(
            config.hidden_size,
            config.changed_cell_classes,
        )
        self.player_moved = nn.Linear(config.hidden_size, 1)
        self.progress = nn.Linear(config.hidden_size, 1)
        self.terminal = nn.Linear(config.hidden_size, 1)
        self.risk = nn.Linear(config.hidden_size, 1)
        self.noop = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        latent: Tensor,
        streaming_features: Tensor,
    ) -> Dict[str, Tensor]:
        hidden = self.trunk(torch.cat(
            (latent, streaming_features),
            dim=-1,
        ))
        return {
            "next_latent": F.normalize(self.next_latent(hidden), dim=-1),
            "changed_cells_logits": self.changed_cells(hidden),
            "player_moved_logit": self.player_moved(hidden).squeeze(-1),
            "progress_logit": self.progress(hidden).squeeze(-1),
            "terminal_logit": self.terminal(hidden).squeeze(-1),
            "risk_logit": self.risk(hidden).squeeze(-1),
            "noop_logit": self.noop(hidden).squeeze(-1),
        }


class Sage11GraphWorldModel(nn.Module):
    """Graph encoder plus five bootstrap dynamics/calibration heads."""

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        if (
            self.config.streaming_feature_version
            != STREAMING_FEATURE_FORMAT_VERSION
        ):
            raise ValueError("world model requires streaming feature v2")
        self.encoder = GraphAtomEncoder(self.config)
        self.heads = nn.ModuleList([
            BootstrapPredictionHead(self.config)
            for _ in range(self.config.bootstrap_heads)
        ])
        self.register_buffer(
            "_strong_terminal_events",
            torch.tensor(0, dtype=torch.long),
        )
        count = self.parameter_count
        if count >= self.config.max_parameters:
            raise ValueError(
                f"SAGE.11 model has {count:,} parameters; "
                f"limit is <{self.config.max_parameters:,}"
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def terminal_head_enabled(self) -> bool:
        return (
            int(self._strong_terminal_events.item())
            >= MINIMUM_STRONG_TERMINAL_EVENTS
        )

    def set_strong_terminal_events(self, count: int) -> None:
        self._strong_terminal_events.fill_(max(0, int(count)))

    def forward(
        self,
        atom_ids: Tensor,
        atom_mask: Tensor,
        streaming_features: Tensor,
    ) -> Dict[str, Any]:
        if (
            streaming_features.ndim != 2
            or streaming_features.shape[-1]
            != self.config.streaming_features
        ):
            raise ValueError(
                "world model streaming feature width does not match config"
            )
        latent = self.encoder(atom_ids, atom_mask)
        members = tuple(
            head(latent, streaming_features)
            for head in self.heads
        )
        aggregate: Dict[str, Tensor] = {"latent": latent}
        for key in members[0]:
            values = torch.stack([member[key] for member in members], dim=0)
            aggregate[key] = values.mean(dim=0)
            aggregate[f"{key}_variance"] = values.var(
                dim=0,
                unbiased=False,
            )
        if not self.terminal_head_enabled:
            aggregate["terminal_logit"] = torch.full_like(
                aggregate["terminal_logit"],
                -20.0,
            )
        aggregate["members"] = members
        return aggregate

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "format_version": self.config.format_version,
            "config": asdict(self.config),
            "parameter_count": self.parameter_count,
            "strong_terminal_events": int(
                self._strong_terminal_events.item()
            ),
            "terminal_head_enabled": self.terminal_head_enabled,
            "legacy_weights_loaded": False,
        }

    def load_legacy_state_dict(
        self,
        _state: Mapping[str, Tensor],
    ) -> None:
        raise ValueError("SAGE.11 forbids M2/v4 legacy weights")


__all__ = [
    "GraphAtomEncoder",
    "Sage11GraphWorldModel",
    "WorldModelConfig",
]
