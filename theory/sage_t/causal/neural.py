"""Shared graph-masked neural mechanisms for SAGE.T12."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from theory.m2.arc_lewm_model import GridEncoder

from .contracts import GroundedAction, ValueDistribution, encode_value


class CausalStateEncoder(nn.Module):
    """ARC-LeWM visual encoder reused as an observation encoder, not executor."""

    def __init__(self, *, latent_dim: int = 64) -> None:
        super().__init__()
        self.grid_encoder = GridEncoder(latent_dim=latent_dim)

    def forward(
        self, grid: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.grid_encoder(grid, mask)


class GraphMaskedMechanismHead(nn.Module):
    """A small head that receives only its declared-parent vector."""

    def __init__(
        self,
        *,
        parent_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.parent_dim = int(parent_dim)
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(self.parent_dim + self.action_dim + self.context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(
        self,
        parents: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if parents.shape[-1] != self.parent_dim:
            raise ValueError("parent tensor violates the declared causal mask")
        if actions.shape[-1] != self.action_dim:
            raise ValueError("action tensor has the wrong dimension")
        features = [parents, actions]
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError("context tensor has the wrong dimension")
            features.append(context)
        elif context is not None and context.shape[-1] != 0:
            raise ValueError("this mechanism does not declare a context input")
        return self.network(torch.cat(features, dim=-1))


class SharedMechanismBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()

    def register_head(self, module_id: str, head: GraphMaskedMechanismHead) -> None:
        if module_id in self.heads:
            raise ValueError(f"mechanism head already registered: {module_id}")
        self.heads[module_id] = head

    def forward(
        self,
        module_id: str,
        parents: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if module_id not in self.heads:
            raise KeyError(f"unknown shared mechanism head: {module_id}")
        return self.heads[module_id](parents, actions, context)


class ObservationLikelihoodHead(nn.Module):
    """Calibrated compatibility logit between predicted and observed deltas."""

    def __init__(self, *, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, predicted_features: torch.Tensor, observed_features: torch.Tensor
    ) -> torch.Tensor:
        if predicted_features.shape[-1] != self.feature_dim or observed_features.shape[-1] != self.feature_dim:
            raise ValueError("observation-likelihood features have the wrong dimension")
        return self.network(torch.cat((predicted_features, observed_features), dim=-1)).squeeze(-1)


class TorchObservationLikelihood:
    """Inference adapter; the feature builder is observation-only and auditable."""

    def __init__(
        self,
        *,
        head: ObservationLikelihoodHead,
        feature_builder: Any,
        device: str = "cpu",
        minimum_log_likelihood: float = -20.0,
    ) -> None:
        self.head = head
        self.feature_builder = feature_builder
        self.device = torch.device(device)
        self.minimum_log_likelihood = float(minimum_log_likelihood)

    def log_likelihood(self, prediction: Any, evidence: Any) -> float:
        predicted, observed = self.feature_builder(prediction, evidence)
        predicted_tensor = torch.as_tensor(predicted, dtype=torch.float32, device=self.device).reshape(1, -1)
        observed_tensor = torch.as_tensor(observed, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.head.eval()
        with torch.no_grad():
            log_probability = F.logsigmoid(
                self.head(predicted_tensor, observed_tensor)
            )[0]
        return max(self.minimum_log_likelihood, float(log_probability.cpu().item()))


class TorchCategoricalMechanism:
    """Inference adapter from a shared torch head to ValueDistribution."""

    def __init__(
        self,
        *,
        bank: SharedMechanismBank,
        module_id: str,
        parent_vocabulary: Sequence[Any],
        action_vocabulary: Sequence[str],
        output_vocabulary: Sequence[Any],
        device: str = "cpu",
    ) -> None:
        self.bank = bank
        self.module_id = str(module_id)
        self.parent_vocabulary = tuple(encode_value(item) for item in parent_vocabulary)
        self.action_vocabulary = tuple(str(item) for item in action_vocabulary)
        self.output_vocabulary = tuple(output_vocabulary)
        self.device = torch.device(device)
        if self.module_id not in self.bank.heads:
            raise ValueError(f"unregistered mechanism head: {self.module_id}")

    def predict(
        self,
        parents: Sequence[ValueDistribution],
        action: GroundedAction,
        parameters: Mapping[str, Any],
    ) -> ValueDistribution:
        parent_vector = []
        for parent in parents:
            parent_vector.extend(
                float(parent.probabilities.get(value, 0.0))
                for value in self.parent_vocabulary
            )
        action_vector = [
            float(action.action_name == action_name)
            for action_name in self.action_vocabulary
        ]
        head = self.bank.heads[self.module_id]
        if len(parent_vector) != head.parent_dim:
            raise ValueError("encoded parents do not match the shared head mask")
        if len(action_vector) != head.action_dim:
            raise ValueError("encoded action does not match the shared head")
        self.bank.eval()
        with torch.no_grad():
            logits = self.bank(
                self.module_id,
                torch.tensor([parent_vector], dtype=torch.float32, device=self.device),
                torch.tensor([action_vector], dtype=torch.float32, device=self.device),
            )
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return ValueDistribution(
            {
                encode_value(value): float(probability)
                for value, probability in zip(self.output_vocabulary, probabilities)
            }
        )


@dataclass(frozen=True)
class CausalLossBreakdown:
    total: torch.Tensor
    transition: torch.Tensor
    branch: torch.Tensor
    invariance: torch.Tensor
    sparsity: torch.Tensor
    calibration: torch.Tensor


def causal_mechanism_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    branch_logits: torch.Tensor | None = None,
    branch_targets: torch.Tensor | None = None,
    invariant_pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
    parent_gate: torch.Tensor | None = None,
    lambda_branch: float = 1.0,
    lambda_invariance: float = 0.1,
    lambda_sparsity: float = 0.01,
    lambda_calibration: float = 0.1,
) -> CausalLossBreakdown:
    transition = F.cross_entropy(logits, targets)
    branch = logits.sum() * 0.0
    if branch_logits is not None and branch_targets is not None:
        branch = F.cross_entropy(branch_logits, branch_targets)
    invariance = logits.sum() * 0.0
    if invariant_pairs is not None:
        left, right = invariant_pairs
        invariance = F.mse_loss(torch.softmax(left, -1), torch.softmax(right, -1))
    sparsity = logits.sum() * 0.0 if parent_gate is None else parent_gate.abs().mean()
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(probabilities.dtype)
    calibration = torch.mean(torch.sum((probabilities - one_hot) ** 2, dim=-1))
    total = (
        transition
        + float(lambda_branch) * branch
        + float(lambda_invariance) * invariance
        + float(lambda_sparsity) * sparsity
        + float(lambda_calibration) * calibration
    )
    return CausalLossBreakdown(total, transition, branch, invariance, sparsity, calibration)


def module_content_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "CausalLossBreakdown", "CausalStateEncoder", "GraphMaskedMechanismHead",
    "ObservationLikelihoodHead", "SharedMechanismBank", "TorchCategoricalMechanism",
    "TorchObservationLikelihood", "causal_mechanism_loss",
    "module_content_hash",
]
