"""
State encoder — maps the current reasoning context into a fixed-size
latent vector z_t.

Inputs aggregated:
  - parsed task features (domain, # constraints, objective sense, …)
  - partial solution features (current score, feasibility flags, …)
  - verifier feedback (pass/fail counts, violation severity, …)
  - recent reasoning history (last-K action embeddings)

Architecture: lightweight MLP/Transformer, 5-20M parameters.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """Encodes heterogeneous reasoning context into latent z_t."""

    def __init__(
        self,
        task_dim: int = 32,
        solution_dim: int = 32,
        feedback_dim: int = 16,
        history_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Per-source projections
        self.task_proj = nn.Linear(task_dim, hidden_dim)
        self.solution_proj = nn.Linear(solution_dim, hidden_dim)
        self.feedback_proj = nn.Linear(feedback_dim, hidden_dim)
        self.history_proj = nn.Linear(history_dim, hidden_dim)

        # Fusion MLP
        layers = []
        in_dim = hidden_dim * 4
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.fusion = nn.Sequential(*layers)

    def forward(
        self,
        task_features: torch.Tensor,
        solution_features: torch.Tensor,
        feedback_features: torch.Tensor,
        history_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        All inputs: (batch, *_dim)  →  output: (batch, latent_dim)
        """
        t = self.task_proj(task_features)
        s = self.solution_proj(solution_features)
        f = self.feedback_proj(feedback_features)
        h = self.history_proj(history_features)
        combined = torch.cat([t, s, f, h], dim=-1)
        z = self.fusion(combined)
        return z

    # ------------------------------------------------------------------
    # Feature extraction helpers (called before forward)
    # ------------------------------------------------------------------
    @staticmethod
    def extract_task_features(task_dict: Dict, dim: int = 32) -> torch.Tensor:
        """Convert a TaskObject-like dict to a float tensor."""
        domain_map = {
            "planning": 0, "scheduling": 1,
            "optimization": 2, "coding": 3, "unknown": 4,
        }
        vec = torch.zeros(dim)
        vec[0] = domain_map.get(task_dict.get("domain", "unknown"), 4)
        vec[1] = len(task_dict.get("constraints", []))
        vec[2] = len(task_dict.get("entities", []))
        vec[3] = len(task_dict.get("ambiguities", []))
        sense_map = {"minimize": -1.0, "maximize": 1.0, "satisfy": 0.0}
        obj = task_dict.get("objective")
        if obj:
            vec[4] = sense_map.get(obj.get("sense", "satisfy"), 0.0)
        return vec

    @staticmethod
    def extract_solution_features(solution_dict: Optional[Dict], dim: int = 32) -> torch.Tensor:
        """Convert partial solution state to a float tensor."""
        vec = torch.zeros(dim)
        if solution_dict is None:
            return vec
        vec[0] = float(solution_dict.get("score", 0.0))
        vec[1] = float(solution_dict.get("feasible", 0))
        vec[2] = float(solution_dict.get("num_violations", 0))
        vec[3] = float(solution_dict.get("completeness", 0.0))
        vec[4] = float(solution_dict.get("iteration", 0))
        return vec

    @staticmethod
    def extract_feedback_features(feedback_dict: Optional[Dict], dim: int = 16) -> torch.Tensor:
        """Convert verifier feedback to a float tensor."""
        vec = torch.zeros(dim)
        if feedback_dict is None:
            return vec
        vec[0] = float(feedback_dict.get("tests_passed", 0))
        vec[1] = float(feedback_dict.get("tests_total", 0))
        vec[2] = float(feedback_dict.get("violation_severity", 0.0))
        vec[3] = float(feedback_dict.get("feasible", 0))
        vec[4] = float(feedback_dict.get("objective_value", 0.0))
        return vec

    @staticmethod
    def extract_history_features(
        history: list,
        action_dim: int = 16,
        max_k: int = 4,
        dim: int = 64,
    ) -> torch.Tensor:
        """Flatten last-K reasoning action embeddings into a vector."""
        vec = torch.zeros(dim)
        mode_map = {
            "hierarchical": 0, "global_opt": 1,
            "repair": 2, "llm_codegen": 3,
        }
        for i, entry in enumerate(history[-max_k:]):
            offset = i * action_dim
            if offset + action_dim > dim:
                break
            mode = entry.get("mode", "hierarchical")
            budget_map = {"low": 0.25, "medium": 0.5, "high": 0.75}
            raw_budget = entry.get("budget", 0.5)
            vec[offset] = mode_map.get(mode, 0)
            vec[offset + 1] = budget_map.get(raw_budget, raw_budget) if isinstance(raw_budget, str) else float(raw_budget)
            vec[offset + 2] = float(entry.get("success", 0))
            vec[offset + 3] = float(entry.get("score_delta", 0.0))
        return vec
