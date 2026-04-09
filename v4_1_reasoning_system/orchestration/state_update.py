"""
State manager — maintains and updates the reasoning state across
the control loop.

Responsibilities:
  - Aggregate features from task, solution, feedback, and history
  - Call the state encoder to produce z_t
  - Track iteration count, budget consumption, and loop metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..world_model.encoder import StateEncoder


@dataclass
class ReasoningState:
    """Full reasoning state at time t."""
    iteration: int = 0
    z_t: Optional[torch.Tensor] = None

    # Raw features (before encoding)
    task_features: Optional[torch.Tensor] = None
    solution_features: Optional[torch.Tensor] = None
    feedback_features: Optional[torch.Tensor] = None
    history_features: Optional[torch.Tensor] = None

    # Metadata
    current_score: float = 0.0
    feasible: bool = False
    last_mode: Optional[str] = None
    last_success: bool = True
    total_elapsed: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Current solution and feedback
    current_solution: Optional[Any] = None
    current_feedback: Optional[Dict[str, Any]] = None

    @property
    def is_initial(self) -> bool:
        return self.iteration == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "current_score": self.current_score,
            "feasible": self.feasible,
            "last_mode": self.last_mode,
            "last_success": self.last_success,
            "total_elapsed": self.total_elapsed,
            "history_length": len(self.history),
        }


class StateManager:
    """
    Manages the reasoning state lifecycle:
      1. Initialize from a parsed task
      2. Encode features into latent z_t
      3. Update after each reasoning step
    """

    def __init__(
        self,
        encoder: StateEncoder,
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.device = device

    def initialize(self, task_dict: Dict[str, Any]) -> ReasoningState:
        """Create initial reasoning state from a parsed task."""
        state = ReasoningState()

        # Extract features
        state.task_features = StateEncoder.extract_task_features(task_dict).unsqueeze(0).to(self.device)
        state.solution_features = StateEncoder.extract_solution_features(None).unsqueeze(0).to(self.device)
        state.feedback_features = StateEncoder.extract_feedback_features(None).unsqueeze(0).to(self.device)
        state.history_features = StateEncoder.extract_history_features([]).unsqueeze(0).to(self.device)

        # Encode
        self.encoder.eval()
        with torch.no_grad():
            state.z_t = self.encoder(
                state.task_features,
                state.solution_features,
                state.feedback_features,
                state.history_features,
            )

        return state

    def update(
        self,
        state: ReasoningState,
        task_dict: Dict[str, Any],
        solver_result: Dict[str, Any],
        verifier_feedback: Dict[str, Any],
        mode: str,
        budget: str,
        success: bool,
        elapsed: float,
    ) -> ReasoningState:
        """Update reasoning state after a reasoning step."""
        state.iteration += 1
        state.last_mode = mode
        state.last_success = success
        state.total_elapsed += elapsed

        # Update score and feasibility
        prev_score = state.current_score
        state.current_score = verifier_feedback.get("score", solver_result.get("score", 0.0))
        state.feasible = verifier_feedback.get("feasible", solver_result.get("feasible", False))
        state.current_solution = solver_result.get("solution")
        state.current_feedback = verifier_feedback

        # Update history
        state.history.append({
            "mode": mode,
            "budget": budget,
            "success": success,
            "score_delta": state.current_score - prev_score,
        })

        # Re-extract features
        solution_dict = {
            "score": state.current_score,
            "feasible": int(state.feasible),
            "num_violations": len(verifier_feedback.get("violations", [])),
            "completeness": state.current_score,
            "iteration": state.iteration,
        }
        state.solution_features = StateEncoder.extract_solution_features(solution_dict).unsqueeze(0).to(self.device)

        feedback_dict = {
            "tests_passed": verifier_feedback.get("tests_passed", 0),
            "tests_total": verifier_feedback.get("tests_total", 0),
            "violation_severity": verifier_feedback.get("violation_severity", 0.0),
            "feasible": int(state.feasible),
            "objective_value": verifier_feedback.get("objective_value", 0.0),
        }
        state.feedback_features = StateEncoder.extract_feedback_features(feedback_dict).unsqueeze(0).to(self.device)
        state.history_features = StateEncoder.extract_history_features(state.history).unsqueeze(0).to(self.device)

        # Re-encode
        self.encoder.eval()
        with torch.no_grad():
            state.z_t = self.encoder(
                state.task_features,
                state.solution_features,
                state.feedback_features,
                state.history_features,
            )

        return state
