"""
Base solver interface and common result type.

Every solver in the portfolio implements this interface so the
orchestration controller can invoke them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SolverResult:
    """Uniform output from any solver."""
    success: bool
    solution: Any = None
    score: float = 0.0
    feasible: bool = False
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def num_violations(self) -> int:
        return len(self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "solution": self.solution,
            "score": self.score,
            "feasible": self.feasible,
            "violations": self.violations,
            "metadata": self.metadata,
            "logs": self.logs,
            "elapsed_seconds": self.elapsed_seconds,
        }


class BaseSolver(ABC):
    """Abstract base for all solvers in the portfolio."""

    name: str = "base"

    @abstractmethod
    def solve(
        self,
        task: Dict[str, Any],
        budget: str = "medium",
        strictness: str = "verified",
        tool_hint: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> SolverResult:
        """
        Execute the solver on a task.

        Args:
            task: parsed task as dict (from TaskObject.model_dump())
            budget: low | medium | high
            strictness: fast | verified | strict
            tool_hint: optional solver-specific hint
            context: previous solution, verifier feedback, etc.

        Returns:
            SolverResult
        """
        ...

    def budget_to_seconds(self, budget: str) -> float:
        """Convert budget label to approximate time limit."""
        return {"low": 5.0, "medium": 30.0, "high": 120.0}.get(budget, 30.0)

    def budget_to_iterations(self, budget: str) -> int:
        """Convert budget label to iteration limit."""
        return {"low": 100, "medium": 1000, "high": 10000}.get(budget, 1000)
