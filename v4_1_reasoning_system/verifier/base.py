"""
Base verifier interface and common result type.

The verifier is the real source of truth in the architecture.
It checks correctness, feasibility, constraint satisfaction,
objective quality, runtime behavior, and safety conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    """Uniform output from any verifier."""
    valid: bool
    feasible: bool = True
    score: float = 0.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    tests_passed: int = 0
    tests_total: int = 0
    objective_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    @property
    def violation_severity(self) -> float:
        if not self.violations:
            return 0.0
        return max(v.get("severity", 0.5) for v in self.violations)

    @property
    def pass_rate(self) -> float:
        if self.tests_total == 0:
            return 1.0
        return self.tests_passed / self.tests_total

    def to_feedback_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for the state encoder."""
        return {
            "valid": self.valid,
            "feasible": self.feasible,
            "score": self.score,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "violation_severity": self.violation_severity,
            "objective_value": self.objective_value,
            "violations": [
                {"name": v.get("name", "unnamed"), "description": v.get("description", "")}
                for v in self.violations
            ],
        }


class BaseVerifier(ABC):
    """Abstract base for all verifiers."""

    name: str = "base"

    @abstractmethod
    def verify(
        self,
        task: Dict[str, Any],
        solution: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify a solution against a task specification.

        Args:
            task: parsed task dict
            solution: solver output
            context: optional additional context

        Returns:
            VerificationResult
        """
        ...

    def check_constraints(
        self, task: Dict[str, Any], solution: Any
    ) -> List[Dict[str, Any]]:
        """Check all constraints and return list of violations."""
        violations = []
        for c in task.get("constraints", []):
            ok = self._check_single_constraint(c, solution, task)
            if not ok:
                violations.append({
                    "name": c.get("name", "unnamed"),
                    "description": c.get("description", ""),
                    "ctype": c.get("ctype", "hard"),
                    "severity": 1.0 if c.get("ctype") == "hard" else 0.5,
                })
        return violations

    def _check_single_constraint(
        self, constraint: Dict, solution: Any, task: Dict
    ) -> bool:
        """Override in subclasses for domain-specific constraint checking."""
        return True
