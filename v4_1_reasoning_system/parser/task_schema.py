"""
Structured task representation produced by the semantic parser.

Every problem the system encounters is normalized into a TaskObject
before any reasoning begins.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from adaptive_reasoning_compat.pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Domain taxonomy (extensible)
# ------------------------------------------------------------------
class DomainGuess(str, Enum):
    PLANNING = "planning"
    SCHEDULING = "scheduling"
    OPTIMIZATION = "optimization"
    CODING = "coding"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------
# Constraint representation
# ------------------------------------------------------------------
class ConstraintType(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class Constraint(BaseModel):
    name: str
    description: str
    ctype: ConstraintType = ConstraintType.HARD
    params: Dict[str, Any] = Field(default_factory=dict)

    def is_hard(self) -> bool:
        return self.ctype == ConstraintType.HARD


# ------------------------------------------------------------------
# Objective
# ------------------------------------------------------------------
class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    SATISFY = "satisfy"


class Objective(BaseModel):
    description: str
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    metric: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------
# Ambiguity markers
# ------------------------------------------------------------------
class AmbiguityMarker(BaseModel):
    field: str
    reason: str
    severity: float = 0.5  # 0 = trivial, 1 = blocking


# ------------------------------------------------------------------
# Core task object
# ------------------------------------------------------------------
class TaskObject(BaseModel):
    """Normalized representation of a user problem."""

    raw_input: str
    domain: DomainGuess = DomainGuess.UNKNOWN
    description: str = ""
    entities: List[str] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    objective: Optional[Objective] = None
    ambiguities: List[AmbiguityMarker] = Field(default_factory=list)
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    # ----- convenience helpers -----
    @property
    def hard_constraints(self) -> List[Constraint]:
        return [c for c in self.constraints if c.is_hard()]

    @property
    def soft_constraints(self) -> List[Constraint]:
        return [c for c in self.constraints if not c.is_hard()]

    @property
    def has_ambiguity(self) -> bool:
        return len(self.ambiguities) > 0

    def summary(self) -> str:
        parts = [
            f"Domain: {self.domain.value}",
            f"Entities: {len(self.entities)}",
            f"Hard constraints: {len(self.hard_constraints)}",
            f"Soft constraints: {len(self.soft_constraints)}",
            f"Objective: {self.objective.sense.value if self.objective else 'none'}",
            f"Ambiguities: {len(self.ambiguities)}",
        ]
        return " | ".join(parts)
