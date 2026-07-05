"""V5 schema extensions — types ported from V4 that complement V3's schemas.

V3 schemas use classes with behaviour (Predicate.check, Effect.matches).
These are V4-style value objects: ontologies, surprise, rituals, dissent.
They are stored separately to avoid polluting v5/schemas.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schemas import PrimitiveAction


# ---------------------------------------------------------------------
# Surprise field (ported from V4)
# ---------------------------------------------------------------------

@dataclass
class SurpriseField:
    """Five-channel surprise signal used by the progress tracker."""

    pixel_surprise: float = 0.0
    object_surprise: float = 0.0
    causal_surprise: float = 0.0
    topology_surprise: float = 0.0
    semantic_surprise: float = 0.0
    salient_cells: List[tuple[int, int]] = field(default_factory=list)

    @property
    def total(self) -> float:
        return min(
            1.0,
            (
                self.pixel_surprise
                + self.object_surprise
                + self.causal_surprise
                + self.topology_surprise
                + self.semantic_surprise
            )
            / 5.0,
        )


# ---------------------------------------------------------------------
# Ontology hypotheses (ported from V4, trimmed)
# ---------------------------------------------------------------------

@dataclass
class OntologyHypothesis:
    """A hypothesis about what kind of game this is."""

    ontology_id: str
    kind: str                                 # navigator / click / token / physics
    confidence: float = 0.5
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    salient_object_ids: List[int] = field(default_factory=list)
    active_affordance_biases: Dict[str, float] = field(default_factory=dict)
    terminal_hypotheses: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Rituals (compact prefix + terminal signature templates)
# ---------------------------------------------------------------------

@dataclass
class Ritual:
    """A compiled winning prefix paired with its terminal signature."""

    ritual_id: str
    ontology_kind: str
    prefix: List[PrimitiveAction]
    terminal_signature: Dict[str, Any]
    success_rate: float = 0.0
    survival_score: float = 0.0


# ---------------------------------------------------------------------
# Goal skeleton (V5-native, replaces V4_1 LLM decomposer)
# ---------------------------------------------------------------------

@dataclass
class GoalSkeleton:
    """Tiny 2-level symbolic goal hint derived from ontology and TP.

    Not a constraint — just a hint passed to minds so they prefer
    compatible operators. Refreshed once per action.
    """

    goal: str = "complete_level"
    active_subgoal: str = "explore"          # one of: explore, navigate, click, transform, push, closure
    top_ontology: str = "unknown"
    tp_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Dissent report (ported from V4)
# ---------------------------------------------------------------------

@dataclass
class DissentReport:
    false_progress: Dict[str, float] = field(default_factory=dict)
    ontology_warnings: List[str] = field(default_factory=list)
    loop_warning: bool = False
    suggested_actions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Redirect intent (emitted by DissentController when it interrupts)
# ---------------------------------------------------------------------

@dataclass
class RedirectIntent:
    reason: str
    primitive: Optional[PrimitiveAction] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
