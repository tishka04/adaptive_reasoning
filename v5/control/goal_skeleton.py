"""Goal skeleton — a tiny, symbolic 2-level goal hint for V5.

Replaces V4_1's LLM goal decomposer with ~50 lines of deterministic
code. The skeleton is produced from the top ontology + current TP
estimate and broadcast to specialist minds. Minds are free to ignore
it, but the arbiter will mildly prefer proposals that align.
"""

from __future__ import annotations

from typing import Optional

from ..schemas import GameObservation
from ..schemas_ext import GoalSkeleton, OntologyHypothesis


_ONTOLOGY_TO_SUBGOAL = {
    "navigator": "navigate",
    "click":     "click",
    "token":     "exhaust_class",
    "transform": "trigger_transform",
    "physics":   "push",
}


def refresh_goal(
    obs: GameObservation,
    top_ontology: Optional[OntologyHypothesis],
    tp_estimate: float,
    prev_goal: Optional[GoalSkeleton] = None,
) -> GoalSkeleton:
    """Produce a fresh goal skeleton.

    Rules:
      - If TP ≥ 0.4 (strong terminal proximity) → subgoal = 'closure'
      - Else if TP ≥ 0.15                       → subgoal from ontology
      - Else                                     → subgoal = 'explore'
      - If there are no objects at all           → subgoal = 'explore'
    """
    ontology_kind = top_ontology.kind if top_ontology else "unknown"

    if not obs.objects:
        subgoal = "explore"
    elif tp_estimate >= 0.4:
        subgoal = "closure"
    elif tp_estimate >= 0.15:
        subgoal = _ONTOLOGY_TO_SUBGOAL.get(ontology_kind, "explore")
    else:
        subgoal = "explore"

    level = int(getattr(obs, "levels_completed", 0) or 0)
    return GoalSkeleton(
        goal=f"complete_level_{level + 1}",
        active_subgoal=subgoal,
        top_ontology=ontology_kind,
        tp_estimate=round(float(tp_estimate), 3),
        metadata={
            "n_objects": len(obs.objects),
            "prev_subgoal": prev_goal.active_subgoal if prev_goal else None,
        },
    )


def mind_bias_for_subgoal(subgoal: str) -> dict[str, float]:
    """Return small additive biases on mind confidence from a subgoal."""
    if subgoal == "navigate":
        return {"navigator": 0.10, "physics": 0.04}
    if subgoal == "click":
        return {"click": 0.10, "closure": 0.04}
    if subgoal == "exhaust_class":
        return {"click": 0.06, "closure": 0.10}
    if subgoal == "trigger_transform":
        return {"transform": 0.10}
    if subgoal == "push":
        return {"physics": 0.10, "navigator": 0.05}
    if subgoal == "closure":
        return {"closure": 0.15, "sequence": 0.05}
    # explore
    return {"sequence": 0.04, "transform": 0.02}
