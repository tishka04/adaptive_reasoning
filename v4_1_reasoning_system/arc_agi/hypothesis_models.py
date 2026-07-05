"""Small data models for hypothesis-driven trajectory sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActionHypothesis:
    """One local belief about what an action does in the current game."""

    action_name: str
    kind: str
    confidence: float
    support: int = 0
    information_gain: float = 0.0
    risk: float = 0.0
    displacement: Optional[Tuple[float, float]] = None
    predicted_signal: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_probe(self) -> bool:
        return self.kind in {"unknown", "probe"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_name,
            "kind": self.kind,
            "confidence": round(float(self.confidence), 4),
            "support": int(self.support),
            "information_gain": round(float(self.information_gain), 4),
            "risk": round(float(self.risk), 4),
            "displacement": self.displacement,
            "predicted_signal": self.predicted_signal,
            "evidence": dict(self.evidence),
        }


@dataclass
class BeliefState:
    """Current set of action-effect hypotheses."""

    hypotheses: List[ActionHypothesis] = field(default_factory=list)

    def by_action(self) -> Dict[str, ActionHypothesis]:
        return {hyp.action_name: hyp for hyp in self.hypotheses}

    def top(self, limit: int = 5) -> List[ActionHypothesis]:
        return sorted(
            self.hypotheses,
            key=lambda hyp: (
                -float(hyp.confidence),
                -float(hyp.information_gain),
                float(hyp.risk),
                hyp.action_name,
            ),
        )[: max(0, limit)]

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for hyp in self.hypotheses:
            counts[hyp.kind] = counts.get(hyp.kind, 0) + 1
        return {
            "counts": counts,
            "top": [hyp.to_dict() for hyp in self.top(3)],
        }


@dataclass
class HypothesisTrajectory:
    """A short sampled future anchored in one lead action hypothesis."""

    actions: List[str]
    lead_hypothesis: ActionHypothesis
    support_hypotheses: List[ActionHypothesis] = field(default_factory=list)
    goal_alignment: float = 0.0

    def metadata(self, belief: BeliefState) -> Dict[str, Any]:
        return {
            "hypothesis_kind": self.lead_hypothesis.kind,
            "hypothesis_confidence": float(self.lead_hypothesis.confidence),
            "hypothesis_support": int(self.lead_hypothesis.support),
            "hypothesis_information_gain": float(self.lead_hypothesis.information_gain),
            "hypothesis_risk": float(self.lead_hypothesis.risk),
            "hypothesis_predicted_signal": self.lead_hypothesis.predicted_signal,
            "hypothesis_goal_alignment": float(self.goal_alignment),
            "hypothesis_action": self.lead_hypothesis.action_name,
            "hypothesis_belief_summary": belief.summary(),
            "support_hypotheses": [hyp.to_dict() for hyp in self.support_hypotheses[:3]],
        }
