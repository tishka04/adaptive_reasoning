"""Goal-family recognition for trajectory-conditioned planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .game_memory import GameMemory
from .goal_decomposer import GameGoal, SubGoal
from .state_describer import GameObservation


@dataclass
class GoalHypothesis:
    """Shared semantic target used by the sampler, memory, and scorer."""

    family: str
    confidence: float
    target_objects: List[Dict[str, Any]] = field(default_factory=list)
    relevant_colors: List[str] = field(default_factory=list)
    possible_player: Optional[Dict[str, Any]] = None
    source: str = "heuristic"
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Render a compact JSON-friendly representation."""
        return {
            "family": self.family,
            "confidence": float(self.confidence),
            "target_objects": list(self.target_objects),
            "relevant_colors": list(self.relevant_colors),
            "possible_player": dict(self.possible_player or {}),
            "source": self.source,
            "evidence": dict(self.evidence),
        }


class GoalRecognizer:
    """Best-effort goal recognizer for ARC-AGI-3 trajectory planning."""

    def predict(
        self,
        observation: GameObservation,
        *,
        memory: Optional[GameMemory] = None,
        current_goal: Optional[GameGoal] = None,
        current_subgoal: Optional[SubGoal] = None,
        task_program: Optional[Any] = None,
    ) -> GoalHypothesis:
        """Infer the current goal family and salient target semantics."""
        metadata = dict(getattr(current_subgoal, "metadata", {}) or {})
        evidence: Dict[str, Any] = {}

        family = self._from_task_program(task_program)
        if family is not None:
            source = "task_program"
            confidence = 0.92
            evidence["task_program_goal_family"] = family
        else:
            family = self._from_goal(current_goal)
            if family is not None:
                source = "goal_bank"
                confidence = max(0.55, float(getattr(current_goal, "confidence", 0.6)))
                evidence["goal_bank_family"] = family
            else:
                family = self._from_memory(memory)
                if family is not None:
                    source = "human_prior"
                    confidence = 0.74
                    evidence["memory_goal_family"] = family
                else:
                    family = self._from_observation(observation)
                    source = "observation"
                    confidence = 0.58 if family != "unknown" else 0.35
                    evidence["observation_goal_family"] = family

        relevant_colors = self._relevant_colors(observation, metadata)
        target_objects = self._target_objects(observation, metadata, relevant_colors)
        possible_player = dict(observation.player_info) if observation.player_info else None
        evidence["subgoal_description"] = getattr(current_subgoal, "description", "")
        evidence["expected_signal"] = metadata.get("expected_signal")
        evidence["task_program_id"] = metadata.get("task_program_id")

        return GoalHypothesis(
            family=family,
            confidence=max(0.05, min(0.99, confidence)),
            target_objects=target_objects,
            relevant_colors=relevant_colors,
            possible_player=possible_player,
            source=source,
            evidence=evidence,
        )

    @staticmethod
    def _from_task_program(task_program: Optional[Any]) -> Optional[str]:
        family = getattr(task_program, "goal_family", None)
        if family:
            return str(family)
        return None

    @staticmethod
    def _from_goal(current_goal: Optional[GameGoal]) -> Optional[str]:
        if current_goal is None:
            return None
        hypothesis = getattr(current_goal, "hypothesis", "") or ""
        if "goal_family=" not in hypothesis:
            return None
        frag = hypothesis.split("goal_family=", 1)[1]
        return frag.split(" ", 1)[0].split("|", 1)[0].strip("_| ")

    @staticmethod
    def _from_memory(memory: Optional[GameMemory]) -> Optional[str]:
        if memory is None:
            return None
        try:
            hypotheses = getattr(memory, "hypotheses", {}) or {}
            for key, _value in hypotheses.items():
                text = str(key)
                if text.startswith("game_type::"):
                    return text.split("::", 1)[1]
        except Exception:
            return None
        return None

    @staticmethod
    def _from_observation(observation: GameObservation) -> str:
        if observation.player_info and observation.objects:
            return "navigation"
        if "ACTION6" in observation.action_semantics:
            return "click_puzzle"
        return "unknown"

    @staticmethod
    def _relevant_colors(
        observation: GameObservation,
        metadata: Dict[str, Any],
    ) -> List[str]:
        colors: List[str] = []
        target_color = metadata.get("click_target_color")
        if target_color:
            colors.append(str(target_color))
        for obj in observation.objects:
            color = str(obj.get("color", "")).strip()
            if not color or color == "unknown":
                continue
            if obj.get("is_player"):
                continue
            if color not in colors:
                colors.append(color)
        return colors[:4]

    @staticmethod
    def _target_objects(
        observation: GameObservation,
        metadata: Dict[str, Any],
        relevant_colors: List[str],
    ) -> List[Dict[str, Any]]:
        target_objects: List[Dict[str, Any]] = []
        click_targets = metadata.get("click_targets", []) or []
        for target in click_targets:
            if isinstance(target, dict):
                target_objects.append(dict(target))
        if target_objects:
            return target_objects[:4]

        color_set = set(relevant_colors)
        for obj in observation.objects:
            if obj.get("is_player"):
                continue
            color = str(obj.get("color", "")).strip()
            if color_set and color not in color_set:
                continue
            target_objects.append(
                {
                    "value": obj.get("value"),
                    "color": color,
                    "center_x": int(obj.get("center_x", 0)),
                    "center_y": int(obj.get("center_y", 0)),
                    "size": int(obj.get("size", 0)),
                }
            )
        return target_objects[:6]
