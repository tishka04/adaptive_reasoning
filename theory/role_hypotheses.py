"""Human-facing action-role and goal-family hypotheses.

This layer is deliberately thinner than the low-level action-effect mechanics:
it names what an action is *for* and what family of objective the game appears
to instantiate. The keys emitted here are scored by the same epistemic metric
as mechanic hypotheses, especially for human-alignment recall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .epistemic_metrics import HypothesisRecord, HypothesisStatus


ACTION_ROLE_ALIASES = {
    "movement": "move",
    "move_like": "move",
    "move": "move",
    "selector": "select",
    "select": "select",
    "click": "select",
    "click_activation": "select",
    "confirm": "confirm",
    "validation": "confirm",
    "validate": "confirm",
    "toggle": "toggle",
    "switch": "control_switch",
    "role_switch": "control_switch",
    "control": "control_switch",
    "control_switch": "control_switch",
    "interact": "interact",
    "interaction": "interact",
    "submit": "submit",
    "submit_like": "submit",
    "reset": "reset_like",
    "reset_like": "reset_like",
    "unknown": "unknown",
}

CANDIDATE_ACTION_ROLES = (
    "move",
    "select",
    "confirm",
    "toggle",
    "control_switch",
    "interact",
    "submit",
    "reset_like",
    "unknown",
)

GOAL_FAMILY_ALIASES = {
    "match": "correspondence",
    "matching": "correspondence",
    "correspondence": "correspondence",
    "navigation": "navigation",
    "collection": "collection",
    "collect": "collection",
    "elimination": "elimination",
    "eliminate": "elimination",
    "construction": "construction",
    "construct": "construction",
    "transformation": "transformation",
    "transform": "transformation",
    "unknown": "unknown",
}

CANDIDATE_GOAL_FAMILIES = (
    "correspondence",
    "navigation",
    "collection",
    "elimination",
    "construction",
    "transformation",
    "unknown",
)

SEMANTIC_CONFIRM_CONFIDENCE = 0.60
SEMANTIC_REFUTE_CONTRADICTIONS = 2
SMALL_STRUCTURED_CHANGE_MAX = 75


def normalize_action_role(raw: str) -> str:
    """Normalize human-facing action-role labels."""
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ACTION_ROLE_ALIASES.get(key, key or "unknown")


def action_role_key(action: str, role: str) -> str:
    """Canonical metric key for an action-role hypothesis."""
    return f"action_role::{str(action).upper()}::{normalize_action_role(role)}"


def normalize_goal_family(raw: str) -> str:
    """Normalize a goal-family label."""
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return GOAL_FAMILY_ALIASES.get(key, key or "unknown")


def goal_family_key(family: str) -> str:
    """Canonical metric key for a goal-family hypothesis."""
    return f"goal_family::{normalize_goal_family(family)}"


@dataclass
class _SemanticHypothesis:
    """Common evidence policy for semantic hypotheses."""

    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    prior_confidence: float = 0.0
    experiments_spent: int = 0
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED

    @property
    def support(self) -> int:
        return len(self.evidence_for)

    @property
    def contradictions(self) -> int:
        return len(self.evidence_against)

    @property
    def confidence(self) -> float:
        total = self.support + self.contradictions
        if total == 0:
            return max(0.0, min(1.0, self.prior_confidence))
        observed = self.support / total
        evidence_scale = min(1.0, total / 2.0)
        scaled_observed = observed * evidence_scale
        if self.support > 0:
            scaled_observed = max(scaled_observed, self.prior_confidence)
        return max(0.0, min(1.0, scaled_observed))

    def _record_for(self, key: str, description: str) -> HypothesisRecord:
        return HypothesisRecord(
            key=key,
            description=description,
            status=self.status,
            support=self.support,
            contradictions=self.contradictions,
            experiments_spent=self.experiments_spent,
        )

    def add_evidence_for(self, label: str) -> None:
        self.evidence_for.append(str(label))
        self._recompute_status()

    def add_evidence_against(self, label: str) -> None:
        self.evidence_against.append(str(label))
        self._recompute_status()

    def _recompute_status(self) -> None:
        if (
            self.contradictions >= SEMANTIC_REFUTE_CONTRADICTIONS
            and self.contradictions > self.support
        ):
            self.status = HypothesisStatus.REFUTED
        elif self.support > self.contradictions and self.confidence >= SEMANTIC_CONFIRM_CONFIDENCE:
            self.status = HypothesisStatus.CONFIRMED
        else:
            self.status = HypothesisStatus.UNRESOLVED


@dataclass
class ActionRoleHypothesis(_SemanticHypothesis):
    """A falsifiable claim about the role an action plays for the player."""

    action: str = ""
    role: str = "unknown"
    statement: str = ""

    def __post_init__(self) -> None:
        self.action = str(self.action).upper()
        self.role = normalize_action_role(self.role)
        if not self.statement:
            self.statement = f"{self.action} plays the role '{self.role}'"
        self._recompute_status()

    @property
    def key(self) -> str:
        return action_role_key(self.action, self.role)

    def predicts(self, effect: Any) -> Optional[bool]:
        changed = int(getattr(effect, "num_changed", 0) or 0)
        player_moved = bool(getattr(effect, "player_moved", False))
        game_over = bool(getattr(effect, "game_over", False))
        level_complete = bool(getattr(effect, "level_complete", False))

        if self.role == "move":
            return player_moved
        if self.role in {"select", "toggle", "control_switch"}:
            if (
                0 < changed <= SMALL_STRUCTURED_CHANGE_MAX
                and not player_moved
                and not game_over
                and not level_complete
            ):
                return True
            # Large downstream changes can be caused by the newly selected
            # controllable/branch; they do not falsify the role by themselves.
            return None
        if self.role in {"confirm", "submit"}:
            return level_complete
        if self.role == "interact":
            return changed > 0 or level_complete
        if self.role == "reset_like":
            return game_over and changed > 0
        return None

    def observe(self, effect: Any, *, was_experiment: bool = False) -> None:
        holds = self.predicts(effect)
        if holds is None:
            return
        if was_experiment:
            self.experiments_spent += 1
        if holds:
            self.add_evidence_for(f"observed:{self.role}")
        else:
            self.add_evidence_against(f"missed:{self.role}")

    def to_record(self) -> HypothesisRecord:
        return self._record_for(self.key, self.statement)


@dataclass
class GoalFamilyHypothesis(_SemanticHypothesis):
    """A falsifiable claim about the game's objective family."""

    family: str = "unknown"
    statement: str = ""

    def __post_init__(self) -> None:
        self.family = normalize_goal_family(self.family)
        if not self.statement:
            self.statement = f"goal family is '{self.family}'"
        self._recompute_status()

    @property
    def key(self) -> str:
        return goal_family_key(self.family)

    def predicts(self, effect: Any) -> Optional[bool]:
        changed = int(getattr(effect, "num_changed", 0) or 0)
        player_moved = bool(getattr(effect, "player_moved", False))
        level_complete = bool(getattr(effect, "level_complete", False))

        if self.family == "navigation":
            return True if player_moved else None
        if self.family == "transformation":
            return True if changed >= SMALL_STRUCTURED_CHANGE_MAX else None
        if self.family == "correspondence":
            # A level advance is positive evidence for a goal hypothesis, but
            # ordinary non-winning transitions are not contradictions.
            return True if level_complete else None
        return None

    def observe(self, effect: Any, *, was_experiment: bool = False) -> None:
        holds = self.predicts(effect)
        if holds is None:
            return
        if was_experiment:
            self.experiments_spent += 1
        if holds:
            self.add_evidence_for(f"observed:{self.family}")
        else:
            self.add_evidence_against(f"missed:{self.family}")

    def to_record(self) -> HypothesisRecord:
        return self._record_for(self.key, self.statement)


def load_task_program_semantic_hypotheses(
    path: Path,
) -> Tuple[List[ActionRoleHypothesis], List[GoalFamilyHypothesis]]:
    """Build semantic hypotheses from a compiled human task program.

    The task program is treated as a human-prior evidence source, not as the
    metric oracle. The returned hypotheses still enter the normal ledger and
    can gain or lose evidence from later observations.
    """
    path = Path(path)
    if not path.is_file():
        return [], []
    with open(path, "r", encoding="utf-8") as handle:
        program = json.load(handle)

    default_confidence = _as_float(program.get("confidence"), 0.7)
    action_roles: List[ActionRoleHypothesis] = []
    for item in program.get("action_roles", []) or []:
        action = str(item.get("action", "")).upper()
        role = normalize_action_role(str(item.get("role", "")))
        if not action or not role:
            continue
        evidence = str(item.get("evidence") or "task_program_action_role")
        confidence = _as_float(item.get("confidence"), default_confidence)
        action_roles.append(ActionRoleHypothesis(
            action=action,
            role=role,
            evidence_for=[evidence],
            prior_confidence=confidence,
        ))

    goal_families: List[GoalFamilyHypothesis] = []
    family = normalize_goal_family(str(program.get("goal_family", "")))
    if family and family != "unknown":
        evidence_for = ["task_program_goal_family"]
        macro_goal = str(program.get("macro_goal", "")).lower()
        if family in macro_goal:
            evidence_for.append("macro_goal_mentions_family")
        for subgoal in program.get("subgoal_tests", []) or []:
            text = " ".join(
                str(subgoal.get(field_name, ""))
                for field_name in ("id", "description", "verification", "expected_signal")
            ).lower()
            if family in text or "match" in text or "level advance" in text:
                evidence_for.append(f"subgoal:{subgoal.get('id', family)}")
                break
        goal_families.append(GoalFamilyHypothesis(
            family=family,
            evidence_for=evidence_for,
            prior_confidence=default_confidence,
        ))

    return action_roles, goal_families


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
