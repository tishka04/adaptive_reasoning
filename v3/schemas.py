"""V3 shared data structures.

All core dataclasses used across the V3 architecture live here to avoid
circular imports.  Modules import from ``v3.schemas`` only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Primitives
# =====================================================================

@dataclass
class PrimitiveAction:
    """Lowest-level game action (button press or click)."""
    name: str                             # RESET / ACTION1 / … / ACTION7
    x: Optional[int] = None
    y: Optional[int] = None

    def __repr__(self) -> str:
        if self.x is not None:
            return f"{self.name}({self.x},{self.y})"
        return self.name


# =====================================================================
# Perception
# =====================================================================

@dataclass
class ObjectInfo:
    """Connected-component object extracted from the grid."""
    object_id: int
    value: int
    cells: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]       # (r_min, c_min, r_max, c_max)
    center: Tuple[float, float]
    area: int
    shape_signature: Tuple[int, ...] = ()  # e.g. sorted row-offsets from center

    def overlaps(self, other: "ObjectInfo") -> bool:
        return bool(set(self.cells) & set(other.cells))


@dataclass
class PlayerHypothesis:
    """A candidate player entity in the grid."""
    value: int
    position: Tuple[int, int]             # (row, col)
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameDiff:
    """Structured diff between two consecutive game frames."""
    changed_cells: List[Tuple[int, int]]
    changed_values_before: List[int]
    changed_values_after: List[int]
    created_objects: List[int]            # object_ids that appeared
    removed_objects: List[int]            # object_ids that disappeared
    moved_objects: List[Tuple[int, Tuple[int, int], Tuple[int, int]]]  # (obj_id, old_pos, new_pos)
    player_displacement: Optional[Tuple[int, int]] = None  # (dy, dx)
    game_over: bool = False
    level_complete: bool = False
    num_changed: int = 0

    @property
    def is_noop(self) -> bool:
        return self.num_changed == 0 and not self.game_over and not self.level_complete


@dataclass
class LocalContext:
    """Neighbourhood descriptor around a cell."""
    center: Tuple[int, int]
    patch: np.ndarray                     # small NxN window
    nearby_object_values: List[int]
    free_directions: List[str]            # "up", "down", "left", "right"
    danger_score: float = 0.0


class AffordanceKind(str, enum.Enum):
    CLICKABLE = "clickable"
    TRAVERSABLE = "traversable"
    HAZARDOUS = "hazardous"
    COLLECTIBLE = "collectible"
    MOVABLE = "movable"
    UNKNOWN = "unknown"


@dataclass
class Affordance:
    """An inferred interaction possibility."""
    kind: AffordanceKind
    target: int | Tuple[int, int]         # object_id or cell
    confidence: float = 0.5


@dataclass
class GameObservation:
    """Full structured perception output for one frame."""
    raw_grid: np.ndarray
    grid_hash: int
    game_state: str                       # NOT_PLAYED / NOT_FINISHED / WIN / GAME_OVER
    levels_completed: int
    available_actions: List[str]

    objects: List[ObjectInfo] = field(default_factory=list)
    player_candidates: List[PlayerHypothesis] = field(default_factory=list)
    salient_regions: List[LocalContext] = field(default_factory=list)
    affordances: List[Affordance] = field(default_factory=list)
    danger_map: Optional[np.ndarray] = None

    frame_diff: Optional[FrameDiff] = None
    local_contexts: List[LocalContext] = field(default_factory=list)

    # Injected downstream context
    visual_cortex_summary: str = ""

    @property
    def best_player(self) -> Optional[PlayerHypothesis]:
        if not self.player_candidates:
            return None
        return max(self.player_candidates, key=lambda p: p.confidence)


# =====================================================================
# Mechanic Inference — Predicates & Effects (symbolic DSL)
# =====================================================================

class Predicate:
    """Base class for state predicates."""
    def check(self, obs: GameObservation) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class PlayerExists(Predicate):
    def check(self, obs: GameObservation) -> bool:
        return obs.best_player is not None and obs.best_player.confidence > 0.4


class CellFree(Predicate):
    """Cell at relative offset from player is traversable (value == 0)."""
    def __init__(self, relative: Tuple[int, int]):
        self.relative = relative

    def check(self, obs: GameObservation) -> bool:
        p = obs.best_player
        if p is None:
            return False
        r, c = p.position[0] + self.relative[0], p.position[1] + self.relative[1]
        grid = obs.raw_grid
        if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
            return False
        return int(grid[r, c]) == 0

    def __repr__(self) -> str:
        return f"CellFree(rel={self.relative})"


class AdjacentToValue(Predicate):
    def __init__(self, value: int):
        self.value = value

    def check(self, obs: GameObservation) -> bool:
        p = obs.best_player
        if p is None:
            return False
        r, c = p.position
        grid = obs.raw_grid
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if int(grid[nr, nc]) == self.value:
                    return True
        return False

    def __repr__(self) -> str:
        return f"AdjacentToValue({self.value})"


class ObjectExists(Predicate):
    def __init__(self, value: Optional[int] = None):
        self.value = value

    def check(self, obs: GameObservation) -> bool:
        if self.value is None:
            return len(obs.objects) > 0
        return any(o.value == self.value for o in obs.objects)

    def __repr__(self) -> str:
        return f"ObjectExists(v={self.value})"


class InDangerZone(Predicate):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check(self, obs: GameObservation) -> bool:
        if obs.danger_map is None or obs.best_player is None:
            return False
        r, c = obs.best_player.position
        return float(obs.danger_map[r, c]) >= self.threshold

    def __repr__(self) -> str:
        return f"InDangerZone(>{self.threshold})"


class ObjectCount(Predicate):
    """True when count of objects with given value meets comparison."""
    def __init__(self, value: int, cmp: str = "==", n: int = 0):
        self.value = value
        self.cmp = cmp
        self.n = n

    def check(self, obs: GameObservation) -> bool:
        count = sum(1 for o in obs.objects if o.value == self.value)
        if self.cmp == "==":
            return count == self.n
        elif self.cmp == ">=":
            return count >= self.n
        elif self.cmp == "<=":
            return count <= self.n
        elif self.cmp == ">":
            return count > self.n
        elif self.cmp == "<":
            return count < self.n
        return False

    def __repr__(self) -> str:
        return f"ObjectCount(v={self.value}, {self.cmp}{self.n})"


# ── Effects ──

class Effect:
    """Base class for predicted operator effects."""
    def __repr__(self) -> str:
        return self.__class__.__name__


class PlayerDisplacement(Effect):
    def __init__(self, dy: int, dx: int):
        self.dy = dy
        self.dx = dx

    def matches(self, diff: FrameDiff) -> bool:
        if diff.player_displacement is None:
            return False
        return diff.player_displacement == (self.dy, self.dx)

    def __repr__(self) -> str:
        return f"PlayerDisp(dy={self.dy},dx={self.dx})"


class ChangedCells(Effect):
    def __init__(self, count_range: Tuple[int, int] = (1, 100)):
        self.count_range = count_range

    def matches(self, diff: FrameDiff) -> bool:
        return self.count_range[0] <= diff.num_changed <= self.count_range[1]

    def __repr__(self) -> str:
        return f"ChangedCells({self.count_range})"


class RemovesObject(Effect):
    def __init__(self, value: Optional[int] = None):
        self.value = value

    def matches(self, diff: FrameDiff) -> bool:
        return bool(diff.removed_objects)

    def __repr__(self) -> str:
        return f"RemovesObject(v={self.value})"


class CreatesObject(Effect):
    def __init__(self, value: Optional[int] = None):
        self.value = value

    def matches(self, diff: FrameDiff) -> bool:
        return bool(diff.created_objects)

    def __repr__(self) -> str:
        return f"CreatesObject(v={self.value})"


class GlobalGridChange(Effect):
    """Many cells change at once (e.g. toggle/transform)."""
    def __init__(self, min_cells: int = 5):
        self.min_cells = min_cells

    def matches(self, diff: FrameDiff) -> bool:
        return diff.num_changed >= self.min_cells

    def __repr__(self) -> str:
        return f"GlobalGridChange(min={self.min_cells})"


class GameOverEffect(Effect):
    def matches(self, diff: FrameDiff) -> bool:
        return diff.game_over


class LevelCompleteEffect(Effect):
    def matches(self, diff: FrameDiff) -> bool:
        return diff.level_complete


class NoEffect(Effect):
    def matches(self, diff: FrameDiff) -> bool:
        return diff.is_noop


# =====================================================================
# Operators
# =====================================================================

class OperatorKind(str, enum.Enum):
    MOVE = "move"
    CLICK = "click"
    INTERACT = "interact"
    AVOID = "avoid"
    PUSH = "push"
    TOGGLE = "toggle"
    SEQUENCE = "sequence"
    NOOP = "noop"
    LETHAL = "lethal"
    GLOBAL_TRANSFORM = "global_transform"
    UNKNOWN = "unknown"


@dataclass
class Operator:
    """A reusable state-conditioned action schema."""
    operator_id: str
    kind: OperatorKind
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[Predicate] = field(default_factory=list)
    expected_effects: List[Effect] = field(default_factory=list)
    termination: List[Predicate] = field(default_factory=list)

    # Tracking
    confidence: float = 0.0
    support: int = 0
    contradictions: int = 0
    cost_estimate: float = 1.0
    risk_estimate: float = 0.0

    # Raw action mapping
    primitive_action: Optional[str] = None   # e.g. "ACTION1"
    primitive_x: Optional[int] = None
    primitive_y: Optional[int] = None

    def preconditions_met(self, obs: GameObservation) -> bool:
        return all(p.check(obs) for p in self.preconditions)

    def update_confidence(self) -> None:
        total = self.support + self.contradictions
        if total == 0:
            self.confidence = 0.0
        else:
            raw = self.support / total
            # Sigmoid-like scaling: need enough evidence
            scale = min(1.0, total / 8.0)
            self.confidence = raw * scale

    def __repr__(self) -> str:
        return (f"Op({self.operator_id}, {self.kind.value}, "
                f"conf={self.confidence:.2f}, n={self.support})")


@dataclass
class OperatorCall:
    """A planned invocation of an operator with arguments."""
    operator_id: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannedAction:
    """One step in a plan: either a primitive or operator call."""
    primitive: Optional[PrimitiveAction] = None
    operator_call: Optional[OperatorCall] = None
    purpose: str = ""


# =====================================================================
# Rules (symbolic compression of mechanics)
# =====================================================================

@dataclass
class Rule:
    """A compact causal rule inferred from repeated observations."""
    rule_id: str
    conditions: List[Predicate]
    operator_kind: str
    effects: List[Effect]
    confidence: float = 0.0
    support: int = 0
    contradictions: int = 0

    def matches_context(self, obs: GameObservation) -> bool:
        return all(c.check(obs) for c in self.conditions)

    def __repr__(self) -> str:
        return f"Rule({self.rule_id}, conf={self.confidence:.2f})"


# =====================================================================
# Macros / Options
# =====================================================================

@dataclass
class MacroAction:
    """A compiled reusable multi-step action sequence."""
    macro_id: str
    name: str
    initiation_conditions: List[Predicate] = field(default_factory=list)
    steps: List[PlannedAction] = field(default_factory=list)
    termination_conditions: List[Predicate] = field(default_factory=list)
    expected_effects: List[Effect] = field(default_factory=list)
    success_rate: float = 0.0
    avg_cost: float = 0.0
    times_used: int = 0
    times_succeeded: int = 0


# =====================================================================
# Solved trajectories
# =====================================================================

@dataclass
class SolvedTrajectory:
    """Record of a successfully solved level."""
    level_index: int
    primitive_actions: List[PrimitiveAction]
    operator_trace: List[OperatorCall] = field(default_factory=list)
    action_count: int = 0
    solved: bool = True


# =====================================================================
# Failure patterns
# =====================================================================

@dataclass
class FailurePattern:
    """Recorded failure attractor for avoidance."""
    motif_hash: str
    operator_trace: List[str]
    failure_type: str                     # game_over / no_progress / contradiction / trap
    count: int = 1


# =====================================================================
# Mind proposals
# =====================================================================

@dataclass
class MindProposal:
    """A specialist mind's suggested plan."""
    mind_name: str
    objective: str
    candidate_plan: List[OperatorCall]
    confidence: float = 0.0
    expected_progress: float = 0.0
    expected_info_gain: float = 0.0
    estimated_cost: float = 1.0
    estimated_risk: float = 0.0
    justification: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Experiment plans
# =====================================================================

@dataclass
class ExperimentPlan:
    """A plan designed to reduce uncertainty about mechanics."""
    objective: str
    candidate_hypotheses: List[str]
    planned_actions: List[PlannedAction]
    expected_info_gain: float = 0.0


# =====================================================================
# Search nodes
# =====================================================================

@dataclass
class SearchNode:
    """A node in the operator-level beam search."""
    state_hash: int
    predicted_state_summary: Dict[str, Any]
    operator_trace: List[OperatorCall]
    cumulative_cost: float = 0.0
    heuristic_value: float = 0.0
    depth: int = 0

    @property
    def total_score(self) -> float:
        return self.heuristic_value - 0.1 * self.cumulative_cost


# =====================================================================
# Transition record (raw material for mechanic inference)
# =====================================================================

@dataclass
class TransitionRecord:
    """One observed action-effect pair with full context."""
    action: PrimitiveAction
    obs_before: GameObservation
    obs_after: GameObservation
    diff: FrameDiff
    timestamp: int = 0                    # action counter
