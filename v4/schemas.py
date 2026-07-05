"""Shared data structures for the V4 adaptive reasoning architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ObjectInfo:
    object_id: int
    value: int
    cells: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int
    shape_signature: tuple[int, ...] = ()


@dataclass
class PlayerHypothesis:
    value: int
    position: tuple[int, int]
    confidence: float
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass
class FrameDiff:
    changed_cells: list[tuple[int, int]]
    before_values: list[int]
    after_values: list[int]
    created_object_ids: list[int]
    removed_object_ids: list[int]
    moved_objects: list[tuple[int, tuple[int, int], tuple[int, int]]]
    player_displacement: Optional[tuple[int, int]]
    is_noop: bool
    game_over: bool
    level_complete: bool
    num_changed: int = 0


@dataclass
class SurpriseField:
    pixel_surprise: float = 0.0
    object_surprise: float = 0.0
    causal_surprise: float = 0.0
    topology_surprise: float = 0.0
    semantic_surprise: float = 0.0
    salient_cells: list[tuple[int, int]] = field(default_factory=list)

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
            ) / 5.0,
        )


@dataclass
class TopologyState:
    reachable_regions: list[set[tuple[int, int]]] = field(default_factory=list)
    region_graph_edges: list[tuple[int, int]] = field(default_factory=list)
    player_region_id: Optional[int] = None
    unlocked_regions: list[int] = field(default_factory=list)


@dataclass
class ObservationV4:
    raw_grid: np.ndarray
    grid_hash: int
    game_state: str
    levels_completed: int
    available_actions: list[str]
    objects: list[ObjectInfo] = field(default_factory=list)
    player_hypotheses: list[PlayerHypothesis] = field(default_factory=list)
    frame_diff: Optional[FrameDiff] = None
    surprise: SurpriseField = field(default_factory=SurpriseField)
    topology: TopologyState = field(default_factory=TopologyState)
    affordances: list[dict[str, Any]] = field(default_factory=list)
    local_contexts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def best_player(self) -> Optional[PlayerHypothesis]:
        if not self.player_hypotheses:
            return None
        return max(self.player_hypotheses, key=lambda p: p.confidence)


@dataclass
class PrimitiveAction:
    name: str
    x: Optional[int] = None
    y: Optional[int] = None

    def __repr__(self) -> str:
        if self.x is not None and self.y is not None:
            return f"{self.name}({self.x},{self.y})"
        return self.name


@dataclass
class TransitionRecord:
    prev_hash: int
    next_hash: int
    action: PrimitiveAction
    diff: FrameDiff
    surprise: SurpriseField
    level_completed: bool
    game_over: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Predicate:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Effect:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Operator:
    operator_id: str
    kind: str
    primitive_action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Predicate] = field(default_factory=list)
    expected_effects: list[Effect] = field(default_factory=list)
    confidence: float = 0.0
    support: int = 0
    contradictions: int = 0
    cost_estimate: float = 1.0
    risk_estimate: float = 0.0
    contexts_supported: list[str] = field(default_factory=list)
    survival_score: float = 0.0


@dataclass
class Rule:
    rule_id: str
    family: str
    conditions: list[Predicate]
    effect: Effect
    confidence: float = 0.0
    support: int = 0
    contradictions: int = 0
    ontology_tags: list[str] = field(default_factory=list)
    stage: str = "validated"
    survival_score: float = 0.0


@dataclass
class OntologyHypothesis:
    ontology_id: str
    kind: str
    confidence: float = 0.5
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    salient_object_ids: list[int] = field(default_factory=list)
    active_affordance_biases: dict[str, float] = field(default_factory=dict)
    terminal_hypotheses: list[str] = field(default_factory=list)


@dataclass
class Project:
    project_id: str
    description: str
    ontology_id: str
    law_dependencies: list[str]
    kind: str
    expected_info_gain: float
    expected_structural_gain: float
    expected_terminal_gain: float
    estimated_cost: float
    fragility: float
    dignity: float
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    survival_score: float = 0.0


@dataclass
class MindProposal:
    mind_name: str
    project: Project
    operator_plan: list[str] = field(default_factory=list)
    primitive_plan: list[PrimitiveAction] = field(default_factory=list)
    confidence: float = 0.0
    expected_lp: float = 0.0
    expected_sp: float = 0.0
    expected_tp: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Motif:
    motif_id: str
    kind: str
    content: dict[str, Any]
    support: int = 0
    utility: float = 0.0
    terminal_association: float = 0.0
    structural_association: float = 0.0
    survival_score: float = 0.0


@dataclass
class Ritual:
    ritual_id: str
    ontology_kind: str
    prefix: list[PrimitiveAction]
    terminal_signature: dict[str, Any]
    success_rate: float = 0.0
    survival_score: float = 0.0


@dataclass
class MemoryFrame:
    frame_id: str
    ontology_kind: str
    dominant_laws: list[str]
    dominant_projects: list[str]
    salient_entities: list[str]
    terminal_style: str
    atoms: list[str] = field(default_factory=list)
    reliability: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    embedding: list[float] = field(default_factory=list)
    prototype_weight: float = 0.0


@dataclass
class BissociationBridge:
    bridge_id: str
    frame_a_id: str
    frame_b_id: str
    trigger_signature: dict[str, Any]
    hybrid_hypothesis_template: dict[str, Any]
    success_count: int = 0
    failure_count: int = 0
    novelty_score: float = 0.0
    utility_score: float = 0.0


@dataclass
class NegativeBridge:
    bridge_id: str
    frame_a_id: str
    frame_b_id: str
    failure_signature: dict[str, Any]
    penalty_weight: float = 0.0


@dataclass
class BissociativeProposal:
    proposal_id: str
    source_frame_a: str
    source_frame_b: str
    tension_type: str
    hybrid_project: Optional[Project] = None
    hybrid_operator_chain: list[str] = field(default_factory=list)
    closure_hint: Optional[str] = None
    anti_loop_warning: Optional[str] = None
    confidence: float = 0.0


@dataclass
class DissentReport:
    false_progress: dict[str, float] = field(default_factory=dict)
    ontology_warnings: list[str] = field(default_factory=list)
    chamber_warnings: dict[str, list[str]] = field(default_factory=dict)
    loop_warning: bool = False
    suggested_actions: list[str] = field(default_factory=list)


@dataclass
class ActionIntent:
    source: str
    primitive_plan: list[PrimitiveAction] = field(default_factory=list)
    operator_plan: list[str] = field(default_factory=list)
    project_id: Optional[str] = None
    ontology_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
