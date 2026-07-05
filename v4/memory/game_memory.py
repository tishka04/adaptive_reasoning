"""Per-game memory for V4."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..control import BranchScheduler, PhaseController, ProgressTracker
from ..learning import LearningSuite
from .bissociation_engine import BissociationEngine
from .bridge_memory import BridgeMemory
from .frame_memory import FrameMemory
from ..physics import (
    ActionProfiler,
    ConstraintEngine,
    LawCompetition,
    OperatorInducer,
    TeleologyEngine,
)
from ..schemas import Motif, OntologyHypothesis, PrimitiveAction, Project, Ritual, TransitionRecord


@dataclass
class GameMemoryV4:
    """Medium-timescale per-game memory."""

    profiler: ActionProfiler = field(default_factory=ActionProfiler)
    inducer: OperatorInducer = field(default_factory=OperatorInducer)
    constraints: ConstraintEngine = field(default_factory=ConstraintEngine)
    teleology: TeleologyEngine = field(default_factory=TeleologyEngine)
    laws: LawCompetition = field(default_factory=LawCompetition)
    progress: ProgressTracker = field(default_factory=ProgressTracker)
    phase_controller: PhaseController = field(default_factory=PhaseController)
    branch_scheduler: BranchScheduler = field(default_factory=BranchScheduler)
    frame_memory: FrameMemory = field(default_factory=FrameMemory)
    bridge_memory: BridgeMemory = field(default_factory=BridgeMemory)
    bissociation_engine: BissociationEngine = field(default_factory=BissociationEngine)
    learning: LearningSuite = field(default_factory=LearningSuite)
    ontology_priors: dict[str, float] = field(default_factory=dict)
    motifs: dict[str, Motif] = field(default_factory=dict)
    rituals: dict[str, Ritual] = field(default_factory=dict)
    anti_patterns: dict[str, dict[str, Any]] = field(default_factory=dict)
    successful_trace: list[PrimitiveAction] = field(default_factory=list)
    all_traces: list[list[PrimitiveAction]] = field(default_factory=list)
    ontology_history: list[list[tuple[str, float]]] = field(default_factory=list)
    phase_history: list[str] = field(default_factory=list)
    selected_projects: list[str] = field(default_factory=list)
    current_ontologies: list[OntologyHypothesis] = field(default_factory=list)
    previous_frame: Any = None
    current_frame: Any = None
    last_bissociative_proposals: list[Any] = field(default_factory=list)
    mind_stats: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"used": 0.0, "correct": 0.0, "progress": 0.0, "selected": 0.0}
        )
    )
    project_market: Any = None
    total_actions: int = 0
    total_levels_completed: int = 0
    max_level_reached: int = 0
    states_visited: set[int] = field(default_factory=set)

    def on_transition(self, transition: TransitionRecord) -> None:
        self.total_actions += 1
        self.states_visited.add(transition.next_hash)
        if transition.level_completed:
            self.total_levels_completed += 1
            self.max_level_reached = max(self.max_level_reached, self.total_levels_completed)

    def avg_operator_confidence(self) -> float:
        operators = list(self.inducer.operators.values())
        if not operators:
            return 0.0
        top = sorted(operators, key=lambda op: op.confidence, reverse=True)[:8]
        return sum(op.confidence for op in top) / len(top)

    def mind_reliability(self, mind_name: str) -> float:
        stats = self.mind_stats[mind_name]
        if stats["used"] <= 0:
            return 0.5
        accuracy = stats["correct"] / max(stats["used"], 1.0)
        progress = stats["progress"] / max(stats["used"], 1.0)
        return min(0.95, max(0.05, 0.65 * accuracy + 0.35 * progress))

    def record_mind_selection(self, mind_name: str) -> None:
        self.mind_stats[mind_name]["selected"] += 1

    def record_mind_outcome(self, mind_name: str, predicted_ok: bool, progress_gain: float) -> None:
        stats = self.mind_stats[mind_name]
        stats["used"] += 1
        if predicted_ok:
            stats["correct"] += 1
        stats["progress"] += max(0.0, progress_gain)

    def record_phase(self, phase: str) -> None:
        self.phase_history.append(phase)
        if len(self.phase_history) > 200:
            self.phase_history = self.phase_history[-200:]

    def record_ontology_ranking(self, ranked: list[OntologyHypothesis]) -> None:
        snapshot = [(item.kind, round(item.confidence, 3)) for item in ranked[:3]]
        self.ontology_history.append(snapshot)
        if len(self.ontology_history) > 200:
            self.ontology_history = self.ontology_history[-200:]

    def record_project_selection(self, project: Project) -> None:
        self.selected_projects.append(project.project_id)
        if len(self.selected_projects) > 200:
            self.selected_projects = self.selected_projects[-200:]

    def add_motif(self, motif: Motif) -> None:
        existing = self.motifs.get(motif.motif_id)
        if existing is None:
            motif.survival_score = self._motif_score(motif)
            self.motifs[motif.motif_id] = motif
            return
        existing.support += motif.support
        existing.utility = max(existing.utility, motif.utility)
        existing.terminal_association = max(
            existing.terminal_association, motif.terminal_association
        )
        existing.structural_association = max(
            existing.structural_association, motif.structural_association
        )
        existing.survival_score = self._motif_score(existing)

    def add_ritual(self, ritual: Ritual) -> None:
        existing = self.rituals.get(ritual.ritual_id)
        if existing is None:
            ritual.survival_score = self._ritual_score(ritual)
            self.rituals[ritual.ritual_id] = ritual
            return
        existing.success_rate = max(existing.success_rate, ritual.success_rate)
        if len(ritual.prefix) < len(existing.prefix):
            existing.prefix = ritual.prefix
        existing.terminal_signature = ritual.terminal_signature
        existing.survival_score = self._ritual_score(existing)

    def best_ritual_for(self, ontology_kind: str) -> Ritual | None:
        candidates = [
            ritual
            for ritual in self.rituals.values()
            if ritual.ontology_kind == ontology_kind
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.success_rate, -len(item.prefix)))

    def knowledge_level(self) -> float:
        coverage = self.profiler.action_coverage(list(self.profiler.stats.keys())) if self.profiler.stats else 0.0
        pred_accuracy = self.inducer.operator_predictive_accuracy()
        control_success = self.inducer.operator_control_success()
        if self.constraints.rules:
            valid_rules = [
                rule for rule in self.constraints.rules.values() if rule.confidence >= 0.35
            ]
            rule_quality = len(valid_rules) / max(len(self.constraints.rules), 1)
        else:
            rule_quality = 0.0
        ritual_value = min(1.0, len(self.rituals) / 5.0)
        progress = sum(self.progress.scores()) / 3.0
        return (
            0.18 * coverage
            + 0.20 * pred_accuracy
            + 0.18 * control_success
            + 0.16 * rule_quality
            + 0.12 * ritual_value
            + 0.16 * progress
        )

    def enforce_budgets(self, memory: Any | None = None) -> dict[str, Any]:
        before = {
            "ontologies": len(self.current_ontologies),
            "operators": len(self.inducer.operators),
            "rules": len(self.constraints.rules),
            "validated_teleology": len(self.teleology.hypotheses()),
            "speculative_teleology": len(self.teleology.speculative_hypotheses()),
            "projects": len(self.project_market.projects) if self.project_market is not None else 0,
            "motifs": len(self.motifs),
            "rituals": len(self.rituals),
        }

        removed_projects: list[str] = []
        self.current_ontologies = self.current_ontologies[:3]
        removed_operators = self.inducer.prune(max_active=10, memory=memory)
        removed_rules = self.constraints.prune(max_active=6)
        validated_before_ids = {rule.rule_id for rule in self.teleology.hypotheses()}
        speculative_before_ids = {rule.rule_id for rule in self.teleology.speculative_hypotheses()}
        self.teleology.prune(max_validated=3, max_speculative=4)
        validated_after_ids = {rule.rule_id for rule in self.teleology.hypotheses()}
        speculative_after_ids = {rule.rule_id for rule in self.teleology.speculative_hypotheses()}
        if self.project_market is not None:
            removed_projects = self.project_market.prune(max_active=6, memory=memory)
        removed_motifs = self._prune_motifs(max_active=5)
        removed_rituals = self._prune_rituals(max_active=2)

        after = {
            "ontologies": len(self.current_ontologies),
            "operators": len(self.inducer.operators),
            "rules": len(self.constraints.rules),
            "validated_teleology": len(validated_after_ids),
            "speculative_teleology": len(speculative_after_ids),
            "projects": len(self.project_market.projects) if self.project_market is not None else 0,
            "motifs": len(self.motifs),
            "rituals": len(self.rituals),
        }
        return {
            "before": before,
            "after": after,
            "removed": {
                "operators": removed_operators,
                "rules": removed_rules,
                "validated_teleology": sorted(validated_before_ids - validated_after_ids),
                "speculative_teleology": sorted(speculative_before_ids - speculative_after_ids),
                "projects": removed_projects,
                "motifs": removed_motifs,
                "rituals": removed_rituals,
            },
        }

    def _prune_motifs(self, max_active: int) -> list[str]:
        if len(self.motifs) <= max_active:
            return []
        ranked = sorted(self.motifs.values(), key=self._motif_score, reverse=True)
        keep_ids = {motif.motif_id for motif in ranked[:max_active]}
        removed = [motif_id for motif_id in self.motifs if motif_id not in keep_ids]
        self.motifs = {motif.motif_id: motif for motif in ranked[:max_active]}
        return removed

    def _prune_rituals(self, max_active: int) -> list[str]:
        if len(self.rituals) <= max_active:
            return []
        ranked = sorted(self.rituals.values(), key=self._ritual_score, reverse=True)
        keep_ids = {ritual.ritual_id for ritual in ranked[:max_active]}
        removed = [ritual_id for ritual_id in self.rituals if ritual_id not in keep_ids]
        self.rituals = {ritual.ritual_id: ritual for ritual in ranked[:max_active]}
        return removed

    def _motif_score(self, motif: Motif) -> float:
        support_value = min(1.0, motif.support / 4.0)
        base = (
            0.35 * motif.utility
            + 0.25 * motif.structural_association
            + 0.25 * motif.terminal_association
            + 0.15 * support_value
        )
        if motif.support <= 1:
            base -= 0.10
        learned = self.learning.compression_value.estimate_motif(motif, self)
        score = 0.75 * base + 0.25 * learned
        motif.survival_score = score
        return score

    def _ritual_score(self, ritual: Ritual) -> float:
        brevity = 1.0 / max(len(ritual.prefix), 1)
        base = 0.65 * ritual.success_rate + 0.20 * brevity + 0.15 * min(
            1.0, ritual.terminal_signature.get("levels_completed", 0) / 3.0
        )
        learned = self.learning.compression_value.estimate_ritual(ritual, self)
        score = 0.75 * base + 0.25 * learned
        ritual.survival_score = score
        return score
