"""Guarded high-semantic planning and receding-horizon action arbitration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence, Tuple

from v3.schemas import GameObservation, TransitionRecord

from .compiler import (
    CompiledSemanticOption,
    HypothesisCompiler,
    SemanticActionSlot,
    SlotAnnotation,
)
from .dataset import SemanticTraceWriter, SemanticTrajectoryRecord
from .energy import EnergyBreakdown, HeuristicTrajectoryEnergy
from .llm import HypothesisGenerationResult, TemplateHypothesisGenerator
from .scene_graph import SemanticMemory, build_scene_graph
from .world_model import (
    SemanticTrajectory,
    SemanticWorldModel,
    observed_semantic_effects,
)


class Sage12Mode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    BOUNDED = "bounded"
    ACTIVE = "active"


@dataclass(frozen=True)
class Sage12Config:
    mode: Sage12Mode = Sage12Mode.OFF
    proposal_gate_passed: bool = False
    world_model_gate_passed: bool = False
    energy_gate_passed: bool = False
    active_gate_passed: bool = False
    maximum_depth: int = 3
    beam_width: int = 8
    maximum_advisory_risk: float = 0.10
    maximum_hypotheses: int = 8
    use_learned_energy: bool = False


@dataclass(frozen=True)
class HierarchicalSubgoal:
    goal_id: str
    predicate: str
    parent_goal_id: str = ""
    priority: float = 1.0
    depth: int = 0


@dataclass(frozen=True)
class SemanticActionCandidate:
    action_name: str
    action_data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return _action_key(self.action_name, self.action_data)


@dataclass(frozen=True)
class Sage12Arbitration:
    action_name: str
    action_data: Mapping[str, Any]
    source: str
    reason: str
    confidence: float
    applied: bool
    configured_mode: str
    effective_mode: str
    scene_signature: str = ""
    selected_option_id: str = ""
    trajectory_length: int = 0
    energy: float | None = None
    mt_advisory_id: str = ""
    mt_suggested_action: str = ""
    mt_score: float | None = None


@dataclass(frozen=True)
class _PendingSemanticDecision:
    option: CompiledSemanticOption
    record: SemanticTrajectoryRecord
    predicted_effects: Tuple[str, ...]


class SemanticPlanningController:
    """LLM proposals -> grounded options -> rollouts -> energy -> one action."""

    def __init__(
        self,
        *,
        game_id: str = "",
        generator: Any | None = None,
        compiler: HypothesisCompiler | None = None,
        world_model: SemanticWorldModel | None = None,
        energy: HeuristicTrajectoryEnergy | None = None,
        learned_energy: Any | None = None,
        config: Sage12Config | None = None,
        trace_writer: SemanticTraceWriter | None = None,
        transformation_advisor: Any | None = None,
    ) -> None:
        self.game_id = str(game_id)
        self.generator = generator or TemplateHypothesisGenerator()
        self.compiler = compiler or HypothesisCompiler()
        self.world_model = world_model or SemanticWorldModel()
        self.energy = energy or HeuristicTrajectoryEnergy()
        self.learned_energy = learned_energy
        self.config = config or Sage12Config()
        self.trace_writer = trace_writer
        self.transformation_advisor = transformation_advisor
        self.memory = SemanticMemory()
        self._branch_index = 0
        self._step_index = 0
        self._probed_contexts: set[str] = set()
        self._pending: _PendingSemanticDecision | None = None
        self._evaluations = 0
        self._applied = 0
        self._shadow = 0
        self._parse_failures = 0
        self._rejected = 0
        self._danger_vetoes = 0
        self._protected_blocks = 0
        self._gate_downgrades = 0
        self._observed_outcomes = 0
        self._productive_outcomes = 0
        self._unsafe_outcomes = 0
        self._mt_failures = 0
        self._last_record: SemanticTrajectoryRecord | None = None

    @property
    def configured_mode(self) -> Sage12Mode:
        mode = self.config.mode
        return mode if isinstance(mode, Sage12Mode) else Sage12Mode(str(mode))

    @property
    def effective_mode(self) -> Sage12Mode:
        mode = self.configured_mode
        gates = (
            self.config.proposal_gate_passed
            and self.config.world_model_gate_passed
            and self.config.energy_gate_passed
        )
        if mode in {Sage12Mode.BOUNDED, Sage12Mode.ACTIVE} and not gates:
            return Sage12Mode.SHADOW
        if mode == Sage12Mode.ACTIVE and not self.config.active_gate_passed:
            return Sage12Mode.BOUNDED
        return mode

    def arbitrate(
        self,
        *,
        symbolic_action_name: str,
        symbolic_action_data: Mapping[str, Any] | None,
        symbolic_source: str,
        observation: GameObservation,
        candidates: Sequence[Any],
        protected_competence_available: bool,
        danger_veto: Callable[[str, Mapping[str, Any]], bool],
        subgoals: Sequence[HierarchicalSubgoal] | None = None,
        prebuilt_scene_graph: Any | None = None,
        rollout_initial_state: frozenset[str] | None = None,
    ) -> Sage12Arbitration:
        configured = self.configured_mode
        effective = self.effective_mode
        legal_candidates = _normalize_candidates(
            candidates,
            include=SemanticActionCandidate(
                str(symbolic_action_name),
                dict(symbolic_action_data or {}),
            ),
        )
        mt_advisory = self._advise_transformations(
            observation=observation,
            candidates=legal_candidates,
            executed_action_name=str(symbolic_action_name),
            executed_action_data=dict(symbolic_action_data or {}),
        )
        unchanged = Sage12Arbitration(
            action_name=str(symbolic_action_name),
            action_data=dict(symbolic_action_data or {}),
            source=str(symbolic_source),
            reason="SAGE12 semantic authority disabled",
            confidence=0.0,
            applied=False,
            configured_mode=configured.value,
            effective_mode=effective.value,
            mt_advisory_id=str(
                getattr(mt_advisory, "advisory_id", "")
            ),
            mt_suggested_action=str(
                getattr(mt_advisory, "suggested_action_name", "")
            ),
            mt_score=(
                float(mt_advisory.score)
                if mt_advisory is not None
                and getattr(mt_advisory, "advisory_id", "")
                else None
            ),
        )
        if configured == Sage12Mode.OFF:
            return unchanged
        if configured != effective:
            self._gate_downgrades += 1
        self._step_index += 1
        self._evaluations += 1
        graph = (
            prebuilt_scene_graph
            if prebuilt_scene_graph is not None
            else build_scene_graph(observation)
        )
        goal = _select_subgoal(subgoals)
        try:
            generation: HypothesisGenerationResult = self.generator.generate(
                graph=graph,
                available_actions=tuple(
                    candidate.action_name for candidate in legal_candidates
                ),
                subgoal=goal.predicate,
            )
        except Exception as exc:
            self._parse_failures += 1
            self._pending = None
            return replace(
                unchanged,
                reason=(
                    "SAGE12 proposal backend failed closed: "
                    f"{type(exc).__name__}"
                ),
                scene_signature=graph.signature,
            )
        if generation.parse_error:
            self._parse_failures += 1
        compilation = self.compiler.compile(
            generation.hypotheses[: self.config.maximum_hypotheses],
            graph=graph,
            legal_candidates=legal_candidates,
        )
        self._rejected += len(compilation.rejected)
        trajectories = self.world_model.rollout(
            initial_state=(
                graph.state_predicates
                if rollout_initial_state is None
                else rollout_initial_state
            ),
            options=compilation.options,
            maximum_depth=self.config.maximum_depth,
            beam_width=self.config.beam_width,
        )
        safe_trajectories = []
        for trajectory in trajectories:
            option = trajectory.first_option
            if danger_veto(option.action_name, option.action_data):
                self._danger_vetoes += 1
                continue
            breakdown = self.energy.score(
                trajectory,
                goal_predicate=goal.predicate,
            )
            if breakdown.risk > self.config.maximum_advisory_risk:
                self._danger_vetoes += 1
                continue
            safe_trajectories.append((trajectory, breakdown))
        ranked = self._rank(safe_trajectories)
        if not ranked:
            self._pending = None
            return replace(
                unchanged,
                reason="SAGE12 found no safe grounded semantic trajectory",
                scene_signature=graph.signature,
            )
        selected_trajectory, selected_energy = ranked[0]
        selected_option = selected_trajectory.first_option
        context = graph.signature
        can_apply = effective == Sage12Mode.ACTIVE
        if effective == Sage12Mode.BOUNDED:
            can_apply = context not in self._probed_contexts
        if protected_competence_available:
            can_apply = False
            self._protected_blocks += 1
        applied = bool(can_apply)
        if applied:
            self._applied += 1
            if effective == Sage12Mode.BOUNDED:
                self._probed_contexts.add(context)
            selected_action_name = selected_option.action_name
            selected_action_data = dict(selected_option.action_data)
            selected_source = "sage12_semantic_planner"
            reason = (
                "lowest-energy grounded semantic trajectory; executing only "
                "its first action under receding-horizon control"
            )
        else:
            self._shadow += 1
            selected_action_name = str(symbolic_action_name)
            selected_action_data = dict(symbolic_action_data or {})
            selected_source = str(symbolic_source)
            reason = (
                "SAGE12 ranked a counterfactual trajectory but symbolic "
                "authority was preserved"
            )
        actual_key = _action_key(selected_action_name, selected_action_data)
        pending_option = next(
            (
                option
                for option in compilation.options
                if option.action_key == actual_key
            ),
            None,
        )
        record = SemanticTrajectoryRecord(
            source_game_id=self.game_id,
            branch_index=self._branch_index,
            step_index=self._step_index,
            scene_signature=graph.signature,
            subgoal=goal.predicate,
            proposal_ids=tuple(
                hypothesis.hypothesis_id
                for hypothesis in generation.hypotheses
            ),
            rejected_proposals=compilation.rejected,
            trajectory_option_ids=tuple(
                tuple(step.option.option_id for step in trajectory.steps)
                for trajectory, _ in ranked
            ),
            trajectory_energies=tuple(
                breakdown.total for _, breakdown in ranked
            ),
            counterfactual_option_id=selected_option.option_id,
            selected_option_id=(
                pending_option.option_id
                if pending_option is not None
                else ""
            ),
            selected_action_name=selected_action_name,
            selected_action_data=selected_action_data,
            applied=applied,
        )
        self._pending = (
            _PendingSemanticDecision(
                option=pending_option,
                record=record,
                predicted_effects=pending_option.asserted_effects,
            )
            if pending_option is not None
            else None
        )
        self._last_record = record
        return Sage12Arbitration(
            action_name=selected_action_name,
            action_data=selected_action_data,
            source=selected_source,
            reason=reason,
            confidence=max(
                0.0,
                min(1.0, selected_trajectory.probability),
            ),
            applied=applied,
            configured_mode=configured.value,
            effective_mode=effective.value,
            scene_signature=graph.signature,
            selected_option_id=selected_option.option_id,
            trajectory_length=selected_trajectory.length,
            energy=selected_energy.total,
            mt_advisory_id=unchanged.mt_advisory_id,
            mt_suggested_action=unchanged.mt_suggested_action,
            mt_score=unchanged.mt_score,
        )

    def select_slot_action(
        self,
        *,
        slots: Sequence[SemanticActionSlot],
        annotations: Sequence[SlotAnnotation],
        initial_state: frozenset[str],
        goal_predicate: str = "level_complete",
        maximum_depth: int | None = None,
    ) -> SemanticActionCandidate | None:
        """Rank candidate-complete slot trajectories and return only step one.

        This offline-facing entry point deliberately does not grant authority
        or execute an environment action.  Callers retain the normal SAGE12
        arbitration boundary around the returned receding-horizon candidate.
        """
        compilation = self.compiler.compile_slots(
            slots,
            annotations=annotations,
        )
        trajectories = self.world_model.rollout(
            initial_state=initial_state,
            options=compilation.options,
            maximum_depth=(
                self.config.maximum_depth
                if maximum_depth is None
                else max(1, int(maximum_depth))
            ),
            beam_width=self.config.beam_width,
        )
        scored = tuple(
            (
                trajectory,
                self.energy.score(
                    trajectory,
                    goal_predicate=goal_predicate,
                ),
            )
            for trajectory in trajectories
        )
        ranked = self._rank(scored)
        if not ranked:
            return None
        first = ranked[0][0].first_option
        return SemanticActionCandidate(
            action_name=first.action_name,
            action_data=dict(first.action_data),
        )

    def observe_transition(self, record: TransitionRecord) -> None:
        if self.transformation_advisor is not None:
            try:
                self.transformation_advisor.observe_transition(record)
            except Exception:  # noqa: BLE001 - SAGE-MT is fail-closed
                # SAGE-MT is advisory-only in V4.16 and must not interfere
                # with observed evidence updates in the protected planner.
                self._mt_failures += 1
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        observed = observed_semantic_effects(record)
        self.world_model.observe(pending.option, observed)
        self.memory.observe(pending.predicted_effects, observed)
        productive = bool(
            not record.diff.is_noop
            or record.diff.level_complete
        )
        unsafe = bool(record.diff.game_over)
        self._observed_outcomes += 1
        self._productive_outcomes += int(productive)
        self._unsafe_outcomes += int(unsafe)
        completed = pending.record.with_outcome(
            observed_effects=tuple(observed),
            productive=productive,
            unsafe=unsafe,
        )
        self._last_record = completed
        if self.trace_writer is not None:
            self.trace_writer.append(completed)

    def start_branch(self) -> None:
        self._branch_index += 1
        self._probed_contexts.clear()
        self._pending = None
        self.memory.start_branch()
        if self.transformation_advisor is not None:
            try:
                self.transformation_advisor.start_branch()
            except Exception:  # noqa: BLE001 - SAGE-MT is fail-closed
                self._mt_failures += 1

    def summary(self) -> Mapping[str, Any]:
        return {
            "configured_mode": self.configured_mode.value,
            "effective_mode": self.effective_mode.value,
            "evaluations": self._evaluations,
            "applied": self._applied,
            "shadow_evaluations": self._shadow,
            "parse_failures": self._parse_failures,
            "rejected_proposals": self._rejected,
            "danger_vetoes": self._danger_vetoes,
            "protected_competence_blocks": self._protected_blocks,
            "gate_downgrades": self._gate_downgrades,
            "observed_outcomes": self._observed_outcomes,
            "productive_outcomes": self._productive_outcomes,
            "unsafe_outcomes": self._unsafe_outcomes,
            "world_model": self.world_model.summary(),
            "semantic_memory": self.memory.snapshot(),
            "last_record_digest": (
                self._last_record.digest if self._last_record else ""
            ),
            "sage_mt": (
                {
                    **dict(self.transformation_advisor.summary()),
                    "controller_failures": self._mt_failures,
                }
                if self.transformation_advisor is not None
                else {
                    "mode": "off",
                    "evaluations": 0,
                    "observations": 0,
                }
            ),
        }

    def _advise_transformations(
        self,
        *,
        observation: GameObservation,
        candidates: Sequence[SemanticActionCandidate],
        executed_action_name: str,
        executed_action_data: Mapping[str, Any],
    ) -> Any | None:
        if self.transformation_advisor is None:
            return None
        try:
            return self.transformation_advisor.advise(
                observation=observation,
                candidates=candidates,
                executed_action_name=executed_action_name,
                executed_action_data=executed_action_data,
            )
        except Exception:  # noqa: BLE001 - SAGE-MT is fail-closed
            self._mt_failures += 1
            return None

    def _rank(
        self,
        scored: Sequence[tuple[SemanticTrajectory, EnergyBreakdown]],
    ) -> Tuple[tuple[SemanticTrajectory, EnergyBreakdown], ...]:
        if (
            self.config.use_learned_energy
            and self.learned_energy is not None
            and getattr(self.learned_energy, "trained_pairs", 0) > 0
        ):
            learned = self.learned_energy.energies(
                [breakdown.features() for _, breakdown in scored]
            )
            ranked = [
                (trajectory, replace(breakdown, total=float(value)))
                for (trajectory, breakdown), value in zip(scored, learned)
            ]
        else:
            ranked = list(scored)
        ranked.sort(
            key=lambda item: (
                item[1].total,
                item[0].first_option.option_id,
            )
        )
        return tuple(ranked)


def _select_subgoal(
    subgoals: Sequence[HierarchicalSubgoal] | None,
) -> HierarchicalSubgoal:
    if not subgoals:
        return HierarchicalSubgoal(
            goal_id="complete_level",
            predicate="level_complete",
        )
    return min(
        subgoals,
        key=lambda goal: (-float(goal.priority), int(goal.depth), goal.goal_id),
    )


def _normalize_candidates(
    candidates: Sequence[Any],
    *,
    include: SemanticActionCandidate,
) -> Tuple[SemanticActionCandidate, ...]:
    normalized = [include]
    for candidate in candidates:
        if isinstance(candidate, SemanticActionCandidate):
            normalized.append(candidate)
        elif isinstance(candidate, str):
            normalized.append(SemanticActionCandidate(candidate, {}))
        else:
            name = getattr(candidate, "action_name", None)
            if name is not None:
                normalized.append(
                    SemanticActionCandidate(
                        str(name),
                        dict(getattr(candidate, "action_data", {}) or {}),
                    )
                )
    by_key = {candidate.key: candidate for candidate in normalized}
    return tuple(by_key[key] for key in sorted(by_key))


def _action_key(action_name: str, action_data: Mapping[str, Any]) -> str:
    return (
        str(action_name).strip().upper()
        + ":"
        + json.dumps(
            {
                str(key): value
                for key, value in dict(action_data or {}).items()
                if value is not None
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


__all__ = [
    "HierarchicalSubgoal",
    "Sage12Arbitration",
    "Sage12Config",
    "Sage12Mode",
    "SemanticActionCandidate",
    "SemanticPlanningController",
]
