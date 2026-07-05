"""V4 adaptive reasoning agent."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .composition import ForgettingManager, MotifComposer, PrefixCompressor, Ritualizer
from .dissent import DissentController
from .execution import ActionExecutor, EmergencyReset, ReactiveController
from .memory import CrossGameMemoryV4, FastMemory, GameMemoryV4
from .ontology import AffordanceReframer, OntologyCompetition
from .schemas import ActionIntent, ObservationV4, PrimitiveAction, SurpriseField, TransitionRecord
from .sensorium import FrameDiffer, ObjectTracker, SurpriseFieldBuilder, TopologyMonitor
from .strategy import Arbiter, OperatorSearcher, ProjectGenerator, ProjectMarket, create_default_minds

logger = logging.getLogger(__name__)

INDUCTION_INTERVAL = 4
LAW_INTERVAL = 3
FORGET_INTERVAL = 20
TIME_BUDGET = 60.0


@dataclass
class MemoryBundle:
    fast: FastMemory
    game: GameMemoryV4
    cross_game: Optional[CrossGameMemoryV4] = None
    ontology_competition: Optional[OntologyCompetition] = None


class Sensorium:
    """Orchestrate low-level V4 perception."""

    def __init__(self) -> None:
        self.object_tracker = ObjectTracker()
        self.frame_differ = FrameDiffer()
        self.topology_monitor = TopologyMonitor()
        self.surprise_builder = SurpriseFieldBuilder()

    def observe(
        self,
        prev_obs: ObservationV4 | None,
        prev_frame: np.ndarray | None,
        grid: np.ndarray,
        available_actions: list[str],
        game_state: str,
        levels_completed: int,
        predicted_effects,
    ) -> ObservationV4:
        objects = self.object_tracker.extract(grid)
        players = self.object_tracker.player_hypotheses(prev_obs, objects)

        frame_diff = None
        if prev_frame is not None and prev_obs is not None:
            prev_player = prev_obs.best_player.position if prev_obs.best_player else None
            next_player = players[0].position if players else None
            frame_diff = self.frame_differ.diff(
                prev_frame,
                grid,
                prev_obs.objects,
                objects,
                prev_player,
                next_player,
                game_state,
                prev_obs.levels_completed,
                levels_completed,
            )

        obs = ObservationV4(
            raw_grid=grid,
            grid_hash=hash(grid.tobytes()),
            game_state=game_state,
            levels_completed=levels_completed,
            available_actions=available_actions,
            objects=objects,
            player_hypotheses=players,
            frame_diff=frame_diff,
            surprise=SurpriseField(),
        )
        obs.topology = self.topology_monitor.analyze(obs)
        obs.affordances = self._build_affordances(obs)
        obs.local_contexts = self._build_local_contexts(obs)
        topology_delta = len(obs.topology.unlocked_regions) / max(len(obs.topology.reachable_regions), 1)
        semantic_novelty = self._semantic_novelty(prev_obs, obs)
        obs.surprise = self.surprise_builder.build(
            predicted_effects=predicted_effects,
            observed_diff=frame_diff,
            topology_delta=topology_delta,
            semantic_novelty=semantic_novelty,
        )
        return obs

    def make_transition(
        self,
        prev_obs: ObservationV4,
        action: PrimitiveAction,
        obs: ObservationV4,
        predicted_effects,
        last_intent: ActionIntent | None,
        last_operator_id: str | None,
    ) -> TransitionRecord:
        if obs.frame_diff is None:
            raise ValueError("Cannot create a transition without a frame diff.")

        metadata = {
            "obs_before": prev_obs,
            "obs_after": obs,
            "project_id": last_intent.project_id if last_intent else None,
            "project_kind": last_intent.metadata.get("project_kind") if last_intent else None,
            "ontology_id": last_intent.ontology_id if last_intent else None,
            "ontology_kind": last_intent.metadata.get("ontology_kind") if last_intent else None,
            "source": last_intent.source if last_intent else "unknown",
            "phase": last_intent.metadata.get("phase") if last_intent else None,
            "mind_name": last_intent.metadata.get("mind_name") if last_intent else None,
            "bandit_signature": last_intent.metadata.get("bandit_signature") if last_intent else None,
            "ritual_id": last_intent.metadata.get("ritual_id") if last_intent else None,
            "source_bridge": last_intent.metadata.get("source_bridge") if last_intent else None,
            "operator_id": last_operator_id,
            "removed_values": self._removed_values(prev_obs, obs),
            "phase_shift": obs.frame_diff.num_changed >= 8,
            "prefix_replayed": bool(
                last_intent
                and (
                    last_intent.metadata.get("ritual_id")
                    or last_intent.metadata.get("project_kind") == "replay_prefix_then_deviate"
                )
            ),
            "closure_signal": sum(
                1 for obj in obs.objects if obj.value != 0 and obj.area <= 12
            ) <= 4,
        }
        if action.x is not None:
            y = int(action.y if action.y is not None else 0)
            x = int(action.x)
            if 0 <= y < prev_obs.raw_grid.shape[0] and 0 <= x < prev_obs.raw_grid.shape[1]:
                metadata["click_value_before"] = int(prev_obs.raw_grid[y, x])

        predicted_ok = self._predicted_ok(predicted_effects, obs.frame_diff)
        metadata["predicted_ok"] = predicted_ok
        return TransitionRecord(
            prev_hash=prev_obs.grid_hash,
            next_hash=obs.grid_hash,
            action=action,
            diff=obs.frame_diff,
            surprise=obs.surprise,
            level_completed=obs.frame_diff.level_complete,
            game_over=obs.frame_diff.game_over,
            metadata=metadata,
        )

    def _build_affordances(self, obs: ObservationV4) -> list[dict[str, Any]]:
        affordances: list[dict[str, Any]] = []
        player = obs.best_player
        for obj in obs.objects[:20]:
            entry = {
                "value": obj.value,
                "position": (int(round(obj.center[0])), int(round(obj.center[1]))),
                "area": obj.area,
                "clickable": obj.area <= 10,
                "countable": True,
                "salience": 1.0 / max(obj.area, 1),
            }
            if player is not None:
                entry["distance"] = abs(player.position[0] - entry["position"][0]) + abs(
                    player.position[1] - entry["position"][1]
                )
            affordances.append(entry)
        return affordances

    def _build_local_contexts(self, obs: ObservationV4) -> list[dict[str, Any]]:
        player = obs.best_player
        if player is None:
            return []
        r, c = player.position
        grid = obs.raw_grid
        cells = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    cells.append({"offset": (dr, dc), "value": int(grid[nr, nc])})
        return [{"player": player.position, "cells": cells}]

    def _removed_values(self, prev_obs: ObservationV4, obs: ObservationV4) -> dict[int, int]:
        before: dict[int, int] = {}
        after: dict[int, int] = {}
        for obj in prev_obs.objects:
            before[obj.value] = before.get(obj.value, 0) + 1
        for obj in obs.objects:
            after[obj.value] = after.get(obj.value, 0) + 1
        removed = {}
        for value, count in before.items():
            delta = count - after.get(value, 0)
            if delta > 0:
                removed[value] = delta
        return removed

    def _semantic_novelty(self, prev_obs: ObservationV4 | None, obs: ObservationV4) -> float:
        if prev_obs is None:
            return 0.0
        before_values = {obj.value for obj in prev_obs.objects}
        after_values = {obj.value for obj in obs.objects}
        novel = len(after_values - before_values)
        return min(1.0, novel / 5.0)

    def _predicted_ok(self, predicted_effects, diff) -> bool:
        if diff is None:
            return False
        effects = list(predicted_effects or [])
        if not effects:
            return not diff.is_noop
        matched = 0
        for effect in effects:
            if effect.kind == "player_displacement" and effect.args.get("delta") == diff.player_displacement:
                matched += 1
            elif effect.kind == "noop" and diff.is_noop:
                matched += 1
            elif effect.kind == "grid_change" and diff.num_changed >= int(effect.args.get("min_cells", 1)):
                matched += 1
            elif effect.kind == "game_over" and diff.game_over:
                matched += 1
        return matched > 0


class StrategyChamber:
    """Project-market-driven strategy chamber."""

    def __init__(self) -> None:
        self.generator = ProjectGenerator()
        self.market = ProjectMarket()
        self.minds = {mind.name: mind for mind in create_default_minds()}
        self.arbiter = Arbiter()
        self.searcher = OperatorSearcher()
        self.reactive = ReactiveController()

    def propose_control(self, obs: ObservationV4, memory: MemoryBundle, phase: str) -> ActionIntent:
        new_projects = self.generator.generate(obs, memory)
        self.market.update(new_projects, memory, obs)
        memory.game.project_market = self.market

        ranked = self.market.ranked()[:4]
        proposals = []
        for project in ranked:
            for mind in self.minds.values():
                proposals.extend(mind.propose(obs, memory, project))

        chosen = self.arbiter.select(proposals, memory) if proposals else None
        if chosen is not None:
            memory.game.record_project_selection(chosen.project)
            operator_plan = self.searcher.search(obs, chosen, memory)
            metadata = dict(chosen.project.metadata)
            metadata.update(chosen.metadata)
            metadata["project_kind"] = chosen.project.kind
            metadata["phase"] = phase
            metadata["mind_name"] = chosen.mind_name
            metadata["ontology_kind"] = next(
                (
                    ontology.kind
                    for ontology in memory.game.current_ontologies
                    if ontology.ontology_id == chosen.project.ontology_id
                ),
                memory.game.current_ontologies[0].kind if memory.game.current_ontologies else "unknown",
            )
            return ActionIntent(
                source=chosen.mind_name,
                primitive_plan=chosen.primitive_plan,
                operator_plan=operator_plan,
                project_id=chosen.project.project_id,
                ontology_id=chosen.project.ontology_id,
                metadata=metadata,
            )

        reactive_action = self.reactive.act(obs, memory)
        return ActionIntent(
            source="reactive",
            primitive_plan=[reactive_action],
            ontology_id=memory.game.current_ontologies[0].ontology_id if memory.game.current_ontologies else None,
            metadata={
                "project_kind": "reactive",
                "phase": phase,
                "ontology_kind": (
                    memory.game.current_ontologies[0].kind if memory.game.current_ontologies else "unknown"
                ),
            },
        )


class AdaptiveReasoningAgentV4:
    """World-constituting adaptive reasoner with competing chambers."""

    def __init__(
        self,
        cross_game: Optional[CrossGameMemoryV4] = None,
        time_budget: float = TIME_BUDGET,
        freeze_transfer: bool = False,
        freeze_learning_updates: bool = False,
        progress_profile: str = "strict_plus",
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> None:
        self.sensorium = Sensorium()
        self.ontology = OntologyCompetition()
        self.affordance_reframer = AffordanceReframer()
        self.strategy = StrategyChamber()
        self.dissent = DissentController()
        self.motif_composer = MotifComposer()
        self.prefix_compressor = PrefixCompressor()
        self.ritualizer = Ritualizer()
        self.forgetting = ForgettingManager()
        self.executor = ActionExecutor()
        self.emergency_reset = EmergencyReset()

        self.memory = MemoryBundle(
            fast=FastMemory(),
            game=GameMemoryV4(),
            cross_game=cross_game,
            ontology_competition=self.ontology,
        )
        self._freeze_transfer = freeze_transfer
        self._freeze_learning_updates = freeze_learning_updates
        self._progress_profile = progress_profile
        self._diagnostics = dict(diagnostics or {})
        self._diagnostics_enabled = bool(self._diagnostics.get("enabled", False))
        self._diagnostic_dump_path = self._diagnostics.get("dump_path")
        self._assert_causal_chain = bool(self._diagnostics.get("assert_causal_chain", False))
        self._diagnostic_transition_limit = self._diagnostics.get("transition_limit")
        self._transition_diagnostics: list[dict[str, Any]] = []
        self._assertion_failures: list[str] = []
        self.memory.game.current_ontologies = []
        self.memory.game.progress.set_profile(progress_profile)
        if cross_game is not None and not self._freeze_transfer:
            cross_game.seed_game(self.memory)

        self._time_budget = time_budget
        self._start_time: Optional[float] = None
        self._level_trace: list[PrimitiveAction] = []
        self._intent_counts: dict[str, int] = {}
        self._visited_hashes: set[int] = set()

    def reset_game(self) -> None:
        self.__init__(
            cross_game=self.memory.cross_game,
            time_budget=self._time_budget,
            freeze_transfer=self._freeze_transfer,
            freeze_learning_updates=self._freeze_learning_updates,
            progress_profile=self._progress_profile,
            diagnostics=self._diagnostics,
        )

    def choose_action(
        self,
        frames: list[np.ndarray],
        available_actions: list[str],
        game_state: str,
        levels_completed: int,
    ) -> dict[str, Any]:
        if self._start_time is None:
            self._start_time = time.time()

        grid = frames[-1] if frames else np.zeros((10, 10), dtype=np.int32)
        predicted_effects = self._predicted_effects()
        obs = self.sensorium.observe(
            prev_obs=self.memory.fast.prev_obs,
            prev_frame=self.memory.fast.prev_frame,
            grid=np.array(grid, dtype=np.int32),
            available_actions=available_actions,
            game_state=game_state,
            levels_completed=levels_completed,
            predicted_effects=predicted_effects,
        )
        self.memory.fast.on_observation(obs)

        if self.memory.fast.prev_obs is not None and self.memory.fast.last_action is not None and obs.frame_diff is not None:
            self._record_transition(obs)

        self.ontology.update(obs, self.memory)
        self.memory.game.current_ontologies = self.ontology.ranked()
        top = self.memory.game.current_ontologies[0] if self.memory.game.current_ontologies else None
        if top is not None:
            weights = self.affordance_reframer.reweight(obs, top)
            if self.memory.game.total_actions < 25:
                weights = {key: 1.0 + (value - 1.0) * 0.30 for key, value in weights.items()}
            elif self.memory.game.total_actions < 50:
                weights = {key: 1.0 + (value - 1.0) * 0.60 for key, value in weights.items()}
            top.active_affordance_biases = weights

        previous_frame = self.memory.game.current_frame
        current_frame = self.memory.game.frame_memory.observe(obs, self.memory)
        self.memory.game.previous_frame = previous_frame
        self.memory.game.current_frame = current_frame
        if self.memory.fast.last_transition is not None:
            self.memory.game.bridge_memory.observe_shift(
                previous_frame,
                current_frame,
                sp_gain=float(self.memory.fast.last_transition.metadata.get("sp_delta", 0.0)),
                tp_gain=float(self.memory.fast.last_transition.metadata.get("tp_delta", 0.0)),
                loop_warning=self.dissent.loop_critic.analyze(self.memory),
            )

        phase = self.memory.game.phase_controller.select_phase(obs, self.memory)
        self.memory.fast.current_phase = phase
        self.memory.game.record_phase(phase)

        if self.memory.game.total_actions > 0 and self.memory.game.total_actions % FORGET_INTERVAL == 0:
            self.forgetting.decay(self.memory)

        if self.emergency_reset.should_reset(self.memory):
            if not self._freeze_learning_updates:
                self.memory.game.learning.credit.on_branch_reset(self.memory)
            self.memory.game.branch_scheduler.on_branch_kill(self.memory)
            self.emergency_reset.mark_reset(self.memory)
            intent = ActionIntent(source="reset", primitive_plan=[PrimitiveAction("RESET")])
        elif self.memory.fast.has_plan():
            intent = None
        else:
            report = self.dissent.update(obs, self.memory)
            if self.dissent.should_interrupt(self.memory):
                intent = self.dissent.interrupt_and_redirect(obs, self.memory)
            else:
                intent = self.strategy.propose_control(obs, self.memory, phase)

        primitive = self.executor.act(intent, obs, self.memory)
        self._intent_counts[self.memory.fast.last_intent.source] = (
            self._intent_counts.get(self.memory.fast.last_intent.source, 0) + 1
        )

        self.memory.fast.prev_frame = obs.raw_grid.copy()
        self.memory.fast.prev_obs = obs
        self._visited_hashes.add(obs.grid_hash)
        self._level_trace.append(primitive)

        result: dict[str, Any] = {"action": primitive.name}
        if primitive.x is not None:
            result["x"] = primitive.x
            result["y"] = primitive.y
        return result

    def end_game(self, won: bool) -> dict[str, Any]:
        if not self._freeze_learning_updates:
            self.memory.game.learning.world_embedding.observe_episode(self.memory, won)
            self.memory.game.learning.credit.on_game_end(self.memory, won)
        if self.memory.cross_game is not None and not self._freeze_transfer:
            self.memory.cross_game.export_game(self.memory, won=won)

        self._flush_diagnostics(won)

        return {
            "total_actions": self.memory.game.total_actions,
            "levels_completed": self.memory.game.total_levels_completed,
            "operators": len(self.memory.game.inducer.operators),
            "rules": len(self.memory.game.constraints.rules),
            "teleology": len(self.memory.game.teleology.hypotheses()),
            "motifs": len(self.memory.game.motifs),
            "rituals": len(self.memory.game.rituals),
            "states_visited": len(self._visited_hashes),
            "knowledge_level": round(self.memory.game.knowledge_level(), 3),
            "pred_accuracy": round(self.memory.game.inducer.operator_predictive_accuracy(), 3),
            "control_success": round(self.memory.game.inducer.operator_control_success(), 3),
            "progress": self.memory.game.progress.summary(),
            "mind_selections": dict(self.strategy.arbiter.selections),
            "intent_counts": dict(self._intent_counts),
            "phase_counts": self._phase_counts(),
            "learning": {
                "world_reliability": round(
                    self.memory.game.learning.world_reliability.estimate(self.memory), 3
                ),
                "sterility_risk": round(
                    self.memory.game.learning.sterility_predictor.predict(self.memory), 3
                ),
                "frame_embeddings": len(self.memory.game.learning.world_embedding.frame_prototypes),
                "episode_embeddings": len(self.memory.game.learning.world_embedding.episode_prototypes),
            },
            "current_ontologies": [
                (item.kind, round(item.confidence, 3))
                for item in self.memory.game.current_ontologies[:3]
            ],
            "diagnostics": {
                "enabled": self._diagnostics_enabled,
                "freeze_transfer": self._freeze_transfer,
                "freeze_learning_updates": self._freeze_learning_updates,
                "records": len(self._transition_diagnostics),
                "assertion_failures": list(self._assertion_failures),
                "dump_path": self._diagnostic_dump_path,
            },
            "won": won,
        }

    def _record_transition(self, obs: ObservationV4) -> None:
        prev_obs = self.memory.fast.prev_obs
        action = self.memory.fast.last_action
        if prev_obs is None or action is None:
            return

        predicted_effects = self._predicted_effects()
        transition = self.sensorium.make_transition(
            prev_obs=prev_obs,
            action=action,
            obs=obs,
            predicted_effects=predicted_effects,
            last_intent=self.memory.fast.last_intent,
            last_operator_id=self.memory.fast.last_operator_id,
        )

        ontology_id = self.memory.fast.current_ontology_id or (
            self.memory.game.current_ontologies[0].ontology_id
            if self.memory.game.current_ontologies else "unknown"
        )
        if not transition.metadata.get("ontology_kind"):
            transition.metadata["ontology_kind"] = next(
                (
                    ontology.kind
                    for ontology in self.memory.game.current_ontologies
                    if ontology.ontology_id == ontology_id
                ),
                str(ontology_id),
            )
        if not transition.metadata.get("phase"):
            transition.metadata["phase"] = self.memory.fast.current_phase
        is_reset_transition = action.name == "RESET"
        if not is_reset_transition:
            self.memory.game.profiler.update(prev_obs, action, transition, ontology_id)

            if self.memory.fast.last_operator_id is not None:
                self.memory.game.inducer.record_validation(
                    self.memory.fast.last_operator_id,
                    bool(transition.metadata.get("predicted_ok")),
                    not transition.diff.is_noop,
                )

        structure_before = self._structure_counts()
        project_before = self._project_snapshot(str(transition.metadata.get("project_id") or ""))
        market_before = self._market_snapshot()
        prev_lp, prev_sp, prev_tp = self.memory.game.progress.scores()
        progress_updates_before = self.memory.game.progress.num_updates
        induction_called = False
        law_update_called = False

        if (
            not is_reset_transition
            and (
                self.memory.game.total_actions == 0
                or self.memory.game.total_actions % INDUCTION_INTERVAL == 0
            )
        ):
            induction_called = True
            self.memory.game.inducer.induce(self.memory.game.profiler, self.memory)

        rules_added = []
        if (
            not is_reset_transition
            and (
                self.memory.game.total_actions == 0
                or self.memory.game.total_actions % LAW_INTERVAL == 0
            )
        ):
            law_update_called = True
            rules_added = self.memory.game.constraints.update(
                obs,
                transition,
                self.memory.game.inducer,
            )
        transition.metadata["new_rules"] = len(rules_added)
        if is_reset_transition:
            validated_hits, speculative_hits = 0, 0
        else:
            validated_hits, speculative_hits = self.memory.game.teleology.evidence_hits(transition, obs)
        transition.metadata["validated_teleology_hits"] = validated_hits
        transition.metadata["speculative_teleology_hits"] = speculative_hits

        self.memory.fast.on_transition(transition)
        self.memory.game.on_transition(transition)
        self.memory.game.progress.on_transition(obs, transition, self.memory)

        lp, sp, tp = self.memory.game.progress.scores()
        transition.metadata["lp_delta"] = max(0.0, lp - prev_lp)
        transition.metadata["sp_delta"] = max(0.0, sp - prev_sp)
        transition.metadata["tp_delta"] = max(0.0, tp - prev_tp)
        if not is_reset_transition:
            self.memory.game.teleology.update(obs, self.memory)
            self.memory.game.laws.update(
                list(self.memory.game.inducer.operators.values()),
                list(self.memory.game.constraints.rules.values()),
                self.memory.game.teleology.hypotheses(),
            )
        project_outcome = None
        if self.memory.game.project_market is not None and not is_reset_transition:
            project_outcome = self._apply_project_outcome(transition)
            self.memory.game.project_market.mark_feedback_applied(transition)

        if not self._freeze_learning_updates and not is_reset_transition:
            self.memory.game.learning.credit.on_transition(self.memory, transition)

        source = transition.metadata.get("source")
        if source in self.strategy.minds and not is_reset_transition:
            progress_gain = transition.metadata["lp_delta"] + transition.metadata["sp_delta"] + 2.0 * transition.metadata["tp_delta"]
            self.memory.game.record_mind_outcome(
                source,
                bool(transition.metadata.get("predicted_ok")),
                progress_gain,
            )

        if not is_reset_transition:
            self.motif_composer.update(self.memory.fast.recent_transitions, self.memory)
        prune_info = self.memory.game.enforce_budgets(self.memory)
        structure_after = self._structure_counts()
        project_after = self._project_snapshot(str(transition.metadata.get("project_id") or ""))
        market_after = self._market_snapshot()
        progress_diag = self.memory.game.progress.diagnostics()

        if transition.level_completed:
            compressed = self.prefix_compressor.compress(self._level_trace)
            self.memory.game.successful_trace = compressed
            self.memory.game.all_traces.append(compressed)
            self.ritualizer.compile(self.memory, compressed)
            self._level_trace = []

        if self._diagnostics_enabled:
            record = self._build_transition_diagnostic(
                transition=transition,
                progress_updates_before=progress_updates_before,
                induction_called=induction_called,
                law_update_called=law_update_called,
                rules_added=rules_added,
                structure_before=structure_before,
                structure_after=structure_after,
                project_before=project_before,
                project_after=project_after,
                project_outcome=project_outcome,
                market_before=market_before,
                market_after=market_after,
                prune_info=prune_info,
                progress_diag=progress_diag,
            )
            if (
                self._diagnostic_transition_limit is None
                or len(self._transition_diagnostics) < int(self._diagnostic_transition_limit)
            ):
                self._transition_diagnostics.append(record)
            if self._assert_causal_chain:
                self._assert_transition_diagnostic(record)

    def _predicted_effects(self):
        operator_id = self.memory.fast.last_operator_id
        if operator_id is None:
            return []
        operator = self.memory.game.inducer.operators.get(operator_id)
        return operator.expected_effects if operator is not None else []

    def _phase_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for phase in self.memory.game.phase_history:
            counts[phase] = counts.get(phase, 0) + 1
        return counts

    def _structure_counts(self) -> dict[str, int]:
        return {
            "operators": len(self.memory.game.inducer.operators),
            "rules": len(self.memory.game.constraints.rules),
            "validated_teleology": len(self.memory.game.teleology.hypotheses()),
            "speculative_teleology": len(self.memory.game.teleology.speculative_hypotheses()),
            "projects": len(self.memory.game.project_market.projects)
            if self.memory.game.project_market is not None else 0,
            "motifs": len(self.memory.game.motifs),
            "rituals": len(self.memory.game.rituals),
        }

    def _project_snapshot(self, project_id: str) -> dict[str, Any]:
        if not project_id or self.memory.game.project_market is None:
            return {"exists": False}
        return self.memory.game.project_market.snapshot(project_id)

    def _market_snapshot(self) -> dict[str, Any]:
        if self.memory.game.project_market is None:
            return {"top": []}
        return self.memory.game.project_market.snapshot(limit=3)

    def _apply_project_outcome(self, transition: TransitionRecord) -> dict[str, Any]:
        project_id = str(transition.metadata.get("project_id", "") or "")
        if not project_id or self.memory.game.project_market is None:
            return {"project_id": project_id, "applied": False}

        before = self.memory.game.project_market.snapshot(project_id)
        self.memory.game.project_market.apply_outcome(
            project_id,
            float(transition.metadata.get("sp_delta", 0.0)),
            float(transition.metadata.get("tp_delta", 0.0)),
        )
        after = self.memory.game.project_market.snapshot(project_id)
        return {
            "project_id": project_id,
            "applied": before.get("exists", False) or after.get("exists", False),
            "before": before,
            "after": after,
        }

    def _build_transition_diagnostic(
        self,
        transition: TransitionRecord,
        progress_updates_before: int,
        induction_called: bool,
        law_update_called: bool,
        rules_added: list[Any],
        structure_before: dict[str, int],
        structure_after: dict[str, int],
        project_before: dict[str, Any],
        project_after: dict[str, Any],
        project_outcome: dict[str, Any] | None,
        market_before: dict[str, Any],
        market_after: dict[str, Any],
        prune_info: dict[str, Any],
        progress_diag: dict[str, Any],
    ) -> dict[str, Any]:
        justification_flags = {
            "induction_called": induction_called,
            "law_update_called": law_update_called,
            "rules_added": len(rules_added),
            "teleology_updated": True,
            "motif_gain_signal": (
                float(transition.metadata.get("sp_delta", 0.0)) > 0.0
                or float(transition.metadata.get("tp_delta", 0.0)) > 0.0
            ),
            "level_completed": transition.level_completed,
            "project_outcome_applied": bool(project_outcome and project_outcome.get("applied")),
            "learning_update_applied": not self._freeze_learning_updates,
            "transfer_update_applied": False,
        }
        deltas = {
            key: structure_after.get(key, 0) - structure_before.get(key, 0)
            for key in sorted(set(structure_before) | set(structure_after))
        }
        record = {
            "step": self.memory.game.total_actions,
            "action": repr(transition.action),
            "transition": {
                "prev_hash": transition.prev_hash,
                "next_hash": transition.next_hash,
                "num_changed": transition.diff.num_changed,
                "is_noop": transition.diff.is_noop,
                "game_over": transition.diff.game_over,
                "level_completed": transition.level_completed,
                "player_displacement": transition.diff.player_displacement,
                "predicted_ok": bool(transition.metadata.get("predicted_ok")),
                "changed_cells": len(transition.diff.changed_cells),
            },
            "progress": {
                "updates_before": progress_updates_before,
                "updates_after": self.memory.game.progress.num_updates,
                "diagnostic": progress_diag,
            },
            "project": {
                "selected_id": transition.metadata.get("project_id"),
                "before": project_before,
                "after": project_after,
                "outcome": project_outcome or {"applied": False},
                "market_before": market_before,
                "market_after": market_after,
            },
            "pruning": prune_info,
            "memory": {
                "before": structure_before,
                "after": structure_after,
                "deltas": deltas,
                "justifications": justification_flags,
            },
            "freeze": {
                "learning_updates": self._freeze_learning_updates,
                "transfer": self._freeze_transfer,
            },
        }
        return _json_safe(record)

    def _assert_transition_diagnostic(self, record: dict[str, Any]) -> None:
        messages: list[str] = []
        progress = record["progress"]
        memory = record["memory"]
        project = record["project"]

        if progress["updates_after"] != progress["updates_before"] + 1:
            messages.append("progress tracker was not updated exactly once")

        diagnostic = progress.get("diagnostic", {})
        if diagnostic.get("step") != record["step"]:
            messages.append("progress diagnostic step does not match transition step")

        project_id = project.get("selected_id")
        if project_id:
            outcome = project.get("outcome", {})
            before = outcome.get("before", {})
            after = outcome.get("after", {})
            changed = (
                before.get("dignity") != after.get("dignity")
                or before.get("score") != after.get("score")
                or not after.get("exists", True)
            )
            if not changed:
                messages.append("project outcome did not change score/dignity or trigger pruning")

        deltas = memory.get("deltas", {})
        justifications = memory.get("justifications", {})
        if deltas.get("operators", 0) > 0 and not justifications.get("induction_called"):
            messages.append("operators increased without induction")
        if deltas.get("rules", 0) > 0 and int(justifications.get("rules_added", 0)) <= 0:
            messages.append("rules increased without rule evidence")
        if deltas.get("motifs", 0) > 0 and not justifications.get("motif_gain_signal"):
            messages.append("motifs increased without progress signal")
        if deltas.get("rituals", 0) > 0 and not justifications.get("level_completed"):
            messages.append("rituals increased without level completion")

        if messages:
            self._assertion_failures.extend(messages)
            raise AssertionError("; ".join(messages))

    def _flush_diagnostics(self, won: bool) -> None:
        if not self._diagnostics_enabled or not self._diagnostic_dump_path:
            return
        dump_path = Path(str(self._diagnostic_dump_path))
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "won": won,
            "total_actions": self.memory.game.total_actions,
            "levels_completed": self.memory.game.total_levels_completed,
            "knowledge_level": round(self.memory.game.knowledge_level(), 3),
            "freeze_transfer": self._freeze_transfer,
            "freeze_learning_updates": self._freeze_learning_updates,
            "assertion_failures": self._assertion_failures,
            "progress_summary": self.memory.game.progress.summary(),
            "current_ontologies": [
                (item.kind, round(item.confidence, 3))
                for item in self.memory.game.current_ontologies[:3]
            ],
            "records": self._transition_diagnostics,
        }
        dump_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
