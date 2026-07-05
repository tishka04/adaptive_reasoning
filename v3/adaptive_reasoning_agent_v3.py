"""V3 Adaptive Reasoning Agent — operator-centric, mechanic-inducing.

Main runtime loop:
  1. Perceive → structured observation
  2. Update beliefs (profiler, operators, rules)
  3. Decide mode: experiment / search / reactive
  4. Execute via operator→primitive translation
  5. Record transition, compress if solved

This replaces the V2 ε-based exploration/exploitation loop with a
semantics-driven controller that asks "do I need information, search,
or execution right now?"
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    FrameDiff,
    GameObservation,
    MacroAction,
    MindProposal,
    OperatorCall,
    PrimitiveAction,
    SolvedTrajectory,
    TransitionRecord,
)
from .perception.frame_diff import compute_frame_diff, diff_summary
from .perception.object_extractor import (
    extract_objects,
    generate_player_hypotheses,
)
from .perception.affordance_mapper import map_affordances, build_local_contexts
from .mechanics.action_profiler import ActionProfiler
from .mechanics.operator_inducer import OperatorInducer
from .mechanics.experiment_designer import ExperimentDesigner
from .mechanics.rule_engine import RuleEngine
from .memory.game_memory import GameMemoryV3
from .memory.episodic_graph import EpisodicGraph
from .memory.belief_debugger import BeliefDebugger
from .memory.cross_game_memory import CrossGameMemoryV3
from .compression.solution_shortener import SolutionShortener
from .compression.macro_compiler import MacroCompiler
from .control.specialist_minds import SpecialistMind, create_default_minds
from .control.arbiter import Arbiter
from .control.operator_search import OperatorSearcher
from .control.reactive_controller import ReactiveController
from .control.executor import ActionExecutor
from .control.options import Option, create_default_options
from .control.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

# =====================================================================
# Configuration
# =====================================================================

INDUCTION_INTERVAL = 5        # re-induce operators every N actions
RULE_INTERVAL = 10            # re-run rule engine every N actions
BELIEF_AUDIT_INTERVAL = 20    # audit beliefs every N actions
MACRO_COMPILE_INTERVAL = 20   # compile micro-macros every N actions
EXPLOIT_CONFIDENCE = 0.7      # operator confidence for reactive mode
SEARCH_CONFIDENCE = 0.5       # min confidence to run search
TIME_BUDGET = 60.0            # seconds per game


class AdaptiveReasoningAgentV3:
    """V3 agent: discover mechanics → induce operators → search efficiently."""

    def __init__(
        self,
        cross_game: Optional[CrossGameMemoryV3] = None,
        time_budget: float = TIME_BUDGET,
    ) -> None:
        # Memory
        self.memory = GameMemoryV3()
        self.episodic = EpisodicGraph()
        self.cross_game = cross_game

        # Mechanics
        self.profiler = self.memory.profiler
        self.inducer = self.memory.inducer
        self.experiment_designer = ExperimentDesigner()
        self.rule_engine = self.memory.rules

        # Control
        self.minds = {m.name: m for m in create_default_minds()}
        self.arbiter = Arbiter()
        self.searcher = OperatorSearcher()
        self.reactive = ReactiveController(self.inducer, self.rule_engine)
        self.executor = ActionExecutor(self.inducer)

        # Compression
        self.shortener = SolutionShortener()
        self.macro_compiler = MacroCompiler()

        # Options (compositional behaviours built on operators)
        self.options = create_default_options()

        # Belief maintenance
        self.debugger = BeliefDebugger()

        # Three-tier progress tracking + branch killing
        self.progress = ProgressTracker()

        # State
        self._prev_grid: Optional[np.ndarray] = None
        self._prev_obs: Optional[GameObservation] = None
        self._prev_objects: List = []
        self._prev_player_hypotheses: List = []
        self._action_counter: int = 0
        self._current_plan: List[OperatorCall] = []
        self._plan_index: int = 0
        self._current_mind: Optional[str] = None
        self._current_operator_id: Optional[str] = None
        self._level_action_trace: List[PrimitiveAction] = []
        self._prev_levels: int = 0
        self._time_budget = time_budget
        self._start_time: Optional[float] = None
        self._visited_hashes: set = set()

        # Control leverage metrics
        self._mode_counts: Dict[str, int] = {
            "experiment": 0, "exploit": 0, "search": 0, "fallback": 0,
        }
        self._operator_driven_actions: int = 0
        self._fallback_actions: int = 0
        self._plan_driven_actions: int = 0
        self._mind_diversity_penalty: Dict[str, float] = {}

        # Mode cycling: prevent exploit death spiral
        self._exploit_streak: int = 0
        self._actions_since_last_level: int = 0
        self._last_novel_state_action: int = 0  # action# when last novel state seen
        self._induction_ticks: int = 0
        self._bootstrap_warning_emitted: bool = False
        self._debug_action_limit: int = int(os.getenv("V3_DEBUG_ACTIONS", "0"))

        # Seed from cross-game memory
        if cross_game is not None:
            cross_game.seed_game(self.memory)

    # ─── Main entry points ──────────────────────────────────────

    def is_done(self) -> bool:
        """Check if the agent should stop."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= self._time_budget:
                return True
        return False

    def choose_action(
        self,
        frames: List[np.ndarray],
        available_actions: List[str],
        game_state: str,
        levels_completed: int,
    ) -> Dict[str, Any]:
        """Main decision function called each step.

        Returns: {"action": str, "x": int|None, "y": int|None, ...}
        """
        if self._start_time is None:
            self._start_time = time.time()

        latest_frame = frames[-1] if frames else np.zeros((10, 10), dtype=int)

        # ── 1. Perceive ──
        obs = self._perceive(latest_frame, available_actions, game_state,
                             levels_completed)

        # ── 2. Record transition from previous step ──
        if self._prev_obs is not None and self._prev_grid is not None:
            self._record_transition(obs, game_state)
            self._maybe_warn_bootstrap_stall(obs)

        # ── 3. Check for level completion → compress ──
        self._actions_since_last_level += 1
        if levels_completed > self._prev_levels:
            self._on_level_complete(levels_completed)
            self._exploit_streak = 0
            self._actions_since_last_level = 0

        # ── 4. Periodic mechanic inference ──
        if self._action_counter > 0:
            if self._action_counter % INDUCTION_INTERVAL == 0:
                self._induction_ticks += 1
                self.inducer.induce(
                    self.profiler,
                    self.profiler.transitions,
                )
            if self._action_counter % RULE_INTERVAL == 0:
                self.rule_engine.propose_and_verify(
                    self.profiler,
                    self.profiler.transitions,
                )
            if self._action_counter % BELIEF_AUDIT_INTERVAL == 0:
                self.debugger.audit_and_repair(self.memory)
            # Compile micro-macros from action trace (two-tier: ephemeral → promoted)
            if (self._action_counter % MACRO_COMPILE_INTERVAL == 0
                    and len(self._level_action_trace) >= 4
                    and len(self.memory.macros) < 10):
                _, sp, _ = self.progress.scores()
                new_macros = self.macro_compiler.compile_from_trace(
                    self._level_action_trace,
                    control_success_rate=self.inducer.operator_control_success(),
                    structural_gain=sp,
                )
                for m in new_macros:
                    if len(self.memory.macros) >= 10:
                        break
                    self.memory.macros[m.macro_id] = m
                    self.executor.set_macros(self.memory.macros)

        # ── 5. Choose action (the V3 decision logic) ──
        primitive = self._decide(obs)

        # ── 6. Bookkeeping ──
        self._prev_grid = latest_frame.copy()
        self._prev_obs = obs
        self._prev_levels = levels_completed
        self._action_counter += 1
        self.memory.total_actions = self._action_counter
        self._level_action_trace.append(primitive)
        # Track novel state discovery
        if obs.grid_hash not in self._visited_hashes:
            self._last_novel_state_action = self._action_counter
        self._visited_hashes.add(obs.grid_hash)

        result: Dict[str, Any] = {"action": primitive.name}
        if primitive.x is not None:
            result["x"] = primitive.x
            result["y"] = primitive.y

        return result

    # ─── Perception ─────────────────────────────────────────────

    def _perceive(
        self,
        grid: np.ndarray,
        available_actions: List[str],
        game_state: str,
        levels_completed: int,
    ) -> GameObservation:
        """Build structured observation from raw frame."""
        grid_hash = hash(grid.tobytes())
        objects = extract_objects(grid)

        player_hypotheses = generate_player_hypotheses(
            grid, objects,
            prev_hypotheses=self._prev_player_hypotheses,
        )

        # Frame diff
        frame_diff: Optional[FrameDiff] = None
        if self._prev_grid is not None:
            prev_player = (self._prev_player_hypotheses[0].position
                           if self._prev_player_hypotheses else None)
            curr_player = (player_hypotheses[0].position
                           if player_hypotheses else None)
            frame_diff = compute_frame_diff(
                self._prev_grid, grid,
                self._prev_objects, objects,
                prev_player, curr_player,
                game_state, self._prev_levels, levels_completed,
            )

        # Affordances
        player_pos = (player_hypotheses[0].position
                      if player_hypotheses else None)
        affordances = map_affordances(
            objects, grid,
            player_pos=player_pos,
            known_lethal_values=self.memory.lethal_values,
        )

        # Local contexts around player
        local_ctx = []
        if player_pos:
            local_ctx = build_local_contexts(grid, [player_pos])

        obs = GameObservation(
            raw_grid=grid,
            grid_hash=grid_hash,
            game_state=game_state,
            levels_completed=levels_completed,
            available_actions=available_actions,
            objects=objects,
            player_candidates=player_hypotheses,
            affordances=affordances,
            frame_diff=frame_diff,
            local_contexts=local_ctx,
        )

        # Track novel state
        is_novel = self.memory.record_state_visit(grid_hash)

        self._prev_objects = objects
        self._prev_player_hypotheses = player_hypotheses

        return obs

    # ─── Transition recording ───────────────────────────────────

    def _record_transition(
        self, obs: GameObservation, game_state: str
    ) -> None:
        """Record the last transition into profiler + episodic graph."""
        if obs.frame_diff is None or not self._level_action_trace:
            return

        last_action = self._level_action_trace[-1]
        diff = obs.frame_diff
        had_progress = not diff.is_noop
        predicted_ok = False
        active_operator_id = self._current_operator_id
        active_operator = (
            self.inducer.operators.get(active_operator_id)
            if active_operator_id is not None else None
        )
        record = TransitionRecord(
            action=last_action,
            obs_before=self._prev_obs,
            obs_after=obs,
            diff=diff,
            timestamp=self._action_counter,
        )
        self.profiler.record_transition(record)

        # Reactive controller: track effect
        self.reactive.record_effect(had_progress)

        # Operator validation: check if the operator's prediction matched.
        # Synthetic plan IDs such as experiment_1 are not real operators and
        # should not enter the validation pool.
        if active_operator is not None and active_operator_id is not None:
            predicted_ok = True
            if active_operator.kind.name == "MOVE":
                if diff.player_displacement:
                    exp_dy = active_operator.parameters.get("dy", 0)
                    exp_dx = active_operator.parameters.get("dx", 0)
                    act_dy, act_dx = diff.player_displacement
                    predicted_ok = (
                        round(act_dy) == exp_dy and round(act_dx) == exp_dx
                    )
                else:
                    predicted_ok = False
            elif active_operator.kind.name == "NOOP":
                predicted_ok = diff.is_noop
            elif active_operator.kind.name == "GLOBAL_TRANSFORM":
                predicted_ok = diff.num_changed >= 5
            elif active_operator.kind.name == "CLICK":
                predicted_ok = diff.num_changed > 0

            self.inducer.record_validation(
                active_operator_id, predicted_ok, had_progress,
            )

        if self._current_mind and self._current_mind != "reactive":
            self.arbiter.update_mind_outcomes(
                self._current_mind, self.minds,
                predicted_ok if active_operator is not None else had_progress,
                1.0 if had_progress else 0.0,
            )

        # Update mind diversity penalties every 10 actions
        if self._action_counter > 0 and self._action_counter % 10 == 0:
            self._update_mind_diversity()

        # ── Three-tier progress tracking ──
        is_click = (active_operator_id or "").startswith("click") or (
            self._level_action_trace and self._level_action_trace[-1].name == "ACTION6"
        )
        is_transform = (active_operator_id or "").startswith("global_transform")
        diff_sig = None
        if diff:
            diff_sig = (
                f"{diff.num_changed}_{len(diff.removed_objects)}_"
                f"{len(diff.created_objects)}"
            )
        self.progress.on_action(
            grid_hash=obs.grid_hash,
            diff_signature=diff_sig,
            macro_id=None,
            is_noop=diff.is_noop,
            game_over=diff.game_over,
            player_moved=bool(diff.player_displacement),
            num_changed=diff.num_changed,
            objects=obs.objects,
            current_validated_ops=self.inducer.num_validated(),
            current_validated_rules=len(self.rule_engine.rules),
            operator_predicted_ok=(predicted_ok if active_operator is not None else False),
            is_click=is_click,
            is_transform=is_transform,
        )
        self._log_transition_debug(last_action, diff, active_operator_id)

        # Episodic graph
        state_motif = f"s_{self._prev_obs.grid_hash}"
        op_id = self._current_mind or last_action.name
        outcome = "success" if diff.level_complete else (
            "failure" if diff.game_over else
            "noop" if diff.is_noop else "progress"
        )
        self.episodic.record_transition(state_motif, op_id, outcome)

        # Track lethal values
        if diff.game_over and diff.player_displacement:
            if self._prev_obs.best_player:
                p = self._prev_obs.best_player
                dy, dx = diff.player_displacement
                nr = p.position[0] + dy
                nc = p.position[1] + dx
                grid = obs.raw_grid
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    val = int(grid[nr, nc])
                    if val != 0:
                        self.memory.mark_lethal(val)

        # Record failure pattern
        if diff.game_over:
            motif = f"death_{self._prev_obs.grid_hash}"
            trace = [a.name for a in self._level_action_trace[-5:]]
            self.memory.record_failure(motif, trace, "game_over")

    def _log_transition_debug(
        self,
        last_action: PrimitiveAction,
        diff: FrameDiff,
        operator_id: Optional[str],
    ) -> None:
        """Emit step-level diagnostics for early bootstrapping."""
        if self._debug_action_limit <= 0 or self._action_counter >= self._debug_action_limit:
            return

        lp, sp, tp = self.progress.scores()
        branch_diag = self.progress.branch_diagnostics()
        logger.info(
            "V3 bootstrap step=%s action=%s changed=%s noop=%s "
            "profiler=%s progress_updates=%s lp=%.2f sp=%.2f tp=%.2f "
            "ops=%s validated=%s branch=%s kill=%s operator=%s",
            self._action_counter,
            last_action,
            diff.num_changed,
            diff.is_noop,
            self.profiler.total_transitions,
            self.progress.num_updates,
            lp,
            sp,
            tp,
            len(self.inducer.operators),
            self.inducer.num_validated(),
            branch_diag["branch_id"],
            self.progress.should_kill_branch(),
            operator_id or "-",
        )

    def _maybe_warn_bootstrap_stall(self, obs: GameObservation) -> None:
        """Warn loudly when the perception→transition→progress chain is still dead."""
        if self._bootstrap_warning_emitted or self._action_counter < 10:
            return

        lp, sp, tp = self.progress.scores()
        if (
            self.profiler.total_transitions == 0
            or self.progress.num_updates == 0
            or (
                self._action_counter >= 20
                and lp == 0.0 and sp == 0.0 and tp == 0.0
                and len(self.inducer.operators) == 0
            )
        ):
            self._bootstrap_warning_emitted = True
            logger.warning(
                "V3 bootstrap stall at action=%s transitions=%s progress_updates=%s "
                "lp=%.2f sp=%.2f tp=%.2f ops=%s rules=%s branch_diag=%s "
                "grid_hash=%s",
                self._action_counter,
                self.profiler.total_transitions,
                self.progress.num_updates,
                lp,
                sp,
                tp,
                len(self.inducer.operators),
                len(self.rule_engine.rules),
                self.progress.branch_diagnostics(),
                obs.grid_hash,
            )

    # ─── V3 decision logic ──────────────────────────────────────

    def _decide(self, obs: GameObservation) -> PrimitiveAction:
        """The core V3 decision: experiment / search / react.

        Replaces V2's ε-based exploration/exploitation.
        """
        # ── Continue executing current plan if exists ──
        if self._current_plan and self._plan_index < len(self._current_plan):
            call = self._current_plan[self._plan_index]
            self._plan_index += 1
            self._current_operator_id = call.operator_id
            self._plan_driven_actions += 1
            primitive = self.executor.next_primitive(call, obs)
            return primitive

        # Clear finished plan
        self._current_plan = []
        self._plan_index = 0

        # ── Mode selection (progress-driven) ──
        # Phase 1: Bootstrap mechanics (first 15 actions = pure experiment)
        if self._action_counter < 15:
            return self._do_experiment(obs)

        # Get three-tier progress scores
        lp, sp, tp = self.progress.scores()

        # Branch killer: if sterile, force new branch
        if self.progress.should_kill_branch():
            self.progress.start_new_branch(
                current_validated_ops=self.inducer.num_validated(),
                current_validated_rules=len(self.rule_engine.rules),
            )
            # After killing a branch, do a burst of experiments
            return self._do_experiment(obs)

        # Determine mode probabilities from progress tier
        if self._actions_since_last_level < 20:
            # Just completed a level — explore for new mechanics
            p_experiment, p_search, p_exploit = 0.5, 0.3, 0.2
        elif tp > 0.3:
            # Terminal progress detected — focus on closure
            p_experiment, p_search, p_exploit = 0.05, 0.35, 0.60
        elif sp > 0.3 and self._should_exploit(obs):
            # Structural progress + validated mechanics — balanced
            p_experiment, p_search, p_exploit = 0.10, 0.30, 0.60
        elif sp < 0.1 and lp > 0.3:
            # Local competence but no structural progress — stalling
            p_experiment, p_search, p_exploit = 0.35, 0.45, 0.20
        elif self._should_exploit(obs):
            # Have validated mechanics, moderate structural
            p_experiment, p_search, p_exploit = 0.15, 0.30, 0.55
        else:
            # Weak mechanics — keep learning
            p_experiment, p_search, p_exploit = 0.35, 0.40, 0.25

        # Periodic forced modes (ensure diversity)
        action_mod = self._action_counter % 10
        if action_mod == 0:
            return self._do_search(obs)
        if action_mod == 5 and self._action_counter > 20:
            return self._do_experiment(obs)

        # Force SequenceMind when stalling (local competence, no structural gain)
        if (self._action_counter % 20 == 10 and lp > 0.3 and sp < 0.15
                and self._action_counter > 40):
            return self._do_forced_mind(obs, "sequence")
        # Force ClosureMind when terminal progress shows promise
        if self._action_counter % 15 == 7 and tp > 0.1:
            return self._do_forced_mind(obs, "closure")

        # Stochastic mode selection
        r = random.random()
        if r < p_experiment:
            self._exploit_streak = 0
            return self._do_experiment(obs)
        elif r < p_experiment + p_search:
            self._exploit_streak = 0
            return self._do_search(obs)
        else:
            self._exploit_streak += 1
            return self._do_exploit(obs)

    def _do_forced_mind(self, obs: GameObservation, mind_name: str) -> PrimitiveAction:
        """Force a specific mind to propose and execute its plan."""
        self._mode_counts["search"] += 1
        mind = self.minds.get(mind_name)
        if mind is None:
            return self._do_experiment(obs)

        try:
            proposal = mind.propose(
                obs, self.profiler, self.inducer, self.rule_engine,
            )
            if proposal is not None and proposal.candidate_plan:
                self._current_mind = mind_name
                self._current_plan = proposal.candidate_plan
                self._plan_index = 0
                call = self._current_plan[0]
                self._plan_index = 1
                self._current_operator_id = call.operator_id
                self._operator_driven_actions += 1
                return self.executor.next_primitive(call, obs)
        except Exception as e:
            logger.warning(f"Forced mind {mind_name} error: {e}")

        return self._do_search(obs)

    def _should_exploit(self, obs: GameObservation) -> bool:
        """Decide if we should exploit (reactive/search) vs experiment.

        Gates on VALIDATED mechanics, not just induction.
        """
        if obs.levels_completed > self._prev_levels:
            return True
        # Require validated operators, not just induced ones
        if self.inducer.num_validated(min_uses=3, min_accuracy=0.5) >= 2:
            return True
        if self.inducer.best_validated_confidence() >= EXPLOIT_CONFIDENCE:
            return True
        if self.memory.knowledge_level() >= 0.5:
            return True
        return False

    def _do_experiment(self, obs: GameObservation) -> PrimitiveAction:
        """Run an uncertainty-reducing experiment."""
        self._mode_counts["experiment"] += 1
        self._current_mind = "experiment"
        self._current_operator_id = None
        plan = self.experiment_designer.design(
            obs, self.profiler, self.inducer, max_steps=3,
        )
        if plan.planned_actions:
            self._current_plan = [
                OperatorCall(
                    operator_id=f"experiment_{i}",
                    args={"action": pa.primitive.name}
                    if pa.primitive else {},
                )
                for i, pa in enumerate(plan.planned_actions)
            ]
            # Execute first action
            first = plan.planned_actions[0]
            if first.primitive:
                self._plan_index = 1  # skip first in plan
                return first.primitive

        # Fallback
        return PrimitiveAction(name=random.choice(obs.available_actions))

    def _do_exploit(self, obs: GameObservation) -> PrimitiveAction:
        """Exploit known mechanics: reactive if possible, search otherwise."""
        self._mode_counts["exploit"] += 1

        # If no compositional evidence, 30% chance to explore instead
        # This prevents premature exploitation on shallow local success
        if (not self._has_compositional_evidence()
                and random.random() < 0.3):
            return self._do_experiment(obs)

        # Try reactive control first (cheapest)
        reactive_call = self.reactive.act(obs)
        if reactive_call is not None:
            self._current_mind = "reactive"
            self._current_operator_id = reactive_call.operator_id
            self._operator_driven_actions += 1
            return self.executor.next_primitive(reactive_call, obs)

        # Try option-based plans (compositional layer)
        option_plan = self._try_options(obs)
        if option_plan:
            self._current_mind = "option"
            self._current_plan = option_plan
            self._plan_index = 0
            call = self._current_plan[0]
            self._plan_index = 1
            self._current_operator_id = call.operator_id
            self._operator_driven_actions += 1
            return self.executor.next_primitive(call, obs)

        # Otherwise, use minds + search
        return self._do_search(obs)

    def _try_options(self, obs: GameObservation) -> List[OperatorCall]:
        """Try each option and return the best applicable plan."""
        best_plan: List[OperatorCall] = []
        best_score = -1.0

        for option in self.options:
            if not option.is_applicable(obs, self.inducer):
                continue
            try:
                plan = option.generate_plan(
                    obs, self.inducer, self.rule_engine,
                    lethal_values=self.memory.lethal_values,
                )
                if not plan:
                    continue
                # Score: length-penalized, prefer medium-length plans
                score = min(len(plan), 8) / 8.0
                if len(plan) > 12:
                    score *= 0.7  # penalize very long plans
                if option.name in ("approach", "click_all", "avoid_and_reach"):
                    score *= 1.2  # boost goal-directed options
                if score > best_score:
                    best_score = score
                    best_plan = plan
            except Exception as e:
                logger.warning(f"Option {option.name} error: {e}")

        return best_plan if best_score > 0 else []

    def _do_search(self, obs: GameObservation) -> PrimitiveAction:
        """Collect mind proposals → arbiter selects → search → execute."""
        self._mode_counts["search"] += 1
        # Collect proposals from all minds
        proposals: List[MindProposal] = []
        for mind in self.minds.values():
            try:
                p = mind.propose(
                    obs, self.profiler, self.inducer, self.rule_engine,
                )
                if p is not None:
                    # Apply diversity penalty: reduce confidence for overused minds
                    penalty = self._mind_diversity_penalty.get(mind.name, 0.0)
                    if penalty > 0:
                        p = MindProposal(
                            mind_name=p.mind_name,
                            objective=p.objective,
                            candidate_plan=p.candidate_plan,
                            confidence=max(0.05, p.confidence - penalty),
                            expected_progress=p.expected_progress,
                            expected_info_gain=p.expected_info_gain,
                            estimated_cost=p.estimated_cost,
                            estimated_risk=p.estimated_risk,
                            justification=p.justification,
                        )
                    proposals.append(p)
            except Exception as e:
                logger.warning(f"Mind {mind.name} error: {e}")

        # Arbiter selects best proposal
        chosen = self.arbiter.select(proposals, self.minds)

        if chosen is not None and chosen.candidate_plan:
            self._current_mind = chosen.mind_name
            self._current_plan = chosen.candidate_plan
            self._plan_index = 0

            # If confidence is high enough, run search to refine
            if chosen.confidence >= SEARCH_CONFIDENCE:
                target = chosen.justification.get("target")
                target_dict = ({"position": target} if target and
                               isinstance(target, tuple) else None)
                search_plan = self.searcher.search(
                    obs, self.inducer, self.rule_engine,
                    target=target_dict,
                    macros=list(self.memory.macros.values()),
                    visited_hashes=self._visited_hashes,
                )
                if search_plan:
                    self._current_plan = search_plan

            # Execute first action from plan
            if self._current_plan:
                call = self._current_plan[self._plan_index]
                self._plan_index += 1
                self._current_operator_id = call.operator_id
                self._operator_driven_actions += 1
                return self.executor.next_primitive(call, obs)

        # Fallback: random non-reset action
        self._fallback_actions += 1
        self._mode_counts["fallback"] += 1
        self._current_mind = None
        self._current_operator_id = None
        non_reset = [a for a in obs.available_actions if a != "RESET"]
        action = random.choice(non_reset) if non_reset else "ACTION1"
        return PrimitiveAction(name=action)

    # ─── Compositional evidence ─────────────────────────────────

    def _has_compositional_evidence(self) -> bool:
        """Check if the system has evidence of compositional control.

        Uses three-tier progress: require structural progress, not just local.
        """
        lp, sp, tp = self.progress.scores()
        # Terminal progress is strong evidence
        if tp > 0.2:
            return True
        # Structural progress + validated ops
        if sp > 0.2 and self.inducer.num_validated(min_uses=2, min_accuracy=0.4) >= 3:
            return True
        # Macros + structural gain
        if len(self.memory.macros) > 0 and sp > 0.1:
            return True
        return False

    # ─── Mind diversity ────────────────────────────────────────

    def _update_mind_diversity(self) -> None:
        """Penalize overused minds to prevent committee collapse."""
        selections = self.arbiter._mind_selections
        total = sum(selections.values())
        if total < 5:
            return
        fair_share = 1.0 / max(len(self.minds), 1)
        for name in self.minds:
            count = selections.get(name, 0)
            share = count / total
            # Penalize minds that are >2x their fair share
            if share > fair_share * 2.0:
                excess = share - fair_share
                self._mind_diversity_penalty[name] = min(0.3, excess)
            else:
                self._mind_diversity_penalty[name] = 0.0

    # ─── Level completion handler ───────────────────────────────

    def _on_level_complete(self, levels_completed: int) -> None:
        """Handle level completion: record solution, try to compress."""
        level_idx = levels_completed - 1
        trajectory = SolvedTrajectory(
            level_index=level_idx,
            primitive_actions=list(self._level_action_trace),
            action_count=len(self._level_action_trace),
        )
        self.memory.record_solution(trajectory)

        # Compile macros from solution
        new_macros = self.macro_compiler.compile_from_solution(trajectory)
        inserted_macros = 0
        for m in new_macros:
            if len(self.memory.macros) >= 10:
                break
            self.memory.macros[m.macro_id] = m
            inserted_macros += 1
        self.executor.set_macros(self.memory.macros)

        # Feed solution to sequence mind
        seq_mind = self.minds.get("sequence")
        if seq_mind is not None:
            seq_mind.learn_sequence(
                [a.name for a in self._level_action_trace],
                success=True,
            )

        logger.info(
            f"Level {level_idx} solved in {len(self._level_action_trace)} actions! "
            f"Added {inserted_macros} macros."
        )

        # Reset level trace for next level
        self._level_action_trace = []

    # ─── Game lifecycle ─────────────────────────────────────────

    def end_game(self, won: bool) -> Dict[str, Any]:
        """Called at game end. Export to cross-game memory.

        Returns diagnostic summary.
        """
        # Export cross-game
        if self.cross_game is not None:
            mind_stats = {
                name: mind.recent_accuracy
                for name, mind in self.minds.items()
            }
            self.cross_game.export_game(
                self.memory,
                won=won,
                mind_stats=mind_stats,
                graph_export=self.episodic.export_compact(),
            )

        # Build summary
        # Control leverage metrics
        total = max(self._action_counter, 1)
        prog_summary = self.progress.summary()
        leverage = {
            "operator_driven_pct": round(100 * self._operator_driven_actions / total, 1),
            "plan_driven_pct": round(100 * self._plan_driven_actions / total, 1),
            "fallback_pct": round(100 * self._fallback_actions / total, 1),
            "mode_counts": dict(self._mode_counts),
            "profiled_transitions": self.profiler.total_transitions,
            "progress_updates": self.progress.num_updates,
            "induction_ticks": self._induction_ticks,
            "pred_accuracy": round(self.inducer.operator_predictive_accuracy(), 3),
            "control_success": round(self.inducer.operator_control_success(), 3),
            "validated_ops": self.inducer.num_validated(),
            "macros": len(self.memory.macros),
            "progress": prog_summary,
            "compositional": self._has_compositional_evidence(),
        }

        summary = {
            "total_actions": self._action_counter,
            "levels_completed": self.memory.total_levels_completed,
            "operators": len(self.inducer.operators),
            "rules": len(self.rule_engine.rules),
            "macros": len(self.memory.macros),
            "states_visited": len(self._visited_hashes),
            "knowledge_level": self.memory.knowledge_level(),
            "won": won,
            "mind_selections": dict(self.arbiter._mind_selections),
            "leverage": leverage,
        }

        logger.info(f"Game end: {summary}")
        return summary
