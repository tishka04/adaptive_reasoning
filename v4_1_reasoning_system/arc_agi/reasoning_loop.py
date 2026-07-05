"""
Adaptive Reasoning Loop for ARC-AGI-3 — **hierarchical goal decomposition**.

The loop follows the user's specified architecture:

  1. **Observe** game state (StateDescriber → GameObservation)
  2. **Explore** — try a few actions, memorise game mechanics
  3. **Decompose** — the GoalDecomposer (LLM / templates) produces
     an overarching goal AND an ordered set of subgoals
  4. **For each subgoal**:
     a. **Generate** candidate strategies (StrategyGenerator)
     b. **Predict** latent outcomes (GameWorldModel / JEPA)
     c. **Score** outcomes with EBM and select best strategy
     d. **Act** — the Actioner converts the strategy into concrete
        action(s) using memory's directional knowledge
  5. **Update** memory with actions taken and new game state → repeat

This mirrors the v4_1 ReasoningController.solve() loop, adapted from
abstract reasoning to interactive grid-based game environments.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state_describer import StateDescriber, GameObservation
from .strategy_generator import StrategyGenerator, GameStrategy, StrategyType
from .game_world_model import GameWorldModel, WorldModelConfig
from .energy_scorer import (
    GameEnergyScorer,
    ScoringDecision,
    TrajectoryScoringDecision,
)
from .goal_decomposer import GoalDecomposer, GameGoal, SubGoal, SubGoalStatus
from .actioner import Actioner, ActionResult
from .grid_analyzer import GridAnalyzer, FrameDiff
from .game_memory import GameMemory
from .associative_memory import AssociativeMemory, CrossGameMemory
from .visual_cortex import VisualCortex, VisualCortexConfig
from .goal_pursuit import (
    GoalContext,
    GoalProgressManager,
    GameObjective,
    PARTIAL_THRESHOLD,
    SUCCESS_THRESHOLD,
    StrategyOutcome,
    TrajectoryOutcome,
)
from .trajectory_memory import TrajectoryMemory
from .trajectory_sampler import (
    ContinuationMode,
    SampledTrajectory,
    TrajectorySampler,
)
from .latent_task_programmer import LatentTaskProgram, LatentTaskProgramGenerator
from .goal_recognizer import GoalHypothesis, GoalRecognizer

logger = logging.getLogger(__name__)


ABLATION_STAGES = (
    "symbolic_only",
    "game_memory",
    "goal_pursuit",
    "trajectory_memory",
    "short_horizon",
    "visual_cortex",
    "jepa_ebm",
)
_ABLATION_INDEX = {name: idx for idx, name in enumerate(ABLATION_STAGES)}
_ABLATION_ALIASES = {
    "symbolic": "symbolic_only",
    "symbolic_core": "short_horizon",
    "ascetic": "short_horizon",
    "memory": "game_memory",
    "goals": "goal_pursuit",
    "goal": "goal_pursuit",
    "trajectory": "trajectory_memory",
    "traj_memory": "trajectory_memory",
    "sampling": "short_horizon",
    "short": "short_horizon",
    "vc": "visual_cortex",
    "visual": "visual_cortex",
    "full": "jepa_ebm",
    "jepa": "jepa_ebm",
    "ebm": "jepa_ebm",
}


def _normalise_ablation_stage(stage: Optional[str], reasoning_mode: str) -> str:
    if stage is None or str(stage).strip() == "":
        if str(reasoning_mode).strip().lower() in {"ascetic", "symbolic_core"}:
            return "short_horizon"
        return "jepa_ebm"
    raw = str(stage).strip().lower().replace("-", "_")
    stage_name = _ABLATION_ALIASES.get(raw, raw)
    if stage_name not in _ABLATION_INDEX:
        valid = ", ".join(ABLATION_STAGES)
        raise ValueError(f"Unknown ablation_stage={stage!r}; expected one of: {valid}")
    return stage_name


# ------------------------------------------------------------------
# Phase enum
# ------------------------------------------------------------------
class Phase(Enum):
    """Current phase of the reasoning loop."""
    EXPLORE = "explore"             # trying actions to learn mechanics
    DECOMPOSE = "decompose"         # building / revising goal hierarchy
    STRATEGIZE = "strategize"       # generating & scoring per-subgoal strategies
    EXECUTE = "execute"             # actioner implements chosen strategy
    UPDATE = "update"               # online model update


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass
class LoopConfig:
    """Configuration for the reasoning loop."""
    reasoning_mode: str = "full"       # full | ascetic | symbolic_core
    ablation_stage: Optional[str] = None  # symbolic_only -> ... -> jepa_ebm
    explore_budget: int = 14          # actions spent in pure exploration
    subgoal_budget: int = 30          # max actions per subgoal before skip
    redecompose_interval: int = 40    # re-decompose goals every N actions w/o progress
    online_update_interval: int = 50  # train world model / EBM every N actions
    train_jepa_online: bool = False    # disable JEPA training to reduce noise
    train_ebm_online: bool = False     # disable EBM training to reduce noise
    max_strategies: int = 6           # legacy strategy generation cap
    trajectory_samples: int = 12      # K candidate trajectories
    trajectory_horizon: int = 4       # H steps per sampled trajectory
    sampler_stage: str = "v0"         # staged sampler rebuild: v0 -> v3
    planner_mode: str = "prior"       # prior | hypothesis
    enable_trajectory_continuation: bool = False
    enable_latent_task_programmer: bool = False
    jepa_ebm_rerank_weight: float = 0.10  # learned models stay a weak reranker
    llm_model_name: Optional[str] = None
    llm_device: str = "cpu"
    use_llm: bool = False
    world_model_config: WorldModelConfig = field(default_factory=WorldModelConfig)
    # Pre-trained checkpoint paths (None = train from scratch online)
    world_model_checkpoint: Optional[str] = None
    ebm_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        self.reasoning_mode = str(self.reasoning_mode or "full").strip().lower()
        self.ablation_stage = _normalise_ablation_stage(
            self.ablation_stage,
            self.reasoning_mode,
        )
        if not self.uses_jepa_ebm():
            self.train_jepa_online = False
            self.train_ebm_online = False
        self.jepa_ebm_rerank_weight = max(
            0.0,
            min(1.0, float(self.jepa_ebm_rerank_weight)),
        )

    def ablation_level(self) -> int:
        return _ABLATION_INDEX[str(self.ablation_stage)]

    def uses_game_memory_policy(self) -> bool:
        return self.ablation_level() >= _ABLATION_INDEX["game_memory"]

    def uses_goal_pursuit(self) -> bool:
        return self.ablation_level() >= _ABLATION_INDEX["goal_pursuit"]

    def uses_trajectory_memory(self) -> bool:
        return self.ablation_level() >= _ABLATION_INDEX["trajectory_memory"]

    def uses_short_horizon_sampling(self) -> bool:
        return self.ablation_level() >= _ABLATION_INDEX["short_horizon"]

    def uses_visual_cortex(self) -> bool:
        return self.ablation_level() >= _ABLATION_INDEX["visual_cortex"]

    def uses_jepa_ebm(self) -> bool:
        return self.ablation_level() >= _ABLATION_INDEX["jepa_ebm"]

    def enabled_features(self) -> Dict[str, bool]:
        return {
            "symbolic_observer": True,
            "game_memory_policy": self.uses_game_memory_policy(),
            "goal_pursuit": self.uses_goal_pursuit(),
            "trajectory_memory": self.uses_trajectory_memory(),
            "short_horizon_sampling": self.uses_short_horizon_sampling(),
            "visual_cortex": self.uses_visual_cortex(),
            "jepa_ebm_rerank": self.uses_jepa_ebm(),
        }


# ------------------------------------------------------------------
# Mutable loop state
# ------------------------------------------------------------------
@dataclass
class LoopState:
    """Mutable state of the reasoning loop."""
    phase: Phase = Phase.EXPLORE
    action_counter: int = 0

    # Goal hierarchy
    goal: Optional[GameGoal] = None
    current_subgoal: Optional[SubGoal] = None
    subgoal_start_action: int = 0

    # Current strategy for the active subgoal
    current_strategy: Optional[GameStrategy] = None
    current_trajectory: Optional[SampledTrajectory] = None
    current_goal_context: Optional[GoalContext] = None
    current_goal_hypothesis: Optional[GoalHypothesis] = None
    current_latent_task_program: Optional[LatentTaskProgram] = None
    current_continuation: Optional[ContinuationMode] = None
    pending_continuation: Optional[ContinuationMode] = None
    strategy_start_action: int = 0

    # Observation tracking
    last_observation: Optional[GameObservation] = None
    last_scoring: Optional[ScoringDecision] = None
    last_trajectory_scoring: Optional[TrajectoryScoringDecision] = None
    last_trajectory_debug: Optional[Dict[str, Any]] = None

    # Counters
    strategies_tried: int = 0
    strategies_succeeded: int = 0
    subgoals_completed: int = 0
    decompositions: int = 0
    last_progress_action: int = 0     # last action that produced progress


class _ScalarAuxValue:
    """Small tensor-like scalar used when JEPA aux heads are disabled."""

    def __init__(self, value: float):
        self._value = float(value)

    def item(self) -> float:
        return self._value


@dataclass
class _SymbolicAux:
    progress_prob: _ScalarAuxValue
    risk_prob: _ScalarAuxValue
    novelty_score: _ScalarAuxValue


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
class AdaptiveReasoningLoop:
    """
    Hierarchical adaptive reasoning loop for ARC-AGI-3.

    Architecture:
      observe → explore → decompose goals
        → per subgoal: generate → predict → score → act
        → update memory → repeat
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        memory: Optional[GameMemory] = None,
        cross_game: Optional[CrossGameMemory] = None,
    ):
        self.config = config or LoopConfig()
        self.memory = memory or GameMemory()
        self._cross_game = cross_game

        # ── Components ──────────────────────────────────────────
        self.describer = StateDescriber()
        self.decomposer = GoalDecomposer(
            model_name=self.config.llm_model_name,
            device=self.config.llm_device,
            use_llm=self.config.use_llm,
        )
        self.generator = StrategyGenerator(
            model_name=None,
            device=self.config.llm_device,
            use_llm=False,  # templates for speed; LLM only for goal bank
            max_candidates=self.config.max_strategies,
        )
        self.world_model = GameWorldModel(self.config.world_model_config)
        self.scorer = GameEnergyScorer(self.config.world_model_config)
        # Load pre-trained checkpoints if provided
        if self.config.uses_jepa_ebm() and self.config.world_model_checkpoint:
            try:
                self.world_model.load_checkpoint(self.config.world_model_checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load world model checkpoint: {e}")
        if self.config.uses_jepa_ebm() and self.config.ebm_checkpoint:
            try:
                self.scorer.load_checkpoint(self.config.ebm_checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load EBM checkpoint: {e}")
        self.actioner = Actioner()
        self.analyzer = GridAnalyzer()

        # ── Visual cortex (CNN frame predictor) ──────────────
        self.visual_cortex = VisualCortex(VisualCortexConfig(
            device=self.config.world_model_config.device,
        ))

        # ── Associative memory (brain-inspired LTP/LTD) ──────
        self.assoc_memory = AssociativeMemory(
            ltp_rate=0.3, ltd_rate=0.1, decay_rate=0.005,
            max_episodes=200, max_procedures=30,
        )
        # Inherit cross-game meta-knowledge (NN weights, action priors)
        if self._cross_game is not None:
            self.assoc_memory.new_game(self._cross_game)
        self._assoc_recent_actions: List[int] = []

        # ── Goal pursuit controller ─────────────────────────────
        self.goal_pursuit = GoalProgressManager(
            max_goals=6, max_attempts=5,
        )
        self.trajectory_memory = TrajectoryMemory(self._cross_game)
        self.trajectory_sampler = TrajectorySampler(
            stage=self.config.sampler_stage,
            planner_mode=self.config.planner_mode,
            enable_continuation=self.config.enable_trajectory_continuation,
        )
        self.latent_task_programmer = LatentTaskProgramGenerator()
        self.goal_recognizer = GoalRecognizer()
        self.human_trace_memory: Optional[Any] = None

        # ── State ───────────────────────────────────────────────
        self.state = LoopState()
        self._prev_grid: Optional[np.ndarray] = None
        self._prev_obs: Optional[GameObservation] = None
        self._explore_idx: int = 0
        self._post_level_switch_guard_remaining: int = 0
        self._active_trace_bootstrap: Optional[Any] = None
        self._active_trace_bootstrap_index: int = 0
        self._trace_bootstrap_replan_requested: bool = False
        self._trace_bootstrap_avoid_actions: List[str] = []
        self._trace_bootstrap_avoid_suffix: List[str] = []

    def _allow_continuation(self, level: int) -> bool:
        """Continuation stays off in the V0 sampler until reintroduced deliberately."""
        return (
            self.config.uses_short_horizon_sampling()
            and bool(self.config.enable_trajectory_continuation)
            and int(level) >= 1
        )

    def _allow_latent_task_programmer(self) -> bool:
        """V2: synthesize runtime programs from scored trajectory futures."""
        return (
            self.config.uses_jepa_ebm()
            and (
                bool(self.config.enable_latent_task_programmer)
                or str(self.config.sampler_stage).lower() in {"v2", "v3"}
            )
        )

    def _allow_runtime_human_trace_sampling(self, level: int) -> bool:
        """Runtime human-trace replay starts at V1."""
        del level
        if not self.config.uses_trajectory_memory():
            return False
        if str(self.config.planner_mode).lower() == "hypothesis":
            return False
        return str(self.config.sampler_stage).lower() in {"v1", "v2", "v3"}

    def _consume_post_level_switch_guard(
        self,
        action_name: str,
        goal_context: Optional[GoalContext],
        *,
        game_over: bool,
    ) -> None:
        """V0 sampler disables post-level action guards entirely."""
        del action_name, goal_context, game_over
        return

    def prime_post_level_followup(self, current_level: int) -> None:
        """No-op in V0: level-specific follow-up guards are disabled."""
        del current_level
        return

    def set_human_trace_memory(self, human_trace_memory: Optional[Any]) -> None:
        """Attach runtime human trace memory for retrieval-based sampling."""
        self.human_trace_memory = human_trace_memory

    def _trajectory_bootstrap_policy(
        self,
        *,
        current_grid: np.ndarray,
        obs: GameObservation,
        available: List[str],
        levels_completed: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Replay a distilled level-up prefix before EXPLORE when aligned."""
        if not self.config.uses_trajectory_memory():
            return None
        if self.human_trace_memory is None or not available:
            return None

        segment = self._active_trace_bootstrap
        idx = self._active_trace_bootstrap_index
        if segment is not None:
            if self._trace_frame_matches(segment, idx, current_grid):
                return self._emit_trace_bootstrap_action(segment, idx, obs, available)

            self._clear_trace_bootstrap()
            self._trace_bootstrap_replan_requested = True
            self.state.last_trajectory_debug = {
                "policy": "human_success_prefix",
                "event": "mismatch_replan",
            }
            return None

        finder = getattr(self.human_trace_memory, "aligned_success_prefix_at", None)
        if finder is None:
            return None
        try:
            candidate = finder(
                current_grid,
                available_actions=available,
                current_level=int(levels_completed),
            )
        except Exception:
            logger.debug("aligned success-prefix lookup failed", exc_info=True)
            return None
        if candidate is not None:
            segment, idx = candidate
            self._active_trace_bootstrap = segment
            self._active_trace_bootstrap_index = int(idx)
            return self._emit_trace_bootstrap_action(segment, int(idx), obs, available)

        frontier_finder = getattr(self.human_trace_memory, "recovery_frontier_at", None)
        if frontier_finder is None:
            return None
        try:
            frontier = frontier_finder(current_grid)
        except Exception:
            logger.debug("recovery frontier lookup failed", exc_info=True)
            return None
        if frontier is None:
            return None

        danger_suffix = getattr(frontier, "danger_suffix", None)
        self._trace_bootstrap_avoid_actions = list(getattr(frontier, "avoid_actions", []) or [])
        self._trace_bootstrap_avoid_suffix = list(getattr(danger_suffix, "actions", []) or [])
        self._trace_bootstrap_replan_requested = True
        self.state.last_trajectory_debug = {
            "policy": "recovery_frontier",
            "event": "danger_suffix_replan",
            "episode_id": getattr(frontier, "episode_id", None),
            "trace_index": int(getattr(frontier, "trace_index", -1)),
            "avoid_actions": list(self._trace_bootstrap_avoid_actions),
            "danger_suffix": list(self._trace_bootstrap_avoid_suffix),
        }
        return None

    @staticmethod
    def _trace_frame_matches(trace: Any, idx: int, current_grid: np.ndarray) -> bool:
        frames = getattr(trace, "frames", []) or []
        if idx < 0 or idx >= len(frames):
            return False
        expected = frames[idx]
        if hasattr(expected, "tolist"):
            expected = expected.tolist()
        actual = current_grid.tolist() if hasattr(current_grid, "tolist") else current_grid
        return actual == expected

    def _emit_trace_bootstrap_action(
        self,
        segment: Any,
        idx: int,
        obs: GameObservation,
        available: List[str],
    ) -> Optional[Dict[str, Any]]:
        actions = list(getattr(segment, "actions", []) or [])
        if idx < 0 or idx >= len(actions):
            self._clear_trace_bootstrap()
            return None

        action = str(actions[idx])
        if action not in available:
            self._clear_trace_bootstrap()
            self._trace_bootstrap_replan_requested = True
            self.state.last_trajectory_debug = {
                "policy": "human_success_prefix",
                "event": "action_unavailable_replan",
                "trace_action": action,
                "available": list(available),
            }
            return None

        action_data = self._trace_action_data(segment, idx)
        next_idx = idx + 1
        if next_idx >= len(actions):
            self._clear_trace_bootstrap()
        else:
            self._active_trace_bootstrap = segment
            self._active_trace_bootstrap_index = next_idx

        trajectory = SampledTrajectory(
            actions=[action],
            action_data=[action_data],
            goal_context=self.state.current_goal_context,
            source="human_success_prefix",
            score=float(getattr(segment, "score", 1.0) or 1.0),
            metadata={
                "episode_id": getattr(segment, "episode_id", None),
                "trace_index": int(getattr(segment, "start_index", 0) or 0) + int(idx),
                "segment_index": int(idx),
                "start_level": int(getattr(segment, "start_level", 0) or 0),
                "end_level": int(getattr(segment, "end_level", 0) or 0),
                "trace_levels_completed": int(getattr(segment, "trace_levels_completed", 0) or 0),
                "final_state": getattr(segment, "final_state", None),
                "goal_progress": 1.0,
                "human_compatibility": 1.0,
            },
        )
        self.state.phase = Phase.EXECUTE
        self.state.current_trajectory = trajectory
        self.state.last_trajectory_debug = {
            "ablation_stage": self.config.ablation_stage,
            "policy": "human_success_prefix",
            "selected_source": trajectory.source,
            "selected_first_action": action,
            "episode_id": getattr(segment, "episode_id", None),
            "trace_index": int(getattr(segment, "start_index", 0) or 0) + int(idx),
            "segment_index": int(idx),
            "end_level": int(getattr(segment, "end_level", 0) or 0),
            "trace_levels_completed": int(getattr(segment, "trace_levels_completed", 0) or 0),
        }
        payload = self._result(action, action_data, None, None, Phase.EXECUTE, obs)
        payload["trajectory"] = trajectory
        payload["goal_context"] = self.state.current_goal_context
        payload["goal_hypothesis"] = self.state.current_goal_hypothesis
        payload["trajectory_debug"] = self.state.last_trajectory_debug
        return payload

    @staticmethod
    def _trace_action_data(trace: Any, idx: int) -> Optional[Dict[str, Any]]:
        data_items = list(getattr(trace, "action_data", []) or [])
        if idx < 0 or idx >= len(data_items):
            return None
        data = data_items[idx]
        return dict(data) if isinstance(data, dict) else None

    def _clear_trace_bootstrap(self) -> None:
        self._active_trace_bootstrap = None
        self._active_trace_bootstrap_index = 0

    def _replan_after_trace_mismatch(
        self,
        obs: GameObservation,
        available: List[str],
    ) -> Dict[str, Any]:
        """After an aligned trace breaks, replan instead of re-entering explore."""
        avoid_actions = list(self._trace_bootstrap_avoid_actions)
        danger_suffix = list(self._trace_bootstrap_avoid_suffix)
        safe_available = [
            action for action in available
            if action not in set(avoid_actions)
        ] or list(available)

        if self.config.uses_goal_pursuit() and self.state.goal is None:
            self.state.phase = Phase.DECOMPOSE
            result = self._phase_decompose(obs, safe_available)
        elif self.config.uses_goal_pursuit():
            self.state.current_strategy = None
            self.state.current_trajectory = None
            self.state.phase = Phase.STRATEGIZE
            result = self._phase_strategize(obs, safe_available)
        else:
            self.state.phase = Phase.STRATEGIZE
            result = self._phase_symbolic_policy(obs, safe_available)

        if avoid_actions or danger_suffix:
            debug = result.get("trajectory_debug") or self.state.last_trajectory_debug or {}
            if isinstance(debug, dict):
                debug = dict(debug)
                debug["replan_reason"] = "recovery_frontier"
                debug["avoid_actions"] = avoid_actions
                debug["danger_suffix"] = danger_suffix
                result["trajectory_debug"] = debug
                self.state.last_trajectory_debug = debug
        self._trace_bootstrap_avoid_actions = []
        self._trace_bootstrap_avoid_suffix = []
        return result

    # ==================================================================
    # Main entry point — called every action step
    # ==================================================================
    # How often to build a full observation during EXECUTE phase
    _FULL_OBS_INTERVAL: int = 8

    def step(
        self,
        current_grid: np.ndarray,
        game_state: str,
        levels_completed: int,
        available_actions: List[str],
    ) -> Dict[str, Any]:
        """
        Execute one step of the hierarchical reasoning loop.

        Uses a fast path during EXECUTE: skips expensive observation
        building and just tracks grid changes + dispatches to actioner.
        Full observation is built only on phase transitions or every
        _FULL_OBS_INTERVAL steps.

        Returns dict with keys:
            action, action_data, strategy, subgoal, goal, phase, observation
        """
        self.state.action_counter += 1
        ac = self.state.action_counter

        # ── 1. LIGHTWEIGHT DIFF (always) ─────────────────────────
        diff = None
        if self._prev_grid is not None:
            diff = self.analyzer.compute_diff(self._prev_grid, current_grid)

        changed = diff is not None and diff.anything_changed
        self.actioner.on_action_effect(changed)
        if changed:
            self.state.last_progress_action = ac

        # ── 2. FAST PATH: pure execution ─────────────────────────
        #    Skip full observation when we're just executing a strategy
        #    and nothing interesting happened.
        in_execute = (
            self.state.phase == Phase.EXECUTE
            and self.state.current_strategy is not None
            and ac > self.config.explore_budget
        )
        steps_since_full_obs = ac - getattr(self, "_last_full_obs_ac", 0)
        need_full_obs = (
            not in_execute
            or changed                        # something changed → re-observe
            or steps_since_full_obs >= self._FULL_OBS_INTERVAL
        )

        if need_full_obs:
            obs = self.describer.describe(
                grid=current_grid,
                memory=self.memory,
                game_state=game_state,
                levels_completed=levels_completed,
                action_counter=ac,
                diff=diff,
            )
            self.state.last_observation = obs
            self._last_full_obs_ac = ac
        else:
            obs = self._prev_obs  # reuse previous observation

        # ── 3. TRAJECTORY BOOTSTRAP ──────────────────────────────
        result = self._trajectory_bootstrap_policy(
            current_grid=current_grid,
            obs=obs,
            available=available_actions,
            levels_completed=levels_completed,
        )

        if result is None:
            if self._trace_bootstrap_replan_requested:
                self._trace_bootstrap_replan_requested = False
                result = self._replan_after_trace_mismatch(obs, available_actions)
            else:
                # ── 4. PHASE MANAGEMENT ─────────────────────────
                self._update_phase(obs, available_actions)

                # ── 5. DISPATCH to current phase ────────────────
                phase = self.state.phase
                if phase == Phase.EXPLORE:
                    result = self._phase_explore(obs, available_actions)
                elif phase == Phase.DECOMPOSE:
                    result = self._phase_decompose(obs, available_actions)
                elif phase == Phase.STRATEGIZE:
                    result = self._phase_strategize(obs, available_actions)
                elif phase == Phase.EXECUTE:
                    result = self._phase_execute(obs, available_actions)
                else:  # UPDATE
                    result = self._phase_update(obs, available_actions)

        # ── 5. Feed associative memory ─────────────────────────
        act_name = result.get("action", "ACTION1")
        act_int = int(act_name.replace("ACTION", "")) if act_name.startswith("ACTION") else 0
        level_changed = levels_completed > getattr(self, "_prev_levels_completed", 0)
        game_over = game_state == "GAME_OVER"
        self.assoc_memory.record_step(
            grid=self._prev_grid,
            action=act_int,
            action_data=result.get("action_data"),
            changed=changed,
            level_changed=level_changed,
            game_over=game_over,
        )
        self._assoc_recent_actions.append(act_int)
        if len(self._assoc_recent_actions) > 10:
            self._assoc_recent_actions = self._assoc_recent_actions[-10:]
        self._prev_levels_completed = levels_completed

        # ── 6. Feed visual cortex ────────────────────────────
        if self.config.uses_visual_cortex() and self._prev_grid is not None:
            self.visual_cortex.record_transition(
                grid_before=self._prev_grid,
                action=act_int,
                action_data=result.get("action_data"),
                grid_after=current_grid,
            )

        # ── 7. Periodic online model training ───────────────────
        if ac > 0 and ac % self.config.online_update_interval == 0:
            wm_loss = (
                self.world_model.update(train_steps=3)
                if self.config.uses_jepa_ebm() and self.config.train_jepa_online
                else 0.0
            )
            ebm_loss = (
                self.scorer.update(train_steps=3)
                if self.config.uses_jepa_ebm() and self.config.train_ebm_online
                else 0.0
            )
            vc_loss = (
                self.visual_cortex.train(steps=3)
                if self.config.uses_visual_cortex()
                else 0.0
            )
            logger.debug(
                f"Online update: WM={wm_loss:.4f}, EBM={ebm_loss:.4f}, VC={vc_loss:.4f}"
            )

        # ── 8. Book-keeping ─────────────────────────────────────
        self._prev_grid = current_grid.copy()
        if need_full_obs:
            self._prev_obs = obs

        return result

    # ==================================================================
    # Phase management
    # ==================================================================
    def _update_phase(
        self, obs: GameObservation, available_actions: List[str]
    ) -> None:
        ac = self.state.action_counter

        # Phase 1: Explore (first N actions)
        if ac <= self.config.explore_budget:
            self.state.phase = Phase.EXPLORE
            return

        # Early ablations deliberately stop before goal decomposition.
        if not self.config.uses_goal_pursuit():
            self.state.phase = Phase.STRATEGIZE
            return

        # Need to decompose goals?
        if self.state.goal is None:
            self.state.phase = Phase.DECOMPOSE
            return

        # Check if we should re-decompose (stuck for too long)
        actions_since_progress = ac - self.state.last_progress_action
        if actions_since_progress >= self.config.redecompose_interval:
            logger.info(
                f"No progress for {actions_since_progress} actions → re-decomposing goals"
            )
            self.state.phase = Phase.DECOMPOSE
            return

        # Advance subgoal if current one is done or timed out
        sg = self.state.current_subgoal
        if sg is None or sg.is_terminal:
            self._advance_subgoal()
            sg = self.state.current_subgoal

        # If all subgoals done, re-decompose
        if sg is None:
            self.state.phase = Phase.DECOMPOSE
            return

        # Subgoal budget exceeded → skip it
        if sg.actions_spent >= sg.max_actions:
            logger.info(f"Subgoal {sg.id} budget exceeded → skipping")
            sg.status = SubGoalStatus.SKIPPED
            self._advance_subgoal()
            if self.state.current_subgoal is None:
                self.state.phase = Phase.DECOMPOSE
                return

        # Need a strategy for the current subgoal?
        if self.state.current_strategy is None:
            self.state.phase = Phase.STRATEGIZE
            return

        # Strategy budget exceeded? Re-strategize
        strat_actions = ac - self.state.strategy_start_action
        if strat_actions >= self.config.subgoal_budget // 2:
            self.state.current_strategy = None
            self.state.phase = Phase.STRATEGIZE
            return

        # Default: execute
        self.state.phase = Phase.EXECUTE

    # ==================================================================
    # Phase implementations
    # ==================================================================

    # ── EXPLORE ──────────────────────────────────────────────────
    def _phase_explore(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Try each action 2× to learn mechanics.

        For ACTION6 (click), cycles through: objects → non-zero cells → grid scan,
        so click-only games discover which clicks change the grid."""
        import numpy as np

        n = len(available)
        cycle_idx = self._explore_idx // 2
        action_idx = cycle_idx % n
        action = available[action_idx]

        action_data = None
        if action == "ACTION6":
            action_data = self._explore_click_position(obs)

        self._explore_idx += 1
        logger.debug(f"EXPLORE: {action} ({self._explore_idx})")

        return self._result(action, action_data, None, None, Phase.EXPLORE, obs)

    def _phase_fast_random(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Fast random exploration — random agent speed with click intelligence."""
        import random as _rand

        action = _rand.choice(available)
        action_data = None

        if action == "ACTION6":
            # Use novelty-driven clicking (avoid already-clicked positions)
            action_data = self._explore_click_position(obs)
        
        return self._result(action, action_data, None, None, Phase.EXPLORE, obs)

    def _phase_symbolic_policy(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Early ablation policy: symbolic observation, optionally GameMemory ranking."""
        if not available:
            return self._result("ACTION1", None, None, None, Phase.STRATEGIZE, obs)

        if self.config.uses_game_memory_policy():
            ranked = self.memory.rank_actions(available)
            action = ranked[0] if ranked else available[self.state.action_counter % len(available)]
        else:
            action = available[self.state.action_counter % len(available)]

        action_data = None
        if action == "ACTION6":
            action_data = (
                self._explore_click_position(obs)
                if self.config.uses_game_memory_policy()
                else self._symbolic_click_position(obs)
            )

        self.state.last_trajectory_debug = {
            "ablation_stage": self.config.ablation_stage,
            "policy": (
                "game_memory_ranked"
                if self.config.uses_game_memory_policy()
                else "symbolic_cycle"
            ),
            "selected_action": action,
        }
        return self._result(action, action_data, None, None, Phase.STRATEGIZE, obs)

    def _symbolic_click_position(self, obs: GameObservation) -> Dict[str, int]:
        """Pick a deterministic object/cell center without learned memory."""
        objects = [
            obj for obj in obs.objects
            if not obj.get("is_player")
        ]
        if objects:
            obj = objects[self.state.action_counter % len(objects)]
            return {
                "x": int(obj.get("center_x", 0)),
                "y": int(obj.get("center_y", 0)),
            }
        if obs.raw_grid is not None:
            non_zero = list(zip(*np.nonzero(obs.raw_grid)))
            if non_zero:
                y, x = non_zero[self.state.action_counter % len(non_zero)]
                return {"x": int(x), "y": int(y)}
            h, w = obs.raw_grid.shape[:2]
            return {"x": int(w // 2), "y": int(h // 2)}
        return {"x": 32, "y": 32}

    def _explore_click_position(self, obs: GameObservation) -> dict:
        """Delegate to actioner's novelty-driven click exploration."""
        result = self.actioner._explore_click(obs, self.memory)
        return result.action_data or {"x": 32, "y": 32}

    def _phase_goal_strategy_symbolic(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Goal-pursuit ablation: strategy generation with symbolic scoring only."""
        strategies: List[GameStrategy] = []
        memory_strategy = self._strategy_from_trajectory_memory(obs, available)
        if memory_strategy is not None:
            strategies.append(memory_strategy)
        strategies.extend(self.generator.generate(obs, available))
        if not strategies:
            return self._phase_symbolic_policy(obs, available)

        scored = [
            (self._score_strategy_symbolically(strategy, obs, available), strategy)
            for strategy in strategies
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        selected_score, selected = scored[0]
        top_candidates = [
            {
                "rank": idx + 1,
                "strategy_type": strategy.strategy_type.value,
                "first_action": strategy.action_plan[0] if strategy.action_plan else None,
                "score": round(float(score), 3),
                "source": strategy.metadata.get("trajectory_source", "strategy"),
            }
            for idx, (score, strategy) in enumerate(scored[:3])
        ]

        selected.metadata.update({
            "symbolic_strategy_score": float(selected_score),
            "trajectory_top_candidates": top_candidates,
        })
        selected_trajectory = None
        if self.config.uses_trajectory_memory() and selected.action_plan:
            goal_hypothesis = self.goal_recognizer.predict(
                obs,
                memory=self.memory,
                current_goal=self.state.goal,
                current_subgoal=self.state.current_subgoal,
                task_program=getattr(self.decomposer, "_task_program", None),
            )
            goal_context = self._build_goal_context(obs, goal_hypothesis)
            action_data = [None for _ in selected.action_plan[: self.config.trajectory_horizon]]
            click_targets = list(selected.metadata.get("click_targets", []) or [])
            for idx, action in enumerate(selected.action_plan[: len(action_data)]):
                if action == "ACTION6" and click_targets:
                    action_data[idx] = click_targets[idx % len(click_targets)]
            selected_trajectory = SampledTrajectory(
                actions=list(selected.action_plan[: self.config.trajectory_horizon]),
                action_data=action_data,
                goal_context=goal_context,
                source=str(selected.metadata.get("trajectory_source", "strategy_symbolic")),
                metadata={
                    "goal_progress": max(0.0, min(1.0, float(selected_score))),
                    "generator_confidence": float(selected.confidence),
                    "symbolic_score": float(selected_score),
                },
            )
            self.state.current_goal_hypothesis = goal_hypothesis
            self.state.current_goal_context = goal_context
        self.state.last_scoring = None
        self.state.last_trajectory_scoring = None
        self.state.last_trajectory_debug = {
            "ablation_stage": self.config.ablation_stage,
            "policy": "goal_strategy_symbolic",
            "selected_score": round(float(selected_score), 3),
            "selected_first_action": (
                selected.action_plan[0] if selected.action_plan else None
            ),
            "top_candidates": top_candidates,
        }
        self.state.current_strategy = selected
        self.state.current_trajectory = selected_trajectory
        self.state.strategy_start_action = self.state.action_counter
        self.state.strategies_tried += 1
        self.state.phase = Phase.EXECUTE
        return self._phase_execute(obs, available)

    def _strategy_from_trajectory_memory(
        self,
        obs: GameObservation,
        available: List[str],
    ) -> Optional[GameStrategy]:
        if not self.config.uses_trajectory_memory():
            return None
        goal_family = self._goal_family_from_state(obs)
        records = self.trajectory_memory.retrieve_similar(
            goal_family,
            state_embedding=None,
            top_k=5,
        )
        for record in records:
            actions = [
                action for action in TrajectorySampler._normalize_action_sequence(record.actions)
                if action in available
            ]
            if not actions or self.trajectory_memory.is_failed_prefix(actions):
                continue
            action_data = list(record.action_data[: len(actions)])
            click_targets = [
                data for action, data in zip(actions, action_data)
                if action == "ACTION6" and isinstance(data, dict)
            ]
            metadata = {
                "trajectory_source": "trajectory_memory",
                "trajectory_memory_goal_family": record.goal_family,
                "trajectory_memory_score": float(record.score),
                "trajectory_memory_progress": float(record.progress_delta),
            }
            if click_targets:
                metadata["click_targets"] = click_targets
            return GameStrategy(
                strategy_type=StrategyType.SEQUENCE_ACTIONS,
                description=f"Replay useful prefix for {goal_family}",
                action_plan=actions,
                rationale="TrajectoryMemory fragment previously produced measured progress.",
                confidence=max(0.30, min(0.90, 0.45 + float(record.score))),
                metadata=metadata,
            )
        return None

    def _trajectory_memory_trajectories(
        self,
        *,
        goal_context: GoalContext,
        available: List[str],
        state_embedding: Optional[np.ndarray],
        horizon: int,
    ) -> List[SampledTrajectory]:
        records = self.trajectory_memory.retrieve_similar(
            goal_context.goal_family,
            state_embedding=state_embedding,
            top_k=3,
        )
        out: List[SampledTrajectory] = []
        seen: set[tuple[str, ...]] = set()
        for record in records:
            actions = [
                action for action in TrajectorySampler._normalize_action_sequence(record.actions)
                if action in available
            ][: max(1, int(horizon))]
            if not actions:
                continue
            sig = tuple(actions)
            if sig in seen or self.trajectory_memory.is_failed_prefix(actions):
                continue
            seen.add(sig)
            action_data = list(record.action_data[: len(actions)])
            while len(action_data) < len(actions):
                action_data.append(None)
            out.append(
                SampledTrajectory(
                    actions=actions,
                    action_data=action_data,
                    goal_context=goal_context,
                    source="trajectory_memory",
                    metadata={
                        "preferred_fraction": self._memory_preferred_fraction(
                            actions, goal_context
                        ),
                        "trajectory_memory_score": float(record.score),
                        "trajectory_memory_progress": float(record.progress_delta),
                        "trajectory_memory_prediction_match": float(record.prediction_match),
                        "generator_confidence": max(
                            0.30,
                            min(0.90, 0.45 + float(record.score)),
                        ),
                    },
                )
            )
        return out

    @staticmethod
    def _memory_preferred_fraction(
        actions: List[str],
        goal_context: GoalContext,
    ) -> float:
        preferred = set(goal_context.preferred_actions)
        if not actions or not preferred:
            return 0.0
        return sum(1 for action in actions if action in preferred) / len(actions)

    def _score_strategy_symbolically(
        self,
        strategy: GameStrategy,
        obs: GameObservation,
        available: List[str],
    ) -> float:
        plan = [action for action in strategy.action_plan if action in available]
        if not plan:
            return 0.0
        action_scores = [self.memory.score_action(action) for action in plan]
        action_quality = sum(action_scores) / max(len(action_scores), 1)
        confidence = max(0.0, min(1.0, float(strategy.confidence)))
        locked = self.memory.get_locked_mechanisms()
        lock_bonus = 0.0
        if locked:
            lock_bonus = sum(1 for action in plan if action in locked) / max(len(plan), 1)

        subgoal = self.state.current_subgoal
        metadata = getattr(subgoal, "metadata", {}) if subgoal is not None else {}
        preferred = set(TrajectorySampler._normalize_actions(
            list(metadata.get("prefer_actions", []) or [])
        ))
        preferred_bonus = (
            sum(1 for action in plan if action in preferred) / max(len(plan), 1)
            if preferred else 0.0
        )

        hazard_penalty = 0.0
        for action in plan:
            profile = self.memory.action_profiles.get(action)
            if profile is None:
                continue
            hazard_penalty += (
                profile.times_caused_game_over / max(profile.times_tried, 1)
            )
        hazard_penalty /= max(len(plan), 1)

        type_bonus = 0.0
        if strategy.strategy_type == StrategyType.CLICK_OBJECTS and "ACTION6" in plan:
            type_bonus += 0.10
        if strategy.strategy_type in {
            StrategyType.NAVIGATE_TO_GOAL,
            StrategyType.COLLECT_ITEMS,
        } and obs.player_info is not None:
            type_bonus += 0.10
        if strategy.metadata.get("trajectory_source") == "trajectory_memory":
            type_bonus += 0.18

        return (
            0.42 * action_quality
            + 0.25 * confidence
            + 0.15 * lock_bonus
            + 0.12 * preferred_bonus
            + type_bonus
            - 0.35 * hazard_penalty
        )

    # ── DECOMPOSE ────────────────────────────────────────────────
    def _phase_decompose(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Build / revise the hierarchical goal structure."""
        goal = self.decomposer.decompose(
            observation=obs,
            memory=self.memory,
            previous_goal=self.state.goal,
        )
        self.state.goal = goal
        self.state.decompositions += 1
        self.state.current_strategy = None
        self.state.last_progress_action = self.state.action_counter

        # Activate the first subgoal
        self._advance_subgoal()

        logger.info(
            f"DECOMPOSE (rev {goal.revision}): \"{goal.overarching_goal}\" "
            f"→ {len(goal.subgoals)} subgoals, "
            f"confidence={goal.confidence:.0%}"
        )
        for sg in goal.subgoals:
            logger.info(f"  [{sg.status.value}] SG{sg.id}: {sg.description}")

        # Immediately strategize for the first subgoal
        self.state.phase = Phase.STRATEGIZE
        return self._phase_strategize(obs, available)

    # ── STRATEGIZE ───────────────────────────────────────────────
    def _phase_strategize(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Sample futures, score them, and execute the first action of the best."""
        if not self.config.uses_goal_pursuit():
            return self._phase_symbolic_policy(obs, available)
        if not self.config.uses_short_horizon_sampling():
            return self._phase_goal_strategy_symbolic(obs, available)

        # Visual cortex: predict action effects and feed all downstream systems
        if (
            self.config.uses_visual_cortex()
            and self.visual_cortex.trained_steps >= 10
            and obs.raw_grid is not None
        ):
            avail_ints = []
            for a in available:
                try:
                    avail_ints.append(int(a.replace("ACTION", "")))
                except ValueError:
                    pass
            if avail_ints:
                # (A) NL summary for strategy generator prompt
                action_summary = self.visual_cortex.get_action_summary(
                    obs.raw_grid, avail_ints,
                )
                obs.visual_cortex_summary = action_summary

                # (B) Structured analysis for associative memory
                vc_analysis = self.visual_cortex.analyze_action_effects(
                    obs.raw_grid, avail_ints,
                )
                vc_similarity = self.visual_cortex.compute_action_similarity(
                    obs.raw_grid, avail_ints,
                )
                self.assoc_memory.ingest_vc_predictions(
                    vc_analysis, vc_similarity, obs.raw_grid,
                )

                # (C) Propagate VC directions to GameMemory for actioner
                for act_int, info in vc_analysis.items():
                    direction = info.get("direction")
                    if direction is not None:
                        act_name = f"ACTION{act_int}"
                        if act_name not in self.memory.action_directions:
                            self.memory.vc_directions[act_name] = direction

                logger.info(f"Visual cortex: {action_summary}")

        z_t = None
        state_embedding = None
        if self.config.uses_jepa_ebm():
            z_t = self.world_model.encode_observation(obs)
            state_embedding = z_t.squeeze(0).detach().cpu().numpy()
        goal_hypothesis = self.goal_recognizer.predict(
            obs,
            memory=self.memory,
            current_goal=self.state.goal,
            current_subgoal=self.state.current_subgoal,
            task_program=getattr(self.decomposer, "_task_program", None),
        )
        self.state.current_goal_hypothesis = goal_hypothesis
        goal_context = self._build_goal_context(obs, goal_hypothesis)
        self.state.current_goal_context = goal_context
        current_level = int(getattr(obs, "level", 0) or 0)
        retrieved_human_traces = []
        if (
            self.human_trace_memory is not None
            and self._allow_runtime_human_trace_sampling(current_level)
        ):
            try:
                retrieved_human_traces = self.human_trace_memory.retrieve(
                    obs,
                    goal_family=goal_hypothesis.family,
                    k=5,
                )
            except Exception:
                logger.debug("human trace retrieval failed", exc_info=True)
        if not self._allow_continuation(current_level):
            self.state.current_continuation = None
            self.state.pending_continuation = None
        if not self._allow_latent_task_programmer():
            self.state.current_latent_task_program = None

        trajectories = self.trajectory_sampler.sample(
            observation=obs,
            goal_context=goal_context,
            memory=self.memory,
            assoc_memory=self.assoc_memory,
            trajectory_memory=self.trajectory_memory,
            available_actions=available,
            human_traces=retrieved_human_traces,
            continuation=(
                self.state.current_continuation
                if self._allow_continuation(current_level)
                else None
            ),
            action_counter=self.state.action_counter,
            k=self.config.trajectory_samples,
            horizon=self.config.trajectory_horizon,
            state_embedding=state_embedding,
        )
        if self.config.uses_trajectory_memory():
            trajectories = self._trajectory_memory_trajectories(
                goal_context=goal_context,
                available=available,
                state_embedding=state_embedding,
                horizon=self.config.trajectory_horizon,
            ) + trajectories
        if not trajectories:
            action = random.choice(available)
            return self._result(action, None, None, self.state.current_subgoal, Phase.STRATEGIZE, obs)

        strategies = [traj.to_strategy() for traj in trajectories]
        z_hats = None
        s_embs = None
        if self.config.uses_jepa_ebm() and z_t is not None:
            z_hats, aux_list = self.world_model.predict_strategy_outcomes(z_t, strategies)
            s_embs = self.world_model.strategy_encoder.encode_batch(
                strategies, self.world_model.device
            )
        else:
            aux_list = [
                self._symbolic_aux_for_trajectory(obs, goal_context, trajectory)
                for trajectory in trajectories
            ]

        for idx, trajectory in enumerate(trajectories):
            if z_hats is not None:
                trajectory.predicted_latents = [z_hats[:, idx, :].detach().cpu().numpy()]
            if (
                self.config.uses_visual_cortex()
                and self.visual_cortex.trained_steps >= 10
                and obs.raw_grid is not None
            ):
                seq = []
                for action_name, action_data in zip(trajectory.actions, trajectory.action_data):
                    if not action_name.startswith("ACTION"):
                        continue
                    try:
                        seq.append((int(action_name.replace("ACTION", "")), action_data))
                    except ValueError:
                        continue
                if seq:
                    trajectory.predicted_observations = self.visual_cortex.imagine_sequence(
                        obs.raw_grid,
                        seq[: self.config.trajectory_horizon],
                    )
            trajectory.metadata.update(
                self._trajectory_heuristics(obs, goal_context, trajectory, aux_list[idx])
            )

        if (
            self.config.uses_jepa_ebm()
            and z_t is not None
            and s_embs is not None
            and z_hats is not None
        ):
            learned_decision = self.scorer.score_trajectories(
                z_t, trajectories, s_embs, z_hats, aux_list
            )
            decision = self._rerank_with_symbolic_core(trajectories, learned_decision)
        else:
            decision = self._score_trajectories_symbolically(trajectories)
        self.state.last_scoring = None
        self.state.last_trajectory_scoring = decision

        selected = strategies[decision.selected_idx]
        selected_trajectory = decision.selected_trajectory
        source_counts: Dict[str, int] = {}
        for trajectory in trajectories:
            source_counts[trajectory.source] = source_counts.get(trajectory.source, 0) + 1

        latent_program: Optional[LatentTaskProgram] = None
        if self._allow_latent_task_programmer():
            latent_program = self.latent_task_programmer.build(
                goal_context=goal_context,
                trajectories=trajectories,
                selected=selected_trajectory,
                top_indices=decision.top_k_indices,
            )
            if latent_program is not None:
                self._install_latent_task_program(latent_program)

        top_candidates: List[Dict[str, Any]] = []
        for rank, top_idx in enumerate(decision.top_k_indices, start=1):
            candidate = trajectories[top_idx]
            top_candidates.append({
                "rank": rank,
                "source": candidate.source,
                "first_action": candidate.actions[0] if candidate.actions else None,
                "score": round(float(candidate.score), 3),
                "energy": round(float(candidate.energy), 3),
                "goal_progress": round(
                    float(candidate.metadata.get("goal_progress", 0.0)), 3
                ),
                "novelty": round(float(candidate.metadata.get("novelty", 0.0)), 3),
                "risk": round(float(candidate.metadata.get("risk", 0.0)), 3),
                "human_compatibility": round(
                    float(candidate.metadata.get("human_compatibility", 0.0)), 3
                ),
                "hypothesis_kind": candidate.metadata.get("hypothesis_kind"),
                "hypothesis_confidence": round(
                    float(candidate.metadata.get("hypothesis_confidence", 0.0)), 3
                ),
                "hypothesis_experiment_score": round(
                    float(candidate.metadata.get("hypothesis_experiment_score", 0.0)), 3
                ),
                "hypothesis_recent_penalty": round(
                    float(candidate.metadata.get("hypothesis_recent_penalty", 0.0)), 3
                ),
                "control_switch_repeat_penalty": round(
                    float(candidate.metadata.get("control_switch_repeat_penalty", 0.0)), 3
                ),
                "symbolic_score": round(
                    float(candidate.metadata.get("symbolic_score", candidate.score)), 3
                ),
                "learned_score": round(
                    float(candidate.metadata.get("learned_score", 0.0)), 3
                ),
                "rerank_score": round(
                    float(candidate.metadata.get("rerank_score", candidate.score)), 3
                ),
                "hypothesis_support": int(
                    candidate.metadata.get("hypothesis_support", 0) or 0
                ),
            })

        latent_summary = None
        if latent_program is not None:
            latent_summary = {
                "id": latent_program.objective_id,
                "preferred_actions": list(latent_program.preferred_actions),
                "preferred_sequences": [
                    list(seq) for seq in latent_program.preferred_sequences[:3]
                ],
                "avoid_actions": list(latent_program.avoid_actions),
                "confidence": round(float(latent_program.confidence), 3),
                "override_static": bool(
                    latent_program.metadata.get("latent_override_static_program", False)
                ),
            }

        selected_trajectory.metadata.update({
            "source_counts": dict(source_counts),
            "top_candidates": top_candidates,
            "goal_hypothesis_source": goal_hypothesis.source,
            "goal_hypothesis_family": goal_hypothesis.family,
            "latent_task_program": latent_summary,
        })
        selected.metadata.update({
            "trajectory_source": selected_trajectory.source,
            "trajectory_score": float(decision.selected_score),
            "trajectory_energy": float(decision.selected_energy),
            "trajectory_goal_progress": float(
                selected_trajectory.metadata.get("goal_progress", 0.0)
            ),
            "trajectory_novelty": float(
                selected_trajectory.metadata.get("novelty", 0.0)
            ),
            "trajectory_risk": float(selected_trajectory.metadata.get("risk", 0.0)),
            "trajectory_human_compatibility": float(
                selected_trajectory.metadata.get("human_compatibility", 0.0)
            ),
            "goal_hypothesis_family": goal_hypothesis.family,
            "goal_hypothesis_confidence": float(goal_hypothesis.confidence),
            "goal_hypothesis_source": goal_hypothesis.source,
            "trajectory_source_counts": dict(source_counts),
            "trajectory_top_candidates": top_candidates,
            "trajectory_hypothesis_kind": selected_trajectory.metadata.get("hypothesis_kind"),
            "trajectory_hypothesis_confidence": float(
                selected_trajectory.metadata.get("hypothesis_confidence", 0.0)
            ),
            "trajectory_hypothesis_experiment_score": float(
                selected_trajectory.metadata.get("hypothesis_experiment_score", 0.0)
            ),
            "trajectory_hypothesis_recent_penalty": float(
                selected_trajectory.metadata.get("hypothesis_recent_penalty", 0.0)
            ),
            "trajectory_control_switch_repeat_penalty": float(
                selected_trajectory.metadata.get("control_switch_repeat_penalty", 0.0)
            ),
            "trajectory_latent_task_program": latent_summary,
            "trajectory_symbolic_score": float(
                selected_trajectory.metadata.get("symbolic_score", selected_trajectory.score)
            ),
            "trajectory_learned_score": float(
                selected_trajectory.metadata.get("learned_score", 0.0)
            ),
            "trajectory_rerank_score": float(
                selected_trajectory.metadata.get("rerank_score", selected_trajectory.score)
            ),
        })
        self.state.last_trajectory_debug = {
            "ablation_stage": self.config.ablation_stage,
            "jepa_ebm_rerank_weight": self.config.jepa_ebm_rerank_weight,
            "source_counts": dict(source_counts),
            "selected_source": selected_trajectory.source,
            "selected_first_action": (
                selected_trajectory.actions[0] if selected_trajectory.actions else None
            ),
            "selected_score": round(float(decision.selected_score), 3),
            "selected_energy": round(float(decision.selected_energy), 3),
            "top_candidates": top_candidates,
            "latent_task_program": latent_summary,
        }

        logger.info(
            f"STRATEGIZE for SG{self.state.current_subgoal.id if self.state.current_subgoal else '?'}: "
            f"{len(trajectories)} sampled trajectories {source_counts}"
        )
        logger.info(
            f"Trajectory → {selected_trajectory.source} "
            f"(score={decision.selected_score:.3f}, energy={decision.selected_energy:.3f}): "
            f"{selected.description[:100]}"
        )

        self.state.current_strategy = selected
        self.state.current_trajectory = selected_trajectory
        self.state.strategy_start_action = self.state.action_counter
        self.state.strategies_tried += 1

        # Immediately execute the first action
        self.state.phase = Phase.EXECUTE
        return self._phase_execute(obs, available)

    # ── EXECUTE ──────────────────────────────────────────────────
    def _phase_execute(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Delegate to the Actioner to emit the chosen trajectory prefix action."""
        strategy = self.state.current_strategy
        trajectory = self.state.current_trajectory
        subgoal = self.state.current_subgoal

        if strategy is None:
            self.state.phase = Phase.STRATEGIZE
            return self._phase_strategize(obs, available)

        result = self.actioner.act(
            strategy=strategy,
            subgoal=subgoal,
            observation=obs,
            memory=self.memory,
            available_actions=available,
        )

        # Track subgoal budget
        if subgoal is not None:
            subgoal.actions_spent += 1

        logger.debug(
            f"EXECUTE [SG{subgoal.id if subgoal else '?'} / "
            f"{strategy.strategy_type.value}]: {result.action} — {result.reason}"
        )
        payload = self._result(
            result.action, result.action_data, strategy, subgoal,
            Phase.EXECUTE, obs,
        )
        payload["trajectory"] = trajectory
        payload["goal_context"] = self.state.current_goal_context
        payload["goal_hypothesis"] = self.state.current_goal_hypothesis
        payload["trajectory_debug"] = self.state.last_trajectory_debug
        self.state.pending_continuation = self._continuation_after_execute(
            trajectory,
            self.state.current_goal_context,
            self.state.current_goal_hypothesis,
        )

        # Replan every step: execute only the first action of the chosen future.
        self.state.current_strategy = None
        self.state.current_trajectory = None

        return payload

    def _continuation_after_execute(
        self,
        trajectory: Optional[SampledTrajectory],
        goal_context: Optional[GoalContext],
        goal_hypothesis: Optional[GoalHypothesis],
    ) -> Optional[ContinuationMode]:
        """Create a soft continuation from the selected future remainder."""
        if trajectory is None or len(trajectory.actions) <= 1:
            return None
        current_level = (
            self.state.last_observation.level
            if self.state.last_observation is not None else 0
        )
        if not self._allow_continuation(current_level):
            return None
        metadata = goal_context.metadata if goal_context is not None else {}
        task_program_id = str(metadata.get("task_program_id", "") or "").lower()
        is_probe = bool(metadata.get("probe")) or task_program_id.startswith("probe") or task_program_id.startswith("identify")
        if is_probe:
            return None
        predicted_progress = float(trajectory.metadata.get("goal_progress", 0.0))
        source = trajectory.source
        worthy_source = source in {
            "task_program",
            "human",
            "continuation",
            "trajectory_memory",
            "latent_program",
        }
        worthy_goal = goal_hypothesis is not None and goal_hypothesis.confidence >= 0.55
        if predicted_progress < 0.45 and not (worthy_source and worthy_goal):
            return None
        return ContinuationMode(
            source_trajectory=trajectory,
            remaining_actions=list(trajectory.actions[1:]),
            remaining_action_data=list(trajectory.action_data[1:]),
            current_index=1,
            adaptation_rules={
                "goal_family": goal_context.goal_family if goal_context is not None else "unknown",
                "goal_hypothesis_source": goal_hypothesis.source if goal_hypothesis is not None else "heuristic",
            },
            contradiction_score=0.0,
            max_commit_steps=3,
            committed_steps=0,
            origin_level=self.state.last_observation.level if self.state.last_observation is not None else 0,
        )

    # ── UPDATE (online training phase) ───────────────────────────
    def _phase_update(
        self, obs: GameObservation, available: List[str]
    ) -> Dict[str, Any]:
        """Run online model training then delegate to strategize."""
        wm_loss = (
            self.world_model.update(train_steps=5)
            if self.config.uses_jepa_ebm() and self.config.train_jepa_online
            else 0.0
        )
        ebm_loss = (
            self.scorer.update(train_steps=5)
            if self.config.uses_jepa_ebm() and self.config.train_ebm_online
            else 0.0
        )
        logger.debug(f"UPDATE phase: WM={wm_loss:.4f}, EBM={ebm_loss:.4f}")
        self.state.phase = Phase.STRATEGIZE
        return self._phase_strategize(obs, available)

    # ==================================================================
    # Subgoal management
    # ==================================================================
    def _advance_subgoal(self) -> None:
        """Move to the next pending subgoal."""
        goal = self.state.goal
        if goal is None:
            self.state.current_subgoal = None
            return

        # Mark current subgoal as active → pending if not terminal
        next_sg = goal.current_subgoal
        if next_sg is not None:
            next_sg.status = SubGoalStatus.ACTIVE
            self.state.current_subgoal = next_sg
            self.state.subgoal_start_action = self.state.action_counter
            self.state.current_strategy = None  # need new strategy
            self.state.current_trajectory = None
            self.state.current_goal_context = None
            self.state.current_goal_hypothesis = None
            self.state.current_latent_task_program = None
            self.state.current_continuation = None
            self.state.pending_continuation = None
            self.actioner._on_subgoal_change(next_sg.id)
            logger.info(f"Advancing to SG{next_sg.id}: {next_sg.description}")
        else:
            self.state.current_subgoal = None
            self.state.current_trajectory = None
            self.state.current_goal_context = None
            self.state.current_goal_hypothesis = None
            self.state.current_latent_task_program = None
            self.state.current_continuation = None
            self.state.pending_continuation = None
            logger.info("All subgoals complete or exhausted")

    def mark_subgoal_achieved(self) -> None:
        """Called externally when progress indicates subgoal success."""
        sg = self.state.current_subgoal
        if sg is not None:
            sg.status = SubGoalStatus.ACHIEVED
            self.state.subgoals_completed += 1
            logger.info(f"SG{sg.id} achieved!")
            self._advance_subgoal()

    # ==================================================================
    # Feedback — called by the agent after observing action results
    # ==================================================================
    def record_result(
        self,
        obs_before: GameObservation,
        obs_after: GameObservation,
        strategy: Optional[GameStrategy],
        level_changed: bool,
        game_over: bool,
        new_states: int,
        trajectory: Optional[SampledTrajectory] = None,
        goal_context: Optional[GoalContext] = None,
    ) -> None:
        """Record the outcome and update models."""
        if strategy is None and trajectory is None:
            return

        feedback_strategy = strategy
        if strategy is not None and trajectory is not None and strategy.action_plan:
            feedback_metadata = dict(strategy.metadata)
            if feedback_metadata.get("click_targets"):
                feedback_metadata["click_targets"] = feedback_metadata["click_targets"][:1]
            feedback_strategy = GameStrategy(
                strategy_type=strategy.strategy_type,
                description=f"{strategy.description} [executed-prefix]",
                action_plan=strategy.action_plan[:1],
                rationale=strategy.rationale,
                confidence=strategy.confidence,
                metadata=feedback_metadata,
            )

        # Record in world model buffer
        if self.config.uses_jepa_ebm() and feedback_strategy is not None:
            self.world_model.record_transition(
                obs_before=obs_before,
                strategy=feedback_strategy,
                obs_after=obs_after,
                level_changed=level_changed,
                game_over=game_over,
                states_discovered=new_states,
            )

        resolved_goal_context = goal_context
        if resolved_goal_context is None and trajectory is not None and trajectory.goal_context is not None:
            resolved_goal_context = trajectory.goal_context
        if resolved_goal_context is None:
            resolved_goal_context = self._build_goal_context(obs_before)

        progress_score, progress_components = self.goal_pursuit.measure_goal_context_progress(
            resolved_goal_context,
            grid_before=obs_before.raw_grid,
            grid_after=obs_after.raw_grid,
            levels_before=obs_before.level,
            levels_after=obs_after.level,
            states_before=int(obs_before.memory_summary.get("states_visited", 0)),
            states_after=int(obs_after.memory_summary.get("states_visited", 0)),
            game_over=game_over,
        )

        was_good = level_changed or progress_score >= PARTIAL_THRESHOLD or new_states > 0
        if was_good:
            self.state.strategies_succeeded += 1
            if level_changed or progress_score >= SUCCESS_THRESHOLD:
                self.mark_subgoal_achieved()

        if trajectory is not None:
            predicted_first = None
            if trajectory.predicted_observations:
                predicted_first = trajectory.predicted_observations[0]
            executed_action = (
                strategy.action_plan[0]
                if strategy and strategy.action_plan
                else (trajectory.actions[0] if trajectory.actions else "")
            )
            prediction_match = self._prediction_match(predicted_first, obs_after.raw_grid)
            prefix_executed = [{
                "action": executed_action,
                "action_data": strategy.metadata.get("click_targets", [None])[0]
                if strategy and strategy.metadata.get("click_targets")
                else (trajectory.action_data[0] if trajectory.action_data else None),
            }]
            traj_outcome = TrajectoryOutcome(
                prefix_executed=prefix_executed,
                observed_after=obs_after.raw_grid.copy(),
                progress_delta=progress_score,
                prediction_match=prediction_match,
                game_over=game_over,
                levels_delta=max(0, obs_after.level - obs_before.level),
                source=trajectory.source,
                goal_context=resolved_goal_context,
                metadata={"progress_components": progress_components},
            )
            self.goal_pursuit.record_trajectory_outcome(traj_outcome)
            state_embedding = None
            if self.config.uses_jepa_ebm():
                z_before = self.world_model.encode_observation(obs_before)
                state_embedding = z_before.squeeze(0).detach().cpu().numpy()
            self.trajectory_memory.store(
                traj_outcome,
                state_embedding=state_embedding,
            )
            self.trajectory_memory.update_prior_trust(
                trajectory.source,
                prediction_match=prediction_match,
                progress_delta=progress_score,
            )
            if self.state.current_continuation is not None:
                self.state.current_continuation.on_feedback(
                    progress_delta=progress_score,
                    prediction_match=prediction_match,
                    levels_delta=max(0, obs_after.level - obs_before.level),
                    game_over=game_over,
                )
                if not self.state.current_continuation.active():
                    self.state.current_continuation = None
            if trajectory.source == "continuation":
                if was_good and self.state.pending_continuation is not None:
                    self.state.current_continuation = self.state.pending_continuation
                elif not was_good:
                    self.state.current_continuation = None
            elif was_good and self.state.pending_continuation is not None:
                self.state.current_continuation = self.state.pending_continuation
            if progress_score < 0.05:
                self.trajectory_memory.remember_failed_prefix(trajectory.actions)
            self._export_trajectory_prior(traj_outcome)
            self._consume_post_level_switch_guard(
                executed_action,
                resolved_goal_context,
                game_over=game_over,
            )

        self.state.pending_continuation = None

        if self.config.uses_jepa_ebm() and feedback_strategy is not None:
            z_t = self.world_model.encode_observation(obs_before)
            s_emb = self.world_model.strategy_encoder.encode_strategy(
                feedback_strategy, self.world_model.device
            )
            z_hat = self.world_model.predictor(z_t, s_emb)
            aux = self.world_model.aux_heads(z_hat)
            self.scorer.record_goal_feedback(z_t, s_emb, z_hat, aux, progress_score)

    # ==================================================================
    # Event handlers
    # ==================================================================
    def on_level_change(self, new_level: int) -> None:
        """Called when a level is completed — re-explore + re-decompose."""
        logger.info(f"Level completed → {new_level}. Re-decomposing.")
        self.state.goal = None
        self.state.current_subgoal = None
        self.state.current_strategy = None
        self.state.current_trajectory = None
        self.state.current_goal_context = None
        self.state.current_goal_hypothesis = None
        if self.state.current_continuation is not None:
            self.state.current_continuation.carry_into_level(new_level)
        self.state.pending_continuation = None
        self._post_level_switch_guard_remaining = 0
        self.state.phase = Phase.EXPLORE
        self._explore_idx = 0
        self.state.last_progress_action = self.state.action_counter

    def new_iteration(self) -> None:
        """Reset phase state for a new game iteration.

        Memory and model weights are preserved — only the planning
        state is cleared so the agent re-explores with accumulated
        knowledge.
        """
        # End previous episode (consolidation: LTP/LTD)
        if self.assoc_memory._current_episode is not None:
            self.assoc_memory.end_episode()
        # Begin new episode
        self.assoc_memory.begin_episode()
        self._assoc_recent_actions = []
        self._prev_levels_completed = 0

        self.state.phase = Phase.EXPLORE
        self.state.action_counter = 0
        self.state.goal = None
        self.state.current_subgoal = None
        self.state.current_strategy = None
        self.state.current_trajectory = None
        self.state.current_goal_context = None
        self.state.current_goal_hypothesis = None
        self.state.current_latent_task_program = None
        self.state.current_continuation = None
        self.state.pending_continuation = None
        self.state.last_observation = None
        self.state.last_scoring = None
        self.state.last_trajectory_scoring = None
        self.state.last_progress_action = 0
        self._explore_idx = 0
        self._prev_grid = None
        self._prev_obs = None
        # Reset actioner per-subgoal state but keep game-over positions
        self.actioner._current_subgoal_id = None
        self.actioner._step_in_subgoal = 0
        self.actioner._nav_target_idx = 0
        self.actioner._click_target_idx = 0
        self.actioner._plan_idx = 0
        self.actioner._stuck_counter = 0
        self.actioner._explore_idx = 0
        self.actioner._blocked_actions.clear()
        self.actioner._clicked_positions.clear()
        # Keep _gameover_positions — accumulated knowledge!

    def on_game_over(self) -> None:
        """Called on game over — re-explore and revise goals."""
        logger.info("Game over. Re-exploring with revised goals.")
        self.actioner.on_game_over()
        # Keep the old goal so decomposer can revise it
        self.state.current_subgoal = None
        self.state.current_strategy = None
        self.state.current_trajectory = None
        self.state.current_goal_context = None
        self.state.current_goal_hypothesis = None
        self.state.current_latent_task_program = None
        self.state.current_continuation = None
        self.state.pending_continuation = None
        self.state.phase = Phase.EXPLORE
        self._explore_idx = 0

    # ==================================================================
    # Stats
    # ==================================================================
    def get_stats(self) -> Dict[str, Any]:
        """Return reasoning loop statistics."""
        goal = self.state.goal
        sg = self.state.current_subgoal
        return {
            "reasoning_mode": self.config.reasoning_mode,
            "ablation_stage": self.config.ablation_stage,
            "enabled_features": self.config.enabled_features(),
            "phase": self.state.phase.value,
            "action_counter": self.state.action_counter,
            "strategies_tried": self.state.strategies_tried,
            "strategies_succeeded": self.state.strategies_succeeded,
            "subgoals_completed": self.state.subgoals_completed,
            "decompositions": self.state.decompositions,
            "overarching_goal": goal.overarching_goal if goal else None,
            "goal_progress": f"{goal.progress_fraction:.0%}" if goal else "0%",
            "current_subgoal": sg.description[:80] if sg else None,
            "current_strategy": (
                self.state.current_strategy.strategy_type.value
                if self.state.current_strategy else None
            ),
            "current_goal_hypothesis": (
                self.state.current_goal_hypothesis.to_dict()
                if self.state.current_goal_hypothesis is not None else None
            ),
            "current_latent_task_program": (
                {
                    "id": self.state.current_latent_task_program.objective_id,
                    "description": self.state.current_latent_task_program.description,
                    "preferred_actions": list(
                        self.state.current_latent_task_program.preferred_actions
                    ),
                    "preferred_sequences": [
                        list(seq)
                        for seq in self.state.current_latent_task_program.preferred_sequences[:3]
                    ],
                    "confidence": self.state.current_latent_task_program.confidence,
                }
                if self.state.current_latent_task_program is not None else None
            ),
            "current_trajectory_source": (
                self.state.current_trajectory.source
                if self.state.current_trajectory else None
            ),
            "current_continuation": (
                {
                    "remaining_actions": list(self.state.current_continuation.remaining_actions),
                    "contradiction_score": self.state.current_continuation.contradiction_score,
                    "origin_level": self.state.current_continuation.origin_level,
                }
                if self.state.current_continuation is not None else None
            ),
            "pending_continuation": (
                list(self.state.pending_continuation.remaining_actions)
                if self.state.pending_continuation is not None else None
            ),
            "last_trajectory_debug": self.state.last_trajectory_debug,
            "wm_transitions": len(self.world_model._transition_buffer),
            "ebm_feedback": len(self.scorer._feedback_buffer),
            "wm_trained_steps": self.world_model._trained_steps,
            "assoc_memory": self.assoc_memory.stats(),
            "trajectory_memory": self.trajectory_memory.stats(),
            "visual_cortex": self.visual_cortex.stats(),
            "goal_pursuit": self.goal_pursuit.stats(),
        }

    # ==================================================================
    # Goal pursuit integration
    # ==================================================================
    def generate_goal_bank(
        self, obs: GameObservation, available_actions: List[str],
    ) -> List[GameObjective]:
        """Generate a bank of plausible objectives from current knowledge.

        Uses GoalDecomposer.generate_goal_bank() with LLM + template fallback.
        Previous strategy outcomes are passed as context for refinement.
        """
        previous_outcomes = [
            o.to_dict() for o in self.goal_pursuit._all_outcomes[-10:]
        ] if self.goal_pursuit._all_outcomes else None

        bank = self.decomposer.generate_goal_bank(
            observation=obs,
            memory=self.memory,
            previous_outcomes=previous_outcomes,
        )
        self.goal_pursuit.set_goal_bank(bank)
        return bank

    def strategize_for_goal(
        self, obs: GameObservation, available_actions: List[str],
    ) -> Optional[GameStrategy]:
        """Generate and score goal-conditioned strategies.

        1. Get context from GoalProgressManager (goal + failure history)
        2. Generate strategies conditioned on the active goal
        3. Score goal-strategy pairs with JEPA + EBM
        4. Return the best strategy

        Returns None if no goal is active.
        """
        goal = self.goal_pursuit.active_goal
        if goal is None:
            return None

        ctx = self.goal_pursuit.strategy_context()

        # Visual cortex: predict action effects and feed downstream
        if (
            self.config.uses_visual_cortex()
            and self.visual_cortex.trained_steps >= 10
            and obs.raw_grid is not None
        ):
            avail_ints = []
            for a in available_actions:
                try:
                    avail_ints.append(int(a.replace("ACTION", "")))
                except ValueError:
                    pass
            if avail_ints:
                action_summary = self.visual_cortex.get_action_summary(
                    obs.raw_grid, avail_ints,
                )
                obs.visual_cortex_summary = action_summary

                vc_analysis = self.visual_cortex.analyze_action_effects(
                    obs.raw_grid, avail_ints,
                )
                vc_similarity = self.visual_cortex.compute_action_similarity(
                    obs.raw_grid, avail_ints,
                )
                self.assoc_memory.ingest_vc_predictions(
                    vc_analysis, vc_similarity, obs.raw_grid,
                )
                for act_int, info in vc_analysis.items():
                    direction = info.get("direction")
                    if direction is not None:
                        act_name = f"ACTION{act_int}"
                        if act_name not in self.memory.action_directions:
                            self.memory.vc_directions[act_name] = direction

        # Generate goal-conditioned strategies
        strategies = self.generator.generate_for_goal(
            observation=obs,
            available_actions=available_actions,
            goal=goal,
            failed_strategies=ctx.get("failed_strategies", []),
            partial_strategies=ctx.get("partial_strategies", []),
        )
        if not strategies:
            return None

        logger.info(
            f"STRATEGIZE for goal '{goal.id}' (attempt {goal.attempts+1}): "
            f"{len(strategies)} candidates"
        )

        # ── Action-effect-driven scoring (same as _phase_strategize) ──
        if not self.config.uses_jepa_ebm():
            scored = [
                (self._score_strategy_symbolically(strategy, obs, available_actions), strategy)
                for strategy in strategies
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = scored[0][1]
            self.state.last_scoring = None
            self.state.current_strategy = selected
            self.state.strategy_start_action = self.state.action_counter
            self.state.strategies_tried += 1
            return selected

        z_t = self.world_model.encode_observation(obs)
        z_hats, aux_list = self.world_model.predict_strategy_outcomes(z_t, strategies)
        s_embs = self.world_model.strategy_encoder.encode_batch(
            strategies, self.world_model.device
        )
        decision = self.scorer.score_for_goal(
            z_t, strategies, s_embs, z_hats, aux_list, goal,
        )
        self.state.last_scoring = decision

        # Score each strategy by action plan quality + mechanism locking
        locked = self.memory.get_locked_mechanisms()
        strategy_scores = []
        for s in strategies:
            action_score = 0.0
            if s.action_plan:
                scores = [self.memory.score_action(a) for a in s.action_plan
                          if a in available_actions]
                action_score = sum(scores) / max(len(scores), 1)
            lock_bonus = 0.0
            if locked and s.action_plan:
                locked_used = sum(1 for a in s.action_plan if a in locked)
                lock_bonus = 0.3 * (locked_used / max(len(s.action_plan), 1))
            conf = s.confidence
            ebm_idx = strategies.index(s)
            ebm_score = 0.1 * (1.0 / (1.0 + abs(decision.energies[ebm_idx]))) \
                if hasattr(decision, 'energies') and ebm_idx < len(decision.energies) else 0.0
            total = 0.40 * action_score + 0.25 * lock_bonus + 0.25 * conf + 0.10 * ebm_score
            strategy_scores.append(total)

        best_idx = max(range(len(strategies)), key=lambda i: strategy_scores[i])
        selected = strategies[best_idx]

        logger.info(
            f"Strategy → {selected.strategy_type.value} for '{goal.id}' "
            f"(score={strategy_scores[best_idx]:.3f}): {selected.description[:80]}"
        )

        self.state.current_strategy = selected
        self.state.strategy_start_action = self.state.action_counter
        self.state.strategies_tried += 1

        return selected

    # ==================================================================
    # Trajectory planning helpers
    # ==================================================================
    def _build_goal_context(
        self,
        obs: GameObservation,
        goal_hypothesis: Optional[GoalHypothesis] = None,
    ) -> GoalContext:
        """Construct a goal-conditioned planning context for sampling."""
        subgoal = self.state.current_subgoal
        goal = self.state.goal
        metadata = dict(subgoal.metadata) if subgoal is not None else {}

        if goal_hypothesis is None:
            goal_hypothesis = self.goal_recognizer.predict(
                obs,
                memory=self.memory,
                current_goal=goal,
                current_subgoal=subgoal,
                task_program=getattr(self.decomposer, "_task_program", None),
            )
        goal_family = goal_hypothesis.family
        objective_id = metadata.get("task_program_id")
        if not objective_id:
            objective_id = (
                f"sg_{subgoal.id}_{subgoal.description[:32].replace(' ', '_')}"
                if subgoal is not None else "explore"
            )
        progress_signals: List[str] = []
        if subgoal is not None and subgoal.success_hint:
            progress_signals.append(subgoal.success_hint)
        expected_signal = metadata.get("expected_signal")
        if expected_signal:
            progress_signals.append(str(expected_signal))

        anti_signals: List[str] = []
        if goal is not None and "avoid:" in goal.hypothesis:
            anti_signals.append(goal.hypothesis.split("avoid:", 1)[1][:120])
        preferred_actions = list(metadata.get("prefer_actions", []) or [])
        click_targets = list(metadata.get("click_targets", []) or [])
        human_weight = (
            0.0
            if str(self.config.sampler_stage).lower() == "v0"
            else self.trajectory_memory.human_prior_weight(self.state.action_counter)
        )
        recent_actions = [
            effect.action_name
            for effect in self.memory.action_history[-4:]
            if getattr(effect, "action_name", None)
        ]
        if recent_actions:
            metadata["recent_actions"] = recent_actions
            metadata["last_action"] = recent_actions[-1]
        latent_pref = TrajectorySampler._normalize_actions(
            metadata.get("latent_preferred_actions", [])
        )
        if latent_pref:
            override_static = bool(metadata.get("latent_override_static_program", False))
            existing_pref = TrajectorySampler._normalize_actions(preferred_actions)
            if override_static:
                preferred_actions = latent_pref + [
                    action for action in existing_pref if action not in latent_pref
                ]
            else:
                preferred_actions = existing_pref + [
                    action for action in latent_pref if action not in existing_pref
                ]
        latent_avoid = TrajectorySampler._normalize_actions(
            metadata.get("latent_avoid_actions", [])
        )
        if latent_avoid:
            anti_signals.append("latent_avoid: " + ",".join(latent_avoid[:3]))
        latent_sequences = TrajectorySampler._latent_sequences(
            GoalContext(
                goal_family=goal_family,
                objective_id=str(objective_id),
                preferred_actions=preferred_actions,
                metadata=metadata,
            )
        )
        if latent_sequences:
            metadata["latent_preferred_sequences"] = latent_sequences
            metadata["latent_macro_actions"] = list(latent_sequences[0])
            for action in latent_sequences[0]:
                if action not in preferred_actions:
                    preferred_actions.append(action)
        metadata["goal_hypothesis"] = goal_hypothesis.to_dict()
        metadata["target_objects"] = list(goal_hypothesis.target_objects)
        metadata["relevant_colors"] = list(goal_hypothesis.relevant_colors)
        metadata["goal_hypothesis_source"] = goal_hypothesis.source
        metadata["current_level"] = int(getattr(obs, "level", 0) or 0)
        preferred_actions = list(dict.fromkeys(preferred_actions))

        return GoalContext(
            goal_family=goal_family,
            objective_id=str(objective_id),
            progress_signals=progress_signals,
            anti_signals=anti_signals,
            source_confidence=float(goal_hypothesis.confidence),
            human_prior_weight=human_weight,
            preferred_actions=preferred_actions,
            click_targets=click_targets,
            metadata=metadata,
        )

    def _install_latent_task_program(self, program: LatentTaskProgram) -> None:
        """Install a scorer-generated program into the active subgoal metadata."""
        if not self._allow_latent_task_programmer():
            return
        if program.confidence < 0.20:
            return

        self.state.current_latent_task_program = program
        subgoal = self.state.current_subgoal
        if subgoal is None:
            return

        metadata = dict(getattr(subgoal, "metadata", {}) or {})
        patch = program.metadata_patch()
        latent_pref = TrajectorySampler._normalize_actions(program.preferred_actions)
        existing_pref = TrajectorySampler._normalize_actions(
            metadata.get("prefer_actions", [])
        )
        if bool(patch.get("latent_override_static_program", False)):
            merged_pref = latent_pref + [
                action for action in existing_pref if action not in latent_pref
            ]
        else:
            merged_pref = existing_pref + [
                action for action in latent_pref if action not in existing_pref
            ]

        metadata.update(patch)
        if merged_pref:
            metadata["prefer_actions"] = merged_pref[:7]
        if program.preferred_sequences:
            metadata["latent_preferred_sequences"] = [
                list(seq) for seq in program.preferred_sequences[:4]
            ]
            metadata["latent_macro_actions"] = list(program.preferred_sequences[0])
        subgoal.metadata = metadata

    def _goal_family_from_state(self, obs: GameObservation) -> str:
        """Best-effort goal-family inference for trajectory retrieval."""
        program = getattr(self.decomposer, "_task_program", None)
        if program is not None:
            family = getattr(program, "goal_family", None)
            if family:
                return str(family)
        if self.state.goal is not None and "goal_family=" in self.state.goal.hypothesis:
            frag = self.state.goal.hypothesis.split("goal_family=", 1)[1]
            return frag.split(" ", 1)[0].split("|", 1)[0].strip("_| ")
        priors = self.decomposer._extract_human_priors(self.memory)
        gt_prior = priors.get("game_type")
        if gt_prior:
            return str(gt_prior[0])
        if obs.player_info and obs.objects:
            return "navigation"
        if "ACTION6" in obs.action_semantics:
            return "click_puzzle"
        return "unknown"

    def _symbolic_aux_for_trajectory(
        self,
        obs: GameObservation,
        goal_context: GoalContext,
        trajectory: SampledTrajectory,
    ) -> _SymbolicAux:
        del obs
        actions = [a for a in trajectory.actions if a]
        if not actions:
            return _SymbolicAux(
                _ScalarAuxValue(0.0),
                _ScalarAuxValue(1.0),
                _ScalarAuxValue(0.0),
            )

        action_scores = [self.memory.score_action(action) for action in actions]
        action_quality = sum(action_scores) / max(len(action_scores), 1)
        preferred = set(goal_context.preferred_actions)
        preferred_fraction = (
            sum(1 for action in actions if action in preferred) / max(len(actions), 1)
            if preferred else 0.0
        )
        untried_fraction = sum(
            1 for action in actions
            if action not in self.memory.action_profiles
        ) / max(len(actions), 1)

        death_rates = []
        no_effect_rates = []
        for action in actions:
            profile = self.memory.action_profiles.get(action)
            if profile is None:
                death_rates.append(0.0)
                no_effect_rates.append(0.0)
                continue
            death_rates.append(profile.times_caused_game_over / max(profile.times_tried, 1))
            no_effect_rates.append(profile.times_no_effect / max(profile.times_tried, 1))
        death_risk = sum(death_rates) / max(len(death_rates), 1)
        no_effect_risk = sum(no_effect_rates) / max(len(no_effect_rates), 1)

        source_bonus = 0.0
        if trajectory.source in {"prior", "trajectory_memory"}:
            source_bonus += 0.08
        if trajectory.source == "hypothesis":
            source_bonus += 0.05 * float(
                trajectory.metadata.get("hypothesis_experiment_score", 0.0)
            )

        progress = (
            0.20
            + 0.45 * max(0.0, min(1.0, action_quality))
            + 0.25 * preferred_fraction
            + source_bonus
        )
        novelty = 0.10 + 0.45 * untried_fraction + 0.15 * max(0.0, 0.5 - action_quality)
        risk = 0.65 * death_risk + 0.25 * max(0.0, no_effect_risk - 0.55)
        return _SymbolicAux(
            _ScalarAuxValue(max(0.0, min(1.0, progress))),
            _ScalarAuxValue(max(0.0, min(1.0, risk))),
            _ScalarAuxValue(max(0.0, min(1.0, novelty))),
        )

    def _score_trajectories_symbolically(
        self,
        trajectories: List[SampledTrajectory],
    ) -> TrajectoryScoringDecision:
        scores: List[float] = []
        energies: List[float] = []
        for trajectory in trajectories:
            goal_progress = float(trajectory.metadata.get("goal_progress", 0.0))
            novelty = float(trajectory.metadata.get("novelty", 0.0))
            risk = float(trajectory.metadata.get("risk", 0.0))
            human_compatibility = float(
                trajectory.metadata.get("human_compatibility", 0.0)
            )
            generator_confidence = float(
                trajectory.metadata.get("generator_confidence", 0.5)
            )
            source_bonus = 0.0
            if trajectory.source == "trajectory_memory":
                source_bonus += 0.20
            elif trajectory.source == "prior":
                source_bonus += 0.08
            elif trajectory.source == "hypothesis":
                source_bonus += 0.06

            score = (
                1.20 * goal_progress
                + 0.45 * novelty
                + 0.30 * human_compatibility
                + 0.25 * generator_confidence
                + source_bonus
                - 0.85 * risk
            )
            trajectory.score = score
            trajectory.energy = -score
            trajectory.metadata["symbolic_score"] = score
            scores.append(score)
            energies.append(-score)

        sorted_indices = sorted(range(len(trajectories)), key=lambda i: scores[i], reverse=True)
        top_k_idx = sorted_indices[: min(3, len(sorted_indices))]
        return TrajectoryScoringDecision(
            selected_idx=sorted_indices[0],
            selected_energy=energies[sorted_indices[0]],
            selected_score=scores[sorted_indices[0]],
            all_energies=energies,
            all_scores=scores,
            selected_trajectory=trajectories[sorted_indices[0]],
            top_k_indices=top_k_idx,
        )

    def _rerank_with_symbolic_core(
        self,
        trajectories: List[SampledTrajectory],
        learned_decision: TrajectoryScoringDecision,
    ) -> TrajectoryScoringDecision:
        symbolic_decision = self._score_trajectories_symbolically(trajectories)
        symbolic_scores = list(symbolic_decision.all_scores)
        learned_scores = list(learned_decision.all_scores)

        def _normalise(values: List[float]) -> List[float]:
            if not values:
                return []
            lo = min(values)
            hi = max(values)
            if abs(hi - lo) < 1e-9:
                return [0.5 for _ in values]
            return [(value - lo) / (hi - lo) for value in values]

        symbolic_norm = _normalise(symbolic_scores)
        learned_norm = _normalise(learned_scores)
        weight = self.config.jepa_ebm_rerank_weight
        final_scores = [
            (1.0 - weight) * symbolic_norm[i] + weight * learned_norm[i]
            for i in range(len(trajectories))
        ]
        for i, trajectory in enumerate(trajectories):
            trajectory.score = final_scores[i]
            trajectory.energy = learned_decision.all_energies[i]
            trajectory.metadata["symbolic_score"] = symbolic_scores[i]
            trajectory.metadata["learned_score"] = learned_scores[i]
            trajectory.metadata["rerank_score"] = final_scores[i]
            trajectory.metadata["jepa_ebm_rerank_weight"] = weight

        sorted_indices = sorted(range(len(trajectories)), key=lambda i: final_scores[i], reverse=True)
        top_k_idx = sorted_indices[: min(3, len(sorted_indices))]
        return TrajectoryScoringDecision(
            selected_idx=sorted_indices[0],
            selected_energy=learned_decision.all_energies[sorted_indices[0]],
            selected_score=final_scores[sorted_indices[0]],
            all_energies=list(learned_decision.all_energies),
            all_scores=final_scores,
            selected_trajectory=trajectories[sorted_indices[0]],
            top_k_indices=top_k_idx,
        )

    def _trajectory_heuristics(
        self,
        obs: GameObservation,
        goal_context: GoalContext,
        trajectory: SampledTrajectory,
        aux: Any,
    ) -> Dict[str, float]:
        """Heuristics layered on top of learned energy for trajectory scoring."""
        metadata = goal_context.metadata or {}

        def _metadata_action_set(key: str) -> set[str]:
            raw = metadata.get(key, [])
            if isinstance(raw, str):
                raw = [raw]
            return set(TrajectorySampler._normalize_actions(list(raw or [])))

        preferred = set(goal_context.preferred_actions)
        control_switch_actions = _metadata_action_set("control_switch_actions")
        movement_actions = _metadata_action_set("movement_actions")
        recent_actions = TrajectorySampler._normalize_actions(
            metadata.get("recent_actions", [])
            if not isinstance(metadata.get("recent_actions", []), str)
            else [metadata.get("recent_actions", "")]
        )
        action_scores = [self.memory.score_action(a) for a in trajectory.actions]
        action_quality = sum(action_scores) / max(len(action_scores), 1)
        preferred_fraction = max(
            float(trajectory.metadata.get("preferred_fraction", 0.0)),
            sum(1 for action in trajectory.actions if action in preferred) / len(trajectory.actions)
            if preferred and trajectory.actions
            else 0.0
        )
        base_human_compat = float(trajectory.metadata.get("human_compatibility", 0.0))
        if str(self.config.sampler_stage).lower() == "v0":
            human_compatibility = 0.0
        else:
            human_compatibility = min(
                0.25,
                base_human_compat * max(0.0, goal_context.human_prior_weight),
            )
        goal_progress = float(aux.progress_prob.item()) + 0.20 * preferred_fraction
        novelty = float(aux.novelty_score.item())
        risk = float(aux.risk_prob.item())
        generator_confidence = max(0.05, min(0.95, 0.5 + 0.4 * action_quality))

        first_action = trajectory.actions[0] if trajectory.actions else ""
        expected_signal = str(metadata.get("expected_signal", "") or "").lower()
        progress_text = " ".join(str(s).lower() for s in goal_context.progress_signals)
        switch_probe_expected = (
            expected_signal == "role_switch" or "role switch" in progress_text
        )
        movement_or_level_expected = (
            expected_signal in {"level_advance", "object_moved", "overlap_achieved"}
            or "level" in progress_text
            or "move" in progress_text
        )
        switch_repeat_penalty = 0.0
        switch_followup_bonus = 0.0
        if control_switch_actions and trajectory.actions:
            recent_switches = [
                action for action in recent_actions[-4:]
                if action in control_switch_actions
            ]
            last_was_switch = bool(recent_actions and recent_actions[-1] in control_switch_actions)
            switch_lead = first_action in control_switch_actions
            switch_density = (
                sum(1 for action in trajectory.actions if action in control_switch_actions)
                / max(len(trajectory.actions), 1)
            )

            if switch_lead and not switch_probe_expected:
                if last_was_switch:
                    switch_repeat_penalty += 0.55
                elif len(recent_switches) >= 2:
                    switch_repeat_penalty += 0.35
                if switch_density > 0.5:
                    switch_repeat_penalty += 0.20
                if movement_or_level_expected:
                    switch_repeat_penalty += 0.10
            elif last_was_switch and movement_or_level_expected:
                if not movement_actions or first_action in movement_actions:
                    switch_followup_bonus = 0.12

        if switch_repeat_penalty > 0.0:
            risk += switch_repeat_penalty
            goal_progress -= 0.25 * switch_repeat_penalty
            generator_confidence = max(
                0.05,
                generator_confidence - 0.35 * switch_repeat_penalty,
            )
        if switch_followup_bonus > 0.0:
            goal_progress += switch_followup_bonus
            novelty += 0.03
            generator_confidence = min(
                0.95,
                generator_confidence + 0.08,
            )
        trajectory.metadata["control_switch_repeat_penalty"] = switch_repeat_penalty
        trajectory.metadata["control_switch_followup_bonus"] = switch_followup_bonus

        if trajectory.source == "prior":
            goal_progress += 0.05
        if trajectory.source == "latent_program":
            latent_confidence = float(
                trajectory.metadata.get("latent_program_confidence", 0.0)
            )
            macro_len = len(trajectory.metadata.get("latent_macro_sequence", []) or [])
            goal_progress += 0.06 + 0.08 * min(1.0, max(0.0, latent_confidence))
            novelty += 0.02 * min(3, max(0, macro_len - 1))
            risk = max(0.0, risk - 0.03 * min(1.0, max(0.0, latent_confidence)))
            generator_confidence = min(
                0.95,
                generator_confidence
                + 0.06 * min(1.0, max(0.0, latent_confidence)),
            )
        if trajectory.source == "hypothesis":
            info_gain = float(trajectory.metadata.get("hypothesis_information_gain", 0.0))
            alignment = float(trajectory.metadata.get("hypothesis_goal_alignment", 0.0))
            confidence = float(trajectory.metadata.get("hypothesis_confidence", 0.0))
            hypothesis_risk = float(trajectory.metadata.get("hypothesis_risk", 0.0))
            experiment_score = float(
                trajectory.metadata.get("hypothesis_experiment_score", 0.0)
            )
            coverage_bonus = float(
                trajectory.metadata.get("hypothesis_coverage_bonus", 0.0)
            )
            recent_penalty = float(
                trajectory.metadata.get("hypothesis_recent_penalty", 0.0)
            )
            support_saturation = float(
                trajectory.metadata.get("hypothesis_support_saturation", 0.0)
            )
            goal_progress += 0.12 * alignment + 0.05 * confidence + 0.10 * experiment_score
            novelty += 0.40 * info_gain + 0.25 * coverage_bonus + 0.25 * experiment_score
            risk += 0.30 * hypothesis_risk + 0.50 * recent_penalty + 0.20 * support_saturation
            generator_confidence = max(
                0.05,
                min(
                    0.95,
                    0.35
                    + 0.35 * experiment_score
                    + 0.12 * alignment
                    + 0.08 * confidence
                    - 0.25 * recent_penalty
                    - 0.15 * support_saturation,
                ),
            )

        if trajectory.predicted_observations:
            pred = trajectory.predicted_observations[-1]
            if not self.memory.is_state_visited(pred):
                novelty += 0.25

        risk += max(0.0, 0.35 - action_quality)
        return {
            "goal_progress": max(0.0, min(1.0, goal_progress)),
            "novelty": max(0.0, min(1.25, novelty)),
            "risk": max(0.0, min(1.5, risk)),
            "human_compatibility": human_compatibility,
            "generator_confidence": generator_confidence,
        }

    @staticmethod
    def _prediction_match(
        predicted: Optional[np.ndarray],
        observed: np.ndarray,
    ) -> float:
        """How closely a predicted frame matches the observed next frame."""
        if predicted is None or observed is None:
            return 0.0
        if predicted.shape != observed.shape:
            h = min(predicted.shape[0], observed.shape[0])
            w = min(predicted.shape[1], observed.shape[1])
            predicted = predicted[:h, :w]
            observed = observed[:h, :w]
        if predicted.size == 0:
            return 0.0
        diff = np.mean(predicted != observed)
        return max(0.0, min(1.0, 1.0 - float(diff)))

    def _export_trajectory_prior(self, outcome: TrajectoryOutcome) -> None:
        """Persist compact trajectory priors into cross-game memory."""
        if self._cross_game is None:
            return
        if outcome.progress_delta < 0.05 and outcome.levels_delta <= 0:
            return
        bucket = self._cross_game.trajectory_priors.setdefault(
            outcome.goal_context.goal_family,
            [],
        )
        exported = self.trajectory_memory.export_record(outcome)
        sig = (tuple(exported.get("actions", [])), exported.get("source"))
        existing = {
            (tuple(item.get("actions", [])), item.get("source"))
            for item in bucket if isinstance(item, dict)
        }
        if sig in existing:
            return
        bucket.append(exported)
        bucket.sort(key=lambda item: -float(item.get("progress_delta", 0.0)))
        del bucket[8:]

    # ==================================================================
    # Cross-game learning
    # ==================================================================
    def export_cross_game(self) -> None:
        """Export this game's learnings to the cross-game memory."""
        if self._cross_game is not None:
            self.assoc_memory.export_to_cross_game(self._cross_game)

    # ==================================================================
    # Associative memory interface
    # ==================================================================
    def retrieve_from_memory(
        self,
        grid: Optional[np.ndarray],
        available_actions: List[int],
        temperature: float = 1.0,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Query associative memory for the best action given current state."""
        return self.assoc_memory.retrieve_action(
            grid, available_actions, self._assoc_recent_actions, temperature,
        )

    # ==================================================================
    # Result helper
    # ==================================================================
    def _result(
        self,
        action: str,
        action_data: Optional[Dict[str, Any]],
        strategy: Optional[GameStrategy],
        subgoal: Optional[SubGoal],
        phase: Phase,
        obs: GameObservation,
    ) -> Dict[str, Any]:
        return {
            "action": action,
            "action_data": action_data,
            "strategy": strategy,
            "trajectory": None,
            "subgoal": subgoal,
            "goal": self.state.goal,
            "goal_context": self.state.current_goal_context,
            "phase": phase.value,
            "observation": obs,
        }
