"""
Adaptive Reasoning Agent for ARC-AGI-3.

Bridges the v4_1_reasoning_system architecture with the ARC-AGI-3
game environment.  Implements the full adaptive reasoning loop:

  1. Observe  → StateDescriber produces natural-language observation
  2. Explore  → try actions, memorise effects (first N actions)
  3. Generate → StrategyGenerator (LLM / templates) produces candidates
  4. Predict  → GameWorldModel (JEPA) predicts latent outcomes
  5. Score    → GameEnergyScorer (EBM) ranks candidates by energy
  6. Execute  → run the winning strategy's action plan
  7. Learn    → record transition, update world model + EBM online

Designed to work offline (no internet) within Kaggle's 6-hour limit.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

# Add the project root to path so we can import v4_1_reasoning_system
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from v4_1_reasoning_system.arc_agi.grid_analyzer import GridAnalyzer, FrameDiff
from v4_1_reasoning_system.arc_agi.game_memory import GameMemory
from v4_1_reasoning_system.arc_agi.reasoning_loop import (
    AdaptiveReasoningLoop, LoopConfig, Phase,
)
from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory
from v4_1_reasoning_system.arc_agi.runtime_bootstrap import (
    DEFAULT_ABLATION_STAGE,
    DEFAULT_ENABLE_ONLINE_EBM_TRAINING,
    DEFAULT_ENABLE_ONLINE_JEPA_TRAINING,
    DEFAULT_ENABLE_VISUAL_CORTEX_WARMUP,
    DEFAULT_REASONING_MODE,
    build_adaptive_loop_config,
    resolve_checkpoint_path,
)
from theory.unified_cognitive_controller import UnifiedCognitiveController

logger = logging.getLogger(__name__)


# ── Action name / enum helpers ──────────────────────────────────────

_NAME_TO_ACTION = {
    "RESET": GameAction.RESET,
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
}

_INT_TO_ACTION_NAME = {
    0: "RESET",
    1: "ACTION1", 2: "ACTION2", 3: "ACTION3",
    4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}


def _normalize_action_name(a: Any) -> str:
    """Convert an available_action entry (int, enum, str) to 'ACTIONn' string."""
    if isinstance(a, int):
        return _INT_TO_ACTION_NAME.get(a, f"ACTION{a}")
    name = a.name if hasattr(a, "name") else str(a)
    return name


# ── Agent ───────────────────────────────────────────────────────────

class AdaptiveReasoning(Agent):
    """
    An ARC-AGI-3 agent powered by the v4_1 adaptive reasoning architecture.

    Lifecycle per game:
      1. RESET to start the game
      2. Run the unified adaptive loop (explore ↔ strategize)
      3. On GAME_OVER or stall: restart with accumulated knowledge
      4. Continue until WIN or TIME_BUDGET exhausted
    """

    TIME_BUDGET: float = 60.0   # total seconds for the entire game
    STALL_FRACTION: float = 0.25  # restart iteration if no progress for this fraction of remaining time
    MAX_WINS: int = 999           # outer-loop cap on total level-up counts (set high to keep pushing)
    LEVEL_UP_BONUS_SECONDS: float = 30.0  # after a level-up mid-iter, grant at least this much stall-free budget
    PRESERVE_ENV_ACROSS_ITERS: bool = True  # skip env rebuild + RESET when previous iter ended mid-progress
    LIVE_REVISE_AFTER_ITERS: int = 3       # iters of plateau (no level-up beyond best_level) before triggering a LIVE LLM revision; also the minimum gap between successive live revisions
    MAX_LIVE_REVISIONS: int = 2            # cap on live LLM compilations per game (each takes 30-90s on CPU, 5-15s on GPU)
    LIVE_REVISE_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"  # ~6 GB fp16; same id is used for the goal-decomposer LLM (see LoopConfig below) so the shared llm_cache loads it ONCE and both consumers reuse the same VRAM-resident copy.
    LIVE_REVISE_DEVICE: str = "auto"  # "auto" -> "cuda" if available else "cpu"; instance attr `_live_revise_device` carries the resolved value
    LIVE_REVISE_PAUSES_BUDGET: bool = True  # if True, the time spent in the LLM compile is added back to the game time budget (so the agent gets to actually try the new program)
    ENABLE_ONLINE_JEPA_TRAINING: bool = DEFAULT_ENABLE_ONLINE_JEPA_TRAINING
    ENABLE_ONLINE_EBM_TRAINING: bool = DEFAULT_ENABLE_ONLINE_EBM_TRAINING
    ENABLE_VISUAL_CORTEX_WARMUP: bool = DEFAULT_ENABLE_VISUAL_CORTEX_WARMUP
    REASONING_MODE: str = DEFAULT_REASONING_MODE
    ABLATION_STAGE: Optional[str] = DEFAULT_ABLATION_STAGE
    ENABLE_UNIFIED_COGNITION: bool = True

    def __init__(self, *args: Any, cross_game: Optional[CrossGameMemory] = None, arcade: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cross_game = cross_game
        self._arcade = arcade  # Arcade instance for creating fresh envs

        # Seed for reproducibility per game
        seed = int(time.time() * 1000) + hash(self.game_id) % 1000000
        random.seed(seed)

        # ── Build the reasoning loop ────────────────────────────
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loop_cfg = self._build_loop_config(device)
        self.memory = GameMemory()
        self.reasoning = AdaptiveReasoningLoop(
            config=loop_cfg,
            memory=self.memory,
            cross_game=self._cross_game,
        )
        self.analyzer = GridAnalyzer()

        # ── Frame-level state tracking ──────────────────────────
        self._prev_grid: Optional[np.ndarray] = None
        self._prev_levels: int = 0
        self._prev_obs = None          # last GameObservation
        self._prev_strategy = None     # last executed GameStrategy
        self._prev_trajectory = None   # last sampled trajectory
        self._prev_goal_context = None # last sampling goal context
        self._needs_reset: bool = False
        self._game_started: bool = False
        self._last_action_name: Optional[str] = None
        self._last_action_data: Dict[str, Any] = {}
        self._last_click_pos: Optional[tuple] = None
        self._prev_game_state: str = "NOT_FINISHED"
        self._available_action_names: List[str] = [
            "ACTION1", "ACTION2", "ACTION3", "ACTION4",
            "ACTION5", "ACTION6", "ACTION7",
        ]
        self.cognitive_controller: Optional[UnifiedCognitiveController] = None
        if self.ENABLE_UNIFIED_COGNITION:
            self.cognitive_controller = UnifiedCognitiveController(
                self.game_id,
                available_actions=self._available_action_names,
            )
        # ── Env-persistence across outer-loop iterations ──────────
        # When True, the next `_run_*_iteration()` call will NOT rebuild
        # the arcade env or send a RESET; it continues from the current
        # live level state. Set after a successful level-up that did
        # not terminate the session. Cleared whenever we need a fresh env
        # (GAME_OVER, full stall without progress, explicit reset).
        self._preserve_env_next_iter: bool = False
        # Current level observed in the live env (used for continuation)
        self._live_levels_completed: int = 0
        # ── Live LLM revision (online plateau-driven) ─────────────
        # Directory used to PERSIST the revised TaskPrograms produced
        # by the live LLM compiler (e.g. `<game>.lvl<N>.live<k>.json`).
        # Set by the launcher (`run_with_human_priors.py`).
        self._task_program_dir: Optional[Path] = None
        # Plateau detection: counts iters since `best_level` last increased.
        self._iters_no_progress: int = 0
        # Snapshot of best_level at the END of the previous iter; used
        # by plateau detection to compare across iterations.
        self._iter_best_level_snapshot: int = 0
        # Set by the launcher; required for live revision to work.
        self._human_traces_dir: Optional[Path] = None
        # Track which target level's TaskProgram is currently attached.
        self._attached_task_program_target_level: Optional[int] = None
        self._attached_task_program_kind: Optional[str] = None
        # Number of live LLM revisions performed so far (capped by
        # MAX_LIVE_REVISIONS).
        self._live_revision_attempts: int = 0
        # Iters_no_progress value at the moment we last triggered a
        # live revision; used to require N additional plateau iters
        # before triggering another.
        self._live_revision_last_trigger_iter: int = -10**9
        # Resolve LIVE_REVISE_DEVICE: "auto" -> "cuda" if torch sees a
        # GPU, else "cpu". Concrete strings ("cuda", "cuda:0", "cpu")
        # are passed through verbatim.
        self._live_revise_device: str = self._resolve_live_revise_device()
        self._live_revise_disabled_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_checkpoint_path(ckpt_dir: Path, *candidates: str) -> Optional[Path]:
        """Return the first existing checkpoint path from a preference-ordered list."""
        return resolve_checkpoint_path(ckpt_dir, *candidates)

    @classmethod
    def _build_loop_config(cls, device: str) -> LoopConfig:
        """Build the reasoning config with explicit checkpoint and training choices."""
        return build_adaptive_loop_config(
            device=device,
            llm_model_name=cls.LIVE_REVISE_MODEL,
            reasoning_mode=cls.REASONING_MODE,
            ablation_stage=cls.ABLATION_STAGE,
            enable_online_jepa_training=cls.ENABLE_ONLINE_JEPA_TRAINING,
            enable_online_ebm_training=cls.ENABLE_ONLINE_EBM_TRAINING,
        )

    def _resolve_live_revise_device(self) -> str:
        """Pick the runtime device for the live-revise LLM.

        Honours the class attribute ``LIVE_REVISE_DEVICE``:
          - "auto" (default): "cuda" if ``torch.cuda.is_available()``
            else "cpu".
          - any other string ("cpu", "cuda", "cuda:0", ...) is passed
            through verbatim.
        """
        requested = (self.LIVE_REVISE_DEVICE or "auto").strip().lower()
        if requested != "auto":
            return requested
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _disable_live_revise_for_run(self, reason: str) -> None:
        """Stop retrying live revision when the runtime cannot support it."""
        if self._live_revise_disabled_reason is not None:
            return
        self._live_revise_disabled_reason = reason
        print(
            f"[DIAG {self.game_id}] LIVE-REVISE disabled for this run: {reason}",
            flush=True,
        )

    @staticmethod
    def _is_live_revise_hard_failure(error: Exception) -> bool:
        """True when retrying live revise is extremely unlikely to help."""
        msg = f"{type(error).__name__}: {error}".lower()
        hard_patterns = (
            "could not load llm",
            "install transformers",
            "check hf cache",
            "no module named 'transformers'",
            'no module named "transformers"',
            "no module named 'sentencepiece'",
            'no module named "sentencepiece"',
            "no module named 'accelerate'",
            'no module named "accelerate"',
        )
        return any(pattern in msg for pattern in hard_patterns)

    def _build_agent_experience_block(self, current_level: int) -> str:
        """Render the agent's per-action stats and discovered mechanics
        on the level it is currently stuck on, for inclusion in an LLM
        revision prompt.

        Pulls everything from ``self.memory`` (a ``GameMemory``):
          - per-action ``ActionProfile`` (tries / change / move / death rates)
          - inferred semantics (move_up / interact / no_effect / ...)
          - movement actions, safe actions, effective click values
          - level action sequences (recent actions per level)
          - hypotheses with confidence
        """
        try:
            mem = self.memory
        except Exception:
            return "(no GameMemory available)"

        lines: List[str] = [
            f"Stuck on level: {current_level + 1}  "
            f"(human-trace levels_completed=={current_level} when transitioning here).",
            f"Total actions taken so far: {getattr(mem, 'total_actions', 0)}; "
            f"deaths: {getattr(mem, 'total_game_overs', 0)}; "
            f"states visited: {len(getattr(mem, '_visited_hashes', []) or [])}.",
            "",
            "Per-action runtime profile (all levels combined; non-zero tries only):",
        ]
        try:
            profiles = list(mem.action_profiles.items())
        except Exception:
            profiles = []
        if not profiles:
            lines.append("  (no actions recorded yet)")
        else:
            # Sort by tries descending so the most-used actions come first.
            profiles.sort(key=lambda kv: -kv[1].times_tried)
            for name, p in profiles:
                if p.times_tried == 0 or name == "RESET":
                    continue
                death_rate = p.times_caused_game_over / max(p.times_tried, 1)
                disp = p.dominant_displacement
                disp_s = (
                    f" disp=({disp[0]:+.1f},{disp[1]:+.1f})" if disp else ""
                )
                lines.append(
                    f"  {name}: tried={p.times_tried} "
                    f"change_rate={p.change_rate:.2f} "
                    f"move_rate={p.move_rate:.2f} "
                    f"death_rate={death_rate:.2f}"
                    f"{disp_s}"
                )

        # Inferred semantics
        try:
            sem = mem.infer_action_semantics()
        except Exception:
            sem = {}
        if sem:
            lines.append("")
            lines.append("Inferred action semantics (from observations):")
            for k in sorted(sem.keys()):
                if k == "RESET":
                    continue
                lines.append(f"  {k} -> {sem[k]}")

        # Movement / safe actions
        try:
            move_actions = mem.get_movement_actions()
        except Exception:
            move_actions = []
        try:
            safe_actions = mem.get_safe_actions()
        except Exception:
            safe_actions = []
        if move_actions or safe_actions:
            lines.append("")
            lines.append(
                f"Movement actions identified: {move_actions or '(none)'}"
            )
            lines.append(
                f"Actions never causing GAME_OVER: {safe_actions or '(none)'}"
            )

        # Effective click values
        try:
            click_vals = sorted(mem.get_effective_click_values())
        except Exception:
            click_vals = []
        if click_vals:
            lines.append(
                f"Grid colours that responded to ACTION6 click: {click_vals}"
            )

        # Recent action sequence on the stuck level (last 25 entries)
        try:
            seq = list(mem.level_action_sequences.get(current_level, []))
        except Exception:
            seq = []
        if seq:
            lines.append("")
            lines.append(
                f"Recent action sequence the agent tried on this level "
                f"(last 25 of {len(seq)}, oldest first):"
            )
            tail = seq[-25:]
            lines.append("  " + " ".join(tail))

        # Hypotheses
        try:
            hyps = sorted(
                ((k, v) for k, v in mem.hypotheses.items()),
                key=lambda kv: -kv[1],
            )[:10]
        except Exception:
            hyps = []
        if hyps:
            lines.append("")
            lines.append("Top hypotheses currently held by agent (key, conf):")
            for k, v in hyps:
                lines.append(f"  {k}: {v:.2f}")

        return "\n".join(lines)

    def _maybe_attach_task_program_for_level(self, levels_completed: int) -> None:
        """Attach the best available TaskProgram for the current target level.

        The environment reports `levels_completed`, so the active target
        level is `levels_completed + 1`. Level 1 keeps the validated
        generic per-game program; level-specific programs are only
        preferred once we have actually advanced into later levels.
        """
        if self._task_program_dir is None:
            return

        target_level = max(1, int(levels_completed) + 1)
        if self._attached_task_program_target_level == target_level:
            return

        try:
            from human_trace.task_program import (
                try_load_task_program,
                try_load_task_program_for_level,
            )
        except Exception as e:
            print(
                f"[DIAG {self.game_id}] TASK-PROGRAM load helpers unavailable: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return

        program = None
        kind = "generic"
        if target_level > 1:
            program = (
                try_load_task_program_for_level(self._task_program_dir, self.game_id, target_level)
                or try_load_task_program_for_level(
                    self._task_program_dir,
                    self.game_id.split("-", 1)[0],
                    target_level,
                )
            )
            if program is not None:
                kind = "level-specific"
        if program is None:
            program = (
                try_load_task_program(self._task_program_dir, self.game_id)
                or try_load_task_program(
                    self._task_program_dir,
                    self.game_id.split("-", 1)[0],
                )
            )
        if program is None:
            return

        setattr(program, "_target_level", target_level)
        setattr(program, "_attachment_kind", kind)
        self.reasoning.decomposer.set_task_program(program)
        self.reasoning.state.goal = None
        self.reasoning.state.current_subgoal = None
        self.reasoning.state.current_strategy = None
        self.reasoning.state.current_trajectory = None
        self.reasoning.state.current_goal_context = None
        self.reasoning.state.current_goal_hypothesis = None
        self.reasoning.state.current_continuation = None
        self.reasoning.state.pending_continuation = None
        self.reasoning.goal_pursuit.set_goal_bank([])
        self.reasoning.state.phase = Phase.DECOMPOSE
        self._attached_task_program_target_level = target_level
        self._attached_task_program_kind = kind
        print(
            f"[DIAG {self.game_id}] TASK-PROGRAM attach "
            f"target_level={target_level} kind={kind} "
            f"family={program.goal_family} subgoals={len(program.subgoal_tests)} "
            f"conf={program.confidence:.2f}",
            flush=True,
        )

    def _try_live_llm_revise_task_program(self, best_level: int) -> bool:
        """Trigger a *live* LLM revision: re-compile a TaskProgram using
        BOTH the human traces AND the agent's runtime observations.

        Cost: ~30-90 s on CPU with Qwen-2.5-3B-Instruct. Used as a
        fallback when the pre-compiled per-level programs (Phase 1+2)
        aren't enough to crack the current level.

        Returns True iff a new program was successfully compiled and
        attached to the goal decomposer.
        """
        if self._live_revise_disabled_reason is not None:
            return False
        if self._live_revision_attempts >= self.MAX_LIVE_REVISIONS:
            return False
        if self._human_traces_dir is None or self._task_program_dir is None:
            print(
                f"[DIAG {self.game_id}] LIVE-REVISE skipped: "
                f"traces_dir={self._human_traces_dir} "
                f"programs_dir={self._task_program_dir}",
                flush=True,
            )
            return False
        target_level = best_level + 1
        attempt_idx = self._live_revision_attempts + 1
        device = self._live_revise_device
        eta_hint = (
            "(this will block ~30-90s on CPU)"
            if device.startswith("cpu")
            else f"(running on {device}; expect ~5-20s)"
        )
        print(
            f"[DIAG {self.game_id}] LIVE-REVISE START attempt={attempt_idx}/"
            f"{self.MAX_LIVE_REVISIONS} target_level={target_level} "
            f"device={device} {eta_hint}",
            flush=True,
        )
        t0 = time.time()
        try:
            from human_trace.compile_trace import (
                compile_with_llm,
                filter_pack_by_min_level,
            )
            from human_trace.integration import build_prior_pack
            from human_trace.loader import load_traces
        except Exception as e:
            print(
                f"[DIAG {self.game_id}] LIVE-REVISE import error: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            self._disable_live_revise_for_run(f"import error: {type(e).__name__}")
            return False

        try:
            corpus = load_traces(self._human_traces_dir)
            # Resolve full game id (loader keys by full id like "ar25-e3c63847").
            full_ids = sorted(corpus.by_game.keys())
            short = self.game_id.split("-", 1)[0]
            match = None
            for gid in full_ids:
                if gid == self.game_id or gid.startswith(f"{short}-"):
                    match = gid
                    break
            if match is None:
                match = self.game_id
            pack = build_prior_pack(corpus, match)
            # Filter to the level we're stuck on (humans had completed
            # at least `best_level` levels at this point).
            pack = filter_pack_by_min_level(pack, best_level)
        except Exception as e:
            print(
                f"[DIAG {self.game_id}] LIVE-REVISE pack build failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return False

        agent_experience = self._build_agent_experience_block(best_level)
        # Identify which actions the agent has ALREADY hammered (these
        # are over-saturated and should NOT dominate the new probes).
        over_used: List[str] = []
        under_used: List[str] = []
        try:
            for name, p in self.memory.action_profiles.items():
                if name == "RESET":
                    continue
                if p.times_tried >= 50:
                    over_used.append(name)
                elif p.times_tried <= 10:
                    under_used.append(name)
        except Exception:
            pass
        # Movement actions to surface in the achieve subgoal's
        # `prefer_actions` requirement (so the LLM doesn't pick a
        # nonsensical set).
        try:
            move_actions_for_prompt = self.memory.get_movement_actions() or [
                "ACTION1", "ACTION2", "ACTION3", "ACTION4",
            ]
        except Exception:
            move_actions_for_prompt = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

        revision_context = (
            f"This is LIVE REVISION #{attempt_idx} for game {self.game_id}.\n"
            f"The agent previously completed level {best_level} but is now "
            f"stuck on level {target_level}. Many deaths and very little "
            f"signal from the level-{best_level} probes carrying over.\n"
            f"\n"
            f"PRODUCE A TASKPROGRAM TAILORED TO LEVEL {target_level} ONLY.\n"
            f"\n"
            f"=== HARD REQUIREMENTS for `subgoal_tests` ===\n"
            f"\n"
            f"You MUST emit BOTH (a) probe subgoals AND (b) one final "
            f"achieve subgoal. Without the achieve subgoal the agent "
            f"will never combine actions and will stay stuck.\n"
            f"\n"
            f"(a) PROBES (2-3 entries):\n"
            f"  - `id` MUST start with \"probe_\" (e.g. probe_action1, "
            f"probe_switch_controller).\n"
            f"  - EVERY probe MUST have a DIFFERENT `prefer_actions` set. "
            f"Do NOT emit two probes with the same `prefer_actions`.\n"
            f"  - PRIORITISE under-used actions {under_used or '[none]'} "
            f"in `prefer_actions`. AVOID putting only over-used actions "
            f"{over_used or '[none]'} (the agent already tried those "
            f"hundreds of times to no effect on this level).\n"
            f"  - Small `max_actions` (4-8).\n"
            f"  - `expected_signal` exploratory: `object_moved`, "
            f"`role_switch`, `color_changed`, `anything_changed`.\n"
            f"\n"
            f"(b) ACHIEVE (EXACTLY 1 entry, MANDATORY, listed LAST):\n"
            f"  - `id` MUST start with \"achieve_\" (e.g. "
            f"achieve_level{target_level}, achieve_progress).\n"
            f"  - `prefer_actions` MUST list AT LEAST 4 distinct actions, "
            f"covering BOTH movement actions {move_actions_for_prompt} "
            f"AND interact/switch actions like ACTION5 / ACTION6. This is "
            f"the EXPLOITATION phase where the agent is finally allowed "
            f"to combine mechanics.\n"
            f"  - `max_actions` MUST be between 30 and 60 (give the agent "
            f"enough room to chain actions).\n"
            f"  - `expected_signal` SHOULD be a high-level progress "
            f"indicator: `level_changed`, `grid_cell_change`, or "
            f"`unique_states_increasing`.\n"
            f"  - Even if the exact win condition is unclear, emit this "
            f"subgoal anyway — it is REQUIRED.\n"
            f"\n"
            f"Do NOT simply repeat the level-1 program — surface what is "
            f"DIFFERENT about level {target_level} from the agent's "
            f"runtime observations below."
        )

        # Pause the game-time clock during the (potentially long) LLM call:
        # without this the entire compile time is charged against the agent's
        # budget, leaving no time to actually use the new program.
        compile_t0 = time.time()
        try:
            program = compile_with_llm(
                pack,
                model_name=self.LIVE_REVISE_MODEL,
                device=device,
                max_retries=3,  # leave headroom for the achieve_* validator (rule 3b) to bounce a probe-only first draft
                agent_experience=agent_experience,
                revision_context=revision_context,
            )
        except Exception as e:
            print(
                f"[DIAG {self.game_id}] LIVE-REVISE compile failed after "
                f"{time.time() - t0:.1f}s: {type(e).__name__}: {e}",
                flush=True,
            )
            if self._is_live_revise_hard_failure(e):
                self._disable_live_revise_for_run(
                    f"runtime unavailable ({type(e).__name__})"
                )
            # Still refund the time we spent so we don't punish the run.
            if self.LIVE_REVISE_PAUSES_BUDGET:
                self.timer += time.time() - compile_t0
            return False
        compile_dt = time.time() - compile_t0
        if self.LIVE_REVISE_PAUSES_BUDGET:
            # Shift the game-start timestamp forward by the compile
            # duration; `_time_left()` is computed as
            # `time_budget - (now - self.timer)` so this effectively
            # pauses the clock during the LLM call.
            self.timer += compile_dt
            print(
                f"[DIAG {self.game_id}] LIVE-REVISE paused game clock "
                f"by {compile_dt:.1f}s",
                flush=True,
            )

        # Persist the revised program for inspection / reuse next run.
        out_path = (
            self._task_program_dir
            / f"{short}.lvl{target_level}.live{attempt_idx}.json"
        )
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            program.save(out_path)
        except Exception as e:
            print(
                f"[DIAG {self.game_id}] LIVE-REVISE could not save program "
                f"to {out_path}: {type(e).__name__}: {e}",
                flush=True,
            )
            # continue anyway — we can still attach in-memory

        try:
            self.reasoning.decomposer.set_task_program(program)
        except Exception as e:
            print(
                f"[DIAG {self.game_id}] LIVE-REVISE attach failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return False

        # Force regeneration of the goal bank against the new program.
        try:
            self.reasoning.goal_pursuit.active_goal = None
        except Exception:
            pass

        self._live_revision_attempts += 1
        print(
            f"[DIAG {self.game_id}] LIVE-REVISE SUCCESS attempt={attempt_idx} "
            f"target_level={target_level} took={time.time() - t0:.1f}s "
            f"family={program.goal_family} "
            f"subgoals={len(program.subgoal_tests)} "
            f"conf={program.confidence:.2f} "
            f"saved_to={out_path.name}",
            flush=True,
        )
        return True

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        try:
            return self._choose_action_impl(frames, latest_frame)
        except Exception as e:
            logger.error(f"[{self.game_id}] Error in choose_action: {e}", exc_info=True)
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return GameAction.RESET
            return random.choice(
                [GameAction.ACTION1, GameAction.ACTION2,
                 GameAction.ACTION3, GameAction.ACTION4]
            )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _choose_action_impl(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:

        # Parse first so terminal transitions are learned before the mandatory
        # RESET. The former ordering returned early on GAME_OVER and hid the
        # most important negative observation from every learner.
        current_grid = self.analyzer.parse_frame(latest_frame.frame)
        if latest_frame.available_actions:
            self._available_action_names = [
                _normalize_action_name(a)
                for a in latest_frame.available_actions
                if _normalize_action_name(a) != "RESET"
            ]

        diff = None
        if self._prev_grid is not None and self._last_action_name is not None:
            diff = self.analyzer.compute_diff(self._prev_grid, current_grid)
            if (
                self.cognitive_controller is not None
                and self._last_action_name != "RESET"
            ):
                try:
                    self.cognitive_controller.observe_transition(
                        action=self._last_action_name,
                        action_data=self._last_action_data,
                        grid_before=self._prev_grid,
                        grid_after=current_grid,
                        available_actions=self._available_action_names,
                        game_state_before=self._prev_game_state,
                        game_state_after=latest_frame.state.name,
                        levels_completed_before=self._prev_levels,
                        levels_completed_after=latest_frame.levels_completed,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] unified cognition observation failed: %s",
                        self.game_id,
                        exc,
                        exc_info=True,
                    )

        # ── Handle mandatory resets ─────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._needs_reset = True

        if self._needs_reset:
            self._needs_reset = False
            self._game_started = True
            if (
                latest_frame.state == GameState.GAME_OVER
                and self._prev_grid is not None
                and self._last_action_name not in (None, "RESET")
                and diff is not None
            ):
                self.memory.record_action(
                    action_name=self._last_action_name,
                    grid_before=self._prev_grid,
                    grid_after=current_grid,
                    diff=diff,
                    game_state=latest_frame.state.name,
                    levels_completed=latest_frame.levels_completed,
                )
                if self._last_action_name == "ACTION6" and self._last_click_pos is not None:
                    self.memory.record_click(
                        pos=self._last_click_pos,
                        grid_before=self._prev_grid,
                        changed=diff.anything_changed,
                        level_changed=False,
                    )
            self._last_action_name = "RESET"
            self._last_action_data = {}
            self._prev_grid = current_grid.copy()
            self._prev_levels = latest_frame.levels_completed
            self._prev_game_state = latest_frame.state.name
            if self.cognitive_controller is not None:
                self.cognitive_controller.on_reset()
            if latest_frame.state == GameState.GAME_OVER:
                self.memory.on_game_over()
                self.reasoning.on_game_over()
                logger.info(
                    f"[{self.game_id}] GAME_OVER at action {self.action_counter}, "
                    f"level {latest_frame.levels_completed}. Retrying."
                )
            return GameAction.RESET

        if latest_frame.full_reset and self._last_action_name != "RESET":
            self._last_action_name = "RESET"
            self._last_action_data = {}
            self._prev_grid = current_grid.copy()
            self._prev_levels = latest_frame.levels_completed
            self._prev_game_state = latest_frame.state.name
            if self.cognitive_controller is not None:
                self.cognitive_controller.on_reset()
            return GameAction.RESET

        # ── Parse current frame ─────────────────────────────────
        current_grid = self.analyzer.parse_frame(latest_frame.frame)

        if latest_frame.available_actions:
            self._available_action_names = [
                _normalize_action_name(a)
                for a in latest_frame.available_actions
                if _normalize_action_name(a) != "RESET"
            ]

        # ── Record effect of previous action in memory ──────────
        diff = None
        if self._prev_grid is not None:
            diff = self.analyzer.compute_diff(self._prev_grid, current_grid)
            if self._last_action_name is not None:
                self.memory.record_action(
                    action_name=self._last_action_name,
                    grid_before=self._prev_grid,
                    grid_after=current_grid,
                    diff=diff,
                    game_state=latest_frame.state.name,
                    levels_completed=latest_frame.levels_completed,
                )
                if self._last_action_name == "ACTION6" and self._last_click_pos is not None:
                    self.memory.record_click(
                        pos=self._last_click_pos,
                        grid_before=self._prev_grid,
                        changed=diff.anything_changed,
                        level_changed=(latest_frame.levels_completed > self._prev_levels),
                    )

        # ── Detect level progression ────────────────────────────
        level_changed = latest_frame.levels_completed > self._prev_levels
        if level_changed:
            self.memory.on_level_change(latest_frame.levels_completed)
            self.reasoning.on_level_change(latest_frame.levels_completed)
            if self.cognitive_controller is not None:
                self.cognitive_controller.on_level_change()
            logger.info(
                f"[{self.game_id}] Level up! Now at level {latest_frame.levels_completed}"
            )

        # ── Step 7 (feedback): record result of previous strategy ──
        if self._prev_obs is not None and self._prev_strategy is not None:
            new_states = 1 if diff and diff.anything_changed else 0
            cur_obs_for_feedback = self.reasoning.describer.describe(
                grid=current_grid,
                memory=self.memory,
                game_state=latest_frame.state.name,
                levels_completed=latest_frame.levels_completed,
                action_counter=self.action_counter,
                diff=diff,
            )
            self.reasoning.record_result(
                obs_before=self._prev_obs,
                obs_after=cur_obs_for_feedback,
                strategy=self._prev_strategy,
                level_changed=level_changed,
                game_over=False,
                new_states=new_states,
                trajectory=self._prev_trajectory,
                goal_context=self._prev_goal_context,
            )

        # ── Steps 1-6: reasoning loop step ──────────────────────
        self._maybe_attach_task_program_for_level(latest_frame.levels_completed)

        result = self.reasoning.step(
            current_grid=current_grid,
            game_state=latest_frame.state.name,
            levels_completed=latest_frame.levels_completed,
            available_actions=self._available_action_names,
        )

        action_name: str = result["action"]
        action_data = result.get("action_data")
        legacy_action_name = action_name
        legacy_action_data = dict(action_data or {})
        strategy = result.get("strategy")
        trajectory = result.get("trajectory")
        subgoal = result.get("subgoal")
        goal = result.get("goal")
        goal_context = result.get("goal_context")
        goal_hypothesis = result.get("goal_hypothesis")
        observation = result.get("observation")
        trajectory_debug = result.get("trajectory_debug")

        cognitive_decision = None
        if self.cognitive_controller is not None:
            try:
                cognitive_decision = self.cognitive_controller.select_action(
                    current_grid=current_grid,
                    available_actions=self._available_action_names,
                    legacy_action=legacy_action_name,
                    legacy_action_data=legacy_action_data,
                    game_state=latest_frame.state.name,
                    levels_completed=latest_frame.levels_completed,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] unified cognition decision failed; using legacy: %s",
                    self.game_id,
                    exc,
                    exc_info=True,
                )
            if cognitive_decision is not None:
                action_name = cognitive_decision.action_name
                action_data = dict(cognitive_decision.action_data)

        # ── Convert to GameAction ───────────────────────────────
        action = _NAME_TO_ACTION.get(action_name, GameAction.ACTION1)

        if action_name == "ACTION6":
            # Safety: ACTION6 always needs x,y data
            if not action_data:
                # Generate a click position from observation or grid center
                if observation and observation.objects:
                    idx = self.action_counter % len(observation.objects)
                    obj = observation.objects[idx]
                    action_data = {"x": int(obj["center_x"]), "y": int(obj["center_y"])}
                elif current_grid is not None:
                    h, w = current_grid.shape[:2] if current_grid.ndim >= 2 else (64, 64)
                    # Scan position based on action counter
                    scan_idx = self.action_counter % 64
                    row, col = divmod(scan_idx, 8)
                    action_data = {"x": int((col + 0.5) * w / 8), "y": int((row + 0.5) * h / 8)}
                else:
                    action_data = {"x": 32, "y": 32}
            action.set_data(action_data)
            self._last_click_pos = (
                action_data.get("y", 0),
                action_data.get("x", 0),
            )
        else:
            self._last_click_pos = None

        # Attach reasoning metadata
        loop_stats = self.reasoning.get_stats()
        action.reasoning = {
            "phase": result.get("phase", "unknown"),
            "goal": goal.overarching_goal if goal else None,
            "subgoal": subgoal.description[:80] if subgoal else None,
            "strategy": strategy.strategy_type.value if strategy else None,
            "strategy_desc": strategy.description[:120] if strategy else None,
            "trajectory_source": getattr(trajectory, "source", None) if trajectory else None,
            "trajectory_score": getattr(trajectory, "score", None) if trajectory else None,
            "trajectory_energy": getattr(trajectory, "energy", None) if trajectory else None,
            "trajectory_goal_progress": (
                trajectory.metadata.get("goal_progress") if trajectory else None
            ),
            "trajectory_novelty": getattr(trajectory, "novelty", None) if trajectory else None,
            "trajectory_risk": getattr(trajectory, "risk", None) if trajectory else None,
            "trajectory_human_compatibility": (
                trajectory.metadata.get("human_compatibility") if trajectory else None
            ),
            "trajectory_source_counts": (
                trajectory.metadata.get("source_counts") if trajectory else None
            ),
            "trajectory_top_candidates": (
                trajectory.metadata.get("top_candidates") if trajectory else None
            ),
            "trajectory_latent_task_program": (
                trajectory.metadata.get("latent_task_program") if trajectory else None
            ),
            "trajectory_debug": trajectory_debug,
            "goal_hypothesis_family": (
                getattr(goal_hypothesis, "family", None)
                if goal_hypothesis is not None else None
            ),
            "goal_hypothesis_confidence": (
                getattr(goal_hypothesis, "confidence", None)
                if goal_hypothesis is not None else None
            ),
            "goal_hypothesis_source": (
                getattr(goal_hypothesis, "source", None)
                if goal_hypothesis is not None else None
            ),
            "action_counter": self.action_counter,
            "level": latest_frame.levels_completed,
            "exploration_score": self.memory.get_exploration_score(),
            "loop_stats": loop_stats,
            "unified_cognition_decision": (
                cognitive_decision.to_dict()
                if cognitive_decision is not None
                else None
            ),
            "unified_cognition": (
                self.cognitive_controller.summary()
                if self.cognitive_controller is not None
                else None
            ),
        }

        # ── Update tracking ─────────────────────────────────────
        self._last_action_name = action_name
        self._last_action_data = dict(action_data or {})
        self._prev_grid = current_grid.copy()
        self._prev_levels = latest_frame.levels_completed
        self._prev_game_state = latest_frame.state.name
        self._prev_obs = observation
        executed_legacy_plan = (
            cognitive_decision is None
            or cognitive_decision.source == "legacy_fallback"
        )
        self._prev_strategy = strategy if executed_legacy_plan else None
        self._prev_trajectory = trajectory if executed_legacy_plan else None
        self._prev_goal_context = goal_context if executed_legacy_plan else None

        return action

    # ------------------------------------------------------------------
    # Unified adaptive loop
    # ------------------------------------------------------------------
    def _compute_knowledge_level(self) -> float:
        """Compute how much we know about this game (0.0 = nothing, 1.0 = well-understood).

        Drives the exploration→exploitation transition.  Heavily weighted
        toward *actionable* knowledge: do we know what actions DO?
        """
        profiles = self.memory.action_profiles
        n_avail = max(len(self._available_action_names), 1)

        # 1. Action coverage: fraction tried ≥2 times (0→1 quickly)
        tried = sum(1 for p in profiles.values() if p.times_tried >= 2)
        action_coverage = min(1.0, tried / n_avail)

        # 2. Mechanism confidence: how many actions have consistent behavior?
        #    (change_rate near 0 or near 1 = we understand it)
        confident = 0
        for p in profiles.values():
            if p.times_tried >= 3:
                cr = p.change_rate
                if cr >= 0.7 or cr <= 0.15:  # clearly does/doesn't change grid
                    confident += 1
        mechanism_confidence = min(1.0, confident / max(n_avail, 1))

        # 3. Level completion (binary — massive signal)
        level_signal = 1.0 if self.memory.max_level_reached > 0 else 0.0

        # 4. Player identified (useful for navigation games)
        player_signal = 1.0 if self.memory.player.identified else 0.0

        # Weighted: action understanding dominates
        return (
            0.30 * action_coverage
            + 0.30 * mechanism_confidence
            + 0.25 * level_signal
            + 0.15 * player_signal
        )

    # ------------------------------------------------------------------
    # Phase 2b: mandatory exploration floor
    # ------------------------------------------------------------------
    # Regardless of how confident the priors make `knowledge_level` look,
    # the agent must spend at least this many iterations in exploration
    # mode (ε ≥ 0.7) before switching to exploitation. This prevents the
    # "confidently wrong" failure mode where seeded action profiles push
    # knowledge to 0.6+ on step 0 and the agent skips Phase 1 entirely.
    #
    # One exploration iteration ≈ 100 actions (fast_iter_size). So 3
    # iterations ≈ 300 actions of guaranteed exploration — ~3% of a
    # 10k-action budget. Cheap insurance against bad priors.
    MIN_EXPLORE_ITERATIONS: int = 3

    def _compute_epsilon(self, knowledge: float, iteration: int) -> float:
        """Exploration rate with hard exploitation shifts.

        Returns value in [0.05, 1.0].
        Key insight: ARC rewards fast commitment once a pattern is detected.
        """
        # Mandatory exploration floor (Phase 2b). Applies even when a
        # primed knowledge_level would otherwise trigger exploitation.
        if iteration <= self.MIN_EXPLORE_ITERATIONS:
            return max(0.7, 1.0 - knowledge * 0.3)

        # Hard shift: if we've completed a level, exploit aggressively
        if self.memory.max_level_reached > 0:
            return 0.10

        # Hard shift: if we have confident mechanics for most actions, commit
        if knowledge >= 0.6:
            return 0.25

        # Normal decay: steeper than before (was 80 iterations, now 20)
        base = max(0.15, 1.0 - knowledge * 1.5)
        iter_decay = max(0.0, 1.0 - iteration / 20.0)
        eps = base * (0.3 + 0.7 * iter_decay)

        return max(0.05, min(1.0, eps))

    def main(self) -> None:
        """Unified adaptive loop — exploration and strategy bootstrap each other.

        Single time budget for the entire game.  Every iteration sits on a
        continuous exploration ↔ exploitation spectrum controlled by ε
        (exploration rate).  ε starts high and decays with knowledge.

        If no progress is made for STALL_FRACTION of remaining time,
        the current iteration is abandoned and a fresh one starts.

        Key feedback loops:
          • Visual cortex trains continuously → feeds memory + strategy
          • Associative memory retrieve_action() guides exploration (never blind random)
          • Strategic discoveries (directions, danger) flow back to memory
          • Winning sequences become procedures that are replayed early in future iterations
        """
        from arcengine import GameState as GS

        self.timer = time.time()
        time_budget = self.TIME_BUDGET
        best_level = 0
        best_seq: list = []
        iteration = 0
        total_wins = 0
        fast_iter_size = 100       # actions per fast iteration
        vc_train_interval = 10     # train VC every N iterations
        stuck_threshold = 3        # consecutive no-progress iters → re-explore

        avail_ints = list(range(1, 8))

        def _time_left():
            return max(0.0, time_budget - (time.time() - self.timer))

        def _stall_limit():
            """Max seconds for an iteration with no progress before restart."""
            return max(2.0, _time_left() * self.STALL_FRACTION)

        # ── Helpers ───────────────────────────────────────────────
        def _fast_parse(frame_data):
            if frame_data is None:
                return None
            f = frame_data.frame if hasattr(frame_data, 'frame') else frame_data
            if f is None:
                return None
            arr = np.array(f, dtype=np.uint8)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            return arr

        def _informed_click(grid, it, step):
            """Click position using memory + fallback strategies."""
            if grid is None:
                return {"x": random.randint(0, 31), "y": random.randint(0, 31)}
            h, w = grid.shape[:2]
            # Try effective click values from memory first
            effective_vals = self.memory.get_effective_click_values()
            if effective_vals and random.random() < 0.5:
                for val in effective_vals:
                    ys, xs = np.where(grid == val)
                    if len(ys) > 0:
                        idx = random.randint(0, len(ys) - 1)
                        return {"x": int(xs[idx]), "y": int(ys[idx])}
            # Fallback: varied strategies
            strategy = it % 6
            if strategy == 0:
                return {"x": random.randint(0, w - 1), "y": random.randint(0, h - 1)}
            elif strategy == 1:
                nz_y, nz_x = np.nonzero(grid)
                if len(nz_y) > 0:
                    idx = random.randint(0, len(nz_y) - 1)
                    return {"x": int(nz_x[idx]), "y": int(nz_y[idx])}
            elif strategy == 2:
                return {"x": w // 2, "y": h // 2}
            elif strategy == 3:
                row, col = step % 4, step // 4 % 4
                return {"x": int((col + 0.5) * w / 4), "y": int((row + 0.5) * h / 4)}
            scan_idx = (it * 13 + step) % 256
            row, col = divmod(scan_idx, 16)
            return {"x": int(min(w - 1, (col + 0.5) * w / 16)),
                    "y": int(min(h - 1, (row + 0.5) * h / 16))}

        def _unified_fast_choice(grid, legacy_int, legacy_data, frame_data):
            """Route fast-path and replay decisions through the same controller."""
            if self.cognitive_controller is None or grid is None:
                return legacy_int, legacy_data, None
            legacy_name = _INT_TO_ACTION_NAME.get(
                legacy_int,
                f"ACTION{legacy_int}",
            )
            available_names = [
                _INT_TO_ACTION_NAME.get(value, f"ACTION{value}")
                for value in avail_ints
            ]
            try:
                decision = self.cognitive_controller.select_action(
                    current_grid=grid,
                    available_actions=available_names,
                    legacy_action=legacy_name,
                    legacy_action_data=legacy_data or {},
                    game_state=getattr(
                        getattr(frame_data, "state", None),
                        "name",
                        "NOT_FINISHED",
                    ),
                    levels_completed=int(
                        getattr(frame_data, "levels_completed", 0) or 0
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "[%s] fast unified decision failed; using legacy: %s",
                    self.game_id,
                    exc,
                    exc_info=True,
                )
                return legacy_int, legacy_data, None
            try:
                selected_int = int(decision.action_name.replace("ACTION", ""))
            except (TypeError, ValueError):
                selected_int = legacy_int
            return selected_int, dict(decision.action_data), decision

        def _unified_fast_observe(
            before_grid,
            after_grid,
            action_name,
            action_data,
            before_level,
            frame_data,
        ):
            if (
                self.cognitive_controller is None
                or before_grid is None
                or after_grid is None
                or action_name == "RESET"
            ):
                return
            try:
                self.cognitive_controller.observe_transition(
                    action=action_name,
                    action_data=action_data or {},
                    grid_before=before_grid,
                    grid_after=after_grid,
                    available_actions=[
                        _INT_TO_ACTION_NAME.get(value, f"ACTION{value}")
                        for value in avail_ints
                    ],
                    game_state_before="NOT_FINISHED",
                    game_state_after=getattr(
                        getattr(frame_data, "state", None),
                        "name",
                        "NOT_FINISHED",
                    ),
                    levels_completed_before=int(before_level),
                    levels_completed_after=int(
                        getattr(frame_data, "levels_completed", before_level) or 0
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "[%s] fast unified observation failed: %s",
                    self.game_id,
                    exc,
                    exc_info=True,
                )

        def _run_exploration_iteration():
            """One fast informed-exploration iteration. Returns (level, sequence)."""
            nonlocal avail_ints

            # ── Fresh env + Reset ──
            if self._arcade is not None:
                try:
                    self.arc_env = self._arcade.make(self.game_id)
                except Exception:
                    pass
            env = self.arc_env
            try:
                f = env.step(GameAction.RESET)
                if f is None or f.frame is None:
                    return 0, []
            except Exception:
                return 0, []

            if self.cognitive_controller is not None:
                self.cognitive_controller.on_reset()

            prev_grid = _fast_parse(f)
            self.action_counter += 1
            self.reasoning.assoc_memory.begin_episode()

            if f.available_actions:
                avail_ints = [a for a in f.available_actions if a != 0]

            iter_actions = 1
            level = 0
            run_seq: list = []

            # ── Replay best procedure prefix (if we have one) ──
            procedure = self.reasoning.assoc_memory.get_best_procedure()
            if procedure and best_level >= 0:
                replay_len = min(len(procedure), fast_iter_size // 2)
                for ri in range(replay_len):
                    if iter_actions >= fast_iter_size:
                        break
                    act_int, act_data = procedure[ri]
                    act_int, act_data, cognitive_decision = _unified_fast_choice(
                        prev_grid,
                        act_int,
                        act_data,
                        f,
                    )
                    act = _NAME_TO_ACTION.get(
                        _INT_TO_ACTION_NAME.get(act_int, f"ACTION{act_int}"),
                        GameAction.ACTION1,
                    )
                    if act_int == 6 and act_data:
                        act.set_data(act_data)
                    try:
                        f = env.step(act)
                    except Exception:
                        break
                    if f is None or f.frame is None:
                        break
                    cur_grid = _fast_parse(f)
                    _unified_fast_observe(
                        prev_grid,
                        cur_grid,
                        _INT_TO_ACTION_NAME.get(act_int, f"ACTION{act_int}"),
                        act_data,
                        level,
                        f,
                    )
                    changed = prev_grid is not None and cur_grid is not None and not np.array_equal(prev_grid, cur_grid)
                    lvl_changed = f.levels_completed > level
                    game_over = (f.state == GS.GAME_OVER)
                    self.reasoning.assoc_memory.record_step(
                        prev_grid, act_int, act_data, changed, lvl_changed, game_over,
                    )
                    act_name_r = _INT_TO_ACTION_NAME.get(act_int, f"ACTION{act_int}")
                    if prev_grid is not None and cur_grid is not None:
                        diff = self.analyzer.compute_diff(prev_grid, cur_grid)
                        self.memory.record_action(
                            action_name=act_name_r, grid_before=prev_grid,
                            grid_after=cur_grid, diff=diff,
                            game_state=f.state.name, levels_completed=f.levels_completed,
                        )
                        if self.reasoning.config.uses_visual_cortex():
                            self.reasoning.visual_cortex.record_transition(
                                prev_grid, act_int, act_data, cur_grid,
                            )
                    prev_grid = cur_grid if cur_grid is not None else prev_grid
                    if lvl_changed:
                        level = f.levels_completed
                        print(
                            f"[DIAG {self.game_id}] LEVEL-UP (replay) iter={iteration} "
                            f"replay_idx={ri} levels_completed={f.levels_completed} "
                            f"state={f.state.name}",
                            flush=True,
                        )
                    self.action_counter += 1
                    iter_actions += 1
                    run_seq.append((act_int, act_data))
                    if f.state in (GS.WIN, GS.GAME_OVER):
                        print(
                            f"[DIAG {self.game_id}] {f.state.name} (replay) at replay_idx={ri} "
                            f"levels_completed={f.levels_completed} -- breaking replay",
                            flush=True,
                        )
                        break

            # ── Informed exploration (memory-guided, not blind random) ──
            episode_actions: list = []
            while iter_actions < fast_iter_size:
                if f is None or f.state in (GS.GAME_OVER, GS.NOT_PLAYED):
                    self.reasoning.assoc_memory.record_step(prev_grid, 0, None, False, False, True)
                    self.memory.on_game_over()
                    try:
                        f = env.step(GameAction.RESET)
                        if f is None or f.frame is None:
                            break
                        prev_grid = _fast_parse(f)
                        if self.cognitive_controller is not None:
                            self.cognitive_controller.on_reset()
                    except Exception:
                        break
                    self.action_counter += 1
                    iter_actions += 1
                    episode_actions = []
                    continue
                if f.state == GS.WIN:
                    break

                # Action selection: GameMemory-driven when we have mechanism
                # knowledge, novelty-biased otherwise.
                locked = self.memory.get_locked_mechanisms()
                ranked = self.memory.rank_actions(
                    [_INT_TO_ACTION_NAME.get(a, f"ACTION{a}") for a in avail_ints]
                )
                knowledge_here = self._compute_knowledge_level()

                if knowledge_here >= 0.4 and ranked and random.random() < 0.7:
                    # 70%: use GameMemory-ranked action (mechanism-driven)
                    top_name = ranked[0] if random.random() < 0.6 else random.choice(ranked[:3])
                    act_int = int(top_name.replace("ACTION", ""))
                    act_data = None
                elif (self.reasoning.assoc_memory._train_steps > 10
                        and random.random() < 0.3):
                    # Use full retrieval (associations + NN + danger map)
                    act_int, retrieved_data = self.reasoning.assoc_memory.retrieve_action(
                        prev_grid, avail_ints, episode_actions, temperature=1.5,
                    )
                    act_data = retrieved_data
                else:
                    # Novelty-biased selection (with VC priors)
                    act_int = self.reasoning.assoc_memory.pick_novel_action(avail_ints, episode_actions)
                    act_data = None

                act_int, act_data, cognitive_decision = _unified_fast_choice(
                    prev_grid,
                    act_int,
                    act_data,
                    f,
                )
                act_name = _INT_TO_ACTION_NAME.get(act_int, f"ACTION{act_int}")
                act = _NAME_TO_ACTION.get(act_name, GameAction.ACTION1)

                if act_int == 6 and not act_data:
                    act_data = _informed_click(prev_grid, iteration, iter_actions)
                if act_int == 6 and act_data:
                    act.set_data(act_data)

                try:
                    f = env.step(act)
                except Exception:
                    break
                if f is None or f.frame is None:
                    break

                cur_grid = _fast_parse(f)
                _unified_fast_observe(
                    prev_grid,
                    cur_grid,
                    act_name,
                    act_data,
                    level,
                    f,
                )
                changed = prev_grid is not None and cur_grid is not None and not np.array_equal(prev_grid, cur_grid)
                lvl_changed = f.levels_completed > level
                game_over = (f.state == GS.GAME_OVER)

                self.reasoning.assoc_memory.record_step(
                    prev_grid, act_int, act_data, changed, lvl_changed, game_over,
                )
                if prev_grid is not None and cur_grid is not None:
                    diff = self.analyzer.compute_diff(prev_grid, cur_grid)
                    self.memory.record_action(
                        action_name=act_name, grid_before=prev_grid,
                        grid_after=cur_grid, diff=diff,
                        game_state=f.state.name, levels_completed=f.levels_completed,
                    )
                    if self.reasoning.config.uses_visual_cortex():
                        self.reasoning.visual_cortex.record_transition(
                            prev_grid, act_int, act_data, cur_grid,
                        )
                prev_grid = cur_grid if cur_grid is not None else prev_grid
                if lvl_changed:
                    level = f.levels_completed
                    print(
                        f"[DIAG {self.game_id}] LEVEL-UP iter={iteration} "
                        f"iter_actions={iter_actions} levels_completed={f.levels_completed} "
                        f"state={f.state.name} action={act_name}",
                        flush=True,
                    )
                episode_actions.append(act_int)
                self.action_counter += 1
                iter_actions += 1
                run_seq.append((act_int, act_data))
                if f.state == GS.WIN:
                    print(
                        f"[DIAG {self.game_id}] GS.WIN at iter_actions={iter_actions} "
                        f"levels_completed={f.levels_completed} -- breaking exploration iter",
                        flush=True,
                    )
                    break

            self.reasoning.assoc_memory.end_episode()
            return level, run_seq

        def _run_strategic_iteration():
            """One strategic iteration using the full reasoning loop.

            Time-bounded: runs until time runs out or stall detected.
            Returns (level_reached, initial_grid, final_grid).
            """
            # ── Env lifecycle: rebuild + RESET only when needed ──
            # When the previous iter ended mid-progress (level-up without
            # terminal game_over/WIN), `self._preserve_env_next_iter` is
            # True and we skip the rebuild so the session continues at
            # the current level instead of snapping back to level 0.
            preserved = (
                self.PRESERVE_ENV_ACROSS_ITERS
                and self._preserve_env_next_iter
                and self.arc_env is not None
            )
            print(
                f"[DIAG {self.game_id}] STRAT-ENTRY iter={iteration} "
                f"preserved={preserved} "
                f"flag={self._preserve_env_next_iter} "
                f"cls_enable={self.PRESERVE_ENV_ACROSS_ITERS} "
                f"arc_env_none={self.arc_env is None} "
                f"live_level={self._live_levels_completed}",
                flush=True,
            )
            if preserved:
                # Continue from where the last iter left off.
                print(
                    f"[DIAG {self.game_id}] STRAT iter={iteration} PRESERVE-ENV "
                    f"continuing from level={self._live_levels_completed}",
                    flush=True,
                )
                self._preserve_env_next_iter = False
                iter_level = self._live_levels_completed
                _strat_initial_grid = self._prev_grid
                frame = self.frames[-1] if self.frames else None
                self.reasoning.new_iteration()
                self._maybe_attach_task_program_for_level(iter_level)
                self.reasoning.prime_post_level_followup(iter_level)
            else:
                if self._arcade is not None:
                    try:
                        self.arc_env = self._arcade.make(self.game_id)
                    except Exception:
                        pass

                frame = self.take_action(GameAction.RESET)
                if frame:
                    self.append_frame(frame)
                self.action_counter += 1

                self.reasoning.new_iteration()
                # Capture initial grid right after RESET for progress measurement
                _strat_initial_grid = None
                if frame and frame.frame is not None:
                    _strat_initial_grid = self.analyzer.parse_frame(frame.frame)
                self._prev_grid = _strat_initial_grid
                self._prev_levels = 0
                self._prev_obs = None
                self._prev_strategy = None
                self._prev_trajectory = None
                self._prev_goal_context = None
                self._last_action_name = "RESET"
                self._last_action_data = {}
                self._prev_game_state = "NOT_FINISHED"
                self._needs_reset = False
                self._game_started = True
                if self.cognitive_controller is not None:
                    self.cognitive_controller.on_reset()
                iter_level = 0
                self._live_levels_completed = 0

            iter_start = time.time()
            stall_limit = _stall_limit()
            last_progress_time = iter_start
            # Tracks an extra stall-free window granted after a level-up.
            # While `time.time() < stall_bonus_deadline`, the stall check
            # is suppressed (lets the agent investigate the new level).
            stall_bonus_deadline = 0.0

            # Replay best sequence prefix (if any) — skip entirely when
            # we're preserving env (the replayed prefix no longer aligns
            # with the current mid-game state).
            replay_len = 0 if preserved else min(len(best_seq), 60)
            for ri in range(replay_len):
                if _time_left() <= 0:
                    break
                act_int, act_data = best_seq[ri]
                act_int, act_data, cognitive_decision = _unified_fast_choice(
                    self._prev_grid,
                    act_int,
                    act_data,
                    frame,
                )
                act_name = _INT_TO_ACTION_NAME.get(act_int, f"ACTION{act_int}")
                action = _NAME_TO_ACTION.get(act_name, GameAction.ACTION1)
                if act_int == 6 and act_data:
                    action.set_data(act_data)
                frame = self.take_action(action)
                if frame:
                    self.append_frame(frame)
                    observed_grid = (
                        self.analyzer.parse_frame(frame.frame)
                        if frame.frame is not None
                        else None
                    )
                    _unified_fast_observe(
                        self._prev_grid,
                        observed_grid,
                        act_name,
                        act_data,
                        iter_level,
                        frame,
                    )
                    if frame.state == GS.WIN:
                        break
                    if frame.state == GS.GAME_OVER:
                        self.memory.on_game_over()
                        self.reasoning.on_game_over()
                        break
                    cur_grid = self.analyzer.parse_frame(frame.frame) if frame.frame is not None else None
                    if self._prev_grid is not None and cur_grid is not None:
                        diff = self.analyzer.compute_diff(self._prev_grid, cur_grid)
                        self.memory.record_action(
                            action_name=act_name, grid_before=self._prev_grid,
                            grid_after=cur_grid, diff=diff,
                            game_state=frame.state.name,
                            levels_completed=frame.levels_completed,
                        )
                    if frame.levels_completed > iter_level:
                        iter_level = frame.levels_completed
                        self._live_levels_completed = frame.levels_completed
                        last_progress_time = time.time()
                        stall_bonus_deadline = time.time() + self.LEVEL_UP_BONUS_SECONDS
                        print(
                            f"[DIAG {self.game_id}] STRAT-REPLAY LEVEL-UP "
                            f"iter={iteration} ri={ri} "
                            f"levels_completed={frame.levels_completed} "
                            f"state={frame.state.name} "
                            f"bonus_until={self.LEVEL_UP_BONUS_SECONDS:.0f}s",
                            flush=True,
                        )
                        self._maybe_attach_task_program_for_level(iter_level)
                    self._prev_grid = cur_grid
                    self._prev_levels = frame.levels_completed
                self.action_counter += 1

            # Continue with full reasoning loop — time-bounded with stall restart
            reached_win = False
            game_over_count = 0   # how many deaths this iter
            _first_obs_logged = False
            planning_steps = 0
            trajectory_source_counts: Dict[str, int] = {}
            last_plan_signature: Optional[tuple] = None
            while _time_left() > 0:
                # One-shot probe: what level does the env claim we're on?
                if not _first_obs_logged:
                    try:
                        _probe = self._convert_raw_frame_data(
                            self.arc_env.observation_space if self.arc_env else None
                        )
                        print(
                            f"[DIAG {self.game_id}] STRAT-PROBE iter={iteration} "
                            f"env_state={_probe.state.name} "
                            f"env_levels_completed={_probe.levels_completed} "
                            f"(our iter_level={iter_level})",
                            flush=True,
                        )
                    except Exception as _e:
                        print(
                            f"[DIAG {self.game_id}] STRAT-PROBE iter={iteration} "
                            f"FAILED ({type(_e).__name__})",
                            flush=True,
                        )
                    _first_obs_logged = True

                now_t = time.time()
                in_bonus = now_t < stall_bonus_deadline
                # Stall check: no progress for too long → restart.
                # Suppressed while we're inside a post-level-up bonus
                # window, which lets the agent probe the new level.
                if (now_t - last_progress_time) > stall_limit and not in_bonus:
                    logger.info(
                        f"[{self.game_id}] Strategic stall after "
                        f"{now_t - iter_start:.1f}s → restart"
                    )
                    break

                latest = self._convert_raw_frame_data(
                    self.arc_env.observation_space if self.arc_env else None
                )
                if latest.state == GS.WIN:
                    reached_win = True
                    break

                # On GAME_OVER we DON'T break. `choose_action` will set
                # `_needs_reset` and return RESET, which restarts the
                # level within the SAME env. That lets the agent keep
                # attacking higher levels across many deaths within one
                # outer-loop iteration — deaths are cheap, a fresh
                # iter is not.
                action = self.choose_action(self.frames, latest)
                reasoning_meta = getattr(action, "reasoning", {}) or {}
                traj_source = reasoning_meta.get("trajectory_source")
                if traj_source:
                    planning_steps += 1
                    trajectory_source_counts[traj_source] = (
                        trajectory_source_counts.get(traj_source, 0) + 1
                    )
                    plan_signature = (
                        traj_source,
                        reasoning_meta.get("subgoal"),
                        action.name,
                    )
                    if (
                        planning_steps <= 6
                        or planning_steps % 20 == 0
                        or plan_signature != last_plan_signature
                    ):
                        source_counts = reasoning_meta.get("trajectory_source_counts") or {}
                        top_candidates = reasoning_meta.get("trajectory_top_candidates") or []
                        top_summary = ", ".join(
                            (
                                f"#{item.get('rank')} {item.get('source')} "
                                f"{item.get('first_action')} s={item.get('score')}"
                            )
                            for item in top_candidates[:3]
                        )
                        latent_program = reasoning_meta.get("trajectory_latent_task_program") or {}
                        latent_actions = ",".join(
                            (latent_program.get("preferred_actions") or [])[:4]
                        )
                        latent_sequences = latent_program.get("preferred_sequences") or []
                        latent_macro = ""
                        if latent_sequences:
                            latent_macro = ">".join(latent_sequences[0][:5])
                        print(
                            f"[DIAG {self.game_id}] PLAN iter={iteration} "
                            f"plan_step={planning_steps} action={action.name} "
                            f"subgoal={reasoning_meta.get('subgoal')} "
                            f"source={traj_source} "
                            f"score={reasoning_meta.get('trajectory_score')} "
                            f"energy={reasoning_meta.get('trajectory_energy')} "
                            f"goal_prog={reasoning_meta.get('trajectory_goal_progress')} "
                            f"risk={reasoning_meta.get('trajectory_risk')} "
                            f"sources={source_counts} "
                            f"latent=[{latent_actions}] "
                            f"macro=[{latent_macro}] "
                            f"lat_conf={latent_program.get('confidence')} "
                            f"top=[{top_summary}]",
                            flush=True,
                        )
                    last_plan_signature = plan_signature
                frame = self.take_action(action)
                if frame:
                    self.append_frame(frame)
                self.action_counter += 1

                # Track progress
                if frame and frame.levels_completed > iter_level:
                    iter_level = frame.levels_completed
                    self._live_levels_completed = frame.levels_completed
                    last_progress_time = time.time()
                    stall_bonus_deadline = time.time() + self.LEVEL_UP_BONUS_SECONDS
                    print(
                        f"[DIAG {self.game_id}] STRAT LEVEL-UP iter={iteration} "
                        f"levels_completed={frame.levels_completed} "
                        f"state={frame.state.name} "
                        f"t_since_iter_start={time.time() - iter_start:.1f}s "
                        f"bonus={self.LEVEL_UP_BONUS_SECONDS:.0f}s",
                        flush=True,
                    )
                    self._maybe_attach_task_program_for_level(iter_level)

                if frame and frame.state == GS.GAME_OVER:
                    game_over_count += 1
                    self.memory.on_game_over()
                    self.reasoning.on_game_over()
                    # stay in the loop — next choose_action will RESET

                if frame and frame.state == GS.WIN:
                    reached_win = True
                    print(
                        f"[DIAG {self.game_id}] STRAT GS.WIN iter={iteration} "
                        f"levels_completed={frame.levels_completed} "
                        f"t_since_iter_start={time.time() - iter_start:.1f}s",
                        flush=True,
                    )
                    break

            # Decide whether to preserve the env for the next iter.
            # Preserve iff we made level progress AND did NOT end with a
            # final WIN. We preserve even after deaths, because RESET
            # inside a live env may return us to the progressed level
            # rather than all the way back to level 0 (depends on env).
            if (
                self.PRESERVE_ENV_ACROSS_ITERS
                and iter_level > 0
                and not reached_win
            ):
                self._preserve_env_next_iter = True
            else:
                self._preserve_env_next_iter = False
            print(
                f"[DIAG {self.game_id}] STRAT-EXIT iter={iteration} "
                f"iter_level={iter_level} deaths_in_iter={game_over_count} "
                f"reached_win={reached_win} "
                f"set_preserve_flag={self._preserve_env_next_iter} "
                f"trajectory_sources={trajectory_source_counts}",
                flush=True,
            )

            cur_level = self.frames[-1].levels_completed if self.frames else iter_level
            _strat_final_grid = self._prev_grid  # last grid seen
            return cur_level, _strat_initial_grid, _strat_final_grid

        # ══════════════════════════════════════════════════════════
        # UNIFIED ADAPTIVE LOOP — goal-directed, time-based
        # ══════════════════════════════════════════════════════════
        #
        # Three modes:
        #   1. EXPLORE (ε > 0.5): fast informed exploration to build knowledge
        #   2. GOAL_BANK: generate/regenerate goal hypotheses once ε drops
        #   3. GOAL_PURSUIT: goal-conditioned strategy → execute → measure
        #      → retry strategy / switch goal / regenerate bank
        #
        gpm = self.reasoning.goal_pursuit
        goal_bank_ready = False

        def _has_trace_bootstrap_prior() -> bool:
            if not self.reasoning.config.uses_trajectory_memory():
                return False
            trace_memory = getattr(self.reasoning, "human_trace_memory", None)
            traces = list(getattr(trace_memory, "traces", []) or [])
            for trace in traces:
                if not getattr(trace, "actions", None):
                    continue
                if getattr(trace, "final_state", "") == "WIN":
                    return True
                if int(getattr(trace, "levels_completed", 0) or 0) > 0:
                    return True
            return False

        while _time_left() > 0 and total_wins < self.MAX_WINS:
            iteration += 1
            knowledge = self._compute_knowledge_level()
            epsilon = self._compute_epsilon(knowledge, iteration)
            trace_bootstrap_ready = _has_trace_bootstrap_prior()
            path_label = (
                "goal_pursuit(trace_bootstrap)"
                if trace_bootstrap_ready
                else ("explore" if epsilon > 0.5 else "goal_pursuit")
            )
            level = 0
            mode = "explore"
            print(
                f"[DIAG {self.game_id}] ITER-START iter={iteration} "
                f"knowledge={knowledge:.2f} epsilon={epsilon:.2f} "
                f"total_wins={total_wins} best_level={best_level} "
                f"path={path_label}",
                flush=True,
            )

            # ── Phase 1: EXPLORE while knowledge is low ──
            if epsilon > 0.5 and not trace_bootstrap_ready:
                level, seq = _run_exploration_iteration()
                mode = "explore"
                _strat_grids = (None, None)  # no grid tracking for explore

                if level > best_level:
                    best_level = level
                    if seq:
                        best_seq = seq.copy()
                if level > 0:
                    total_wins += 1
                print(
                    f"[DIAG {self.game_id}] ITER-END (explore) iter={iteration} "
                    f"level={level} best_level={best_level} total_wins={total_wins}",
                    flush=True,
                )

            else:
                # ── Phase 2: GOAL PURSUIT ──

                # Generate goal bank if needed
                if not goal_bank_ready or gpm.should_regenerate_bank():
                    obs = self.reasoning.describer.describe(
                        grid=self._prev_grid if self._prev_grid is not None else np.zeros((8, 8), dtype=np.uint8),
                        memory=self.memory,
                        game_state="NOT_FINISHED",
                        levels_completed=best_level,
                        action_counter=self.action_counter,
                    )
                    self.reasoning.generate_goal_bank(
                        obs, self._available_action_names,
                    )
                    goal_bank_ready = True
                    logger.info(
                        f"[{self.game_id}] Goal bank: "
                        f"{[g.id for g in gpm.goals]}"
                    )

                # Switch goal if current one is exhausted
                if gpm.should_switch_goal():
                    new_goal = gpm.select_next_goal()
                    if new_goal is None:
                        # All goals exhausted → regenerate bank
                        goal_bank_ready = False
                        continue

                goal = gpm.active_goal
                if goal is None:
                    # Fallback: explore more
                    level, seq = _run_exploration_iteration()
                    mode = "explore"
                    _strat_grids = (None, None)
                    if level > best_level:
                        best_level = level
                        if seq:
                            best_seq = seq.copy()
                    if level > 0:
                        total_wins += 1
                else:
                    # ── Goal-conditioned strategic iteration ──
                    mode = f"goal:{goal.id}"

                    # Build observation for strategy generation
                    obs = self.reasoning.describer.describe(
                        grid=self._prev_grid if self._prev_grid is not None else np.zeros((8, 8), dtype=np.uint8),
                        memory=self.memory,
                        game_state="NOT_FINISHED",
                        levels_completed=best_level,
                        action_counter=self.action_counter,
                    )

                    # Generate goal-conditioned strategy (fresh each time
                    # for diversity — templates are instant, LLM only for bank)
                    strategy = self.reasoning.strategize_for_goal(
                        obs, self._available_action_names,
                    )

                    # Execute the strategy — captures initial/final grids
                    states_before = len(self.memory._visited_hashes)
                    level, grid_init, grid_final = _run_strategic_iteration()

                    # Snapshot + measure using the actual episode grids
                    gpm.begin_strategy(
                        strategy_description=strategy.description if strategy else "explore_fallback",
                        strategy_type=strategy.strategy_type.value if strategy else "explore",
                        grid=grid_init,
                        levels_completed=best_level,
                        states_visited=states_before,
                        game_id=self.game_id,
                    )
                    outcome = gpm.end_strategy(
                        grid_after=grid_final,
                        levels_completed=max(level, best_level),
                        states_visited=len(self.memory._visited_hashes),
                        game_over=False,
                    )

                    # Store in associative memory
                    self.reasoning.assoc_memory.record_strategy_outcome(outcome)

                    # Track best
                    if level > best_level:
                        best_level = level
                    if level > 0:
                        total_wins += 1

                    # Decide next action
                    decision = gpm.decide_next_action(outcome)
                    logger.info(
                        f"[{self.game_id}] Goal '{goal.id}' attempt {goal.attempts}: "
                        f"progress={outcome.progress_score:.2f} "
                        f"({outcome.terminal_status}) → {decision}"
                    )

                    if decision == "regenerate":
                        goal_bank_ready = False
                    elif decision == "new_goal":
                        gpm.select_next_goal()

            # ── Plateau detection + LIVE LLM revision ─────────
            # If `best_level` advanced this iter, reset counter; else
            # increment. After LIVE_REVISE_AFTER_ITERS iters of plateau
            # (and at least the same gap since the last live revision),
            # trigger a LIVE LLM revision: re-compile a TaskProgram on
            # the fly using the agent's accumulated runtime observations.
            if level > self._iter_best_level_snapshot:
                self._iters_no_progress = 0
            else:
                self._iters_no_progress += 1
            self._iter_best_level_snapshot = best_level

            iters_since_last_live = (
                iteration - self._live_revision_last_trigger_iter
            )
            if (
                self._iters_no_progress >= self.LIVE_REVISE_AFTER_ITERS
                and best_level >= 0
                and iters_since_last_live >= self.LIVE_REVISE_AFTER_ITERS
                and self._live_revision_attempts < self.MAX_LIVE_REVISIONS
                and self._live_revise_disabled_reason is None
            ):
                self._live_revision_last_trigger_iter = iteration
                if self._try_live_llm_revise_task_program(best_level):
                    self._iters_no_progress = 0
                    goal_bank_ready = False

            # ── Continuous VC training (every N iterations) ──
            vc = self.reasoning.visual_cortex
            if (
                self.ENABLE_VISUAL_CORTEX_WARMUP
                and self.reasoning.config.uses_visual_cortex()
                and iteration % vc_train_interval == 0
                and vc.buffer_size > 0
            ):
                train_steps = 10 if vc.trained_steps < 30 else 5
                vc.train(steps=train_steps)

            logger.info(
                f"[{self.game_id}] iter {iteration} [{mode}]: "
                f"level={level}, best={best_level}, ε={epsilon:.2f}, "
                f"knowledge={knowledge:.2f}, wins={total_wins}, "
                f"time_left={_time_left():.1f}s"
                + (f", goal={gpm.active_goal.id}" if gpm.active_goal else "")
            )

        # ── Final VC training pass (use any remaining buffer) ──
        vc = self.reasoning.visual_cortex
        if (
            self.ENABLE_VISUAL_CORTEX_WARMUP
            and self.reasoning.config.uses_visual_cortex()
            and vc.buffer_size > 10
            and vc.trained_steps < 80
        ):
            vc.train(steps=min(20, vc.buffer_size // 4))

        self.cleanup()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{super().name}.adaptive_reasoning_v3.{self.TIME_BUDGET:.0f}s"

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        """Export cross-game learnings and log final state before cleanup."""
        # Export learnings to cross-game memory before cleanup
        self.reasoning.export_cross_game()
        if self._cleanup:
            summary = self.memory.summary()
            loop_stats = self.reasoning.get_stats()
            logger.info(
                f"[{self.game_id}] Final: "
                f"actions={summary['total_actions']}, "
                f"max_level={summary['max_level']}, "
                f"explore={summary['exploration_score']:.2f}, "
                f"goal={loop_stats.get('overarching_goal', 'none')}, "
                f"subgoals_done={loop_stats['subgoals_completed']}, "
                f"strategies={loop_stats['strategies_tried']}, "
                f"wm_steps={loop_stats['wm_trained_steps']}, "
                f"goal_pursuit={loop_stats.get('goal_pursuit', {})}"
            )
            if hasattr(self, "recorder") and not self.is_playback:
                self.recorder.record({
                    "adaptive_reasoning_summary": summary,
                    "reasoning_loop_stats": loop_stats,
                })
        super().cleanup(*args, **kwargs)
