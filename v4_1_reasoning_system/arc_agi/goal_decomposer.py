"""
Goal Decomposer — hierarchical win-condition decomposition for ARC-AGI-3.

Given observations + memorised game mechanics, produces:
  1. An **overarching goal** (what we believe the game's win condition is).
  2. An ordered list of **subgoals** that, when achieved sequentially,
     should satisfy the overarching goal.

Supports two backends:
  - **LLM mode**: uses a causal LM to reason about mechanics → goals.
  - **Template mode**: heuristic decomposition (fallback when no LLM).

The decomposer is re-invoked whenever:
  - A level is completed (new level may have new win conditions).
  - Enough actions have elapsed without progress (re-think).
  - A game-over occurs (previous hypothesis was wrong).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .state_describer import GameObservation
from .game_memory import GameMemory
from .llm_cache import get_shared_llm
from .goal_pursuit import GameObjective, ProgressSignal, ObjectiveStatus

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
class SubGoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubGoal:
    """A concrete, verifiable sub-objective."""
    id: int
    description: str                        # NL description
    success_hint: str                       # how to detect success
    priority: int = 0                       # lower = do first
    status: SubGoalStatus = SubGoalStatus.PENDING
    actions_spent: int = 0                  # actions spent on this subgoal
    max_actions: int = 30                   # budget before giving up
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            SubGoalStatus.ACHIEVED,
            SubGoalStatus.FAILED,
            SubGoalStatus.SKIPPED,
        )


@dataclass
class GameGoal:
    """Hierarchical goal structure for the current game / level."""
    overarching_goal: str                   # e.g. "Navigate to exit"
    hypothesis: str                         # our theory about win condition
    subgoals: List[SubGoal] = field(default_factory=list)
    confidence: float = 0.5                 # how confident we are
    revision: int = 0                       # how many times we re-decomposed

    @property
    def current_subgoal(self) -> Optional[SubGoal]:
        for sg in self.subgoals:
            if sg.status in (SubGoalStatus.PENDING, SubGoalStatus.ACTIVE):
                return sg
        return None

    @property
    def all_achieved(self) -> bool:
        return all(
            sg.status == SubGoalStatus.ACHIEVED for sg in self.subgoals
        )

    @property
    def progress_fraction(self) -> float:
        if not self.subgoals:
            return 0.0
        achieved = sum(1 for sg in self.subgoals if sg.status == SubGoalStatus.ACHIEVED)
        return achieved / len(self.subgoals)

    def to_prompt(self) -> str:
        lines = [
            f"## Current Plan (revision {self.revision}, confidence {self.confidence:.0%})",
            f"**Goal**: {self.overarching_goal}",
            f"**Hypothesis**: {self.hypothesis}",
            "",
            "### Subgoals",
        ]
        for sg in self.subgoals:
            marker = {"achieved": "✓", "active": "→", "failed": "✗",
                      "skipped": "–", "pending": " "}[sg.status.value]
            lines.append(
                f"  [{marker}] {sg.id}. {sg.description} "
                f"(hint: {sg.success_hint}, budget: {sg.actions_spent}/{sg.max_actions})"
            )
        return "\n".join(lines)


# ------------------------------------------------------------------
# LLM prompt for goal decomposition
# ------------------------------------------------------------------
_DECOMPOSE_SYSTEM = """\
You are an expert game analyst for interactive grid-based games.
Given observations about game mechanics (discovered via exploration),
you must:

1. State an **overarching goal** — what you believe the win condition is.
2. State a **hypothesis** — your theory about how the game works.
3. Produce an ordered list of **subgoals** that achieve the win condition.

Each subgoal must be:
- Concrete (can be attempted with available actions)
- Verifiable (has a success hint the agent can check)
- Small enough to accomplish in ≤30 actions

Respond with a JSON object:
{
  "overarching_goal": "...",
  "hypothesis": "...",
  "confidence": 0.0-1.0,
  "subgoals": [
    {
      "description": "...",
      "success_hint": "...",
      "priority": 0,
      "max_actions": 30
    },
    ...
  ]
}

Common game patterns:
- Navigation: move player to specific tiles/exits
- Collection: visit all items available and bring them back to the warehouse
- Puzzle: push objects to target locations
- Sequence: perform actions in specific order
- Click-based: click on specific objects to toggle/activate them
- Avoidance: reach goal while avoiding hazards (game-over tiles)

Be creative but grounded in the observations provided.
"""


# ------------------------------------------------------------------
# Goal Decomposer
# ------------------------------------------------------------------
class GoalDecomposer:
    """
    Produces a hierarchical GameGoal from observations + memory.

    Operates in two modes:
      1. LLM: feeds observation summary → LM → JSON goal structure
      2. Template: rule-based heuristic decomposition (always available)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        use_llm: bool = False,
    ):
        self.use_llm = use_llm and model_name is not None
        self._model = None
        self._tokenizer = None
        self._model_name = model_name
        self._device = device
        if self.use_llm:
            self._init_model()

    def _init_model(self) -> None:
        model, tokenizer = get_shared_llm(self._model_name, self._device)
        if model is not None:
            self._model = model
            self._tokenizer = tokenizer
        else:
            logger.warning("LLM unavailable for goal decomposition. Using templates.")
            self.use_llm = False

    # ------------------------------------------------------------------
    # Task-program integration (Phase 2a)
    # ------------------------------------------------------------------
    # A compiled `TaskProgram` (structured human-trace distillation, see
    # `human_trace.task_program.TaskProgram`) can be attached here. When
    # present, it takes priority over LLM / template decomposition so
    # that the agent's strategic layer pursues the exact subgoals the
    # human-trace compiler produced, not generic heuristics.
    #
    # The program is duck-typed — we only access documented attributes
    # so this module doesn't need to import `human_trace.task_program`
    # and create a cross-package dep the other way.
    _task_program: Optional[Any] = None   # type: ignore[assignment]

    def set_task_program(self, program: Optional[Any]) -> None:
        """Attach or clear a compiled TaskProgram.

        Pass None to remove a previously-attached program.
        """
        self._task_program = program
        if program is not None:
            logger.info(
                "GoalDecomposer: task program loaded (goal_family=%s, "
                "%d subgoal_tests, confidence=%.2f)",
                getattr(program, "goal_family", "?"),
                len(getattr(program, "subgoal_tests", []) or []),
                float(getattr(program, "confidence", 0.0) or 0.0),
            )

    # ------------------------------------------------------------------
    # Human-prior extraction (opt-in; empty dict if nothing seeded)
    # ------------------------------------------------------------------
    # Minimum confidence for a `game_type::*` hypothesis to override the
    # classification heuristics. Below this we still surface priors in the
    # LLM prompt / hypothesis string but don't force the game_type.
    _GAME_TYPE_OVERRIDE_CONF: float = 0.6

    # Canonical goal-id (used by human_trace + CrossGameMemory) → the
    # internal template-type label `_classify_game` produces.
    _CANON_TO_GAME_TYPE: Dict[str, str] = {
        "click_puzzle":     "click_puzzle",
        "sequence_puzzle":  "sequence_puzzle",
        "navigate_exit":    "navigation",
        "navigate_puzzle":  "navigation",
        "collection":       "collection",
        "push_puzzle":      "push_puzzle",
        "transform_puzzle": "sequence_puzzle",
        "unknown":          "unknown",
    }

    def _extract_human_priors(self, memory: GameMemory) -> Dict[str, Any]:
        """Pull the human-seeded hypotheses out of `memory.hypotheses`.

        The `human_trace.integration` pipeline inserts four flavours of keys:
            game_type::<canonical_goal_id>
            objective::<free text>
            mechanic::<free text>
            human::<free text>        (sticky per-step hypotheses)

        If the agent was not primed with human traces, all of these will be
        absent and this returns an empty dict — callers must tolerate that.
        """
        priors: Dict[str, Any] = {
            "game_type": None,          # (canonical_goal_id, confidence) or None
            "objective": None,          # (text, confidence) or None
            "mechanics": [],            # list[(text, confidence)]
            "observations": [],         # list[(text, confidence)] from human:: keys
        }
        hyps = getattr(memory, "hypotheses", None)
        if not hyps:
            return priors

        best_game_type: Optional[tuple[str, float]] = None
        best_objective: Optional[tuple[str, float]] = None
        for key, conf in hyps.items():
            if not isinstance(key, str):
                continue
            if key.startswith("game_type::"):
                gid = key.split("::", 1)[1].strip()
                if best_game_type is None or conf > best_game_type[1]:
                    best_game_type = (gid, float(conf))
            elif key.startswith("objective::"):
                text = key.split("::", 1)[1].strip()
                if best_objective is None or conf > best_objective[1]:
                    best_objective = (text, float(conf))
            elif key.startswith("mechanic::"):
                priors["mechanics"].append((key.split("::", 1)[1].strip(), float(conf)))
            elif key.startswith("human::"):
                priors["observations"].append((key.split("::", 1)[1].strip(), float(conf)))

        priors["game_type"] = best_game_type
        priors["objective"] = best_objective
        # Keep the list sizes bounded so prompts stay small
        priors["mechanics"].sort(key=lambda t: -t[1])
        priors["mechanics"] = priors["mechanics"][:6]
        priors["observations"].sort(key=lambda t: -t[1])
        priors["observations"] = priors["observations"][:4]
        return priors

    def _human_priors_prompt(self, priors: Dict[str, Any]) -> str:
        """Render human priors as a small NL block for LLM consumption.

        Returns an empty string if no priors are present, so it's cheap to
        unconditionally concatenate.
        """
        lines: List[str] = []
        gt = priors.get("game_type")
        if gt:
            lines.append(f"- A prior human player believes this is a **{gt[0]}** game "
                         f"(confidence {gt[1]:.0%}).")
        obj = priors.get("objective")
        if obj:
            lines.append(f"- Stated human objective: \"{obj[0]}\" (confidence {obj[1]:.0%}).")
        for m, c in priors.get("mechanics", []):
            lines.append(f"- Mechanic observed by human: {m} (confidence {c:.0%}).")
        for o, c in priors.get("observations", []):
            lines.append(f"- Human hypothesis: {o} (confidence {c:.0%}).")
        if not lines:
            return ""
        return "## Prior knowledge from a human playthrough\n" + "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decompose(
        self,
        observation: GameObservation,
        memory: GameMemory,
        previous_goal: Optional[GameGoal] = None,
    ) -> GameGoal:
        """
        Produce a GameGoal (overarching goal + subgoals).

        Args:
            observation: current structured game observation
            memory: accumulated game knowledge
            previous_goal: the previous goal (if re-decomposing)
        """
        # Phase 2a: if a compiled TaskProgram is attached, it takes
        # priority. The program encodes human-trace-derived subgoal
        # tests that are far richer than generic template heuristics.
        if self._task_program is not None:
            goal = self._decompose_from_program(
                self._task_program, observation, memory, previous_goal
            )
            if goal is not None:
                return goal

        if self.use_llm and self._model is not None:
            goal = self._decompose_with_llm(observation, memory, previous_goal)
            if goal is not None:
                return goal

        return self._decompose_from_templates(observation, memory, previous_goal)

    # ------------------------------------------------------------------
    # LLM-based decomposition
    # ------------------------------------------------------------------
    def _decompose_with_llm(
        self,
        observation: GameObservation,
        memory: GameMemory,
        previous_goal: Optional[GameGoal],
    ) -> Optional[GameGoal]:
        prompt = observation.to_prompt()
        # Inject human priors if the agent was primed from recorded traces.
        priors_block = self._human_priors_prompt(self._extract_human_priors(memory))
        if priors_block:
            prompt = priors_block + "\n" + prompt
        if previous_goal:
            prompt += "\n\n" + previous_goal.to_prompt()
            prompt += "\n\n⚠ The previous plan did not work. Revise it."
        prompt += "\n\nDecompose the win condition into subgoals (JSON):"

        # Truncate prompt to keep inference fast
        if len(prompt) > 1500:
            prompt = prompt[:1500] + "\n...(truncated)"

        messages = [
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            import torch
            inputs = self._tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True
            ).to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=192, temperature=0.7,
                    do_sample=True, top_p=0.9,
                )
            raw = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
            return self._parse_llm_response(raw, previous_goal)
        except Exception as e:
            logger.warning(f"LLM goal decomposition failed: {e}")
            return None

    def _parse_llm_response(
        self, raw: str, previous_goal: Optional[GameGoal]
    ) -> Optional[GameGoal]:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

        subgoals = []
        for i, sg in enumerate(data.get("subgoals", [])):
            subgoals.append(SubGoal(
                id=i,
                description=sg.get("description", ""),
                success_hint=sg.get("success_hint", ""),
                priority=sg.get("priority", i),
                max_actions=sg.get("max_actions", 30),
            ))

        revision = (previous_goal.revision + 1) if previous_goal else 0
        return GameGoal(
            overarching_goal=data.get("overarching_goal", "Win the game"),
            hypothesis=data.get("hypothesis", "Unknown"),
            subgoals=subgoals,
            confidence=float(data.get("confidence", 0.5)),
            revision=revision,
        )

    # ------------------------------------------------------------------
    # TaskProgram-based decomposition (Phase 2a)
    # ------------------------------------------------------------------
    # Canonical goal-family (from human_trace.task_program.GoalFamily)
    # → internal game_type label (used for `_estimate_confidence`).
    _GOAL_FAMILY_TO_GAME_TYPE: Dict[str, str] = {
        "navigation":        "navigation",
        "collection":        "collection",
        "ordering":          "sequence_puzzle",
        "alignment":         "navigation",
        "correspondence":    "navigation",
        "symmetry":          "navigation",
        "exclusion":         "collection",
        "toggle_activation": "click_puzzle",
        "click_sequence":    "click_puzzle",
        "click_selection":   "click_puzzle",
        "unknown":           "unknown",
        "other":             "unknown",
    }

    # Mapping SubgoalTest.expected_signal → a natural-language hint
    # suitable for SubGoal.success_hint (consumed by the goal-pursuit
    # progress manager as a tie-breaker).
    _SIGNAL_TO_HINT: Dict[str, str] = {
        "grid_change":            "grid state changes",
        "object_moved":           "a non-player object changes position",
        "level_advance":          "levels_completed increments",
        "role_switch":            "an entity's behaviour changes role",
        "overlap_achieved":       "player overlaps or aligns with target",
        "click_triggered_change": "clicking produces a grid change",
        "click_absorbed_no_change": "click registers without visible effect",
        "object_disappeared":     "a non-player object is removed",
        "new_region_revealed":    "previously hidden cells are revealed",
        "other":                  "observable progress signal",
    }

    @staticmethod
    def _probe_already_satisfied(test: Any, memory: GameMemory) -> bool:
        """True when a probe-style TaskProgram test is already grounded."""
        prefer = [
            str(action).strip().upper()
            for action in list(getattr(test, "prefer_actions", []) or [])
            if str(action).strip()
        ]
        if not prefer:
            return False

        expected_signal = str(getattr(test, "expected_signal", "other") or "other")
        profiles = [memory.action_profiles.get(action) for action in prefer]
        profiles = [profile for profile in profiles if profile is not None]
        if not profiles:
            return False

        if expected_signal == "role_switch":
            return any(
                profile.times_tried >= 3 and profile.change_rate >= 0.60
                for profile in profiles
            )
        if expected_signal in {"object_moved", "overlap_achieved"}:
            return any(
                profile.times_tried >= 2 and (
                    profile.is_movement_action(threshold=0.25)
                    or profile.dominant_displacement is not None
                )
                for profile in profiles
            )
        if expected_signal in {"grid_change", "click_triggered_change"}:
            return any(
                profile.times_tried >= 2 and profile.change_rate >= 0.30
                for profile in profiles
            )
        return False

    def _decompose_from_program(
        self,
        program: Any,
        observation: GameObservation,
        memory: GameMemory,
        previous_goal: Optional[GameGoal],
    ) -> Optional[GameGoal]:
        """Build a GameGoal whose subgoals come from a TaskProgram.

        Graceful degradation: if the program has no subgoal tests (rare
        but possible with small-LLM compiler output), we fall back to
        template decomposition so the agent always has *something* to
        execute.
        """
        tests = list(getattr(program, "subgoal_tests", []) or [])
        if not tests:
            logger.warning(
                "TaskProgram has no subgoal_tests; falling back to templates"
            )
            return None

        revision = (previous_goal.revision + 1) if previous_goal else 0

        # If the previous plan revision already exhausted the human
        # program's subgoals, rotate: move the first subgoal to the
        # end so the agent tries a different starting point. This is
        # the light-weight "on_plan_stalled" hook for Phase 2a — a
        # full version lives in the reasoning loop (Phase 2b).
        if revision > 0 and len(tests) >= 2:
            tests = tests[revision % len(tests):] + tests[: revision % len(tests)]

        program_level = int(getattr(program, "_target_level", 1) or 1)
        program_anti_patterns = list(getattr(program, "anti_patterns", []) or [])
        action_role_actions: Dict[str, List[str]] = {}
        for role in list(getattr(program, "action_roles", []) or []):
            action_name = str(getattr(role, "action", "") or "").strip().upper()
            role_name = str(getattr(role, "role", "") or "").strip().lower()
            if not action_name.startswith("ACTION") or not role_name:
                continue
            bucket = action_role_actions.setdefault(role_name, [])
            if action_name not in bucket:
                bucket.append(action_name)

        def _actions_for_roles(*role_names: str) -> List[str]:
            actions: List[str] = []
            for role_name in role_names:
                for action_name in action_role_actions.get(role_name, []):
                    if action_name not in actions:
                        actions.append(action_name)
            return actions

        control_switch_actions = _actions_for_roles("control_switch")
        movement_actions = _actions_for_roles("movement", "push")
        subgoals: List[SubGoal] = []
        for i, test in enumerate(tests):
            test_id = str(getattr(test, "id", "") or "").lower()
            is_probe = test_id.startswith("probe") or test_id.startswith("identify")
            if (
                program_level > 1
                and is_probe
                and self._probe_already_satisfied(test, memory)
            ):
                logger.info(
                    "Skipping already-confirmed TaskProgram probe %s for level %s",
                    test_id or f"test_{i}",
                    program_level,
                )
                continue
            desc = getattr(test, "description", "") or getattr(test, "id", f"test_{i}")
            verify = getattr(test, "verification", "") or ""
            sig = getattr(test, "expected_signal", "other") or "other"
            hint_tail = self._SIGNAL_TO_HINT.get(sig, "")
            success_hint = verify or hint_tail or "progress signal observed"

            # If the test specifies preferred actions or a click target,
            # surface them in metadata keys the Actioner already consumes
            # (see arc_agi.actioner._act_click / _get_target).
            metadata: Dict[str, Any] = {
                "from_task_program": True,
                "expected_signal": sig,
                "program_level": program_level,
                # Preserve the original TaskProgram test id so the
                # Actioner can key on it (e.g. "probe_*" → hard bias).
                "task_program_id": str(getattr(test, "id", "") or ""),
            }
            if control_switch_actions:
                metadata["control_switch_actions"] = list(control_switch_actions)
            if movement_actions:
                metadata["movement_actions"] = list(movement_actions)
            if program_anti_patterns:
                metadata["program_anti_patterns"] = [
                    str(item)[:120] for item in program_anti_patterns[:3]
                ]
            prefer = list(getattr(test, "prefer_actions", []) or [])
            if prefer:
                metadata["prefer_actions"] = prefer
            click_xy = getattr(test, "click_target_xy", None)
            if click_xy is not None and len(click_xy) == 2:
                cx, cy = int(click_xy[0]), int(click_xy[1])
                # Actioner contract: list of {"x": int, "y": int} dicts
                metadata["click_targets"] = [{"x": cx, "y": cy}]
                # And for nav targets it reads target_y / target_x
                metadata["target_y"] = cy
                metadata["target_x"] = cx
            click_color = getattr(test, "click_target_color", None)
            if click_color is not None:
                metadata["click_target_color"] = int(click_color)
            preconds = list(getattr(test, "preconditions", []) or [])
            if preconds:
                metadata["preconditions"] = preconds

            subgoals.append(SubGoal(
                id=i,
                priority=i,
                description=f"[human-program] {desc[:120]}",
                success_hint=success_hint[:120],
                max_actions=int(getattr(test, "max_actions", 30) or 30),
                metadata=metadata,
            ))

        # Optional: prepend a minimum-exploration subgoal on the very
        # first revision so the agent doesn't leap straight into
        # exploitation with an untested program. (Phase 2b will add a
        # proper exploration floor in the reasoning loop; this is the
        # cheap version.)
        if revision == 0 and program_level <= 1:
            subgoals.insert(0, SubGoal(
                id=-1, priority=-1, max_actions=12,
                description="[human-program] Probe: try each available action at least once to confirm the program's assumptions about mechanics",
                success_hint="each action tried ≥1× and its effect recorded",
                metadata={
                    "from_task_program": True,
                    "probe": True,
                    "program_level": program_level,
                    "task_program_id": "probe_bootstrap_task_program",
                },
            ))
            # re-number ids 0..N
            for new_id, sg in enumerate(subgoals):
                sg.id = new_id
                sg.priority = new_id

        goal_family = getattr(program, "goal_family", "unknown") or "unknown"
        mapped_type = self._GOAL_FAMILY_TO_GAME_TYPE.get(goal_family, "unknown")

        macro = getattr(program, "macro_goal", "") or "Pursue human-defined objective"
        win_hyps = list(getattr(program, "win_condition_hypotheses", []) or [])
        anti = program_anti_patterns
        hyp_bits = [f"goal_family={goal_family}"]
        if win_hyps:
            hyp_bits.append("win_cond: " + "; ".join(win_hyps[:2])[:160])
        if anti:
            hyp_bits.append("avoid: " + "; ".join(anti[:2])[:160])

        prog_conf = float(getattr(program, "confidence", 0.5) or 0.5)
        # Damp the program's self-reported confidence so it doesn't
        # fully suppress exploration — the program is a prior, not truth.
        goal_conf = min(0.85, 0.4 + 0.4 * prog_conf)

        goal = GameGoal(
            overarching_goal=f"[human] {macro}"[:160],
            hypothesis=" || ".join(hyp_bits)[:300],
            subgoals=subgoals,
            confidence=goal_conf,
            revision=revision,
        )
        logger.info(
            "Goal decomposed from task_program (rev=%d): family=%s, "
            "%d subgoals (conf=%.2f)",
            revision, goal_family, len(subgoals), goal_conf,
        )
        return goal

    # ------------------------------------------------------------------
    # Template-based decomposition (always-available fallback)
    # ------------------------------------------------------------------
    def _decompose_from_templates(
        self,
        obs: GameObservation,
        memory: GameMemory,
        previous_goal: Optional[GameGoal],
    ) -> GameGoal:
        """
        Heuristic goal decomposition based on observed game properties.

        Analyses:
          - Player movement capabilities
          - Object types and distributions
          - Action semantics learned during exploration
          - Click effectiveness
          - Game-over patterns
        """
        revision = (previous_goal.revision + 1) if previous_goal else 0
        subgoals: List[SubGoal] = []
        sg_id = 0

        has_player = obs.player_info is not None
        move_actions = memory.get_movement_actions()
        semantics = memory.infer_action_semantics()
        n_objects = len(obs.objects) if obs.objects else 0
        effective_clicks = memory.get_effective_click_values()
        has_game_overs = memory.total_game_overs > 0
        has_interact = any(
            v in ("interact", "activate", "toggle")
            for v in semantics.values()
        )

        # Human priors (empty if the agent wasn't primed from traces).
        human_priors = self._extract_human_priors(memory)

        # ── Classify the likely game type ─────────────────────────
        available_actions = list(memory.action_profiles.keys()) or [
            f"ACTION{i}" for i in range(1, 8)
        ]

        # If a human prior for `game_type::*` exists above the override
        # threshold, use it instead of the observation heuristics. Below
        # the threshold, priors still appear in the hypothesis string but
        # heuristics decide the classification.
        game_type = None
        hypothesis = None
        gt_prior = human_priors.get("game_type")
        if gt_prior and gt_prior[1] >= self._GAME_TYPE_OVERRIDE_CONF:
            canon_id, conf = gt_prior
            mapped = self._CANON_TO_GAME_TYPE.get(canon_id)
            if mapped is not None:
                game_type = mapped
                hypothesis = (
                    f"Human prior ({conf:.0%}) classifies this as "
                    f"'{canon_id}'."
                )

        if game_type is None:
            game_type, hypothesis = self._classify_game(
                obs, memory, has_player, move_actions, n_objects,
                effective_clicks, has_game_overs, has_interact,
                available_actions=available_actions,
            )

        # ── Build subgoals based on game type ─────────────────────
        if game_type == "navigation":
            overarching = "Navigate the player to the goal location(s)"
            subgoals = self._navigation_subgoals(
                obs, memory, move_actions, sg_id
            )
        elif game_type == "collection":
            overarching = "Collect all target items on the grid"
            subgoals = self._collection_subgoals(
                obs, memory, move_actions, has_interact, sg_id
            )
        elif game_type == "click_puzzle":
            overarching = "Click on the correct objects to solve the puzzle"
            subgoals = self._click_puzzle_subgoals(
                obs, memory, effective_clicks, sg_id, revision=revision
            )
        elif game_type == "sequence_puzzle":
            overarching = "Find and execute the correct action sequence"
            subgoals = self._sequence_puzzle_subgoals(
                obs, memory, sg_id
            )
        elif game_type == "push_puzzle":
            overarching = "Push objects to their target positions"
            subgoals = self._push_puzzle_subgoals(
                obs, memory, move_actions, has_interact, sg_id
            )
        else:
            overarching = "Explore and discover the win condition"
            subgoals = self._unknown_game_subgoals(
                obs, memory, move_actions, effective_clicks, sg_id
            )

        # If previous plan failed, add an exploration subgoal first
        if previous_goal and previous_goal.revision > 0:
            explore_sg = SubGoal(
                id=0,
                description="Re-explore: try untested actions and positions",
                success_hint="Discover new grid states or action effects",
                priority=-1,
                max_actions=15,
            )
            for sg in subgoals:
                sg.id += 1
            subgoals.insert(0, explore_sg)

        # ── Human-prior overrides on the goal NL strings ──────────
        # Replace the default `overarching` with the human's stated
        # objective when available, and append mechanic hints to the
        # hypothesis so downstream consumers (LLM prompts, logs, strategy
        # generator) see them.
        obj_prior = human_priors.get("objective")
        if obj_prior:
            overarching = f"{obj_prior[0]} (human, {obj_prior[1]:.0%})"
        mech_hints = human_priors.get("mechanics") or []
        obs_hints = human_priors.get("observations") or []
        extra_hyp_bits: List[str] = []
        for m, c in mech_hints[:3]:
            extra_hyp_bits.append(f"mechanic[{c:.0%}]: {m}")
        for o, c in obs_hints[:2]:
            extra_hyp_bits.append(f"human[{c:.0%}]: {o}")
        if extra_hyp_bits:
            hypothesis = (hypothesis or "") + "  ||  " + " ; ".join(extra_hyp_bits)

        goal = GameGoal(
            overarching_goal=overarching,
            hypothesis=hypothesis,
            subgoals=subgoals,
            confidence=self._estimate_confidence(obs, memory, game_type),
            revision=revision,
        )

        logger.info(
            f"Goal decomposed (type={game_type}, rev={revision}): "
            f"{overarching} → {len(subgoals)} subgoals"
        )
        return goal

    # ------------------------------------------------------------------
    # Game classification heuristics
    # ------------------------------------------------------------------
    def _classify_game(
        self,
        obs: GameObservation,
        memory: GameMemory,
        has_player: bool,
        move_actions: List[str],
        n_objects: int,
        effective_clicks: set,
        has_game_overs: bool,
        has_interact: bool,
        available_actions: Optional[List[str]] = None,
    ) -> tuple[str, str]:
        """Classify the game type using available_actions as the primary signal."""

        aa = set(available_actions or [])
        has_movement_actions = bool(aa & {"ACTION1", "ACTION2", "ACTION3", "ACTION4"})
        has_click = "ACTION6" in aa
        has_undo = "ACTION7" in aa

        # ── Click-only games (no movement actions) ──
        if has_click and not has_movement_actions:
            return (
                "click_puzzle",
                "Click-only game (no movement actions). "
                "The goal is to click on the right objects/cells "
                "in the correct order or pattern.",
            )

        # ── Mixed click+movement games ──
        if has_click and has_movement_actions:
            # If clicking has proven effective, prioritise clicks
            if effective_clicks:
                return (
                    "click_puzzle",
                    "Clicking changes the grid. "
                    "The goal may involve clicking specific objects.",
                )

        # ── Collection game: player + many same-colored objects ──
        if has_player and len(move_actions) >= 2 and n_objects >= 4:
            obj_values = [o["value"] for o in (obs.objects or []) if not o.get("is_player")]
            from collections import Counter
            counts = Counter(obj_values)
            most_common_val, most_common_count = counts.most_common(1)[0] if counts else (0, 0)
            if most_common_count >= 3:
                return (
                    "collection",
                    f"Multiple objects of color {most_common_val} "
                    f"({most_common_count} instances). "
                    f"The goal is likely to visit/collect them.",
                )

        # ── Push puzzle: player + movable objects + interact ──
        if has_player and has_interact and n_objects >= 2:
            return (
                "push_puzzle",
                "Player can interact with objects. "
                "Likely need to push/move objects to target positions.",
            )

        # ── Navigation game: player moves ──
        if has_player and len(move_actions) >= 2:
            return (
                "navigation",
                "Player can move in multiple directions. "
                "The goal is likely to reach a specific location or object.",
            )

        # ── Sequence puzzle: no clear player, actions change grid ──
        change_actions = [
            name for name, prof in memory.action_profiles.items()
            if prof.change_rate > 0.3 and name not in ("RESET",)
        ]
        if len(change_actions) >= 2:
            return (
                "sequence_puzzle",
                "Multiple actions change the grid. "
                "The goal may be to find the right action sequence.",
            )

        return (
            "unknown",
            "Game mechanics unclear. Need more exploration to determine win condition.",
        )

    # ------------------------------------------------------------------
    # Subgoal builders per game type
    # ------------------------------------------------------------------
    def _navigation_subgoals(
        self,
        obs: GameObservation,
        memory: GameMemory,
        move_actions: List[str],
        sg_id: int,
    ) -> List[SubGoal]:
        subgoals = []
        objects = [
            o for o in (obs.objects or [])
            if not o.get("is_player") and o.get("value", 0) != 0
        ]

        if not objects:
            subgoals.append(SubGoal(
                id=sg_id, priority=0, max_actions=40,
                description="Explore the grid to find goal location",
                success_hint="Discover a new region or object",
            ))
            return subgoals

        # Sort objects by distance from player (closest first → try reaching each)
        if obs.player_info:
            py, px = obs.player_info["y"], obs.player_info["x"]
            objects.sort(key=lambda o: (o["center_y"] - py) ** 2 + (o["center_x"] - px) ** 2)

        # Subgoal: navigate to each distinct object cluster
        visited_values = set()
        for obj in objects:
            val = obj["value"]
            if val in visited_values:
                continue
            visited_values.add(val)
            subgoals.append(SubGoal(
                id=sg_id,
                priority=sg_id,
                max_actions=25,
                description=f"Navigate to color-{val} object at ({obj['center_y']:.0f}, {obj['center_x']:.0f})",
                success_hint=f"Player reaches within 2 cells of ({obj['center_y']:.0f}, {obj['center_x']:.0f})",
                metadata={"target_value": val, "target_y": obj["center_y"], "target_x": obj["center_x"]},
            ))
            sg_id += 1

        # Final subgoal: interact/click at goal
        subgoals.append(SubGoal(
            id=sg_id, priority=sg_id, max_actions=10,
            description="Try interacting at the goal location (ACTION5/ACTION6)",
            success_hint="Level completion or grid state change",
        ))
        return subgoals

    def _collection_subgoals(
        self,
        obs: GameObservation,
        memory: GameMemory,
        move_actions: List[str],
        has_interact: bool,
        sg_id: int,
    ) -> List[SubGoal]:
        subgoals = []
        objects = [
            o for o in (obs.objects or [])
            if not o.get("is_player") and o.get("value", 0) != 0
        ]
        from collections import Counter
        counts = Counter(o["value"] for o in objects)
        target_val = counts.most_common(1)[0][0] if counts else None

        targets = [o for o in objects if o.get("value") == target_val]
        if obs.player_info:
            py, px = obs.player_info["y"], obs.player_info["x"]
            targets.sort(key=lambda o: (o["center_y"] - py) ** 2 + (o["center_x"] - px) ** 2)

        for i, t in enumerate(targets):
            subgoals.append(SubGoal(
                id=sg_id, priority=sg_id, max_actions=20,
                description=f"Reach item {i+1}/{len(targets)}: color-{t['value']} at ({t['center_y']:.0f}, {t['center_x']:.0f})",
                success_hint=f"Player is adjacent to ({t['center_y']:.0f}, {t['center_x']:.0f}) or item disappears",
                metadata={"target_y": t["center_y"], "target_x": t["center_x"], "target_value": t["value"]},
            ))
            sg_id += 1
            if has_interact:
                subgoals.append(SubGoal(
                    id=sg_id, priority=sg_id, max_actions=5,
                    description=f"Interact with item {i+1} (ACTION5)",
                    success_hint="Item collected or grid changes",
                ))
                sg_id += 1
        return subgoals

    def _click_puzzle_subgoals(
        self,
        obs: GameObservation,
        memory: GameMemory,
        effective_clicks: set,
        sg_id: int,
        revision: int = 0,
    ) -> List[SubGoal]:
        import numpy as np
        subgoals = []
        objects = [
            o for o in (obs.objects or [])
            if o.get("value", 0) != 0
        ]

        # SG0: Click on each object (prioritise ones of effective values)
        click_targets_all = []
        if objects:
            # Prioritise effective click values, then all objects
            if effective_clicks:
                prio = [o for o in objects if o["value"] in effective_clicks]
                rest = [o for o in objects if o["value"] not in effective_clicks]
                ordered = prio + rest
            else:
                ordered = objects

            # On revision > 0, try different orderings
            if revision == 1:
                ordered = list(reversed(ordered))  # reverse order
            elif revision >= 2:
                # Shuffle based on revision to try different orderings
                import random
                rng = random.Random(revision)
                ordered = list(ordered)
                rng.shuffle(ordered)

            for obj in ordered:
                click_targets_all.append(
                    {"x": int(obj["center_x"]), "y": int(obj["center_y"])}
                )

        if click_targets_all:
            subgoals.append(SubGoal(
                id=sg_id, priority=sg_id, max_actions=min(len(click_targets_all) * 2, 40),
                description=f"Click on all {len(click_targets_all)} detected objects",
                success_hint="Grid state changes after clicks",
                metadata={
                    "action": "ACTION6",
                    "click_targets": click_targets_all[:20],
                },
            ))
            sg_id += 1

        # SG1: Systematic grid scan — click non-zero cells
        grid = obs.raw_grid
        if grid is not None:
            nz = list(zip(*np.nonzero(grid)))
            if nz:
                step = max(1, len(nz) // 30)
                # On revision>0, offset the sampling to cover different cells
                offset = (revision * step // 2) % step if revision > 0 else 0
                scan_targets = [
                    {"x": int(nz[i][1]), "y": int(nz[i][0])}
                    for i in range(offset, len(nz), step)
                ]
                subgoals.append(SubGoal(
                    id=sg_id, priority=sg_id, max_actions=min(len(scan_targets), 40),
                    description=f"Click on {len(scan_targets)} non-zero cells across the grid",
                    success_hint="Level completion or significant grid changes",
                    metadata={
                        "action": "ACTION6",
                        "click_targets": scan_targets,
                    },
                ))
                sg_id += 1

        # SG2: Click same cells again (some games need toggle/repeat)
        if click_targets_all:
            subgoals.append(SubGoal(
                id=sg_id, priority=sg_id, max_actions=min(len(click_targets_all) * 2, 30),
                description="Re-click objects in reverse order (toggle/repeat pattern)",
                success_hint="Level completion",
                metadata={
                    "action": "ACTION6",
                    "click_targets": list(reversed(click_targets_all[:15])),
                },
            ))

        return subgoals

    def _sequence_puzzle_subgoals(
        self,
        obs: GameObservation,
        memory: GameMemory,
        sg_id: int,
    ) -> List[SubGoal]:
        subgoals = []
        # Find which actions change the grid
        change_actions = [
            name for name, prof in memory.action_profiles.items()
            if prof.change_rate > 0.2 and name not in ("RESET",)
        ]

        subgoals.append(SubGoal(
            id=sg_id, priority=0, max_actions=20,
            description=f"Try combinations of effective actions: {change_actions}",
            success_hint="New grid state or level progress",
        ))
        sg_id += 1

        subgoals.append(SubGoal(
            id=sg_id, priority=1, max_actions=20,
            description="Try each effective action 3x in a row",
            success_hint="Cumulative grid change or level progress",
        ))
        sg_id += 1

        subgoals.append(SubGoal(
            id=sg_id, priority=2, max_actions=15,
            description="Undo with ACTION7/RESET and try reverse order",
            success_hint="Different outcome from previous attempts",
        ))
        return subgoals

    def _push_puzzle_subgoals(
        self,
        obs: GameObservation,
        memory: GameMemory,
        move_actions: List[str],
        has_interact: bool,
        sg_id: int,
    ) -> List[SubGoal]:
        subgoals = []
        objects = [
            o for o in (obs.objects or [])
            if not o.get("is_player") and o.get("value", 0) != 0
        ]

        for i, obj in enumerate(objects[:5]):
            subgoals.append(SubGoal(
                id=sg_id, priority=sg_id, max_actions=20,
                description=f"Navigate to object {i+1} (color-{obj['value']}) and interact",
                success_hint="Object moves or grid changes after interaction",
                metadata={"target_y": obj["center_y"], "target_x": obj["center_x"]},
            ))
            sg_id += 1
        return subgoals

    def _unknown_game_subgoals(
        self,
        obs: GameObservation,
        memory: GameMemory,
        move_actions: List[str],
        effective_clicks: set,
        sg_id: int,
    ) -> List[SubGoal]:
        subgoals = []

        # First: thorough exploration
        subgoals.append(SubGoal(
            id=sg_id, priority=0, max_actions=20,
            description="Try every available action at least 2x and observe effects",
            success_hint="All actions have been tried; effects recorded in memory",
        ))
        sg_id += 1

        # If movement exists, explore the grid
        if move_actions:
            subgoals.append(SubGoal(
                id=sg_id, priority=1, max_actions=25,
                description="Explore the full grid using movement actions",
                success_hint="Visit at least 50% of reachable grid positions",
            ))
            sg_id += 1

        # If clicks work, try systematic clicking
        if effective_clicks:
            subgoals.append(SubGoal(
                id=sg_id, priority=2, max_actions=15,
                description="Click on every distinct object type",
                success_hint="Discover clickable patterns",
            ))
            sg_id += 1

        # Try every action at every position
        subgoals.append(SubGoal(
            id=sg_id, priority=3, max_actions=30,
            description="Try non-movement actions at different grid positions",
            success_hint="Level completion or significant grid change",
        ))
        return subgoals

    # ------------------------------------------------------------------
    # Goal bank generation (multiple hypotheses)
    # ------------------------------------------------------------------
    def generate_goal_bank(
        self,
        observation: GameObservation,
        memory: GameMemory,
        previous_outcomes: Optional[List[Dict]] = None,
    ) -> List[GameObjective]:
        """Generate a bank of 3-6 plausible game objectives.

        Each objective carries measurable progress signals, not just
        a name.  The LLM is asked: *what would observable changes in
        the environment indicate partial progress and what final
        condition would indicate success?*

        Falls back to template-based heuristics if no LLM.
        """
        # Phase 2a: if a TaskProgram is attached, its SubgoalTests are
        # the canonical hypothesis bank. We still append a small number
        # of template-derived goals as fallback candidates so that if
        # the program's goals all fail, the goal-pursuit layer has
        # alternatives to try.
        if self._task_program is not None:
            bank = self._goal_bank_from_program(self._task_program, observation, memory)
            if bank:
                # Append 2-3 top template goals as lower-confidence
                # fallbacks. Prefixes the bank so program goals lead.
                templates = self._goal_bank_from_templates(observation, memory)
                # Reduce template confidence so program goals win
                # confidence-sorted selection.
                for t in templates[:3]:
                    t.confidence = min(t.confidence, 0.45)
                    bank.append(t)
                return bank

        if self.use_llm and self._model is not None:
            bank = self._goal_bank_with_llm(observation, memory, previous_outcomes)
            if bank:
                # LLM gives good hypotheses but often omits progress signals;
                # enrich with template-derived signals so measurement works.
                templates = self._goal_bank_from_templates(observation, memory)
                bank = self._enrich_llm_goals(bank, templates)
                return bank

        return self._goal_bank_from_templates(observation, memory)

    # ── TaskProgram-based goal bank (Phase 2a) ───────────────────
    # expected_signal → list of ProgressSignal templates so the goal-
    # pursuit layer has something concrete to measure.
    _SIGNAL_TO_PROGRESS_SIGNALS: Dict[str, List[tuple]] = {
        "grid_change": [
            ("grid_cell_change", "Grid state changes after actions", "increase", 1.5),
            ("unique_states_increasing", "New grid states discovered", "increase", 1.0),
        ],
        "object_moved": [
            ("object_positions_changed", "Non-player objects change position", "increase", 2.0),
            ("grid_cell_change", "Grid changes near object locations", "increase", 1.0),
        ],
        "level_advance": [
            ("level_completed", "Level counter increments", "increase", 3.0),
        ],
        "role_switch": [
            ("player_position_changed", "Player identity may have switched", "increase", 1.0),
            ("grid_cell_change", "Grid signals a control handover", "increase", 1.0),
        ],
        "overlap_achieved": [
            ("player_distance_to_target_decreasing", "Player closes distance to key object", "decrease", 2.0),
            ("player_position_changed", "Player advances", "increase", 1.0),
        ],
        "click_triggered_change": [
            ("grid_cell_change", "Grid state changes after clicks", "increase", 1.5),
            ("unique_states_increasing", "New grid states via clicking", "increase", 1.0),
        ],
        "click_absorbed_no_change": [
            ("grid_cell_change", "Clicks register but state is stable", "increase", 0.5),
        ],
        "object_disappeared": [
            ("grid_cell_change", "Target items disappear from the grid", "increase", 2.0),
            ("player_position_changed", "Player visits item positions", "increase", 1.0),
        ],
        "new_region_revealed": [
            ("unique_states_increasing", "New regions of the grid are visited", "increase", 1.5),
            ("player_position_changed", "Player explores further", "increase", 1.0),
        ],
    }

    def _goal_bank_from_program(
        self,
        program: Any,
        observation: GameObservation,
        memory: GameMemory,
    ) -> List[GameObjective]:
        """Turn a TaskProgram's SubgoalTests into a goal-pursuit bank.

        Each `SubgoalTest` becomes one `GameObjective` with:
          - id           = test.id (or `human_prior_<n>` if missing)
          - description  = test.description
          - success_cond = test.verification
          - progress     = templates chosen by test.expected_signal
          - confidence   = decreasing with rank (top cheapest = highest)
        """
        tests = list(getattr(program, "subgoal_tests", []) or [])
        if not tests:
            return []

        anti = list(getattr(program, "anti_patterns", []) or [])
        # Truncate anti_signals so they fit the GameObjective schema.
        anti_sigs_global = [a[:80] for a in anti[:3]]
        anti_sigs_global.append("game_over")

        prog_conf = float(getattr(program, "confidence", 0.5) or 0.5)
        # Rank confidence: first subgoal gets program_conf, last gets
        # 0.4 min. Cheaper-to-test comes first by compiler convention.
        n = len(tests)
        def _conf_at(i: int) -> float:
            top = min(0.9, 0.4 + 0.5 * prog_conf)
            # Linear decay to 0.4 at i = n-1
            if n <= 1:
                return top
            bottom = max(0.4, top - 0.3)
            frac = i / (n - 1)
            return top * (1 - frac) + bottom * frac

        bank: List[GameObjective] = []
        for i, test in enumerate(tests):
            raw_id = str(getattr(test, "id", "") or f"human_prior_{i}")
            gid = f"human_prior::{raw_id}"[:80]
            desc = str(getattr(test, "description", "") or "").strip() or raw_id
            verify = str(getattr(test, "verification", "") or "").strip() or "observable signal"
            sig = str(getattr(test, "expected_signal", "other") or "other")

            signal_specs = self._SIGNAL_TO_PROGRESS_SIGNALS.get(sig, [
                ("grid_cell_change", "Grid changes indicate progress", "increase", 1.5),
                ("unique_states_increasing", "New grid states discovered", "increase", 1.0),
            ])
            progress_signals = [
                ProgressSignal(name=s[0], description=s[1], direction=s[2], weight=s[3])
                for s in signal_specs
            ]

            bank.append(GameObjective(
                id=gid,
                description=f"[human] {desc}"[:200],
                success_condition=f"[human-verified] {verify}"[:200],
                progress_signals=progress_signals,
                anti_signals=list(anti_sigs_global),
                confidence=_conf_at(i),
            ))

        return bank

    # ── LLM + template merge ──────────────────────────────────────

    def _enrich_llm_goals(
        self,
        llm_goals: List[GameObjective],
        template_goals: List[GameObjective],
    ) -> List[GameObjective]:
        """Back-fill progress signals on LLM goals that lack them.

        Strategy:
        1. If an LLM goal already has ≥2 signals, keep them.
        2. Otherwise, find the best-matching template goal (by keyword
           overlap in id + description) and copy its signals.
        3. If no good match, assign generic grid-change signals.
        """
        def _keywords(g: GameObjective) -> set:
            text = f"{g.id} {g.description}".lower()
            return set(text.split())

        template_kws = [(_keywords(t), t) for t in template_goals]

        enriched = []
        for g in llm_goals:
            if len(g.progress_signals) >= 2:
                enriched.append(g)
                continue

            # Find best-matching template by keyword overlap
            g_kws = _keywords(g)
            best_match = None
            best_score = 0
            for t_kws, t in template_kws:
                overlap = len(g_kws & t_kws)
                if overlap > best_score:
                    best_score = overlap
                    best_match = t

            if best_match and best_score >= 2:
                # Copy signals from template match
                g.progress_signals = list(best_match.progress_signals)
                if not g.anti_signals and best_match.anti_signals:
                    g.anti_signals = list(best_match.anti_signals)
            else:
                # Generic fallback signals
                g.progress_signals = [
                    ProgressSignal("grid_cell_change",
                                   "Grid cells change after actions", "increase"),
                    ProgressSignal("unique_states_increasing",
                                   "New grid states discovered", "increase"),
                    ProgressSignal("player_position_changed",
                                   "Player moves to new positions", "increase"),
                ]

            enriched.append(g)

        return enriched

    # ── LLM-based goal bank ──────────────────────────────────────

    _GOAL_BANK_SYSTEM = """\
You are an expert game analyst for interactive grid-based games.
Given observations about game mechanics (discovered via exploration),
produce 3-6 **plausible objective hypotheses** for the game.

For EACH hypothesis, output:
- "id": short identifier (e.g. "navigate_exit")
- "description": what the player needs to do
- "success_condition": what observable state means the objective is achieved
- "confidence": float 0-1 (how likely this is the real objective)
- "progress_signals": list of {"name": ..., "description": ..., "direction": "increase"/"decrease"}
  These are MEASURABLE indicators of partial progress.
- "anti_signals": list of strings describing what indicates WRONG direction

Critical: progress_signals must be things the agent can actually observe
from the grid state (cell counts, object positions, new regions, etc.).

IMPORTANT: Base your hypotheses on the ACTION EFFECT PROFILES. If most
actions cause player movement, prioritise navigation/collection hypotheses.
Only hypothesise click-based objectives if clicking (ACTION6) is the main
way to change the grid.

Respond with a JSON array of hypothesis objects.
"""

    def _goal_bank_with_llm(
        self,
        observation: GameObservation,
        memory: GameMemory,
        previous_outcomes: Optional[List[Dict]],
    ) -> List[GameObjective]:
        prompt = observation.to_prompt()

        # Surface human priors (if any) at the top of the prompt so the LLM
        # can prefer hypotheses aligned with what a human observed.
        priors_block = self._human_priors_prompt(self._extract_human_priors(memory))
        if priors_block:
            prompt = priors_block + "\n" + prompt

        # Inject detailed action effect profiles so the LLM understands
        # which actions cause movement vs interaction vs danger
        action_details = []
        for name, prof in memory.action_profiles.items():
            if name == "RESET" or prof.times_tried == 0:
                continue
            cr = prof.change_rate
            dr = prof.times_caused_game_over / max(prof.times_tried, 1)
            mv = prof.times_moved_player / max(prof.times_tried, 1)
            line = f"{name}: tried={prof.times_tried}, grid_change={cr:.0%}, player_move={mv:.0%}, death={dr:.0%}"
            if prof.avg_displacement != (0.0, 0.0):
                dy, dx = prof.avg_displacement
                line += f", avg_move=({dy:+.1f},{dx:+.1f})"
            action_details.append(line)
        if action_details:
            prompt += "\n\n### Action Effect Profiles (from exploration)\n"
            prompt += "\n".join(action_details)
            # Add interpretive hint
            movement_acts = [n for n, p in memory.action_profiles.items()
                            if p.times_moved_player > 0 and n != "RESET"]
            if movement_acts:
                prompt += f"\n\nMovement actions: {', '.join(movement_acts)}"
                prompt += "\nThis game likely involves NAVIGATION, not clicking."
            elif memory.get_effective_click_values():
                prompt += f"\nEffective click targets: color values {memory.get_effective_click_values()}"
                prompt += "\nThis game may involve clicking on specific objects."
        if previous_outcomes:
            prompt += "\n\n## Previous strategy outcomes:\n"
            for po in previous_outcomes[-5:]:
                prompt += f"  - {po.get('strategy', '?')}: progress={po.get('progress', 0):.2f}, status={po.get('status', '?')}\n"
            prompt += "\nRevise your hypotheses in light of this evidence.\n"
        prompt += "\n\nGenerate 3-6 plausible game objectives as JSON:"

        if len(prompt) > 2000:
            prompt = prompt[:2000] + "\n...(truncated)"

        messages = [
            {"role": "system", "content": self._GOAL_BANK_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            import torch
            inputs = self._tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True
            ).to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=200, temperature=0.7,
                    do_sample=True, top_p=0.9,
                )
            raw = self._tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return self._parse_goal_bank_response(raw)
        except Exception as e:
            logger.warning(f"LLM goal bank generation failed: {e}")
            return []

    def _parse_goal_bank_response(self, raw: str) -> List[GameObjective]:
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

        objectives = []
        for item in data:
            signals = []
            for s in item.get("progress_signals", []):
                if isinstance(s, dict):
                    signals.append(ProgressSignal(
                        name=s.get("name", "unknown"),
                        description=s.get("description", ""),
                        direction=s.get("direction", "increase"),
                    ))
            objectives.append(GameObjective(
                id=item.get("id", f"obj_{len(objectives)}"),
                description=item.get("description", "Unknown"),
                success_condition=item.get("success_condition", "Level completion"),
                progress_signals=signals,
                anti_signals=item.get("anti_signals", []),
                confidence=float(item.get("confidence", 0.5)),
            ))
        return objectives

    # ── Template-based goal bank ─────────────────────────────────
    def _goal_bank_from_templates(
        self,
        obs: GameObservation,
        memory: GameMemory,
    ) -> List[GameObjective]:
        """Generate objective hypotheses from heuristics.

        Uses rich game state: player position, object locations/colors,
        grid dimensions, action effect profiles, click data, death zones.
        """
        from collections import Counter

        objectives: List[GameObjective] = []

        # ── Extract rich context ─────────────────────────────────
        has_player = obs.player_info is not None
        player_pos = None
        player_val = None
        if has_player:
            player_pos = (
                int(obs.player_info.get("y", 0)),
                int(obs.player_info.get("x", 0)),
            )
            player_val = obs.player_info.get("value", -1)

        move_actions = memory.get_movement_actions()
        n_objects = len(obs.objects) if obs.objects else 0
        effective_clicks = memory.get_effective_click_values()
        has_game_overs = memory.total_game_overs > 0

        available_actions = list(memory.action_profiles.keys()) or [
            f"ACTION{i}" for i in range(1, 8)
        ]
        aa = set(available_actions)
        has_movement = bool(aa & {"ACTION1", "ACTION2", "ACTION3", "ACTION4"})
        has_click = "ACTION6" in aa

        # Non-player objects sorted by size (largest first — likely targets)
        non_player_objs = sorted(
            [o for o in (obs.objects or []) if not o.get("is_player")],
            key=lambda o: -o.get("size", 0),
        )
        obj_values = [o["value"] for o in non_player_objs]
        value_counts = Counter(obj_values)
        unique_colors = sorted(value_counts.keys())

        # Grid dimensions
        h, w = obs.raw_grid.shape[:2] if obs.raw_grid is not None else (16, 16)

        # Action effect profiles
        high_change_actions = [
            name for name, prof in memory.action_profiles.items()
            if prof.change_rate > 0.3 and name not in ("RESET",)
        ]
        low_change_actions = [
            name for name, prof in memory.action_profiles.items()
            if 0 < prof.change_rate <= 0.1 and name not in ("RESET",)
        ]
        deadly_actions = [
            name for name, prof in memory.action_profiles.items()
            if prof.times_tried > 0
            and (prof.times_caused_game_over / prof.times_tried) > 0.2
            and name not in ("RESET",)
        ]
        semantics = memory.infer_action_semantics()

        # ── 1. Navigate to specific target ──
        if has_player and has_movement and non_player_objs:
            # Find the most prominent non-player object as target
            target = non_player_objs[0]
            ty, tx = int(target["center_y"]), int(target["center_x"])
            tval = target["value"]
            if player_pos:
                dist = abs(player_pos[0] - ty) + abs(player_pos[1] - tx)
                objectives.append(GameObjective(
                    id=f"navigate_to_color{tval}",
                    description=(
                        f"Navigate player from ({player_pos[0]},{player_pos[1]}) "
                        f"to color-{tval} object at ({ty},{tx}), distance={dist}"
                    ),
                    success_condition=f"Player overlaps or is adjacent to the color-{tval} region",
                    progress_signals=[
                        ProgressSignal("player_distance_to_target_decreasing",
                                       f"Manhattan distance to ({ty},{tx}) decreases",
                                       "decrease", weight=2.0),
                        ProgressSignal("player_position_changed",
                                       "Player position changes each step", "increase"),
                    ],
                    anti_signals=[f"player_moves_away_from_({ty},{tx})", "game_over"],
                    confidence=0.75 if len(move_actions) >= 2 else 0.45,
                ))

            # If there's a second prominent object, add a second nav hypothesis
            if len(non_player_objs) >= 2:
                t2 = non_player_objs[1]
                t2y, t2x = int(t2["center_y"]), int(t2["center_x"])
                t2val = t2["value"]
                if t2val != tval and player_pos:
                    d2 = abs(player_pos[0] - t2y) + abs(player_pos[1] - t2x)
                    objectives.append(GameObjective(
                        id=f"navigate_to_color{t2val}",
                        description=(
                            f"Navigate player to color-{t2val} object "
                            f"at ({t2y},{t2x}), distance={d2}"
                        ),
                        success_condition=f"Player reaches the color-{t2val} region",
                        progress_signals=[
                            ProgressSignal("player_distance_to_target_decreasing",
                                           f"Manhattan distance to ({t2y},{t2x}) decreases",
                                           "decrease", weight=2.0),
                            ProgressSignal("player_position_changed",
                                           "Player position changes each step", "increase"),
                        ],
                        confidence=0.55,
                    ))

        # ── 2. Collection: gather repeated objects ──
        if has_player and value_counts:
            for val, cnt in value_counts.most_common(3):
                if cnt < 2:
                    break
                positions = [(int(o["center_y"]), int(o["center_x"]))
                             for o in non_player_objs if o["value"] == val]
                pos_str = ", ".join(f"({y},{x})" for y, x in positions[:4])
                objectives.append(GameObjective(
                    id=f"collect_color{val}",
                    description=(
                        f"Collect all {cnt} color-{val} items at {pos_str}"
                    ),
                    success_condition=f"All {cnt} color-{val} cells disappear from the grid",
                    progress_signals=[
                        ProgressSignal(f"color{val}_count_decreasing",
                                       f"Number of color-{val} cells in grid decreases",
                                       "decrease", weight=2.5),
                        ProgressSignal("player_visits_item_positions",
                                       "Player moves to cells where items exist", "increase"),
                        ProgressSignal("grid_cell_change",
                                       "Grid cells change value (items consumed)", "increase"),
                    ],
                    anti_signals=[f"color{val}_count_increases"],
                    confidence=0.65 if cnt >= 3 else 0.45,
                ))
                if len(objectives) >= 5:
                    break

        # ── 3. Click puzzle ──
        if has_click:
            click_vals = sorted(effective_clicks) if effective_clicks else []
            conf = 0.8 if not has_movement else (0.65 if click_vals else 0.35)
            desc = "Click on specific cells to solve the puzzle"
            if click_vals:
                desc = f"Click on cells with values {click_vals} (known effective clicks)"
            objectives.append(GameObjective(
                id="click_puzzle",
                description=desc,
                success_condition="Correct click pattern triggers level completion",
                progress_signals=[
                    ProgressSignal("grid_cell_change",
                                   "Grid state changes after clicks", "increase", weight=1.5),
                    ProgressSignal("unique_states_increasing",
                                   "New grid states discovered via clicking", "increase"),
                ],
                anti_signals=["clicks_cause_game_over", "no_change_after_click"],
                confidence=conf,
            ))

        # ── 4. Sequence / action combo puzzle ──
        if len(high_change_actions) >= 2:
            act_str = ", ".join(high_change_actions[:4])
            objectives.append(GameObjective(
                id="sequence_puzzle",
                description=f"Execute a specific sequence of [{act_str}] to transform the grid",
                success_condition="Grid reaches a target configuration via correct action sequence",
                progress_signals=[
                    ProgressSignal("cumulative_grid_change",
                                   "Grid accumulates progressive change across actions",
                                   "increase", weight=2.0),
                    ProgressSignal("unique_states_increasing",
                                   "New states discovered via action combos", "increase"),
                    ProgressSignal("grid_cell_change",
                                   "Each action produces meaningful grid change", "increase"),
                ],
                anti_signals=["grid_resets", "same_state_cycling"],
                confidence=0.50,
            ))

        # ── 5. Push / interact puzzle ──
        has_interact = any(
            v in ("interact", "activate", "toggle") for v in semantics.values()
        )
        if has_player and has_interact and n_objects >= 2:
            objectives.append(GameObjective(
                id="push_puzzle",
                description="Push or manipulate objects to their target positions",
                success_condition="All movable objects reach target cells",
                progress_signals=[
                    ProgressSignal("object_positions_changed",
                                   "Non-player objects change position", "increase", weight=2.0),
                    ProgressSignal("grid_cell_change",
                                   "Grid changes near object locations", "increase"),
                ],
                anti_signals=["objects_stuck", "pushing_causes_game_over"],
                confidence=0.50,
            ))

        # ── 6. Avoidance navigation (if deaths occur) ──
        if has_game_overs and has_player:
            dead_str = ", ".join(deadly_actions[:3]) if deadly_actions else "unknown"
            objectives.append(GameObjective(
                id="navigate_avoid_hazards",
                description=f"Reach the goal while avoiding death (dangerous: {dead_str})",
                success_condition="Player reaches exit without game-over",
                progress_signals=[
                    ProgressSignal("player_position_changed",
                                   "Player advances despite hazards", "increase", weight=1.5),
                    ProgressSignal("unique_states_increasing",
                                   "New safe regions discovered", "increase"),
                ],
                anti_signals=["immediate_game_over", "stuck_in_safe_area"],
                confidence=0.55 if deadly_actions else 0.40,
            ))

        # ── 6b. Human prior: always surface if a human objective exists ──
        # This guarantees the human hypothesis reaches the goal bank even
        # when template heuristics would not have generated it, so the
        # downstream goal_pursuit loop can actually try it.
        human_priors = self._extract_human_priors(memory)
        gt_prior = human_priors.get("game_type")
        obj_prior = human_priors.get("objective")
        if gt_prior or obj_prior:
            gt_id, gt_conf = (gt_prior if gt_prior else ("unknown", 0.5))
            obj_text = obj_prior[0] if obj_prior else f"Pursue human-hypothesised goal ({gt_id})"
            obj_conf = obj_prior[1] if obj_prior else gt_conf
            human_conf = min(0.9, max(gt_conf, obj_conf))
            # Pick progress signals that match the canonical goal bucket so
            # the goal_pursuit layer has something to measure.
            _signal_templates: Dict[str, List[ProgressSignal]] = {
                "navigate_exit": [
                    ProgressSignal("player_position_changed",
                                   "Player advances toward new regions", "increase", weight=1.5),
                    ProgressSignal("unique_states_increasing",
                                   "New regions of the grid are visited", "increase"),
                ],
                "navigate_puzzle": [
                    ProgressSignal("player_position_changed",
                                   "Player advances toward new regions", "increase", weight=1.5),
                    ProgressSignal("unique_states_increasing",
                                   "New regions of the grid are visited", "increase"),
                ],
                "collection": [
                    ProgressSignal("grid_cell_change",
                                   "Target items disappear from the grid", "increase", weight=2.0),
                    ProgressSignal("player_position_changed",
                                   "Player visits item positions", "increase"),
                ],
                "click_puzzle": [
                    ProgressSignal("grid_cell_change",
                                   "Grid state changes after clicks", "increase", weight=1.5),
                    ProgressSignal("unique_states_increasing",
                                   "New grid states via clicking", "increase"),
                ],
                "sequence_puzzle": [
                    ProgressSignal("cumulative_grid_change",
                                   "Grid accumulates progressive change", "increase", weight=2.0),
                    ProgressSignal("unique_states_increasing",
                                   "New states discovered via action combos", "increase"),
                ],
                "push_puzzle": [
                    ProgressSignal("object_positions_changed",
                                   "Non-player objects change position", "increase", weight=2.0),
                    ProgressSignal("grid_cell_change",
                                   "Grid changes near object locations", "increase"),
                ],
                "transform_puzzle": [
                    ProgressSignal("cumulative_grid_change",
                                   "Grid progressively transforms", "increase", weight=2.0),
                    ProgressSignal("grid_cell_change",
                                   "Each action produces a targeted change", "increase"),
                ],
            }
            sigs = _signal_templates.get(gt_id, [
                ProgressSignal("grid_cell_change",
                               "Grid changes indicate progress", "increase", weight=1.5),
                ProgressSignal("unique_states_increasing",
                               "New grid states discovered", "increase"),
            ])
            # Append a failure-pattern anti-signal from human-logged mistakes.
            anti = ["game_over"]
            for obs_text, _ in human_priors.get("observations", [])[:2]:
                anti.append(f"human_flagged::{obs_text}"[:80])

            human_desc = f"HUMAN PRIOR: {obj_text}"
            if gt_prior:
                human_desc += f" (type={gt_id}, conf={gt_conf:.0%})"
            objectives.append(GameObjective(
                id=f"human_prior::{gt_id}",
                description=human_desc[:200],
                success_condition="Objective matches the human-stated win condition",
                progress_signals=sigs,
                anti_signals=anti,
                confidence=human_conf,
            ))

        # ── 7. Fallback: explore to discover mechanics ──
        n_tried = sum(1 for p in memory.action_profiles.values() if p.times_tried > 0)
        n_total = max(len(available_actions), 1)
        objectives.append(GameObjective(
            id="discover_mechanics",
            description=(
                f"Explore the {h}x{w} grid to discover win condition "
                f"({n_tried}/{n_total} actions tested, "
                f"{len(memory._visited_hashes)} states seen)"
            ),
            success_condition="Enough mechanics discovered to form a specific hypothesis",
            progress_signals=[
                ProgressSignal("unique_states_increasing",
                               "New grid states visited", "increase", weight=1.5),
                ProgressSignal("grid_cell_change",
                               "Actions produce observable effects", "increase"),
            ],
            anti_signals=["same_states_revisited"],
            confidence=0.25,
        ))

        # Sort by confidence descending, cap at 6
        objectives.sort(key=lambda o: -o.confidence)
        return objectives[:6]

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------
    def _estimate_confidence(
        self,
        obs: GameObservation,
        memory: GameMemory,
        game_type: str,
    ) -> float:
        """Estimate confidence in the goal decomposition."""
        base = 0.3
        # More exploration → more confidence
        base += min(memory.get_exploration_score() * 0.3, 0.3)
        # Known game type → more confidence
        if game_type != "unknown":
            base += 0.2
        # Player identified → more confidence
        if obs.player_info:
            base += 0.1
        return min(base, 0.95)
