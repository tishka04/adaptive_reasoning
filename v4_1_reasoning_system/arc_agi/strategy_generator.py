"""
Strategy Generator — produces candidate game strategies via LLM.

Adapts the v4_1 CandidateGenerator concept:
  - Instead of (mode, budget, strictness) reasoning candidates,
    generates (strategy_type, action_sequence, rationale) game strategies.
  - Uses an LLM to parse observations into hypotheses about game mechanics
    and generate strategies in natural language.
  - Falls back to template-based generation when no LLM is available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state_describer import GameObservation
from .llm_cache import get_shared_llm
from .goal_pursuit import GameObjective

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
class StrategyType(Enum):
    """High-level strategy categories for ARC-AGI-3 games."""
    NAVIGATE_TO_GOAL = "navigate_to_goal"
    COLLECT_ITEMS = "collect_items"
    SOLVE_PUZZLE = "solve_puzzle"
    CLICK_OBJECTS = "click_objects"
    SEQUENCE_ACTIONS = "sequence_actions"
    AVOID_HAZARDS = "avoid_hazards"
    EXPLORE_SYSTEMATICALLY = "explore_systematically"
    UNDO_AND_RETRY = "undo_and_retry"


@dataclass
class GameStrategy:
    """A candidate strategy for the current game state."""
    strategy_type: StrategyType
    description: str                            # NL description of what to do
    action_plan: List[str]                      # ordered action names to try
    rationale: str                              # why this strategy might work
    confidence: float = 0.5                     # generator's confidence [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_type": self.strategy_type.value,
            "description": self.description,
            "action_plan": self.action_plan,
            "rationale": self.rationale,
            "confidence": self.confidence,
            **self.metadata,
        }


# ------------------------------------------------------------------
# LLM-based strategy generation
# ------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert game-playing AI analyzing an interactive grid-based game.
You receive an observation of the game state and must generate candidate
strategies to make progress (complete the current level).

Each game is UNIQUE — you must infer the rules from observations.

Respond with a JSON array of strategy objects, each with:
- "strategy_type": one of [navigate_to_goal, collect_items, solve_puzzle,
  click_objects, sequence_actions, avoid_hazards, explore_systematically,
  undo_and_retry]
- "description": what to do in plain English
- "action_plan": ordered list of action names (ACTION1-ACTION7, RESET)
- "rationale": why this strategy might work
- "confidence": float 0-1

Generate 2-4 diverse strategies. Consider:
1. What objects exist and what might be goals vs obstacles
2. Which actions cause movement vs interaction vs undo
3. Whether clicking (ACTION6) on specific objects could trigger progress
4. Whether a specific sequence of actions might solve a puzzle
"""


class StrategyGenerator:
    """
    Generates candidate game strategies from observations.

    Supports two modes:
    1. LLM mode: uses a HuggingFace causal LM to generate strategies
    2. Template mode: uses rule-based heuristics (fallback)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_candidates: int = 6,
        use_llm: bool = False,
    ):
        self.max_candidates = max_candidates
        self.use_llm = use_llm and model_name is not None
        self._model = None
        self._tokenizer = None
        self._model_name = model_name
        self._device = device

        if self.use_llm:
            self._init_model()

    def _init_model(self) -> None:
        """Load or reuse shared LLM for strategy generation."""
        model, tokenizer = get_shared_llm(self._model_name, self._device)
        if model is not None:
            self._model = model
            self._tokenizer = tokenizer
        else:
            logger.warning("LLM unavailable for strategy generation. Using templates.")
            self.use_llm = False

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------
    def generate(
        self,
        observation: GameObservation,
        available_actions: List[str],
    ) -> List[GameStrategy]:
        """
        Generate candidate strategies from the current observation.

        Args:
            observation: structured game observation
            available_actions: list of available action names

        Returns:
            List of GameStrategy candidates
        """
        if self.use_llm and self._model is not None:
            strategies = self._generate_with_llm(observation, available_actions)
            if strategies:
                return strategies[:self.max_candidates]

        # Fallback: template-based generation
        return self._generate_from_templates(observation, available_actions)

    # ------------------------------------------------------------------
    # LLM-based generation
    # ------------------------------------------------------------------
    def _generate_with_llm(
        self,
        observation: GameObservation,
        available_actions: List[str],
    ) -> List[GameStrategy]:
        """Use the loaded LLM to generate strategies."""
        prompt = observation.to_prompt()
        prompt += f"\n\nAvailable actions: {available_actions}"
        prompt += "\n\nGenerate 2-4 diverse strategies as a JSON array:"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            # Build input for the model
            if hasattr(self._tokenizer, "apply_chat_template"):
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = f"{_SYSTEM_PROMPT}\n\n{prompt}"

            import torch
            inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )
            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return self._parse_llm_response(response, available_actions)
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return []

    def _parse_llm_response(
        self,
        response: str,
        available_actions: List[str],
    ) -> List[GameStrategy]:
        """Parse LLM JSON response into GameStrategy objects."""
        # Extract JSON array from response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        try:
            data = json.loads(response[start:end])
        except json.JSONDecodeError:
            return []

        strategies = []
        for item in data:
            try:
                st = StrategyType(item.get("strategy_type", "explore_systematically"))
                plan = item.get("action_plan", [])
                # Filter to available actions
                plan = [a for a in plan if a in available_actions or a == "RESET"]
                strategies.append(GameStrategy(
                    strategy_type=st,
                    description=item.get("description", ""),
                    action_plan=plan if plan else available_actions[:3],
                    rationale=item.get("rationale", ""),
                    confidence=float(item.get("confidence", 0.5)),
                ))
            except (ValueError, KeyError):
                continue

        return strategies

    # ------------------------------------------------------------------
    # Template-based generation (fallback)
    # ------------------------------------------------------------------
    def _generate_from_templates(
        self,
        obs: GameObservation,
        available_actions: List[str],
    ) -> List[GameStrategy]:
        """
        Rule-based strategy generation from observation heuristics.

        Analyses the observation to infer game type and generate
        appropriate strategies.
        """
        strategies: List[GameStrategy] = []
        ms = obs.memory_summary
        has_player = obs.player_info is not None
        has_movement = bool(obs.action_semantics)
        has_click = "ACTION6" in available_actions
        has_interact = "ACTION5" in available_actions
        has_undo = "ACTION7" in available_actions
        non_player_objects = [o for o in obs.objects if not o.get("is_player")]
        total_actions = ms.get("total_actions", 0)
        states_visited = ms.get("states_visited", 0)
        max_level = ms.get("max_level", 0)
        game_overs = ms.get("total_game_overs", 0)

        # --- Strategy 1: Navigate to objects (if player + objects) ---
        # Use action_semantics OR memory's movement_actions
        # Exclude ACTION6 (click) and ACTION7 (undo) — not directional movement
        _NON_MOVE = {"ACTION5", "ACTION6", "ACTION7", "RESET"}
        move_actions = [
            act for act, sem in obs.action_semantics.items()
            if "move" in sem and act not in _NON_MOVE
        ]
        # Also check memory for discovered movement actions
        mem_moves = ms.get("movement_actions", [])
        for m in mem_moves:
            if m not in move_actions and m in available_actions and m not in _NON_MOVE:
                move_actions.append(m)

        if has_player and move_actions and non_player_objects:
            strategies.append(GameStrategy(
                strategy_type=StrategyType.NAVIGATE_TO_GOAL,
                description=(
                    f"Navigate player toward {len(non_player_objects)} detected objects. "
                    f"Use movement actions: {move_actions}."
                ),
                action_plan=move_actions,
                rationale=(
                    "Many games require reaching specific locations. "
                    "Moving toward non-player objects may trigger level completion."
                ),
                confidence=0.8 if max_level == 0 else 0.5,
                metadata={"target_objects": [o["value"] for o in non_player_objects[:5]]},
            ))
        elif has_player and non_player_objects:
            # Player exists but no movement found yet — try non-click actions as movement
            fallback_move = [a for a in available_actions if a not in _NON_MOVE]
            if fallback_move:  # Only create navigate if we have real directional candidates
                strategies.append(GameStrategy(
                    strategy_type=StrategyType.NAVIGATE_TO_GOAL,
                    description=(
                        f"Try reaching {len(non_player_objects)} objects. "
                        f"Movement actions not yet identified — trying {fallback_move}."
                    ),
                    action_plan=fallback_move,
                    rationale="Player detected but movement semantics unknown. Try directional actions.",
                    confidence=0.6,
                ))

        # --- Strategy 2: Click on objects (if ACTION6 available) ---
        click_only = has_click and not any(
            a in available_actions for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
        )
        if has_click and (non_player_objects or click_only):
            click_values = ms.get("effective_click_values", [])
            if non_player_objects:
                targets = (
                    [o for o in non_player_objects if o["value"] in click_values]
                    if click_values
                    else non_player_objects
                )
                click_targets = [
                    {"y": int(o["center_y"]), "x": int(o["center_x"])}
                    for o in targets[:10]
                ]
            else:
                # No objects detected — scan all non-zero cells
                targets = []
                click_targets = []
                if obs.raw_grid is not None:
                    nz = list(zip(*obs.raw_grid.nonzero()))
                    # Sample up to 20 positions across the grid
                    step = max(1, len(nz) // 20)
                    for i in range(0, len(nz), step):
                        y, x = int(nz[i][0]), int(nz[i][1])
                        click_targets.append({"y": y, "x": x})

            strategies.append(GameStrategy(
                strategy_type=StrategyType.CLICK_OBJECTS,
                description=(
                    f"Click on {len(click_targets)} targets using ACTION6. "
                    f"{'Click-only game.' if click_only else ''}"
                ),
                action_plan=["ACTION6"],
                rationale=(
                    "Many ARC-AGI-3 games use click mechanics (sys_click sprites). "
                    "Clicking on objects may toggle states or trigger progression."
                ),
                confidence=0.85 if click_only else (0.6 if not click_values else 0.8),
                metadata={"click_targets": click_targets},
            ))

        # --- Strategy 3: Navigate then interact ---
        if has_player and move_actions and has_interact:
            strategies.append(GameStrategy(
                strategy_type=StrategyType.COLLECT_ITEMS,
                description=(
                    "Move to each object and interact (ACTION5). "
                    "This covers games where items must be collected."
                ),
                action_plan=move_actions + ["ACTION5"],
                rationale=(
                    "ACTION5 often switches focus or interacts. "
                    "Moving to an object then pressing ACTION5 may collect it."
                ),
                confidence=0.5,
            ))

        # --- Strategy 4: Puzzle solving via undo ---
        if has_undo and total_actions > 20 and max_level == 0:
            strategies.append(GameStrategy(
                strategy_type=StrategyType.UNDO_AND_RETRY,
                description=(
                    "Undo recent moves with ACTION7, then try different actions. "
                    "Systematically explore action sequences via backtracking."
                ),
                action_plan=["ACTION7"] * 3 + available_actions[:4],
                rationale=(
                    "If simple exploration hasn't worked, backtracking and "
                    "trying alternative paths may solve puzzles."
                ),
                confidence=0.4,
            ))

        # --- Strategy 5: Systematic exploration (always include as fallback) ---
        strategies.append(GameStrategy(
            strategy_type=StrategyType.EXPLORE_SYSTEMATICALLY,
            description=(
                "Try each available action systematically to learn game mechanics. "
                f"Available: {available_actions}."
            ),
            action_plan=available_actions,
            rationale="Explore to discover what each action does.",
            confidence=0.9 if total_actions < 15 else 0.2,
        ))

        # --- Strategy 6: Avoid hazards (if game-overs have occurred) ---
        if game_overs > 0:
            safe_actions = ms.get("action_effects", {})
            dangerous = [
                name for name, eff in safe_actions.items()
                if eff.get("game_over_rate", 0) > 0.1
            ]
            safe = [a for a in available_actions if a not in dangerous]
            strategies.append(GameStrategy(
                strategy_type=StrategyType.AVOID_HAZARDS,
                description=(
                    f"Avoid dangerous actions: {dangerous}. "
                    f"Use safe actions: {safe}."
                ),
                action_plan=safe if safe else available_actions,
                rationale=f"Agent has died {game_overs} times. Avoid risky actions.",
                confidence=0.6,
            ))

        # Ensure we always have at least one strategy
        if not strategies:
            strategies.append(GameStrategy(
                strategy_type=StrategyType.EXPLORE_SYSTEMATICALLY,
                description="Try all available actions to learn the game.",
                action_plan=available_actions,
                rationale="No clear strategy identified; explore to gather information.",
                confidence=0.3,
            ))

        return strategies[:self.max_candidates]

    # ------------------------------------------------------------------
    # Goal-conditioned generation
    # ------------------------------------------------------------------
    def generate_for_goal(
        self,
        observation: GameObservation,
        available_actions: List[str],
        goal: GameObjective,
        failed_strategies: Optional[List[str]] = None,
        partial_strategies: Optional[List[Tuple[str, float]]] = None,
    ) -> List[GameStrategy]:
        """Generate strategies conditioned on a specific goal hypothesis.

        Unlike generate(), this method:
          - Tells the LLM/template which objective to pursue
          - Provides failure history so new strategies differ meaningfully
          - Tags each strategy with the goal_id
        """
        if failed_strategies is None:
            failed_strategies = []
        if partial_strategies is None:
            partial_strategies = []

        if self.use_llm and self._model is not None:
            strategies = self._generate_for_goal_llm(
                observation, available_actions, goal,
                failed_strategies, partial_strategies,
            )
            if strategies:
                return strategies[:self.max_candidates]

        return self._generate_for_goal_templates(
            observation, available_actions, goal,
            failed_strategies,
        )

    # ── LLM goal-conditioned generation ───────────────────────────

    _GOAL_STRATEGY_SYSTEM = """\
You are an expert game-playing AI. You are given:
1. An observation of the current game state
2. A specific OBJECTIVE hypothesis to pursue
3. Previous strategies that FAILED for this objective

Generate 2-4 strategies specifically designed to make progress toward
the given objective. Each strategy MUST be meaningfully different from
the failed ones.

Variation axes: different target object, different board region,
different action family, different sequencing, different causal test.

Respond with a JSON array of strategy objects, each with:
- "strategy_type": one of [navigate_to_goal, collect_items, solve_puzzle,
  click_objects, sequence_actions, avoid_hazards, explore_systematically,
  undo_and_retry]
- "description": what to do (be specific — which objects, which direction)
- "action_plan": ordered list of action names
- "rationale": why this advances the objective and differs from failures
- "confidence": float 0-1
"""

    def _generate_for_goal_llm(
        self,
        observation: GameObservation,
        available_actions: List[str],
        goal: GameObjective,
        failed: List[str],
        partial: List[Tuple[str, float]],
    ) -> List[GameStrategy]:
        prompt = observation.to_prompt()
        prompt += f"\n\n## OBJECTIVE: {goal.description}"
        prompt += f"\nSuccess condition: {goal.success_condition}"
        if goal.progress_signals:
            sigs = ", ".join(s.name for s in goal.progress_signals)
            prompt += f"\nProgress signals: {sigs}"
        if failed:
            prompt += "\n\n## FAILED strategies (do NOT repeat these):\n"
            for f_desc in failed[-5:]:
                prompt += f"  - {f_desc}\n"
        if partial:
            prompt += "\n## PARTIALLY successful strategies (build on these):\n"
            for p_desc, p_score in partial[-3:]:
                prompt += f"  - {p_desc} (progress={p_score:.2f})\n"
        prompt += f"\nAvailable actions: {available_actions}"
        prompt += "\n\nGenerate 2-4 NEW strategies as JSON:"

        if len(prompt) > 1500:
            prompt = prompt[:1500] + "\n...(truncated)"

        messages = [
            {"role": "system", "content": self._GOAL_STRATEGY_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            import torch
            inputs = self._tokenizer(
                input_text, return_tensors="pt", max_length=512, truncation=True
            ).to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=200, temperature=0.8,
                    do_sample=True, top_p=0.9,
                )
            raw = self._tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            strategies = self._parse_llm_response(raw, available_actions)
            for s in strategies:
                s.metadata["goal_id"] = goal.id
            return strategies
        except Exception as e:
            logger.warning(f"LLM goal-conditioned generation failed: {e}")
            return []

    # ── Template goal-conditioned generation ──────────────────────
    def _generate_for_goal_templates(
        self,
        obs: GameObservation,
        available_actions: List[str],
        goal: GameObjective,
        failed: List[str],
    ) -> List[GameStrategy]:
        """Generate strategies targeted at a specific goal from templates."""
        # Start with generic strategies and filter/prioritise by goal type
        all_strategies = self._generate_from_templates(obs, available_actions)

        # Map goal id → preferred strategy types
        goal_type_map = {
            "navigate_to_target": {StrategyType.NAVIGATE_TO_GOAL, StrategyType.AVOID_HAZARDS},
            "collect_items": {StrategyType.COLLECT_ITEMS, StrategyType.NAVIGATE_TO_GOAL},
            "click_puzzle": {StrategyType.CLICK_OBJECTS},
            "sequence_puzzle": {StrategyType.SEQUENCE_ACTIONS, StrategyType.UNDO_AND_RETRY},
            "push_puzzle": {StrategyType.COLLECT_ITEMS, StrategyType.NAVIGATE_TO_GOAL},
            "navigate_avoid_hazards": {StrategyType.AVOID_HAZARDS, StrategyType.NAVIGATE_TO_GOAL},
            "discover_mechanics": {StrategyType.EXPLORE_SYSTEMATICALLY},
        }
        preferred = goal_type_map.get(goal.id, set())

        # Boost confidence of matching strategies
        for s in all_strategies:
            s.metadata["goal_id"] = goal.id
            if s.strategy_type in preferred:
                s.confidence = min(1.0, s.confidence + 0.2)

        # Filter out strategies whose description is in the failed list
        if failed:
            failed_lower = {f.lower()[:60] for f in failed}
            all_strategies = [
                s for s in all_strategies
                if s.description.lower()[:60] not in failed_lower
            ]

        # Sort: preferred types first, then by confidence
        all_strategies.sort(
            key=lambda s: (-int(s.strategy_type in preferred), -s.confidence)
        )

        # Ensure at least one strategy
        if not all_strategies:
            all_strategies.append(GameStrategy(
                strategy_type=StrategyType.EXPLORE_SYSTEMATICALLY,
                description=f"Explore to learn more about: {goal.description}",
                action_plan=available_actions,
                rationale=f"No matching strategy for goal '{goal.id}'. Explore to gather info.",
                confidence=0.3,
                metadata={"goal_id": goal.id},
            ))

        return all_strategies[:self.max_candidates]
