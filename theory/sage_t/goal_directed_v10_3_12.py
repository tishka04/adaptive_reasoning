"""Relational-mechanism controller for the SAGE.T10.3.12 core experiment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from v3.schemas import TransitionRecord

from .contracts import AbstractState, ActionCandidate
from .goal_directed_v10_3_2 import (
    GoalDirectedOption,
    OnlineOptionAutomatonInducer,
    OptionStep,
    _effect_signature,
)
from .goal_directed_v10_3_3 import DYNAMIC_SUCCESSOR, _same_action_data
from .goal_directed_v10_3_6 import _candidate_order_key
from .goal_directed_v10_3_11 import GoalConditionedSageTController
from .progress_witness_v10 import GroundedAction, SearchConfig, chain_successor_macro
from .relational_program_v10_3_12 import (
    ARMS,
    RelationalProgram,
    RelationalProgramRegistry,
    boundary_distance,
)

FORMAT_VERSION = "sage-t10.3.12-relational-mechanism-controller-v1"
RELATIONAL_CAUSAL_ROLE = "t10_3_12_reset_local_causal_role"
GENERIC_FIRST_ROLE = "t10_3_12_generic_first_role"
WRONG_DISTINCT_SUCCESSOR = "t10_3_12_wrong_distinct_successor"
HASH_OFFSET_ABLATION = "t10_3_12_hash_offset_ablation"
LEXICOGRAPHIC_PATH_ABLATION = "t10_3_12_lexicographic_path_ablation"


class OptionLocalAutomatonInducer(OnlineOptionAutomatonInducer):
    """Attach effects from the start of the active option, not reset index zero."""

    def __init__(self) -> None:
        super().__init__()
        self._active_option_id: str | None = None
        self._option_start = 0
        self._prefix_lengths: list[int] = []

    def start_branch(self) -> None:
        super().start_branch()
        self._active_option_id = None
        self._option_start = 0
        self._prefix_lengths.clear()

    @staticmethod
    def align_effects(
        option: GoalDirectedOption,
        branch_effects: Sequence[str],
        *,
        option_start: int,
    ) -> GoalDirectedOption:
        start = max(0, int(option_start))
        return GoalDirectedOption(
            schema=option.schema,
            steps=tuple(
                OptionStep(
                    action_name=step.action_name,
                    binding_method=step.binding_method,
                    structural_signature=step.structural_signature,
                    expected_effect=(
                        str(branch_effects[start + index])
                        if start + index < len(branch_effects)
                        else step.expected_effect
                    ),
                )
                for index, step in enumerate(option.steps)
            ),
            source="option_local_level_progress_reproduction",
        )

    def observe(
        self,
        record: TransitionRecord,
        *,
        selected_step: OptionStep | None,
        active_option: GoalDirectedOption | None,
    ) -> GoalDirectedOption | None:
        action_name = str(record.action.name).strip().upper()
        step = selected_step or OptionStep(action_name=action_name)
        effect = _effect_signature(record)
        productive = not record.diff.is_noop and not record.diff.game_over
        step = OptionStep(
            action_name=step.action_name,
            binding_method=step.binding_method,
            structural_signature=step.structural_signature,
            expected_effect=effect,
        )
        if active_option is not None and active_option.option_id != self._active_option_id:
            self._active_option_id = active_option.option_id
            self._option_start = len(self._branch)
            self._prefix_lengths.append(self._option_start)
        self._branch.append((step, effect, productive))
        self._effects_by_action[action_name][effect] += 1
        progressed = bool(
            record.diff.level_complete
            or record.obs_after.levels_completed > record.obs_before.levels_completed
        )
        if not progressed:
            if active_option is None:
                self._active_option_id = None
            return None
        if active_option is not None:
            learned = self.align_effects(
                active_option,
                tuple(row[1] for row in self._branch),
                option_start=self._option_start,
            )
        else:
            suffix = self._minimal_productive_suffix()
            schema = (
                "mixed_automaton"
                if len({item.action_name for item in suffix}) >= 2
                else "repeat_target"
            )
            learned = GoalDirectedOption(
                schema=schema,
                steps=suffix,
                source="option_local_level_progress_induction",
            )
        self._successful_options.append(learned)
        self._active_option_id = None
        return learned

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base.update(
            {
                "option_local_effect_alignment": True,
                "observed_prefix_lengths": list(self._prefix_lengths),
            }
        )
        return base


class RelationalMechanismSageTController(GoalConditionedSageTController):
    """Execute one preregistered relational arm with no cross-reset memory."""

    def __init__(
        self,
        *args: Any,
        arm: str,
        relational_registry: RelationalProgramRegistry,
        **kwargs: Any,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported relational arm: {arm}")
        kwargs.setdefault("goal_conditioning_enabled", False)
        kwargs.setdefault("warmup_actions", 0)
        kwargs.setdefault("exploration_interval", 1)
        super().__init__(*args, **kwargs)
        self.arm = arm
        self.relational_registry = relational_registry
        self.inducer = OptionLocalAutomatonInducer()
        self._current_shape: tuple[int, int] = (64, 64)
        self._recognized_context = ""
        self._current_program: RelationalProgram | None = None
        self._program_hashes_used: set[str] = set()
        self._relational_inspections = 0
        self._relational_abstentions: Counter[str] = Counter()
        self._role_binding_uses: Counter[str] = Counter()
        self._relation_attestations: Counter[str] = Counter()
        self._ablated_successor_plan: tuple[GroundedAction, ...] = ()

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self._recognized_context = ""
        self._current_program = None
        self._program_hashes_used.clear()
        self._relational_inspections = 0
        self._relational_abstentions.clear()
        self._role_binding_uses.clear()
        self._relation_attestations.clear()
        self._ablated_successor_plan = ()

    def decide(self, *args: Any, observation: Any, **kwargs: Any):
        raw = getattr(observation, "raw_grid", None)
        if raw is None:
            raw = getattr(observation, "grid", None)
        shape = tuple(getattr(raw, "shape", ()))
        if len(shape) >= 2:
            self._current_shape = (int(shape[0]), int(shape[1]))
        return super().decide(*args, observation=observation, **kwargs)

    @staticmethod
    def _grounded(candidates: Sequence[ActionCandidate]) -> tuple[GroundedAction, ...]:
        return tuple(
            GroundedAction(
                candidate.action_name,
                tuple(dict(candidate.action_data).items()),
            )
            for candidate in candidates
        )

    def _context(
        self, state: AbstractState, candidates: Sequence[ActionCandidate]
    ) -> tuple[str, Any | None]:
        macro = chain_successor_macro(
            state,
            self._grounded(candidates),
            config=SearchConfig(maximum_horizon=16),
        )
        self._relational_inspections += min(32, len(candidates))
        if macro is not None and macro.actions:
            return "path_context", macro
        if any(candidate.action_data for candidate in candidates):
            return "repeat_context", None
        return "", None

    @staticmethod
    def _parameterized_action_name(candidates: Sequence[ActionCandidate]) -> str | None:
        by_name: dict[str, int] = {}
        for candidate in candidates:
            if candidate.action_data:
                by_name[candidate.action_name] = by_name.get(candidate.action_name, 0) + 1
        return (
            None
            if not by_name
            else min(by_name, key=lambda name: (-by_name[name], name))
        )

    def _choose_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any] = (),
    ) -> GoalDirectedOption | None:
        del goal_hypotheses
        context, macro = self._context(state, candidates)
        if not context:
            self._relational_abstentions["no_relational_context"] += 1
            return None
        program = self.relational_registry.program_for(self.arm, context)
        self._recognized_context = context
        self._current_program = program
        self._program_hashes_used.add(program.program_hash)
        if context == "path_context" and program.mechanism in {
            "salient_path_successor",
            "generic_path_search",
        }:
            if macro is None or not macro.actions:
                self._relational_abstentions["path_grounding_miss"] += 1
                return None
            action_name = macro.actions[0].action_name
            return GoalDirectedOption(
                schema="path_successor",
                steps=tuple(
                    OptionStep(
                        action_name,
                        binding_method=DYNAMIC_SUCCESSOR,
                        expected_effect="successor_toward_unique_salient_end",
                    )
                    for _ in range(program.safety_horizon)
                ),
                source=f"t10_3_12_{self.arm}",
            )
        action_name = self._parameterized_action_name(candidates)
        if action_name is None:
            self._relational_abstentions["parameterized_schema_miss"] += 1
            return None
        binding = {
            "repeat_causal_role": RELATIONAL_CAUSAL_ROLE,
            "generic_repeat_search": GENERIC_FIRST_ROLE,
            "path_schema_on_repeat_context": WRONG_DISTINCT_SUCCESSOR,
            "repeat_schema_on_path_context": GENERIC_FIRST_ROLE,
            "hash_offset_repeat_ablation": HASH_OFFSET_ABLATION,
            "lexicographic_path_ablation": LEXICOGRAPHIC_PATH_ABLATION,
        }[program.mechanism]
        schema = "path_successor" if binding == LEXICOGRAPHIC_PATH_ABLATION else "repeat_target"
        return GoalDirectedOption(
            schema=schema,
            steps=tuple(
                OptionStep(
                    action_name,
                    binding_method=binding,
                    expected_effect=program.transition_relation,
                )
                for _ in range(program.safety_horizon)
            ),
            source=f"t10_3_12_{self.arm}",
        )

    def _select_repeat_candidate(
        self,
        step: OptionStep,
        candidates: Sequence[ActionCandidate],
    ) -> ActionCandidate | None:
        matches = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.action_name == step.action_name and candidate.action_data
            ),
            key=_candidate_order_key,
        )
        self._relational_inspections += len(matches)
        if not matches:
            return None
        if step.binding_method == RELATIONAL_CAUSAL_ROLE:
            scored = [
                (boundary_distance(candidate.action_data, self._current_shape), candidate)
                for candidate in matches
            ]
            scored = [(score, candidate) for score, candidate in scored if score is not None]
            if not scored:
                return None
            best = min(score for score, _ in scored)
            winners = [candidate for score, candidate in scored if score == best]
            if len(winners) != 1:
                self._relational_abstentions["ambiguous_boundary_role"] += 1
                return None
            return winners[0]
        if step.binding_method == HASH_OFFSET_ABLATION:
            return matches[self._exploration_rotation % len(matches)]
        if step.binding_method == WRONG_DISTINCT_SUCCESSOR:
            return matches[self._active_cursor % len(matches)]
        return matches[0]

    def _continue_active_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
    ) -> ActionCandidate | None:
        option = self._active_option
        if option is None or self._active_cursor >= len(option.steps):
            return super()._continue_active_option(state, candidates)
        step = option.steps[self._active_cursor]
        custom_repeat = {
            RELATIONAL_CAUSAL_ROLE,
            GENERIC_FIRST_ROLE,
            WRONG_DISTINCT_SUCCESSOR,
            HASH_OFFSET_ABLATION,
        }
        if step.binding_method in custom_repeat:
            selected = self._select_repeat_candidate(step, candidates)
            if selected is None:
                self._last_grounding_failure = "t10_3_12_repeat_role_miss"
                return None
            self._pending_grounded_candidate = selected
            self._binding_method_uses[step.binding_method] += 1
            self._role_binding_uses[step.binding_method] += 1
            return selected
        if step.binding_method == LEXICOGRAPHIC_PATH_ABLATION:
            if not self._ablated_successor_plan:
                macro = chain_successor_macro(
                    state,
                    self._grounded(candidates),
                    config=SearchConfig(maximum_horizon=16),
                )
                if macro is None or not macro.actions:
                    self._last_grounding_failure = "ablated_path_miss"
                    return None
                # Deliberately remove enclosure salience by using the opposite
                # orientation of the relational generator.
                self._ablated_successor_plan = tuple(reversed(macro.actions))
            if self._active_cursor >= len(self._ablated_successor_plan):
                return None
            wanted = self._ablated_successor_plan[self._active_cursor]
            matches = [
                candidate
                for candidate in candidates
                if candidate.action_name == wanted.action_name
                and _same_action_data(candidate.action_data, wanted.data)
            ]
            self._relational_inspections += len(candidates)
            if len(matches) != 1:
                return None
            self._pending_grounded_candidate = matches[0]
            self._binding_method_uses[step.binding_method] += 1
            return matches[0]
        selected = super()._continue_active_option(state, candidates)
        if selected is not None and step.binding_method == DYNAMIC_SUCCESSOR:
            self._relation_attestations["successor"] += 1
            self._relation_attestations["orientation"] += 1
        return selected

    def observe_transition(self, record: TransitionRecord) -> None:
        pending = self._pending_option
        super().observe_transition(record)
        if pending is not None and (
            record.diff.level_complete
            or record.obs_after.levels_completed > record.obs_before.levels_completed
        ):
            self._relation_attestations["level_stop"] += 1

    def _finish_active_option(self, *, progressed: bool, reason: str) -> None:
        super()._finish_active_option(progressed=progressed, reason=reason)
        self._ablated_successor_plan = ()

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "relational_arm": self.arm,
                "recognized_context": self._recognized_context,
                "relational_program_hashes_used": sorted(self._program_hashes_used),
                "source_information_loaded": self.arm == ARMS[0],
                "relational_inspections": self._relational_inspections,
                "relational_abstentions": dict(self._relational_abstentions),
                "role_binding_uses": dict(self._role_binding_uses),
                "relation_attestations": dict(self._relation_attestations),
                "option_local_effect_alignment": True,
                "cross_reset_memory": False,
                "grounded_arguments_persisted": False,
            }
        )
        return base


__all__ = [
    "FORMAT_VERSION",
    "GENERIC_FIRST_ROLE",
    "HASH_OFFSET_ABLATION",
    "LEXICOGRAPHIC_PATH_ABLATION",
    "OptionLocalAutomatonInducer",
    "RELATIONAL_CAUSAL_ROLE",
    "RelationalMechanismSageTController",
    "WRONG_DISTINCT_SUCCESSOR",
]
