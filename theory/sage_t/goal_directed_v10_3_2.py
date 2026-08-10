"""Goal-directed, closed-loop SAGE.T controller for the T10.3.2 source pilot.

The module deliberately extends SAGE.T through its existing controller injection
boundary.  It does not change the frozen generic executor or decision engine.
Full frames are consumed transiently by the normal compiler, while transferable
options retain only action schemas and coordinate-free structural bindings.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from v3.schemas import TransitionRecord

from .compiler import compile_observation
from .contracts import (
    AbstractState,
    ActionCandidate,
    normalized_action_candidates,
)
from .controller import SageTArbitration, SageTConfig, SageTController
from .frame_adapters_v10_3 import (
    resolve_pre_action_root,
    structural_signature,
)
from .progress_witness_v10 import (
    GroundedAction,
    SearchConfig,
    chain_successor_macro,
)

FORMAT_VERSION = "sage-t10.3.2-goal-directed-v1"
REGISTRY_FORMAT_VERSION = "sage-t10.3.2-progress-program-registry-v1"
MAXIMUM_OPTION_HORIZON = 32
SHORTLIST_HORIZON = 8
SHORTLIST_SIZE = 8
DISCOVERY_WARMUP_ACTIONS = 32
EXPLORATION_ACTIONS_BETWEEN_OPTIONS = 8
MAXIMUM_STERILE_OPTION_ACTIONS = 4

_FORBIDDEN_SERIALIZED_KEYS = frozenset(
    {
        "game_id",
        "seed",
        "x",
        "y",
        "coordinate",
        "coordinates",
        "color",
        "colour",
        "raw_grid",
        "grid",
        "entity_id",
        "object_id",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _signed(payload: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    result = dict(payload)
    result[field_name] = _sha(result)
    return result


def _assert_transfer_safe(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_SERIALIZED_KEYS:
                raise ValueError(f"forbidden transferable field at {path}.{key}")
            _assert_transfer_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_transfer_safe(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class OptionStep:
    """One re-groundable, identity-free action in an option automaton."""

    action_name: str
    binding_method: str = "unique_action_schema"
    structural_signature: str | None = None
    expected_effect: str = "unknown"

    def __post_init__(self) -> None:
        name = str(self.action_name).strip().upper()
        if not name.startswith("ACTION"):
            raise ValueError("option steps require an ACTION schema")
        object.__setattr__(self, "action_name", name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "binding_method": self.binding_method,
            "structural_signature": self.structural_signature,
            "expected_effect": self.expected_effect,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OptionStep:
        return cls(
            action_name=str(payload["action_name"]),
            binding_method=str(payload.get("binding_method", "unique_action_schema")),
            structural_signature=(
                None
                if payload.get("structural_signature") is None
                else str(payload["structural_signature"])
            ),
            expected_effect=str(payload.get("expected_effect", "unknown")),
        )


@dataclass(frozen=True)
class GoalDirectedOption:
    """A bounded initiation/dynamics/termination automaton."""

    schema: str
    steps: tuple[OptionStep, ...]
    initiation: str = "groundable"
    termination: str = "level_progress_or_option_stop"
    source: str = "online_induction"

    def __post_init__(self) -> None:
        if not 1 <= len(self.steps) <= MAXIMUM_OPTION_HORIZON:
            raise ValueError("goal-directed options require 1-32 steps")
        if self.schema not in {"repeat_target", "path_successor", "mixed_automaton"}:
            raise ValueError("unsupported goal-directed option schema")

    @property
    def action_schemas(self) -> tuple[str, ...]:
        return tuple(step.action_name for step in self.steps)

    @property
    def mixed(self) -> bool:
        return len(set(self.action_schemas)) >= 2

    @property
    def option_id(self) -> str:
        return _sha(self.safe_payload)

    @property
    def safe_payload(self) -> dict[str, Any]:
        payload = {
            "format_version": FORMAT_VERSION,
            "schema": self.schema,
            "initiation": self.initiation,
            "dynamics": [step.as_dict() for step in self.steps],
            "termination": self.termination,
            "source": self.source,
        }
        _assert_transfer_safe(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GoalDirectedOption:
        _assert_transfer_safe(payload)
        return cls(
            schema=str(payload["schema"]),
            steps=tuple(OptionStep.from_dict(row) for row in payload["dynamics"]),
            initiation=str(payload.get("initiation", "groundable")),
            termination=str(
                payload.get("termination", "level_progress_or_option_stop")
            ),
            source=str(payload.get("source", "transferred_registry")),
        )


@dataclass
class _RegistryEvidence:
    option: GoalDirectedOption
    success_attestations: set[str] = field(default_factory=set)
    support_scopes: set[str] = field(default_factory=set)
    contradiction_attestations: set[str] = field(default_factory=set)
    binding_swap_passed: bool = False
    order_permutation_passed: bool = False
    automaton_ablation_passed: bool = False

    @property
    def reproducible(self) -> bool:
        return len(self.support_scopes) >= 2

    @property
    def controls_passed(self) -> bool:
        return bool(
            self.binding_swap_passed
            and self.order_permutation_passed
            and self.automaton_ablation_passed
        )

    @property
    def promoted(self) -> bool:
        return self.reproducible and self.controls_passed


class ProgressProgramRegistry:
    """Coordinate-free option registry with explicit local support-zero transfer."""

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self._evidence: dict[str, _RegistryEvidence] = {}
        self._local_support: Counter[str] = Counter()
        self._local_contradictions: Counter[str] = Counter()
        self.loaded_checksum: str | None = None
        if payload is not None:
            self._load(payload)

    def _load(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format_version") != REGISTRY_FORMAT_VERSION:
            raise ValueError("unsupported progress registry format")
        expected = str(payload.get("registry_checksum", ""))
        core = {key: value for key, value in payload.items() if key != "registry_checksum"}
        if not expected or _sha(core) != expected:
            raise ValueError("progress registry checksum mismatch")
        _assert_transfer_safe(core)
        for row in payload.get("programs", ()):
            option = GoalDirectedOption.from_payload(row["program"])
            if str(row.get("option_id")) != option.option_id:
                raise ValueError("progress registry option id mismatch")
            current = self._evidence.setdefault(
                option.option_id, _RegistryEvidence(option=option)
            )
            current.success_attestations.update(
                str(item) for item in row.get("success_attestations", ())
            )
            current.support_scopes.update(
                str(item) for item in row.get("support_scopes", ())
            )
            current.contradiction_attestations.update(
                str(item) for item in row.get("contradiction_attestations", ())
            )
            current.binding_swap_passed = bool(
                current.binding_swap_passed or row.get("binding_swap_passed")
            )
            current.order_permutation_passed = bool(
                current.order_permutation_passed or row.get("order_permutation_passed")
            )
            current.automaton_ablation_passed = bool(
                current.automaton_ablation_passed
                or row.get("automaton_ablation_passed")
            )
        self.loaded_checksum = expected

    def merge(self, payload: Mapping[str, Any]) -> None:
        """Merge a signed safe snapshot without importing local support."""

        self._load(payload)

    def register_candidate(self, option: GoalDirectedOption) -> None:
        self._evidence.setdefault(option.option_id, _RegistryEvidence(option=option))

    def note_success(
        self,
        option: GoalDirectedOption,
        attestation: str,
        *,
        scope: str | None = None,
    ) -> None:
        self.register_candidate(option)
        token = _sha({"attestation": str(attestation), "option": option.option_id})
        self._evidence[option.option_id].success_attestations.add(token)
        self._evidence[option.option_id].support_scopes.add(
            _sha({"scope": str(scope if scope is not None else attestation)})
        )
        self._local_support[option.option_id] += 1

    def note_contradiction(self, option: GoalDirectedOption, attestation: str) -> None:
        self.register_candidate(option)
        token = _sha({"contradiction": str(attestation), "option": option.option_id})
        self._evidence[option.option_id].contradiction_attestations.add(token)
        self._local_contradictions[option.option_id] += 1

    def note_controls(
        self,
        option_id: str,
        *,
        binding_swap: bool,
        order_permutation: bool,
        automaton_ablation: bool,
    ) -> None:
        row = self._evidence[str(option_id)]
        row.binding_swap_passed = bool(binding_swap)
        row.order_permutation_passed = bool(order_permutation)
        row.automaton_ablation_passed = bool(automaton_ablation)

    def transferred_options(self) -> tuple[GoalDirectedOption, ...]:
        return tuple(
            row.option
            for _, row in sorted(self._evidence.items())
            if row.promoted
        )

    def eligible_transferred_options(self) -> tuple[GoalDirectedOption, ...]:
        """Promoted options not contradicted in the current fresh controller."""

        return tuple(
            option
            for option in self.transferred_options()
            if self.local_contradictions(option.option_id) == 0
        )

    def candidates(self) -> tuple[GoalDirectedOption, ...]:
        return tuple(row.option for _, row in sorted(self._evidence.items()))

    def reproduction_candidates(self) -> tuple[GoalDirectedOption, ...]:
        return tuple(
            row.option
            for _, row in sorted(self._evidence.items())
            if row.success_attestations and not row.promoted
        )

    def local_support(self, option_id: str) -> int:
        return int(self._local_support[str(option_id)])

    def local_contradictions(self, option_id: str) -> int:
        return int(self._local_contradictions[str(option_id)])

    def evidence(self, option_id: str) -> _RegistryEvidence | None:
        return self._evidence.get(str(option_id))

    def snapshot(self, *, promoted_only: bool = False) -> dict[str, Any]:
        programs = []
        for option_id, row in sorted(self._evidence.items()):
            if promoted_only and not row.promoted:
                continue
            programs.append(
                {
                    "option_id": option_id,
                    "program": row.option.safe_payload,
                    "success_attestations": sorted(row.success_attestations),
                    "support_scopes": sorted(row.support_scopes),
                    "contradiction_attestations": sorted(
                        row.contradiction_attestations
                    ),
                    "binding_swap_passed": row.binding_swap_passed,
                    "order_permutation_passed": row.order_permutation_passed,
                    "automaton_ablation_passed": row.automaton_ablation_passed,
                    "promoted": row.promoted,
                }
            )
        core = {
            "format_version": REGISTRY_FORMAT_VERSION,
            "transfer_support_policy": "support_zero_until_local_observation",
            "programs": programs,
        }
        _assert_transfer_safe(core)
        return _signed(core, "registry_checksum")


def _effect_signature(record: TransitionRecord) -> str:
    diff = record.diff
    moved_bucket = min(4, len(diff.moved_objects))
    created_bucket = min(4, len(diff.created_objects))
    removed_bucket = min(4, len(diff.removed_objects))
    changed_bucket = min(8, int(diff.num_changed) // 4)
    displacement = diff.player_displacement
    direction = "none"
    if displacement is not None:
        dy, dx = displacement
        direction = (
            "vertical"
            if abs(int(dy)) > abs(int(dx))
            else "horizontal"
            if abs(int(dx)) > 0
            else "none"
        )
    return _sha(
        {
            "noop": bool(diff.is_noop),
            "moved_bucket": moved_bucket,
            "created_bucket": created_bucket,
            "removed_bucket": removed_bucket,
            "changed_bucket": changed_bucket,
            "actor_axis": direction,
            "level_progress": bool(
                diff.level_complete
                or record.obs_after.levels_completed
                > record.obs_before.levels_completed
            ),
            "terminal": bool(diff.game_over),
        }
    )


class OnlineOptionAutomatonInducer:
    """Induce productive mixed schemas from the current branch only."""

    def __init__(self) -> None:
        self._branch: list[tuple[OptionStep, str, bool]] = []
        self._effects_by_action: dict[str, Counter[str]] = defaultdict(Counter)
        self._successful_options: list[GoalDirectedOption] = []

    def start_branch(self) -> None:
        self._branch.clear()

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
        self._branch.append((step, effect, productive))
        self._effects_by_action[action_name][effect] += 1
        progressed = bool(
            record.diff.level_complete
            or record.obs_after.levels_completed > record.obs_before.levels_completed
        )
        if not progressed:
            return None
        if active_option is not None:
            learned = GoalDirectedOption(
                schema=active_option.schema,
                steps=tuple(
                    OptionStep(
                        action_name=item.action_name,
                        binding_method=item.binding_method,
                        structural_signature=item.structural_signature,
                        expected_effect=(
                            self._branch[index][1]
                            if index < len(self._branch)
                            else item.expected_effect
                        ),
                    )
                    for index, item in enumerate(active_option.steps)
                ),
                source="level_progress_reproduction",
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
                source="level_progress_induction",
            )
        self._successful_options.append(learned)
        return learned

    def _minimal_productive_suffix(self) -> tuple[OptionStep, ...]:
        rows = self._branch[-MAXIMUM_OPTION_HORIZON:]
        start = 0
        for index, (_, _, productive) in enumerate(rows):
            if productive:
                start = index
                break
        suffix = tuple(row[0] for row in rows[start:])
        return suffix or (rows[-1][0],)

    def productive_action_names(self) -> tuple[str, ...]:
        names = []
        for name, effects in sorted(self._effects_by_action.items()):
            if effects:
                names.append(name)
        return tuple(names)

    def mixed_candidates(
        self,
        legal_action_names: Sequence[str],
        *,
        subgoal_action_names: Sequence[str] = (),
    ) -> tuple[GoalDirectedOption, ...]:
        productive = self.productive_action_names()
        legal = {str(item).strip().upper() for item in legal_action_names}
        subgoal = tuple(
            name
            for name in dict.fromkeys(
                str(item).strip().upper() for item in subgoal_action_names
            )
            if name in legal
        )
        effect_graph = tuple(name for name in productive if name in legal)
        available = tuple(dict.fromkeys((*subgoal, *effect_graph)))
        if len(available) < 2:
            available = tuple(sorted(legal))
        if len(available) < 2:
            return ()
        options = []
        for length in (16, 24, 32):
            steps = tuple(
                OptionStep(action_name=available[index % len(available)])
                for index in range(length)
            )
            options.append(
                GoalDirectedOption(
                    schema="mixed_automaton",
                    steps=steps,
                    source="effect_graph_composition",
                )
            )
        return tuple(options)

    def summary(self) -> dict[str, Any]:
        return {
            "branch_steps": len(self._branch),
            "action_schema_count": len(self._effects_by_action),
            "successful_option_count": len(self._successful_options),
        }


@dataclass(frozen=True)
class OptionAssessment:
    option: GoalDirectedOption
    short_score: float
    extended_score: float
    predicted_progress_probability: float


class ExtendedOptionEvaluator:
    """Two-stage, bounded option evaluator extending the frozen 8-step engine."""

    def assess(
        self,
        options: Sequence[GoalDirectedOption],
        *,
        registry: ProgressProgramRegistry,
        tried_option_ids: set[str],
    ) -> tuple[OptionAssessment, ...]:
        short = []
        for option in options:
            evidence = registry.evidence(option.option_id)
            successes = 0 if evidence is None else len(evidence.success_attestations)
            contradictions = (
                0 if evidence is None else len(evidence.contradiction_attestations)
            )
            probability = (successes + 1.0) / (successes + contradictions + 2.0)
            prefix = option.steps[:SHORTLIST_HORIZON]
            diversity = len({step.action_name for step in prefix})
            short_score = (
                4.0 * probability
                + (2.0 if option.option_id not in tried_option_ids else -2.0)
                + (1.0 if option.schema == "path_successor" else 0.0)
                + 0.2 * diversity
                - 0.01 * len(prefix)
                + 0.5 * registry.local_support(option.option_id)
                - 1.0 * registry.local_contradictions(option.option_id)
            )
            short.append((option, short_score, probability))
        shortlist = sorted(
            short,
            key=lambda row: (-row[1], row[0].option_id),
        )[:SHORTLIST_SIZE]
        assessments = []
        for option, short_score, probability in shortlist:
            extended_score = (
                short_score
                + 0.35 * min(4, len(set(option.action_schemas)))
                + (1.5 if option.mixed else 0.0)
                - 0.015 * len(option.steps)
            )
            assessments.append(
                OptionAssessment(
                    option=option,
                    short_score=short_score,
                    extended_score=extended_score,
                    predicted_progress_probability=probability,
                )
            )
        return tuple(
            sorted(
                assessments,
                key=lambda row: (-row.extended_score, row.option.option_id),
            )
        )


def _step_for_candidate(state: AbstractState, candidate: ActionCandidate) -> OptionStep:
    resolved = resolve_pre_action_root(state, candidate)
    signature = (
        structural_signature(state, resolved.entity_id)
        if resolved.entity_id is not None and resolved.unique
        else None
    )
    return OptionStep(
        action_name=candidate.action_name,
        binding_method=resolved.method,
        structural_signature=signature,
    )


def _candidate_by_grounded(
    candidates: Sequence[ActionCandidate], grounded: GroundedAction
) -> ActionCandidate | None:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.action_name == grounded.action_name
            and dict(candidate.action_data) == dict(grounded.data)
        ),
        None,
    )


class GoalDirectedSageTController(SageTController):
    """Source-only SAGE.T authority that learns and executes goal options live."""

    def __init__(
        self,
        *,
        phase: str = "discovery",
        registry: ProgressProgramRegistry | None = None,
        registry_checksum: str | None = None,
        attestation_scope: str = "",
        warmup_actions: int = DISCOVERY_WARMUP_ACTIONS,
        exploration_interval: int = EXPLORATION_ACTIONS_BETWEEN_OPTIONS,
    ) -> None:
        super().__init__(
            config=SageTConfig(
                mode="shadow",
                maximum_programs=64,
                maximum_sequences=64,
                maximum_particles_per_decision=16,
                ordinary_horizon=3,
            )
        )
        if phase not in {"discovery", "confirmation", "preflight"}:
            raise ValueError("goal-directed phase must be discovery, confirmation or preflight")
        self.phase = phase
        self.registry = registry or ProgressProgramRegistry()
        if phase == "confirmation" and not registry_checksum:
            raise ValueError("confirmation controller requires a registry checksum")
        if (
            registry_checksum is not None
            and self.registry.loaded_checksum is not None
            and registry_checksum != self.registry.loaded_checksum
        ):
            raise ValueError("loaded registry checksum does not match controller receipt")
        self.registry_checksum = registry_checksum or self.registry.loaded_checksum
        self.attestation_scope = _sha({"scope": str(attestation_scope)})
        self.inducer = OnlineOptionAutomatonInducer()
        self.evaluator = ExtendedOptionEvaluator()
        self.warmup_actions = max(0, int(warmup_actions))
        self.exploration_interval = max(1, int(exploration_interval))
        self._decision_index = 0
        self._exploration_since_option = 0
        self._active_option: GoalDirectedOption | None = None
        self._active_cursor = 0
        self._active_sterile = 0
        self._pending_step: OptionStep | None = None
        self._pending_option: GoalDirectedOption | None = None
        self._tried_option_ids: set[str] = set()
        self._source_counts: Counter[str] = Counter()
        self._option_successes = 0
        self._option_contradictions = 0
        self._registry_used_in_decision = False
        self._last_decision_registry_checksum: str | None = None
        self._successful_option_action_schemas: list[tuple[str, ...]] = []

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self.inducer.start_branch()
        self._decision_index = 0
        self._exploration_since_option = 0
        self._active_option = None
        self._active_cursor = 0
        self._active_sterile = 0
        self._pending_step = None
        self._pending_option = None
        self._tried_option_ids.clear()
        self._last_decision_registry_checksum = None
        self._successful_option_action_schemas.clear()

    def decide(
        self,
        *,
        symbolic_action_name: str,
        symbolic_action_data: Mapping[str, Any] | None,
        observation: Any,
        legal_actions: Sequence[Any],
        mechanic_theory: Any | None = None,
        goal_hypotheses: Sequence[Any] = (),
        route_memory: Any | None = None,
        danger_veto: Callable[[ActionCandidate], bool] | None = None,
        protected_route: bool = False,
    ) -> SageTArbitration:
        shadow = super().decide(
            symbolic_action_name=symbolic_action_name,
            symbolic_action_data=symbolic_action_data,
            observation=observation,
            legal_actions=legal_actions,
            mechanic_theory=mechanic_theory,
            goal_hypotheses=goal_hypotheses,
            route_memory=route_memory,
            danger_veto=danger_veto,
            protected_route=protected_route,
        )
        self._last_decision_registry_checksum = None
        self._decision_index += 1
        try:
            candidates = normalized_action_candidates(legal_actions)
            state = compile_observation(observation, regime_index=self._regime_index)
        except (TypeError, ValueError):
            self._source_counts["uncompilable_fallback"] += 1
            return shadow
        if protected_route:
            self._source_counts["protected_route"] += 1
            return shadow

        selected = self._continue_active_option(state, candidates)
        if selected is None and self._active_option is not None:
            self._finish_active_option(progressed=False, reason="grounding_miss")

        transferred_ready = bool(self.registry.eligible_transferred_options())
        immediate_confirmation = bool(
            self.phase == "confirmation"
            and self._decision_index == 1
            and transferred_ready
        )
        scheduled_trial = bool(
            self._decision_index > self.warmup_actions
            and self._exploration_since_option >= self.exploration_interval
        )
        preflight_trial = self.phase == "preflight" and self._active_option is None
        if selected is None and (immediate_confirmation or scheduled_trial or preflight_trial):
            option = self._choose_option(
                state,
                candidates,
                goal_hypotheses=goal_hypotheses,
            )
            if option is not None:
                self._active_option = option
                self._active_cursor = 0
                self._active_sterile = 0
                self._tried_option_ids.add(option.option_id)
                selected = self._continue_active_option(state, candidates)
                self._exploration_since_option = 0

        if selected is None:
            self._pending_step = None
            self._pending_option = None
            self._exploration_since_option += 1
            self._source_counts["unified_exploration"] += 1
            return shadow
        if danger_veto is not None and danger_veto(selected):
            self._finish_active_option(progressed=False, reason="danger_veto")
            self._source_counts["danger_veto"] += 1
            return shadow
        self._pending_step = self._active_option.steps[self._active_cursor]
        self._pending_option = self._active_option
        self._source_counts["sage_t_goal_option"] += 1
        transferred_ids = {
            option.option_id for option in self.registry.eligible_transferred_options()
        }
        if (
            self.registry_checksum is not None
            and self._active_option.option_id in transferred_ids
        ):
            self._registry_used_in_decision = True
            self._last_decision_registry_checksum = self.registry_checksum
        return SageTArbitration(
            action_name=selected.action_name,
            action_data=dict(selected.action_data),
            applied=True,
            requested_mode="source_experimental_active",
            effective_mode="source_experimental_active",
            reason="goal_option_receding_horizon",
            decision=shadow.decision,
            posterior=shadow.posterior,
        )

    def _choose_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any] = (),
    ) -> GoalDirectedOption | None:
        untried_transferred = [
            option
            for option in self.registry.eligible_transferred_options()
            if option.option_id not in self._tried_option_ids
        ]
        if self.phase == "confirmation" and untried_transferred:
            assessments = self.evaluator.assess(
                tuple(untried_transferred),
                registry=self.registry,
                tried_option_ids=self._tried_option_ids,
            )
            return assessments[0].option if assessments else None
        untried_reproductions = [
            option
            for option in self.registry.reproduction_candidates()
            if option.option_id not in self._tried_option_ids
        ]
        if self.phase == "discovery" and untried_reproductions:
            assessments = self.evaluator.assess(
                tuple(untried_reproductions),
                registry=self.registry,
                tried_option_ids=self._tried_option_ids,
            )
            return assessments[0].option if assessments else None
        generated = list(untried_transferred)
        parameterized = [candidate for candidate in candidates if candidate.action_data]
        for candidate in parameterized[:8]:
            step = _step_for_candidate(state, candidate)
            for length in (2, 4, 8, 16):
                generated.append(
                    GoalDirectedOption(
                        schema="repeat_target",
                        steps=tuple(step for _ in range(length)),
                        source="dynamic_repeat_search",
                    )
                )
        grounded = tuple(
            GroundedAction(
                candidate.action_name,
                tuple(dict(candidate.action_data).items()),
            )
            for candidate in candidates
        )
        chain = chain_successor_macro(
            state,
            grounded,
            config=SearchConfig(maximum_horizon=MAXIMUM_OPTION_HORIZON),
        )
        if chain is not None:
            chain_steps = []
            for item in chain.actions[:MAXIMUM_OPTION_HORIZON]:
                candidate = _candidate_by_grounded(candidates, item)
                if candidate is not None:
                    chain_steps.append(_step_for_candidate(state, candidate))
            if chain_steps:
                generated.append(
                    GoalDirectedOption(
                        schema="path_successor",
                        steps=tuple(chain_steps),
                        source="dynamic_successor_search",
                    )
                )
        generated.extend(
            self.inducer.mixed_candidates(
                tuple(candidate.action_name for candidate in candidates),
                subgoal_action_names=tuple(
                    action
                    for hypothesis in goal_hypotheses
                    for action in getattr(hypothesis, "supporting_actions", ())
                ),
            )
        )
        unique = {option.option_id: option for option in generated}
        assessments = self.evaluator.assess(
            tuple(unique.values()),
            registry=self.registry,
            tried_option_ids=self._tried_option_ids,
        )
        return assessments[0].option if assessments else None

    def _continue_active_option(
        self, state: AbstractState, candidates: Sequence[ActionCandidate]
    ) -> ActionCandidate | None:
        if self._active_option is None:
            return None
        if self._active_cursor >= len(self._active_option.steps):
            self._finish_active_option(progressed=False, reason="option_exhausted")
            return None
        step = self._active_option.steps[self._active_cursor]
        matches = [
            candidate
            for candidate in candidates
            if candidate.action_name == step.action_name
        ]
        if step.structural_signature is not None:
            matches = [
                candidate
                for candidate in matches
                if self._candidate_signature(state, candidate)
                == step.structural_signature
            ]
        if len(matches) != 1:
            return None
        return matches[0]

    @staticmethod
    def _candidate_signature(
        state: AbstractState, candidate: ActionCandidate
    ) -> str | None:
        resolved = resolve_pre_action_root(state, candidate)
        if resolved.entity_id is None or not resolved.unique:
            return None
        return structural_signature(state, resolved.entity_id)

    def observe_transition(self, record: TransitionRecord) -> None:
        super().observe_transition(record)
        active = self._pending_option
        selected_step = self._pending_step
        learned = self.inducer.observe(
            record,
            selected_step=selected_step,
            active_option=active,
        )
        progressed = bool(
            record.diff.level_complete
            or record.obs_after.levels_completed > record.obs_before.levels_completed
        )
        if progressed:
            option = learned or active
            if option is not None:
                attestation = _sha(
                    {
                        "option": option.option_id,
                        "transition": _effect_signature(record),
                        "branch": self._branch_index,
                        "scope": self.attestation_scope,
                    }
                )
                self.registry.note_success(
                    option,
                    attestation,
                    scope=self.attestation_scope,
                )
                self._option_successes += 1
                self._successful_option_action_schemas.append(option.action_schemas)
            self._finish_active_option(progressed=True, reason="level_progress")
            return
        if active is None:
            return
        if record.diff.game_over:
            self._finish_active_option(progressed=False, reason="terminal")
            return
        self._active_cursor += 1
        if record.diff.is_noop:
            self._active_sterile += 1
        else:
            self._active_sterile = 0
        if self._active_sterile >= MAXIMUM_STERILE_OPTION_ACTIONS:
            self._finish_active_option(progressed=False, reason="sterile_suffix")
        elif self._active_cursor >= min(
            MAXIMUM_OPTION_HORIZON, len(active.steps)
        ):
            self._finish_active_option(progressed=False, reason="option_exhausted")

    def _finish_active_option(self, *, progressed: bool, reason: str) -> None:
        option = self._active_option
        if option is not None and not progressed:
            self.registry.note_contradiction(
                option,
                _sha(
                    {
                        "option": option.option_id,
                        "reason": str(reason),
                        "branch": self._branch_index,
                        "decision": self._decision_index,
                        "scope": self.attestation_scope,
                    }
                ),
            )
            self._option_contradictions += 1
        self._active_option = None
        self._active_cursor = 0
        self._active_sterile = 0
        self._pending_step = None
        self._pending_option = None

    def note_level_change(self) -> None:
        self._finish_active_option(progressed=True, reason="controller_level_change")
        self.inducer.start_branch()
        super().note_level_change()

    @property
    def last_decision_registry_checksum(self) -> str | None:
        """Checksum attested by the most recent transferred-program decision."""

        return self._last_decision_registry_checksum

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "requested_mode": "source_experimental_active",
                "effective_mode": "source_experimental_active",
                "phase": self.phase,
                "decision_sources": dict(self._source_counts),
                "active_option_id": (
                    None if self._active_option is None else self._active_option.option_id
                ),
                "option_successes": self._option_successes,
                "option_contradictions": self._option_contradictions,
                "registry_checksum": self.registry_checksum,
                "registry_used_in_decision": self._registry_used_in_decision,
                "successful_option_action_schemas": [
                    list(items) for items in self._successful_option_action_schemas
                ],
                "registry": self.registry.snapshot(),
                "inducer": self.inducer.summary(),
                "maximum_option_horizon": MAXIMUM_OPTION_HORIZON,
            }
        )
        return base


__all__ = [
    "DISCOVERY_WARMUP_ACTIONS",
    "EXPLORATION_ACTIONS_BETWEEN_OPTIONS",
    "FORMAT_VERSION",
    "MAXIMUM_OPTION_HORIZON",
    "REGISTRY_FORMAT_VERSION",
    "ExtendedOptionEvaluator",
    "GoalDirectedOption",
    "GoalDirectedSageTController",
    "OnlineOptionAutomatonInducer",
    "OptionAssessment",
    "OptionStep",
    "ProgressProgramRegistry",
]
