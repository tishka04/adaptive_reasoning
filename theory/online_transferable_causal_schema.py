"""Frozen causal-schema transfer with target-local epistemic safeguards.

The source side exports only terminal-grounded abstract chains:

    structural precondition -> action family/object role -> effect -> subgoal

No game identifier, palette value, coordinate, concrete action payload, grid
snapshot, or state hash is serialized.  A frozen source schema is a candidate
experiment prior on a target.  It cannot become a target policy until its
predicted effects are repeatably observed locally and a complete locally
executed chain receives target-terminal credit.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

from v3.schemas import GameObservation, ObjectInfo

from .online_multiform_relational_learner import (
    MultiformRelationPattern,
    extract_multiform_relation_patterns,
)


@dataclass(frozen=True)
class CausalEffectTemplate:
    """Palette/position-free transition effect with relative role bindings."""

    family: str
    predicate: str
    direction: str = ""
    source_binding: str = ""
    target_binding: str = ""

    @property
    def key(self) -> str:
        return repr((
            self.family,
            self.predicate,
            self.direction,
            self.source_binding,
            self.target_binding,
        ))

    @property
    def core_key(self) -> str:
        """Effect identity without a source actuator's role binding."""
        return repr((self.family, self.predicate, self.direction))

    def to_dict(self) -> Dict[str, str]:
        return {
            "family": self.family,
            "predicate": self.predicate,
            "direction": self.direction,
            "source_binding": self.source_binding,
            "target_binding": self.target_binding,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CausalEffectTemplate:
        return cls(
            family=str(payload.get("family", "")),
            predicate=str(payload.get("predicate", "")),
            direction=str(payload.get("direction", "")),
            source_binding=str(payload.get("source_binding", "")),
            target_binding=str(payload.get("target_binding", "")),
        )


@dataclass(frozen=True)
class TransferableCausalSchemaStep:
    """One abstract intervention and the effect/subgoal it predicts."""

    action_family: str
    target_role: str
    effects: Tuple[CausalEffectTemplate, ...]
    next_subgoal: str

    @property
    def key(self) -> str:
        return repr((
            self.action_family,
            self.target_role,
            tuple(effect.key for effect in self.effects),
            self.next_subgoal,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precondition": {
                "action_family_available": self.action_family,
                "target_role_present": self.target_role,
            },
            "action_family": self.action_family,
            "target_role": self.target_role,
            "effects": [effect.to_dict() for effect in self.effects],
            "next_subgoal": self.next_subgoal,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TransferableCausalSchemaStep:
        precondition = dict(payload.get("precondition", {}) or {})
        return cls(
            action_family=str(
                payload.get(
                    "action_family",
                    precondition.get("action_family_available", ""),
                )
            ),
            target_role=str(
                payload.get(
                    "target_role",
                    precondition.get("target_role_present", ""),
                )
            ),
            effects=tuple(
                CausalEffectTemplate.from_dict(dict(effect))
                for effect in tuple(payload.get("effects", ()) or ())
            ),
            next_subgoal=str(payload.get("next_subgoal", "")),
        )


@dataclass(frozen=True)
class CausalSchemaProvenance:
    """Non-behavioral provenance for auditing a frozen schema."""

    source_tag: str
    terminal_context: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "source_tag": self.source_tag,
            "terminal_context": self.terminal_context,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> CausalSchemaProvenance:
        return cls(
            source_tag=str(payload.get("source_tag", "")),
            terminal_context=str(payload.get("terminal_context", "")),
        )


@dataclass(frozen=True)
class TransferableCausalSchema:
    """Immutable terminal-grounded abstract chain."""

    schema_id: str
    steps: Tuple[TransferableCausalSchemaStep, ...]
    terminal_support: int
    provenance: Tuple[CausalSchemaProvenance, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "terminal_support": self.terminal_support,
            "steps": [step.to_dict() for step in self.steps],
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> TransferableCausalSchema:
        steps = tuple(
            TransferableCausalSchemaStep.from_dict(dict(step))
            for step in tuple(payload.get("steps", ()) or ())
        )
        if not steps:
            raise ValueError("causal schema must contain at least one step")
        computed_schema_id = _schema_id(steps)
        schema_id = str(payload.get("schema_id", "")) or computed_schema_id
        if schema_id != computed_schema_id:
            raise ValueError(
                "causal-schema content does not match its schema_id"
            )
        return cls(
            schema_id=schema_id,
            steps=steps,
            terminal_support=max(
                1,
                int(payload.get("terminal_support", 1) or 1),
            ),
            provenance=tuple(
                CausalSchemaProvenance.from_dict(dict(item))
                for item in tuple(payload.get("provenance", ()) or ())
            ),
        )


@dataclass(frozen=True)
class FrozenCausalSchemaLibrary:
    """Read-only transfer artifact produced before target evaluation."""

    schemas: Tuple[TransferableCausalSchema, ...] = ()
    format_version: str = "sage-causal-schema-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "frozen": True,
            "schema_count": len(self.schemas),
            "schemas": [schema.to_dict() for schema in self.schemas],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> FrozenCausalSchemaLibrary:
        version = str(
            payload.get("format_version", "sage-causal-schema-v1")
        )
        if version != "sage-causal-schema-v1":
            raise ValueError(
                f"unsupported causal-schema format: {version}"
            )
        if payload.get("frozen") is False:
            raise ValueError("causal-schema import must be frozen")
        schemas = tuple(
            TransferableCausalSchema.from_dict(dict(schema))
            for schema in tuple(payload.get("schemas", ()) or ())
        )
        schema_ids = [schema.schema_id for schema in schemas]
        if len(schema_ids) != len(set(schema_ids)):
            raise ValueError("causal-schema library contains duplicate ids")
        return cls(schemas=schemas, format_version=version)


@dataclass(frozen=True)
class CausalSchemaSelection:
    """One target-local schema experiment or terminal-promoted policy action."""

    action_name: str
    action_data: Dict[str, Any]
    schema_id: str
    step_index: int
    step_count: int
    source_action_family: str
    action_family: str
    target_role: str
    predicted_effects: Tuple[CausalEffectTemplate, ...]
    terminal_support: int
    local_effect_confirmations: int
    promoted: bool
    score: float
    context_signature: str
    reason: str


@dataclass
class _SchemaEvidence:
    steps: Tuple[TransferableCausalSchemaStep, ...]
    terminal_contexts: set[str] = field(default_factory=set)
    provenance: list[CausalSchemaProvenance] = field(default_factory=list)


@dataclass(frozen=True)
class _PendingSchemaSelection:
    selection: CausalSchemaSelection
    branch_index: int
    confirmation_context: str


class OnlineCausalSchemaExporter:
    """Compile bounded terminal chains and freeze them for later targets."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        source_tag: str = "online-source",
        max_steps_per_schema: int = 4,
        max_schemas: int = 64,
        max_effects_per_step: int = 12,
    ) -> None:
        self.enabled = bool(enabled)
        self.source_tag = str(source_tag)
        self.max_steps_per_schema = max(1, int(max_steps_per_schema))
        self.max_schemas = max(1, int(max_schemas))
        self.max_effects_per_step = max(1, int(max_effects_per_step))
        self._schemas: Dict[str, _SchemaEvidence] = {}
        self._branch_steps: list[TransferableCausalSchemaStep] = []
        self._branch_index = 0
        self._terminal_serial = 0
        self._transitions_abstracted = 0
        self._terminal_chains_observed = 0

    def start_branch(self) -> None:
        self._branch_index += 1
        self._branch_steps = []

    def observe_transition(
        self,
        *,
        observation_before: GameObservation,
        observation_after: GameObservation,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        terminal_success: bool,
        game_over: bool,
        patterns: Sequence[MultiformRelationPattern] | None = None,
    ) -> Tuple[CausalEffectTemplate, ...]:
        """Record one abstract effect and compile on observed terminal success."""
        if not self.enabled:
            return ()
        observed_patterns = (
            tuple(patterns)
            if patterns is not None
            else extract_multiform_relation_patterns(
                observation_before,
                observation_after,
            )
        )
        acted_exact_role = _acted_exact_role(
            observation_before,
            action_data,
        )
        effects = tuple(sorted(
            {
                _effect_template(pattern, acted_exact_role)
                for pattern in observed_patterns
            },
            key=lambda effect: effect.key,
        ))[: self.max_effects_per_step]
        if effects:
            self._transitions_abstracted += 1
            step = TransferableCausalSchemaStep(
                action_family=_action_family(action_name, action_data),
                target_role=_acted_coarse_role(
                    observation_before,
                    action_data,
                ),
                effects=effects,
                next_subgoal=_subgoal_key(effects),
            )
            if not self._branch_steps or self._branch_steps[-1].key != step.key:
                self._branch_steps.append(step)
                self._branch_steps = self._branch_steps[
                    -self.max_steps_per_schema:
                ]
        if terminal_success:
            self._compile_terminal_chain()
            self._branch_steps = []
        elif game_over:
            self._branch_steps = []
        return effects

    def freeze(
        self,
        *,
        minimum_terminal_support: int = 1,
    ) -> FrozenCausalSchemaLibrary:
        """Return an immutable snapshot; later source learning cannot mutate it."""
        support = max(1, int(minimum_terminal_support))
        schemas = []
        for schema_id, evidence in self._schemas.items():
            if len(evidence.terminal_contexts) < support:
                continue
            schemas.append(TransferableCausalSchema(
                schema_id=schema_id,
                steps=evidence.steps,
                terminal_support=len(evidence.terminal_contexts),
                provenance=tuple(evidence.provenance),
            ))
        schemas.sort(
            key=lambda schema: (
                -schema.terminal_support,
                len(schema.steps),
                schema.schema_id,
            )
        )
        return FrozenCausalSchemaLibrary(tuple(schemas[: self.max_schemas]))

    def summary(self) -> Dict[str, Any]:
        frozen = self.freeze()
        return {
            "enabled": self.enabled,
            "format_version": frozen.format_version,
            "schemas": len(self._schemas),
            "frozen_schemas": len(frozen.schemas),
            "transitions_abstracted": self._transitions_abstracted,
            "terminal_chains_observed": self._terminal_chains_observed,
            "branch_steps_pending": len(self._branch_steps),
            "contains_game_identity": False,
            "contains_palette_values": False,
            "contains_coordinates": False,
            "contains_grid_or_state_hashes": False,
        }

    def _compile_terminal_chain(self) -> None:
        if not self._branch_steps:
            return
        steps = _annotate_next_subgoals(tuple(self._branch_steps))
        schema_id = _schema_id(steps)
        terminal_context = (
            f"branch-{self._branch_index:06d}:"
            f"terminal-{self._terminal_serial:06d}"
        )
        self._terminal_serial += 1
        evidence = self._schemas.get(schema_id)
        if evidence is None:
            if len(self._schemas) >= self.max_schemas:
                return
            evidence = _SchemaEvidence(steps=steps)
            self._schemas[schema_id] = evidence
        if terminal_context not in evidence.terminal_contexts:
            evidence.terminal_contexts.add(terminal_context)
            evidence.provenance.append(CausalSchemaProvenance(
                source_tag=self.source_tag,
                terminal_context=terminal_context,
            ))
        self._terminal_chains_observed += 1


class OnlineCausalSchemaTransfer:
    """Use frozen schemas only as bounded, locally falsifiable target priors."""

    def __init__(
        self,
        library: FrozenCausalSchemaLibrary | None = None,
        *,
        enabled: bool = True,
        local_effect_confirmation_threshold: int = 2,
        max_probes_per_branch: int = 4,
        max_probes_per_schema_context: int = 1,
        nonprogress_demotion_threshold: int = 2,
        minimum_effect_match_fraction: float = 0.5,
    ) -> None:
        self.enabled = bool(enabled)
        self.library = library or FrozenCausalSchemaLibrary()
        self.local_effect_confirmation_threshold = max(
            1,
            int(local_effect_confirmation_threshold),
        )
        self.max_probes_per_branch = max(1, int(max_probes_per_branch))
        self.max_probes_per_schema_context = max(
            1,
            int(max_probes_per_schema_context),
        )
        self.nonprogress_demotion_threshold = max(
            1,
            int(nonprogress_demotion_threshold),
        )
        self.minimum_effect_match_fraction = max(
            0.0,
            min(1.0, float(minimum_effect_match_fraction)),
        )
        self._schema_by_id = {
            schema.schema_id: schema for schema in self.library.schemas
        }
        self._branch_index = 0
        self._branch_probes = 0
        self._context_probes: Counter[Tuple[str, str, int]] = Counter()
        self._pending: _PendingSchemaSelection | None = None
        self._local_confirmation_contexts: Dict[
            Tuple[str, int],
            set[str],
        ] = defaultdict(set)
        self._context_nonprogress: Counter[
            Tuple[str, str, int]
        ] = Counter()
        self._demoted_context_steps: set[Tuple[str, str, int]] = set()
        self._active_chain: Tuple[str, int] | None = None
        self._branch_matched_steps: Dict[str, set[int]] = defaultdict(set)
        self._promoted_schemas: set[str] = set()
        self._selections = 0
        self._candidate_probe_selections = 0
        self._promoted_policy_selections = 0
        self._effect_confirmations = 0
        self._effect_mismatches = 0
        self._chain_advances = 0
        self._terminal_backcredits = 0
        self._promotions = 0
        self._nonprogress_outcomes = 0
        self._unsafe_outcomes = 0
        self._demotions = 0
        self._demotion_blocks = 0
        self._branch_cap_blocks = 0
        self._context_cap_blocks = 0
        self._protected_competence_blocks = 0
        self._cross_family_adapter_probes = 0
        self._cross_family_adapter_confirmations = 0

    def start_branch(self) -> None:
        self._branch_index += 1
        self._branch_probes = 0
        self._context_probes.clear()
        self._pending = None
        self._active_chain = None
        self._branch_matched_steps.clear()

    def select(
        self,
        *,
        observation: GameObservation,
        available_actions: Sequence[str],
        available_action_candidates: Sequence[Any] | None,
        experiment_eligible: bool,
        protected_competence_available: bool = False,
    ) -> CausalSchemaSelection | None:
        """Select one bounded local test; source evidence never counts as proof."""
        if not self.enabled or not self.library.schemas:
            return None
        if protected_competence_available:
            self._protected_competence_blocks += 1
            return None
        context = _target_context_signature(observation)
        candidates = _concrete_candidates(
            observation,
            available_actions,
            available_action_candidates,
        )
        if not candidates:
            return None
        ranked = []
        for schema in self.library.schemas:
            promoted = schema.schema_id in self._promoted_schemas
            if not experiment_eligible and not promoted:
                continue
            step_indices = self._eligible_step_indices(schema)
            for step_index in step_indices:
                step = schema.steps[step_index]
                context_key = (context, schema.schema_id, step_index)
                if context_key in self._demoted_context_steps:
                    self._demotion_blocks += 1
                    continue
                if (
                    self._context_probes[context_key]
                    >= self.max_probes_per_schema_context
                ):
                    self._context_cap_blocks += 1
                    continue
                local_support = len(
                    self._local_confirmation_contexts[
                        (schema.schema_id, step_index)
                    ]
                )
                for action_name, action_data, action_family, role in candidates:
                    family_exact = action_family == step.action_family
                    role_score = _role_match_score(step.target_role, role)
                    if role_score < 0 and not (
                        not family_exact and role == "global"
                    ):
                        continue
                    role_score = max(0.0, role_score)
                    score = (
                        5.0 * float(promoted)
                        + 2.0 * float(schema.terminal_support)
                        + 1.5 * float(local_support)
                        + float(role_score)
                        + 3.0 * float(family_exact)
                        + 0.5 * float(
                            self._active_chain
                            == (schema.schema_id, step_index)
                        )
                    )
                    ranked.append((
                        score,
                        int(promoted),
                        schema.terminal_support,
                        -step_index,
                        schema.schema_id,
                        action_name,
                        repr(action_data),
                        action_data,
                        role,
                        local_support,
                        step,
                        family_exact,
                        action_family,
                    ))
        if not ranked:
            return None
        promoted_available = any(item[1] for item in ranked)
        if (
            self._branch_probes >= self.max_probes_per_branch
            and not promoted_available
        ):
            self._branch_cap_blocks += 1
            return None
        ranked.sort(reverse=True)
        (
            score,
            promoted_raw,
            terminal_support,
            negative_step_index,
            schema_id,
            action_name,
            _,
            action_data,
            role,
            local_support,
            step,
            family_exact,
            target_action_family,
        ) = ranked[0]
        step_index = -negative_step_index
        promoted = bool(promoted_raw)
        selection = CausalSchemaSelection(
            action_name=action_name,
            action_data=dict(action_data),
            schema_id=schema_id,
            step_index=step_index,
            step_count=len(self._schema_by_id[schema_id].steps),
            source_action_family=step.action_family,
            action_family=target_action_family,
            target_role=role,
            predicted_effects=step.effects,
            terminal_support=int(terminal_support),
            local_effect_confirmations=local_support,
            promoted=promoted,
            score=float(score),
            context_signature=context,
            reason=(
                "target-terminal-promoted causal schema"
                if promoted
                else (
                    (
                        "bounded cross-family actuator-adapter probe; "
                        if not family_exact
                        else "bounded frozen-schema probe; "
                    )
                    + "predicted effect must be confirmed locally before "
                    "chain authority advances"
                )
            ),
        )
        confirmation_context = (
            f"branch-{self._branch_index:06d}:{context}"
        )
        self._pending = _PendingSchemaSelection(
            selection=selection,
            branch_index=self._branch_index,
            confirmation_context=confirmation_context,
        )
        context_key = (context, schema_id, step_index)
        self._context_probes[context_key] += 1
        self._branch_probes += int(not promoted)
        self._selections += 1
        self._candidate_probe_selections += int(not promoted)
        self._promoted_policy_selections += int(promoted)
        self._cross_family_adapter_probes += int(
            not family_exact and not promoted
        )
        return selection

    def observe_transition(
        self,
        *,
        observation_before: GameObservation,
        observation_after: GameObservation,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        terminal_success: bool,
        game_over: bool,
        no_effect: bool,
        patterns: Sequence[MultiformRelationPattern] | None = None,
    ) -> Dict[str, Any]:
        """Test predicted effects and back-credit only a complete local chain."""
        pending = self._pending
        self._pending = None
        result: Dict[str, Any] = {"observed": False}
        if (
            pending is not None
            and pending.selection.action_name == str(action_name)
            and dict(pending.selection.action_data) == dict(action_data or {})
        ):
            observed_patterns = (
                tuple(patterns)
                if patterns is not None
                else extract_multiform_relation_patterns(
                    observation_before,
                    observation_after,
                )
            )
            acted_exact_role = _acted_exact_role(
                observation_before,
                action_data,
            )
            cross_family = bool(
                pending.selection.action_family
                != pending.selection.source_action_family
            )
            observed_templates = {
                _effect_template(pattern, acted_exact_role)
                for pattern in observed_patterns
            }
            observed_effects = {
                (
                    effect.core_key
                    if cross_family
                    else effect.key
                )
                for effect in observed_templates
            }
            predicted_effects = {
                (
                    effect.core_key
                    if cross_family
                    else effect.key
                )
                for effect in pending.selection.predicted_effects
            }
            overlap = predicted_effects.intersection(observed_effects)
            fraction = (
                0.0
                if not predicted_effects
                else len(overlap) / len(predicted_effects)
            )
            matched = bool(
                not no_effect
                and not (game_over and not terminal_success)
                and overlap
                and fraction >= self.minimum_effect_match_fraction
            )
            key = (
                pending.selection.schema_id,
                pending.selection.step_index,
            )
            context_key = (
                pending.selection.context_signature,
                pending.selection.schema_id,
                pending.selection.step_index,
            )
            if matched:
                contexts = self._local_confirmation_contexts[key]
                before_support = len(contexts)
                contexts.add(pending.confirmation_context)
                after_support = len(contexts)
                self._effect_confirmations += int(
                    after_support > before_support
                )
                self._cross_family_adapter_confirmations += int(
                    cross_family and after_support > before_support
                )
                self._context_nonprogress[context_key] = 0
                self._branch_matched_steps[
                    pending.selection.schema_id
                ].add(pending.selection.step_index)
                schema = self._schema_by_id[pending.selection.schema_id]
                if (
                    after_support
                    >= self.local_effect_confirmation_threshold
                    and pending.selection.step_index + 1 < len(schema.steps)
                ):
                    self._active_chain = (
                        schema.schema_id,
                        pending.selection.step_index + 1,
                    )
                    self._chain_advances += 1
            else:
                self._effect_mismatches += 1
                self._nonprogress_outcomes += 1
                self._unsafe_outcomes += int(
                    bool(game_over and not terminal_success)
                )
                self._context_nonprogress[context_key] += 1
                if (
                    self._context_nonprogress[context_key]
                    >= self.nonprogress_demotion_threshold
                    and context_key not in self._demoted_context_steps
                ):
                    self._demoted_context_steps.add(context_key)
                    self._demotions += 1
                self._active_chain = None
            result = {
                "observed": True,
                "effect_matched": matched,
                "effect_match_fraction": fraction,
                "matched_effects": len(overlap),
                "predicted_effects": len(predicted_effects),
                "schema_id": pending.selection.schema_id,
                "step_index": pending.selection.step_index,
                "local_effect_confirmations": len(
                    self._local_confirmation_contexts[key]
                ),
            }
        if terminal_success:
            promoted = self._backcredit_complete_chains()
            result["terminal_backcredited_schemas"] = promoted
            self._active_chain = None
        elif game_over:
            self._active_chain = None
            self._branch_matched_steps.clear()
        return result

    def cancel_pending(self) -> None:
        """Cancel a vetoed action without fabricating target evidence."""
        pending = self._pending
        if pending is None:
            return
        selection = pending.selection
        context_key = (
            selection.context_signature,
            selection.schema_id,
            selection.step_index,
        )
        self._context_probes[context_key] = max(
            0,
            self._context_probes[context_key] - 1,
        )
        self._branch_probes = max(
            0,
            self._branch_probes - int(not selection.promoted),
        )
        self._selections = max(0, self._selections - 1)
        if selection.promoted:
            self._promoted_policy_selections = max(
                0,
                self._promoted_policy_selections - 1,
            )
        else:
            self._candidate_probe_selections = max(
                0,
                self._candidate_probe_selections - 1,
            )
        self._pending = None

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "library_format_version": self.library.format_version,
            "source_schemas": len(self.library.schemas),
            "library_frozen": True,
            "local_effect_confirmation_threshold": (
                self.local_effect_confirmation_threshold
            ),
            "max_probes_per_branch": self.max_probes_per_branch,
            "max_probes_per_schema_context": (
                self.max_probes_per_schema_context
            ),
            "selections": self._selections,
            "candidate_probe_selections": self._candidate_probe_selections,
            "promoted_policy_selections": self._promoted_policy_selections,
            "effect_confirmations": self._effect_confirmations,
            "effect_mismatches": self._effect_mismatches,
            "locally_confirmed_steps": sum(
                len(contexts) >= self.local_effect_confirmation_threshold
                for contexts in self._local_confirmation_contexts.values()
            ),
            "chain_advances": self._chain_advances,
            "terminal_backcredits": self._terminal_backcredits,
            "promotions": self._promotions,
            "promoted_schemas": len(self._promoted_schemas),
            "nonprogress_outcomes": self._nonprogress_outcomes,
            "unsafe_outcomes": self._unsafe_outcomes,
            "demotions": self._demotions,
            "demotion_blocks": self._demotion_blocks,
            "demoted_context_steps": len(self._demoted_context_steps),
            "branch_cap_blocks": self._branch_cap_blocks,
            "context_cap_blocks": self._context_cap_blocks,
            "protected_competence_blocks": (
                self._protected_competence_blocks
            ),
            "cross_family_adapter_probes": (
                self._cross_family_adapter_probes
            ),
            "cross_family_adapter_confirmations": (
                self._cross_family_adapter_confirmations
            ),
            "source_evidence_grants_policy_authority": False,
            "promotion_requires_target_terminal": True,
        }

    def _eligible_step_indices(
        self,
        schema: TransferableCausalSchema,
    ) -> Tuple[int, ...]:
        if self._active_chain is not None:
            schema_id, step_index = self._active_chain
            if schema_id == schema.schema_id:
                return (step_index,)
        if schema.schema_id in self._promoted_schemas:
            matched = self._branch_matched_steps[schema.schema_id]
            for step_index in range(len(schema.steps)):
                if step_index not in matched:
                    return (step_index,)
        trusted_prefix = 0
        for step_index in range(len(schema.steps)):
            if step_index == 0:
                trusted_prefix = 0
                continue
            previous_support = len(
                self._local_confirmation_contexts[
                    (schema.schema_id, step_index - 1)
                ]
            )
            if previous_support < self.local_effect_confirmation_threshold:
                break
            trusted_prefix = step_index
        return (trusted_prefix,)

    def _backcredit_complete_chains(self) -> Tuple[str, ...]:
        promoted = []
        for schema_id, matched_steps in self._branch_matched_steps.items():
            schema = self._schema_by_id.get(schema_id)
            if schema is None:
                continue
            if matched_steps != set(range(len(schema.steps))):
                continue
            if schema_id not in self._promoted_schemas:
                self._promoted_schemas.add(schema_id)
                self._promotions += 1
            self._terminal_backcredits += 1
            promoted.append(schema_id)
        return tuple(sorted(promoted))


def _effect_template(
    pattern: MultiformRelationPattern,
    acted_exact_role: str,
) -> CausalEffectTemplate:
    def binding(role: str) -> str:
        if not role:
            return ""
        if acted_exact_role and role == acted_exact_role:
            return "acted_object"
        return "other_object"

    return CausalEffectTemplate(
        family=pattern.family,
        predicate=pattern.predicate,
        direction=pattern.direction,
        source_binding=binding(pattern.source_role),
        target_binding=binding(pattern.target_role),
    )


def _action_family(
    action_name: str,
    action_data: Mapping[str, Any] | None,
) -> str:
    data = dict(action_data or {})
    if "x" in data and "y" in data:
        return "point"
    argument_schema = tuple(sorted(str(key) for key in data))
    if argument_schema:
        return f"parameterized:{str(action_name)}:{','.join(argument_schema)}"
    return f"primitive:{str(action_name)}"


def _acted_exact_role(
    observation: GameObservation,
    action_data: Mapping[str, Any] | None,
) -> str:
    obj = _acted_object(observation, action_data)
    return "" if obj is None else _exact_object_role(obj)


def _acted_coarse_role(
    observation: GameObservation,
    action_data: Mapping[str, Any] | None,
) -> str:
    data = dict(action_data or {})
    if "x" not in data or "y" not in data:
        return "global"
    obj = _acted_object(observation, data)
    if obj is None:
        return "background"
    height = int(obj.bbox[2] - obj.bbox[0] + 1)
    width = int(obj.bbox[3] - obj.bbox[1] + 1)
    area = int(obj.area)
    area_bucket = (
        "single"
        if area == 1
        else "small"
        if area <= 4
        else "medium"
        if area <= 15
        else "large"
    )
    aspect = (
        "square"
        if height == width
        else "vertical"
        if height > width
        else "horizontal"
    )
    density = area / max(1, height * width)
    topology = (
        "solid"
        if density >= 0.9
        else "sparse"
        if density <= 0.5
        else "mixed"
    )
    return f"object:{area_bucket}:{aspect}:{topology}"


def _acted_object(
    observation: GameObservation,
    action_data: Mapping[str, Any] | None,
) -> ObjectInfo | None:
    data = dict(action_data or {})
    if "x" not in data or "y" not in data:
        return None
    try:
        x = int(data["x"])
        y = int(data["y"])
    except (TypeError, ValueError):
        return None
    containing = [
        obj for obj in observation.objects if (y, x) in set(obj.cells)
    ]
    if not containing:
        return None
    return min(containing, key=lambda obj: (obj.area, obj.object_id))


def _exact_object_role(obj: ObjectInfo) -> str:
    height = int(obj.bbox[2] - obj.bbox[0] + 1)
    width = int(obj.bbox[3] - obj.bbox[1] + 1)
    area_bucket = (
        "single"
        if obj.area == 1
        else "small"
        if obj.area <= 4
        else "medium"
        if obj.area <= 15
        else "large"
    )
    min_row = min(row for row, _ in obj.cells)
    min_col = min(col for _, col in obj.cells)
    normalized = tuple(sorted(
        (int(row - min_row), int(col - min_col))
        for row, col in obj.cells
    ))
    shape_key = hashlib.sha1(
        repr(normalized).encode("utf-8")
    ).hexdigest()[:12]
    payload = (area_bucket, min(height, 7), min(width, 7), shape_key)
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _concrete_candidates(
    observation: GameObservation,
    available_actions: Sequence[str],
    raw_candidates: Sequence[Any] | None,
) -> Tuple[Tuple[str, Dict[str, Any], str, str], ...]:
    allowed = {
        str(action)
        for action in available_actions
        if str(action) and str(action) != "RESET"
    }
    concrete: list[Tuple[str, Dict[str, Any]]] = []
    for raw in tuple(raw_candidates or ()):
        name = str(getattr(raw, "name", ""))
        if name not in allowed:
            continue
        concrete.append((
            name,
            dict(getattr(raw, "action_args", {}) or {}),
        ))
    represented = {name for name, _ in concrete}
    concrete.extend((name, {}) for name in sorted(allowed - represented))
    seen = set()
    result = []
    for name, data in concrete:
        stable = tuple(sorted(
            (str(key), repr(value)) for key, value in data.items()
        ))
        if (name, stable) in seen:
            continue
        seen.add((name, stable))
        result.append((
            name,
            data,
            _action_family(name, data),
            _acted_coarse_role(observation, data),
        ))
    return tuple(result)


def _role_match_score(source_role: str, target_role: str) -> float:
    if source_role == target_role:
        return 3.0
    if source_role.startswith("object:") and target_role.startswith("object:"):
        source_parts = source_role.split(":")
        target_parts = target_role.split(":")
        return 1.0 + 0.5 * sum(
            left == right
            for left, right in zip(source_parts[1:], target_parts[1:])
        )
    if source_role in {"global", "background"}:
        return 1.0 if source_role == target_role else -1.0
    return -1.0


def _subgoal_key(effects: Sequence[CausalEffectTemplate]) -> str:
    payload = tuple(effect.key for effect in effects)
    return "effect::" + hashlib.sha1(
        repr(payload).encode("utf-8")
    ).hexdigest()[:16]


def _annotate_next_subgoals(
    steps: Tuple[TransferableCausalSchemaStep, ...],
) -> Tuple[TransferableCausalSchemaStep, ...]:
    result = []
    for index, step in enumerate(steps):
        next_subgoal = (
            "terminal_progress"
            if index + 1 >= len(steps)
            else _subgoal_key(steps[index + 1].effects)
        )
        result.append(TransferableCausalSchemaStep(
            action_family=step.action_family,
            target_role=step.target_role,
            effects=step.effects,
            next_subgoal=next_subgoal,
        ))
    return tuple(result)


def _schema_id(
    steps: Sequence[TransferableCausalSchemaStep],
) -> str:
    payload = tuple(step.key for step in steps)
    return "causal-schema::" + hashlib.sha1(
        repr(payload).encode("utf-8")
    ).hexdigest()[:16]


def _target_context_signature(observation: GameObservation) -> str:
    roles = Counter(
        _coarse_object_role(obj) for obj in observation.objects
    )
    payload = (
        tuple(int(value) for value in observation.raw_grid.shape),
        tuple(sorted(roles.items())),
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _coarse_object_role(obj: ObjectInfo) -> str:
    height = int(obj.bbox[2] - obj.bbox[0] + 1)
    width = int(obj.bbox[3] - obj.bbox[1] + 1)
    area = int(obj.area)
    area_bucket = (
        "single"
        if area == 1
        else "small"
        if area <= 4
        else "medium"
        if area <= 15
        else "large"
    )
    aspect = (
        "square"
        if height == width
        else "vertical"
        if height > width
        else "horizontal"
    )
    return f"object:{area_bucket}:{aspect}"


__all__ = [
    "CausalEffectTemplate",
    "CausalSchemaProvenance",
    "CausalSchemaSelection",
    "FrozenCausalSchemaLibrary",
    "OnlineCausalSchemaExporter",
    "OnlineCausalSchemaTransfer",
    "TransferableCausalSchema",
    "TransferableCausalSchemaStep",
]
