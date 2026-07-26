"""Terminally grounded online learning over multiple relation families.

SAGE.9w generalizes the stencil-specific terminal learner to the structured
scene representation already produced by the unified controller.  It induces
palette- and position-invariant patterns for:

* object correspondence;
* attribute/shape/scale transformation;
* object counting, appearance, disappearance, and exhaustion;
* object trajectories;
* spatial relations that appear or break.

Patterns are hypotheses after ordinary transitions.  A pattern can authorize
an action only after the same relation family has occurred on terminally
successful transitions in independent online branches.  No game identity,
level index, reward oracle, or answer trace is available to the learner.

SAGE.10a can also ground the patterns caused by a productive frontier action
when the terminal arrives several actions later in the same branch.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Mapping, Sequence, Tuple

from v3.schemas import GameObservation, ObjectInfo


@dataclass(frozen=True)
class MultiformRelationPattern:
    """One abstract relation observed across a live transition."""

    family: str
    signature: str
    source_role: str = ""
    target_role: str = ""
    predicate: str = ""
    direction: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "signature": self.signature,
            "source_role": self.source_role,
            "target_role": self.target_role,
            "predicate": self.predicate,
            "direction": self.direction,
        }


@dataclass
class MultiformPatternEvidence:
    pattern: MultiformRelationPattern
    observations: int = 0
    terminal_contexts: set[str] = field(default_factory=set)
    terminal_levels: set[int] = field(default_factory=set)
    action_names: Counter[str] = field(default_factory=Counter)
    actuator_signatures: Counter[str] = field(default_factory=Counter)

    def to_dict(self, *, minimum_terminal_support: int) -> Dict[str, Any]:
        return {
            **self.pattern.to_dict(),
            "observations": self.observations,
            "terminal_confirmations": len(self.terminal_contexts),
            "terminal_levels": sorted(self.terminal_levels),
            "confirmed": (
                len(self.terminal_contexts) >= minimum_terminal_support
            ),
            "action_names": dict(self.action_names),
            "actuator_signatures": dict(self.actuator_signatures),
        }


@dataclass
class MultiformActuatorModel:
    actuator_signature: str
    action_name: str
    target_role: str
    observations: int = 0
    terminal_outcomes: int = 0
    unsafe_outcomes: int = 0
    pattern_counts: Counter[str] = field(default_factory=Counter)
    concrete_action_keys: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _DelayedFrontierPatternEligibility:
    eligibility_id: str
    branch_index: int
    source_level: int
    actuator_signature: str
    pattern_signatures: Tuple[str, ...]


@dataclass(frozen=True)
class _PendingMultiformSelection:
    action_name: str
    context_signature: str
    pattern_signatures: Tuple[str, ...]


@dataclass(frozen=True)
class MultiformRelationSelection:
    """One live action supported by terminally grounded relation families."""

    action_name: str
    action_data: Dict[str, Any]
    actuator_signature: str
    target_role: str
    predicted_pattern_signatures: Tuple[str, ...]
    predicted_families: Tuple[str, ...]
    terminal_support: int
    score: float
    reason: str


class OnlineMultiformRelationalLearner:
    """Learn and reuse terminal relations over generic scene structure."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_terminal_support: int = 2,
        maximum_patterns_per_transition: int = 32,
        max_selections_per_branch: int = 2,
        max_selections_per_context: int = 2,
        nonprogress_demotion_threshold: int = 2,
    ) -> None:
        self.enabled = bool(enabled)
        self.minimum_terminal_support = max(
            1,
            int(minimum_terminal_support),
        )
        self.maximum_patterns_per_transition = max(
            1,
            int(maximum_patterns_per_transition),
        )
        self.max_selections_per_branch = max(
            1,
            int(max_selections_per_branch),
        )
        self.max_selections_per_context = max(
            1,
            int(max_selections_per_context),
        )
        self.nonprogress_demotion_threshold = max(
            1,
            int(nonprogress_demotion_threshold),
        )
        self._patterns: Dict[str, MultiformPatternEvidence] = {}
        self._actuators: Dict[str, MultiformActuatorModel] = {}
        self._branch_index = 0
        self._branch_selections = 0
        self._context_selections: Counter[str] = Counter()
        self._context_nonprogress: Counter[str] = Counter()
        self._context_pattern_support: Dict[str, set[str]] = {}
        self._demoted_context_versions: Dict[
            str,
            Dict[str, int],
        ] = {}
        self._pattern_terminal_versions: Counter[str] = Counter()
        self._pending_selection: _PendingMultiformSelection | None = None
        self._observations = 0
        self._terminal_examples = 0
        self._patterns_observed = 0
        self._terminal_pattern_credits = 0
        self._selections = 0
        self._transferred_selections = 0
        self._unsafe_model_blocks = 0
        self._branch_cap_blocks = 0
        self._context_cap_blocks = 0
        self._nonprogress_outcomes = 0
        self._productive_outcomes = 0
        self._demotions = 0
        self._demotion_blocks = 0
        self._reactivations = 0
        self._level_gating_blocks = 0
        self._protected_route_reactivations = 0
        self._family_counts: Counter[str] = Counter()
        self._terminal_family_counts: Counter[str] = Counter()
        self._selection_family_counts: Counter[str] = Counter()
        self._delayed_frontier_eligibilities: Dict[
            str,
            _DelayedFrontierPatternEligibility,
        ] = {}
        self._delayed_frontier_eligibilities_registered = 0
        self._delayed_frontier_eligibilities_credited = 0
        self._delayed_frontier_pattern_credits = 0
        self._delayed_frontier_credit_branches: set[int] = set()
        self._delayed_frontier_eligibilities_expired = 0
        self._delayed_frontier_eligibilities_discarded = 0

    def start_branch(self) -> None:
        self._branch_index += 1
        self._branch_selections = 0
        self._context_selections.clear()
        self._context_nonprogress.clear()
        self._pending_selection = None

    def observe_transition(
        self,
        *,
        observation_before: GameObservation,
        observation_after: GameObservation,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        terminal_success: bool,
        game_over: bool,
        delayed_frontier_eligibility_id: str = "",
    ) -> Tuple[MultiformRelationPattern, ...]:
        """Induce patterns; grant goal authority only on terminal success."""
        if not self.enabled:
            return ()
        patterns = extract_multiform_relation_patterns(
            observation_before,
            observation_after,
        )[: self.maximum_patterns_per_transition]
        context_signature = _selection_context_signature(
            observation_before
        )
        context_support = self._context_pattern_support.setdefault(
            context_signature,
            set(),
        )
        observed_signatures = {
            pattern.signature for pattern in patterns
        }
        novel_context_effect = bool(
            observed_signatures.difference(context_support)
        )
        pending = self._pending_selection
        if (
            pending is not None
            and pending.action_name == str(action_name)
            and pending.context_signature == context_signature
        ):
            productive = bool(terminal_success or novel_context_effect)
            if productive:
                self._productive_outcomes += 1
                self._context_nonprogress[context_signature] = 0
            else:
                self._nonprogress_outcomes += 1
                self._context_nonprogress[context_signature] += 1
                if (
                    self._context_nonprogress[context_signature]
                    >= self.nonprogress_demotion_threshold
                    and context_signature
                    not in self._demoted_context_versions
                ):
                    self._demoted_context_versions[context_signature] = {
                        signature: self._pattern_terminal_versions[
                            signature
                        ]
                        for signature in pending.pattern_signatures
                    }
                    self._demotions += 1
            self._pending_selection = None
        elif pending is not None:
            self._pending_selection = None
        context_support.update(observed_signatures)
        target_role = _acted_role(observation_before, action_data)
        actuator_signature = _actuator_signature(
            str(action_name),
            target_role,
            action_data,
        )
        model = self._actuators.get(actuator_signature)
        if model is None:
            model = MultiformActuatorModel(
                actuator_signature=actuator_signature,
                action_name=str(action_name),
                target_role=target_role,
            )
            self._actuators[actuator_signature] = model
        model.observations += 1
        model.terminal_outcomes += int(bool(terminal_success))
        model.unsafe_outcomes += int(
            bool(game_over and not terminal_success)
        )
        model.concrete_action_keys.add(
            _concrete_action_key(str(action_name), action_data)
        )

        context = f"branch-{self._branch_index}"
        source_level = int(observation_before.levels_completed)
        credited = 0
        for pattern in patterns:
            evidence = self._patterns.get(pattern.signature)
            if evidence is None:
                evidence = MultiformPatternEvidence(pattern=pattern)
                self._patterns[pattern.signature] = evidence
            evidence.observations += 1
            evidence.action_names[str(action_name)] += 1
            evidence.actuator_signatures[actuator_signature] += 1
            model.pattern_counts[pattern.signature] += 1
            self._family_counts[pattern.family] += 1
            if terminal_success:
                before = len(evidence.terminal_contexts)
                evidence.terminal_contexts.add(context)
                evidence.terminal_levels.add(source_level)
                if len(evidence.terminal_contexts) > before:
                    credited += 1
                    self._pattern_terminal_versions[
                        pattern.signature
                    ] += 1
                    self._terminal_family_counts[pattern.family] += 1
        self._observations += 1
        self._patterns_observed += len(patterns)
        self._terminal_pattern_credits += credited
        self._terminal_examples += int(bool(terminal_success))
        eligibility_id = str(delayed_frontier_eligibility_id)
        if (
            eligibility_id
            and patterns
            and not terminal_success
            and not game_over
        ):
            self._delayed_frontier_eligibilities[eligibility_id] = (
                _DelayedFrontierPatternEligibility(
                    eligibility_id=eligibility_id,
                    branch_index=self._branch_index,
                    source_level=source_level,
                    actuator_signature=actuator_signature,
                    pattern_signatures=tuple(
                        pattern.signature for pattern in patterns
                    ),
                )
            )
            self._delayed_frontier_eligibilities_registered += 1
        return patterns

    def resolve_delayed_frontier_credit(
        self,
        *,
        credited_eligibility_ids: Sequence[str] = (),
        expired_eligibility_ids: Sequence[str] = (),
        discarded_eligibility_ids: Sequence[str] = (),
    ) -> Dict[str, int]:
        """Ground earlier frontier effects in a later branch terminal."""
        credited_patterns = 0
        credited_records = 0
        for raw_id in credited_eligibility_ids:
            eligibility_id = str(raw_id)
            record = self._delayed_frontier_eligibilities.pop(
                eligibility_id,
                None,
            )
            if record is None:
                continue
            context = f"branch-{record.branch_index}"
            record_credits = 0
            for signature in record.pattern_signatures:
                evidence = self._patterns.get(signature)
                if evidence is None:
                    continue
                before = len(evidence.terminal_contexts)
                evidence.terminal_contexts.add(context)
                evidence.terminal_levels.add(record.source_level)
                if len(evidence.terminal_contexts) > before:
                    record_credits += 1
                    self._pattern_terminal_versions[signature] += 1
                    self._terminal_family_counts[
                        evidence.pattern.family
                    ] += 1
            model = self._actuators.get(record.actuator_signature)
            if model is not None:
                model.terminal_outcomes += 1
            credited_patterns += record_credits
            credited_records += 1
            self._delayed_frontier_credit_branches.add(
                record.branch_index
            )
        expired = self._discard_delayed_frontier_eligibilities(
            expired_eligibility_ids
        )
        discarded = self._discard_delayed_frontier_eligibilities(
            discarded_eligibility_ids
        )
        self._delayed_frontier_eligibilities_credited += (
            credited_records
        )
        self._delayed_frontier_pattern_credits += credited_patterns
        self._terminal_pattern_credits += credited_patterns
        self._delayed_frontier_eligibilities_expired += expired
        self._delayed_frontier_eligibilities_discarded += discarded
        return {
            "credited_eligibilities": credited_records,
            "credited_patterns": credited_patterns,
            "expired_eligibilities": expired,
            "discarded_eligibilities": discarded,
        }

    def select(
        self,
        *,
        observation: GameObservation,
        available_actions: Sequence[str],
        available_action_candidates: Sequence[Any] | None,
    ) -> MultiformRelationSelection | None:
        """Choose a concrete action whose learned effects match terminal rules."""
        if not self.enabled:
            return None
        if self._branch_selections >= self.max_selections_per_branch:
            self._branch_cap_blocks += 1
            return None
        context_signature = _selection_context_signature(observation)
        if (
            self._context_selections[context_signature]
            >= self.max_selections_per_context
        ):
            self._context_cap_blocks += 1
            return None
        demoted_versions = self._demoted_context_versions.get(
            context_signature
        )
        if demoted_versions is not None:
            new_terminal_support = any(
                self._pattern_terminal_versions[signature] > version
                for signature, version in demoted_versions.items()
            )
            if new_terminal_support:
                self._demoted_context_versions.pop(
                    context_signature,
                    None,
                )
                self._context_nonprogress[context_signature] = 0
                self._reactivations += 1
            else:
                self._demotion_blocks += 1
                return None
        confirmed = {
            signature: evidence
            for signature, evidence in self._patterns.items()
            if len(evidence.terminal_contexts)
            >= self.minimum_terminal_support
        }
        if not confirmed:
            return None
        current_level = int(observation.levels_completed)
        context_support = self._context_pattern_support.get(
            context_signature,
            set(),
        )
        level_eligible = {
            signature: evidence
            for signature, evidence in confirmed.items()
            if (
                not evidence.terminal_levels
                or current_level in evidence.terminal_levels
                or signature in context_support
            )
        }
        self._level_gating_blocks += len(confirmed) - len(level_eligible)
        if not level_eligible:
            return None
        candidates = _concrete_candidates(
            observation,
            available_actions,
            available_action_candidates,
        )
        ranked = []
        for action_name, action_data, actuator, target_role in candidates:
            model = self._actuators.get(actuator)
            if model is None or model.observations <= 0:
                continue
            if (
                model.unsafe_outcomes > 0
                and model.unsafe_outcomes >= model.terminal_outcomes
            ):
                self._unsafe_model_blocks += 1
                continue
            supported = [
                (signature, level_eligible[signature])
                for signature in model.pattern_counts
                if signature in level_eligible
            ]
            if not supported:
                continue
            terminal_support = sum(
                len(evidence.terminal_contexts)
                for _, evidence in supported
            )
            pattern_support = sum(
                model.pattern_counts[signature]
                for signature, _ in supported
            )
            families = tuple(sorted({
                evidence.pattern.family
                for _, evidence in supported
            }))
            score = (
                4.0 * terminal_support
                + pattern_support / max(1, model.observations)
                + len(families)
                - 3.0 * model.unsafe_outcomes
            )
            ranked.append((
                score,
                terminal_support,
                pattern_support,
                action_name,
                repr(action_data),
                action_data,
                actuator,
                target_role,
                _concrete_action_key(action_name, action_data),
                tuple(sorted(signature for signature, _ in supported)),
                families,
            ))
        if not ranked:
            return None
        ranked.sort(reverse=True)
        (
            score,
            terminal_support,
            _,
            action_name,
            _,
            action_data,
            actuator,
            target_role,
            concrete_action_key,
            signatures,
            families,
        ) = ranked[0]
        self._selections += 1
        self._branch_selections += 1
        self._context_selections[context_signature] += 1
        self._pending_selection = _PendingMultiformSelection(
            action_name=action_name,
            context_signature=context_signature,
            pattern_signatures=signatures,
        )
        transferred = bool(
            concrete_action_key
            not in self._actuators[actuator].concrete_action_keys
        )
        self._transferred_selections += int(transferred)
        for family in families:
            self._selection_family_counts[family] += 1
        return MultiformRelationSelection(
            action_name=action_name,
            action_data=dict(action_data),
            actuator_signature=actuator,
            target_role=target_role,
            predicted_pattern_signatures=signatures,
            predicted_families=families,
            terminal_support=int(terminal_support),
            score=float(score),
            reason=(
                "terminally confirmed multi-form relation policy: "
                + ",".join(families)
            ),
        )

    def cancel_pending_selection(self) -> None:
        """Withdraw a guarded selection without charging its outcome."""
        self._pending_selection = None

    def note_protected_route_failure(
        self,
        observation: GameObservation,
    ) -> None:
        """Re-enable this context after exact competence is refuted/diverges."""
        context_signature = _selection_context_signature(observation)
        changed = bool(
            context_signature in self._demoted_context_versions
            or self._context_nonprogress[context_signature]
        )
        self._demoted_context_versions.pop(context_signature, None)
        self._context_nonprogress[context_signature] = 0
        self._context_selections[context_signature] = 0
        if changed:
            self._protected_route_reactivations += 1

    def confirmed_patterns(self) -> Tuple[MultiformRelationPattern, ...]:
        return tuple(
            evidence.pattern
            for evidence in self._patterns.values()
            if len(evidence.terminal_contexts)
            >= self.minimum_terminal_support
        )

    def summary(self) -> Dict[str, Any]:
        confirmed = self.confirmed_patterns()
        return {
            "enabled": self.enabled,
            "minimum_terminal_support": self.minimum_terminal_support,
            "max_selections_per_branch": self.max_selections_per_branch,
            "max_selections_per_context": self.max_selections_per_context,
            "nonprogress_demotion_threshold": (
                self.nonprogress_demotion_threshold
            ),
            "observations": self._observations,
            "terminal_examples": self._terminal_examples,
            "patterns_observed": self._patterns_observed,
            "pattern_hypotheses": len(self._patterns),
            "confirmed_patterns": len(confirmed),
            "confirmed_families": sorted({
                pattern.family for pattern in confirmed
            }),
            "actuator_models": len(self._actuators),
            "terminal_pattern_credits": self._terminal_pattern_credits,
            "selections": self._selections,
            "transferred_selections": self._transferred_selections,
            "unsafe_model_blocks": self._unsafe_model_blocks,
            "branch_cap_blocks": self._branch_cap_blocks,
            "context_cap_blocks": self._context_cap_blocks,
            "productive_selection_outcomes": self._productive_outcomes,
            "nonprogress_selection_outcomes": self._nonprogress_outcomes,
            "demotions": self._demotions,
            "demotion_blocks": self._demotion_blocks,
            "reactivations": self._reactivations,
            "level_gating_blocks": self._level_gating_blocks,
            "protected_route_reactivations": (
                self._protected_route_reactivations
            ),
            "demoted_contexts": len(self._demoted_context_versions),
            "delayed_frontier_eligibilities_registered": (
                self._delayed_frontier_eligibilities_registered
            ),
            "delayed_frontier_eligibilities_pending": len(
                self._delayed_frontier_eligibilities
            ),
            "delayed_frontier_eligibilities_credited": (
                self._delayed_frontier_eligibilities_credited
            ),
            "delayed_frontier_pattern_credits": (
                self._delayed_frontier_pattern_credits
            ),
            "delayed_frontier_credit_branches": len(
                self._delayed_frontier_credit_branches
            ),
            "delayed_frontier_eligibilities_expired": (
                self._delayed_frontier_eligibilities_expired
            ),
            "delayed_frontier_eligibilities_discarded": (
                self._delayed_frontier_eligibilities_discarded
            ),
            "family_observations": dict(self._family_counts),
            "terminal_family_credits": dict(
                self._terminal_family_counts
            ),
            "selection_families": dict(
                self._selection_family_counts
            ),
            "patterns": [
                evidence.to_dict(
                    minimum_terminal_support=(
                        self.minimum_terminal_support
                    ),
                )
                for evidence in self._patterns.values()
            ],
        }

    def _discard_delayed_frontier_eligibilities(
        self,
        eligibility_ids: Sequence[str],
    ) -> int:
        discarded = 0
        for raw_id in eligibility_ids:
            if self._delayed_frontier_eligibilities.pop(
                str(raw_id),
                None,
            ) is not None:
                discarded += 1
        return discarded


def extract_multiform_relation_patterns(
    before: GameObservation,
    after: GameObservation,
) -> Tuple[MultiformRelationPattern, ...]:
    """Describe one transition with generic, composable relation families."""
    patterns: Dict[str, MultiformRelationPattern] = {}
    before_objects = tuple(before.objects)
    after_objects = tuple(after.objects)
    matched, removed, appeared = _match_objects(
        before_objects,
        after_objects,
    )

    before_roles = Counter(_object_role(obj) for obj in before_objects)
    after_roles = Counter(_object_role(obj) for obj in after_objects)
    for role in sorted(set(before_roles) | set(after_roles)):
        delta = after_roles[role] - before_roles[role]
        if not delta:
            continue
        direction = "increase" if delta > 0 else "decrease"
        if after_roles[role] == 0:
            direction = "exhaust"
        pattern = _pattern(
            family="count",
            predicate=f"object_count_{direction}",
            source_role=role,
            direction=_magnitude_bucket(abs(delta)),
        )
        patterns[pattern.signature] = pattern

    for obj in removed:
        role = _object_role(obj)
        pattern = _pattern(
            family="disappearance",
            predicate="object_disappeared",
            source_role=role,
        )
        patterns[pattern.signature] = pattern
    for obj in appeared:
        role = _object_role(obj)
        pattern = _pattern(
            family="appearance",
            predicate="object_appeared",
            target_role=role,
        )
        patterns[pattern.signature] = pattern

    for source, target in matched:
        source_role = _object_role(source)
        target_role = _object_role(target)
        shape_same = _shape_key(source) == _shape_key(target)
        value_same = int(source.value) == int(target.value)
        dy = int(round(target.center[0] - source.center[0]))
        dx = int(round(target.center[1] - source.center[1]))
        if dy or dx:
            direction = _direction(dy, dx)
            trajectory = _pattern(
                family="trajectory",
                predicate="object_moved",
                source_role=source_role,
                target_role=target_role,
                direction=direction,
            )
            patterns[trajectory.signature] = trajectory
        if not shape_same or not value_same or source.area != target.area:
            if shape_same and source.area == target.area and not value_same:
                predicate = "attribute_converted"
            elif source.area != target.area:
                predicate = "scale_transformed"
            else:
                predicate = "shape_transformed"
            transformed = _pattern(
                family="transformation",
                predicate=predicate,
                source_role=source_role,
                target_role=target_role,
            )
            patterns[transformed.signature] = transformed
        if (dy or dx) or not shape_same or not value_same:
            status = (
                "moved"
                if (dy or dx) and shape_same and value_same
                else "transformed"
            )
            correspondence = _pattern(
                family="correspondence",
                predicate=f"object_correspondence_{status}",
                source_role=source_role,
                target_role=target_role,
            )
            patterns[correspondence.signature] = correspondence

    before_spatial = _spatial_facts(before_objects)
    after_spatial = _spatial_facts(after_objects)
    for fact in sorted(set(before_spatial) | set(after_spatial)):
        delta = after_spatial[fact] - before_spatial[fact]
        if not delta:
            continue
        relation, source_role, target_role = fact
        spatial = _pattern(
            family="spatial",
            predicate=relation,
            source_role=source_role,
            target_role=target_role,
            direction="appeared" if delta > 0 else "broken",
        )
        patterns[spatial.signature] = spatial

    return tuple(sorted(
        patterns.values(),
        key=lambda pattern: (pattern.family, pattern.signature),
    ))


def _match_objects(
    before: Sequence[ObjectInfo],
    after: Sequence[ObjectInfo],
) -> Tuple[
    Tuple[Tuple[ObjectInfo, ObjectInfo], ...],
    Tuple[ObjectInfo, ...],
    Tuple[ObjectInfo, ...],
]:
    before_descriptors = tuple(
        _match_descriptor(obj)
        for obj in before
    )
    after_descriptors = tuple(
        _match_descriptor(obj)
        for obj in after
    )
    matched_indices, removed_indices, appeared_indices = (
        _match_object_indices(before_descriptors, after_descriptors)
    )
    return (
        tuple((before[left], after[right]) for left, right in matched_indices),
        tuple(before[index] for index in removed_indices),
        tuple(after[index] for index in appeared_indices),
    )


def _match_descriptor(obj: ObjectInfo) -> Tuple[Any, ...]:
    return (
        int(obj.object_id),
        _shape_key(obj),
        int(obj.area),
        float(obj.center[0]),
        float(obj.center[1]),
        int(obj.value),
    )


@lru_cache(maxsize=4_096)
def _match_object_indices(
    before: Tuple[Tuple[Any, ...], ...],
    after: Tuple[Tuple[Any, ...], ...],
) -> Tuple[
    Tuple[Tuple[int, int], ...],
    Tuple[int, ...],
    Tuple[int, ...],
]:
    scored = []
    diagonal = 1.0
    all_objects = tuple(before) + tuple(after)
    if all_objects:
        diagonal = max(
            1.0,
            math.sqrt(max(
                (obj[3] ** 2 + obj[4] ** 2)
                for obj in all_objects
            )),
        )
    for source_index, source in enumerate(before):
        for target_index, target in enumerate(after):
            shape_same = source[1] == target[1]
            area_similarity = min(source[2], target[2]) / max(
                1,
                max(source[2], target[2]),
            )
            distance = math.dist(
                (source[3], source[4]),
                (target[3], target[4]),
            ) / diagonal
            score = (
                4.0 * float(shape_same)
                + 2.0 * area_similarity
                + 0.25 * float(source[5] == target[5])
                - distance
            )
            scored.append((
                score,
                source_index,
                target_index,
                source[0],
                target[0],
            ))
    scored.sort(
        key=lambda item: (
            item[0],
            -item[3],
            -item[4],
        ),
        reverse=True,
    )
    used_before = set()
    used_after = set()
    matched = []
    for score, source_index, target_index, _, _ in scored:
        if score < 1.5:
            break
        if (
            source_index in used_before
            or target_index in used_after
        ):
            continue
        used_before.add(source_index)
        used_after.add(target_index)
        matched.append((source_index, target_index))
    removed = tuple(
        index for index in range(len(before)) if index not in used_before
    )
    appeared = tuple(
        index for index in range(len(after)) if index not in used_after
    )
    return tuple(matched), removed, appeared


def _spatial_facts(
    objects: Sequence[ObjectInfo],
) -> Counter[Tuple[str, str, str]]:
    descriptors = tuple(
        (
            _object_role(obj),
            float(obj.center[0]),
            float(obj.center[1]),
            int(obj.bbox[0]),
            int(obj.bbox[1]),
            int(obj.bbox[2]),
            int(obj.bbox[3]),
        )
        for obj in objects
    )
    return Counter(dict(_cached_spatial_fact_items(descriptors)))


@lru_cache(maxsize=4_096)
def _cached_spatial_fact_items(
    objects: Tuple[Tuple[Any, ...], ...],
) -> Tuple[Tuple[Tuple[str, str, str], int], ...]:
    facts: Counter[Tuple[str, str, str]] = Counter()
    for index, source in enumerate(objects):
        for target in objects[index + 1:]:
            left_role, right_role = sorted((
                source[0],
                target[0],
            ))
            sr, sc = source[1], source[2]
            tr, tc = target[1], target[2]
            row_gap = _interval_gap(
                source[3],
                source[5],
                target[3],
                target[5],
            )
            col_gap = _interval_gap(
                source[4],
                source[6],
                target[4],
                target[6],
            )
            if row_gap + col_gap <= 1:
                facts[("adjacent", left_role, right_role)] += 1
            if int(round(sr)) == int(round(tr)):
                facts[("aligned_row", left_role, right_role)] += 1
            if int(round(sc)) == int(round(tc)):
                facts[("aligned_column", left_role, right_role)] += 1
            if math.dist((sr, sc), (tr, tc)) <= 3.0:
                facts[("near", left_role, right_role)] += 1
    return tuple(sorted(facts.items()))


def _interval_gap(
    first_min: int,
    first_max: int,
    second_min: int,
    second_max: int,
) -> int:
    if first_max < second_min:
        return int(second_min - first_max - 1)
    if second_max < first_min:
        return int(first_min - second_max - 1)
    return 0


def _pattern(
    *,
    family: str,
    predicate: str,
    source_role: str = "",
    target_role: str = "",
    direction: str = "",
) -> MultiformRelationPattern:
    payload = (
        str(family),
        str(predicate),
        str(source_role),
        str(target_role),
        str(direction),
    )
    signature = (
        f"{family}::"
        + hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]
    )
    return MultiformRelationPattern(
        family=str(family),
        signature=signature,
        source_role=str(source_role),
        target_role=str(target_role),
        predicate=str(predicate),
        direction=str(direction),
    )


def _object_role(obj: ObjectInfo) -> str:
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
    payload = (
        area_bucket,
        min(height, 7),
        min(width, 7),
        _shape_key(obj),
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _shape_key(obj: ObjectInfo) -> str:
    if not obj.cells:
        return "empty"
    min_row = min(row for row, _ in obj.cells)
    min_col = min(col for _, col in obj.cells)
    normalized = tuple(sorted(
        (int(row - min_row), int(col - min_col))
        for row, col in obj.cells
    ))
    return _normalized_shape_key(normalized)


@lru_cache(maxsize=65_536)
def _normalized_shape_key(
    normalized: Tuple[Tuple[int, int], ...],
) -> str:
    return hashlib.sha1(
        repr(normalized).encode("utf-8")
    ).hexdigest()[:12]


def _acted_role(
    observation: GameObservation,
    action_data: Mapping[str, Any] | None,
) -> str:
    data = dict(action_data or {})
    if "x" not in data or "y" not in data:
        return ""
    try:
        x = int(data["x"])
        y = int(data["y"])
    except (TypeError, ValueError):
        return ""
    containing = [
        obj for obj in observation.objects
        if (y, x) in set(obj.cells)
    ]
    if containing:
        return _object_role(min(
            containing,
            key=lambda obj: (obj.area, obj.object_id),
        ))
    return "background"


def _actuator_signature(
    action_name: str,
    target_role: str,
    action_data: Mapping[str, Any] | None,
) -> str:
    schema = tuple(sorted(str(key) for key in dict(action_data or {})))
    payload = (str(action_name), str(target_role), schema)
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _concrete_action_key(
    action_name: str,
    action_data: Mapping[str, Any] | None,
) -> str:
    stable_data = tuple(sorted(
        (str(key), repr(value))
        for key, value in dict(action_data or {}).items()
    ))
    return repr((str(action_name), stable_data))


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
    concrete = []
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
    result = []
    seen = set()
    for name, data in concrete:
        stable_data = tuple(sorted(
            (str(key), repr(value))
            for key, value in data.items()
        ))
        if (name, stable_data) in seen:
            continue
        seen.add((name, stable_data))
        target_role = _acted_role(observation, data)
        actuator = _actuator_signature(name, target_role, data)
        result.append((name, data, actuator, target_role))
    return tuple(result)


def _observation_signature(observation: GameObservation) -> str:
    roles = Counter(_object_role(obj) for obj in observation.objects)
    spatial = _spatial_facts(observation.objects)
    payload = (
        tuple(sorted(roles.items())),
        tuple(sorted(spatial.items())),
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]


def _selection_context_signature(observation: GameObservation) -> str:
    """Identify the live level/state context without using game identity."""
    return (
        f"level-{int(observation.levels_completed)}:"
        f"grid-{int(observation.grid_hash)}"
    )


def _direction(dy: int, dx: int) -> str:
    vertical = "down" if dy > 0 else "up" if dy < 0 else ""
    horizontal = "right" if dx > 0 else "left" if dx < 0 else ""
    direction = "_".join(part for part in (vertical, horizontal) if part)
    magnitude = _magnitude_bucket(max(abs(dy), abs(dx)))
    return f"{direction or 'stationary'}_{magnitude}"


def _magnitude_bucket(value: int) -> str:
    return "one" if value <= 1 else "few" if value <= 3 else "many"


__all__ = [
    "MultiformRelationPattern",
    "MultiformRelationSelection",
    "OnlineMultiformRelationalLearner",
    "extract_multiform_relation_patterns",
]
