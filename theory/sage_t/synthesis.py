"""Deterministic fragment proposal, typed assembly and local program repair."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import (
    ActionBinding,
    ActionCandidate,
    Effect,
    Expression,
    GoalRule,
    JointProgramHypothesis,
    ObjectSchema,
    ObservedTransition,
    ProgramFragment,
    ProgressRule,
    TerminalRule,
    TransitionRule,
)

_KIND_TO_OPERATOR = {
    "move": "move",
    "global_transform": "transform",
    "click": "apply",
    "interact": "apply",
    "noop": "noop",
    "lethal": "apply",
}


@dataclass(frozen=True)
class AssembledProgram:
    program: JointProgramHypothesis
    prior_logprob: float


@dataclass(frozen=True)
class FragmentProposal:
    fragments: tuple[ProgramFragment, ...]
    plan_sequences: tuple[tuple[ActionCandidate, ...], ...] = ()


class DeterministicFragmentProposer:
    """Generate a bounded grammar from observed effects and weak priors."""

    def __init__(
        self,
        *,
        maximum_operator_candidates_per_action: int = 5,
    ) -> None:
        self.maximum_operator_candidates_per_action = max(
            1,
            int(maximum_operator_candidates_per_action),
        )

    def propose(
        self,
        *,
        available_actions: Sequence[str],
        transitions: Sequence[ObservedTransition] = (),
        mechanic_theory: Any | None = None,
        goal_hypotheses: Sequence[Any] = (),
        route_memory: Any | None = None,
    ) -> FragmentProposal:
        actions = tuple(
            sorted(
                {
                    str(action).strip().upper()
                    for action in available_actions
                    if str(action).strip().upper() not in {"", "RESET"}
                }
            )
        )
        fragments: list[ProgramFragment] = [
            ProgramFragment(
                fragment_id="schema_core",
                kind="schema",
                payload=ObjectSchema(
                    roles=(
                        "object",
                        "player",
                        "target",
                        "movable",
                        "hazardous",
                        "collectible",
                    )
                ),
                roles=("object", "player", "target"),
                provenance=("sage_t_deterministic_grammar",),
            )
        ]
        by_action: dict[str, list[ObservedTransition]] = defaultdict(list)
        for transition in transitions:
            by_action[transition.action.action_name].append(transition)
        for action in actions:
            fragments.extend(
                self._dynamics_for_action(action, by_action.get(action, ()))
            )
        fragments.extend(_legacy_mechanic_fragments(mechanic_theory))
        fragments.extend(_default_goal_fragments())
        fragments.extend(_legacy_goal_fragments(goal_hypotheses))
        plans = _route_sequences(route_memory)
        fragments.extend(
            ProgramFragment(
                fragment_id=f"plan_{index:03d}",
                kind="plan",
                payload=sequence,
                provenance=("level_route_memory",),
                prior_logprob=0.25,
            )
            for index, sequence in enumerate(plans)
        )
        unique: dict[str, ProgramFragment] = {}
        for fragment in fragments:
            previous = unique.get(fragment.fragment_id)
            if previous is None or fragment.prior_logprob > previous.prior_logprob:
                unique[fragment.fragment_id] = fragment
        return FragmentProposal(
            fragments=tuple(unique.values()),
            plan_sequences=plans,
        )

    def _dynamics_for_action(
        self,
        action: str,
        transitions: Sequence[ObservedTransition],
    ) -> list[ProgramFragment]:
        events = {event for transition in transitions for event in transition.events}
        operators = _operators_for_events(events, action)
        output = []
        for rank, operator in enumerate(
            operators[: self.maximum_operator_candidates_per_action]
        ):
            effects = _effects_for_events(events, operator)
            binding = ActionBinding(
                action_name=action,
                operator=operator,
                target_role=("target" if operator in {"select", "apply"} else ""),
            )
            rule = TransitionRule(
                rule_id=f"rule_{action.lower()}_{operator}",
                action_operator=operator,
                condition=Expression.constant(True),
                effects=effects,
            )
            output.append(
                ProgramFragment(
                    fragment_id=f"dyn_{action.lower()}_{operator}",
                    kind="dynamics",
                    payload=(binding, rule),
                    roles=("object", "target"),
                    predicted_events=tuple(sorted(events)),
                    provenance=(
                        "observed_transition" if transitions else "weak_operator_prior",
                    ),
                    prior_logprob=(-0.05 * rank if transitions else -0.25 - 0.1 * rank),
                )
            )
        return output


class ProgramAssembler:
    """Combine compatible fragments into a diverse bounded program beam."""

    def __init__(
        self,
        *,
        maximum_programs: int = 64,
        maximum_dynamics_beam: int = 16,
    ) -> None:
        self.maximum_programs = max(1, int(maximum_programs))
        self.maximum_dynamics_beam = max(1, int(maximum_dynamics_beam))

    def assemble(
        self,
        fragments: Sequence[ProgramFragment],
        *,
        available_actions: Sequence[str],
    ) -> tuple[AssembledProgram, ...]:
        schema = next(
            (
                fragment.payload
                for fragment in fragments
                if fragment.kind == "schema"
                and isinstance(fragment.payload, ObjectSchema)
            ),
            ObjectSchema(("object", "player", "target")),
        )
        dynamics_by_action: dict[
            str,
            list[tuple[ProgramFragment, ActionBinding, TransitionRule]],
        ] = defaultdict(list)
        goal_bundles = []
        for fragment in fragments:
            if fragment.kind == "dynamics":
                try:
                    binding, rule = fragment.payload
                except (TypeError, ValueError):
                    continue
                if isinstance(binding, ActionBinding) and isinstance(
                    rule,
                    TransitionRule,
                ):
                    dynamics_by_action[binding.action_name].append(
                        (fragment, binding, rule)
                    )
            elif fragment.kind == "goal_bundle":
                try:
                    progress, goal, terminal = fragment.payload
                except (TypeError, ValueError):
                    continue
                if (
                    isinstance(progress, ProgressRule)
                    and isinstance(goal, GoalRule)
                    and isinstance(terminal, tuple)
                    and terminal
                    and all(isinstance(item, TerminalRule) for item in terminal)
                ):
                    goal_bundles.append((fragment, progress, goal, terminal))
        actions = tuple(
            sorted(
                {
                    str(action).strip().upper()
                    for action in available_actions
                    if str(action).strip().upper() not in {"", "RESET"}
                }
            )
        )
        if not actions or not goal_bundles:
            return ()
        beam: list[
            tuple[
                float,
                tuple[ActionBinding, ...],
                tuple[TransitionRule, ...],
                tuple[str, ...],
                frozenset[str],
                frozenset[str],
            ]
        ] = [(0.0, (), (), (), frozenset(), frozenset())]
        for action in actions:
            candidates = dynamics_by_action.get(action)
            if not candidates:
                candidates = [
                    (
                        ProgramFragment(
                            fragment_id=f"fallback_{action.lower()}_noop",
                            kind="dynamics",
                            payload=None,
                            prior_logprob=-1.0,
                        ),
                        ActionBinding(action, "noop"),
                        TransitionRule(
                            rule_id=f"rule_{action.lower()}_noop",
                            action_operator="noop",
                            condition=Expression.constant(True),
                            effects=(Effect("assert", "no_effect"),),
                        ),
                    )
                ]
            expanded = []
            for score, bindings, rules, provenance, roles, events in beam:
                for fragment, binding, rule in candidates:
                    expanded.append(
                        (
                            score + float(fragment.prior_logprob),
                            bindings + (binding,),
                            rules + (rule,),
                            provenance + tuple(fragment.provenance),
                            roles | frozenset(fragment.roles),
                            events | frozenset(fragment.predicted_events),
                        )
                    )
            expanded.sort(key=lambda item: item[0], reverse=True)
            beam = expanded[: self.maximum_dynamics_beam]

        assembled: list[AssembledProgram] = []
        seen: set[str] = set()
        program_index = 0
        for (
            dynamics_score,
            bindings,
            rules,
            dynamics_provenance,
            dynamics_roles,
            dynamics_events,
        ) in beam:
            for fragment, progress, goal, terminal in goal_bundles:
                if not _compatible_fragments(
                    schema=schema,
                    dynamics_roles=dynamics_roles,
                    dynamics_events=dynamics_events,
                    goal_fragment=fragment,
                    goal=goal,
                ):
                    continue
                program = JointProgramHypothesis(
                    program_id=f"program_{program_index:04d}",
                    object_schema=schema,
                    action_bindings=bindings,
                    transition_rules=rules,
                    progress_rule=progress,
                    terminal_rules=terminal,
                    goal_rule=goal,
                    provenance=tuple(
                        sorted(set(dynamics_provenance + tuple(fragment.provenance)))
                    ),
                )
                program_index += 1
                if program.canonical_hash in seen:
                    continue
                seen.add(program.canonical_hash)
                complexity_prior = (
                    -0.05 * program.node_count - 0.25 * program.local_constant_count
                )
                assembled.append(
                    AssembledProgram(
                        program=program,
                        prior_logprob=(
                            dynamics_score
                            + float(fragment.prior_logprob)
                            + complexity_prior
                        ),
                    )
                )
        assembled.sort(key=lambda item: item.prior_logprob, reverse=True)
        return tuple(
            _preserve_family_diversity(
                assembled,
                maximum=self.maximum_programs,
            )
        )


class ProgramMutator:
    """Localized repair proposals that must be replayed before admission."""

    def __init__(self, *, maximum_children: int = 8) -> None:
        self.maximum_children = max(0, int(maximum_children))

    def mutate(
        self,
        program: JointProgramHypothesis,
        evidence: ObservedTransition,
    ) -> tuple[JointProgramHypothesis, ...]:
        if self.maximum_children <= 0:
            return ()
        binding = next(
            (
                item
                for item in program.action_bindings
                if item.action_name == evidence.action.action_name
            ),
            None,
        )
        if binding is None:
            return ()
        observed_effects = _effects_for_events(
            set(evidence.events),
            binding.operator,
        )
        children: list[JointProgramHypothesis] = []
        matching = [
            index
            for index, rule in enumerate(program.transition_rules)
            if rule.action_operator == binding.operator
        ]
        if matching:
            index = matching[0]
            original = program.transition_rules[index]
            rules = list(program.transition_rules)
            rules[index] = replace(rules[index], effects=observed_effects)
            children.append(
                _repair_child(
                    program,
                    suffix="effects",
                    transition_rules=tuple(rules),
                    edit_distance=1,
                )
            )
            rules = list(program.transition_rules)
            if original.condition == Expression.constant(True):
                rules[index] = replace(
                    original,
                    condition=Expression(
                        op="and",
                        args=(
                            original.condition,
                            Expression.fact("exists", "$target"),
                        ),
                    ),
                )
                suffix = "precondition_add"
            else:
                rules[index] = replace(
                    original,
                    condition=Expression.constant(True),
                )
                suffix = "precondition_remove"
            children.append(
                _repair_child(
                    program,
                    suffix=suffix,
                    transition_rules=tuple(rules),
                    edit_distance=1,
                )
            )
            replacement_operator = next(
                (
                    operator
                    for operator in _operators_for_events(
                        set(evidence.events),
                        evidence.action.action_name,
                    )
                    if operator != binding.operator
                ),
                "",
            )
            if replacement_operator:
                bindings = tuple(
                    replace(item, operator=replacement_operator)
                    if item.action_name == binding.action_name
                    else item
                    for item in program.action_bindings
                )
                old_operator_still_bound = any(
                    item.action_name != binding.action_name
                    and item.operator == binding.operator
                    for item in program.action_bindings
                )
                replacement_rule = replace(
                    original,
                    rule_id=f"{original.rule_id}_operator",
                    action_operator=replacement_operator,
                    effects=_effects_for_events(
                        set(evidence.events),
                        replacement_operator,
                    ),
                )
                operator_rules = list(program.transition_rules)
                if old_operator_still_bound:
                    operator_rules.append(replacement_rule)
                else:
                    operator_rules[index] = replacement_rule
                children.append(
                    _repair_child(
                        program,
                        suffix="action_semantics",
                        action_bindings=bindings,
                        transition_rules=tuple(operator_rules),
                        edit_distance=1,
                    )
                )
            replacement_role = next(
                (
                    role
                    for role in ("target", "object", "player")
                    if role != binding.target_role
                ),
                "",
            )
            if replacement_role:
                children.append(
                    _repair_child(
                        program,
                        suffix="target_role",
                        action_bindings=tuple(
                            replace(item, target_role=replacement_role)
                            if item.action_name == binding.action_name
                            else item
                            for item in program.action_bindings
                        ),
                        edit_distance=1,
                    )
                )
        goal_expression = program.goal_rule.expression
        if goal_expression.op in {"exists", "forall"}:
            children.append(
                _repair_child(
                    program,
                    suffix="goal_quantifier",
                    goal_rule=replace(
                        program.goal_rule,
                        expression=replace(
                            goal_expression,
                            op=(
                                "exists" if goal_expression.op == "forall" else "forall"
                            ),
                        ),
                    ),
                    edit_distance=1,
                )
            )
        if (
            evidence.observation.goal_probability is not None
            and evidence.observation.goal_probability >= 0.5
            and program.goal_rule.family != "level_completion"
        ):
            children.append(
                _repair_child(
                    program,
                    suffix="goal",
                    progress_rule=ProgressRule(Expression.fact("level_complete")),
                    goal_rule=GoalRule(
                        Expression.fact("level_complete"),
                        family="level_completion",
                    ),
                    edit_distance=1,
                )
            )
        if (
            evidence.observation.terminal_probability is not None
            and evidence.observation.terminal_probability >= 0.5
        ):
            children.append(
                _repair_child(
                    program,
                    suffix="terminal",
                    terminal_rules=(
                        TerminalRule(
                            Expression.fact("game_over"),
                            "game_over",
                        ),
                        *tuple(
                            rule
                            for rule in program.terminal_rules
                            if rule.outcome != "game_over"
                        ),
                    ),
                    edit_distance=1,
                )
            )
        unique: dict[str, JointProgramHypothesis] = {}
        for child in children:
            unique[child.canonical_hash] = child
        return tuple(unique.values())[: self.maximum_children]


def _operators_for_events(events: set[str], action: str) -> list[str]:
    if events:
        ranked = []
        if "moved" in events:
            ranked.append("move")
        if events.intersection(
            {
                "created",
                "removed",
                "morphology_changed",
                "changed",
            }
        ) or any(event.startswith("relation_") for event in events):
            ranked.extend(("apply", "transform"))
        if "no_effect" in events:
            ranked.append("noop")
        if not ranked:
            ranked.append("transform")
        ranked.extend(("select", "apply", "move", "transform", "noop"))
        return list(dict.fromkeys(ranked))
    weak = ["move", "select", "apply", "transform", "noop"]
    if action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
        weak.remove("move")
        weak.insert(0, "move")
    return weak


def _effects_for_events(
    events: set[str],
    operator: str,
) -> tuple[Effect, ...]:
    effects: list[Effect] = []
    for event in sorted(events):
        if event.startswith("relation_added:"):
            effects.append(Effect("assert", event.split(":", 1)[1]))
        elif event.startswith("relation_removed:"):
            effects.append(Effect("retract", event.split(":", 1)[1]))
        elif event.startswith(("component_", "hole_")):
            predicate = (
                "component_count_changed"
                if event.startswith("component_")
                else "hole_count_changed"
            )
            effects.append(Effect("assert", predicate))
        elif event == "moved":
            effects.append(
                Effect(
                    "move_relative",
                    terms=("$actor",),
                    value="relative",
                )
            )
        elif event == "morphology_changed":
            effects.append(
                Effect(
                    "change_morphology",
                    terms=("$target",),
                    value="unspecified",
                )
            )
        elif event == "progress":
            effects.append(Effect("progress", value=1.0))
        elif event == "level_complete":
            effects.append(Effect("win"))
        elif event == "game_over":
            effects.append(Effect("fail"))
        elif event in {
            "changed",
            "created",
            "removed",
            "no_effect",
        }:
            effects.append(Effect("assert", event))
    if effects:
        return tuple(dict.fromkeys(effects))
    if operator == "select":
        return (
            Effect(
                operation="set_register",
                key="selected",
                value="$target",
            ),
        )
    if operator == "apply":
        return (
            Effect("assert", "changed"),
            Effect("assert", "solved", ("$target",)),
        )
    if operator == "move":
        return (
            Effect(
                "move_relative",
                terms=("$actor",),
                value="relative",
            ),
        )
    if operator in {"transform", "create", "remove", "toggle"}:
        return (Effect("assert", "changed"),)
    return (Effect("assert", "no_effect"),)


def _default_goal_fragments() -> list[ProgramFragment]:
    game_over = TerminalRule(Expression.fact("game_over"), "game_over")
    win = TerminalRule(Expression.fact("level_complete"), "win")
    level = (
        ProgressRule(Expression.fact("level_complete")),
        GoalRule(
            Expression.fact("level_complete"),
            family="level_completion",
        ),
        (game_over, win),
    )
    solved_count = Expression(
        op="count",
        args=(Expression.fact("solved", "?target"),),
        variable="?target",
        role="target",
    )
    target_count = Expression(op="count", role="target")
    solve_all = (
        ProgressRule(
            Expression(
                op="ratio",
                args=(solved_count, target_count),
            )
        ),
        GoalRule(
            Expression(
                op="forall",
                args=(Expression.fact("solved", "?target"),),
                variable="?target",
                role="target",
            ),
            family="solve_all_targets",
        ),
        (game_over, win),
    )
    contact_goal = Expression(
        op="exists",
        args=(Expression.fact("contact", "$actor", "?target"),),
        variable="?target",
        role="target",
    )
    reach = (
        ProgressRule(contact_goal),
        GoalRule(contact_goal, family="reach_target"),
        (game_over, win),
    )
    exhaust_goal = Expression(
        op="eq",
        args=(
            Expression(op="count", role="target"),
            Expression.constant(0.0),
        ),
    )
    exhaust = (
        ProgressRule(exhaust_goal),
        GoalRule(exhaust_goal, family="exhaust_targets"),
        (game_over, win),
    )
    bundles = (
        ("goal_level_completion", level, 0.0),
        ("goal_solve_all_targets", solve_all, -0.10),
        ("goal_reach_target", reach, -0.15),
        ("goal_exhaust_targets", exhaust, -0.20),
    )
    return [
        ProgramFragment(
            fragment_id=identifier,
            kind="goal_bundle",
            payload=payload,
            roles=("player", "target"),
            provenance=("sage_t_deterministic_goal_grammar",),
            prior_logprob=prior,
        )
        for identifier, payload, prior in bundles
    ]


def _legacy_mechanic_fragments(theory: Any | None) -> list[ProgramFragment]:
    if theory is None or not callable(getattr(theory, "hypotheses", None)):
        return []
    output = []
    for hypothesis in theory.hypotheses():
        action = str(getattr(hypothesis, "action", "")).upper()
        kind = str(getattr(hypothesis, "kind", "")).lower()
        operator = _KIND_TO_OPERATOR.get(kind)
        if not action or operator is None:
            continue
        confidence = float(getattr(hypothesis, "confidence", 0.0) or 0.0)
        status = str(getattr(getattr(hypothesis, "status", ""), "value", ""))
        prior = math.log(max(0.05, min(0.95, confidence or 0.1)))
        if status == "refuted":
            prior -= 4.0
        binding = ActionBinding(
            action,
            operator,
            "target" if operator in {"select", "apply"} else "",
        )
        rule = TransitionRule(
            rule_id=f"legacy_{action.lower()}_{operator}",
            action_operator=operator,
            condition=Expression.constant(True),
            effects=_effects_for_events(set(), operator),
        )
        output.append(
            ProgramFragment(
                fragment_id=f"legacy_{action.lower()}_{operator}",
                kind="dynamics",
                payload=(binding, rule),
                provenance=(f"game_theory:{status or 'candidate'}",),
                prior_logprob=prior,
            )
        )
    return output


def _legacy_goal_fragments(
    hypotheses: Sequence[Any],
) -> list[ProgramFragment]:
    defaults = {
        fragment.payload[1].family: fragment for fragment in _default_goal_fragments()
    }
    family_aliases = {
        "exhaust": "exhaust_targets",
        "reach": "reach_target",
        "appear": "reach_target",
        "convert": "solve_all_targets",
        "match": "solve_all_targets",
        "complete": "level_completion",
    }
    output = []
    for index, hypothesis in enumerate(hypotheses):
        raw_family = str(getattr(hypothesis, "family", "")).lower()
        family = next(
            (
                normalized
                for prefix, normalized in family_aliases.items()
                if prefix in raw_family
            ),
            "",
        )
        template = defaults.get(family)
        if template is None:
            continue
        priority = float(
            getattr(
                hypothesis,
                "prior_priority",
                getattr(hypothesis, "selection_utility", 0.0),
            )
            or 0.0
        )
        output.append(
            replace(
                template,
                fragment_id=f"legacy_goal_{family}_{index:03d}",
                provenance=("online_goal_hypothesis",),
                prior_logprob=max(-2.0, min(2.0, priority)),
            )
        )
    return output


def _route_sequences(
    route_memory: Any | None,
) -> tuple[tuple[ActionCandidate, ...], ...]:
    if route_memory is None or not callable(getattr(route_memory, "routes", None)):
        return ()
    sequences = []
    for route in route_memory.routes():
        raw_actions = getattr(
            route,
            "actions",
            getattr(route, "primitive_actions", ()),
        )
        sequence = []
        for raw in tuple(raw_actions or ())[:8]:
            name = getattr(
                raw,
                "action_name",
                getattr(raw, "name", raw),
            )
            data = getattr(
                raw,
                "data",
                getattr(
                    raw,
                    "action_data",
                    getattr(raw, "action_args", {}),
                ),
            )
            try:
                normalized_data = data() if callable(data) else data
                sequence.append(
                    ActionCandidate(
                        str(name),
                        dict(normalized_data or {}),
                    )
                )
            except (TypeError, ValueError):
                continue
        if sequence:
            sequences.append(tuple(sequence))
    return tuple(sequences[:16])


def _compatible_fragments(
    *,
    schema: ObjectSchema,
    dynamics_roles: frozenset[str],
    dynamics_events: frozenset[str],
    goal_fragment: ProgramFragment,
    goal: GoalRule,
) -> bool:
    goal_roles = frozenset(goal_fragment.roles)
    declared_roles = frozenset(schema.roles)
    if not (dynamics_roles | goal_roles).issubset(declared_roles):
        return False
    if dynamics_roles and goal_roles and dynamics_roles & goal_roles:
        return True
    if not dynamics_events:
        # Weak deterministic priors have no observed dependency yet.
        return True
    dependencies = {
        "level_completion": {
            "level_complete",
            "progress",
            "changed",
            "created",
            "removed",
        },
        "solve_all_targets": {
            "changed",
            "created",
            "removed",
            "morphology_changed",
        },
        "reach_target": {
            "moved",
            "relation_added:contact",
            "relation_added:near",
        },
        "exhaust_targets": {"removed", "changed"},
    }
    expected = dependencies.get(goal.family, set())
    return bool(expected & dynamics_events)


def _preserve_family_diversity(
    programs: Sequence[AssembledProgram],
    *,
    maximum: int,
) -> list[AssembledProgram]:
    selected: list[AssembledProgram] = []
    family_counts: dict[tuple[str, str], int] = defaultdict(int)
    for candidate in programs:
        family = candidate.program.semantic_family
        if family_counts[family] >= 4:
            continue
        selected.append(candidate)
        family_counts[family] += 1
        if len(selected) >= maximum:
            return selected
    for candidate in programs:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= maximum:
            break
    return selected


def _repair_child(
    program: JointProgramHypothesis,
    *,
    suffix: str,
    edit_distance: int,
    **changes: Any,
) -> JointProgramHypothesis:
    parent = program.canonical_hash
    digest = hashlib.sha256(f"{parent}:{suffix}:{edit_distance}".encode()).hexdigest()[
        :10
    ]
    return replace(
        program,
        program_id=f"repair_{suffix}_{digest}",
        provenance=tuple(sorted(set(program.provenance + ("localized_repair",)))),
        parent_hash=parent,
        edit_distance=program.edit_distance + int(edit_distance),
        **changes,
    )


__all__ = [
    "AssembledProgram",
    "DeterministicFragmentProposer",
    "FragmentProposal",
    "ProgramAssembler",
    "ProgramMutator",
]
