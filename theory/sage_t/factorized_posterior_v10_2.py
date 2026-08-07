"""Strict five-factor variational control for the T10.2 posterior.

The control posterior is the virtual product

``q(D) q(G) q(F) q(Tau) q(A)``.

That product and the bounded executable particle bank are deliberately kept
separate.  A particle's weight is the *unnormalized* product of its five
marginal probabilities.  Product mass belonging to combinations that are not
materialized is retained in ``residual_mass``; it is never conditioned away by
renormalizing over the bounded bank.

The observation update uses one simultaneous mean-field step.  Each scored
particle likelihood is allocated exactly once:

* dynamics and goal receive complementary halves of the physical term;
* frame and transport receive complementary halves of the frame term after
  the weighted commutativity penalty;
* option receives the option term.

The complementary second half is calculated by subtraction so the five
allocations sum (up to the final floating-point addition) to
``physical + frame - commutativity + option``.  Conditional expectations are
estimated only from the pre-registered, outcome-independent executable bank;
the resulting marginal update is inherited by the virtual combinations in the
residual.  MAP collapse is disabled because it would destroy factorization.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, overload

from .contracts import Expression, JointProgramHypothesis
from .gauge_inference_v10_2 import (
    GaugeParticle,
    GaugeProgramPosterior,
    GaugeUpdate,
    JointGaugeHypothesis,
)

FACTOR_NAMES = ("dynamics", "goal", "frame", "transport", "option")
MAXIMUM_CONTROL_CANDIDATES = 256
MAXIMUM_CONTROL_ENUMERATIONS = 65_536
_MASS_TOLERANCE = 1e-12


class FactorizedControlRefusal(ValueError):
    """The requested scientific control cannot be represented honestly."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("canonical_payload must be finite JSON data") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _component_payload(value: Any) -> Any:
    """Return a component's mandatory canonical payload.

    There is intentionally no ``str(value)`` or canonical-hash fallback.  A
    factor that cannot describe itself canonically cannot enter a reproducible
    scientific control.
    """

    sentinel = object()
    payload = getattr(value, "canonical_payload", sentinel)
    if payload is sentinel:
        raise TypeError(
            f"{type(value).__name__} must expose canonical_payload for factorization"
        )
    if callable(payload):
        payload = payload()
    _canonical_json(payload)
    return payload


def _frame_id(frame: Any) -> str:
    direct = str(getattr(frame, "frame_id", "")).strip()
    if direct:
        return direct
    spec = getattr(frame, "spec", None) or getattr(frame, "frame", None)
    return str(getattr(spec, "frame_id", "")).strip()


def factor_keys(hypothesis: JointGaugeHypothesis) -> tuple[str, ...]:
    """Return canonical keys for ``(D, G, F, Tau, A)``."""

    program = hypothesis.world_program.canonical_payload
    dynamics = {
        "object_schema": program["object_schema"],
        "action_bindings": program["action_bindings"],
        "transition_rules": program["transition_rules"],
    }
    goal = {
        "progress": program["progress"],
        "terminal": program["terminal"],
        "goal": program["goal"],
    }
    transports = sorted(
        (_component_payload(item) for item in hypothesis.transports),
        key=_canonical_json,
    )
    return (
        _hash(dynamics),
        _hash(goal),
        _hash(_component_payload(hypothesis.frame)),
        _hash(transports),
        _hash(_component_payload(hypothesis.option)),
    )


def _logsumexp(values: Sequence[float]) -> float:
    finite = tuple(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return float("-inf")
    maximum = max(finite)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in finite))


@dataclass(frozen=True)
class FactorDistribution:
    """One normalized categorical marginal in canonical log space."""

    name: str
    entries: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.name not in FACTOR_NAMES:
            raise ValueError(f"unknown factor marginal: {self.name}")
        entries = tuple(sorted((str(key), float(value)) for key, value in self.entries))
        if not entries:
            raise ValueError(f"{self.name} marginal cannot be empty")
        if len({key for key, _ in entries}) != len(entries):
            raise ValueError(f"{self.name} marginal keys must be unique")
        if any(not math.isfinite(value) for _, value in entries):
            raise ValueError(f"{self.name} marginal probabilities must be positive")
        if abs(_logsumexp(tuple(value for _, value in entries))) > _MASS_TOLERANCE:
            raise ValueError(f"{self.name} marginal must be normalized")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def uniform(cls, name: str, keys: Sequence[str]) -> FactorDistribution:
        support = tuple(sorted({str(key) for key in keys}))
        if not support:
            raise ValueError(f"{name} factor support cannot be empty")
        log_probability = -math.log(len(support))
        return cls(name, tuple((key, log_probability) for key in support))

    @classmethod
    def from_log_scores(
        cls,
        name: str,
        scores: Mapping[str, float],
    ) -> FactorDistribution:
        if not scores:
            raise ValueError(f"{name} factor scores cannot be empty")
        normalizer = _logsumexp(tuple(float(value) for value in scores.values()))
        if not math.isfinite(normalizer):
            raise ValueError(f"{name} factor scores contain no finite mass")
        return cls(
            name,
            tuple(
                (str(key), float(value) - normalizer) for key, value in scores.items()
            ),
        )

    @property
    def support(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.entries)

    @property
    def log_probabilities(self) -> Mapping[str, float]:
        return dict(self.entries)

    @property
    def probabilities(self) -> Mapping[str, float]:
        return {key: math.exp(value) for key, value in self.entries}

    def log_probability(self, key: str) -> float:
        wanted = str(key)
        for candidate, value in self.entries:
            if candidate == wanted:
                return value
        raise KeyError(f"{wanted} is outside the {self.name} marginal support")


@dataclass(frozen=True)
class FactorMarginals:
    """The five separately stored marginals of the control posterior."""

    dynamics: FactorDistribution
    goal: FactorDistribution
    frame: FactorDistribution
    transport: FactorDistribution
    option: FactorDistribution

    def __post_init__(self) -> None:
        for name, marginal in zip(FACTOR_NAMES, self, strict=True):
            if marginal.name != name:
                raise ValueError(f"expected {name} marginal, received {marginal.name}")

    def __iter__(self) -> Iterator[FactorDistribution]:
        return iter((self.dynamics, self.goal, self.frame, self.transport, self.option))

    def __getitem__(self, name: str) -> FactorDistribution:
        if name not in FACTOR_NAMES:
            raise KeyError(name)
        return getattr(self, name)

    @classmethod
    def uniform_from_rows(
        cls,
        component_rows: Sequence[tuple[str, ...]],
    ) -> FactorMarginals:
        """Create independent uniform priors from factor domains only.

        Outcome weights are deliberately not accepted: truncated joint support
        cannot identify five independent marginals.
        """

        rows = tuple(tuple(str(item) for item in row) for row in component_rows)
        if not rows:
            raise ValueError("five-factor marginals require at least one row")
        if any(len(row) != len(FACTOR_NAMES) for row in rows):
            raise ValueError("each factor row must contain five canonical keys")
        distributions = tuple(
            FactorDistribution.uniform(
                name,
                tuple(row[index] for row in rows),
            )
            for index, name in enumerate(FACTOR_NAMES)
        )
        return cls(*distributions)

    @classmethod
    def mdl_from_hypotheses(
        cls,
        hypotheses: Sequence[JointGaugeHypothesis],
    ) -> FactorMarginals:
        """Factorize the frozen MDL energy without outcome-derived weights.

        Canonically identical factors may have several synthesis derivations.
        Their best (largest) MDL log prior is selected deterministically, just
        as the joint T10.2 posterior keeps the minimum-description derivation.
        """

        scores: list[dict[str, float]] = [{} for _ in FACTOR_NAMES]
        for hypothesis in hypotheses:
            keys = factor_keys(hypothesis)
            components = factor_log_prior_components(hypothesis)
            for index, (key, value) in enumerate(zip(keys, components, strict=True)):
                previous = scores[index].get(key)
                if previous is None or value > previous:
                    scores[index][key] = value
        if any(not values for values in scores):
            raise ValueError("five-factor MDL priors require all factor domains")
        return cls(
            *(
                FactorDistribution.from_log_scores(name, values)
                for name, values in zip(FACTOR_NAMES, scores, strict=True)
            )
        )

    @property
    def support_sizes(self) -> Mapping[str, int]:
        return {marginal.name: len(marginal.entries) for marginal in self}


def factorized_log_weights(
    component_rows: Sequence[tuple[str, ...]],
    marginals: FactorMarginals,
) -> tuple[float, ...]:
    """Materialize unnormalized products from already stored marginals.

    This function never infers marginal probabilities from joint particle
    weights.  Consequently it is idempotent on diagonal or otherwise
    truncated banks: repeated materialization from the same marginals returns
    exactly the same products and leaves omitted product mass outside support.
    """

    if not isinstance(marginals, FactorMarginals):
        raise TypeError(
            "factorized_log_weights requires five explicit marginals; "
            "joint weights cannot identify marginals on truncated support"
        )
    rows = tuple(tuple(str(item) for item in row) for row in component_rows)
    if any(len(row) != len(FACTOR_NAMES) for row in rows):
        raise ValueError("each factor row must contain five canonical keys")
    return tuple(
        math.fsum(
            marginal.log_probability(key)
            for marginal, key in zip(marginals, row, strict=True)
        )
        for row in rows
    )


def _world_program(
    dynamics: JointProgramHypothesis,
    goal: JointProgramHypothesis,
) -> JointProgramHypothesis:
    return JointProgramHypothesis(
        program_id="factorized_control",
        object_schema=dynamics.object_schema,
        action_bindings=dynamics.action_bindings,
        transition_rules=dynamics.transition_rules,
        progress_rule=goal.progress_rule,
        terminal_rules=goal.terminal_rules,
        goal_rule=goal.goal_rule,
        provenance=(),
        parent_hash="",
        # Edit distance is assigned to the dynamics factor in the exact MDL
        # decomposition below.  Taking max(D, G) would make the prior
        # non-separable and would leak the paired source program into q(D)q(G).
        edit_distance=dynamics.edit_distance,
    )


def _ast_nodes(value: Any) -> int:
    for name in ("node_count", "ast_node_count"):
        count = getattr(value, name, None)
        if count is not None:
            return max(1, int(count))
    return 1


def factor_log_prior_components(
    hypothesis: JointGaugeHypothesis,
) -> tuple[float, ...]:
    """Return the exact frozen MDL energies for ``D,G,F,Tau,A``.

    Their sum is exactly :attr:`JointGaugeHypothesis.log_prior`.  The world
    root, local action constants, and edit distance belong to dynamics; goal
    rules form the goal factor; frames have no registered AST charge; every
    transport and option AST node receives ``-0.05``.
    """

    program = hypothesis.world_program
    dynamics_nodes = (
        1
        + len(program.object_schema.roles)
        + len(program.action_bindings)
        + sum(rule.node_count for rule in program.transition_rules)
    )
    goal_nodes = (
        program.progress_rule.node_count
        + sum(rule.node_count for rule in program.terminal_rules)
        + program.goal_rule.node_count
    )
    components = (
        -0.05 * dynamics_nodes
        - 0.25 * program.local_constant_count
        - float(program.edit_distance),
        -0.05 * goal_nodes,
        0.0,
        -0.05 * sum(_ast_nodes(item) for item in hypothesis.transports),
        -0.05 * _ast_nodes(hypothesis.option),
    )
    if not math.isclose(
        math.fsum(components),
        hypothesis.log_prior,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("factor MDL decomposition drifted from joint prior")
    return components


def _walk_expression(expression: Expression) -> Iterator[Expression]:
    yield expression
    for child in expression.args:
        yield from _walk_expression(child)


def _goal_dynamics_compatible(
    dynamics: JointProgramHypothesis,
    goal: JointProgramHypothesis,
) -> bool:
    roles = set(dynamics.object_schema.roles)
    if any(
        binding.target_role and binding.target_role not in roles
        for binding in dynamics.action_bindings
    ):
        return False
    expressions = (
        goal.progress_rule.expression,
        *(rule.expression for rule in goal.terminal_rules),
        goal.goal_rule.expression,
    )
    required_roles = {
        node.role
        for expression in expressions
        for node in _walk_expression(expression)
        if node.role
    }
    if not required_roles <= roles:
        return False
    required_counters = {
        str(node.value)
        for expression in expressions
        for node in _walk_expression(expression)
        if node.op == "counter"
    }
    # ``progress`` is the executor's registered physical outcome counter and
    # may be present in observed state even when a particular candidate has no
    # progress-producing transition.
    available_counters = {"progress"} | {
        effect.key
        for rule in dynamics.transition_rules
        for effect in rule.effects
        if effect.operation in {"set_counter", "increment_counter"}
    }
    return required_counters <= available_counters


def _option_bindings_compatible(
    dynamics: JointProgramHypothesis,
    option: Any,
) -> bool:
    schemas = getattr(option, "action_schemas", None)
    if schemas is None:
        return False
    option_schemas = {
        str(item).strip().lower() for item in schemas if str(item).strip()
    }
    bound = {binding.action_name.lower() for binding in dynamics.action_bindings}
    return bool(option_schemas) and option_schemas <= bound


def _frame_transport_compatible(frame: Any, transports: Sequence[Any]) -> bool:
    native = _frame_id(frame).strip().lower()
    if not native:
        return False
    if not transports:
        return True

    adjacency: dict[str, set[str]] = {}
    declared_frames: set[str] = set()
    for transport in transports:
        source = (
            str(
                getattr(transport, "source_frame_id", "")
                or getattr(transport, "source_frame", "")
            )
            .strip()
            .lower()
        )
        target = (
            str(
                getattr(transport, "target_frame_id", "")
                or getattr(transport, "target_frame", "")
            )
            .strip()
            .lower()
        )
        if not source or not target or source == target:
            return False
        declared_frames.update((source, target))
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    # A transport tuple is one factor.  Every map in it must participate in a
    # path connected to the particle's native frame; an unrelated component
    # would otherwise pass merely because another map touches the native view.
    reachable = {native}
    frontier = [native]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency.get(current, ()):
            if neighbor in reachable:
                continue
            reachable.add(neighbor)
            frontier.append(neighbor)
    return declared_frames <= reachable


def typed_recombination_compatible(
    *,
    dynamics: JointProgramHypothesis,
    goal: JointProgramHypothesis,
    frame: Any,
    transports: Sequence[Any],
    option: Any,
) -> bool:
    """Check the three registered cross-factor interfaces."""

    # Canonicalization is part of the type contract, not an optional hashing
    # convenience.  Raise for missing payloads instead of silently rejecting.
    _component_payload(frame)
    for transport in transports:
        _component_payload(transport)
    _component_payload(option)
    return (
        _goal_dynamics_compatible(dynamics, goal)
        and _option_bindings_compatible(dynamics, option)
        and _frame_transport_compatible(frame, transports)
    )


@dataclass(frozen=True)
class _FactorDomain:
    name: str
    entries: tuple[tuple[str, Any], ...]
    log_priors: tuple[tuple[str, float], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.entries)

    def log_prior(self, key: str) -> float:
        for candidate, value in self.log_priors:
            if candidate == key:
                return value
        raise KeyError(key)


def _factor_representative_payload(index: int, value: Any) -> Any:
    if index == 3:
        return sorted(
            (_component_payload(item) for item in value),
            key=_canonical_json,
        )
    return _component_payload(value)


def _factor_domains(
    candidates: Sequence[JointGaugeHypothesis],
) -> tuple[_FactorDomain, ...]:
    banks: list[dict[str, tuple[float, str, Any]]] = [{} for _ in FACTOR_NAMES]
    for candidate in candidates:
        keys = factor_keys(candidate)
        priors = factor_log_prior_components(candidate)
        values = (
            candidate.world_program,
            candidate.world_program,
            candidate.frame,
            candidate.transports,
            candidate.option,
        )
        for index, (key, value) in enumerate(zip(keys, values, strict=True)):
            prior = priors[index]
            canonical = _canonical_json(_factor_representative_payload(index, value))
            previous = banks[index].get(key)
            if (
                previous is None
                or prior > previous[0]
                or (prior == previous[0] and canonical < previous[1])
            ):
                banks[index][key] = (prior, canonical, value)
    return tuple(
        _FactorDomain(
            name,
            tuple((key, ranked[2]) for key, ranked in sorted(bank.items())),
            tuple((key, ranked[0]) for key, ranked in sorted(bank.items())),
        )
        for name, bank in zip(FACTOR_NAMES, banks, strict=True)
    )


def _recombine(parts: tuple[Any, ...]) -> JointGaugeHypothesis:
    dynamics, goal, frame, transports, option = parts
    return JointGaugeHypothesis(
        world_program=_world_program(dynamics, goal),
        frame=frame,
        transports=tuple(transports),
        option=option,
    )


@dataclass(frozen=True)
class FactorizedBankMetrics:
    source_particles: int
    source_classes: int
    target_particles: int
    target_classes: int
    particle_budget: int
    enumeration_budget: int
    cartesian_combinations: int
    enumeration_complete: bool
    generation_attempts: int
    compatible_combinations_observed: int
    incompatible_combinations_observed: int
    materialized_combinations: int
    unmaterialized_compatible_combinations_lower_bound: int
    novel_recombinations: int

    @property
    def capacity_matched(self) -> bool:
        return (
            self.source_particles == self.target_particles
            and self.source_classes == self.target_classes
            and self.target_particles <= self.particle_budget
        )

    def as_dict(self) -> Mapping[str, int | bool]:
        return {
            "source_particles": self.source_particles,
            "source_classes": self.source_classes,
            "target_particles": self.target_particles,
            "target_classes": self.target_classes,
            "particle_budget": self.particle_budget,
            "enumeration_budget": self.enumeration_budget,
            "cartesian_combinations": self.cartesian_combinations,
            "enumeration_complete": self.enumeration_complete,
            "generation_attempts": self.generation_attempts,
            "compatible_combinations_observed": (self.compatible_combinations_observed),
            "compatible_combinations_lower_bound": (
                self.compatible_combinations_observed
            ),
            "incompatible_combinations_observed": (
                self.incompatible_combinations_observed
            ),
            "materialized_combinations": self.materialized_combinations,
            "unmaterialized_compatible_combinations_lower_bound": (
                self.unmaterialized_compatible_combinations_lower_bound
            ),
            "novel_recombinations": self.novel_recombinations,
            "capacity_matched": self.capacity_matched,
        }


@dataclass(frozen=True)
class FactorizedCandidateBank(Sequence[JointGaugeHypothesis]):
    """An audited capacity-matched executable view of the product posterior."""

    hypotheses: tuple[JointGaugeHypothesis, ...]
    factor_rows: tuple[tuple[str, ...], ...]
    metrics: FactorizedBankMetrics
    prior_marginals: FactorMarginals

    def __post_init__(self) -> None:
        if len(self.hypotheses) != len(self.factor_rows):
            raise ValueError("bank hypotheses and factor rows must have equal length")
        if tuple(factor_keys(item) for item in self.hypotheses) != self.factor_rows:
            raise ValueError("bank factor rows do not match their hypotheses")
        if not self.metrics.capacity_matched:
            raise FactorizedControlRefusal("factorized bank is not capacity matched")
        for index, marginal in enumerate(self.prior_marginals):
            materialized_support = {row[index] for row in self.factor_rows}
            if set(marginal.support) != materialized_support:
                raise FactorizedControlRefusal(
                    f"{marginal.name} MDL support drifted from the control bank"
                )

    @overload
    def __getitem__(self, index: int) -> JointGaugeHypothesis: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[JointGaugeHypothesis]: ...

    def __getitem__(
        self, index: int | slice
    ) -> JointGaugeHypothesis | Sequence[JointGaugeHypothesis]:
        return self.hypotheses[index]

    def __len__(self) -> int:
        return len(self.hypotheses)

    @property
    def marginals(self) -> FactorMarginals:
        # The registered MDL prior is frozen from the deduplicated source bank
        # before recombination.  Domain materialization independently enforces
        # the same best-per-factor derivations in the executable control bank.
        return self.prior_marginals


def _class_count(candidates: Sequence[JointGaugeHypothesis]) -> int:
    return len({candidate.gauge_equivalence_key for candidate in candidates})


def _covers_domains(
    candidates: Sequence[JointGaugeHypothesis],
    domains: Sequence[_FactorDomain],
) -> bool:
    rows = tuple(factor_keys(item) for item in candidates)
    return all(
        {row[index] for row in rows} == set(domain.keys)
        for index, domain in enumerate(domains)
    )


def _uses_best_factor_derivations(
    candidate: JointGaugeHypothesis,
    domains: Sequence[_FactorDomain],
) -> bool:
    keys = factor_keys(candidate)
    priors = factor_log_prior_components(candidate)
    return all(
        prior == domain.log_prior(key)
        for domain, key, prior in zip(domains, keys, priors, strict=True)
    )


def _select_recombined_bank(
    *,
    source: tuple[JointGaugeHypothesis, ...],
    pool: tuple[JointGaugeHypothesis, ...],
    domains: tuple[_FactorDomain, ...],
    target_classes: int,
    search_budget: int,
) -> tuple[JointGaugeHypothesis, ...]:
    """Find a novel exact-capacity bank or refuse.

    The original bank is a known feasible point.  We perform bounded swaps
    against the fully enumerated typed pool and require exact particle count,
    exact gauge-class count and complete factor-domain coverage after every
    accepted solution.  No approximate class count is reported as comparable.
    """

    source_hashes = {item.canonical_hash for item in source}
    novel = tuple(item for item in pool if item.canonical_hash not in source_hashes)
    if not novel:
        raise FactorizedControlRefusal(
            "no novel typed recombination exists at the registered factor domains"
        )

    attempts = 0
    maximum_swaps = min(len(source), len(novel), 4)
    source_indices = tuple(range(len(source)))
    for swap_count in range(1, maximum_swaps + 1):
        for removed_indices in itertools.combinations(source_indices, swap_count):
            retained = tuple(
                item
                for index, item in enumerate(source)
                if index not in removed_indices
            )
            for additions in itertools.combinations(novel, swap_count):
                attempts += 1
                if attempts > search_budget:
                    raise FactorizedControlRefusal(
                        "typed bank selection exhausted its declared search budget"
                    )
                trial_by_hash = {
                    item.canonical_hash: item for item in (*retained, *additions)
                }
                if len(trial_by_hash) != len(source):
                    continue
                trial = tuple(trial_by_hash[key] for key in sorted(trial_by_hash))
                if not all(
                    _uses_best_factor_derivations(item, domains) for item in trial
                ):
                    continue
                if _class_count(trial) != target_classes:
                    continue
                if not _covers_domains(trial, domains):
                    continue
                return trial
    raise FactorizedControlRefusal(
        "no compatible bank preserves particle, gauge-class and factor-domain capacity"
    )


def _exact_recombined_bank(
    candidates: Sequence[JointGaugeHypothesis],
    *,
    source: Sequence[JointGaugeHypothesis],
    domains: Sequence[_FactorDomain],
    target_classes: int,
) -> tuple[JointGaugeHypothesis, ...] | None:
    """Return a deterministically ordered bank only if every capacity is exact."""

    unique = {item.canonical_hash: item for item in candidates}
    if len(unique) != len(source):
        return None
    ordered = tuple(unique[key] for key in sorted(unique))
    if _class_count(ordered) != target_classes:
        return None
    if not _covers_domains(ordered, domains):
        return None
    source_hashes = {item.canonical_hash for item in source}
    if all(item.canonical_hash in source_hashes for item in ordered):
        return None
    return ordered


def _bounded_rotation_pool(
    *,
    source: tuple[JointGaugeHypothesis, ...],
    domains: tuple[_FactorDomain, ...],
    target_classes: int,
    attempt_budget: int,
) -> tuple[
    tuple[JointGaugeHypothesis, ...],
    tuple[JointGaugeHypothesis, ...] | None,
    int,
    int,
    int,
]:
    """Generate a deterministic typed pool without traversing the product.

    Each schedule preserves the source multiset for four factors and rotates
    the fifth against its registered interface partner.  Schedules are tried
    in the fixed order ``D/G``, ``D/A`` and ``F/Tau`` for each rotation.  A
    complete schedule therefore preserves every factor domain by construction;
    it is accepted only if particle and gauge-class counts also match exactly.
    At most ``attempt_budget`` component tuples are inspected.
    """

    generated = {item.canonical_hash: item for item in source}
    source_rows = tuple(factor_keys(item) for item in source)
    compatible_rows = set(source_rows)
    incompatible_rows: set[tuple[str, ...]] = set()
    domain_values = tuple(dict(domain.entries) for domain in domains)
    source_parts = tuple(
        tuple(domain_values[index][key] for index, key in enumerate(row))
        for row in source_rows
    )
    # Rotate the second member of each registered compatibility interface.
    pair_schedules = (
        ("dynamics_goal", 1),
        ("dynamics_option", 4),
        ("frame_transport", 3),
    )
    attempts = 0
    source_count = len(source)
    for rotation in range(1, source_count):
        for _label, rotated_factor in pair_schedules:
            if attempts + source_count > attempt_budget:
                pool = tuple(generated[key] for key in sorted(generated))
                return (
                    pool,
                    None,
                    attempts,
                    len(compatible_rows),
                    len(incompatible_rows),
                )
            schedule: list[JointGaugeHypothesis] = []
            schedule_complete = True
            for index in range(source_count):
                rotated_index = (index + rotation) % source_count
                parts = list(source_parts[index])
                parts[rotated_factor] = source_parts[rotated_index][rotated_factor]
                proposed_row = list(source_rows[index])
                proposed_row[rotated_factor] = source_rows[rotated_index][
                    rotated_factor
                ]
                proposed_keys = tuple(proposed_row)
                attempts += 1
                dynamics, goal, frame, transports, option = parts
                if not typed_recombination_compatible(
                    dynamics=dynamics,
                    goal=goal,
                    frame=frame,
                    transports=transports,
                    option=option,
                ):
                    incompatible_rows.add(proposed_keys)
                    schedule_complete = False
                    continue
                candidate = _recombine(tuple(parts))
                if factor_keys(candidate) != proposed_keys:
                    raise AssertionError(
                        "rotated component keys changed during typed recombination"
                    )
                compatible_rows.add(proposed_keys)
                generated.setdefault(candidate.canonical_hash, candidate)
                schedule.append(candidate)
            if schedule_complete:
                exact = _exact_recombined_bank(
                    schedule,
                    source=source,
                    domains=domains,
                    target_classes=target_classes,
                )
                if exact is not None:
                    pool = tuple(generated[key] for key in sorted(generated))
                    return (
                        pool,
                        exact,
                        attempts,
                        len(compatible_rows),
                        len(incompatible_rows),
                    )
    pool = tuple(generated[key] for key in sorted(generated))
    return (
        pool,
        None,
        attempts,
        len(compatible_rows),
        len(incompatible_rows),
    )


def capacity_matched_factorized_bank(
    candidates: Sequence[JointGaugeHypothesis],
    *,
    particle_budget: int = MAXIMUM_CONTROL_CANDIDATES,
    enumeration_budget: int = MAXIMUM_CONTROL_ENUMERATIONS,
) -> FactorizedCandidateBank:
    """Build an audited, outcome-independent typed recombination bank.

    The function refuses instead of truncating, inventing gauge classes, or
    reporting an incompatible bank as a comparable-capacity control.
    """

    budget = max(1, min(MAXIMUM_CONTROL_CANDIDATES, int(particle_budget)))
    enumeration_limit = max(1, int(enumeration_budget))
    unique: dict[str, JointGaugeHypothesis] = {}
    for item in candidates:
        previous = unique.get(item.canonical_hash)
        if (
            previous is None
            or item.log_prior > previous.log_prior
            or (
                item.log_prior == previous.log_prior
                and _canonical_json(
                    {
                        "program_id": item.world_program.program_id,
                        "parent_hash": item.world_program.parent_hash,
                        "provenance": item.world_program.provenance,
                        "edit_distance": item.world_program.edit_distance,
                    }
                )
                < _canonical_json(
                    {
                        "program_id": previous.world_program.program_id,
                        "parent_hash": previous.world_program.parent_hash,
                        "provenance": previous.world_program.provenance,
                        "edit_distance": previous.world_program.edit_distance,
                    }
                )
            )
        ):
            unique[item.canonical_hash] = item
    source = tuple(unique[key] for key in sorted(unique))
    if len(source) < 2:
        raise FactorizedControlRefusal(
            "factorized control requires at least two distinct source particles"
        )
    if len(source) > budget:
        raise FactorizedControlRefusal(
            f"source has {len(source)} particles but budget is {budget}"
        )
    if any(
        not typed_recombination_compatible(
            dynamics=item.world_program,
            goal=item.world_program,
            frame=item.frame,
            transports=item.transports,
            option=item.option,
        )
        for item in source
    ):
        raise FactorizedControlRefusal(
            "source bank itself violates a registered factor interface"
        )

    domains = _factor_domains(source)
    if any(not domain.entries for domain in domains):
        raise FactorizedControlRefusal("all five factor domains must be non-empty")
    cartesian_count = math.prod(len(domain.entries) for domain in domains)
    source_marginals = FactorMarginals.mdl_from_hypotheses(source)
    source_classes = _class_count(source)
    enumeration_complete = cartesian_count <= enumeration_limit
    selected: tuple[JointGaugeHypothesis, ...] | None = None
    if enumeration_complete:
        generated: dict[str, JointGaugeHypothesis] = {}
        incompatible_observed = 0
        for entries in itertools.product(*(domain.entries for domain in domains)):
            parts = tuple(value for _, value in entries)
            dynamics, goal, frame, transports, option = parts
            if not typed_recombination_compatible(
                dynamics=dynamics,
                goal=goal,
                frame=frame,
                transports=transports,
                option=option,
            ):
                incompatible_observed += 1
                continue
            candidate = _recombine(parts)
            generated.setdefault(candidate.canonical_hash, candidate)
        pool = tuple(generated[key] for key in sorted(generated))
        generation_attempts = cartesian_count
        compatible_observed = len(pool)
    else:
        (
            pool,
            selected,
            generation_attempts,
            compatible_observed,
            incompatible_observed,
        ) = _bounded_rotation_pool(
            source=source,
            domains=domains,
            target_classes=source_classes,
            attempt_budget=enumeration_limit,
        )
    if len(pool) < len(source):
        raise FactorizedControlRefusal(
            "bounded typed pool cannot preserve source particle capacity"
        )

    if selected is None:
        selected = _select_recombined_bank(
            source=source,
            pool=pool,
            domains=domains,
            target_classes=source_classes,
            search_budget=enumeration_limit,
        )
    target_classes = _class_count(selected)
    source_hashes = {item.canonical_hash for item in source}
    metrics = FactorizedBankMetrics(
        source_particles=len(source),
        source_classes=source_classes,
        target_particles=len(selected),
        target_classes=target_classes,
        particle_budget=budget,
        enumeration_budget=enumeration_limit,
        cartesian_combinations=cartesian_count,
        enumeration_complete=enumeration_complete,
        generation_attempts=generation_attempts,
        compatible_combinations_observed=compatible_observed,
        incompatible_combinations_observed=incompatible_observed,
        materialized_combinations=len(selected),
        unmaterialized_compatible_combinations_lower_bound=max(
            0, compatible_observed - len(selected)
        ),
        novel_recombinations=sum(
            item.canonical_hash not in source_hashes for item in selected
        ),
    )
    if not metrics.capacity_matched or metrics.novel_recombinations <= 0:
        raise FactorizedControlRefusal(
            "candidate bank failed exact comparable-capacity checks"
        )
    rows = tuple(factor_keys(item) for item in selected)
    return FactorizedCandidateBank(
        selected,
        rows,
        metrics,
        source_marginals,
    )


def likelihood_allocation(
    particle: GaugeParticle,
    *,
    commutativity_penalty: float,
) -> tuple[float, ...]:
    """Allocate one scored joint likelihood exactly once across five factors."""

    physical = float(particle.latest_physical_log_likelihood)
    frame_net = float(particle.latest_frame_log_likelihood) - (
        float(commutativity_penalty) * float(particle.latest_commutativity_penalty)
    )
    option = float(particle.latest_option_log_likelihood)
    dynamics = 0.5 * physical
    goal = physical - dynamics
    frame = 0.5 * frame_net
    transport = frame_net - frame
    return (dynamics, goal, frame, transport, option)


class FactorizedGaugeProgramPosterior(GaugeProgramPosterior):
    """Variational ``q(D)q(G)q(F)q(Tau)q(A)`` scientific control."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._marginals: FactorMarginals | None = None
        self._bank_metrics: FactorizedBankMetrics | None = None
        self._factorized_updates = 0
        self._last_likelihood_decomposition_error = 0.0
        self._materialized_product_mass = 0.0

    @property
    def factor_marginals(self) -> FactorMarginals:
        if self._marginals is None:
            raise RuntimeError("factorized posterior has not been seeded")
        return self._marginals

    @property
    def bank_metrics(self) -> FactorizedBankMetrics:
        if self._bank_metrics is None:
            raise RuntimeError("factorized posterior has not been seeded")
        return self._bank_metrics

    @property
    def factorized_updates(self) -> int:
        return self._factorized_updates

    @property
    def last_likelihood_decomposition_error(self) -> float:
        return self._last_likelihood_decomposition_error

    @property
    def materialized_product_mass(self) -> float:
        return self._materialized_product_mass

    def seed(
        self,
        hypotheses: Sequence[JointGaugeHypothesis],
        *,
        initial_states: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(hypotheses, FactorizedCandidateBank):
            raise FactorizedControlRefusal(
                "FactorizedGaugeProgramPosterior.seed requires an audited "
                "FactorizedCandidateBank"
            )
        if hypotheses.metrics.target_classes > self.maximum_classes:
            raise FactorizedControlRefusal(
                "posterior class budget is smaller than the audited control bank"
            )
        super().seed(hypotheses.hypotheses, initial_states=initial_states)
        if len(self._particles) != hypotheses.metrics.target_particles:
            raise FactorizedControlRefusal(
                "base posterior changed the audited particle capacity during seed"
            )
        if len(self.classes) != hypotheses.metrics.target_classes:
            raise FactorizedControlRefusal(
                "base posterior changed the audited gauge-class capacity during seed"
            )
        self._marginals = hypotheses.marginals
        self._bank_metrics = hypotheses.metrics
        self._factorized_updates = 0
        self._last_likelihood_decomposition_error = 0.0
        self._materialize_product(initialize_prior=True)

    def observe(self, bundle: Any) -> GaugeUpdate:
        update = super().observe(bundle)
        if bool(getattr(bundle, "reset", False)):
            # start_branch refreshes latent execution states only.  A reset is
            # not physical evidence and must not touch q or its update count.
            exact = replace(
                update,
                classes_after=len(self.classes),
                collapsed=False,
            )
            self._last_update = exact
            return exact

        self._update_marginals_from_scored_bank()
        self._materialize_product()
        self._factorized_updates += 1
        exact = replace(
            update,
            classes_after=len(self.classes),
            collapsed=False,
        )
        self._last_update = exact
        return exact

    def _maybe_collapse(self) -> bool:
        # Collapse selects one joint gauge class and therefore invalidates the
        # registered five-factor control.  It is intentionally unavailable.
        self._collapsed = False
        self._top_class_key = ""
        self._top_class_streak = 0
        return False

    def _update_marginals_from_scored_bank(self) -> None:
        marginals = self.factor_marginals
        if not self._particles:
            raise FactorizedControlRefusal("cannot update an empty factorized bank")
        rows = tuple(factor_keys(item.hypothesis) for item in self._particles)
        allocations = tuple(
            likelihood_allocation(
                item,
                commutativity_penalty=self.commutativity_penalty,
            )
            for item in self._particles
        )
        errors = []
        for particle, allocation in zip(self._particles, allocations, strict=True):
            expected = (
                float(particle.latest_physical_log_likelihood)
                + float(particle.latest_frame_log_likelihood)
                - self.commutativity_penalty
                * float(particle.latest_commutativity_penalty)
                + float(particle.latest_option_log_likelihood)
            )
            errors.append(abs(math.fsum(allocation) - expected))
        self._last_likelihood_decomposition_error = max(errors, default=0.0)
        if self._last_likelihood_decomposition_error > 1e-10:
            raise AssertionError("factor likelihood allocation lost joint likelihood")

        updated: list[FactorDistribution] = []
        for factor_index, marginal in enumerate(marginals):
            scores: dict[str, float] = {}
            for factor_key, current_log_probability in marginal.entries:
                conditional_rows: list[tuple[float, float]] = []
                for row, allocation in zip(rows, allocations, strict=True):
                    if row[factor_index] != factor_key:
                        continue
                    other_log_probability = math.fsum(
                        other.log_probability(row[other_index])
                        for other_index, other in enumerate(marginals)
                        if other_index != factor_index
                    )
                    conditional_rows.append(
                        (other_log_probability, allocation[factor_index])
                    )
                if not conditional_rows:
                    raise FactorizedControlRefusal(
                        f"materialized bank has no coverage for {marginal.name}="
                        f"{factor_key}"
                    )
                maximum_log_weight = max(item[0] for item in conditional_rows)
                stable_rows = tuple(
                    (math.exp(log_weight - maximum_log_weight), value)
                    for log_weight, value in conditional_rows
                )
                denominator = math.fsum(weight for weight, _value in stable_rows)
                expected_log_likelihood = (
                    math.fsum(weight * value for weight, value in stable_rows)
                    / denominator
                )
                scores[factor_key] = current_log_probability + expected_log_likelihood
            updated.append(FactorDistribution.from_log_scores(marginal.name, scores))
        self._marginals = FactorMarginals(*updated)

    def _materialize_product(self, *, initialize_prior: bool = False) -> None:
        rows = tuple(factor_keys(item.hypothesis) for item in self._particles)
        products = factorized_log_weights(rows, self.factor_marginals)
        explicit_mass = math.fsum(math.exp(value) for value in products)
        if explicit_mass > 1.0 + _MASS_TOLERANCE:
            raise AssertionError("materialized factor products exceed unit mass")
        explicit_mass = min(1.0, max(0.0, explicit_mass))
        self._particles = [
            replace(
                particle,
                log_prior=(weight if initialize_prior else particle.log_prior),
                log_weight=weight,
            )
            for particle, weight in zip(self._particles, products, strict=True)
        ]
        residual = max(0.0, 1.0 - explicit_mass)
        self._residual_log_mass = (
            float("-inf") if residual == 0.0 else math.log(residual)
        )
        self._materialized_product_mass = explicit_mass

    def snapshot(self, *, maximum_classes: int = 8) -> Mapping[str, Any]:
        payload = dict(super().snapshot(maximum_classes=maximum_classes))
        metrics = self.bank_metrics
        payload.update(
            {
                "posterior_family": "strict_five_factor_variational_control",
                "factorized_updates": self._factorized_updates,
                "collapse_policy": "disabled_preserve_factorization",
                "factor_support_sizes": dict(self.factor_marginals.support_sizes),
                "materialized_product_mass": round(self._materialized_product_mass, 10),
                "likelihood_decomposition_max_error": round(
                    self._last_likelihood_decomposition_error, 14
                ),
                "bank": dict(metrics.as_dict()),
            }
        )
        return payload


__all__ = [
    "FACTOR_NAMES",
    "FactorDistribution",
    "FactorMarginals",
    "FactorizedBankMetrics",
    "FactorizedCandidateBank",
    "FactorizedControlRefusal",
    "FactorizedGaugeProgramPosterior",
    "capacity_matched_factorized_bank",
    "factor_keys",
    "factor_log_prior_components",
    "factorized_log_weights",
    "likelihood_allocation",
    "typed_recombination_compatible",
]
