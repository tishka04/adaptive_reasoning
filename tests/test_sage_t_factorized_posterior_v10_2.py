from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest

from theory.sage_t.contracts import (
    AbstractState,
    ActionBinding,
    ActionCandidate,
    Effect,
    Expression,
    GoalRule,
    JointProgramHypothesis,
    ObjectSchema,
    PredictionPacket,
    ProgressRule,
    TerminalRule,
    TransitionRule,
)
from theory.sage_t.factorized_posterior_v10_2 import (
    FactorDistribution,
    FactorizedControlRefusal,
    FactorizedGaugeProgramPosterior,
    FactorMarginals,
    capacity_matched_factorized_bank,
    factor_keys,
    factor_log_prior_components,
    factorized_log_weights,
    likelihood_allocation,
    typed_recombination_compatible,
)
from theory.sage_t.gauge_inference_v10_2 import (
    GaugeParticle,
    JointGaugeHypothesis,
)
from theory.sage_t.observer_frames_v10_2 import observer_frame_spec


@dataclass(frozen=True)
class _Frame:
    frame_id: str

    @property
    def canonical_payload(self) -> dict[str, object]:
        return observer_frame_spec(self.frame_id).canonical_payload


@dataclass(frozen=True)
class _Transport:
    source_frame_id: str
    target_frame_id: str
    orbit: str = ""

    @property
    def canonical_payload(self) -> dict[str, str]:
        return {
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "orbit": self.orbit,
        }

    @property
    def node_count(self) -> int:
        return 1

    @property
    def certifies_gauge_equivalence(self) -> bool:
        return bool(self.orbit)

    @property
    def gauge_equivalence_key(self) -> str | None:
        return self.orbit or None

    def transport_prediction(self, packet: PredictionPacket) -> PredictionPacket:
        return packet

    def transport_action(self, action: ActionCandidate) -> ActionCandidate:
        return action


@dataclass(frozen=True)
class _Option:
    option_id: str
    action_schemas: tuple[str, ...] = ("action1",)
    initial_state: int = 0

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "action_schemas": self.action_schemas,
            "states": (0,),
        }

    @property
    def node_count(self) -> int:
        return 1

    def allowed_action_schemas(self, _state: int) -> tuple[str, ...]:
        return self.action_schemas

    def observe(
        self,
        state: int,
        _action: ActionCandidate,
        *,
        events: tuple[str, ...],
    ) -> int:
        del events
        return state


@dataclass(frozen=True)
class _Projection:
    spec: _Frame
    state_before: AbstractState
    action: ActionCandidate
    state_after: AbstractState
    observation: PredictionPacket


@dataclass(frozen=True)
class _Bundle:
    event_id: str
    action: ActionCandidate
    common_observation: PredictionPacket
    projections: tuple[_Projection, ...]
    events: tuple[str, ...] = ()
    reset: bool = False


def _program(
    index: int,
    *,
    action_name: str = "ACTION1",
    role: str = "target",
    goal_role: str | None = None,
    goal_variant: int | None = None,
) -> JointProgramHypothesis:
    dynamics_predicate = ("changed", "moved")[index % 2]
    goal_index = index if goal_variant is None else goal_variant
    goal_predicate = ("changed", "moved")[goal_index % 2]
    operator = ("apply", "move")[index % 2]
    selected_goal_role = role if goal_role is None else goal_role
    goal_expression = Expression(
        op="exists",
        args=(Expression.fact(goal_predicate, "?item"),),
        variable="?item",
        role=selected_goal_role,
    )
    return JointProgramHypothesis(
        program_id=f"program_{index}_{action_name.lower()}_{role}_{goal_index}",
        object_schema=ObjectSchema(("object", role)),
        action_bindings=(ActionBinding(action_name, operator, target_role=role),),
        transition_rules=(
            TransitionRule(
                rule_id=f"rule_{index}",
                action_operator=operator,
                condition=Expression.constant(float(index + 1)),
                effects=(
                    Effect(
                        "assert",
                        predicate=dynamics_predicate,
                        terms=("$target",),
                    ),
                ),
            ),
        ),
        progress_rule=ProgressRule(Expression.constant(float(goal_index))),
        terminal_rules=(
            TerminalRule(Expression.fact("game_over"), outcome="game_over"),
        ),
        goal_rule=GoalRule(
            goal_expression,
            family=f"goal_family_{goal_index}_{selected_goal_role}",
        ),
    )


def _hypothesis(
    dynamics: JointProgramHypothesis,
    goal: JointProgramHypothesis,
    *,
    frame: _Frame | None = None,
    transports: tuple[_Transport, ...] = (),
    option: _Option | None = None,
) -> JointGaugeHypothesis:
    world = replace(
        dynamics,
        progress_rule=goal.progress_rule,
        terminal_rules=goal.terminal_rules,
        goal_rule=goal.goal_rule,
    )
    return JointGaugeHypothesis(
        world_program=world,
        frame=frame or _Frame("root_only"),
        transports=transports,
        option=option or _Option("repeat"),
    )


def _correlated_source() -> tuple[JointGaugeHypothesis, ...]:
    first = _program(0)
    second = _program(1)
    return (_hypothesis(first, first), _hypothesis(second, second))


def _bundle(event_id: str, *, reset: bool = False) -> _Bundle:
    action = ActionCandidate("ACTION1")
    state = AbstractState()
    observation = PredictionPacket(
        progress_mean=0.0,
        terminal_probability=0.0,
        goal_probability=1.0,
        known_channels=frozenset({"progress", "terminal", "goal"}),
        state_after=state,
    )
    projection = _Projection(
        spec=_Frame("root_only"),
        state_before=state,
        action=action,
        state_after=state,
        observation=PredictionPacket(state_after=state),
    )
    return _Bundle(
        event_id=event_id,
        action=action,
        common_observation=observation,
        projections=(projection,),
        reset=reset,
    )


def test_diagonal_support_materialization_is_idempotent_and_keeps_residual() -> None:
    rows = (
        ("d0", "g0", "f", "tau", "a"),
        ("d1", "g1", "f", "tau", "a"),
    )
    marginals = FactorMarginals.uniform_from_rows(rows)
    first = factorized_log_weights(rows, marginals)
    second = factorized_log_weights(rows, marginals)
    assert second == first
    assert [math.exp(value) for value in first] == pytest.approx([0.25, 0.25])
    assert 1.0 - sum(math.exp(value) for value in first) == pytest.approx(0.5)
    with pytest.raises(TypeError, match="joint weights cannot identify"):
        factorized_log_weights(rows, (math.log(0.6), math.log(0.4)))  # type: ignore[arg-type]


def test_control_bank_is_typed_novel_and_exactly_capacity_matched() -> None:
    source = _correlated_source()
    bank = capacity_matched_factorized_bank(source)
    source_rows = {factor_keys(item) for item in source}
    assert len(bank) == len(source)
    assert bank.metrics.source_particles == bank.metrics.target_particles == 2
    assert bank.metrics.source_classes == bank.metrics.target_classes == 2
    assert bank.metrics.cartesian_combinations == 4
    assert bank.metrics.enumeration_complete is True
    assert bank.metrics.compatible_combinations_observed == 4
    assert bank.metrics.novel_recombinations == 2
    assert all(factor_keys(item) not in source_rows for item in bank)
    assert bank.metrics.capacity_matched


def test_factorized_control_preserves_exact_mdl_component_priors() -> None:
    first, second = _correlated_source()
    first = replace(
        first,
        world_program=replace(first.world_program, edit_distance=1),
    )
    second = replace(
        second,
        world_program=replace(second.world_program, edit_distance=4),
    )
    for candidate in (first, second):
        assert math.fsum(factor_log_prior_components(candidate)) == pytest.approx(
            candidate.log_prior
        )

    bank = capacity_matched_factorized_bank((first, second))
    dynamics_edit_by_key = {
        factor_keys(candidate)[0]: candidate.world_program.edit_distance
        for candidate in (first, second)
    }
    for candidate in bank:
        assert (
            candidate.world_program.edit_distance
            == dynamics_edit_by_key[factor_keys(candidate)[0]]
        )
    probabilities = bank.marginals.dynamics.probabilities
    cheap_key = factor_keys(first)[0]
    expensive_key = factor_keys(second)[0]
    assert probabilities[cheap_key] > probabilities[expensive_key]
    assert probabilities[cheap_key] / probabilities[expensive_key] == pytest.approx(
        math.exp(3.0)
    )

    posterior = FactorizedGaugeProgramPosterior()
    posterior.seed(bank)
    expected = factorized_log_weights(bank.factor_rows, bank.marginals)
    assert tuple(item.log_prior for item in posterior.particles) == pytest.approx(
        expected
    )


def test_factorized_bank_freezes_best_source_mdl_derivation_per_factor() -> None:
    dynamics_best = replace(_program(0), edit_distance=0)
    dynamics_costlier = replace(_program(0), edit_distance=9)
    alternate_dynamics = replace(_program(1), edit_distance=2)
    goal_zero = _program(0)
    goal_one = _program(1)
    source = (
        _hypothesis(dynamics_best, goal_zero),
        _hypothesis(dynamics_costlier, goal_one),
        _hypothesis(alternate_dynamics, goal_zero),
    )
    expected = FactorMarginals.mdl_from_hypotheses(source)

    bank = capacity_matched_factorized_bank(source)

    assert bank.marginals == expected
    dynamics_key = factor_keys(source[0])[0]
    selected_derivations = tuple(
        candidate.world_program.edit_distance
        for candidate in bank
        if factor_keys(candidate)[0] == dynamics_key
    )
    assert selected_derivations
    assert selected_derivations == (0,)
    probabilities = bank.marginals.dynamics.probabilities
    assert probabilities[dynamics_key] > probabilities[factor_keys(source[2])[0]]


def test_massive_256_particle_product_uses_bounded_deterministic_rotations() -> None:
    programs = tuple(_program(index) for index in range(256))
    source = tuple(
        _hypothesis(
            program,
            program,
            frame=_Frame(
                "root_only" if index % 2 == 0 else "allocentric_object_relative"
            ),
        )
        for index, program in enumerate(programs)
    )
    bank = capacity_matched_factorized_bank(source)
    assert len(bank) == 256
    assert bank.metrics.source_particles == bank.metrics.target_particles == 256
    assert bank.metrics.source_classes == bank.metrics.target_classes == 256
    assert bank.metrics.cartesian_combinations == 256 * 256 * 2
    assert bank.metrics.enumeration_complete is False
    assert bank.metrics.generation_attempts == 256
    assert bank.metrics.generation_attempts <= bank.metrics.enumeration_budget
    assert bank.metrics.compatible_combinations_observed >= 256
    assert bank.metrics.novel_recombinations == 256
    assert bank.metrics.capacity_matched
    repeated = capacity_matched_factorized_bank(source)
    assert tuple(item.canonical_hash for item in repeated) == tuple(
        item.canonical_hash for item in bank
    )
    assert repeated.metrics == bank.metrics


def test_typed_interfaces_reject_option_frame_and_goal_mismatches() -> None:
    first = _program(0, action_name="ACTION1", goal_variant=0)
    second = _program(1, action_name="ACTION2", goal_variant=0)
    option_correlated = (
        _hypothesis(first, first, option=_Option("a1", ("action1",))),
        _hypothesis(second, first, option=_Option("a2", ("action2",))),
    )
    with pytest.raises(FactorizedControlRefusal, match="novel typed recombination"):
        capacity_matched_factorized_bank(option_correlated)

    common = _program(0)
    frame_correlated = (
        _hypothesis(
            common,
            common,
            frame=_Frame("root_only"),
            transports=(_Transport("root_only", "action_aligned_relational"),),
        ),
        _hypothesis(
            common,
            common,
            frame=_Frame("allocentric_object_relative"),
            transports=(
                _Transport(
                    "allocentric_object_relative",
                    "action_aligned_relational",
                ),
            ),
        ),
    )
    with pytest.raises(FactorizedControlRefusal, match="novel typed recombination"):
        capacity_matched_factorized_bank(frame_correlated)

    dynamics_zero = _program(0, role="role_zero", goal_role="role_zero")
    dynamics_one = _program(1, role="role_one", goal_role="role_one")
    assert not typed_recombination_compatible(
        dynamics=dynamics_zero,
        goal=dynamics_one,
        frame=_Frame("root_only"),
        transports=(),
        option=_Option("repeat"),
    )
    role_correlated = (
        _hypothesis(dynamics_zero, dynamics_zero),
        _hypothesis(dynamics_one, dynamics_one),
    )
    with pytest.raises(FactorizedControlRefusal, match="novel typed recombination"):
        capacity_matched_factorized_bank(role_correlated)


def test_frame_transport_accepts_multihop_chain_and_refuses_disconnected_map() -> None:
    program = _program(0)
    frame = _Frame("root_only")
    option = _Option("repeat")
    connected = (
        _Transport("root_only", "allocentric_object_relative"),
        _Transport("allocentric_object_relative", "action_aligned_relational"),
        _Transport("action_aligned_relational", "action_rooted_topological"),
    )

    assert typed_recombination_compatible(
        dynamics=program,
        goal=program,
        frame=frame,
        transports=connected,
        option=option,
    )
    disconnected = (
        _Transport("root_only", "allocentric_object_relative"),
        _Transport("action_aligned_relational", "action_rooted_topological"),
    )
    assert not typed_recombination_compatible(
        dynamics=program,
        goal=program,
        frame=frame,
        transports=disconnected,
        option=option,
    )


def test_canonical_payload_is_mandatory_without_string_fallback() -> None:
    @dataclass(frozen=True)
    class _NonCanonicalOption:
        action_schemas: tuple[str, ...] = ("action1",)
        initial_state: int = 0

        def allowed_action_schemas(self, _state: int) -> tuple[str, ...]:
            return self.action_schemas

        def observe(
            self,
            state: int,
            _action: ActionCandidate,
            *,
            events: tuple[str, ...],
        ) -> int:
            del events
            return state

    program = _program(0)
    candidate = JointGaugeHypothesis(
        world_program=program,
        frame=_Frame("root_only"),
        transports=(),
        option=_NonCanonicalOption(),
    )
    with pytest.raises(TypeError, match="canonical_payload"):
        factor_keys(candidate)


def test_posterior_stores_five_marginals_and_unmaterialized_mass() -> None:
    bank = capacity_matched_factorized_bank(_correlated_source())
    posterior = FactorizedGaugeProgramPosterior()
    posterior.seed(bank)
    marginals = posterior.factor_marginals
    assert tuple(item.name for item in marginals) == (
        "dynamics",
        "goal",
        "frame",
        "transport",
        "option",
    )
    assert sum(marginals.dynamics.probabilities.values()) == pytest.approx(1.0)
    assert sum(marginals.goal.probabilities.values()) == pytest.approx(1.0)
    assert sum(item.probability for item in posterior.particles) == pytest.approx(0.5)
    assert posterior.residual_mass == pytest.approx(0.5)
    assert (
        posterior.materialized_product_mass + posterior.residual_mass
        == pytest.approx(1.0)
    )
    snapshot = posterior.snapshot()
    assert snapshot["posterior_family"] == "strict_five_factor_variational_control"
    assert snapshot["bank"]["capacity_matched"] is True
    assert snapshot["bank"]["target_particles"] == 2
    assert snapshot["bank"]["target_classes"] == 2


def test_reset_does_not_update_and_observation_preserves_factorization() -> None:
    bank = capacity_matched_factorized_bank(_correlated_source())
    posterior = FactorizedGaugeProgramPosterior()
    posterior.seed(bank)
    before_marginals = posterior.factor_marginals
    before_weights = tuple(item.log_weight for item in posterior.particles)
    reset_update = posterior.observe(_bundle("reset-0", reset=True))
    assert posterior.factor_marginals == before_marginals
    assert tuple(item.log_weight for item in posterior.particles) == before_weights
    assert posterior.factorized_updates == 0
    assert reset_update.physical_scored_particles == 0
    assert reset_update.classes_after == len(posterior.classes)

    update = posterior.observe(_bundle("physical-1"))
    assert posterior.factorized_updates == 1
    assert posterior.factor_marginals != before_marginals
    assert posterior.last_likelihood_decomposition_error <= 1e-12
    assert update.classes_after == len(posterior.classes)
    assert update.collapsed is False
    assert posterior.collapsed is False
    assert sum(
        item.probability for item in posterior.particles
    ) + posterior.residual_mass == pytest.approx(1.0)


def test_factor_update_stays_finite_for_extremely_concentrated_marginals() -> None:
    bank = capacity_matched_factorized_bank(_correlated_source())
    posterior = FactorizedGaugeProgramPosterior()
    posterior.seed(bank)
    concentrated = []
    for marginal in posterior.factor_marginals:
        concentrated.append(
            FactorDistribution.from_log_scores(
                marginal.name,
                {
                    key: (0.0 if index == 0 else -2_000.0)
                    for index, key in enumerate(marginal.support)
                },
            )
        )
    posterior._marginals = FactorMarginals(*concentrated)
    posterior._materialize_product()
    update = posterior.observe(_bundle("concentrated-physical"))
    assert update.classes_after == len(posterior.classes)
    assert all(
        math.isfinite(value)
        for marginal in posterior.factor_marginals
        for _key, value in marginal.entries
    )


def test_likelihood_terms_are_allocated_exactly_once() -> None:
    candidate = _correlated_source()[0]
    particle = GaugeParticle(
        hypothesis=candidate,
        log_prior=0.0,
        log_weight=0.0,
        latest_physical_log_likelihood=-3.25,
        latest_frame_log_likelihood=-1.5,
        latest_commutativity_penalty=0.75,
        latest_option_log_likelihood=-2.0,
    )
    allocation = likelihood_allocation(particle, commutativity_penalty=1.25)
    expected = -3.25 - 1.5 - 1.25 * 0.75 - 2.0
    assert math.fsum(allocation) == pytest.approx(expected)
    assert math.fsum(allocation[:2]) == pytest.approx(-3.25)
    assert math.fsum(allocation[2:4]) == pytest.approx(-1.5 - 1.25 * 0.75)
    assert allocation[4] == -2.0


def test_uncertified_frame_class_capacity_is_exact_and_collapse_stays_disabled() -> (
    None
):
    program = _program(0)
    source = (
        _hypothesis(
            program,
            program,
            frame=_Frame("root_only"),
            transports=(
                _Transport(
                    "root_only",
                    "allocentric_object_relative",
                    orbit="shared",
                ),
            ),
        ),
        _hypothesis(
            program,
            program,
            frame=_Frame("allocentric_object_relative"),
            transports=(
                _Transport(
                    "allocentric_object_relative",
                    "root_only",
                    orbit="shared",
                ),
            ),
        ),
    )
    bank = capacity_matched_factorized_bank(source)
    assert bank.metrics.source_classes == bank.metrics.target_classes == 2
    posterior = FactorizedGaugeProgramPosterior()
    posterior.seed(bank)
    assert len(posterior.particles) == 2
    assert len(posterior.classes) == 2
    update = posterior.observe(_bundle("gauge-reset", reset=True))
    assert update.classes_before == update.classes_after == 2
    assert update.collapsed is False
    assert posterior.snapshot()["collapse_policy"] == "disabled_preserve_factorization"


def test_posterior_refuses_unaudited_or_underbudget_banks() -> None:
    source = _correlated_source()
    posterior = FactorizedGaugeProgramPosterior()
    with pytest.raises(FactorizedControlRefusal, match="audited"):
        posterior.seed(source)
    with pytest.raises(FactorizedControlRefusal, match="budget is 1"):
        capacity_matched_factorized_bank(source, particle_budget=1)
