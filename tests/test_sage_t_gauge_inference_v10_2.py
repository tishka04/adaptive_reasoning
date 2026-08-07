from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from theory.sage_t import gauge_inference_v10_2 as gauge_module
from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionBinding,
    ActionCandidate,
    Effect,
    Expression,
    GoalRule,
    GroundFact,
    JointProgramHypothesis,
    ObjectSchema,
    ObservedTransition,
    PredictionPacket,
    ProgressRule,
    TerminalRule,
    TransitionRule,
)
from theory.sage_t.frame_transport_v10_2 import (
    TransportCertificate,
    TransportMap,
    TransportOrbitWitness,
)
from theory.sage_t.gauge_inference_v10_2 import (
    GaugeDecisionEngine,
    GaugeProgramPosterior,
    JointGaugeHypothesis,
    rank_option_sequence_signatures,
)
from theory.sage_t.mixed_automata_v10_2 import (
    OptionAutomaton,
    OptionInitiationCondition,
    OptionState,
    OptionTransition,
    alternate,
    follow_relation_then_apply,
    prime_then_repeat,
    repeat,
    until_then,
)
from theory.sage_t.observer_frames_v10_2 import (
    ROOT_ONLY_FRAME,
    identity_projector,
    observer_frame_spec,
    project_observed_transition,
)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class FakeFrame:
    frame_id: str

    @property
    def canonical_payload(self) -> dict[str, object]:
        return dict(observer_frame_spec(self.frame_id).canonical_payload)


@dataclass(frozen=True)
class FakeTransport:
    source_frame: str
    target_frame: str
    certified_orbit: str = ""
    corrupt_prediction: bool = False
    nodes: int = 1

    @property
    def canonical_hash(self) -> str:
        return _hash(
            (
                self.source_frame,
                self.target_frame,
                self.certified_orbit,
                self.corrupt_prediction,
                self.nodes,
            )
        )

    @property
    def node_count(self) -> int:
        return self.nodes

    @property
    def certifies_gauge_equivalence(self) -> bool:
        return bool(self.certified_orbit)

    @property
    def gauge_equivalence_key(self) -> str | None:
        return self.certified_orbit or None

    def transport_prediction(self, packet: PredictionPacket) -> PredictionPacket:
        if not self.corrupt_prediction:
            return packet
        return replace(
            packet,
            object_deltas={"transport_mismatch": 1.0},
        )

    def transport_action(self, action: ActionCandidate) -> ActionCandidate:
        return action


@dataclass(frozen=True)
class FakeOption:
    option_id: str
    trigger: str = "never"
    initial_state: int = 0

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "trigger": self.trigger,
            "states": (0, 1),
        }

    @property
    def canonical_hash(self) -> str:
        return _hash(self.canonical_payload)

    @property
    def node_count(self) -> int:
        return 2

    def allowed_action_schemas(self, state: int) -> tuple[str, ...]:
        del state
        return ("ACTION1", "ACTION2")

    def observe(
        self,
        state: int,
        action: ActionCandidate,
        *,
        events: tuple[str, ...],
    ) -> int:
        del action
        return 1 if self.trigger in events else state


@dataclass(frozen=True)
class FakeProjection:
    spec: FakeFrame
    state_before: AbstractState
    action: ActionCandidate
    state_after: AbstractState
    observation: PredictionPacket


@dataclass(frozen=True)
class FakeBundle:
    event_id: str
    action: ActionCandidate
    common_observation: PredictionPacket
    projections: tuple[FakeProjection, ...]
    events: tuple[str, ...] = ()
    reset: bool = False


def _state(*, solved: bool = False) -> AbstractState:
    target = AbstractEntity("target", ("object", "target"), center=(1.0, 1.0))
    solved_fact = GroundFact("solved", ("target",))
    return AbstractState(
        entities=(target,),
        true_facts=frozenset(
            {
                GroundFact("exists", ("target",)),
                GroundFact("role", ("target", "target")),
                *((solved_fact,) if solved else ()),
            }
        ),
        false_facts=frozenset(
            {
                *((solved_fact,) if not solved else ()),
                GroundFact("game_over"),
                GroundFact("level_complete"),
            }
        ),
    )


def _program(program_id: str, *, solves: bool) -> JointProgramHypothesis:
    solved_count = Expression(
        op="count",
        args=(Expression.fact("solved", "?item"),),
        variable="?item",
        role="target",
    )
    effect = (
        Effect("assert", predicate="solved", terms=("$target",))
        if solves
        else Effect("assert", predicate="no_effect")
    )
    return JointProgramHypothesis(
        program_id=program_id,
        object_schema=ObjectSchema(("object", "target")),
        action_bindings=(ActionBinding("ACTION1", "apply", target_role="target"),),
        transition_rules=(
            TransitionRule(
                rule_id=f"rule_{program_id}",
                action_operator="apply",
                condition=Expression.constant(True),
                effects=(effect,),
            ),
        ),
        progress_rule=ProgressRule(solved_count),
        terminal_rules=(
            TerminalRule(Expression.fact("game_over"), outcome="game_over"),
        ),
        goal_rule=GoalRule(
            Expression(
                op="forall",
                args=(Expression.fact("solved", "?goal"),),
                variable="?goal",
                role="target",
            ),
            family="solve_targets",
        ),
    )


def _bundle(event_id: str, frames: tuple[str, ...] = ("root_only",)) -> FakeBundle:
    action = ActionCandidate("ACTION1", {"entity_id": "target"})
    before = _state(solved=False)
    after = _state(solved=True)
    projected = PredictionPacket(
        object_deltas={"solved|target|": 1.0},
        known_channels=frozenset({"objects"}),
        state_after=after,
    )
    projections = tuple(
        FakeProjection(
            spec=FakeFrame(frame),
            state_before=before,
            action=action,
            state_after=after,
            observation=projected,
        )
        for frame in frames
    )
    return FakeBundle(
        event_id=event_id,
        action=action,
        common_observation=PredictionPacket(
            progress_mean=1.0,
            terminal_probability=0.0,
            goal_probability=1.0,
            known_channels=frozenset({"progress", "terminal", "goal"}),
            state_after=after,
        ),
        projections=projections,
        events=("progress", "level_complete"),
    )


def _hypothesis(
    program: JointProgramHypothesis,
    *,
    frame: str = "root_only",
    option: FakeOption | None = None,
    targets: tuple[str, ...] = (),
) -> JointGaugeHypothesis:
    return JointGaugeHypothesis(
        world_program=program,
        frame=FakeFrame(frame),
        transports=tuple(_official_transport(frame, target) for target in targets),
        option=option or FakeOption("repeat"),
    )


def _official_transport(
    source_frame: str,
    target_frame: str,
    *,
    corrupt_prediction: bool = False,
    swap_action: bool = False,
) -> TransportMap:
    return TransportMap(
        source_frame,
        target_frame,
        fact_map=(
            (("solved", "no_effect"),)
            if corrupt_prediction
            else (("action1", "action2"),)
            if swap_action
            else ()
        ),
        action_map=(
            (("ACTION1", "ACTION2"),) if swap_action else (("ACTION1", "ACTION1"),)
        ),
    )


def _orbit_witness() -> TransportOrbitWitness:
    transport = TransportMap(
        "root_only",
        "allocentric_object_relative",
        role_map=(("target", "target"),),
        action_map=(("ACTION1", "ACTION1"),),
    )
    inverse = transport.inverse
    assert inverse is not None
    source_projection = _bundle("orbit-certificate").projections[0]
    target_projection = replace(
        source_projection,
        spec=FakeFrame("allocentric_object_relative"),
    )
    certificate = TransportCertificate.from_projections(
        transport,
        source_projection,
        target_projection,
        inverse=inverse,
    )
    return TransportOrbitWitness.from_certificate(certificate)


def _renamed_option(option: OptionAutomaton) -> OptionAutomaton:
    mapping = {
        state.state_id: f"renamed_{index}" for index, state in enumerate(option.states)
    }
    return OptionAutomaton(
        schema=option.schema,
        states=tuple(
            OptionState(mapping[state.state_id], terminal=state.terminal)
            for state in reversed(option.states)
        ),
        transitions=tuple(
            OptionTransition(
                mapping[item.source_state_id],
                mapping[item.target_state_id],
                item.action_schema,
                predicate=item.predicate,
                predicate_present=item.predicate_present,
                minimum_visits=item.minimum_visits,
                relation=item.relation,
                priority=item.priority,
            )
            for item in reversed(option.transitions)
        ),
        initial_state_id=mapping[option.initial_state_id],
        maximum_horizon=option.maximum_horizon,
        initiation_condition=option.initiation_condition,
    )


def test_joint_hash_contains_all_components_and_rejects_identity_leaks() -> None:
    program = _program("solve", solves=True)
    first = _hypothesis(program)
    second = _hypothesis(program, option=FakeOption("alternate"))

    assert first.canonical_hash != second.canonical_hash
    assert first.world_program.canonical_hash == program.canonical_hash
    with pytest.raises(ValueError, match="outside the four frozen frames"):
        JointGaugeHypothesis(
            world_program=program,
            frame=FakeFrame("bp35-0a0ad940"),
            transports=(),
            option=FakeOption("repeat"),
        )


def test_joint_hash_refuses_components_without_canonical_representation() -> None:
    class _NonCanonicalFrame:
        frame_id = "root_only"

    with pytest.raises(TypeError, match="canonical"):
        JointGaugeHypothesis(
            world_program=_program("noncanonical_frame", solves=True),
            frame=_NonCanonicalFrame(),
            transports=(),
            option=FakeOption("repeat"),
        )


@pytest.mark.parametrize(
    "nonfinite",
    (float("nan"), float("inf"), float("-inf")),
)
def test_canonical_hypothesis_rejects_nonfinite_numbers(nonfinite: float) -> None:
    class _NonFiniteOption:
        initial_state = 0

        @property
        def canonical_payload(self) -> dict[str, object]:
            return {"option_id": "nonfinite", "score": nonfinite}

        def allowed_action_schemas(self, _state: int) -> tuple[str, ...]:
            return ("ACTION1",)

        def observe(
            self,
            state: int,
            _action: ActionCandidate,
            *,
            events: tuple[str, ...],
        ) -> int:
            del events
            return state

    with pytest.raises(ValueError, match="canonical payload numbers must be finite"):
        JointGaugeHypothesis(
            _program("nonfinite", solves=True),
            FakeFrame("root_only"),
            (),
            _NonFiniteOption(),
        )


@pytest.mark.parametrize(
    "raw_role",
    ("red", "red_object", "identity", "persistent_object_17", "bp35"),
)
def test_transferable_program_rejects_raw_color_and_identity_roles(
    raw_role: str,
) -> None:
    program = _program("raw_role", solves=True)
    program = replace(
        program,
        object_schema=ObjectSchema(("object", raw_role)),
        action_bindings=(replace(program.action_bindings[0], target_role=raw_role),),
    )

    with pytest.raises(ValueError, match="forbidden transferable program token"):
        JointGaugeHypothesis(
            program,
            FakeFrame("root_only"),
            (),
            FakeOption("repeat"),
        )


def test_transferable_program_rejects_raw_color_attributes() -> None:
    program = _program("raw_attribute", solves=True)
    rule = replace(
        program.transition_rules[0],
        effects=(Effect("set_register", key="color", value="red"),),
    )

    with pytest.raises(ValueError, match="forbidden transferable program token"):
        JointGaugeHypothesis(
            replace(program, transition_rules=(rule,)),
            FakeFrame("root_only"),
            (),
            FakeOption("repeat"),
        )


def test_same_color_relation_is_not_transferable() -> None:
    program = _program("same_color", solves=True)
    rule = replace(
        program.transition_rules[0],
        condition=Expression.fact("same_color", "$target", "$target"),
    )

    with pytest.raises(ValueError, match="forbidden transferable program token"):
        JointGaugeHypothesis(
            replace(program, transition_rules=(rule,)),
            FakeFrame("root_only"),
            (),
            FakeOption("repeat"),
        )


def test_color_independent_structural_relation_remains_transferable() -> None:
    program = _program("structural_relation", solves=True)
    rule = replace(
        program.transition_rules[0],
        condition=Expression.fact("same_shape", "$target", "$target"),
    )
    hypothesis = JointGaugeHypothesis(
        replace(program, transition_rules=(rule,)),
        FakeFrame("root_only"),
        (),
        FakeOption("repeat"),
    )

    assert hypothesis.world_program.transition_rules[0].condition.predicate == (
        "same_shape"
    )


def test_mdl_prior_charges_transport_and_option_nodes_at_frozen_rate() -> None:
    program = _program("prior", solves=True)
    transport = FakeTransport("root_only", "allocentric_object_relative")
    option = FakeOption("repeat")
    hypothesis = JointGaugeHypothesis(
        program,
        FakeFrame("root_only"),
        (transport,),
        option,
    )
    baseline = (
        -0.05 * program.node_count
        - 0.25 * program.local_constant_count
        - float(program.edit_distance)
    )
    assert hypothesis.log_prior == pytest.approx(
        baseline - 0.05 * (transport.node_count + option.node_count)
    )


def test_uncertified_frame_copies_remain_separate_classes() -> None:
    program = _program("solve", solves=True)
    hypotheses = tuple(
        _hypothesis(program, frame=frame)
        for frame in (
            "root_only",
            "allocentric_object_relative",
            "action_aligned_relational",
            "action_rooted_topological",
        )
    )
    posterior = GaugeProgramPosterior()
    posterior.seed(hypotheses)

    assert len(posterior.classes) == 4
    assert [particle.probability for particle in posterior.particles] == pytest.approx(
        [0.25] * 4
    )


def test_canonical_derivation_dedup_uses_best_mdl_independent_of_order() -> None:
    base = _program("same_semantics", solves=True)
    cheap = _hypothesis(replace(base, edit_distance=1))
    expensive = _hypothesis(replace(base, edit_distance=4))
    other = _hypothesis(_program("other_semantics", solves=False))

    first = GaugeProgramPosterior()
    first.seed((cheap, expensive, other))
    second = GaugeProgramPosterior()
    second.seed((expensive, cheap, other))

    assert tuple(item.hypothesis.canonical_hash for item in first.particles) == tuple(
        item.hypothesis.canonical_hash for item in second.particles
    )
    assert tuple(item.probability for item in first.particles) == pytest.approx(
        tuple(item.probability for item in second.particles)
    )
    retained = next(
        item
        for item in first.particles
        if item.hypothesis.canonical_hash == cheap.canonical_hash
    )
    assert retained.hypothesis.world_program.edit_distance == 1


def test_duck_typed_certification_cannot_merge_gauge_copies() -> None:
    program = _program("solve", solves=True)
    option = FakeOption("repeat")
    orbit = "certified-root-allocentric-orbit"
    hypotheses = (
        JointGaugeHypothesis(
            program,
            FakeFrame("root_only"),
            (FakeTransport("root_only", "allocentric_object_relative", orbit),),
            option,
        ),
        JointGaugeHypothesis(
            program,
            FakeFrame("allocentric_object_relative"),
            (FakeTransport("allocentric_object_relative", "root_only", orbit),),
            option,
        ),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed(hypotheses)

    assert len(posterior.classes) == 2
    assert [particle.probability for particle in posterior.particles] == pytest.approx(
        [0.5, 0.5]
    )


def test_two_certified_gauge_copies_add_two_thirds_class_mass() -> None:
    program = _program("copy_mass", solves=True)
    option = repeat("action1")
    forward = _orbit_witness()
    reverse = forward.inverse
    hypotheses = (
        JointGaugeHypothesis(
            program,
            FakeFrame("root_only"),
            (forward,),
            option,
        ),
        JointGaugeHypothesis(
            program,
            FakeFrame("allocentric_object_relative"),
            (reverse,),
            option,
        ),
        JointGaugeHypothesis(
            program,
            FakeFrame("root_only"),
            (
                FakeTransport(
                    "root_only",
                    "allocentric_object_relative",
                    nodes=forward.node_count,
                ),
            ),
            option,
        ),
    )
    assert len({item.log_prior for item in hypotheses}) == 1

    posterior = GaugeProgramPosterior()
    posterior.seed(hypotheses)

    assert len(posterior.classes) == 2
    assert sorted(
        gauge_class.probability for gauge_class in posterior.classes
    ) == pytest.approx([1.0 / 3.0, 2.0 / 3.0])


def test_stable_structural_orbit_fuses_forward_and_reverse_hypotheses() -> None:
    program = _program("orbit", solves=True)
    option = repeat("action1")
    forward = _orbit_witness()
    reverse = forward.inverse
    hypotheses = (
        JointGaugeHypothesis(
            program,
            FakeFrame("root_only"),
            (forward,),
            option,
        ),
        JointGaugeHypothesis(
            program,
            FakeFrame("allocentric_object_relative"),
            (reverse,),
            option,
        ),
    )

    posterior = GaugeProgramPosterior()
    posterior.seed(hypotheses)

    assert forward.canonical_hash == reverse.canonical_hash
    assert len(posterior.classes) == 1
    assert posterior.classes[0].probability == pytest.approx(1.0)


def test_orbit_domain_must_cover_every_program_and_option_symbol() -> None:
    # The certificate state never mentions ``no_effect``, so this witness is
    # real but too narrow to certify the no-op dynamics below.
    program = _program("narrow_domain", solves=False)
    option = repeat("action1")
    forward = _orbit_witness()
    assert "fact:no_effect" not in forward.certified_source_domain
    hypotheses = (
        JointGaugeHypothesis(
            program,
            FakeFrame("root_only"),
            (forward,),
            option,
        ),
        JointGaugeHypothesis(
            program,
            FakeFrame("allocentric_object_relative"),
            (forward.inverse,),
            option,
        ),
    )

    posterior = GaugeProgramPosterior()
    posterior.seed(hypotheses)

    assert len(posterior.classes) == 2


def test_certified_orbit_cannot_be_attached_to_a_disconnected_native_frame() -> None:
    transport = TransportMap(
        "root_only",
        "allocentric_object_relative",
        role_map=(("target", "target"),),
        action_map=(("ACTION1", "ACTION1"),),
    )
    source_projection = _bundle("disconnected-orbit").projections[0]
    target_projection = replace(
        source_projection,
        spec=FakeFrame("allocentric_object_relative"),
    )
    witness = TransportOrbitWitness.from_certificate(
        TransportCertificate.from_projections(
            transport,
            source_projection,
            target_projection,
        )
    )

    with pytest.raises(ValueError, match="connected to the native frame"):
        JointGaugeHypothesis(
            _program("disconnected", solves=True),
            FakeFrame("action_aligned_relational"),
            (witness,),
            FakeOption("repeat"),
        )


def test_common_outcome_is_scored_once_when_four_projections_are_available() -> None:
    program = _program("solve", solves=True)
    extra = (
        "allocentric_object_relative",
        "action_aligned_relational",
        "action_rooted_topological",
    )
    hypothesis = _hypothesis(program, targets=extra)
    one = GaugeProgramPosterior()
    four = GaugeProgramPosterior()
    one.seed((hypothesis,))
    four.seed((hypothesis,))

    one.observe(_bundle("one"))
    four.observe(_bundle("four", ("root_only", *extra)))
    left = one.particles[0]
    right = four.particles[0]

    assert left.latest_physical_log_likelihood == pytest.approx(
        right.latest_physical_log_likelihood
    )
    assert left.latest_frame_log_likelihood == pytest.approx(
        right.latest_frame_log_likelihood
    )
    assert four.last_update is not None
    assert four.last_update.physical_scored_particles == 1
    assert four.last_update.projection_score_count == 4


@dataclass(frozen=True)
class _DuckActionHijackTransport:
    source_frame: str = "root_only"
    target_frame: str = "allocentric_object_relative"

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "kind": "test_duck_action_hijack",
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
        }

    @property
    def node_count(self) -> int:
        return 1

    def transport_prediction(self, packet: PredictionPacket) -> PredictionPacket:
        return replace(packet, object_deltas={"action6": 1.0})

    def transport_action(self, action: ActionCandidate) -> ActionCandidate:
        return ActionCandidate("ACTION6", dict(action.action_data))


class _ActionRecordingExecutor:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def step(
        self,
        _program: JointProgramHypothesis,
        state: AbstractState,
        action: ActionCandidate,
    ) -> PredictionPacket:
        self.actions.append(action.action_name)
        return PredictionPacket(
            object_deltas={action.action_name.lower(): 1.0},
            known_channels=frozenset({"objects"}),
            state_after=state,
        )


def test_observed_commutativity_executes_the_transported_target_action() -> None:
    executor = _ActionRecordingExecutor()
    hypothesis = JointGaugeHypothesis(
        _program("transported_action", solves=True),
        FakeFrame("root_only"),
        (
            _DuckActionHijackTransport(),
            _official_transport(
                "root_only",
                "allocentric_object_relative",
                swap_action=True,
            ),
        ),
        FakeOption("repeat"),
    )
    posterior = GaugeProgramPosterior(executor=executor)  # type: ignore[arg-type]
    posterior.seed((hypothesis,))

    posterior.observe(
        _bundle(
            "transported-action",
            ("root_only", "allocentric_object_relative"),
        )
    )

    assert executor.actions == ["ACTION1", "ACTION2"]
    assert posterior.particles[0].latest_commutativity_penalty == 0.0


def test_transport_resolution_ignores_ducks_and_delegates_official_multihop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _official_transport("root_only", "allocentric_object_relative")
    second = _official_transport(
        "allocentric_object_relative",
        "action_rooted_topological",
    )
    composed = _official_transport("root_only", "action_rooted_topological")
    duck = FakeTransport("root_only", "action_rooted_topological")
    calls: list[tuple[tuple[object, ...], str, str]] = []

    def official_resolver(
        transports: tuple[object, ...],
        source_frame: str,
        target_frame: str,
    ) -> TransportMap:
        calls.append((transports, source_frame, target_frame))
        return composed

    monkeypatch.setattr(
        gauge_module,
        "resolve_official_transport",
        official_resolver,
    )

    resolved = gauge_module._find_transport(
        (duck, first, second),
        "root_only",
        "action_rooted_topological",
    )

    assert resolved is composed
    assert calls == [
        (
            (first, second),
            "root_only",
            "action_rooted_topological",
        )
    ]


def test_duck_only_transport_is_scored_as_missing_and_noncommuting() -> None:
    hypothesis = JointGaugeHypothesis(
        _program("duck_transport", solves=True),
        FakeFrame("root_only"),
        (FakeTransport("root_only", "allocentric_object_relative"),),
        FakeOption("repeat"),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((hypothesis,))

    posterior.observe(
        _bundle(
            "duck-transport",
            ("root_only", "allocentric_object_relative"),
        )
    )

    assert posterior.particles[0].latest_commutativity_penalty == 1.0


def test_missing_transport_is_scored_as_unknown_and_noncommuting() -> None:
    program = _program("missing_transport", solves=True)
    missing = _hypothesis(program)
    transported = _hypothesis(
        program,
        targets=("allocentric_object_relative",),
    )
    bundle = _bundle(
        "missing-transport",
        ("root_only", "allocentric_object_relative"),
    )
    root_observation = PredictionPacket(
        object_deltas={"solved": 0.95},
        known_channels=frozenset({"objects"}),
        state_after=bundle.projections[0].state_after,
    )
    contradictory_observation = PredictionPacket(
        object_deltas={"contradiction": 1.0},
        known_channels=frozenset({"objects"}),
        state_after=bundle.projections[1].state_after,
    )
    bundle = replace(
        bundle,
        projections=(
            replace(bundle.projections[0], observation=root_observation),
            replace(
                bundle.projections[1],
                observation=contradictory_observation,
            ),
        ),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((missing, transported))
    posterior.observe(bundle)
    by_transport_count = {
        len(particle.hypothesis.transports): particle
        for particle in posterior.particles
    }

    assert posterior.last_update is not None
    assert posterior.last_update.projection_score_count == 4
    assert by_transport_count[0].latest_frame_log_likelihood < 0.0
    assert by_transport_count[0].latest_commutativity_penalty == 1.0
    assert by_transport_count[1].latest_commutativity_penalty == 0.0


def test_duplicate_physical_event_fails_before_a_second_update() -> None:
    posterior = GaugeProgramPosterior()
    posterior.seed((_hypothesis(_program("solve", solves=True)),))
    bundle = _bundle("event-1")
    posterior.observe(bundle)

    with pytest.raises(ValueError, match="duplicate physical event_id"):
        posterior.observe(bundle)
    assert posterior.event_ids == ("event-1",)


def test_map_does_not_collapse_until_three_stable_high_mass_transitions() -> None:
    posterior = GaugeProgramPosterior()
    posterior.seed(
        (
            _hypothesis(_program("solve", solves=True)),
            _hypothesis(
                _program("noop", solves=False),
                option=FakeOption("noop_option"),
            ),
        )
    )

    posterior.observe(_bundle("event-1"))
    assert posterior.collapsed is False
    posterior.observe(_bundle("event-2"))
    assert posterior.collapsed is False
    posterior.observe(_bundle("event-3"))

    assert posterior.collapsed is True
    assert len(posterior.classes) == 1


def test_stateful_option_particles_survive_a_shared_noop_prefix() -> None:
    program = _program("noop", solves=False)
    first = _hypothesis(program, option=FakeOption("first", trigger="terminal_a"))
    second = _hypothesis(program, option=FakeOption("second", trigger="terminal_b"))
    posterior = GaugeProgramPosterior()
    posterior.seed((first, second))

    noop = replace(_bundle("noop-prefix"), events=("no_effect",))
    posterior.observe(noop)

    assert len(posterior.classes) == 2
    assert {particle.option_state for particle in posterior.particles} == {0}


def test_alpha_renamed_options_have_identical_posterior_state_signatures() -> None:
    option = alternate("action1", "action2", maximum_horizon=4)
    renamed = _renamed_option(option)
    program = _program("alpha_option", solves=False)
    first = GaugeProgramPosterior()
    first.seed((_hypothesis(program, option=option),))
    second = GaugeProgramPosterior()
    second.seed((_hypothesis(program, option=renamed),))

    event = replace(_bundle("alpha-noop"), events=("no_effect",))
    first.observe(event)
    second.observe(event)

    assert option.canonical_hash == renamed.canonical_hash
    assert first.particles[0].trace_signature == second.particles[0].trace_signature


def test_observed_sequence_updates_the_option_component_likelihood() -> None:
    program = _program("solve", solves=True)
    compatible = JointGaugeHypothesis(
        program,
        FakeFrame("root_only"),
        (),
        repeat("action1", maximum_horizon=4),
    )
    incompatible = JointGaugeHypothesis(
        program,
        FakeFrame("root_only"),
        (),
        repeat("action2", maximum_horizon=4),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((compatible, incompatible))

    posterior.observe(_bundle("option-evidence"))

    by_option = {
        particle.hypothesis.option.allowed_action_schemas(
            particle.hypothesis.option.initial_state
        )[0]: particle
        for particle in posterior.particles
    }
    assert by_option["action1"].latest_option_log_likelihood == 0.0
    assert by_option["action2"].latest_option_log_likelihood < 0.0
    assert by_option["action1"].probability > by_option["action2"].probability


def test_relation_option_requires_structural_evidence_in_the_posterior() -> None:
    program = _program("relation", solves=True)
    option = follow_relation_then_apply("action1", "action2")
    hypothesis = JointGaugeHypothesis(
        program,
        FakeFrame("root_only"),
        (),
        option,
    )

    missing = GaugeProgramPosterior()
    missing.seed((hypothesis,))
    missing.observe(_bundle("relation-missing"))
    assert missing.particles[0].latest_option_log_likelihood < 0.0

    declared_action = ActionCandidate(
        "ACTION1",
        {
            "entity_id": "target",
            "relation": "successor_toward_enclosure",
        },
    )
    declared_bundle = _bundle("relation-declared-without-proof")
    declared_bundle = replace(
        declared_bundle,
        action=declared_action,
        projections=tuple(
            replace(projection, action=declared_action)
            for projection in declared_bundle.projections
        ),
    )
    declared = GaugeProgramPosterior()
    declared.seed((hypothesis,))
    declared.observe(declared_bundle)
    assert declared.particles[0].latest_option_log_likelihood < 0.0

    relation_state = replace(
        _state(),
        topology=(("successor_toward_enclosure", 1),),
    )
    present_bundle = _bundle("relation-present")
    present_bundle = replace(
        present_bundle,
        projections=tuple(
            replace(projection, state_before=relation_state)
            for projection in present_bundle.projections
        ),
    )
    ungrounded = GaugeProgramPosterior()
    ungrounded.seed((hypothesis,))
    ungrounded.observe(present_bundle)
    assert ungrounded.particles[0].latest_option_log_likelihood < 0.0

    grounded_action = replace(
        present_bundle.action,
        action_data={
            **dict(present_bundle.action.action_data),
            "relation": "successor_toward_enclosure",
        },
    )
    grounded = GaugeProgramPosterior()
    grounded.seed((hypothesis,))
    grounded.observe(
        replace(
            present_bundle,
            event_id="relation-grounded",
            action=grounded_action,
            projections=tuple(
                replace(projection, action=grounded_action)
                for projection in present_bundle.projections
            ),
        )
    )
    assert grounded.particles[0].latest_option_log_likelihood == 0.0


def test_relation_sequence_ranking_requires_native_frame_state_proof() -> None:
    option = follow_relation_then_apply("action1", "action2")
    posterior = GaugeProgramPosterior()
    posterior.seed(
        (
            JointGaugeHypothesis(
                _program("relation_ranking", solves=True),
                FakeFrame("root_only"),
                (),
                option,
            ),
        )
    )
    action = ActionCandidate(
        "ACTION1",
        {
            "entity_id": "target",
            "relation": "successor_toward_enclosure",
        },
    )

    unproved = rank_option_sequence_signatures(
        posterior,
        ((action, ("no_effect",)),),
    )
    proved = rank_option_sequence_signatures(
        posterior,
        (
            (
                action,
                ("no_effect",),
                {
                    "root_only": replace(
                        _state(),
                        topology=(("successor_toward_enclosure", 1),),
                    )
                },
            ),
        ),
    )

    assert unproved.best_compatible_rank is None
    assert unproved.compatible_posterior_mass == 0.0
    assert proved.best_compatible_rank == 1
    assert proved.compatible_posterior_mass == pytest.approx(1.0)


@pytest.mark.parametrize(
    "condition",
    (
        OptionInitiationCondition.fact("exists", role="target"),
        OptionInitiationCondition.entity_role("target"),
    ),
)
def test_seed_passes_native_state_to_fact_and_role_initiation(
    condition: OptionInitiationCondition,
) -> None:
    option = replace(
        repeat("action1"),
        initiation_condition=condition,
    )
    hypothesis = JointGaugeHypothesis(
        _program("guarded_initiation", solves=True),
        FakeFrame("root_only"),
        (),
        option,
    )
    missing = GaugeProgramPosterior()
    with pytest.raises(ValueError, match="initiation condition"):
        missing.seed(
            (hypothesis,),
            initial_states={"root_only": AbstractState()},
        )

    satisfied = GaugeProgramPosterior()
    satisfied.seed(
        (hypothesis,),
        initial_states={"root_only": _state()},
    )
    assert satisfied.particles[0].option_state.state_id == option.initial_state_id


def test_decision_executes_one_action_respects_veto_and_has_safe_fallback() -> None:
    program = _program("solve", solves=True)
    posterior = GaugeProgramPosterior()
    posterior.seed((_hypothesis(program),), initial_states={"root_only": _state()})
    engine = GaugeDecisionEngine()
    action1 = ActionCandidate("ACTION1", {"entity_id": "target"})
    action2 = ActionCandidate("ACTION2")

    decision = engine.decide(
        posterior,
        {"root_only": _state()},
        (action1, action2),
        danger_veto=lambda action: action.action_name == "ACTION1",
    )
    fallback = engine.decide(
        posterior,
        {},
        (action1, action2),
        fallback_action=action2,
    )

    assert decision.action == action2
    assert decision.chosen is not None
    assert fallback.action == action2
    assert fallback.reason == "incomplete_projection_fallback"


def test_fallback_requires_explicit_current_option_support() -> None:
    posterior = GaugeProgramPosterior()
    posterior.seed(
        (
            _hypothesis(
                _program("fallback_support", solves=True),
                option=repeat("action1"),
            ),
        )
    )
    engine = GaugeDecisionEngine()
    supported = ActionCandidate("ACTION1")
    unsupported = ActionCandidate("ACTION2")

    rejected = engine.decide(
        posterior,
        {},
        (unsupported,),
        fallback_action=unsupported,
    )
    accepted = engine.decide(
        posterior,
        {},
        (supported,),
        fallback_action=supported,
    )

    assert rejected.action is None
    assert rejected.reason == "all_vetoed"
    assert accepted.action == supported
    assert accepted.reason == "incomplete_projection_fallback"


def test_decision_penalizes_noncommuting_counterfactual_frames() -> None:
    program = _program("solve", solves=True)
    hypothesis = JointGaugeHypothesis(
        program,
        FakeFrame("root_only"),
        (
            _official_transport(
                "root_only",
                "allocentric_object_relative",
                corrupt_prediction=True,
            ),
        ),
        FakeOption("repeat"),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((hypothesis,), initial_states={"root_only": _state()})

    decision = GaugeDecisionEngine().decide(
        posterior,
        {
            "root_only": _state(),
            "allocentric_object_relative": _state(),
        },
        (ActionCandidate("ACTION1", {"entity_id": "target"}),),
    )

    assert decision.chosen is not None
    assert decision.chosen.commutativity_penalty > 0.0


def test_noncomparable_projection_does_not_dilute_commutativity_failure() -> None:
    program = _program("commutativity", solves=True)
    hypothesis = JointGaugeHypothesis(
        program,
        FakeFrame("root_only"),
        (
            _official_transport(
                "root_only",
                "allocentric_object_relative",
                corrupt_prediction=True,
            ),
            _official_transport("root_only", "action_aligned_relational"),
        ),
        FakeOption("repeat"),
    )
    comparable = GaugeProgramPosterior()
    comparable.seed((hypothesis,))
    comparable.observe(
        _bundle(
            "commute-comparable",
            ("root_only", "allocentric_object_relative"),
        )
    )

    with_unknown = _bundle(
        "commute-with-unknown",
        (
            "root_only",
            "allocentric_object_relative",
            "action_aligned_relational",
        ),
    )
    with_unknown = replace(
        with_unknown,
        projections=tuple(
            replace(projection, observation=PredictionPacket())
            if projection.spec.frame_id == "action_aligned_relational"
            else projection
            for projection in with_unknown.projections
        ),
    )
    augmented = GaugeProgramPosterior()
    augmented.seed((hypothesis,))
    augmented.observe(with_unknown)

    assert comparable.particles[0].latest_commutativity_penalty > 0.0
    assert augmented.particles[0].latest_commutativity_penalty == pytest.approx(
        comparable.particles[0].latest_commutativity_penalty
    )


def test_real_option_automaton_state_is_part_of_each_particle_update() -> None:
    hypothesis = JointGaugeHypothesis(
        world_program=_program("solve", solves=True),
        frame=FakeFrame("root_only"),
        transports=(),
        option=repeat("action1", maximum_horizon=4),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((hypothesis,))

    posterior.observe(_bundle("real-option"))

    state = posterior.particles[0].option_state
    assert state.state_id == "done"
    assert state.terminated is True
    assert state.steps == 1


def test_prime_counter_advances_inside_each_posterior_particle() -> None:
    option = prime_then_repeat("action1", 2, "action2", maximum_horizon=4)
    hypothesis = JointGaugeHypothesis(
        world_program=_program("prime", solves=True),
        frame=FakeFrame("root_only"),
        transports=(),
        option=option,
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((hypothesis,))

    first = replace(_bundle("prime-one"), events=("no_effect",))
    second = replace(_bundle("prime-two"), events=("no_effect",))
    posterior.observe(first)
    after_first = posterior.particles[0].option_state
    posterior.observe(second)
    after_second = posterior.particles[0].option_state

    assert after_first.state_id == "prime"
    assert after_first.state_visits == 1
    assert after_first.steps == 1
    assert after_second.state_id == "repeated"
    assert after_second.state_visits == 0
    assert after_second.steps == 2


def test_pruned_gauge_classes_retain_an_explicit_residual_mass() -> None:
    program = _program("residual", solves=True)
    hypotheses = tuple(
        _hypothesis(program, option=FakeOption(f"option_{index}")) for index in range(3)
    )
    posterior = GaugeProgramPosterior(maximum_classes=2)
    posterior.seed(hypotheses)

    assert len(posterior.classes) == 2
    assert posterior.residual_mass == pytest.approx(1.0 / 3.0)
    assert sum(item.probability for item in posterior.classes) == pytest.approx(
        2.0 / 3.0
    )
    assert posterior.normalized_entropy == pytest.approx(1.0)
    snapshot = posterior.snapshot()
    assert snapshot["classes"] == 3
    assert snapshot["retained_classes"] == 2
    assert snapshot["residual_mass"] == pytest.approx(1.0 / 3.0)

    decision = GaugeDecisionEngine(maximum_option_horizon=1).decide(
        posterior,
        {"root_only": _state()},
        [ActionCandidate("ACTION1", {"entity_id": "target"})],
    )
    assert decision.chosen is not None
    assert decision.chosen.residual_mass == pytest.approx(1.0 / 3.0)

    posterior.observe(_bundle("residual-update"))
    assert posterior.residual_mass == pytest.approx(1.0 / 3.0)
    assert (
        sum(item.probability for item in posterior.classes) + posterior.residual_mass
    ) == pytest.approx(1.0)


def test_reset_clears_all_branch_local_latent_state() -> None:
    legacy = replace(
        _state(solved=True),
        true_facts=frozenset(
            (
                *_state(solved=True).true_facts,
                GroundFact("attached", ("target", "target")),
            )
        ),
        counters=(("legacy_counter", 7),),
        topology=(("legacy_edge", 1),),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed(
        (
            _hypothesis(
                _program("reset_isolation", solves=True),
                option=repeat("action1"),
            ),
        ),
        initial_states={"root_only": legacy},
    )

    posterior.observe(replace(_bundle("reset-boundary"), reset=True))
    assert posterior.particles[0].state is None
    posterior.observe(_bundle("new-branch"))
    state = posterior.particles[0].state

    assert state is not None
    assert GroundFact("attached", ("target", "target")) not in state.true_facts
    assert "legacy_counter" not in dict(state.counters)
    assert "legacy_edge" not in dict(state.topology)


class _SequenceExecutor:
    def step(
        self,
        _program: JointProgramHypothesis,
        state: AbstractState,
        action: ActionCandidate,
    ) -> PredictionPacket:
        progress = 1.0 if action.action_name == "ACTION2" else 0.0
        return PredictionPacket(
            progress_mean=progress,
            terminal_probability=0.0,
            goal_probability=0.0,
            known_channels=frozenset({"progress", "terminal", "goal"}),
            state_after=state,
        )


class _SupportWeightedExecutor(_SequenceExecutor):
    def step(
        self,
        _program: JointProgramHypothesis,
        state: AbstractState,
        action: ActionCandidate,
    ) -> PredictionPacket:
        progress = 1.0 if action.action_name == "ACTION2" else 0.2
        return PredictionPacket(
            progress_mean=progress,
            terminal_probability=0.0,
            goal_probability=0.0,
            known_channels=frozenset({"progress", "terminal", "goal"}),
            state_after=state,
        )


def test_decision_uses_common_posterior_denominator_across_option_support() -> None:
    program = _program("support_weighted", solves=True)
    high_mass = _hypothesis(
        program,
        option=repeat("action1", maximum_horizon=4),
    )
    tail = _hypothesis(
        program,
        option=repeat("action2", maximum_horizon=4),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((high_mass, tail))
    posterior.observe(replace(_bundle("support-evidence"), events=("no_effect",)))

    decision = GaugeDecisionEngine(
        executor=_SupportWeightedExecutor(),  # type: ignore[arg-type]
        maximum_option_horizon=1,
    ).decide(
        posterior,
        {"root_only": _state()},
        (ActionCandidate("ACTION1"), ActionCandidate("ACTION2")),
    )

    assert max(item.probability for item in posterior.particles) > 0.98
    assert decision.action == ActionCandidate("ACTION1"), decision.assessments
    by_action = {item.action.action_name: item for item in decision.assessments}
    assert by_action["ACTION2"].evaluated_mass < 0.02
    assert by_action["ACTION2"].expected_progress < 0.02


def test_counterfactual_option_rollout_separates_a_shared_first_prefix() -> None:
    program = _program("sequence", solves=True)
    hypotheses = (
        _hypothesis(
            program,
            option=repeat(
                "action1",
                termination_predicate="never",
                maximum_horizon=2,
            ),
        ),
        _hypothesis(
            program,
            option=alternate(
                "action1",
                "action2",
                termination_predicate="never",
                maximum_horizon=2,
            ),
        ),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed(hypotheses)
    engine = GaugeDecisionEngine(
        executor=_SequenceExecutor(),  # type: ignore[arg-type]
        maximum_option_horizon=2,
    )

    decision = engine.decide(
        posterior,
        {"root_only": _state()},
        (ActionCandidate("ACTION1"), ActionCandidate("ACTION2")),
    )

    assert decision.action == ActionCandidate("ACTION1")
    assert decision.chosen is not None
    assert decision.chosen.expected_progress > 0.0
    assert decision.chosen.information_gain > 0.0


class _StateChangeSequenceExecutor(_SequenceExecutor):
    def step(
        self,
        program: JointProgramHypothesis,
        state: AbstractState,
        action: ActionCandidate,
    ) -> PredictionPacket:
        packet = super().step(program, state, action)
        if action.action_name != "ACTION1":
            return packet
        return replace(
            packet,
            object_deltas={"moved": 1.0},
            known_channels=frozenset((*packet.known_channels, "objects")),
        )


def test_counterfactual_state_changed_guard_advances_until_option() -> None:
    hypothesis = _hypothesis(
        _program("until_state_changed", solves=True),
        option=until_then(
            "action1",
            "state_changed",
            "action2",
            termination_predicate="never",
            maximum_horizon=2,
        ),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((hypothesis,))

    decision = GaugeDecisionEngine(
        executor=_StateChangeSequenceExecutor(),  # type: ignore[arg-type]
        maximum_option_horizon=2,
    ).decide(
        posterior,
        {"root_only": _state()},
        (ActionCandidate("ACTION1"), ActionCandidate("ACTION2")),
    )

    assert decision.action == ActionCandidate("ACTION1")
    assert decision.chosen is not None
    assert decision.chosen.expected_progress == pytest.approx(1.0)


def test_synthetic_posterior_recovers_all_five_joint_components() -> None:
    correct_program = _program("joint_target", solves=True)
    wrong_goal_program = replace(
        correct_program,
        program_id="wrong_goal",
        goal_rule=GoalRule(Expression.constant(False), family="no_goal"),
    )
    correct_transport = _official_transport(
        "root_only",
        "allocentric_object_relative",
    )
    target = JointGaugeHypothesis(
        correct_program,
        FakeFrame("root_only"),
        (correct_transport,),
        repeat("action1", maximum_horizon=4),
    )
    decoys = (
        JointGaugeHypothesis(
            _program("wrong_dynamics", solves=False),
            FakeFrame("root_only"),
            (correct_transport,),
            repeat("action1", maximum_horizon=4),
        ),
        JointGaugeHypothesis(
            wrong_goal_program,
            FakeFrame("root_only"),
            (correct_transport,),
            repeat("action1", maximum_horizon=4),
        ),
        JointGaugeHypothesis(
            correct_program,
            FakeFrame("action_rooted_topological"),
            (),
            repeat("action1", maximum_horizon=4),
        ),
        JointGaugeHypothesis(
            correct_program,
            FakeFrame("root_only"),
            (
                _official_transport(
                    "root_only",
                    "allocentric_object_relative",
                    corrupt_prediction=True,
                ),
            ),
            repeat("action1", maximum_horizon=4),
        ),
        JointGaugeHypothesis(
            correct_program,
            FakeFrame("root_only"),
            (correct_transport,),
            repeat("action2", maximum_horizon=4),
        ),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((target, *decoys))

    for index in range(3):
        posterior.observe(
            _bundle(
                f"joint-recovery-{index}",
                ("root_only", "allocentric_object_relative"),
            )
        )

    ranked = sorted(
        posterior.particles,
        key=lambda particle: particle.probability,
        reverse=True,
    )
    assert ranked[0].hypothesis.canonical_hash == target.canonical_hash
    assert all(ranked[0].probability > item.probability for item in ranked[1:])


def test_real_physical_bundle_contract_updates_the_gauge_posterior() -> None:
    action = ActionCandidate("ACTION1", {"entity_id": "target"})
    before = _state(solved=False)
    after = _state(solved=True)
    evidence = ObservedTransition(
        state_before=before,
        action=action,
        state_after=after,
        observation=PredictionPacket(
            object_deltas={"solved|target|": 1.0},
            progress_mean=1.0,
            terminal_probability=0.0,
            goal_probability=1.0,
            known_channels=frozenset({"objects", "progress", "terminal", "goal"}),
            state_after=after,
        ),
        events=("progress", "level_complete"),
    )
    bundle = project_observed_transition(
        evidence,
        frames=(ROOT_ONLY_FRAME,),
        projectors={"root_only": identity_projector},
        event_id="real-contract-event",
    )
    hypothesis = JointGaugeHypothesis(
        world_program=_program("solve", solves=True),
        frame=ROOT_ONLY_FRAME,
        transports=(),
        option=repeat("action1"),
    )
    posterior = GaugeProgramPosterior()
    posterior.seed((hypothesis,))

    update = posterior.observe(bundle)

    assert update.physical_scored_particles == 1
    assert update.projection_score_count == 1
    assert posterior.event_ids == ("real-contract-event",)


def test_frozen_sage_t_core_file_hashes_remain_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "contracts.py": "d4a6957c3a7aab3c4fb798316acbe42375db42d7997275fecc10126d353dfee9",
        "posterior.py": "aedb4dad969517e03d3afd0190b3ad6a509f80b1cdc43a7350c0de0c05036e01",
        "decision.py": "fda110efd864fe8036b233d18855bf4ad090bf504cde40955b957be25f5bd595",
        "controller.py": "70f18cc6966ec76a489d28e6370fb5d26a417d95405ba94b039a63567921ed39",
        "executor.py": "1cbe93ecf85da169a45a8c39e9b50ca051671f33589d14e1d5394e5823b4ddf2",
        "synthesis.py": "ec8013d1825f144e34bbe43c227e723850f4017de776a73b4df71c040652fdeb",
        "progress_witness_v10.py": "4322d20798ce3254fe863a391514242f16590be8108237ab4c4599f9ac610966",
    }
    observed = {
        name: hashlib.sha256(
            (root / "theory" / "sage_t" / name).read_bytes()
        ).hexdigest()
        for name in expected
    }
    assert observed == expected
