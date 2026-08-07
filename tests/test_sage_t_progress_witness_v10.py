from __future__ import annotations

import json

import pytest

from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
    program_from_dict,
    program_to_dict,
)
from theory.sage_t.executor import ProgramExecutor
from theory.sage_t.progress_witness_v10 import (
    CONTROL_FAMILY,
    CandidateMacro,
    EffectScanRow,
    GroundedAction,
    ProgressWitness,
    SearchConfig,
    StatefulWitnessPosterior,
    chain_successor_macro,
    compile_progress_program,
    effect_representatives,
)


def _state(*entities: AbstractEntity) -> AbstractState:
    return AbstractState(
        entities=tuple(entities),
        false_facts=frozenset(
            {GroundFact("game_over"), GroundFact("level_complete")}
        ),
    )


def test_progress_program_round_trip_and_rollout() -> None:
    program = compile_progress_program(sequence_length=5)
    restored = program_from_dict(program_to_dict(program))
    assert restored.canonical_hash == program.canonical_hash

    state = _state(
        AbstractEntity(
            "branch_target",
            roles=("object", "target"),
            center=(10.0, 10.0),
        )
    )
    actions = tuple(
        ActionCandidate("ACTION6", {"x": 10, "y": 10}) for _ in range(5)
    )
    rollout = ProgramExecutor().rollout(
        program,
        state,
        actions,
        maximum_actions=5,
    )
    assert [packet.progress_mean for packet in rollout.packets] == [0, 0, 0, 0, 1]
    assert rollout.final_packet.goal_probability == pytest.approx(0.95)
    assert rollout.final_state.counter("witness_step") == 5


def test_posthoc_witness_replay_preserves_stateful_semantic_difference() -> None:
    positive = compile_progress_program(sequence_length=2, positive=True)
    decoy = compile_progress_program(sequence_length=2, positive=False)
    state = _state(
        AbstractEntity(
            "branch_target",
            roles=("object", "target"),
            center=(10.0, 10.0),
        )
    )
    action = ActionCandidate("ACTION6", {"x": 10, "y": 10})
    evidence = (
        ObservedTransition(
            state,
            action,
            state,
            PredictionPacket(
                progress_mean=0.0,
                progress_distribution={"value:0": 1.0},
                terminal_probability=0.0,
                known_channels=frozenset({"progress", "terminal"}),
                state_after=state,
            ),
        ),
        ObservedTransition(
            state,
            action,
            state,
            PredictionPacket(
                progress_mean=1.0,
                progress_distribution={"value:1": 1.0},
                terminal_probability=0.0,
                goal_probability=1.0,
                known_channels=frozenset({"progress", "terminal", "goal"}),
                state_after=state,
            ),
        ),
    )
    posterior = StatefulWitnessPosterior(
        maximum_particles=8,
        repair_ess_threshold=1.0,
    )
    for transition in evidence:
        posterior.observe(transition, allow_repair=False)
    posterior.seed((positive, decoy), initial_state=state)
    assert len(posterior.particles) == 2
    assert posterior.top(1)[0].program.canonical_hash == positive.canonical_hash


def test_transferable_witness_has_no_grounded_coordinates_or_game_id() -> None:
    program = compile_progress_program(sequence_length=2)
    witness = ProgressWitness(
        source_game="lp85-305b61c3",
        context_signature="abcdef",
        macro_schema="repeat_target",
        relation="identity",
        abstract_steps=(),
        grounded_actions=(
            GroundedAction("ACTION6", (("x", 4), ("y", 29))),
            GroundedAction("ACTION6", (("x", 4), ("y", 29))),
        ),
        observed_events=((), ("progress",)),
        level_delta=1,
        program=program,
        posterior_rank=1,
        posterior_mass=0.9,
    )
    transferable = json.dumps(witness.transferable_payload, sort_keys=True).lower()
    audit = json.dumps(witness.to_dict(), sort_keys=True).lower()
    assert "lp85" not in transferable
    assert '"x"' not in transferable
    assert '"y"' not in transferable
    assert "lp85" in audit
    assert '"x"' in audit
    assert witness.transferable_payload["control_family"] == CONTROL_FAMILY


def test_effect_representatives_keep_one_per_causal_class() -> None:
    common = GroundedAction("ACTION6", (("x", 0), ("y", 0)))
    other = GroundedAction("ACTION6", (("x", 4), ("y", 0)))
    duplicate = GroundedAction("ACTION6", (("x", 8), ("y", 0)))
    rows = (
        EffectScanRow(common, "same", 2, 0, "NOT_FINISHED", 1.0),
        EffectScanRow(duplicate, "same", 2, 0, "NOT_FINISHED", 1.0),
        EffectScanRow(other, "different", 20, 0, "NOT_FINISHED", 1.0),
    )
    representatives = effect_representatives(rows, maximum=8)
    assert len(representatives) == 2
    assert {action.key for action in representatives} == {
        min(common.key, duplicate.key),
        other.key,
    }


def test_chain_grounding_uses_relative_successors_toward_salient_end() -> None:
    chain = [
        AbstractEntity(
            f"dot_{index}",
            roles=("object", "target"),
            attributes=(("area", "one"), ("aspect", "square")),
            center=(float(20 - 2 * index), float(2 * index)),
        )
        for index in range(8)
    ]
    source = AbstractEntity(
        "source",
        roles=("object", "movable", "target"),
        attributes=(("area", "medium"), ("aspect", "square")),
        center=(22.0, -2.0),
    )
    enclosure = AbstractEntity(
        "enclosure",
        roles=("object", "target"),
        attributes=(("area", "large"), ("aspect", "square")),
        center=(6.0, 14.0),
    )
    actions = tuple(
        GroundedAction("ACTION6", (("x", x), ("y", y)))
        for x in range(0, 17, 2)
        for y in range(4, 23, 2)
    )
    macro = chain_successor_macro(
        _state(source, enclosure, *chain),
        actions,
        config=SearchConfig(
            maximum_horizon=8,
            chain_link_radius=3.0,
            chain_minimum_nodes=6,
            chain_stride=2,
        ),
    )
    assert macro is not None
    assert macro.schema == "path_successor"
    assert macro.relation == "successor_toward_enclosure"
    assert len(macro.actions) == 3
    # The first grounded successor starts at the source end and heads toward
    # the enclosure; no coordinate is retained by the macro descriptor.
    assert macro.actions[0].data == {"x": 2, "y": 18}
    descriptor = json.dumps(macro.transferable_descriptor, sort_keys=True)
    assert '"x"' not in descriptor
    assert '"y"' not in descriptor


def test_candidate_macro_key_is_grounding_sensitive_but_descriptor_is_not() -> None:
    left = CandidateMacro(
        schema="repeat_target",
        relation="identity",
        actions=(GroundedAction("ACTION6", (("x", 4), ("y", 29))),),
    )
    right = CandidateMacro(
        schema="repeat_target",
        relation="identity",
        actions=(GroundedAction("ACTION6", (("x", 56), ("y", 29))),),
    )
    assert left.key != right.key
    assert left.transferable_descriptor == right.transferable_descriptor


def test_config_rejects_unbounded_horizon() -> None:
    with pytest.raises(ValueError):
        SearchConfig(maximum_horizon=65)
