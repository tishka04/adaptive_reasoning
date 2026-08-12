from dataclasses import replace

import pytest

from tests.test_sage_t_causal_contract_executor import causal_program, initial_state
from theory.sage_t.causal.comparison import compare_particle
from theory.sage_t.causal.contracts import (
    ActionProgram,
    CausalState,
    GroundedAction,
    StructuredDelta,
    TransitionEvidence,
    ValueDistribution,
)
from theory.sage_t.causal.decision import CausalDecisionEngine
from theory.sage_t.causal.executor import CausalExecutor
from theory.sage_t.causal.memory import CausalMemoryStore
from theory.sage_t.causal.posterior import CausalPosterior
from theory.sage_t.causal.repair import CausalProgramRepairer
from theory.sage_t.causal.runtime import CausalRuntime


def matching_evidence(*, evidence_id="evidence-1", noisy=False):
    before = initial_state()
    after_values = dict(before.variables)
    after_values.update(
        {
            "object.color": (
                ValueDistribution({'"blue"': 0.9, '"red"': 0.1})
                if noisy
                else ValueDistribution.deterministic("blue")
            ),
            "pair.aligned": ValueDistribution.deterministic(True),
            "level.complete": ValueDistribution.deterministic(True),
        }
    )
    after = CausalState(variables=after_values)
    return TransitionEvidence(
        evidence_id=evidence_id,
        state_before=before,
        action=GroundedAction("CLICK"),
        state_after=after,
        observed_delta=StructuredDelta(
            variable_changes={
                "object.color": after.value("object.color"),
                "pair.aligned": after.value("pair.aligned"),
                "level.complete": after.value("level.complete"),
            },
            affected_objects=("object", "pair", "level"),
            relation_changes=("pair.aligned",),
            progress=1.0,
        ),
        terminal=True,
        success=True,
        level_change=1,
        prefix_hash="exact-prefix-hash",
        game_id="bp35",
        context_id="context-a",
    )


def test_posterior_concentrates_without_history_only_deduplication():
    executor = CausalExecutor()
    posterior = CausalPosterior(
        executor=executor,
        mdl_beta=0.0,
        repairer=CausalProgramRepairer(maximum_children=0),
    )
    correct = causal_program(program_id="cycle", color_operator="cycle_attribute")
    rival = causal_program(program_id="identity", color_operator="identity")
    posterior.seed((correct, rival))
    assert len(posterior.particles) >= 2

    update = posterior.update(matching_evidence(noisy=True))
    probabilities = {
        particle.program.program_id: particle.probability
        for particle in posterior.particles
    }
    assert probabilities["cycle"] > probabilities["identity"]
    assert len(posterior.particles) >= 2
    assert update.entropy_after < update.entropy_before
    assert all(particle.probability > 0.0 for particle in posterior.particles)

    comparison = compare_particle(
        program=correct,
        evidence=matching_evidence(),
        executor=executor,
    )
    assert set(comparison.responsibility) == {
        "binding",
        "dynamics",
        "goal",
        "observation_model",
    }


def test_exact_duplicate_mass_is_merged_but_rivals_are_retained():
    executor = CausalExecutor()
    posterior = CausalPosterior(executor=executor, mdl_beta=0.0)
    first = causal_program(program_id="first")
    alias = replace(first, program_id="alias")
    rival = causal_program(program_id="identity", color_operator="identity")
    posterior.seed((first, alias, rival))
    assert len(posterior.particles) == 2
    cycle_particle = next(
        particle for particle in posterior.particles
        if particle.program.canonical_hash == first.canonical_hash
    )
    assert cycle_particle.probability > 0.5


def test_a40_roundtrip_restores_posterior_mass(tmp_path):
    memory_path = tmp_path / "causal-memory.jsonl"
    executor = CausalExecutor()
    posterior = CausalPosterior(
        executor=executor,
        repairer=CausalProgramRepairer(maximum_children=0),
    )
    runtime = CausalRuntime(
        executor=executor,
        posterior=posterior,
        memory_path=memory_path,
    )
    runtime.seed(
        (
            causal_program(program_id="cycle", color_operator="cycle_attribute"),
            causal_program(program_id="identity", color_operator="identity"),
        )
    )
    runtime.observe(matching_evidence())
    expected = {
        particle.program.canonical_hash: particle.probability
        for particle in runtime.posterior.particles
    }

    restored = CausalRuntime(memory_path=memory_path)
    assert restored.reload_memory() == len(expected)
    actual = {
        particle.program.canonical_hash: particle.probability
        for particle in restored.posterior.particles
    }
    assert actual == pytest.approx(expected)
    assert len(restored.posterior.evidence) == 1
    assert restored.posterior.evidence[0].evidence_id == "evidence-1"
    assert len(CausalMemoryStore(memory_path).verified_records()) == 1


def test_a40_omits_undeclared_world_state_and_reserves_before_write(tmp_path):
    evidence = matching_evidence()
    noisy_before = dict(evidence.state_before.variables)
    noisy_after = dict(evidence.state_after.variables)
    for index in range(2000):
        key = f"fact.synthetic.{index}"
        noisy_before[key] = ValueDistribution.deterministic(False)
        noisy_after[key] = ValueDistribution.deterministic(True)
    evidence = replace(
        evidence,
        state_before=replace(evidence.state_before, variables=noisy_before),
        state_after=replace(evidence.state_after, variables=noisy_after),
    )
    reservations = []
    path = tmp_path / "compact.jsonl"
    executor = CausalExecutor()
    posterior = CausalPosterior(executor=executor)
    runtime = CausalRuntime(
        executor=executor,
        posterior=posterior,
        memory_path=path,
        reserve_memory_bytes=reservations.append,
    )
    runtime.seed((causal_program(), causal_program(program_id="rival", color_operator="identity")))
    runtime.observe(evidence)
    assert reservations and reservations[0] == path.stat().st_size
    assert path.stat().st_size < 50_000
    payload = CausalMemoryStore(path).verified_records()[0].payload
    assert payload["evidence_compaction"]["declared_variables_only"] is True
    assert not any(
        key.startswith("fact.synthetic")
        for key in payload["evidence"]["state_after"]["variables"]
    )


def test_exact_route_remains_lexicographically_above_causal_probe():
    executor = CausalExecutor()
    posterior = CausalPosterior(executor=executor, mdl_beta=0.0)
    posterior.seed(
        (
            causal_program(program_id="cycle", color_operator="cycle_attribute"),
            causal_program(program_id="identity", color_operator="identity"),
        )
    )
    decision = CausalDecisionEngine(executor=executor).decide(
        posterior,
        initial_state(),
        (
            ActionProgram((GroundedAction("CLICK"),), source="exact_route"),
            ActionProgram((GroundedAction("CLICK", {"probe": True}),), source="causal_probe"),
        ),
    )
    assert decision.chosen is not None
    assert decision.chosen.action_program.source == "exact_route"


def test_same_action_keeps_the_highest_priority_source():
    executor = CausalExecutor()
    posterior = CausalPosterior(executor=executor, mdl_beta=0.0)
    posterior.seed((causal_program(),))
    action = GroundedAction("CLICK")
    decision = CausalDecisionEngine(executor=executor).decide(
        posterior,
        initial_state(),
        (
            ActionProgram((action,), source="exact_route"),
            ActionProgram((action,), source="generic"),
        ),
    )
    assert decision.chosen is not None
    assert decision.chosen.action_program.source == "exact_route"
