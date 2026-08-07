from __future__ import annotations

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.posterior_v8 import T8_6G_POLICIES
from theory.sage_t.posterior_v10 import ContextMemoizedRepairProgramPosterior
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor

SELECTED = "terminal_tempered_20_family_floor_0501_minimum_kl"


def _posterior() -> ContextMemoizedRepairProgramPosterior:
    manifest = v86.load_t7_manifest(verify_code=True)
    config = manifest["posterior"]
    return ContextMemoizedRepairProgramPosterior(
        executor=StructuralRoleProgramExecutor(),
        update_policy=T8_6G_POLICIES[SELECTED].with_repair_v2(),
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=v86._weights("joint"),
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            config["repair_log_likelihood_threshold"]
        ),
    )


def _evidence():
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    sequence = next(iter(v86._signal_sequences(pairs)))
    return next(
        arm
        for arm in sequence["panels"][0].arms
        if arm.action.key == sequence["keys"][0]
    )


def test_same_repair_context_is_attempted_only_once_across_branches() -> None:
    evidence = _evidence()
    manifest = v86.load_t7_manifest(verify_code=True)
    actions = (evidence.action.action_name,)
    programs = programs_for_with_structural_goal_guard(actions, (), manifest)
    posterior = _posterior()
    posterior.seed(programs, initial_state=evidence.state_before)
    posterior.observe(evidence, allow_repair=False)

    posterior.repair(evidence)
    after_first = posterior.snapshot(maximum_programs=0)
    posterior.start_branch(regime_index=evidence.state_before.regime_index)
    skipped = posterior.repair(evidence)
    after_second = posterior.snapshot(maximum_programs=0)

    assert skipped == ()
    assert after_second["repair_cycles"] == after_first["repair_cycles"]
    performance = posterior.performance_snapshot()
    assert performance["unique_repair_contexts"] == 1
    assert performance["repair_context_skips"] == 1


def test_different_observed_result_gets_a_distinct_repair_budget() -> None:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    sequence = next(iter(v86._signal_sequences(pairs)))
    neutral = next(
        arm
        for arm in sequence["panels"][0].arms
        if arm.action.key == sequence["keys"][0]
    )
    positive = sequence["positive"]

    assert (
        ContextMemoizedRepairProgramPosterior._repair_context_key(neutral)
        != ContextMemoizedRepairProgramPosterior._repair_context_key(positive)
    )
