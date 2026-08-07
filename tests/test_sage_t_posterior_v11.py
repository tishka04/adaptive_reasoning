from __future__ import annotations

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.posterior_v8 import T8_6G_POLICIES
from theory.sage_t.posterior_v11 import BudgetedRepairProgramPosterior
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor

SELECTED = "terminal_tempered_20_family_floor_0501_minimum_kl"


def _posterior(maximum: int) -> BudgetedRepairProgramPosterior:
    manifest = v86.load_t7_manifest(verify_code=True)
    config = manifest["posterior"]
    return BudgetedRepairProgramPosterior(
        executor=StructuralRoleProgramExecutor(),
        update_policy=T8_6G_POLICIES[SELECTED].with_repair_v2(),
        maximum_particles=int(config["maximum_particles"]),
        channel_weights=v86._weights("joint"),
        unknown_coverage_penalty=float(config["unknown_coverage_penalty"]),
        repair_ess_threshold=float(config["repair_ess_threshold"]),
        repair_log_likelihood_threshold=float(
            config["repair_log_likelihood_threshold"]
        ),
        maximum_repair_contexts=maximum,
    )


def test_global_repair_budget_rejects_a_new_context() -> None:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    sequence = next(iter(v86._signal_sequences(pairs)))
    first = next(
        arm
        for arm in sequence["panels"][0].arms
        if arm.action.key == sequence["keys"][0]
    )
    second = sequence["positive"]
    manifest = v86.load_t7_manifest(verify_code=True)
    programs = programs_for_with_structural_goal_guard(
        (first.action.action_name, second.action.action_name),
        (),
        manifest,
    )
    posterior = _posterior(1)
    posterior.seed(programs, initial_state=first.state_before)
    posterior.observe(first, allow_repair=False)

    posterior.repair(first)
    skipped = posterior.repair(second)

    assert skipped == ()
    performance = posterior.performance_snapshot()
    assert performance["unique_repair_contexts"] == 1
    assert performance["repair_budget_skips"] == 1
    assert performance["maximum_repair_contexts"] == 1


def test_zero_budget_keeps_updates_but_disables_repair() -> None:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    sequence = next(iter(v86._signal_sequences(pairs)))
    evidence = sequence["positive"]
    manifest = v86.load_t7_manifest(verify_code=True)
    programs = programs_for_with_structural_goal_guard(
        (evidence.action.action_name,), (), manifest
    )
    posterior = _posterior(0)
    posterior.seed(programs, initial_state=evidence.state_before)
    posterior.observe(evidence, allow_repair=False)

    assert posterior.repair(evidence) == ()
    assert posterior.performance_snapshot()["repair_budget_skips"] == 1
    assert len(posterior.history) == 1
