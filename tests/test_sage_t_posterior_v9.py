from __future__ import annotations

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.goal_generation_v3 import (
    programs_for_with_structural_goal_guard,
)
from theory.sage_t.posterior_v8 import (
    T8_6G_POLICIES,
    MinimumKLFamilyFloorProgramPosterior,
)
from theory.sage_t.posterior_v9 import IncrementalMinimumKLProgramPosterior
from theory.sage_t.structural_roles import StructuralRoleProgramExecutor

SELECTED = "terminal_tempered_20_family_floor_0501_minimum_kl"


def _posterior(cls):
    manifest = v86.load_t7_manifest(verify_code=True)
    config = manifest["posterior"]
    return cls(
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


def _goal_sequence():
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    return next(
        sequence
        for sequence in v86._signal_sequences(pairs)
        if sequence["positive_kind"] == "goal"
    )


def _fingerprint(posterior) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            particle.program.canonical_hash,
            particle.log_prior,
            particle.log_joint,
            particle.log_weight,
            particle.latest_log_likelihood,
            particle.latest_raw_log_likelihood,
            particle.observations,
            None
            if particle.state is None
            else particle.state.execution_signature,
        )
        for particle in posterior.particles
    )


def _scientific_diagnostics(posterior) -> dict[str, object] | None:
    diagnostics = posterior.last_update_diagnostics
    if diagnostics is None:
        return None
    payload = diagnostics.to_dict()
    for key in (
        "elapsed_ms",
        "executor_cache_hits_delta",
        "executor_cache_misses_delta",
    ):
        payload.pop(key, None)
    return payload


def test_incremental_path_is_exact_over_neutral_history_and_goal_shock() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    sequence = _goal_sequence()
    revealed = [
        next(arm for arm in panel.arms if arm.action.key == key)
        for panel, key in zip(sequence["panels"], sequence["keys"])
    ]
    actions = tuple(
        sorted(
            {
                arm.action.action_name
                for panel in sequence["panels"]
                for arm in panel.arms
            }
        )
    )
    baseline = _posterior(MinimumKLFamilyFloorProgramPosterior)
    incremental = _posterior(IncrementalMinimumKLProgramPosterior)
    history = []
    for index, evidence in enumerate(revealed):
        programs = programs_for_with_structural_goal_guard(
            actions, history, manifest
        )
        if index == 0:
            baseline.seed(programs, initial_state=evidence.state_before)
            incremental.seed(programs, initial_state=evidence.state_before)
        else:
            baseline.add_programs(programs, initial_state=evidence.state_before)
            incremental.add_programs(
                programs, initial_state=evidence.state_before
            )
        assert _fingerprint(incremental) == _fingerprint(baseline)
        baseline.observe(evidence, allow_repair=False)
        incremental.observe(evidence, allow_repair=False)
        assert _fingerprint(incremental) == _fingerprint(baseline)
        assert _scientific_diagnostics(incremental) == _scientific_diagnostics(
            baseline
        )
        history.append(evidence)

    goal_programs = programs_for_with_structural_goal_guard(
        actions, history, manifest
    )
    baseline.add_programs(goal_programs, initial_state=revealed[-1].state_after)
    incremental.add_programs(
        goal_programs, initial_state=revealed[-1].state_after
    )
    assert _fingerprint(incremental) == _fingerprint(baseline)


def test_repeated_unchanged_program_set_uses_noop_and_incremental_signatures() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    sequence = _goal_sequence()
    evidence = next(
        arm
        for arm in sequence["panels"][0].arms
        if arm.action.key == sequence["keys"][0]
    )
    actions = (evidence.action.action_name,)
    programs = programs_for_with_structural_goal_guard(actions, (), manifest)
    posterior = _posterior(IncrementalMinimumKLProgramPosterior)
    posterior.seed(programs, initial_state=evidence.state_before)

    for _ in range(40):
        posterior.add_programs(programs, initial_state=evidence.state_before)
        posterior.observe(evidence, allow_repair=False)

    performance = posterior.performance_snapshot()
    assert performance["noop_program_additions"] == 40
    assert performance["full_program_additions"] == 0
    assert performance["novel_programs_replayed"] == 0
    assert performance["semantic_cache_extensions"] <= (
        len(programs) * len(posterior.history)
    )
    assert performance["semantic_cache_hits"] > 0
