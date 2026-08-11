from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12e_protocol as protocol


def test_parent_negative_is_exact_and_immutable() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    assert manifest["parent_state"] == protocol.EXPECTED_PARENT
    assert manifest["parent_state"]["verdict"] == "STABLE_EXECUTOR_NO_PROGRESS"
    assert manifest["parent_state"]["authorized_actions"] == 91
    assert manifest["parent_state"]["sealed_events"] == 91
    assert manifest["parent_state"]["receipt_count"] == 36
    assert manifest["post_hoc_diagnostic"] is True


def test_matrix_is_path_only_and_keeps_full_observed_panel() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    specs = protocol.work_specs("active-diagnostic")
    assert manifest["matrix"]["path_context_only"] is True
    assert manifest["matrix"]["non_path_context_policy"] == (
        "uniform_zero_action_abstention"
    )
    assert manifest["matrix"]["games_already_observed_in_t10_3_12c_and_d"] is True
    assert len(specs) == 36
    assert len({(work.game_id, work.arm) for work in specs}) == 36
    assert manifest["matrix"]["maximum_actions"] == 576
    assert manifest["matrix"]["labels_seed_environment"] is False


def test_primary_gate_requires_terminal_advantage_over_all_controls() -> None:
    gates = protocol.build_manifest(Path.cwd())["gates"]
    assert gates["minimum_dynamic_success_games"] == 1
    assert gates["minimum_dynamic_over_frozen_success_advantage"] == 1
    assert gates["minimum_dynamic_over_stateless_success_advantage"] == 1
    assert gates["minimum_dynamic_over_goal_swap_success_advantage"] == 1
    assert gates["maximum_dynamic_grounding_misses"] == 0
    assert gates["minimum_dynamic_exact_grounding_fraction"] == 1.0
    assert gates["minimum_dynamic_frontier_advance_fraction"] == 1.0


def test_claim_and_authority_boundaries_remain_closed() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    claim = manifest["claim_boundary"]
    firewall = manifest["firewall"]
    assert claim["cross_game_generalization_proven"] is False
    assert claim["factor_generalization_proven"] is False
    assert claim["independent_confirmation"] is False
    assert claim["sequence_composition_authorized"] is False
    assert firewall["new_games_opened"] is False
    assert firewall["source_validation_opened"] is False
    assert firewall["holdout_opened"] is False
    assert firewall["production_authority"] is False
    assert firewall["parent_grounded_paths_compiled_into_programs"] is False
    assert firewall["parent_action_checksums_compiled_into_programs"] is False


def test_program_compilation_precedes_active_diagnostic() -> None:
    phases = list(protocol.ARTIFACT_CONTRACT)
    assert phases.index("audit-trajectories") < phases.index("preflight")
    assert phases.index("preflight") < phases.index("compile-programs")
    assert phases.index("compile-programs") < phases.index("active-diagnostic")
    assert protocol.ARTIFACT_CONTRACT["active-diagnostic"]["gate_field"] == (
        "collection_complete"
    )
