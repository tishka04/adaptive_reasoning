from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12c_protocol as protocol


def test_matrix_includes_every_remaining_source_train_game_once_per_arm() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    specs = protocol.work_specs("active-transfer")
    assert len(manifest["matrix"]["games"]) == 9
    assert manifest["matrix"]["selection_rule"] == (
        "all_remaining_source_train_games_excluding_lp85_su15"
    )
    assert len(specs) == 54
    assert len({(work.game_id, work.arm) for work in specs}) == 54
    assert manifest["matrix"]["maximum_actions"] == 864
    assert manifest["matrix"]["labels_seed_environment"] is False


def test_firewalls_and_claim_boundary_remain_closed() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    firewall = manifest["firewall"]
    assert firewall["legacy_fallback_authorized"] is False
    assert firewall["sequence_games_opened"] is False
    assert firewall["source_validation_opened"] is False
    assert firewall["ar25_opened"] is False
    assert firewall["holdout_opened"] is False
    assert firewall["production_authority"] is False
    assert manifest["claim_boundary"]["sequence_composition_authorized"] is False


def test_parent_pass_and_all_four_candidates_are_pinned() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    assert manifest["parent_state"] == protocol.EXPECTED_PARENT
    assert manifest["parent_state"]["identified_factor_candidates"] == [
        "operator", "role_binding", "transition", "termination"
    ]
    assert manifest["negative_result_policy"]["operator_coverage_miss_is_scientific_result"]
    assert manifest["negative_result_policy"]["no_program_promotion"]


def test_compilation_precedes_first_target_action() -> None:
    phases = manifest_phases = list(protocol.ARTIFACT_CONTRACT)
    assert phases.index("audit-targets") < phases.index("compile-transfer")
    assert phases.index("compile-transfer") < phases.index("active-transfer")
    assert manifest_phases[-1] == "report"
