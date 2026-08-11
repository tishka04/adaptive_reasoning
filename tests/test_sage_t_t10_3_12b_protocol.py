from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12b_protocol as protocol


def test_protocol_is_zero_physical_action_and_keeps_authority_closed() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    assert manifest["matrix"]["physical_actions"] == 0
    assert manifest["claim_boundary"]["cross_game_generalization_proven"] is False
    assert manifest["claim_boundary"]["sequence_composition_authorized"] is False
    assert manifest["firewall"]["new_arc_physical_actions_authorized"] is False
    assert manifest["firewall"]["sequence_games_opened"] is False
    assert manifest["firewall"]["holdout_opened"] is False
    assert manifest["firewall"]["production_authority"] is False


def test_factorial_matrix_and_gates_are_preregistered() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    assert manifest["matrix"]["variants"] == 128
    assert manifest["matrix"]["identification_variants"] == 64
    assert manifest["matrix"]["challenge_variants"] == 64
    assert manifest["matrix"]["ambiguous_variants"] == 32
    assert manifest["gates"]["minimum_factor_gap_per_context"] == 8
    assert manifest["gates"]["minimum_distinct_state_hashes_per_context"] == 48
    assert manifest["gates"]["source_role_decoupled_correct"] == 32
    assert manifest["gates"]["minimum_first_decision_divergence"] == 96
    assert manifest["gates"]["maximum_source_to_generic_action_ratio"] == 0.80


def test_parent_is_pinned_as_clean_generic_rediscovery_negative() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    assert manifest["parent_expected"]["verdict"] == "GENERIC_REDISCOVERY_ONLY"
    assert manifest["parent_expected"]["authorized_actions"] == 336
    assert manifest["parent_expected"]["sealed_events"] == 336
    assert manifest["negative_result_policy"]["no_post_freeze_repair"] is True
    assert manifest["negative_result_policy"]["no_program_promotion"] is True


def test_artifact_contract_compiles_before_scoring() -> None:
    phases = list(protocol.ARTIFACT_CONTRACT)
    assert phases.index("materialize-variants") < phases.index("compile-factors")
    assert phases.index("compile-factors") < phases.index("evaluate-interventions")
    assert protocol.ARTIFACT_CONTRACT["evaluate-interventions"]["gate_field"] == "passed"
