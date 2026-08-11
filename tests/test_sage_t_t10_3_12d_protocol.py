from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12d_protocol as protocol


def test_parent_negative_is_exact_and_immutable() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    assert manifest["parent_state"] == protocol.EXPECTED_PARENT
    assert manifest["parent_state"]["verdict"] == "CROSS_GAME_TRANSFER_MISS"
    assert manifest["parent_state"]["authorized_actions"] == 350
    assert manifest["parent_state"]["sealed_events"] == 350
    assert manifest["parent_state"]["receipt_count"] == 54
    assert manifest["post_hoc_diagnostic"] is True


def test_matrix_is_path_only_and_includes_all_observed_games() -> None:
    manifest = protocol.build_manifest(Path.cwd())
    specs = protocol.work_specs("active-diagnostic")
    assert manifest["matrix"]["path_context_only"] is True
    assert manifest["matrix"]["non_path_context_policy"] == (
        "uniform_zero_action_abstention"
    )
    assert manifest["matrix"]["games_already_observed_in_t10_3_12c"] is True
    assert len(specs) == 36
    assert len({(work.game_id, work.arm) for work in specs}) == 36
    assert manifest["matrix"]["maximum_actions"] == 576
    assert manifest["matrix"]["labels_seed_environment"] is False


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
    assert firewall["legacy_fallback_authorized"] is False


def test_executor_compilation_precedes_diagnostic_actions() -> None:
    phases = list(protocol.ARTIFACT_CONTRACT)
    assert phases.index("audit-trajectories") < phases.index("preflight")
    assert phases.index("preflight") < phases.index("compile-executors")
    assert phases.index("compile-executors") < phases.index("active-diagnostic")
    assert protocol.ARTIFACT_CONTRACT["active-diagnostic"]["gate_field"] == (
        "collection_complete"
    )
