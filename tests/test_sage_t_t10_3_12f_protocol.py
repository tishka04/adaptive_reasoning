from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from theory.sage_t import t10_3_12f_protocol as protocol


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    return protocol.build_manifest(Path.cwd())


def test_parent_negative_is_exact_clean_and_immutable() -> None:
    observed = protocol.verify_parent(Path.cwd())
    assert observed == protocol.EXPECTED_PARENT
    assert observed["verdict"] == "CLOSED_LOOP_NO_PROGRESS"
    assert observed["authorized_actions"] == 82
    assert observed["sealed_events"] == 82
    assert observed["inflight_intents"] == 0
    assert observed["unresolved_intents"] == 0
    assert observed["receipt_count"] == 36


def test_source_bindings_are_only_lp85_su15_and_have_success_evidence() -> None:
    observed = protocol.verify_source_evidence(Path.cwd())
    assert observed == protocol.EXPECTED_SOURCE_SUCCESS
    assert observed["canonical_levels"] == {
        "lp85-305b61c3": 1,
        "su15-4c352900": 2,
    }
    bindings = protocol.source_artifact_bindings(Path.cwd())
    assert "lp85_shard" in bindings
    assert "su15_shard" in bindings
    assert "canonical_witness" in bindings
    assert "discovery_core" in bindings
    assert "reproduction_core" in bindings


def test_historical_matrix_is_complete_bounded_and_latin() -> None:
    specs = protocol.work_specs("active-historical")
    assert len(specs) == 144
    assert len({work.work_id for work in specs}) == 144
    assert len({(work.game_id, work.scope_index, work.arm) for work in specs}) == 144
    assert all(work.action_budget == 48 for work in specs)
    assert protocol.maximum_actions_for_specs(specs) == 6_912
    assert protocol.maximum_actions_for_phase("active-historical") == 6_912

    for game in protocol.TARGET_GAMES:
        rows = [work for work in specs if work.game_id == game]
        assert len(rows) == 16
        assert Counter(work.arm for work in rows) == Counter(
            {arm: 4 for arm in protocol.ARMS}
        )
        assert len({work.tie_break_seed for work in rows}) == 4
        for position in range(4):
            position_arms = [rows[scope * 4 + position].arm for scope in range(4)]
            assert set(position_arms) == set(protocol.ARMS)


def test_manifest_separates_source_and_generic_hypotheses(
    manifest: dict[str, object],
) -> None:
    hypotheses = manifest["hypotheses"]
    matrix = manifest["matrix"]
    endpoints = manifest["endpoints"]
    gates = manifest["gates"]
    assert set(hypotheses) == {"procedure", "source"}
    assert matrix["arms"] == list(protocol.ARMS)
    assert matrix["resets"] == 144
    assert matrix["maximum_actions_per_reset"] == 48
    assert matrix["maximum_actions"] == 6_912
    assert matrix["work_scope_affects_only_tie_breaks"] is True
    assert matrix["work_scope_is_environment_seed"] is False
    assert endpoints["terminal_level_delta_is_only_success_credit"] is True
    assert endpoints["statistical_unit"] == "game"
    assert gates["minimum_candidate_success_games"] == 2
    assert gates["minimum_identification_verified_games"] == 2
    assert gates["minimum_identification_better_games_each_control"] == 5
    assert gates["holm_familywise_alpha"] == 0.05
    assert gates["all_receipts_required"] == 144
    arm_contract = manifest["arm_contract"]
    assert arm_contract[
        "shared_representation_learner_observations_thresholds_and_budgets"
    ] is True
    assert arm_contract["source_closed_loop"]["revision"] is True
    assert arm_contract["uniform_closed_loop"]["revision"] is True
    assert arm_contract["permuted_source_closed_loop"]["same_entropy_as_source"] is True
    assert arm_contract["permuted_source_closed_loop"]["same_norm_as_source"] is True
    assert arm_contract["source_open_loop"]["revision"] is False
    assert arm_contract["source_open_loop"]["first_verified_hypothesis_locked"] is True


def test_source_prior_contract_is_balanced_and_transfer_safe(
    manifest: dict[str, object],
) -> None:
    source = manifest["source_policy"]
    gates = manifest["gates"]
    assert source["games"] == list(protocol.SOURCE_GAMES)
    assert source["all_existing_signed_interventions_included"] is True
    assert source["successes_and_failures_included"] is True
    assert source["new_source_physical_actions"] == 0
    assert source["effects_reconstructed_from_frame_pairs"] is True
    assert source["legacy_effect_labels_trusted"] is False
    assert source["minimum_correspondence_confidence"] == 0.60
    assert source["source_game_weights"] == {
        "lp85-305b61c3": 0.5,
        "su15-4c352900": 0.5,
    }
    assert set(source["forbidden_transfer_fields"]) >= {
        "game_id",
        "action_name",
        "coordinates",
        "colors",
        "object_ids",
        "frame_hashes",
        "source_trajectories",
    }
    assert gates["source_minimum_effect_modes_per_game"] == 2
    assert gates["source_maximum_single_label_fraction_exclusive"] == 0.95
    assert gates["source_prior_maximum_family_weight"] == 0.70
    assert gates["minimum_log_loss_improvement_over_permuted_each_source"] == 0.05


def test_cli_artifacts_and_firewalls_are_fail_closed(
    manifest: dict[str, object],
) -> None:
    assert manifest["cli_phases"] == [
        "freeze",
        "status",
        "audit",
        "qa-source",
        "compile-prior",
        "evaluate-source",
        "preflight",
        "active-historical",
        "adjudicate",
        "report",
    ]
    assert list(protocol.ARTIFACT_CONTRACT) == [
        "audit",
        "qa-source",
        "compile-prior",
        "evaluate-source",
        "preflight",
        "active-historical",
        "adjudicate",
        "report",
    ]
    assert protocol.ARTIFACT_CONTRACT["active-historical"]["gate_field"] == (
        "collection_complete"
    )
    assert protocol.ARTIFACT_CONTRACT["compile-prior"]["role"] == "registry"
    firewall = manifest["firewall"]
    for field in (
        "new_games_opened",
        "sequence_games_opened",
        "source_validation_opened",
        "ar25_opened",
        "holdout_opened",
        "production_authority",
        "automatic_retuning",
        "legacy_fallback_authorized",
        "t10_3_13_authorized",
        "source_games_physical_actions_authorized",
        "source_shards_physical_replay_authorized",
        "t10_3_12e_events_training_authorized",
        "t10_3_12c_to_e_events_initialization_authorized",
    ):
        assert firewall[field] is False
    claim = manifest["claim_boundary"]
    assert claim["prospective_generalization_proven"] is False
    assert claim["independent_confirmation"] is False
    assert claim["t10_3_13_holdout_authorized"] is False


def test_manifest_and_custom_freeze_receipt_are_signed(
    tmp_path: Path,
    manifest: dict[str, object],
) -> None:
    protocol.verify_signed(manifest, "manifest_checksum")
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    frozen, receipt = protocol.freeze_manifest(
        Path.cwd(), manifest_path=manifest_path, receipt_path=receipt_path
    )
    protocol.verify_signed(frozen, "manifest_checksum")
    protocol.verify_signed(receipt, "receipt_checksum")
    assert receipt["manifest_checksum"] == frozen["manifest_checksum"]
    assert receipt["maximum_resets"] == 144
    assert receipt["maximum_actions"] == 6_912
    assert receipt["physical_actions_at_freeze"] == 0
    assert receipt["holdout_opened"] is False
    assert receipt["t10_3_13_authorized"] is False
    assert manifest_path.is_file()
    assert receipt_path.is_file()


def test_unknown_physical_phase_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        protocol.work_specs("active-holdout")
