from __future__ import annotations

from theory.sage_t import t10_3_1_protocol as protocol
from theory.sage_t import t10_3_protocol as parent


def test_fresh_matrix_preserves_budget_without_reusing_parent_seeds() -> None:
    matrix = protocol.frozen_matrix()
    assert matrix["total_resets"] == 60
    assert matrix["total_maximum_actions"] == 960
    assert set(protocol.PANEL_SEEDS).isdisjoint(parent.PANEL_SEEDS)
    assert set(protocol.CONFIRMATION_SEEDS).isdisjoint(parent.CONFIRMATION_SEEDS)
    assert matrix["padding_authorized"] is False


def test_migration_receipt_authenticates_rooting_miss_and_excludes_parent_fit() -> None:
    receipt = protocol.build_migration_receipt(repo_root=".")
    assert receipt["parent_terminal_checksum"] == protocol.PARENT_TERMINAL_CHECKSUM
    assert receipt["parent_qa_checksum"] == protocol.PARENT_QA_CHECKSUM
    assert receipt["parent_checkpoint_checksum"] == protocol.PARENT_CHECKPOINT_CHECKSUM
    assert receipt["parent_diagnosis"]["verdict"] == "ROOTING_MISS"
    assert receipt["parent_diagnosis"]["canonical_positive_branches"] == 8
    assert receipt["parent_events_used_for_fit"] == 0
    assert receipt["parent_events_relabelled"] == 0


def test_correction_and_firewall_are_fail_closed() -> None:
    correction = protocol.correction_policy()
    firewall = protocol.firewall_policy()
    assert correction["dynamic_action_regrounding_each_step"] is True
    assert correction["root_only_to_richer_frames_comparable"] is False
    assert correction["parent_physical_events_fit_authorized"] is False
    assert firewall["source_validation_authorized"] is False
    assert firewall["ar25_authorized"] is False
    assert firewall["holdout_authorized"] is False
    assert firewall["production_authority"] is False

