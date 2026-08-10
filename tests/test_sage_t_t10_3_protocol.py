from __future__ import annotations

from theory.sage_t import t10_3_protocol as protocol


def test_frozen_matrix_is_exactly_sixty_resets_and_960_actions() -> None:
    matrix = protocol.frozen_matrix()
    assert matrix["panel"]["resets"] == 48
    assert matrix["panel"]["maximum_actions"] == 768
    assert matrix["confirmation"]["resets"] == 12
    assert matrix["confirmation"]["maximum_actions"] == 192
    assert matrix["total_resets"] == 60
    assert matrix["total_maximum_actions"] == 960
    assert matrix["padding_authorized"] is False
    assert matrix["seed_replacement_authorized"] is False


def test_handoff_authenticates_negative_t10_2_9_and_positive_t10_0b() -> None:
    receipt = protocol.build_handoff_receipt(repo_root=".")
    assert receipt["t10_2_9_terminal_checksum"] == protocol.T10_2_9_TERMINAL_CHECKSUM
    assert receipt["t10_2_9_qa_checksum"] == protocol.T10_2_9_QA_CHECKSUM
    assert receipt["t10_0b_report_checksum"] == protocol.T10_0B_REPORT_CHECKSUM
    assert receipt["t10_2_9_fit_excluded"] is True
    assert len(receipt["canonical_witnesses"]) == 2
    rendered = protocol.canonical_json(receipt["canonical_witnesses"]).lower()
    assert '"x":' not in rendered
    assert '"y":' not in rendered
    assert '"action_data":' not in rendered


def test_firewall_never_authorizes_validation_or_retuning() -> None:
    firewall = protocol.firewall_policy()
    assert firewall["source_validation_authorized"] is False
    assert firewall["ar25_authorized"] is False
    assert firewall["holdout_authorized"] is False
    assert firewall["production_authority"] is False
    assert firewall["automatic_retuning_authorized"] is False
    assert firewall["physical_replay_authorized"] is False

