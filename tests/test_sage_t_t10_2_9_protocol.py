from __future__ import annotations

from pathlib import Path

import pytest

from theory.sage_t import t10_2_9_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]


def test_correction_policy_is_seed_adapter_only_and_fail_closed() -> None:
    policy = protocol.correction_policy()
    assert policy["change_scope"] == "durable_source_seed_registry_adapter_only"
    assert policy["durable_discovery_seeds"] == [101, 102, 103]
    assert policy["durable_confirmation_seeds"] == [111, 112, 113]
    assert policy["scientific_qa_computation_unchanged"] is True
    assert policy["environment_calls_authorized"] == 0
    assert policy["physical_actions_authorized"] == 0
    assert policy["model_fit_authorized"] is False
    assert policy["source_validation_authorized"] is False
    assert policy["ar25_authorized"] is False
    assert policy["holdout_authorized"] is False


def test_handoff_authenticates_fail_closed_t10_2_8() -> None:
    receipt = protocol.build_handoff_receipt(repo_root=ROOT)
    assert receipt["predecessor_terminal_checksum"] == protocol.PREDECESSOR_TERMINAL_CHECKSUM
    assert receipt["adapter_failure"] == {
        "classification": "IMPLEMENTATION_ADAPTER_DEFECT",
        "schema_error": protocol.EXPECTED_ADAPTER_ERROR,
        "scientific_qa_evaluated": False,
        "model_fit_opened": False,
        "active_validation_opened": False,
    }
    assert receipt["source_seed_registry"] == {
        "discovery": [101, 102, 103],
        "leave_one_game_out_confirmation": [111, 112, 113],
        "recovery_confirmation": [3_119_945],
    }
    assert receipt["physical_actions_authorized"] == 0


def test_handoff_checksum_tamper_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = protocol.build_handoff_receipt(repo_root=ROOT)
    tampered = dict(receipt)
    tampered["physical_actions_authorized"] = 1
    with pytest.raises(protocol.ManifestDriftError, match="checksum drifted"):
        protocol.verify_handoff_receipt_live(tampered, repo_root=ROOT)


def test_artifact_contract_uses_append_only_output() -> None:
    contract = protocol.artifact_contract()
    assert contract["predecessor_root"] == "training/sage_t/t10_2_8_offline_qa"
    assert contract["output_root"] == "training/sage_t/t10_2_9_offline_qa"
    assert contract["terminal_report"] == "t10_2_9_report.json"
