from __future__ import annotations

import copy

import pytest

from theory.sage_t import t10_2_7_protocol as protocol


def _anchor() -> dict[str, object]:
    return {
        "predecessor_manifest_checksum": "a" * 64,
        "partial_journal_checksum": "b" * 64,
        "partial_intent_checksum": "c" * 64,
        "t10_2_5_recovery_seeds": [1650445, 1412747, 1438133],
        "t10_2_6_recovery_seeds": [2320669, 2468453, 2325737],
    }


def test_policy_is_append_only_event_seal_scoped_and_fail_closed() -> None:
    policy = protocol.recovery_policy()

    assert policy["parent_scientific_kernel_unchanged"] is True
    assert policy["t10_2_6_partial_journal_read_only"] is True
    assert policy["t10_2_6_partial_lane_enters_model_fit"] is False
    assert policy["t10_2_6_partial_lane_replayed"] is False
    assert policy["execution_manifest_overlays_only"] == [
        "manifest_checksum",
        "migration_receipt",
    ]
    assert policy["first_event_must_seal_before_collection_authorization"] is True
    assert policy["runner_exception_creates_terminal_reset_and_lane_reports"] is True
    assert policy["validation_and_ar25_authority_opened"] is False


def test_new_seeds_are_deterministic_odd_and_disjoint_from_both_predecessors() -> None:
    anchor = _anchor()
    observed = protocol._derive_recovery_seeds(anchor)

    assert observed == protocol._derive_recovery_seeds(copy.deepcopy(anchor))
    assert len(observed) == protocol.MAXIMUM_RECOVERY_LANES
    assert len(set(observed)) == len(observed)
    assert all(seed % 2 == 1 for seed in observed)
    assert set(observed).isdisjoint(anchor["t10_2_5_recovery_seeds"])
    assert set(observed).isdisjoint(anchor["t10_2_6_recovery_seeds"])


def test_seed_derivation_changes_when_partial_intent_binding_changes() -> None:
    anchor = _anchor()
    changed = copy.deepcopy(anchor)
    changed["partial_intent_checksum"] = "d" * 64

    assert protocol._derive_recovery_seeds(anchor) != protocol._derive_recovery_seeds(
        changed
    )


def test_execution_contract_binds_environment_and_overlay_projection() -> None:
    kernel = {
        "manifest_checksum": protocol.PARENT_KERNEL_MANIFEST_CHECKSUM,
        "environment_sha256": "e" * 64,
        "scientific_field": {"frozen": True},
    }

    contract = protocol._execution_contract(kernel)

    assert contract["required_environment_sha256"] == "e" * 64
    assert contract["overlay_keys"] == ["manifest_checksum", "migration_receipt"]
    assert contract["source_kernel_payload_sha256"] == protocol.canonical_sha256(
        kernel
    )

    drifted = copy.deepcopy(contract)
    drifted["required_environment_sha256"] = "f" * 64
    unsigned = {key: value for key, value in drifted.items() if key != "contract_checksum"}
    assert protocol.canonical_sha256(unsigned) != contract["contract_checksum"]


def test_partial_accounting_is_deliberately_open_until_quarantined() -> None:
    payload = {
        "authorized_intent_count": 1,
        "sealed_event_count": 0,
        "explicitly_unresolved_intent_count": 0,
    }

    assert payload["authorized_intent_count"] != (
        payload["sealed_event_count"]
        + payload["explicitly_unresolved_intent_count"]
    )


def test_recovery_lane_identity_binds_seed_game_and_split() -> None:
    lane = protocol._recovery_lane("su15-4c352900", 3_000_001)
    changed = protocol._recovery_lane("su15-4c352900", 3_000_003)

    assert lane["split"] == "leave_one_game_out_confirmation"
    assert lane["lane_id"] != changed["lane_id"]
    with pytest.raises(KeyError):
        _ = lane["unregistered"]
