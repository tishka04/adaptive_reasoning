from __future__ import annotations

import copy

import pytest

from theory.sage_t import t10_2_8_protocol as protocol


def _lane(index: int, *, recovery: bool = False) -> dict[str, object]:
    physical_seed = 3_119_945 if recovery else 100 + index
    physical = {
        "split": (
            "leave_one_game_out_confirmation" if recovery else "discovery"
        ),
        "game_id": "su15-4c352900" if recovery else "bp35-0a0ad940",
        "seed": physical_seed,
        "lane_id": f"physical-{index}",
    }
    payload: dict[str, object] = {
        "lane": physical,
        "sealed_events": 10,
        "report_checksum": f"{index:064x}"[-64:],
    }
    if recovery:
        payload["logical_lane"] = {
            "split": "leave_one_game_out_confirmation",
            "game_id": "su15-4c352900",
            "seed": 111,
            "lane_id": "logical-orphan",
        }
        payload["physical_recovery_lane"] = physical
    return payload


def _collection() -> dict[str, object]:
    lanes = [_lane(index) for index in range(17)] + [_lane(17, recovery=True)]
    return {
        "accepted_lanes": lanes,
        "accepted_event_count": 180,
    }


def test_policy_is_strictly_offline_and_stops_before_fit() -> None:
    policy = protocol.qa_policy()

    assert policy["environment_calls_authorized"] == 0
    assert policy["physical_actions_authorized"] == 0
    assert policy["physical_replay_authorized"] is False
    assert policy["model_fit_authorized"] is False
    assert policy["source_train_authorized"] is False
    assert policy["lineage_validation_required_before_qa"] is True
    assert policy["qa_failure_stops_before_fit"] is True
    assert policy["source_validation_authorized"] is False
    assert policy["ar25_authorized"] is False


def test_lineage_registry_contains_seventeen_parent_and_one_recovery_lane() -> None:
    predecessor = {"manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM}

    registry = protocol._lineage_registry(_collection(), predecessor)

    assert len(registry) == 18
    assert sum(item["lineage"] == "t10_2_2_parent" for item in registry) == 17
    assert sum(item["lineage"] == "t10_2_7_recovery" for item in registry) == 1
    replacement = next(item for item in registry if item["lineage"].endswith("recovery"))
    assert replacement["physical_lane"]["seed"] == 3_119_945
    assert replacement["logical_lane"]["seed"] == 111
    assert (
        replacement["provenance_manifest_checksum"]
        == protocol.PREDECESSOR_MANIFEST_CHECKSUM
    )


def test_lineage_registry_rejects_more_than_one_recovery_lane() -> None:
    collection = _collection()
    collection["accepted_lanes"] = copy.deepcopy(collection["accepted_lanes"])
    collection["accepted_lanes"][0] = _lane(0, recovery=True)
    predecessor = {"manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM}

    with pytest.raises(protocol.JournalIntegrityError, match="exactly one recovery"):
        protocol._lineage_registry(collection, predecessor)


def test_lineage_registry_rejects_event_count_drift() -> None:
    collection = _collection()
    collection["accepted_event_count"] = 181

    with pytest.raises(protocol.JournalIntegrityError, match="event accounting"):
        protocol._lineage_registry(
            collection,
            {"manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM},
        )


def test_artifact_contract_keeps_results_outside_predecessor_root() -> None:
    contract = protocol.artifact_contract()

    assert contract["predecessor_root"] != contract["output_root"]
    assert contract["terminal_report"] == "t10_2_8_report.json"
    assert contract["qa_report"] == "qa_report.json"
