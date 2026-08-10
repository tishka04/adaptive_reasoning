from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from theory.sage_t import t10_2_6_protocol as protocol


def _predecessor() -> dict[str, Any]:
    seeds = [1650445, 1412747, 1438133]
    return {
        "manifest_checksum": protocol.PREDECESSOR_MANIFEST_CHECKSUM,
        "migration_receipt": {
            "recovery_seeds": seeds,
            "recovery_lanes": [
                protocol._predecessor_protocol._recovery_lane(
                    "su15-4c352900", seed
                )
                for seed in seeds
            ],
        },
    }


def _zero_failure() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    predecessor = _predecessor()
    lanes = []
    attempted = []
    for index, lane in enumerate(predecessor["migration_receipt"]["recovery_lanes"]):
        work = {
            "lane": lane,
            "reset_index": 0,
            "controller": "capacity_matched_independent",
            "work_id": f"work-{index}",
        }
        reset = {
            "work": work,
            "status": "ABORTED",
            "stop_reason": "worker_exited",
            "issued_intents": 0,
            "sealed_events": 0,
            "unresolved_intents": 0,
            "posterior_updates": 0,
            "report_checksum": f"{index + 1}" * 64,
        }
        lane_report = {
            "lane": lane,
            "status": "ABORTED",
            "resets": [reset],
            "report_checksum": f"{index + 4}" * 64,
        }
        lanes.append(lane_report)
        attempted.append(
            {
                "lane": lane,
                "lane_id": lane["lane_id"],
                "status": "ABORTED",
                "resets": [],
            }
        )
    accounting = {
        "authorized_intent_count": 0,
        "sealed_event_count": 0,
        "explicitly_unresolved_intent_count": 0,
        "unknown_intent_count": 0,
        "posterior_update_count": 0,
        "equation_holds": True,
    }
    report = {
        "report_checksum": protocol.PREDECESSOR_FAILURE_REPORT_CHECKSUM,
        "status": "FAIL_T10_2_5_RECOVERY",
        "manifest_checksum": predecessor["manifest_checksum"],
        "accepted_lane": None,
        "attempted_lane_count": 3,
        "attempted_lanes": attempted,
        "accounting": accounting,
        "physical_steps_replayed": 0,
        "orphan_events_replayed": 0,
    }
    checkpoint = {"lane_reports": lanes}
    return report, checkpoint, predecessor


def test_policy_is_append_only_spawn_scoped_and_fail_closed() -> None:
    policy = protocol.recovery_policy()

    assert policy["parent_collection_journal_read_only"] is True
    assert policy["predecessor_t10_2_5_failure_immutable"] is True
    assert policy["predecessor_failed_attempts_had_zero_actions"] is True
    assert policy["spawn_child_registers_recovery_seeds_before_work_decode"] is True
    assert policy["recovery_maximum_actions"] == 768
    assert policy["collect_cli_nonzero_on_failed_gate"] is True
    assert policy["validation_and_ar25_authority_opened"] is False


def test_new_seeds_are_deterministic_odd_and_disjoint_from_t10_2_5() -> None:
    anchor = {
        "parent_checkpoint_checksum": "a" * 64,
        "predecessor_failure_report_checksum": "b" * 64,
        "predecessor_recovery_seeds": [1650445, 1412747, 1438133],
    }

    first = protocol._derive_recovery_seeds(anchor)
    second = protocol._derive_recovery_seeds(anchor)

    assert first == second
    assert len(first) == protocol.MAXIMUM_RECOVERY_LANES
    assert len(set(first)) == len(first)
    assert all(seed % 2 == 1 and seed >= 2_000_001 for seed in first)
    assert not set(first) & set(anchor["predecessor_recovery_seeds"])
    assert not set(first) & set(protocol._kernel_runtime.CONFIRMATION_SEEDS)


def test_predecessor_failure_requires_three_zero_action_spawn_exits() -> None:
    report, checkpoint, predecessor = _zero_failure()

    attempts = protocol._validate_zero_action_failure(
        report=report, checkpoint=checkpoint, predecessor=predecessor
    )

    assert len(attempts) == 3
    assert all(item["stop_reason"] == "worker_exited" for item in attempts)
    assert all(item["issued_intents"] == 0 for item in attempts)


def test_predecessor_failure_rejects_any_physical_action() -> None:
    report, checkpoint, predecessor = _zero_failure()
    broken = deepcopy(report)
    broken["accounting"]["authorized_intent_count"] = 1

    with pytest.raises(protocol.JournalIntegrityError, match="pre-action"):
        protocol._validate_zero_action_failure(
            report=broken, checkpoint=checkpoint, predecessor=predecessor
        )


def test_predecessor_failure_rejects_non_spawn_stop_reason() -> None:
    report, checkpoint, predecessor = _zero_failure()
    checkpoint["lane_reports"][0]["resets"][0]["stop_reason"] = "worker_exception"

    with pytest.raises(protocol.JournalIntegrityError, match="spawn exit"):
        protocol._validate_zero_action_failure(
            report=report, checkpoint=checkpoint, predecessor=predecessor
        )
