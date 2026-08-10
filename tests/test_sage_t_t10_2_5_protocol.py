from __future__ import annotations

from typing import Any

import pytest

from theory.sage_t import t10_2_5_protocol as protocol


def _accounting(count: int) -> dict[str, Any]:
    return {
        "authorized_intent_count": count,
        "sealed_event_count": count,
        "explicitly_unresolved_intent_count": 0,
        "unknown_intent_count": 0,
        "posterior_update_count": count,
        "equation_holds": True,
    }


def _work(index: int, controller: str) -> dict[str, Any]:
    return {
        "lane": {"lane_id": "lane"},
        "reset_index": index,
        "controller": controller,
        "work_id": f"work-{index}",
    }


def _binding(
    index: int,
    controller: str,
    *,
    count: int,
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "work": _work(index, controller),
        "accounting": _accounting(count),
        "intent_count": count,
        "event_count": count,
        "posterior_update_count": count,
        "ordered_intents_sha256": "a" * 64,
        "ordered_events_sha256": "b" * 64,
        "ordered_event_ids_sha256": "c" * 64,
        "ordered_updates_sha256": "d" * 64,
        "report": report,
    }


def _snapshot() -> dict[str, Any]:
    complete = {"status": "COMPLETE", "report_checksum": "e" * 64}
    return {
        "completed_lanes": [
            {"lane_id": f"lane-{index}", "status": "COMPLETE", "resets": [1, 2, 3, 4]}
            for index in range(15)
        ],
        "open_lane": {
            "split": "leave_one_game_out_confirmation",
            "game_id": "su15-4c352900",
            "seed": 111,
            "lane_id": "lane",
        },
        "open_resets": [
            _binding(0, "capacity_matched_independent", count=16, report=complete),
            _binding(1, "learned", count=16, report=complete),
            _binding(2, "capacity_matched_independent", count=10, report=None),
            _binding(3, "learned", count=0, report=None),
        ],
    }


def test_recovery_policy_is_bounded_replay_free_and_worker_scoped() -> None:
    policy = protocol.recovery_policy()

    assert policy["orphaned_physical_actions_replayed"] is False
    assert policy["orphaned_lane_enters_model_fit"] is False
    assert policy["replacement_scope"] == "whole_confirmation_lane"
    assert policy["maximum_recovery_lanes"] == 3
    assert policy["recovery_maximum_actions"] == 768
    assert policy["watchdog_kill_scope"] == "reset_worker_process_tree_only"
    assert policy["collector_pid_may_be_killed_by_reset_watchdog"] is False
    assert policy["validation_and_ar25_authority_opened"] is False


def test_recovery_seeds_are_deterministic_distinct_and_odd() -> None:
    anchor = {"checkpoint_checksum": "a" * 64, "orphan_work_id": "work"}

    first = protocol._derive_recovery_seeds(anchor)
    second = protocol._derive_recovery_seeds(anchor)

    assert first == second
    assert len(first) == protocol.MAXIMUM_RECOVERY_LANES
    assert len(set(first)) == len(first)
    assert all(seed % 2 == 1 for seed in first)
    assert not set(first) & set(protocol._kernel_runtime.CONFIRMATION_SEEDS)


def test_orphan_snapshot_requires_two_complete_one_partial_one_empty() -> None:
    lane, orphan = protocol._require_orphan_snapshot(_snapshot())

    assert lane["game_id"] == "su15-4c352900"
    assert orphan["work"]["reset_index"] == 2
    assert orphan["event_count"] == 10


def test_orphan_snapshot_rejects_unsealed_partial() -> None:
    snapshot = _snapshot()
    snapshot["open_resets"][2]["accounting"]["sealed_event_count"] = 9

    with pytest.raises(protocol.JournalIntegrityError, match="fully sealed"):
        protocol._require_orphan_snapshot(snapshot)

