from __future__ import annotations

from types import SimpleNamespace

import pytest

from theory.sage_t import t10_2_3_protocol as protocol


def test_continuation_policy_is_replay_free_and_authority_closed() -> None:
    policy = protocol.continuation_policy()

    assert policy["parent_scientific_kernel_unchanged"] is True
    assert policy["completed_physical_actions_replayed"] is False
    assert policy["cache_authority"] == "none"
    assert policy["cache_built_before_reset_watchdog"] is True
    assert policy["cache_build_time_charged_to_lane_and_collection"] is True
    assert policy["holdout_evidence_enters_cache"] is False
    assert policy["validation_and_ar25_authority_opened"] is False


def test_lane_fingerprint_binds_every_reset_and_event_digest() -> None:
    resets = (
        SimpleNamespace(
            work=SimpleNamespace(work_id="work-a"),
            report_checksum="a" * 64,
            event_ids_sha256="b" * 64,
            issued_intents=3,
            sealed_events=3,
            unresolved_intents=0,
            status="COMPLETE",
        ),
    )
    lane = SimpleNamespace(
        lane_id="c" * 64,
        to_dict=lambda: {"lane_id": "c" * 64},
    )
    report = SimpleNamespace(
        lane=lane,
        status="COMPLETE",
        report_checksum="d" * 64,
        issued_intents=3,
        sealed_events=3,
        unresolved_intents=0,
        resets=resets,
    )

    fingerprint = protocol._lane_fingerprint(report)

    assert fingerprint["report_checksum"] == "d" * 64
    assert fingerprint["resets"][0]["work_id"] == "work-a"
    assert fingerprint["resets"][0]["event_ids_sha256"] == "b" * 64


def _receipt(lanes: list[dict[str, object]]) -> dict[str, object]:
    return protocol.signed_payload(
        {
            "format_version": protocol.MIGRATION_FORMAT_VERSION,
            "parent_t10_2_2_manifest_checksum": "parent",
            "parent_kernel_manifest_checksum": "kernel",
            "initial_checkpoint": {"revision": 1, "checksum": "checkpoint"},
            "initial_cursor": {
                "revision": 2,
                "checksum": "cursor",
                "open_lane_id": "next",
            },
            "completed_lane_count": len(lanes),
            "completed_reset_count": len(lanes) * 4,
            "completed_lanes": lanes,
            "completed_lanes_sha256": protocol.canonical_sha256(lanes),
            "initial_accounting": {
                "authorized_intent_count": 10,
                "sealed_event_count": 10,
                "explicitly_unresolved_intent_count": 0,
                "unknown_intent_count": 0,
                "posterior_update_count": 10,
                "equation_holds": True,
            },
            "initial_discovery_evidence": {"count": 4},
            "next_lane": {"lane_id": "next"},
            "replay_authorized": False,
        },
        checksum_key="receipt_checksum",
    )


def test_live_migration_accepts_append_only_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = [{"lane_id": "lane-1", "report_checksum": "a" * 64}]
    receipt = _receipt(frozen)
    monkeypatch.setattr(
        protocol,
        "_load_parent",
        lambda root: (
            {"manifest_checksum": "parent"},
            {"manifest_checksum": "kernel"},
            root / "kernel.json",
            root / "artifacts",
        ),
    )
    monkeypatch.setattr(
        protocol,
        "_journal_snapshot",
        lambda **kwargs: (
            object(),
            [*frozen, {"lane_id": "lane-2", "report_checksum": "b" * 64}],
            {
                "authorized_intent_count": 12,
                "sealed_event_count": 12,
                "explicitly_unresolved_intent_count": 0,
                "unknown_intent_count": 0,
                "posterior_update_count": 12,
                "equation_holds": True,
            },
            {"count": 5},
        ),
    )

    result = protocol.verify_migration_receipt_live(receipt, repo_root=".")

    assert result["migration_verified"] is True
    assert result["frozen_completed_lanes"] == 1
    assert result["current_completed_lanes"] == 2
    assert result["replay_authorized"] is False


def test_live_migration_rejects_changed_frozen_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = [{"lane_id": "lane-1", "report_checksum": "a" * 64}]
    receipt = _receipt(frozen)
    monkeypatch.setattr(
        protocol,
        "_load_parent",
        lambda root: (
            {"manifest_checksum": "parent"},
            {"manifest_checksum": "kernel"},
            root / "kernel.json",
            root / "artifacts",
        ),
    )
    monkeypatch.setattr(
        protocol,
        "_journal_snapshot",
        lambda **kwargs: (
            object(),
            [{"lane_id": "lane-1", "report_checksum": "changed"}],
            {
                "authorized_intent_count": 10,
                "sealed_event_count": 10,
                "explicitly_unresolved_intent_count": 0,
                "unknown_intent_count": 0,
                "posterior_update_count": 10,
                "equation_holds": True,
            },
            {"count": 4},
        ),
    )

    with pytest.raises(protocol.JournalIntegrityError, match="frozen"):
        protocol.verify_migration_receipt_live(receipt, repo_root=".")

