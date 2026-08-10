from __future__ import annotations

from pathlib import Path

import pytest

from theory.sage_t import t10_2_4_protocol as protocol


def test_policy_registers_dual_exact_cache_and_closed_authority() -> None:
    policy = protocol.continuation_policy()

    assert policy["new_cache_kinds"] == ["gauge", "factorized"]
    assert policy["predecessor_cache_root_read_only"] is True
    assert policy["fit_interception_requires_exact_events"] is True
    assert policy["fit_interception_requires_exact_candidates"] is True
    assert policy["fit_interception_requires_exact_posterior_type"] is True
    assert policy["nonmatching_fits_delegated"] is True
    assert policy["completed_physical_actions_replayed"] is False
    assert policy["validation_and_ar25_authority_opened"] is False


def _receipt(lanes: list[dict[str, object]], caches: list[dict[str, object]]) -> dict[str, object]:
    return protocol.signed_payload(
        {
            "format_version": protocol.MIGRATION_FORMAT_VERSION,
            "predecessor_t10_2_3_manifest_checksum": "predecessor",
            "parent_kernel_manifest_checksum": "kernel",
            "initial_checkpoint": {"revision": 188, "checksum": "checkpoint"},
            "initial_cursor": {
                "revision": 190,
                "checksum": "cursor",
                "open_lane_id": "next",
            },
            "completed_lane_count": len(lanes),
            "completed_reset_count": len(lanes) * 4,
            "completed_lanes": lanes,
            "completed_lanes_sha256": protocol.canonical_sha256(lanes),
            "initial_accounting": {
                "authorized_intent_count": 998,
                "sealed_event_count": 998,
                "posterior_update_count": 998,
                "explicitly_unresolved_intent_count": 0,
                "unknown_intent_count": 0,
                "equation_holds": True,
            },
            "initial_discovery_evidence": {"count": 650},
            "adopted_predecessor_caches": caches,
            "adopted_predecessor_caches_sha256": protocol.canonical_sha256(caches),
            "next_lane": {"lane_id": "next"},
            "replay_authorized": False,
        },
        checksum_key="receipt_checksum",
    )


def _patch_live(
    monkeypatch: pytest.MonkeyPatch,
    lanes: list[dict[str, object]],
    caches: list[dict[str, object]],
) -> None:
    monkeypatch.setattr(
        protocol,
        "_load_predecessor",
        lambda root: {"manifest_checksum": "predecessor"},
    )
    monkeypatch.setattr(
        protocol,
        "_parent_execution",
        lambda root: (
            {},
            {"manifest_checksum": "kernel"},
            Path(root) / "kernel.json",
            Path("artifacts"),
        ),
    )
    monkeypatch.setattr(
        protocol,
        "_snapshot",
        lambda **kwargs: (
            lanes,
            {
                "authorized_intent_count": 1_010,
                "sealed_event_count": 1_010,
                "posterior_update_count": 1_010,
                "explicitly_unresolved_intent_count": 0,
                "unknown_intent_count": 0,
                "equation_holds": True,
            },
            {"count": 700},
        ),
    )
    monkeypatch.setattr(
        protocol, "_predecessor_cache_fingerprints", lambda root: caches
    )


def test_live_migration_accepts_append_only_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = [{"lane_id": "lane-1", "report_checksum": "a" * 64}]
    caches = [{"cache_key": "cache", "cache_checksum": "b" * 64}]
    receipt = _receipt(frozen, caches)
    _patch_live(
        monkeypatch,
        [*frozen, {"lane_id": "lane-2", "report_checksum": "c" * 64}],
        caches,
    )

    result = protocol.verify_migration_receipt_live(receipt, repo_root=".")

    assert result["migration_verified"] is True
    assert result["current_completed_lanes"] == 2
    assert result["adopted_predecessor_cache_count"] == 1
    assert result["replay_authorized"] is False


def test_live_migration_rejects_adopted_cache_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = [{"lane_id": "lane-1", "report_checksum": "a" * 64}]
    caches = [{"cache_key": "cache", "cache_checksum": "b" * 64}]
    receipt = _receipt(frozen, caches)
    changed = [{"cache_key": "cache", "cache_checksum": "changed"}]
    _patch_live(monkeypatch, frozen, changed)

    with pytest.raises(protocol.JournalIntegrityError, match="cache changed"):
        protocol.verify_migration_receipt_live(receipt, repo_root=".")

