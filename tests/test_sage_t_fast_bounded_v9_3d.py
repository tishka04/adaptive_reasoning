from __future__ import annotations

from theory.sage_t import fast_bounded_v9_3d as fast


def test_fast_manifest_preserves_winning_prefixes_and_firewall() -> None:
    manifest = fast.load_manifest()

    assert manifest["controller_caps"]["maximum_sequences"] == 8
    assert manifest["controller_caps"]["maximum_particles_per_decision"] == 4
    assert manifest["fast_prefix_audit"]["correct_first_action"] == 9
    assert manifest["fast_prefix_audit"]["passed"] is True
    assert manifest["firewall"]["active_authority"] is False


def test_partial_result_is_behavioral_evidence_not_gate_authority() -> None:
    report = fast._load_partial_report()

    assert report["behavioral_evidence"] == {
        "bounded_levels": 3,
        "baseline_levels": 0,
        "game_over_delta": 0,
    }
    assert report["complete_gate_result"] is False
    assert report["t9_4_authorized"] is False
