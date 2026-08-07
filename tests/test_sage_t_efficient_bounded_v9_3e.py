from __future__ import annotations

from theory.sage_t import efficient_bounded_v9_3e as efficient


def test_efficient_manifest_preserves_all_winning_prefixes() -> None:
    manifest = efficient.load_manifest()

    assert manifest["controller_caps"]["maximum_programs"] == 16
    assert manifest["controller_caps"]["maximum_sequences"] == 6
    assert manifest["controller_caps"]["maximum_particles_per_decision"] == 2
    assert manifest["winning_prefix_audit"]["exact_sequence_generated"] == 9
    assert manifest["winning_prefix_audit"]["correct_first_action"] == 9
    assert manifest["winning_prefix_audit"]["passed"] is True
    assert manifest["firewall"]["active_authority"] is False


def test_t9_3d_parent_failed_only_latency_after_three_safe_levels() -> None:
    report = efficient._load_parent_report()

    assert report["metrics"]["levels_completed"] == 3
    assert report["metrics"]["game_over_delta"] == 0
    assert report["checks"]["observation_p95"] is False
    assert sum(not value for value in report["checks"].values()) == 1
    assert report["t9_4_authorized"] is False
