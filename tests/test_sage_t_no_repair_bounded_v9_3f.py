from __future__ import annotations

from theory.sage_t import no_repair_bounded_v9_3f as no_repair


def test_no_repair_manifest_preserves_winning_prefixes() -> None:
    manifest = no_repair.load_manifest()

    assert manifest["controller_caps"]["maximum_repair_contexts"] == 0
    assert manifest["winning_prefix_audit"]["exact_sequence_generated"] == 9
    assert manifest["winning_prefix_audit"]["correct_first_action"] == 9
    assert manifest["winning_prefix_audit"]["passed"] is True
    assert manifest["firewall"]["active_authority"] is False


def test_no_repair_controller_skips_every_repair_context() -> None:
    controller = no_repair.build_controller(no_repair.load_manifest())

    assert controller.posterior.maximum_repair_contexts == 0
    assert controller.posterior.performance_snapshot()["maximum_repair_contexts"] == 0
