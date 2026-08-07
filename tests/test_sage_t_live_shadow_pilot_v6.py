from __future__ import annotations

import json
from pathlib import Path

from theory.sage_t.calibration_gate_v8_6 import (
    _checksum,
    freeze_confirmation_manifest,
)
from theory.sage_t.live_shadow_pilot_v6 import (
    MultiPolicyMaterializedController,
    load_confirmation_manifest,
)


def test_multipolicy_controller_contains_all_challengers_and_selected_v2() -> None:
    controller = MultiPolicyMaterializedController(
        selected="tempered",
        caps={
            "maximum_programs": 8,
            "maximum_sequences": 8,
            "maximum_particles_per_decision": 4,
            "ordinary_horizon": 1,
        },
    )

    assert {
        "legacy",
        "tempered",
        "correlation_aware",
        "combined",
        "tempered_repair_v2",
    } == set(controller.controllers)
    assert controller.posterior is controller.controllers[
        "tempered_repair_v2"
    ].posterior
    assert all(
        item.effective_mode.value == "shadow"
        for item in controller.controllers.values()
    )


def test_confirmation_manifest_is_bound_to_exact_selection_report(
    tmp_path: Path,
) -> None:
    selection = {
        "manifest_checksum": "selection-manifest",
        "selected_challenger": "combined",
        "source_validation_authorized": False,
        "offline_repair_v2_confirmation": {"passed": True},
    }
    selection["report_checksum"] = _checksum(selection)
    report_path = tmp_path / "selection_report.json"
    report_path.write_text(json.dumps(selection), encoding="utf-8")
    manifest_path = tmp_path / "confirmation.json"
    frozen = freeze_confirmation_manifest(
        selection,
        output_path=manifest_path,
    )

    loaded, report = load_confirmation_manifest(
        manifest_path,
        selection_report_path=report_path,
    )

    assert loaded == frozen
    assert report == selection
    assert loaded["repair_policy"]["name"] == "combined_repair_v2"
