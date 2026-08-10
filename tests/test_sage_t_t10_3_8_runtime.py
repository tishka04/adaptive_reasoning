from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_8_protocol as protocol
from theory.sage_t import t10_3_8_runtime as runtime
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry
from theory.sage_t.goal_directed_v10_3_7 import StableFreshPathSageTController


def _parent_report() -> dict:
    checks = {
        "all_conditions_present": True,
        "diagnostic_not_training": True,
        "fresh_grounding_only": True,
        "historical_grounded_actions_loaded": False,
        "intent_accounting": True,
        "latency_not_a_gate": True,
        "level_each_core_game": True,
        "posterior_updated_each_event": True,
        "winning_action_from_sage_t": True,
        "zero_controller_errors": True,
        "zero_illegal_actions": True,
        "zero_physical_replay": True,
    }
    core = {
        "format_version": "sage-t10.3.7-canonical-witness-report-v1",
        "manifest_checksum": "parent-manifest",
        "canonical_descriptors": {},
        "metrics": {
            "actions": 38,
            "levels": {"lp85-305b61c3": 1, "su15-4c352900": 2},
        },
        "checks": checks,
        "receipt_checksums": ["a", "b", "c", "d"],
        "passed": False,
        "verdict": "CANONICAL_WITNESS_MISS",
    }
    return runtime._signed(core, "report_checksum")


def test_adjudication_normalizes_only_the_negative_assertion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))
    parent = _parent_report()
    monkeypatch.setattr(runtime, "_parent_report", lambda root: parent)
    audit = runtime._signed(
        {
            "format_version": "audit",
            "status": "PASS_T10_3_8_OFFLINE_AUDIT",
        },
        "audit_checksum",
    )
    protocol.write_json_once(tmp_path / "out" / runtime.AUDIT_FILENAME, audit)
    manifest = {"manifest_checksum": "new-manifest"}

    result = runtime.adjudicate(tmp_path, manifest)

    assert result["passed"] is True
    assert result["checks"]["historical_grounded_actions_absent"] is True
    assert "historical_grounded_actions_loaded" not in result["checks"]
    assert result["metrics"]["new_physical_actions"] == 0
    assert result["physical_actions_replayed"] == 0
    assert result["parent_events_used_for_training"] == 0


def test_any_other_false_check_prevents_adjudication(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))
    parent = _parent_report()
    parent["checks"]["zero_illegal_actions"] = False
    parent.pop("report_checksum")
    parent = runtime._signed(parent, "report_checksum")
    monkeypatch.setattr(runtime, "_parent_report", lambda root: parent)
    audit = runtime._signed(
        {"format_version": "audit", "status": "PASS_T10_3_8_OFFLINE_AUDIT"},
        "audit_checksum",
    )
    protocol.write_json_once(tmp_path / "out" / runtime.AUDIT_FILENAME, audit)

    try:
        runtime.adjudicate(tmp_path, {"manifest_checksum": "new-manifest"})
    except protocol.ScientificGateMiss:
        pass
    else:
        raise AssertionError("an unrelated false gate must remain fail-closed")


def test_blank_discovery_uses_stable_controller_without_witness_prior() -> None:
    work = protocol.work_specs("discover-core")[0]
    _, goal = runtime._controller_pair(
        work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )

    assert isinstance(goal, StableFreshPathSageTController)
    assert goal.witness_schema is None
    assert goal.phase == "discovery"


def test_delegated_reports_are_versioned_as_t10_3_8() -> None:
    transformed = runtime._replace_version(
        {
            "format_version": "sage-t10.3.6-discover-core-report-v1",
            "verdict": "PASS_T10_3_6_BLANK_CORE_DISCOVERY",
        }
    )
    assert transformed["format_version"] == "sage-t10.3.8-discover-core-report-v1"
    assert transformed["verdict"] == "PASS_T10_3_8_BLANK_CORE_DISCOVERY"
