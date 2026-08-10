from __future__ import annotations

from theory.sage_t import t10_3_7_runtime as runtime
from theory.sage_t import t10_3_7_protocol as protocol
from theory.sage_t.goal_directed_v10_3_2 import ProgressProgramRegistry
from theory.sage_t.goal_directed_v10_3_7 import StableFreshPathSageTController


def test_expected_su15_diagnosis_distinguishes_waypoints_nine_and_ten() -> None:
    hashes = runtime.EXPECTED_SU15_WAYPOINT_CHECKSUMS
    assert len(hashes) == 10
    assert len(set(hashes)) == 10
    assert hashes[8] != hashes[9]


def test_synthetic_stable_path_reacquires_all_ten() -> None:
    result = runtime._synthetic_stable_path()

    assert result["selected"] == result["expected"]
    assert result["reacquisitions"] == 10
    assert result["grounding_misses"] == 0
    assert result["plan_persisted"] is False


def test_preflight_keeps_latency_out_of_scientific_checks(tmp_path) -> None:
    manifest = {
        "manifest_checksum": "synthetic-t10-3-7",
        "functional_contract": {"latency_is_telemetry_only": True},
    }
    result = runtime.preflight(tmp_path, manifest)

    assert result["status"] == "PASS_T10_3_7_PREFLIGHT"
    assert result["checks"]["ten_waypoints_in_order"] is True
    assert not any("p95" in key for key in result["checks"])


def test_witness_runtime_uses_stable_controller() -> None:
    work = next(
        item
        for item in protocol.work_specs("witness-core")
        if item.game_id == "su15-4c352900"
    )
    _, goal = runtime._controller_pair(
        work,
        ProgressProgramRegistry(),
        registry_checksum=None,
    )

    assert isinstance(goal, StableFreshPathSageTController)
    assert goal.witness_schema == "path_successor"
    assert goal.witness_horizon == 10


def test_delegated_artifact_versions_are_rewritten_to_t10_3_7() -> None:
    payload = {
        "format_version": "sage-t10.3.6-discover-core-report-v1",
        "verdict": "PASS_T10_3_6_BLANK_CORE_DISCOVERY",
    }

    transformed = runtime._replace_version(payload)

    assert transformed["format_version"] == "sage-t10.3.7-discover-core-report-v1"
    assert transformed["verdict"] == "PASS_T10_3_7_BLANK_CORE_DISCOVERY"
