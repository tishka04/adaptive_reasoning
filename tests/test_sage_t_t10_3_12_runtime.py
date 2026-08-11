from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12_protocol as protocol
from theory.sage_t import t10_3_12_runtime as runtime
from theory.sage_t.relational_program_v10_3_12 import signed


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-manifest",
        "offline_matrix": {
            "maximum_artifact_bytes": 10 * 1024 * 1024,
            "maximum_candidate_inspections": 12_288,
            "maximum_wall_seconds": 600,
        },
        "firewall": {},
    }


def _write_parent_gate(root: Path) -> None:
    payload = signed(
        {
            "format_version": "synthetic",
            "manifest_checksum": "synthetic-manifest",
            "passed": True,
        },
        "receipt_checksum",
    )
    protocol.write_json_once(
        root / protocol.DEFAULT_OUTPUT_DIR / runtime.PARENT_AUDIT_FILENAME,
        payload,
    )


def test_preflight_has_exactly_twelve_zero_action_cases(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))
    _write_parent_gate(tmp_path)
    result = runtime.preflight(tmp_path, _manifest())
    assert result["passed"] is True
    assert len(result["cases"]) == 12
    assert len(result["prefix_alignment"]) == 6
    assert result["physical_actions"] == 0


def test_offline_chain_compiles_and_passes_without_physical_actions(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path.cwd()
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_parent_gate(repo_root)
    manifest = _manifest()
    preflight = runtime.preflight(repo_root, manifest)
    inventory = runtime.materialize_offline(repo_root, manifest)
    registry = runtime.compile_candidates(repo_root, manifest)
    report = runtime.evaluate_offline(repo_root, manifest)
    assert preflight["passed"] is True
    assert inventory["passed"] is True
    assert registry["local_support_total"] == 0
    assert report["passed"] is True
    assert report["physical_actions"] == 0
    assert report["metrics"]["correct_by_arm"]["factorized_relational_source"] == 96


def test_terminal_report_fails_closed_before_offline_gates(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", Path("out"))
    report = runtime.terminal_report(
        tmp_path, {"manifest_checksum": "synthetic-manifest", "firewall": {}}
    )
    assert report["verdict"] == "INVALID_PROVENANCE"
    assert report["physical_actions_replayed"] == 0
