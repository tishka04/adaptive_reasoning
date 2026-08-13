from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage_t.causal import option_contract_protocol as protocol_module
from theory.sage_t.causal.option_contract_cli import build_parser
from theory.sage_t.causal.option_contract_experiment import (
    compile_option_contract,
    option_contract_status,
)
from theory.sage_t.causal.option_contract_protocol import (
    OptionContractProtocol,
    freeze_option_contract,
    load_option_contract_manifest,
)
from theory.sage_t.causal.option_contracts import ContractedCausalOption


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _parent() -> Path:
    return (
        _repo()
        / "training"
        / "sage_t"
        / "option_applicability_t12_4a_4b_bp35"
    )


def _freeze(monkeypatch, tmp_path: Path) -> tuple[Path, dict]:
    monkeypatch.setattr(
        protocol_module,
        "_git_state",
        lambda root: {"commit": "c" * 40, "dirty": False, "dirty_entries": 0},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = freeze_option_contract(
        output_path=manifest_path,
        parent_manifest_path=_parent() / "manifest.json",
        parent_receipt_path=_parent() / "audit" / "applicability_receipt.json",
        root=_repo(),
    )
    return manifest_path, manifest


def test_protocol_is_offline_bounded_and_cli_has_no_active_phase() -> None:
    protocol = OptionContractProtocol()
    assert protocol.maximum_sdk_calls == 0
    assert protocol.maximum_child_particles == 24
    assert protocol.maximum_artifact_bytes_per_run == 3 * 1024**3
    assert protocol.minimum_initiation_particles == 4
    assert protocol.maximum_initiation_particles == 6
    parser = build_parser()
    assert parser.parse_args(
        [
            "freeze",
            "--parent-manifest",
            "manifest.json",
            "--parent-receipt",
            "receipt.json",
        ]
    ).phase == "freeze"
    assert parser.parse_args(["compile"]).phase == "compile"
    assert parser.parse_args(["status"]).phase == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["activate"])


def test_freeze_binds_passed_initiation_dynamics_diagnosis_and_closes_authority(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path, manifest = _freeze(monkeypatch, tmp_path)
    loaded = load_option_contract_manifest(manifest_path, root=_repo())
    assert loaded["manifest_checksum"] == manifest["manifest_checksum"]
    assert loaded["parent"]["receipt"]["status"] == (
        "PASS_T12_4A_4B_APPLICABILITY_AUDIT_GATE"
    )
    assert loaded["firewall"]["option_contract_compile_authorized"] is True
    assert loaded["firewall"]["option_control_authorized"] is False
    assert loaded["firewall"]["neural_training_authorized"] is False
    assert loaded["firewall"]["t12_4a_4d_target_regrounding_freeze_authorized"] is False
    assert loaded["storage"]["maximum_sdk_calls"] == 0


def test_real_sealed_audit_compiles_sparse_guarded_posterior_offline(
    monkeypatch,
    tmp_path,
) -> None:
    manifest_path, _ = _freeze(monkeypatch, tmp_path)
    output = tmp_path / "contract"
    receipt = compile_option_contract(
        manifest_path=manifest_path,
        output_dir=output,
    )
    assert receipt["passed"] is True
    assert receipt["status"] == "PASS_T12_4A_4C_OPTION_CONTRACT_GATE"
    metrics = receipt["metrics"]
    assert metrics["initiation_particle_count"] == 6
    assert metrics["effect_contract_count"] == 5
    assert metrics["effect_atom_count"] == 10
    assert metrics["compiled_program_count"] == 24
    assert metrics["posterior_particle_count"] == 24
    assert metrics["sdk_calls_used"] == 0
    assert metrics["successful_applicable_mass_minimum"] >= 0.8
    assert metrics["failed_applicable_mass_maximum"] == pytest.approx(0.0)
    assert metrics["maximum_parent_mass_error"] < 1e-12
    assert all(metrics["checks"].values())

    registry_payload = json.loads(
        (output / "contracted_option_registry.json").read_text()
    )
    registry = ContractedCausalOption.from_dict(registry_payload)
    assert len(registry.initiation_specs) == 6
    assert len(registry.effect_contracts) == 5
    contract_text = json.dumps(
        {
            "effects": [item.to_dict() for item in registry.effect_contracts],
            "guards": [item.to_dict() for item in registry.initiation_specs],
        }
    ).lower()
    assert "exact_hash" not in contract_text
    assert "levels_completed" not in contract_text
    assert "entity_id" not in contract_text

    status = option_contract_status(
        manifest_path=manifest_path,
        receipt_path=output / "option_contract_receipt.json",
    )
    assert status["next_phase_authorized"] is True
    assert status["firewall"][
        "t12_4a_4d_target_regrounding_freeze_authorized"
    ] is True
    assert status["firewall"]["option_control_authorized"] is False

