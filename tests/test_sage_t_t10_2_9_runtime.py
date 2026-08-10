from __future__ import annotations

from typing import Any

from theory.sage_t import t10_2_9_protocol as protocol
from theory.sage_t import t10_2_9_runtime as runtime


def _manifest() -> dict[str, Any]:
    return {
        "manifest_checksum": "9" * 64,
        "qa_gate": {},
        "handoff_receipt": {
            "receipt_checksum": "8" * 64,
            "predecessor_terminal_checksum": protocol.PREDECESSOR_TERMINAL_CHECKSUM,
            "adapter_failure": {"scientific_qa_evaluated": False},
            "source_seed_registry": {
                "discovery": [101, 102, 103],
                "leave_one_game_out_confirmation": [111, 112, 113],
                "recovery_confirmation": [3_119_945],
            },
        },
    }


def test_durable_seed_binding_sets_and_restores_both_registries() -> None:
    original_discovery = runtime._science.DISCOVERY_SEEDS
    original_confirmation = runtime._science.CONFIRMATION_SEEDS
    with runtime._durable_seed_registry_binding([3_119_945]):
        assert runtime._science.DISCOVERY_SEEDS == (101, 102, 103)
        assert runtime._science.CONFIRMATION_SEEDS == (
            111,
            112,
            113,
            3_119_945,
        )
    assert runtime._science.DISCOVERY_SEEDS == original_discovery
    assert runtime._science.CONFIRMATION_SEEDS == original_confirmation


def test_durable_seed_binding_restores_after_exception() -> None:
    original_discovery = runtime._science.DISCOVERY_SEEDS
    original_confirmation = runtime._science.CONFIRMATION_SEEDS
    try:
        with runtime._durable_seed_registry_binding([3_119_945]):
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    assert runtime._science.DISCOVERY_SEEDS == original_discovery
    assert runtime._science.CONFIRMATION_SEEDS == original_confirmation


def test_not_evaluated_qa_is_fail_closed() -> None:
    report = runtime._not_evaluated_qa(
        manifest=_manifest(),
        lineage_audit={"audit_checksum": "7" * 64},
    )
    assert report["status"] == "NOT_EVALUATED_LINEAGE_FAILURE"
    assert report["metrics"] == {}
    assert report["behavior_diagnostics"] == {}
    assert report["fit_authorized"] is False
    assert report["firewall"]["environment_calls"] == 0


def test_cli_parser_exposes_only_status_and_compile() -> None:
    parser = runtime.build_parser()
    assert parser.parse_args(["status"]).phase == "status"
    assert parser.parse_args(["compile"]).phase == "compile"
