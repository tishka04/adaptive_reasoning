from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12c_protocol as protocol
from theory.sage_t import t10_3_12c_runtime as runtime
from theory.sage_t.cross_game_transfer_v10_3_12c import signed


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-t10-3-12c",
        "matrix": {
            "maximum_artifact_bytes": 20 * 1024 * 1024,
        },
        "gates": {
            "minimum_factorized_applicable_games": 3,
            "minimum_factorized_success_games": 3,
            "minimum_factorized_success_rate_on_applicable": 2 / 3,
            "minimum_paired_ablation_advantage": 2,
            "maximum_reverse_paired_ablation_wins": 0,
            "generic_equal_success_action_ratio_maximum": 0.75,
            "minimum_supported_factors": 1,
        },
    }


def _write_gate(output: Path, filename: str, checksum_field: str) -> None:
    protocol.write_json_once(
        output / filename,
        signed(
            {
                "format_version": "synthetic-gate",
                "manifest_checksum": "synthetic-t10-3-12c",
                "passed": True,
            },
            checksum_field,
        ),
    )


def test_preflight_and_target_audit_execute_zero_actions(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_gate(output, runtime.PARENT_AUDIT_FILENAME, "audit_checksum")
    preflight = runtime.preflight(Path.cwd(), _manifest())
    inventory = runtime.audit_targets(Path.cwd(), _manifest())
    assert preflight["passed"] is True
    assert len(preflight["cases"]) == 14
    assert inventory["passed"] is True
    assert len(inventory["targets"]) == 9
    assert inventory["selection_was_outcome_independent"] is True
    assert inventory["physical_actions"] == 0


def test_transfer_registry_is_compiled_before_target_outcomes(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_gate(output, runtime.TARGET_AUDIT_FILENAME, "inventory_checksum")
    payload = runtime.compile_transfer(Path.cwd(), _manifest())
    assert payload["compiled_before_target_outcomes"] is True
    assert payload["target_schema_inventory_used_for_compilation"] is False
    assert payload["historical_support_imported"] == 0
    assert payload["grounded_arguments_imported"] is False
    assert len(payload["programs"]) == 12


def test_adjudication_names_supported_factors_without_promotion(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    success_games = set(protocol.TARGET_GAMES[:3])
    receipts = []
    for work in protocol.work_specs("active-transfer"):
        succeeds = work.game_id in success_games and work.arm == "factorized_source"
        generic_success = work.game_id == protocol.TARGET_GAMES[0] and work.arm == "generic_source_free"
        applicable = work.arm == "factorized_source" and work.game_id in success_games
        row = signed(
            {
                "format_version": "synthetic-receipt",
                "manifest_checksum": "synthetic-t10-3-12c",
                **work.as_dict(),
                "work_id": work.work_id,
                "complete": True,
                "level_delta": int(succeeds or generic_success),
                "sealed_events": 2 if succeeds else (4 if generic_success else 0),
                "sage_t_option_actions": 2 if succeeds else (4 if generic_success else 0),
                "recognized_contexts": ["repeat_context"] if applicable else [],
                "game_over_actions": 0,
                "illegal_actions": 0,
                "legacy_fallback_actions": 0,
                "errors": [],
                "physical_actions_replayed": 0,
            },
            "receipt_checksum",
        )
        path = output / "journal" / "branches" / work.work_id / "receipt.json"
        protocol.write_json_once(path, row)
        receipts.append(row)
    active = signed(
        {
            "format_version": "synthetic-active",
            "manifest_checksum": "synthetic-t10-3-12c",
            "collection_complete": True,
            "metrics": {
                "initial_frame_hashes": {
                    game: [f"frame-{index}"]
                    for index, game in enumerate(protocol.TARGET_GAMES)
                }
            },
        },
        "report_checksum",
    )
    protocol.write_json_once(output / runtime.ACTIVE_REPORT_FILENAME, active)
    result = runtime.adjudicate(tmp_path, _manifest())
    assert result["passed"] is True
    assert result["supported_cross_game_factors"] == [
        "operator", "role_binding", "transition", "termination"
    ]
    assert result["program_promoted"] is False
    assert result["sequence_composition_authorized"] is False
    assert result["verdict"] == "PASS_T10_3_12C_CROSS_GAME_FACTORS_IDENTIFIED"


def test_terminal_before_adjudication_is_incomplete(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    result = runtime.terminal_report(tmp_path, _manifest())
    assert result["verdict"] == "INCOMPLETE_T10_3_12C"
    assert result["passed"] is False
    assert result["sequence_games_opened"] is False
    assert result["production_authority"] is False
