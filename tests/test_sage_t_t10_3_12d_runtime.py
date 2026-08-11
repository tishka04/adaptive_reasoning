from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12d_protocol as protocol
from theory.sage_t import t10_3_12d_runtime as runtime
from theory.sage_t.executor_correspondence_v10_3_12d import signed


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-t10-3-12d",
        "parent_journal_digest": protocol.parent_journal_digest(Path.cwd()),
        "parent_artifacts": protocol.parent_artifact_bindings(Path.cwd()),
        "gates": {
            "parent_full_path_branches": 3,
            "parent_collapsed_suffix_branches": 3,
            "parent_ablation_wins_on_lf52": 2,
            "preflight_cases": 14,
            "minimum_path_applicable_games": 3,
            "minimum_stable_success_games": 1,
            "minimum_stable_over_stateless_success_advantage": 1,
            "minimum_stable_over_cursor_hold_success_advantage": 1,
            "minimum_stable_reacquisition_fraction": 1.0,
        },
        "matrix": {
            "maximum_artifact_bytes": 15 * 1024 * 1024,
            "maximum_global_wall_seconds": 7200,
        },
    }


def _write_gate(output: Path, filename: str, checksum_field: str) -> None:
    protocol.write_json_once(
        output / filename,
        signed(
            {
                "format_version": "synthetic-gate",
                "manifest_checksum": "synthetic-t10-3-12d",
                "passed": True,
            },
            checksum_field,
        ),
    )


def test_parent_trajectory_audit_and_preflight_are_zero_action(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_gate(output, runtime.PARENT_AUDIT_FILENAME, "audit_checksum")
    audit = runtime.audit_trajectories(Path.cwd(), _manifest())
    preflight = runtime.preflight(Path.cwd(), _manifest())
    assert audit["passed"] is True
    assert len(audit["collapse_summary"]) == 3
    assert all(row["collapsed_suffix"] for row in audit["collapse_summary"])
    assert audit["confirmatory_evidence"] is False
    assert preflight["passed"] is True
    assert len(preflight["cases"]) == 14
    assert preflight["physical_actions"] == 0


def test_executor_registry_compiles_without_parent_paths_or_support(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_gate(output, runtime.PREFLIGHT_FILENAME, "preflight_checksum")
    registry = runtime.compile_executors(Path.cwd(), _manifest())
    assert registry["compiled_before_diagnostic_actions"] is True
    assert registry["parent_outcomes_used_for_program_fit"] is False
    assert registry["parent_grounded_paths_imported"] is False
    assert registry["historical_support_imported"] == 0
    assert len(registry["programs"]) == 4


def test_adjudication_recovers_correspondence_without_generalization_claim(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    applicable_games = set(protocol.TARGET_GAMES[:3])
    success_game = protocol.TARGET_GAMES[0]
    for work in protocol.work_specs("active-diagnostic"):
        applicable = work.game_id in applicable_games
        succeeds = work.arm == "stable_source_cursor" and work.game_id == success_game
        actions = 5 if succeeds else (3 if applicable else 0)
        stable = work.arm == "stable_source_cursor" and applicable
        row = signed(
            {
                "format_version": "synthetic-receipt",
                "manifest_checksum": "synthetic-t10-3-12d",
                **work.as_dict(),
                "work_id": work.work_id,
                "complete": True,
                "level_delta": int(succeeds),
                "sealed_events": actions,
                "sage_t_option_actions": actions,
                "recognized_context": "path_context" if applicable else "",
                "plan_builds": 1 if applicable else 0,
                "replans": 0,
                "plan_length": 10 if applicable else 0,
                "cursor": actions if stable else 0,
                "reacquisitions": actions if stable else 0,
                "grounding_misses": 0,
                "path_plan_persisted": False,
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
    active = signed(
        {
            "format_version": "synthetic-active",
            "manifest_checksum": "synthetic-t10-3-12d",
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
    assert result["executor_correspondence_recovered"] is True
    assert result["cross_game_generalization_proven"] is False
    assert result["factor_generalization_proven"] is False
    assert result["confirmatory_evidence"] is False
    assert result["program_promoted"] is False
    assert result["verdict"] == "PASS_T10_3_12D_EXECUTOR_CORRESPONDENCE_RECOVERED"


def test_terminal_before_adjudication_is_incomplete(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    result = runtime.terminal_report(tmp_path, _manifest())
    assert result["verdict"] == "INCOMPLETE_T10_3_12D"
    assert result["passed"] is False
    assert result["confirmatory_evidence"] is False
    assert result["sequence_games_opened"] is False
    assert result["production_authority"] is False
