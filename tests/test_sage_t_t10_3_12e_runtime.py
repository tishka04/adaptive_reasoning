from __future__ import annotations

from pathlib import Path

from theory.sage_t import t10_3_12e_protocol as protocol
from theory.sage_t import t10_3_12e_runtime as runtime
from theory.sage_t.closed_loop_successor_v10_3_12e import signed


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-t10-3-12e",
        "parent_journal_digest": protocol.parent_journal_digest(Path.cwd()),
        "parent_artifacts": protocol.parent_artifact_bindings(Path.cwd()),
        "gates": {
            "parent_receipts": 36,
            "parent_path_applicable_games": 3,
            "parent_stable_successes": 0,
            "parent_changed_frame_events": 91,
            "preflight_cases": 16,
            "minimum_path_applicable_games": 3,
            "minimum_dynamic_success_games": 1,
            "minimum_dynamic_over_frozen_success_advantage": 1,
            "minimum_dynamic_over_stateless_success_advantage": 1,
            "minimum_dynamic_over_goal_swap_success_advantage": 1,
            "anchor_builds_per_applicable_reset": 1,
            "maximum_dynamic_grounding_misses": 0,
            "minimum_dynamic_exact_grounding_fraction": 1.0,
            "minimum_dynamic_frontier_advance_fraction": 1.0,
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
                "manifest_checksum": "synthetic-t10-3-12e",
                "passed": True,
            },
            checksum_field,
        ),
    )


def test_parent_motivation_audit_and_preflight_are_zero_action(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_gate(output, runtime.PARENT_AUDIT_FILENAME, "audit_checksum")
    audit = runtime.audit_trajectories(Path.cwd(), _manifest())
    preflight = runtime.preflight(Path.cwd(), _manifest())
    assert audit["passed"] is True
    assert len(audit["stable_path_summary"]) == 3
    assert audit["changed_frame_events"] == 91
    assert audit["confirmatory_evidence"] is False
    assert preflight["passed"] is True
    assert len(preflight["cases"]) == 16
    assert preflight["physical_actions"] == 0


def test_registry_compiles_without_parent_paths_checksums_or_support(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_gate(output, runtime.PREFLIGHT_FILENAME, "preflight_checksum")
    registry = runtime.compile_programs(Path.cwd(), _manifest())
    assert registry["compiled_before_diagnostic_actions"] is True
    assert registry["parent_outcomes_used_for_program_fit"] is False
    assert registry["parent_grounded_paths_imported"] is False
    assert registry["parent_action_checksums_imported"] is False
    assert registry["historical_support_imported"] == 0
    assert len(registry["programs"]) == 4


def test_adjudication_requires_dynamic_terminal_advantage(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    applicable_games = set(protocol.TARGET_GAMES[:3])
    success_game = protocol.TARGET_GAMES[0]
    for work in protocol.work_specs("active-diagnostic"):
        applicable = work.game_id in applicable_games
        succeeds = (
            work.arm == "anchored_goal_dynamic_successor"
            and work.game_id == success_game
        )
        actions = 5 if succeeds else (3 if applicable else 0)
        primary = work.arm == "anchored_goal_dynamic_successor" and applicable
        row = signed(
            {
                "format_version": "synthetic-receipt",
                "manifest_checksum": "synthetic-t10-3-12e",
                **work.as_dict(),
                "work_id": work.work_id,
                "complete": True,
                "level_delta": int(succeeds),
                "sealed_events": actions,
                "sage_t_option_actions": actions,
                "recognized_context": "path_context" if applicable else "",
                "anchor_builds": 1 if applicable else 0,
                "relation_evaluations": actions if applicable else 1,
                "dynamic_regrounds": actions if applicable else 1,
                "frontier_advances": actions if primary else 0,
                "repeat_proposals_rejected": 0,
                "exact_groundings": actions if primary else 0,
                "grounding_misses": 0,
                "frontier_size": actions if primary else 0,
                "frozen_cursor": actions if work.arm == "frozen_grounded_cursor" else 0,
                "initial_path_length": 10 if applicable else 0,
                "path_plan_persisted": False,
                "visited_action_keys_persisted": False,
                "grounded_arguments_persisted": False,
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
            "manifest_checksum": "synthetic-t10-3-12e",
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
    assert result["closed_loop_mechanism_recovered"] is True
    assert result["cross_game_generalization_proven"] is False
    assert result["factor_generalization_proven"] is False
    assert result["confirmatory_evidence"] is False
    assert result["program_promoted"] is False
    assert result["verdict"] == "PASS_T10_3_12E_CLOSED_LOOP_RELATIONAL_SUCCESSOR"


def test_terminal_before_adjudication_is_incomplete(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    result = runtime.terminal_report(tmp_path, _manifest())
    assert result["verdict"] == "INCOMPLETE_T10_3_12E"
    assert result["passed"] is False
    assert result["confirmatory_evidence"] is False
    assert result["sequence_games_opened"] is False
    assert result["production_authority"] is False
