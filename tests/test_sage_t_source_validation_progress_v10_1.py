from __future__ import annotations

from theory.sage_t.progress_witness_v10 import (
    AbstractWitnessStep,
    GroundedAction,
    ProgressWitness,
    SearchOutcome,
    compile_progress_program,
)
from theory.sage_t.source_validation_progress_v10_1 import (
    SOURCE_VALIDATION_GAMES,
    build_report,
)


def _outcome(game: str, *, progress: bool) -> SearchOutcome:
    witness = None
    diagnosis = "SEQUENCE_MISS"
    if progress:
        witness = ProgressWitness(
            source_game=game,
            context_signature="validation_context",
            macro_schema="repeat_target",
            relation="identity",
            abstract_steps=(
                AbstractWitnessStep(expected_event="level_progress"),
            ),
            grounded_actions=(
                GroundedAction("ACTION6", (("x", 1), ("y", 2))),
            ),
            observed_events=(("progress", "level_complete"),),
            level_delta=1,
            program=compile_progress_program(sequence_length=1),
            posterior_rank=1,
            posterior_mass=0.99,
        )
        diagnosis = "SUCCESS"
    return SearchOutcome(
        game=game,
        witness=witness,
        diagnosis=diagnosis,
        scan_rows=(),
        effect_groups=1,
        candidate_macros=1,
        macros_executed=1,
        actions_executed=1,
        illegal_actions=0,
        terminal_events=0,
        errors=(),
        wall_seconds=0.1,
    )


def _manifest() -> dict:
    return {
        "manifest_checksum": "frozen",
        "search_config": {},
        "gate": {
            "minimum_progress_games": 2,
            "minimum_total_levels": 2,
            "maximum_posterior_rank": 8,
            "maximum_illegal_actions": 0,
            "maximum_errors": 0,
            "maximum_game_over_events": 0,
            "maximum_wall_seconds": 300.0,
        },
    }


def test_t10_1_gate_passes_two_of_three_safe_progress_games() -> None:
    outcomes = (
        _outcome(SOURCE_VALIDATION_GAMES[0], progress=True),
        _outcome(SOURCE_VALIDATION_GAMES[1], progress=True),
        _outcome(SOURCE_VALIDATION_GAMES[2], progress=False),
    )
    report = build_report(
        manifest=_manifest(),
        outcomes=outcomes,
        wall_seconds=1.0,
        include_scan_rows=False,
    )
    assert report["passed"] is True
    assert report["metrics"]["progress_games"] == 2
    assert report["metrics"]["total_levels"] == 2
    assert report["firewall"]["holdout_opened"] is False


def test_t10_1_gate_fails_closed_on_one_progress_game() -> None:
    outcomes = (
        _outcome(SOURCE_VALIDATION_GAMES[0], progress=True),
        _outcome(SOURCE_VALIDATION_GAMES[1], progress=False),
        _outcome(SOURCE_VALIDATION_GAMES[2], progress=False),
    )
    report = build_report(
        manifest=_manifest(),
        outcomes=outcomes,
        wall_seconds=1.0,
        include_scan_rows=False,
    )
    assert report["passed"] is False
    assert report["status"] == "FAIL_CLOSED"
    assert report["checks"]["minimum_progress_games"] is False
    assert report["firewall"]["integration_pilot_authorized"] is False
