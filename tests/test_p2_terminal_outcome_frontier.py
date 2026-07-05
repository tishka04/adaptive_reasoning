import json

import pytest

from theory.p2 import terminal_outcome_frontier as terminal


def _run_result(
    *,
    budget=96,
    seed=0,
    final_game_state="GAME_OVER",
    levels=0,
    useful_action6=64,
    refresh_triggers=0,
    true_saturation=False,
):
    return {
        "budget": budget,
        "tie_break_seed": seed,
        "summary": {
            "final_game_state": final_game_state,
            "final_levels_completed": levels,
            "useful_action6_steps": useful_action6,
            "action6_steps": useful_action6 + 1,
            "unique_action6_args_selected": 6,
            "repeated_action6_args_selected": 59,
            "progress_proxy": 512.0,
            "conditional_movement_refresh_triggers": refresh_triggers,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P2",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "classification": {
            "true_frontier_triggered": true_saturation,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P2",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P2",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def _source_payload(run_results=None, *, support=0, verdict=False):
    run_results = list(run_results or [])
    return {
        "config": {
            "schema_version": "p2.long_budget_saturation_probe.v1",
            "game_id": "bp35-0a0ad940",
        },
        "run_results": run_results,
        "summary": {
            "real_frontier_ready_for_p2_4": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P2",
            "revision_performed": False,
            "wrong_confirmations": 0,
        },
        "long_budget_probe_counted_as_confirmation": False,
        "policy_result_counted_as_scientific_verdict": verdict,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P2",
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def test_terminal_classifier_flags_local_productive_game_over_without_saturation():
    result = terminal.classify_run_for_terminal_frontier(
        _run_result(),
        min_useful_action6_steps=10,
    )

    assert result["terminal_objective_frontier_triggered"] is True
    assert result["frontier_type"] == terminal.OBJECTIVE_ALIGNMENT_FRONTIER
    assert result["frontier_reason"] == terminal.LOCAL_PRODUCTIVE_TERMINAL_FAILED
    assert result["saturation_handoff_ready"] is False
    assert result["support"] == 0


def test_terminal_classifier_does_not_flag_low_local_signal_or_saturation():
    weak = terminal.classify_run_for_terminal_frontier(
        _run_result(useful_action6=2),
        min_useful_action6_steps=10,
    )
    saturation = terminal.classify_run_for_terminal_frontier(
        _run_result(refresh_triggers=1, true_saturation=True),
        min_useful_action6_steps=10,
    )

    assert weak["terminal_objective_frontier_triggered"] is False
    assert saturation["terminal_objective_frontier_triggered"] is False
    assert weak["frontier_reason"] == terminal.NO_TERMINAL_OBJECTIVE_FRONTIER


def test_terminal_frontier_summary_keeps_no_write_guards(tmp_path):
    path = tmp_path / "p2_3.json"
    path.write_text(
        json.dumps(_source_payload([_run_result(), _run_result(seed=1)])),
        encoding="utf-8",
    )

    payload = terminal.run_terminal_outcome_frontier_classifier(
        long_budget_saturation_probe_path=path,
        min_useful_action6_steps=10,
    )

    assert payload["terminal_outcome_frontier"] is not None
    assert payload["summary"]["runs_seen"] == 2
    assert payload["summary"]["objective_alignment_frontier_runs"] == 2
    assert payload["summary"]["ready_for_objective_frontier_review"] is True
    assert payload["summary"]["ready_for_p2_4_saturation_handoff"] is False
    assert payload["summary"]["a40_write_performed"] is False
    assert payload["summary"]["m2_write_performed"] is False
    assert payload["summary"]["m3_write_performed"] is False
    assert payload["summary"]["support"] == 0
    assert payload["terminal_outcome_frontier_counted_as_confirmation"] is False


def test_terminal_frontier_classifier_returns_none_without_terminal_runs(tmp_path):
    path = tmp_path / "p2_3.json"
    path.write_text(
        json.dumps(
            _source_payload(
                [
                    _run_result(
                        final_game_state="NOT_FINISHED",
                        levels=0,
                        useful_action6=64,
                    )
                ]
            )
        ),
        encoding="utf-8",
    )

    payload = terminal.run_terminal_outcome_frontier_classifier(
        long_budget_saturation_probe_path=path,
        min_useful_action6_steps=10,
    )

    assert payload["terminal_outcome_frontier"] is None
    assert payload["summary"]["ready_for_objective_frontier_review"] is False
    assert payload["summary"]["support"] == 0


def test_terminal_frontier_rejects_source_verdict_flags(tmp_path):
    path = tmp_path / "p2_3.json"
    path.write_text(json.dumps(_source_payload(verdict=True)), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific verdict"):
        terminal.run_terminal_outcome_frontier_classifier(
            long_budget_saturation_probe_path=path,
        )
