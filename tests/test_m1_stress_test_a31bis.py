import json
from pathlib import Path

from theory.m1.stress_test_a31bis import (
    comparison_deltas,
    diagnose_result,
    load_baseline_json,
    summarize_stress_dict,
)


def test_a31bis_loads_baseline_with_log_prefix(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        "2026-06-18 15:16:31 | INFO | noisy prefix\n"
        + json.dumps({"wrong_confirmations": 0, "failure_types": {}}),
        encoding="utf-8",
    )

    loaded = load_baseline_json(baseline)

    assert loaded == {"wrong_confirmations": 0, "failure_types": {}}


def test_a31bis_loads_utf16_baseline_with_log_prefix(tmp_path: Path):
    baseline = tmp_path / "baseline_utf16.json"
    baseline.write_text(
        "2026-06-18 15:16:31 | INFO | noisy prefix\n"
        + json.dumps({"wrong_confirmations": 0, "failure_types": {}}),
        encoding="utf-16",
    )

    loaded = load_baseline_json(baseline)

    assert loaded == {"wrong_confirmations": 0, "failure_types": {}}


def test_a31bis_summary_and_deltas_focus_on_blocker_and_precision():
    baseline = summarize_stress_dict(
        {
            "trace_count": 18,
            "game_count": 6,
            "experiments_run": 624,
            "hypotheses_confirmed": 100,
            "hypotheses_refuted": 200,
            "useful_new_states": 50,
            "wrong_confirmations": 0,
            "failure_types": {"not_enough_relation_candidates": 72},
        },
        label="baseline",
    )
    m1 = summarize_stress_dict(
        {
            "trace_count": 6,
            "game_count": 6,
            "experiments_run": 100,
            "hypotheses_confirmed": 30,
            "hypotheses_refuted": 70,
            "useful_new_states": 20,
            "wrong_confirmations": 0,
            "failure_types": {"not_enough_relation_candidates": 50},
        },
        label="m1",
    )

    deltas = comparison_deltas(baseline, m1)

    assert baseline.not_enough_relation_candidates == 72
    assert m1.not_enough_relation_candidates == 50
    assert deltas["not_enough_relation_candidates"] == -22
    assert deltas["wrong_confirmations"] == 0


def test_a31bis_diagnoses_vocab_expansion_without_pair_expansion():
    baseline = summarize_stress_dict(
        {
            "failure_types": {"not_enough_relation_candidates": 72},
            "wrong_confirmations": 0,
        },
        label="baseline",
    )
    m1 = summarize_stress_dict(
        {
            "failure_types": {"not_enough_relation_candidates": 72},
            "wrong_confirmations": 0,
        },
        label="m1",
    )
    coverage = {
        "averages": {
            "unique_predicates_per_trace": {"delta": 7.0},
            "relation_candidates_generated": {"delta": 100.0},
            "candidate_pairs_per_trace": {"delta": 0.0},
        }
    }

    diagnosis = diagnose_result(baseline, m1, coverage)

    assert diagnosis == "vocabulary_expanded_pairs_flat_blocker_not_reduced"
