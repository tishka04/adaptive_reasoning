"""A3 explicit correspondence hypothesis tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from theory.ar25_oracle import _DEFAULT_TASK_PROGRAM, build_ar25_oracle
from theory.correspondence_hypothesis import (
    CorrespondenceHypothesis,
    correspondence_key,
)
from theory.epistemic_metrics import HypothesisStatus
from theory.live_transition_loop import LiveTransitionBeliefLoop, run_trace_file


def test_correspondence_validates_on_level_advance():
    before = np.zeros((8, 8), dtype=np.int32)
    after = before.copy()
    before[1:3, 1:3] = 10
    before[1:3, 5:7] = 11
    after[1:3, 1:3] = 10
    after[1:3, 5:7] = 11

    loop = LiveTransitionBeliefLoop(
        "synthetic",
        available_actions=["ACTION2"],
        background_value=0,
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    loop.theory.add_semantic_hypotheses(
        correspondence=[
            CorrespondenceHypothesis(
                action="ACTION2",
                relation="validates",
                pair_colors=(10, 11),
            )
        ]
    )

    for idx in range(2):
        loop.observe_grids(
            action="ACTION2",
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION2"],
            levels_completed_before=idx,
            levels_completed_after=idx + 1,
            timestamp=idx,
        )

    key = correspondence_key("ACTION2", "validates", (10, 11))
    records = {record.key: record for record in loop.theory.to_ledger()}
    assert records[key].status == HypothesisStatus.CONFIRMED
    assert records[key].support == 2
    assert records[key].contradictions == 0
    rules = loop.theory.correspondence_rules()
    assert [rule.key for rule in rules] == [key]
    assert "PairedWith" in rules[0].predicates
    assert "CorrespondenceCount" in rules[0].predicates


def test_correspondence_improves_on_better_source_target_match():
    before = np.zeros((10, 10), dtype=np.int32)
    before[1:4, 1:4] = 10
    after = before.copy()
    after[1:4, 6:9] = 11

    loop = LiveTransitionBeliefLoop(
        "synthetic",
        available_actions=["ACTION1"],
        background_value=0,
        infer_players=False,
        correspondence_pair_colors=(10, 11),
    )
    loop.theory.add_semantic_hypotheses(
        correspondence=[
            CorrespondenceHypothesis(
                action="ACTION1",
                relation="improves",
                pair_colors=(10, 11),
            )
        ]
    )

    for idx in range(2):
        loop.observe_grids(
            action="ACTION1",
            grid_before=before,
            grid_after=after,
            available_actions=["ACTION1"],
            timestamp=idx,
        )

    key = correspondence_key("ACTION1", "improves", (10, 11))
    records = {record.key: record for record in loop.theory.to_ledger()}
    assert records[key].status == HypothesisStatus.CONFIRMED
    assert records[key].support == 2
    assert records[key].contradictions == 0


def test_ar25_live_like_confirms_explicit_correspondence_without_bad_confirmations():
    trace_path = Path("human_traces/ar25-e3c63847.20260420-163550.steps.jsonl")
    common_kwargs = dict(
        trace_path=trace_path,
        game_id="ar25-e3c63847",
        task_program_path=_DEFAULT_TASK_PROGRAM,
        max_steps=220,
        background_value=9,
        infer_players=False,
    )
    baseline = run_trace_file(**common_kwargs)
    loop = run_trace_file(
        **common_kwargs,
        correspondence_pair_colors=(10, 11),
    )
    oracle = build_ar25_oracle()
    baseline_score = baseline.score(oracle)
    score = loop.score(oracle)
    correspondence_records = [
        record for record in loop.theory.to_ledger()
        if record.key.startswith("correspondence::")
    ]
    confirmed_correspondence = [
        record for record in correspondence_records
        if record.status == HypothesisStatus.CONFIRMED
    ]

    assert score.confirmation_precision >= 0.95
    assert score.wrong_confirmations == 0
    assert score.human_alignment == 1.0
    assert score.correct_confirmations > baseline_score.correct_confirmations
    assert score.experiment_efficiency >= baseline_score.experiment_efficiency
    assert confirmed_correspondence
    assert any(
        record.key == correspondence_key("ACTION2", "validates", (10, 11))
        for record in confirmed_correspondence
    )
    assert oracle.verdict(correspondence_key("ACTION2", "validates", (10, 11))) is True
