from types import SimpleNamespace

import numpy as np

from theory.m1.actionable_source_alignment import (
    PreparationAction,
    SearchState,
    find_experimental_precondition,
    summarize_pretest_results,
)
from theory.m1.source_reachability import SourceAlignmentProblem


def _problem():
    return SourceAlignmentProblem(
        game_id="zz13-test",
        trace_path="human_traces/zz13.steps.jsonl",
        action="ACTION6",
        desired_source_color=8,
        target_color=2,
        available_live_sources=(5,),
        candidate_pair=("ACTION6", 8, 2),
        block_reason="source_not_selectable_for_action",
        source_scope="ranked_new_pairs",
    )


def _action(name, **action_args):
    return SimpleNamespace(name=name, action_args=dict(action_args))


def test_find_experimental_precondition_makes_source_actionable():
    problem = _problem()

    def evaluate(sequence):
        if not sequence:
            return SearchState(
                grid=np.asarray([[5, 2], [8, 0]], dtype=np.int32),
                actions=(_action("ACTION1"),),
            )
        return SearchState(
            grid=np.asarray([[5, 2], [8, 0]], dtype=np.int32),
            actions=(_action("ACTION6", x=0, y=1),),
        )

    result = find_experimental_precondition(
        problem,
        evaluate_sequence=evaluate,
        max_depth=3,
        max_branching=4,
    )

    assert result.found
    assert result.precondition is not None
    assert result.precondition.status == "UNRESOLVED"
    assert result.precondition.prep_length == 1
    assert result.precondition.then_test_pair == ("ACTION6", 8, 2)
    assert result.precondition.source_becomes_actionable is True
    assert result.precondition.target_still_present is True
    assert result.precondition.final_available_sources == (8,)
    assert result.precondition.trace_support_counted_as_proof is False
    assert result.precondition.prior_counted_as_proof is False


def test_find_experimental_precondition_reports_unresolved_when_not_found():
    problem = _problem()

    def evaluate(_sequence):
        return SearchState(
            grid=np.asarray([[5, 2], [8, 0]], dtype=np.int32),
            actions=(_action("ACTION6", x=0, y=0),),
        )

    result = find_experimental_precondition(
        problem,
        evaluate_sequence=evaluate,
        max_depth=2,
        max_branching=2,
        max_nodes=5,
    )

    assert result.found is False
    assert result.precondition is None
    assert result.evaluated_nodes > 0


def test_summarize_pretest_results_reports_core_kpis():
    problem = _problem()
    result = find_experimental_precondition(
        problem,
        evaluate_sequence=lambda _sequence: SearchState(
            grid=np.asarray([[8, 2]], dtype=np.int32),
            actions=(),
        ),
        max_depth=1,
    )
    # The search only returns preconditions after at least one preparation
    # action, and this state offers no preparation action.
    assert not result.found

    found = find_experimental_precondition(
        problem,
        evaluate_sequence=lambda sequence: SearchState(
            grid=np.asarray([[8, 2]], dtype=np.int32),
            actions=(
                (_action("ACTION1"),)
                if not sequence
                else (_action("ACTION6", x=0, y=0),)
            ),
        ),
        max_depth=1,
    )
    summary = summarize_pretest_results([result, found])

    assert summary["problems_total"] == 2
    assert summary["preconditions_found"] == 1
    assert summary["mean_prep_length"] == 1.0
    assert summary["source_becomes_actionable"] == 1
    assert summary["target_still_present"] == 1
    assert summary["per_game"]["zz13-test"]["problems_total"] == 2


def test_preparation_action_signature_is_stable():
    assert PreparationAction("ACTION1", {"y": 2, "x": 1}).signature == (
        "ACTION1",
        (("x", "1"), ("y", "2")),
    )
