from types import SimpleNamespace

import numpy as np

from theory.cross_game_correspondence_discovery import (
    DiscoveredCorrespondenceCandidate,
    SourceTargetPredicate,
)
from theory.m1.live_anchor_ranking import (
    M1_LIVE_PREFERRED_PREDICATES,
    build_m1_live_candidate_ranker,
    consumability_metrics,
    rank_live_compatible_candidates,
)
from theory.m1.stress_test_a31bis import diagnose_result, summarize_stress_dict


def _candidate(
    source: int,
    target: int,
    *,
    action: str = "ACTION1",
    predicates=("m1_anchor_contact",),
    support: int = 1,
):
    return DiscoveredCorrespondenceCandidate(
        game_id="zz13-live",
        action=action,
        source_color=source,
        target_color=target,
        support=support,
        transition_support=1,
        predicates=tuple(
            SourceTargetPredicate(name=predicate, support=1)
            for predicate in predicates
        ),
    )


def _action(name="ACTION1", x=0, y=0):
    return SimpleNamespace(name=name, action_args={"x": x, "y": y})


def test_live_anchor_ranker_augments_live_preferred_predicates():
    grid = np.asarray(
        [
            [1, 2, 0],
            [1, 2, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    candidate = _candidate(1, 2)

    ranked = rank_live_compatible_candidates(
        [candidate],
        live_grid=grid,
        valid_actions=[_action()],
        max_candidates=5,
    )
    predicate_names = {predicate.name for predicate in ranked[0].predicates}

    assert predicate_names >= set(M1_LIVE_PREFERRED_PREDICATES)


def test_live_anchor_ranker_prioritizes_consumable_pair_over_raw_support():
    grid = np.asarray(
        [
            [1, 2, 0],
            [1, 2, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    consumable = _candidate(1, 2, support=1)
    unsupported_live_source = _candidate(3, 2, support=100)

    ranked = rank_live_compatible_candidates(
        [unsupported_live_source, consumable],
        live_grid=grid,
        valid_actions=[_action()],
        max_candidates=2,
    )

    assert ranked[0].pair_colors == (1, 2)


def test_consumability_metrics_report_new_pairs_entering_agenda_after_ranking():
    grid = np.asarray(
        [
            [1, 2, 0],
            [1, 2, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    candidate = _candidate(1, 2)
    ranker = build_m1_live_candidate_ranker()

    before = consumability_metrics(
        [candidate],
        baseline_candidates=[],
        live_grid=grid,
        valid_actions=[_action()],
    )
    ranked = ranker(
        [candidate],
        live_grid=grid,
        valid_actions=[_action()],
        max_candidates=5,
    )
    after = consumability_metrics(
        ranked,
        baseline_candidates=[],
        live_grid=grid,
        valid_actions=[_action()],
    )

    assert before.new_pairs_total == 1
    assert before.new_pairs_live_color_compatible == 1
    assert before.new_pairs_blocked_by_unselectable_source == 0
    assert before.new_pairs_with_2_preferred_predicates == 0
    assert before.new_pairs_entering_agenda == 0
    assert after.new_pairs_with_2_preferred_predicates == 1
    assert after.new_pairs_entering_agenda == 1


def test_consumability_metrics_report_unselectable_source_blockers():
    grid = np.asarray(
        [
            [1, 2, 0],
            [1, 2, 0],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    candidate = _candidate(3, 2)

    metrics = consumability_metrics(
        [candidate],
        baseline_candidates=[],
        live_grid=grid,
        valid_actions=[_action()],
    )

    assert metrics.new_pairs_total == 1
    assert metrics.new_pairs_live_color_compatible == 0
    assert metrics.new_pairs_blocked_by_unselectable_source == 1
    assert metrics.to_dict()["new_pairs_blocked_by_unselectable_source"] == 1


def test_a31bis_diagnoses_live_anchor_entering_agenda_without_blocker_drop():
    baseline = summarize_stress_dict(
        {"failure_types": {"not_enough_relation_candidates": 8}},
        label="baseline",
    )
    m1 = summarize_stress_dict(
        {"failure_types": {"not_enough_relation_candidates": 8}},
        label="m1",
    )
    coverage = {
        "averages": {
            "unique_predicates_per_trace": {"delta": 1.0},
            "relation_candidates_generated": {"delta": 1.0},
            "candidate_pairs_per_trace": {"delta": 1.0},
        }
    }
    live = {
        "averages_after_ranking": {
            "new_pairs_live_color_compatible": 1.0,
            "new_pairs_entering_agenda": 1.0,
        }
    }

    assert (
        diagnose_result(baseline, m1, coverage, live)
        == "live_anchor_ranking_entering_agenda_blocker_not_reduced"
    )
