from pathlib import Path

from theory.m1.source_reachability import (
    SOURCE_NOT_SELECTABLE_REASON,
    extract_source_alignment_problems,
    load_grounding_autopsy,
    run_source_reachability_analysis,
    summarize_source_alignment_problems,
)


def test_source_alignment_problem_typed_from_blocked_pair():
    payload = {
        "games": [
            {
                "game_id": "bp35-0a0ad940",
                "trace_path": "human_traces/bp35.steps.jsonl",
                "ranked_new_pairs": [
                    {
                        "action": "ACTION6",
                        "source_color": 14,
                        "target_color": 3,
                        "live_source_colors_for_action": [5],
                        "block_reason": SOURCE_NOT_SELECTABLE_REASON,
                        "support": 42,
                        "transition_support": 3,
                        "target_live_present": True,
                        "predicates": ["m1_anchor_motion", "aligned_with"],
                        "preferred_predicates": ["aligned_with"],
                        "live_preferred_predicates": ["aligned_with"],
                    },
                    {
                        "action": "ACTION6",
                        "source_color": 5,
                        "target_color": 3,
                        "live_source_colors_for_action": [5],
                        "block_reason": "agenda_eligible",
                    },
                ],
            }
        ]
    }

    problems = extract_source_alignment_problems(payload)

    assert len(problems) == 1
    problem = problems[0]
    assert problem.game_id == "bp35-0a0ad940"
    assert problem.action == "ACTION6"
    assert problem.desired_source_color == 14
    assert problem.target_color == 3
    assert problem.available_live_sources == (5,)
    assert problem.candidate_pair == ("ACTION6", 14, 3)
    assert problem.block_reason == SOURCE_NOT_SELECTABLE_REASON
    assert problem.status == "UNRESOLVED"
    assert problem.trace_support_counted_as_proof is False
    assert problem.prior_counted_as_proof is False
    assert problem.to_dict()["candidate_pair"] == ["ACTION6", 14, 3]


def test_source_reachability_extracts_bp35_cd82_dc22_new_pair_blockers():
    path = Path("diagnostics/m1/grounding_autopsy.json")
    assert path.exists()
    payload = load_grounding_autopsy(path)

    problems = extract_source_alignment_problems(
        payload,
        source_scope="ranked_new_pairs",
    )
    summary = summarize_source_alignment_problems(problems)

    assert {
        game_id: row["problem_count"]
        for game_id, row in summary.items()
    } == {
        "bp35-0a0ad940": 5,
        "cd82-fb555c5d": 4,
        "dc22-4c9bff3e": 3,
    }
    assert all(
        problem.desired_source_color not in problem.available_live_sources
        for problem in problems
    )
    assert all(problem.status == "UNRESOLVED" for problem in problems)


def test_source_reachability_keeps_global_and_m1_pair_scopes_separate():
    payload = run_source_reachability_analysis(
        grounding_autopsy_path="diagnostics/m1/grounding_autopsy.json",
    )

    assert payload["ranked_pairs_problem_count"] == 55
    assert payload["ranked_new_pairs_problem_count"] == 12
    assert {
        game_id: row["problem_count"]
        for game_id, row in payload["ranked_pairs_summary"].items()
    } == {
        "bp35-0a0ad940": 20,
        "cd82-fb555c5d": 20,
        "dc22-4c9bff3e": 15,
    }
    assert payload["status"] == "UNRESOLVED"
    assert payload["trace_support_counted_as_proof"] is False
    assert payload["prior_counted_as_proof"] is False
