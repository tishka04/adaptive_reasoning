from theory.m2.heuristic_generator import generate_heuristic_proposals
from theory.m2.normalizer import normalize_raw_proposals

from m2_test_helpers import frontier_request


def test_heuristic_generator_produces_bp35_expected_candidates():
    frontiers = [
        frontier_request(reason="context_not_covered_by_scope", context="after_ACTION6"),
        frontier_request(
            reason="confirmed_skill_blocked_by_failed_precondition",
            context="after_ACTION3_live_after_ACTION6",
            fallback_action="ACTION4",
            blocked_skill="ACTION6",
        ),
    ]

    raw = generate_heuristic_proposals(frontiers)
    hypotheses, rejected = normalize_raw_proposals(
        raw,
        frontiers_by_request_id={item["request_id"]: item for item in frontiers},
    )

    assert len(raw) >= 4
    assert not rejected
    assert len(hypotheses) >= 4
    assert {h.candidate_action for h in hypotheses} >= {"ACTION3", "ACTION4"}
    assert all(h.status == "UNRESOLVED" for h in hypotheses)
    assert all(h.support == 0 for h in hypotheses)
