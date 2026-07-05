from theory.m2.normalizer import normalize_raw_proposal
from theory.m2.schema import FrontierConditionedHypothesis, RawHypothesisProposal


def frontier_request(
    *,
    reason="context_not_covered_by_scope",
    context="after_ACTION6",
    request_id=None,
    fallback_action="ACTION3",
    blocked_skill="",
):
    return {
        "request_id": request_id or f"frontier::{context}::step_1::{reason}",
        "source_step": 1,
        "game_id": "bp35-0a0ad940",
        "frontier_context_id": context,
        "context_signature": ["ACTION6"],
        "reason": reason,
        "reason_categories": ["uncovered_context"],
        "recommended_next_scientific_action": "generate_new_candidate_mechanics_from_current_state",
        "live_state_signature": "state:frontier",
        "blocked_skill": blocked_skill,
        "failed_precondition": (
            "target_patch_not_already_saturated=true" if blocked_skill else ""
        ),
        "fallback_action": fallback_action,
        "fallback_progress": False,
        "selected_signal": 0.0,
        "available_actions": ["ACTION3", "ACTION4", "ACTION6"],
        "ready_for_m1_or_m3": True,
        "status": "OPEN",
    }


def raw_proposal(**overrides):
    frontier = overrides.pop("frontier", frontier_request())
    data = {
        "proposal_id": "raw::test::001",
        "source": "heuristic",
        "source_request_id": frontier["request_id"],
        "game_id": frontier["game_id"],
        "frontier_context_id": frontier["frontier_context_id"],
        "frontier_reason": frontier["reason"],
        "frontier_step": frontier["source_step"],
        "hypothesis_family": "post_consumption_transition",
        "candidate_action": "ACTION3",
        "predicted_metric": "local_patch_before_after",
        "predicted_effect": "ACTION3 may expose a new local target",
        "rationale": "Candidate only.",
        "suggested_control_actions": ("ACTION4",),
        "required_context_replay": ("ACTION6",),
    }
    data.update(overrides)
    return RawHypothesisProposal(**data)


def valid_hypothesis(**overrides) -> FrontierConditionedHypothesis:
    frontier = overrides.pop("frontier", frontier_request())
    proposal = raw_proposal(frontier=frontier, **overrides)
    result = normalize_raw_proposal(proposal, frontier_request=frontier)
    assert isinstance(result, FrontierConditionedHypothesis)
    return result
