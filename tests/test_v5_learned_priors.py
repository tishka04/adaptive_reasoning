from types import SimpleNamespace

import numpy as np

from v5.control.arbiter import Arbiter
from v5.control.learned_priors import LearnedPriors
from v5.schemas import GameObservation, MindProposal, OperatorCall, PrimitiveAction


def _score(
    action="ACTION1",
    *,
    danger=0.0,
    noop=0.0,
    break_probability=0.0,
    progress=0.0,
):
    return SimpleNamespace(
        action=action,
        predicted_danger=danger,
        predicted_macro_scores={"explore": noop},
        predicted_break_probability=break_probability,
        predicted_progress=progress,
    )


class _FakeScorer:
    def __init__(self, scores=None):
        self.scores = scores or []
        self.calls = []

    def score_actions(
        self,
        state_features,
        actions,
        history_features,
        largest_component_features_by_action,
    ):
        self.calls.append(
            (
                state_features,
                actions,
                history_features,
                largest_component_features_by_action,
            )
        )
        return self.scores


def test_learned_priors_guards_and_clamps_promote_only_bonus():
    priors = LearnedPriors(
        _FakeScorer(),
        w_break=0.5,
        w_progress=0.25,
        max_bonus=0.1,
        danger_threshold=0.25,
        noop_threshold=0.7,
    )
    priors._scores = {
        "DANGER": _score(danger=0.26, break_probability=1.0),
        "NOOP": _score(noop=0.71, break_probability=1.0),
        "CLAMP": _score(break_probability=1.0, progress=10.0),
        "NEGATIVE": _score(break_probability=-2.0, progress=-3.0),
    }

    assert priors.bonus("DANGER") == 0.0
    assert priors.bonus("NOOP") == 0.0
    assert priors.bonus("CLAMP") == 0.1
    assert priors.bonus("NEGATIVE") == 0.0
    assert priors.bonus("MISSING") == 0.0
    assert priors.danger_guards == 1
    assert priors.noop_guards == 1


def test_begin_step_scores_actions_once_and_adds_action6_cursor_features():
    scorer = _FakeScorer([_score("ACTION1"), _score("ACTION6")])
    priors = LearnedPriors(scorer)
    grid = np.zeros((5, 7), dtype=np.int32)
    grid[1:4, 2:5] = 1

    priors.begin_step(
        grid,
        ["ACTION1", "ACTION6", "ACTION1"],
        {"steps_since_state_change": 2},
    )

    assert len(scorer.calls) == 1
    _, actions, _, local = scorer.calls[0]
    assert actions == ["ACTION1", "ACTION6"]
    assert local["ACTION1"]["cursor_present"] == 0.0
    assert local["ACTION6"]["cursor_present"] == 1.0
    assert priors.states_scored == 1
    assert priors.actions_scored == 2


def _proposal(name, progress):
    return MindProposal(
        mind_name=name,
        objective=name,
        candidate_plan=[OperatorCall(operator_id=name)],
        expected_progress=progress,
    )


def _minds(*names):
    return {name: SimpleNamespace(recent_accuracy=0.0) for name in names}


def test_arbiter_promotes_only_an_in_band_proposal_and_counts_reorder():
    arbiter = Arbiter(
        w_progress=1.0,
        w_info_gain=0.0,
        w_accuracy=0.0,
        w_confidence=0.0,
        w_cost=0.0,
        w_risk=0.0,
    )
    structural = _proposal("structural", 0.60)
    learned = _proposal("learned", 0.55)
    arbiter.set_prior(lambda p: 0.10 if p is learned else 0.0, band=0.10)

    selected = arbiter.select(
        [structural, learned],
        _minds("structural", "learned"),
    )

    assert selected is learned
    assert arbiter.prior_reorders == 1
    assert arbiter.prior_promotions == 1


def test_arbiter_never_calls_prior_for_an_out_of_band_proposal():
    arbiter = Arbiter(
        w_progress=1.0,
        w_info_gain=0.0,
        w_accuracy=0.0,
        w_confidence=0.0,
        w_cost=0.0,
        w_risk=0.0,
    )
    structural = _proposal("structural", 0.60)
    outside = _proposal("outside", 0.49)

    def bonus(proposal):
        assert proposal is not outside
        return 0.0

    arbiter.set_prior(bonus, band=0.10)
    selected = arbiter.select(
        [structural, outside],
        _minds("structural", "outside"),
    )

    assert selected is structural
    assert arbiter.prior_reorders == 0
    assert arbiter.prior_promotions == 0


def test_v5_flag_off_does_not_score_or_wire_learned_priors(monkeypatch):
    from v5.adaptive_reasoning_agent_v5 import AdaptiveReasoningAgentV5

    scorer = _FakeScorer([_score("ACTION1", break_probability=1.0)])
    priors = LearnedPriors(scorer)
    agent = AdaptiveReasoningAgentV5(
        cross_game=None,
        use_danger_memory=False,
        use_anti_attractor=False,
        use_learned_priors=False,
        learned_priors=priors,
    )
    obs = GameObservation(
        raw_grid=np.zeros((4, 4), dtype=np.int32),
        grid_hash=1,
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=["ACTION1", "ACTION2"],
    )
    monkeypatch.setattr(agent, "_decide_inner", lambda _obs: PrimitiveAction("ACTION1"))

    assert agent._decide(obs).name == "ACTION1"
    assert scorer.calls == []
    assert agent.arbiter._prior_bonus_fn is None
