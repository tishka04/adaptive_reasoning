"""Active one-feature lattice discrimination tests for SAGE.8y."""

from __future__ import annotations

import numpy as np

from theory.live_transition_loop import build_observation
from theory.online_mediated_abstraction import parse_mediated_candidate
from theory.online_mediated_discrimination import (
    MediatedDiscriminationStatus,
    OnlineMediatedDiscriminationStore,
)
from theory.online_mediated_entity_effect import (
    prospective_mediator_signature,
)
from theory.online_semantic_intervention import semantic_intervention_anchor


def _observation():
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[3, 3] = 8
    grid[3, 5] = 3
    return build_observation(
        grid,
        available_actions=["ACTION6"],
        infer_players=False,
    )


def _anchor():
    observation = _observation()
    return semantic_intervention_anchor(
        "ACTION6",
        {"x": 3, "y": 3},
        observation,
    )


def _features(*, extra_mismatch: bool = False):
    observation = _observation()
    anchor = _anchor()
    acted = next(
        item for item in observation.objects
        if item.object_id == anchor.source_object_id
    )
    carrier = next(item for item in observation.objects if item.value == 3)
    prospective = parse_mediated_candidate(
        prospective_mediator_signature(
            acted,
            carrier,
            observation,
            expected_change="transformed",
        )
    )
    result = {
        "change": prospective["change"],
        "color": prospective["color"],
        "area": prospective["area"],
        "boundary": prospective["boundary"],
        "proximity": "far",
    }
    if extra_mismatch:
        result["relation_alignment"] = (
            "offset"
            if prospective["relation_alignment"] == "aligned"
            else "aligned"
        )
    return result


def _outcome(features, *, gain: float = 1.0, candidates=()):
    signature = "mediated-abstract::" + "::".join(
        f"{key}:{value}" for key, value in features.items()
    )
    return {
        "observed": True,
        "gain": gain,
        "mode_signature": "mode:one",
        "action_transfer_signature": _anchor().transfer_signature,
        "supported_mediator_is_abstract": True,
        "mediator_abstraction": {
            "signature": signature,
            "features": dict(features),
            "supported": True,
        },
        "candidate_mediator_signatures": list(candidates),
        "target_stable_for_mediation": True,
        "ambiguous_scene_correspondence": False,
    }


def _request(memory, *, features=None):
    abstraction = dict(features or _features())
    request_id = memory.observe_hypothesis(
        option_id="option",
        edge_key="edge",
        objective_id="objective",
        downstream_subgoal_id="subgoal",
        anchor=_anchor(),
        branch_index=1,
        context_signature="source",
        mediated_outcome=_outcome(abstraction),
    )
    assert request_id
    return request_id, abstraction


def _activate(memory):
    memory.start_branch(2)
    return memory.note_opening(
        option_id="option",
        edge_key="edge",
        branch_index=2,
        opening_context="independent-opening",
    )


def test_supported_abstraction_reserves_a_later_branch_not_the_source():
    memory = OnlineMediatedDiscriminationStore()
    request_id, _features_map = _request(memory)

    blocked = memory.note_opening(
        option_id="option",
        edge_key="edge",
        branch_index=1,
        opening_context="same-branch",
    )
    memory.start_branch(2)

    assert blocked == ""
    assert memory.preferred_preparation_edge_key() == "edge"
    assert memory.requests()[0].request_id == request_id
    assert memory.summary()["same_branch_blocks"] == 1


def test_prediction_keeps_action_class_and_varies_exactly_one_feature():
    memory = OnlineMediatedDiscriminationStore()
    request_id, _features_map = _request(memory)
    assert _activate(memory) == request_id

    prediction = memory.predict(
        option_id="option",
        anchor=_anchor(),
        observation=_observation(),
        mode_signature="mode:one",
    )

    assert prediction is not None and prediction.compatible is True
    assert prediction.cross_branch is True
    assert prediction.single_feature_contrast is True
    assert prediction.tested_feature == "proximity"
    assert prediction.expected_value == "far"
    assert prediction.contrast_value == "adjacent"
    assert prediction.action_transfer_signature == _anchor().transfer_signature


def test_multi_feature_difference_is_not_a_matched_discriminator():
    memory = OnlineMediatedDiscriminationStore()
    _request_id, _features_map = _request(
        memory,
        features=_features(extra_mismatch=True),
    )
    _activate(memory)

    prediction = memory.predict(
        option_id="option",
        anchor=_anchor(),
        observation=_observation(),
        mode_signature="mode:one",
    )

    assert prediction is not None
    assert prediction.compatible is False
    assert prediction.single_feature_contrast is False


def test_latent_mode_mismatch_is_not_a_matched_discriminator():
    memory = OnlineMediatedDiscriminationStore()
    _request_id, _features_map = _request(memory)
    _activate(memory)

    prediction = memory.predict(
        option_id="option",
        anchor=_anchor(),
        observation=_observation(),
        mode_signature="mode:other",
    )

    assert prediction is not None
    assert prediction.same_latent_mode is False
    assert prediction.compatible is False
    assert prediction.single_feature_contrast is False
    assert memory.summary()["mode_mismatch_blocks"] == 1


def test_progress_with_observed_contrast_eliminates_the_feature_online():
    memory = OnlineMediatedDiscriminationStore()
    request_id, features = _request(memory)
    _activate(memory)
    prediction = memory.predict(
        option_id="option",
        anchor=_anchor(),
        observation=_observation(),
        mode_signature="mode:one",
    )
    assert prediction is not None and prediction.compatible
    memory.note_selection(prediction)

    memory.observe_hypothesis(
        option_id="option",
        edge_key="edge",
        objective_id="objective",
        downstream_subgoal_id="subgoal",
        anchor=_anchor(),
        branch_index=2,
        context_signature="contrast-progress",
        mediated_outcome=_outcome(
            features,
            gain=1.0,
            candidates=(prediction.prospective_candidate_signature,),
        ),
        selected_request_id=request_id,
    )

    request = memory.requests()[0]
    assert request.status == MediatedDiscriminationStatus.FEATURE_ELIMINATED
    assert memory.summary()["feature_eliminations"] == 1


def test_matched_nonprogress_control_supports_feature_requirement_online():
    memory = OnlineMediatedDiscriminationStore()
    request_id, features = _request(memory)
    _activate(memory)
    prediction = memory.predict(
        option_id="option",
        anchor=_anchor(),
        observation=_observation(),
        mode_signature="mode:one",
    )
    assert prediction is not None and prediction.compatible
    memory.note_selection(prediction)

    memory.observe_hypothesis(
        option_id="option",
        edge_key="edge",
        objective_id="objective",
        downstream_subgoal_id="subgoal",
        anchor=_anchor(),
        branch_index=2,
        context_signature="matched-control",
        mediated_outcome=_outcome(features, gain=0.0),
        selected_request_id=request_id,
    )

    request = memory.requests()[0]
    assert request.status == MediatedDiscriminationStatus.FEATURE_REQUIRED
    assert request.objective_gain == 0.0
    assert memory.summary()["feature_requirements"] == 1
    assert memory.summary()["requests_created"] == 2


def test_ambiguous_nonprogress_cannot_claim_feature_necessity():
    memory = OnlineMediatedDiscriminationStore()
    request_id, features = _request(memory)
    _activate(memory)
    prediction = memory.predict(
        option_id="option",
        anchor=_anchor(),
        observation=_observation(),
        mode_signature="mode:one",
    )
    assert prediction is not None and prediction.compatible
    memory.note_selection(prediction)
    outcome = _outcome(features, gain=0.0)
    outcome["ambiguous_scene_correspondence"] = True

    memory.observe_hypothesis(
        option_id="option",
        edge_key="edge",
        objective_id="objective",
        downstream_subgoal_id="subgoal",
        anchor=_anchor(),
        branch_index=2,
        context_signature="ambiguous-control",
        mediated_outcome=outcome,
        selected_request_id=request_id,
    )

    assert memory.requests()[0].status == MediatedDiscriminationStatus.PENDING
    assert memory.summary()["feature_requirements"] == 0
    assert memory.summary()["inconclusive_attempts"] == 1


def test_discrimination_ablation_creates_no_request_or_prediction():
    memory = OnlineMediatedDiscriminationStore(enabled=False)
    request_id = memory.observe_hypothesis(
        option_id="option",
        edge_key="edge",
        objective_id="objective",
        downstream_subgoal_id="subgoal",
        anchor=_anchor(),
        branch_index=1,
        context_signature="source",
        mediated_outcome=_outcome(_features()),
    )

    assert request_id == ""
    assert memory.requests() == []
    assert memory.summary()["requests_created"] == 0
