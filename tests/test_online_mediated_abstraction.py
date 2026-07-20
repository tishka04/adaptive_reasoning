"""Online structural anti-unification tests for SAGE.8x."""

from __future__ import annotations

from theory.online_mediated_abstraction import (
    abstraction_matches_candidate,
    induce_mediated_abstraction,
    parse_mediated_candidate,
)
from theory.online_mediated_entity_effect import (
    MediatedEffectStatus,
    MediatedEntityEffectEvidence,
)


SOURCE_COLOR0 = (
    "mediated:transformed::"
    "entity:color0:shape15x15-source:area9+::"
    "role:interior:player-far-offset:unique:adj2:align0::"
    "relation:adjacent:above:right:offset:different-color"
)
SECOND_COLOR0 = (
    "mediated:transformed::"
    "entity:color0:shape1x1-second:area1::"
    "role:edge:player-self:unique:adj1:align1::"
    "relation:far:above:left:offset:different-color"
)
SECOND_COLOR12 = (
    "mediated:transformed::"
    "entity:color12:shape15x15-other:area9+::"
    "role:interior:player-far-aligned:unique:adj2:align0::"
    "relation:adjacent:below:left:offset:different-color"
)
SECOND_COLOR4 = (
    "mediated:transformed::"
    "entity:color4:shape1x31-other:area9+::"
    "role:edge:player-adjacent-aligned:unique:adj1:align1::"
    "relation:far:above:right:offset:different-color"
)


def _induce(**kwargs):
    return induce_mediated_abstraction(
        ((SOURCE_COLOR0,), (SECOND_COLOR0, SECOND_COLOR12, SECOND_COLOR4)),
        preferred_colors=(0,),
        **kwargs,
    )


def test_candidate_parser_exposes_generic_entity_role_and_relation_features():
    parsed = parse_mediated_candidate(SOURCE_COLOR0)

    assert parsed["change"] == "transformed"
    assert parsed["color"] == "color0"
    assert parsed["shape"] == "shape15x15-source"
    assert parsed["boundary"] == "interior"
    assert parsed["vertical_relation"] == "above"
    assert parsed["color_relation"] == "different-color"


def test_online_anti_unification_keeps_only_cross_context_invariants():
    abstraction = _induce()

    assert abstraction is not None
    assert abstraction.supported is True
    features = dict(abstraction.features)
    assert features == {
        "change": "transformed",
        "color": "color0",
        "multiplicity": "unique",
        "vertical_relation": "above",
        "relation_alignment": "offset",
        "color_relation": "different-color",
    }
    assert "shape" not in features
    assert "boundary" not in features
    assert abstraction_matches_candidate(abstraction, SECOND_COLOR0) is True
    assert abstraction_matches_candidate(abstraction, SECOND_COLOR12) is False


def test_objective_color_grounding_avoids_a_more_detailed_wrong_color_pattern():
    abstraction = _induce()

    assert abstraction is not None
    assert abstraction.objective_color_aligned is True
    assert dict(abstraction.features)["color"] == "color0"
    assert "color12" not in abstraction.signature


def test_two_nonprogress_controls_prevent_abstract_support():
    abstraction = _induce(
        control_candidate_sets=((SECOND_COLOR0,), (SECOND_COLOR0,)),
    )

    assert abstraction is None


def test_one_regressive_context_vetoes_abstract_support():
    abstraction = _induce(
        regression_candidate_sets=((SECOND_COLOR0,),),
    )

    assert abstraction is None


def test_mediated_evidence_promotes_abstract_not_exact_carrier_online():
    evidence = MediatedEntityEffectEvidence(
        option_id="option",
        objective_id="terminal::exhaust::color0",
        mode_signature="mode",
        action_transfer_signature="action",
        objective_colors=(0,),
        progress_events=2,
        progress_contexts={"first", "second"},
        progress_candidate_sets=[
            (SOURCE_COLOR0,),
            (SECOND_COLOR0, SECOND_COLOR12, SECOND_COLOR4),
        ],
        candidate_supports={
            SOURCE_COLOR0: 1,
            SECOND_COLOR0: 1,
            SECOND_COLOR12: 1,
            SECOND_COLOR4: 1,
        },
    )

    assert evidence.candidate_intersection == set()
    assert evidence.status == MediatedEffectStatus.SUPPORTED
    assert evidence.supported_mediator_is_abstract is True
    assert evidence.supported_mediator_signature.startswith(
        "mediated-abstract::"
    )
    assert evidence.active_candidate_signatures == (
        evidence.supported_mediator_signature,
    )


def test_ablation_retains_exact_refutation_behavior():
    evidence = MediatedEntityEffectEvidence(
        option_id="option",
        objective_id="terminal::exhaust::color0",
        mode_signature="mode",
        action_transfer_signature="action",
        objective_colors=(0,),
        anti_unification_enabled=False,
        progress_events=2,
        progress_contexts={"first", "second"},
        progress_candidate_sets=[(SOURCE_COLOR0,), (SECOND_COLOR0,)],
        candidate_supports={SOURCE_COLOR0: 1, SECOND_COLOR0: 1},
    )

    assert evidence.mediator_abstraction is None
    assert evidence.status == MediatedEffectStatus.CONTRADICTED
    assert evidence.supported_mediator_signature == ""
