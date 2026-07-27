from dataclasses import replace

from theory.sage12 import HypothesisCompiler, SemanticActionCandidate
from theory.sage12.constrained_pilot import (
    DEFAULT_FROZEN_MANIFEST_PATH,
    _load_frozen_manifest,
    invariant_motif,
    invariant_prompt,
    render_hypotheses_json,
)
from theory.sage12.hypotheses import hypotheses_from_json
from theory.sage12.proposal_pilot_data import (
    ProposalPilotTrace,
    graph_from_mapping,
)


def _trace() -> ProposalPilotTrace:
    graph = {
        "entities": [
            {
                "entity_id": "player-a",
                "roles": ["object", "player"],
                "area_bucket": "one",
                "aspect_bucket": "square",
            },
            {
                "entity_id": "target-with-layout-token",
                "roles": ["object", "target"],
                "area_bucket": "large",
                "aspect_bucket": "wide",
            },
        ],
        "relations": [
            {
                "kind": "near",
                "subject_id": "player-a",
                "object_id": "target-with-layout-token",
            },
            {
                "kind": "east_of",
                "subject_id": "target-with-layout-token",
                "object_id": "player-a",
            },
        ],
        "state_predicates": [
            "near|player-a|target-with-layout-token|",
            "east_of|target-with-layout-token|player-a|",
        ],
        "signature": "excluded-signature",
    }
    shuffled = {
        **graph,
        "relations": [
            {
                "kind": "near",
                "subject_id": "target-with-layout-token",
                "object_id": "other",
            }
        ],
    }
    return ProposalPilotTrace(
        game_id="bp35",
        source_split="source_train",
        policy_seed=0,
        reset_index=0,
        step_index=0,
        scene_graph=graph,
        relation_shuffle_graph=shuffled,
        available_action_names=("ACTION1",),
        selected_action_name="ACTION1",
        selected_action_data={},
        observed_effects=("changed|-|-|",),
        changed=True,
        noop=False,
        player_moved=True,
        level_complete=False,
        game_over=False,
        productive=True,
        repeat_index=0,
    )


def test_invariant_motif_discards_counts_shapes_ids_and_orientation() -> None:
    trace = _trace()
    renamed_graph = {
        **trace.scene_graph,
        "entities": [
            {
                **trace.scene_graph["entities"][0],
                "entity_id": "p",
                "area_bucket": "large",
                "aspect_bucket": "tall",
            },
            {
                **trace.scene_graph["entities"][1],
                "entity_id": "x",
                "area_bucket": "one",
                "aspect_bucket": "square",
            },
            {
                "entity_id": "unused-extra",
                "roles": ["object", "target"],
                "area_bucket": "medium",
                "aspect_bucket": "wide",
            },
        ],
        "relations": [
            {
                "kind": "near",
                "subject_id": "x",
                "object_id": "p",
            },
            {
                "kind": "west_of",
                "subject_id": "p",
                "object_id": "x",
            },
        ],
    }

    assert invariant_motif(trace) == invariant_motif(
        replace(trace, scene_graph=renamed_graph)
    )


def test_relation_shuffle_can_remove_actor_local_signal() -> None:
    trace = _trace()

    assert invariant_motif(trace)["actor_interaction"] == 1
    assert (
        invariant_motif(trace, variant="relation_shuffle")[
            "actor_interaction"
        ]
        == 0
    )


def test_prompt_exposes_only_action_and_binary_motif() -> None:
    prompt = invariant_prompt(_trace())

    assert "ACTION1" in prompt
    assert "actor_interaction=yes" in prompt
    assert "player-a" not in prompt
    assert "target-with-layout-token" not in prompt
    assert "excluded-signature" not in prompt


def test_constrained_renderer_is_strict_support_zero_and_grounded() -> None:
    trace = _trace()
    raw = render_hypotheses_json(
        trace,
        {
            "changed": True,
            "player_moved": True,
            "level_complete": False,
            "game_over": False,
        },
    )
    parsed = hypotheses_from_json(raw, maximum=8)
    compilation = HypothesisCompiler().compile(
        parsed,
        graph=graph_from_mapping(trace.scene_graph),
        legal_candidates=(SemanticActionCandidate("ACTION1"),),
    )

    assert len(parsed) == 2
    assert all(item.support == 0 for item in parsed)
    assert len({item.hypothesis_id for item in compilation.options}) == 2
    assert compilation.rejected == ()


def test_v2_manifest_is_frozen_and_cannot_authorize_world_model() -> None:
    manifest = _load_frozen_manifest(DEFAULT_FROZEN_MANIFEST_PATH)

    assert manifest["status"] == "FROZEN_BEFORE_VALIDATION"
    assert manifest["source_corpus"]["source_train_rows"] == 1_624
    assert manifest["source_corpus"]["source_validation_rows"] == 480
    assert manifest["world_model_fit_authorized"] is False
