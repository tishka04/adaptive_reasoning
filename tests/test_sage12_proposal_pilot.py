from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from theory.sage12 import (
    HypothesisCompiler,
    SemanticActionCandidate,
    SemanticEffect,
    SemanticHypothesis,
    SemanticPredicate,
    build_scene_graph,
)
from theory.sage12.proposal_pilot_collection import _balanced_action
from theory.sage12.proposal_pilot_data import (
    ProposalPilotTrace,
    graph_from_mapping,
    graph_to_mapping,
    load_frozen_manifest,
)
from theory.sage12.proposal_pilot_runner import (
    _mechanism_names,
    _representative_sample,
    _shuffle_relations,
)
from theory.sage12.llm import _compact_scene
from theory.sage12.scene_graph import (
    GroundedEntity,
    GroundedRelation,
    SceneGraph,
)
from v3.schemas import GameObservation, ObjectInfo, PlayerHypothesis


@dataclass(frozen=True)
class _Candidate:
    name: str
    action_args: dict[str, int] = field(default_factory=dict)


def _observation() -> GameObservation:
    grid = np.zeros((4, 4), dtype=np.int64)
    grid[1, 1] = 1
    grid[1, 2] = 2
    return GameObservation(
        raw_grid=grid,
        grid_hash=1,
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=["ACTION1", "ACTION6"],
        objects=[
            ObjectInfo(
                object_id=1,
                value=1,
                cells=[(1, 1)],
                bbox=(1, 1, 1, 1),
                center=(1.0, 1.0),
                area=1,
            ),
            ObjectInfo(
                object_id=2,
                value=2,
                cells=[(1, 2)],
                bbox=(1, 2, 1, 2),
                center=(1.0, 2.0),
                area=1,
            ),
        ],
        player_candidates=[
            PlayerHypothesis(
                value=1,
                position=(1, 1),
                confidence=0.9,
            )
        ],
    )


def _trace(game: str, index: int) -> ProposalPilotTrace:
    graph = build_scene_graph(_observation())
    return ProposalPilotTrace(
        game_id=game,
        source_split=(
            "source_validation"
            if game in {"re86", "ls20", "sc25"}
            else "source_train"
        ),
        policy_seed=0,
        reset_index=0,
        step_index=index,
        scene_graph=graph_to_mapping(graph),
        available_action_names=("ACTION1", "ACTION6"),
        selected_action_name="ACTION1",
        selected_action_data={},
        observed_effects=("changed|-|-|", "moved|e0|-|"),
        changed=True,
        noop=False,
        player_moved=True,
        level_complete=False,
        game_over=False,
        productive=True,
        repeat_index=0,
    )


def test_frozen_manifest_is_checksummed_and_source_only() -> None:
    manifest = load_frozen_manifest()

    assert manifest["target_transitions"] == 2_104
    assert set(manifest["game_quotas"]) == set(
        manifest["source_train_games"] + manifest["source_validation_games"]
    )
    assert manifest["firewall"]["holdout_opened"] is False
    assert manifest["world_model_fit_authorized"] is False


def test_graph_archive_excludes_centers_and_raw_values() -> None:
    graph = build_scene_graph(_observation())
    payload = graph_to_mapping(graph)
    restored = graph_from_mapping(payload)

    assert "center" not in payload["entities"][0]
    assert "value_token" not in payload["entities"][0]
    assert restored.signature == graph.signature
    assert [item.key for item in restored.relations] == [
        item.key for item in graph.relations
    ]


def test_action_family_hypothesis_expands_to_exact_legal_arguments() -> None:
    graph = build_scene_graph(_observation())
    hypothesis = SemanticHypothesis(
        hypothesis_id="click_family",
        action_name="ACTION6",
        effects=(SemanticEffect(SemanticPredicate("changed")),),
    )
    candidates = (
        SemanticActionCandidate("ACTION6", {"x": 1, "y": 2}),
        SemanticActionCandidate("ACTION6", {"x": 3, "y": 4}),
    )

    result = HypothesisCompiler().compile(
        (hypothesis,),
        graph=graph,
        legal_candidates=candidates,
    )

    assert {tuple(sorted(option.action_data.items())) for option in result.options} == {
        (("x", 1), ("y", 2)),
        (("x", 3), ("y", 4)),
    }


def test_balanced_policy_prefers_underrepresented_action_family() -> None:
    legal = (
        _Candidate("ACTION1"),
        _Candidate("ACTION6", {"x": 1, "y": 1}),
        _Candidate("ACTION6", {"x": 2, "y": 2}),
    )

    selected = _balanced_action(
        legal,
        counts={"ACTION1": 5, "ACTION6": 1},
        game="bp35",
        seed=0,
        reset_index=0,
        raw_step=0,
    )

    assert selected.name == "ACTION6"


def test_representative_sampling_is_outcome_independent_and_bounded() -> None:
    manifest = load_frozen_manifest()
    traces = tuple(
        _trace(game, index)
        for game in manifest["game_quotas"]
        for index in range(12)
    )

    first = _representative_sample(traces, manifest)
    second = _representative_sample(tuple(reversed(traces)), manifest)

    assert [trace.digest for trace in first] == [
        trace.digest for trace in second
    ]
    assert len(first) == 8 * len(manifest["game_quotas"])


def test_relation_shuffle_preserves_entities_but_changes_bindings() -> None:
    graph = build_scene_graph(_observation())
    shuffled = _shuffle_relations(graph, "1" * 64)

    assert shuffled.entities == graph.entities
    assert shuffled.signature != graph.signature
    assert [item.key for item in shuffled.relations] != [
        item.key for item in graph.relations
    ]


def test_mechanism_labels_exclude_generic_change_and_progress() -> None:
    labels = _mechanism_names(
        (
            "changed|-|-|",
            "progress|-|-|",
            "moved|e0|-|",
            "level_complete|-|-|",
        )
    )

    assert labels == {"moved", "level_complete"}


def test_prompt_scene_compaction_is_deterministic_and_bounded() -> None:
    entities = tuple(
        GroundedEntity(
            entity_id=f"e{index:03d}",
            roles=("player",) if index == 99 else ("object", "target"),
            center=(0.0, 0.0),
            area_bucket="one",
            aspect_bucket="square",
            value_token="excluded",
        )
        for index in range(100)
    )
    relations = tuple(
        GroundedRelation(
            kind=("near" if index % 2 else "aligned"),
            subject_id=f"e{index % 100:03d}",
            object_id=f"e{(index + 1) % 100:03d}",
        )
        for index in range(500)
    )
    graph = SceneGraph(
        entities=entities,
        relations=relations,
        state_predicates=frozenset(item.key for item in relations),
        signature="large",
    )

    selected_entities, selected_relations = _compact_scene(
        graph,
        maximum_entities=24,
        maximum_relations=96,
    )

    assert len(selected_entities) == 24
    assert len(selected_relations) <= 96
    assert selected_entities[0].entity_id == "e099"
    assert {item.kind for item in selected_relations} == {
        "aligned",
        "near",
    }
