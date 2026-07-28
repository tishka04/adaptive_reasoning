from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage12.action_target_data import (
    ActionTargetAnchor,
    ActionTargetTrace,
    ObservedActionTargetEffects,
)
from theory.sage12.object_causal_pilot import (
    COLLECTION_FORMAT_VERSION,
    DiscoveredEventToken,
    InterventionDeltaRecord,
    ObjectCausalExample,
    ObjectCorrespondence,
    ObjectEvent,
    ObjectTrackState,
    RootedTargetGraph,
    _cancel_common_events,
    build_rooted_target_graph,
    default_manifest,
    discover_event_vocabulary,
    match_objects,
    run_source_collection,
    validate_model_view,
)
from theory.sage12.pairwise_causal_pilot import _fit_model
from v3.schemas import ObjectInfo


def _object(
    object_id: int,
    value: int,
    cells: list[tuple[int, int]],
) -> ObjectInfo:
    rows = [row for row, _col in cells]
    cols = [col for _row, col in cells]
    return ObjectInfo(
        object_id=object_id,
        value=value,
        cells=cells,
        bbox=(min(rows), min(cols), max(rows), max(cols)),
        center=(sum(rows) / len(rows), sum(cols) / len(cols)),
        area=len(cells),
    )


def _correspondence() -> ObjectCorrespondence:
    return ObjectCorrespondence(matched=(), appeared=(), disappeared=())


def _event(
    *,
    operation: str = "appeared",
    locus: str = "direct",
    direction: str = "none",
    magnitude: str = "small",
    subject: str = "pre:1",
) -> ObjectEvent:
    return ObjectEvent(
        operation=operation,
        locus=locus,
        direction=direction,
        magnitude=magnitude,
        subject=subject,
        changed_cells=((1, 1),),
        confidence=1.0,
    )


def _delta(
    pair_id: str,
    game_id: str,
    left: tuple[ObjectEvent, ...],
    right: tuple[ObjectEvent, ...],
) -> InterventionDeltaRecord:
    return InterventionDeltaRecord(
        pair_id=pair_id,
        game_id=game_id,
        source_split="source_train",
        left_events=left,
        right_events=right,
        left_correspondence=_correspondence(),
        right_correspondence=_correspondence(),
        common_events_cancelled=0,
        exclusive_localization=1.0,
        pre_state_identical=True,
    )


def _trace(
    *,
    before: list[list[int]],
    after: list[list[int]],
    row: int = 1,
    col: int = 1,
) -> ActionTargetTrace:
    return ActionTargetTrace(
        game_id="bp35",
        source_split="source_train",
        policy_seed=1,
        reset_index=0,
        step_index=0,
        collection_phase="test",
        available_action_names=("ACTION6",),
        selected_action_name="ACTION6",
        selected_action_data={"x": col, "y": row},
        anchor=ActionTargetAnchor(
            kind="click",
            action_family="click",
            row=row,
            col=col,
            in_bounds=True,
            occupied=True,
            target_object_id=0,
            actor_relation="near",
        ),
        effects=ObservedActionTargetEffects(
            labels={
                "actor_displaced": False,
                "target_created": False,
                "target_removed": False,
                "target_moved": False,
            },
            applicable={
                "actor_displaced": True,
                "target_created": True,
                "target_removed": True,
                "target_moved": True,
            },
        ),
        frame_before=before,
        frame_after=after,
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        levels_completed_before=0,
        levels_completed_after=0,
    )


def _graph(*, kind: str, north: int = 0, action: str = "ACTION6") -> RootedTargetGraph:
    return RootedTargetGraph(
        root_kind=kind,
        action_name=action,
        action_family="click",
        requested_direction="none",
        relation_counts=(("near:north", north),) if north else (),
        neighbor_roles=(("object:smaller", 1),),
        actor_relation="near",
    )


def test_correspondence_accepts_translation_and_recoloring() -> None:
    before = [_object(1, 3, [(1, 1), (1, 2)])]
    translated = [_object(7, 3, [(4, 4), (4, 5)])]
    recolored = [_object(8, 9, [(1, 1), (1, 2)])]
    assert match_objects(before, translated).matched[0][:2] == (1, 7)
    assert match_objects(before, recolored).matched[0][:2] == (1, 8)


def test_correspondence_detects_split_and_exposes_ambiguity() -> None:
    source = [_object(1, 3, [(1, 1), (1, 2)])]
    children = [
        _object(2, 3, [(1, 1)]),
        _object(3, 3, [(1, 2)]),
    ]
    split = match_objects(source, children, minimum_score=0.80)
    assert split.splits[0][:2] == (1, (2, 3))

    candidates = [
        _object(4, 3, [(3, 3), (3, 4)]),
        _object(5, 3, [(5, 3), (5, 4)]),
    ]
    ambiguous = match_objects(
        source,
        candidates,
        minimum_score=0.60,
        ambiguity_margin=0.20,
    )
    assert ambiguous.ambiguous_before == (1,)
    assert not ambiguous.matched


def test_common_dynamics_are_cancelled_by_subject_and_semantics() -> None:
    shared = _event(operation="displaced", direction="east")
    exclusive = _event(operation="appeared", subject="new:north")
    left, right, count = _cancel_common_events(
        (shared, exclusive),
        (shared,),
    )
    assert count == 1
    assert left == (exclusive,)
    assert right == ()


def test_vocabulary_uses_deterministic_backoff_and_cross_game_capacity() -> None:
    deltas = []
    games = ("bp35", "cd82", "g50t")
    for game in games:
        for index in range(10):
            magnitude = "small" if index % 2 else "medium"
            deltas.append(
                _delta(
                    f"{game}-{index}",
                    game,
                    (_event(magnitude=magnitude),),
                    (),
                )
            )
    vocabulary, mapping = discover_event_vocabulary(
        deltas,
        minimum_discordant_pairs=30,
        minimum_games_with_10=3,
    )
    assert vocabulary == (
        DiscoveredEventToken(
            token="direct|appeared|none|any",
            projection="no_magnitude",
            discordant_pairs=30,
            games_with_at_least_10=3,
            per_game={"bp35": 10, "cd82": 10, "g50t": 10},
        ),
    )
    assert set(mapping.values()) == {"direct|appeared|none|any"}


def test_rooted_graph_uses_relative_features_and_track_buckets() -> None:
    trace = _trace(
        before=[
            [0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 0],
        ],
        after=[
            [0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 0],
        ],
    )
    graph = build_rooted_target_graph(
        trace,
        track=ObjectTrackState(
            interactions=2,
            last_operation="recolored",
            transitions_since_interaction=1,
            confidence=0.9,
        ),
    )
    features = graph.model_features()
    rendered = json.dumps(features)
    assert graph.grounded
    assert "track:interactions=two_plus" in rendered
    assert '"row"' not in rendered
    assert '"col"' not in rendered
    assert "object_id" not in rendered


def test_model_view_is_identity_free_and_exactly_antisymmetric() -> None:
    example = ObjectCausalExample(
        pair_id="pair",
        game_id="bp35",
        source_split="source_train",
        context=(),
        left_graph=_graph(kind="occupied_object", north=2),
        right_graph=_graph(kind="virtual_cell", north=1),
        outcomes={"direct|appeared|none|any": (True, False)},
    )
    validate_model_view(example, "structured")
    row = example.model_view("structured")
    model = _fit_model([row], [1])
    inverted = {key: -value for key, value in row.items()}
    assert model.predict(inverted) == pytest.approx(
        1.0 - model.predict(row), abs=1e-15
    )
    assert example.model_view("structured", root_swap=True) != row
    assert example.model_view("structured", relation_shuffle=True) != row


def test_manifest_freezes_collection_and_authority_firewall() -> None:
    manifest = default_manifest()
    assert manifest["status"] == "FROZEN_BEFORE_FEASIBILITY"
    assert manifest["fresh_collection"]["format_version"] == COLLECTION_FORMAT_VERSION
    assert manifest["fresh_collection"]["context_full_traces"] == 8
    assert manifest["firewall"]["source_validation_opened"] is False
    assert manifest["firewall"]["world_model_authorized"] is False


def test_failed_feasibility_mechanically_closes_fresh_collection(
    tmp_path: Path,
) -> None:
    (tmp_path / "feasibility_result.json").write_text(
        json.dumps(
            {
                "status": "FAIL_CLOSED",
                "feasibility_checksum": "failed",
                "fresh_source_collection_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    result = run_source_collection(output_dir=tmp_path)
    assert result["status"] == "SKIPPED_FAIL_CLOSED"
    assert result["source_validation_opened"] is False
    assert not (tmp_path / "source_train_shards").exists()
