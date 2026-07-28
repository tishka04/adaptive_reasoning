from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from theory.sage11.splits import SOURCE_TRAIN
from theory.sage12.action_aligned_semantics_v4_10 import (
    _balanced_epoch_indices,
    _game_balanced_calibration_shifts,
    action_aligned_graph,
    freeze_manifest,
    validate_action_aligned_graph,
    write_capacity_amendment,
)
from theory.sage12.semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    ObjectRelativeGraph,
    SemanticTeacherRecord,
    _checksum,
    _write_json,
)


def _graph() -> ObjectRelativeGraph:
    return ObjectRelativeGraph(
        root={
            "action_name": "ACTION4",
            "action_family": "move",
            "requested_direction": "right",
            "root_kind": "occupied_object",
            "root_occupied": 1,
            "root_area_bucket": "small",
            "root_aspect_bucket": "compact",
            "root_affordance": "movable",
            "actor_relation": "adjacent",
            "actor_relative_direction": "east",
            "path_status": "blocked",
            "boundary": "interior",
            "player_available": 1,
        },
        neighbors=(
            {
                "direction": "east",
                "proximity": "contact",
                "relative_size": "equal",
                "area_bucket": "small",
                "aspect_bucket": "compact",
                "is_actor": 0,
                "aligned_row": 1,
                "aligned_col": 0,
                "touches_boundary": 0,
            },
            {
                "direction": "north",
                "proximity": "adjacent",
                "relative_size": "larger",
                "area_bucket": "medium",
                "aspect_bucket": "wide",
                "is_actor": 1,
                "aligned_row": 0,
                "aligned_col": 1,
                "touches_boundary": 0,
            },
        ),
    )


def _record(game: str, index: int) -> SemanticTeacherRecord:
    labels = {effect: False for effect in SEMANTIC_EFFECTS}
    applicable = {effect: True for effect in SEMANTIC_EFFECTS}
    return SemanticTeacherRecord(
        example_id=f"example_{game}_{index}",
        game_id=game,
        source_corpus="unit",
        trace_digest=f"digest_{game}_{index}",
        exact_repeat_key=f"repeat_{game}_{index}",
        same_prestate_keys=(),
        graph=action_aligned_graph(_graph()),
        labels=labels,
        applicable=applicable,
        productive_score=0.0,
        teacher_evidence={},
    )


def test_action_alignment_removes_compass_and_preserves_topology() -> None:
    aligned = action_aligned_graph(_graph())

    assert [row["axis_relation"] for row in aligned.neighbors] == [
        "ahead",
        "lateral_left",
    ]
    assert aligned.root["ahead_contact"] == 1
    assert aligned.root["contact_degree"] == "one"
    assert "requested_direction" not in aligned.root
    assert "actor_relative_direction" not in aligned.root
    assert all("direction" not in row for row in aligned.neighbors)
    validate_action_aligned_graph(aligned)


def test_relation_shuffle_rotates_only_intervention_relative_axes() -> None:
    original = action_aligned_graph(_graph())
    shuffled = action_aligned_graph(_graph(), relation_shuffle=True)

    assert [row["axis_relation"] for row in shuffled.neighbors] == [
        "lateral_right",
        "ahead",
    ]
    for left, right in zip(original.neighbors, shuffled.neighbors):
        assert left["topology"] == right["topology"]
        assert left["relative_size"] == right["relative_size"]


def test_action_aligned_firewall_rejects_compass_neighbor() -> None:
    graph = ObjectRelativeGraph(
        root={"action_name": "ACTION1"},
        neighbors=({"axis_relation": "north"},),
    )
    with pytest.raises(ValueError, match="compass relation leaked"):
        validate_action_aligned_graph(graph)


def test_balanced_epoch_sampler_contributes_equally_per_game() -> None:
    records = tuple(
        [_record("bp35", index) for index in range(2)]
        + [_record("cd82", index) for index in range(7)]
    )
    batches = _balanced_epoch_indices(
        records,
        np.arange(len(records), dtype=np.int64),
        samples_per_game=8,
        per_game_step=2,
        seed=1,
    )

    assert len(batches) == 4
    for batch in batches:
        games = [records[index].game_id for index in batch]
        assert games.count("bp35") == games.count("cd82") == 2


def test_calibration_shift_matches_game_balanced_direction() -> None:
    first = _record("bp35", 0)
    second = replace(
        _record("cd82", 0),
        labels={effect: effect == "changed" for effect in SEMANTIC_EFFECTS},
    )
    records = (first, second)
    logits = np.full((2, len(SEMANTIC_EFFECTS)), -4.0, dtype=np.float64)
    shifts = _game_balanced_calibration_shifts(
        records,
        np.asarray([0, 1], dtype=np.int64),
        logits,
    )

    assert shifts[SEMANTIC_EFFECTS.index("changed")] > 0.0


def test_capacity_amendment_is_limited_to_exhausted_su15(tmp_path) -> None:
    manifest = freeze_manifest(output_dir=tmp_path)
    reports = {}
    shards = []
    for game in SOURCE_TRAIN:
        target = int(manifest["collection"]["rows_per_game"][game])
        rows = 83 if game == "su15" else target
        reports[game] = {
            "game_id": game,
            "rows": rows,
            "target_rows": target,
            "raw_steps": 3_840 if game == "su15" else target,
            "resets_used": 40 if game == "su15" else 1,
            "duplicate_rejections": 1_197 if game == "su15" else 0,
        }
        shards.append({"game_id": game, "rows": rows})
    collection = {
        "manifest_checksum": manifest["manifest_checksum"],
        "collection_checksum": "collection-unit",
        "collection_ready": False,
        "reports": reports,
        "shards": shards,
    }
    _write_json(tmp_path / "collection_manifest.json", collection)

    amendment = write_capacity_amendment(output_dir=tmp_path)

    assert amendment["collection_ready_under_amendment"]
    assert amendment["authorized_minimum_rows_per_game"]["su15"] == 80
    assert amendment["authorized_total_rows_minimum"] == 1_584
    check = dict(amendment)
    expected = check.pop("amendment_checksum")
    assert _checksum(check) == expected
