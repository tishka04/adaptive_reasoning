from __future__ import annotations

import numpy as np
import pytest

from theory.sage12.action_target_data import (
    ActionTargetAnchor,
    ActionTargetTrace,
    ObservedActionTargetEffects,
)
from theory.sage12.object_relative_student_v4_9 import (
    _torch_model,
    tensorize_records,
)
from theory.sage12.semantic_teacher_v4_9 import (
    SEMANTIC_EFFECTS,
    ObjectRelativeGraph,
    SemanticTeacherRecord,
    build_object_relative_graph,
    compile_semantics,
    validate_model_graph,
)


def _trace() -> ActionTargetTrace:
    before = np.zeros((9, 9), dtype=np.int32)
    before[4, 4] = 2
    before[4, 5] = 3
    after = before.copy()
    after[4, 4] = 0
    after[4, 5] = 2
    return ActionTargetTrace(
        game_id="bp35",
        source_split="source_train",
        policy_seed=1,
        reset_index=0,
        step_index=0,
        collection_phase="unit",
        available_action_names=("ACTION4",),
        selected_action_name="ACTION4",
        selected_action_data={},
        anchor=ActionTargetAnchor(
            kind="move_destination",
            action_family="move",
            requested_direction="right",
            row=4,
            col=5,
            in_bounds=True,
            occupied=True,
            target_object_id=1,
            target_area_bucket="single",
            target_aspect_bucket="compact",
            target_affordance="movable",
            actor_relation="adjacent",
            actor_relative_direction="east",
            path_status="blocked",
        ),
        effects=ObservedActionTargetEffects(
            labels={
                "actor_displaced": True,
                "target_created": False,
                "target_removed": True,
                "target_moved": False,
            },
            applicable={
                "actor_displaced": True,
                "target_created": True,
                "target_removed": True,
                "target_moved": True,
            },
            noop=False,
        ),
        frame_before=before,
        frame_after=after,
        game_state_before="NOT_FINISHED",
        game_state_after="NOT_FINISHED",
        levels_completed_before=0,
        levels_completed_after=0,
    )


def _record() -> SemanticTeacherRecord:
    trace = _trace()
    labels, applicable, score, evidence = compile_semantics(trace)
    return SemanticTeacherRecord(
        example_id="sem_unit",
        game_id=trace.game_id,
        source_corpus="unit",
        trace_digest=trace.trace_digest,
        exact_repeat_key=trace.exact_repeat_key(),
        same_prestate_keys=(),
        graph=build_object_relative_graph(trace),
        labels=labels,
        applicable=applicable,
        productive_score=score,
        teacher_evidence=evidence,
    )


def test_teacher_compiles_frozen_physical_and_functional_vocabulary() -> None:
    trace = _trace()
    labels, applicable, score, evidence = compile_semantics(trace)

    assert tuple(labels) == SEMANTIC_EFFECTS
    assert tuple(applicable) == SEMANTIC_EFFECTS
    assert labels["changed"]
    assert labels["moved"]
    assert labels["target_removed"]
    assert labels["local_change"]
    assert score > 0.0
    assert evidence["anchor_grounded"]


def test_student_graph_excludes_identity_coordinates_and_values() -> None:
    record = _record()
    validate_model_graph(record.graph, game_id=record.game_id)
    fields = set(record.graph.root)
    fields.update(key for neighbor in record.graph.neighbors for key in neighbor)

    for forbidden in (
        "row",
        "col",
        "object_id",
        "value",
        "color",
        "frame",
    ):
        assert forbidden not in fields
    assert record.game_id not in str(record.graph.to_dict())


def test_model_graph_firewall_rejects_absolute_coordinates() -> None:
    graph = ObjectRelativeGraph(
        root={"action_name": "ACTION1", "row": 3},
        neighbors=(),
    )
    with pytest.raises(ValueError, match="forbidden model-graph field"):
        validate_model_graph(graph)


def test_deepsets_is_invariant_to_neighbor_order() -> None:
    import torch

    record = _record()
    records = (record,)
    original = tensorize_records(
        records,
        hash_buckets=128,
        maximum_neighbors=16,
    )
    reversed_graph = tensorize_records(
        records,
        hash_buckets=128,
        maximum_neighbors=16,
        reverse_neighbors=True,
    )
    torch.manual_seed(1)
    model = _torch_model(
        hash_buckets=128,
        embedding_width=8,
        hidden_width=16,
        effect_count=len(SEMANTIC_EFFECTS),
        identity_classes=2,
    )
    model.eval()
    with torch.inference_mode():
        left, _ = model(
            torch.as_tensor(original.root_ids),
            torch.as_tensor(original.neighbor_ids),
            torch.as_tensor(original.neighbor_mask),
        )
        right, _ = model(
            torch.as_tensor(reversed_graph.root_ids),
            torch.as_tensor(reversed_graph.neighbor_ids),
            torch.as_tensor(reversed_graph.neighbor_mask),
        )
    assert torch.max(torch.abs(left - right)).item() <= 1e-6


def test_teacher_record_json_roundtrip_preserves_firewall() -> None:
    record = _record()
    restored = SemanticTeacherRecord.from_dict(record.to_dict())

    assert restored.to_dict() == record.to_dict()
    validate_model_graph(restored.graph, game_id=restored.game_id)
